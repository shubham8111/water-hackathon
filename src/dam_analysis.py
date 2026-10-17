"""KRS and Hemavathy: daily inflow, monsoon-total forecast, warning lead time, river-network tests.

1. Daily inflow for both dams with the same gradient-boosting setup (fit 1991-2019, test 2020-2025).
   Rain enters as distance-band features plus per-grid-cell routed rain (cell_response.py: every
   cell has its own delay and multiplier). Hemavathy has no gauged dam upstream, so the like-for-like
   KRS comparison is the rain-only KRS model. Also: peaks, and lead times of 1-7 days with and
   without a (perfect) rain forecast.
2. Total Jun-Sep inflow forecast for both dams from each issue date (leave-one-year-out),
   with and without a (perfect) rainfall forecast.
3. River network (MERIT-style flowlines): distance along the river from each grid cell to the dam
   replaces straight-line distance for the near/mid/far bands; and rain over the Hemavathy
   catchment (held back by Hemavathy dam) is split from the free-draining part of the KRS catchment.
   These experiments use the distance-band features only.
Writes outputs/dam_results.json. Needs data/watershed_hemavati.zip, rivers_ksr.zip, rivers_hemavati.zip.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely.geometry import box
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inflow_model import api, metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
KRS_DAM = (76.575, 12.422)
CAP = {"krs": 49.45, "hemavathy": 37.10}
BANDS = ["near", "mid", "far"]


def km(lon1, lat1, lon2, lat2):
    return np.hypot((lon1 - lon2) * 111.32 * np.cos(np.radians(12.6)), (lat1 - lat2) * 110.57)


def cells_for(poly, cell_ids):
    """Area fraction of each existing 0.25 deg cell inside poly."""
    rows = []
    for cid in cell_ids:
        lat, lon = map(float, cid[2:].split("_"))
        frac = box(lon - 0.125, lat - 0.125, lon + 0.125, lat + 0.125).intersection(poly).area / 0.0625
        if frac > 0.01:
            rows.append({"cell_id": cid, "lon": lon, "lat": lat, "frac": frac})
    return pd.DataFrame(rows)


class RiverNet:
    """Undirected graph of flowline vertices; distances in km along the channel."""

    def __init__(self, lines):
        coords, edges = {}, []
        idx = lambda p: coords.setdefault((round(p[0], 5), round(p[1], 5)), len(coords))
        for geom, order in zip(lines.geometry, lines.sorder):
            pts = list(geom.coords)
            for a, b in zip(pts[:-1], pts[1:]):
                edges.append((idx(a), idx(b), km(a[0], a[1], b[0], b[1])))
        self.xy = np.array(list(coords.keys()))
        e = np.array(edges)
        n = len(self.xy)
        self.g = coo_matrix((e[:, 2], (e[:, 0].astype(int), e[:, 1].astype(int))), shape=(n, n)).tocsr()
        self.n_components = connected_components(self.g, directed=False)[0]

    def nearest(self, lon, lat):
        d = km(self.xy[:, 0], self.xy[:, 1], lon, lat)
        return int(d.argmin()), float(d.min())

    def flow_km(self, cells, outlet):
        """Overland hop to the nearest channel vertex + channel distance to the outlet."""
        o, _ = self.nearest(*outlet)
        dist = dijkstra(self.g, directed=False, indices=o)
        out = []
        for lon, lat in zip(cells.lon, cells.lat):
            # several candidate vertices near the cell centre; take the best overall path
            d0 = km(self.xy[:, 0], self.xy[:, 1], lon, lat)
            cand = np.argsort(d0)[:25]
            out.append(float(np.min(d0[cand] + dist[cand])))
        return np.array(out)


def band_rain(df, cells, dist_col):
    w = cells.set_index("cell_id").frac
    r = pd.DataFrame(index=df.index)
    r["basin"] = (df[w.index] * w).sum(axis=1) / w.sum()
    b = pd.qcut(cells[dist_col].rank(method="first"), 3, labels=BANDS).astype(str).values
    for name in BANDS:
        ids = cells.cell_id[b == name]
        r[name] = (df[ids] * w[ids]).sum(axis=1) / w[ids].sum()
    return r


def features(rain, df, upstream=(), extra_rain=None):
    f = pd.DataFrame(index=df.index)
    for b in BANDS:
        for k in range(8):
            f[f"rain_{b}_lag{k}"] = rain[b].shift(k)
    for w in [3, 7, 14, 30, 60]:
        f[f"rain_sum_{w}d"] = rain["basin"].rolling(w).sum()
    f["api_0.9"] = api(rain["basin"], 0.9)
    f["api_0.97"] = api(rain["basin"], 0.97)
    wy = df.index.year - (df.index.month < 6)
    f["rain_since_jun1"] = rain["basin"].groupby(wy).cumsum()
    if extra_rain is not None:  # rain over a regulated sub-catchment, as its own block
        for k in range(4):
            f[f"rain_reg_lag{k}"] = extra_rain.shift(k)
        f["rain_reg_sum_14d"] = extra_rain.rolling(14).sum()
    for u in upstream:
        for k in range(6):
            f[f"{u}_out_lag{k}"] = df[f"{u}_outflow"].shift(k)
        f[f"{u}_pct"] = df[f"{u}_pct"]
    doy = df.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


def ar_feats(q):
    ar = pd.DataFrame(index=q.index)
    for k in range(1, 8):
        ar[f"in_lag{k}"] = np.log1p(q.shift(k))
    ar["in_mean7"] = np.log1p(q.shift(1).rolling(7).mean())
    return ar


def gb():
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=0)


def run_daily(X, q):
    """Same protocol as inflow_model.py: fit on train+val, cap + Duan smearing, score on test."""
    y = np.log1p(q)
    yrs = q.index.year
    ok = X.notna().all(axis=1) & y.notna()
    trv, te = ok & (yrs <= 2019), ok & (yrs >= 2020)
    m = gb().fit(X[trv], y[trv])
    cap = lambda z: np.clip(z, 0, y[trv].max())
    smear = np.mean(np.exp(y[trv] - cap(m.predict(X[trv]))))
    p = pd.Series(np.expm1(cap(m.predict(X[te]))) * smear, index=X.index[te])
    mon = p.index.month.isin([6, 7, 8, 9])
    return {"all": metrics(q[te], p), "monsoon": metrics(q[te][mon], p[mon])}, p


def baselines(q):
    yrs = q.index.year
    tr, te = yrs <= 2019, (yrs >= 2020) & q.notna() & q.shift(1).notna()
    clim = q[tr].groupby(q.index[tr].dayofyear).mean().rolling(15, center=True, min_periods=1).mean()
    c = pd.Series(q.index.dayofyear.map(clim).values, index=q.index)
    return {"Climatology": (metrics(q[te], c[te]), c[te]),
            "Persistence (yesterday)": (metrics(q[te], q.shift(1)[te]), q.shift(1)[te])}


def peaks(q, p):
    """Same peak checks as peak_analysis.py: same-day size error on p95 peaks, rise capture."""
    yrs = q.index.year
    p95 = q[yrs <= 2014].quantile(0.95)
    t = q.loc[p.index]
    roll = t.rolling(15, center=True, min_periods=1).max()
    ev = t[(t >= p95) & (t == roll)]
    ev = ev[~ev.index.to_series().diff().dt.days.lt(7).fillna(False)]
    err = (p[ev.index] - ev) / ev * 100
    rise = t.diff()
    big = rise >= 5000
    cap_ = (p.diff()[big] / rise[big]).clip(-1, 2)
    return {"p95": round(float(p95)), "events": int(len(ev)), "peak_median_err_pct": round(float(err.median()), 1),
            "peak_within_25pct": round(float((err.abs() <= 25).mean()), 3),
            "rise_capture_median": round(float(cap_.median()), 3)}


def lag_curve(rain, q, max_lag=10):
    X = pd.DataFrame(index=q.index)
    wet = api(rain["basin"], 0.97)
    wet = np.clip(wet / np.nanpercentile(wet, 90), 0, 2)
    for b in BANDS:
        for k in range(max_lag + 1):
            X[f"{b}|{k}"] = rain[b].shift(k)
            X[f"w{b}|{k}"] = rain[b].shift(k) * wet
    m = q.index.month.isin([6, 7, 8, 9, 10]) & X.notna().all(axis=1) & q.notna()
    c = pd.Series(Ridge(alpha=1.0, positive=True).fit(X[m], q[m]).coef_, index=X.columns)
    out = {}
    for b in BANDS:
        v = np.array([c[f"{b}|{k}"] + c[f"w{b}|{k}"] for k in range(max_lag + 1)])
        out[b] = {"cusecs_per_mm": round(float(v.sum()), 1), "peak_lag_days": int(v.argmax()),
                  "mean_lag_days": round(float((v * np.arange(len(v))).sum() / max(v.sum(), 1e-9)), 1),
                  "curve": [round(float(x), 2) for x in v]}
    return out


RAIN_HORIZONS = {"No rain forecast": 0, "+ next 7 days of rain": 7, "+ next 15 days": 15,
                 "+ next 30 days": 30, "+ rest of season (to 30 Sep)": 999}
ISSUES = [(6, 1), (6, 15), (7, 1), (7, 15), (8, 1), (8, 15), (9, 1), (9, 15)]


def monsoon_inflow(df, rain_basin, dam, upstream=()):
    """Total 1 Jun - 30 Sep inflow (TMC) forecast from each issue date, leave-one-year-out.

    Forecast = inflow already observed (1 Jun .. issue date) + predicted remaining inflow.
    Ridge per issue date. Rain-forecast variants add the actual rain over the next H days
    (a perfect forecast: an upper bound on what a real forecast could add)."""
    q = df[f"{dam}_inflow"].interpolate(limit=5).fillna(0) * 86400 / 1e9  # TMC/day
    rows = []
    for year in range(1991, 2026):
        jun1, sep30 = pd.Timestamp(year, 6, 1), pd.Timestamp(year, 9, 30)
        total = q[jun1:sep30].sum()
        for m, d in ISSUES:
            t = pd.Timestamp(year, m, d)
            r = {"year": year, "issue": f"{t:%d %b}", "total": total,
                 "sofar": q[jun1:t].sum(), "last7": q[t - pd.Timedelta(days=6):t].sum(),
                 "rain_sofar": rain_basin[jun1:t].sum(), "rain_last30": rain_basin[t - pd.Timedelta(days=29):t].sum(),
                 "rain_may": rain_basin[pd.Timestamp(year, 5, 1):pd.Timestamp(year, 5, 31)].sum(),
                 "storage_pct": df.at[t, f"{dam}_pct"]}
            for u in upstream:
                r[f"{u}_pct"] = df.at[t, f"{u}_pct"]
            for name, H in RAIN_HORIZONS.items():
                if H:
                    stop = min(t + pd.Timedelta(days=H), sep30)
                    r[f"rain_next_{H}"] = rain_basin[t + pd.Timedelta(days=1):stop].sum()
            rows.append(r)
    rows = pd.DataFrame(rows)
    rows["storage_pct"] = rows.storage_pct.fillna(rows.groupby("issue").storage_pct.transform("median"))
    base = ["sofar", "last7", "rain_sofar", "rain_last30", "rain_may", "storage_pct"] + [f"{u}_pct" for u in upstream]
    rows[base] = rows[base].fillna(rows.groupby("issue")[base].transform("median"))
    mean_total = rows.groupby("year").total.first().mean()
    out = []
    for iss in rows.issue.unique():
        h = rows[rows.issue == iss].reset_index(drop=True)
        y = h.total - h.sofar
        clim = np.array([h.total[h.index != i].mean() for i in range(len(h))])
        rmse_c = float(np.sqrt(np.mean((clim - h.total) ** 2)))
        clim_pct = np.abs((clim - h.total) / h.total) * 100
        low = h.total < h.total.quantile(1 / 3)
        r = {"issue_date": iss, "rmse_climatology_tmc": round(rmse_c, 2),
             "climatology_median_abs_pct_err": round(float(np.median(clim_pct)), 1),
             "climatology_p80_abs_pct_err": round(float(np.quantile(clim_pct, 0.8)), 1),
             "already_observed_pct": round(float((h.sofar / h.total).median() * 100), 0)}
        r["years"] = [{"year": int(yr), "actual": round(float(a), 1)} for yr, a in zip(h.year, h.total)]
        for name, H in RAIN_HORIZONS.items():
            feats = base + ([f"rain_next_{H}"] if H else [])
            pred = np.zeros(len(h))
            for i in range(len(h)):
                tr = h.index != i
                m = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(h.loc[tr, feats], y[tr])
                pred[i] = h.at[i, "sofar"] + max(m.predict(h.loc[[i], feats])[0], 0)
            err = pred - h.total
            rmse = float(np.sqrt(np.mean(err ** 2)))
            # "dry season" call: forecast in the lowest third of training years
            thr = np.array([h.total[h.index != i].quantile(1 / 3) for i in range(len(h))])
            pct = np.abs(err / h.total) * 100  # % error relative to that year's actual total
            r[name] = {"rmse_tmc": round(rmse, 2), "rmse_pct_of_mean": round(rmse / mean_total * 100, 1),
                       "median_abs_pct_err": round(float(np.median(pct)), 1),
                       "p80_abs_pct_err": round(float(np.quantile(pct, 0.8)), 1),
                       "median_bias_pct": round(float(np.median(err / h.total * 100)), 1),
                       "within_20pct": round(float(np.mean(np.abs(err / h.total) <= 0.2)), 3),
                       "skill_vs_climatology": round(1 - rmse / rmse_c, 3),
                       "dry_years_caught": f"{int(((pred < thr) & low).sum())}/{int(low.sum())}",
                       "dry_false_alarms": int(((pred < thr) & ~low).sum())}
            for k, v in enumerate(pred):
                r["years"][k][name] = round(float(v), 1)
        out.append(r)
    years = rows.groupby("year").total.first()
    late = rows[rows.issue == "01 Aug"].set_index("year")
    after_aug1 = {int(y): round(float((1 - late.at[y, "sofar"] / late.at[y, "total"]) * 100), 1) for y in late.index}
    return {"mean_total_tmc": round(float(mean_total), 1), "min_tmc": round(float(years.min()), 1),
            "max_tmc": round(float(years.max()), 1), "by_year": {int(k): round(float(v), 1) for k, v in years.items()},
            "after_aug1_pct": after_aug1, "by_issue": out}


def lead_time(rain, df, q, upstream=(), extra=None):
    """Inflow h days ahead from what is known on day t; optionally + actual rain over t+1..t+h."""
    base = features(rain, df, upstream).join(ar_feats(q))
    if extra is not None:
        base = base.join(extra)
    yrs = q.index.year
    p95 = q[yrs <= 2014].quantile(0.95)
    out = []
    for h in [1, 2, 3, 5, 7]:
        y = np.log1p(q.shift(-h))
        fut = pd.DataFrame({f"fut_{b}": rain[b].rolling(h).sum().shift(-h) for b in ["basin"] + BANDS})
        fut["fut_max_day"] = rain["basin"].rolling(h).max().shift(-h)
        for variant, X in [("Known on the day", base), ("+ rain forecast", base.join(fut))]:
            ok = X.notna().all(axis=1) & y.notna()
            trv, te = ok & (yrs <= 2019), ok & (yrs >= 2020)
            m = gb().fit(X[trv], y[trv])
            cap = lambda z: np.clip(z, 0, y[trv].max())
            smear = np.mean(np.exp(y[trv] - cap(m.predict(X[trv]))))
            pred = pd.Series(np.expm1(cap(m.predict(X[te]))) * smear, index=X.index[te] + pd.Timedelta(days=h))
            obs = q.reindex(pred.index)
            o, pr = obs >= p95, pred >= p95
            hits, miss, fa = int((o & pr).sum()), int((o & ~pr).sum()), int((~o & pr).sum())
            roll = obs.rolling(15, center=True, min_periods=1).max()
            ev = obs[(obs >= p95) & (obs == roll)]
            ev = ev[~ev.index.to_series().diff().dt.days.lt(7).fillna(False)]
            err = (pred[ev.index] - ev) / ev * 100
            out.append({"lead_days": h, "variant": variant, "NSE": metrics(obs, pred)["NSE"],
                        "high_flow_caught_pct": round(hits / max(hits + miss, 1) * 100, 1),
                        "false_alarm_pct": round(fa / max(hits + fa, 1) * 100, 1),
                        "peak_median_err_pct": round(float(err.median()), 1),
                        "peak_within_25pct": round(float((err.abs() <= 25).mean() * 100), 1)})
            print("lead", h, variant, out[-1])
    return out


def hemavathy_outlet(riv_hem, ws_hem):
    """Dam = where the highest-order Hemavathy reach meets its watershed boundary."""
    top = riv_hem[riv_hem.sorder == riv_hem.sorder.max()]
    ends = np.array([p for g in top.geometry for p in (g.coords[0], g.coords[-1])])
    bd = np.array([ws_hem.exterior.distance(gpd.points_from_xy([x], [y])[0]) for x, y in ends])
    return tuple(ends[bd.argmin()])


def main():
    df = pd.read_parquet(DATA / "processed" / "daily.parquet")
    res = pd.read_parquet(DATA / "processed" / "reservoirs_full.parquet")
    cell_ids = [c for c in df.columns if c.startswith("g_")]
    ws_krs = gpd.read_file(ROOT / "watershed_KSR" / "watershed.shp").to_crs(4326).union_all()
    ws_hem = gpd.read_file(f"zip://{DATA / 'watershed_hemavati.zip'}").to_crs(4326).union_all()
    riv_krs = gpd.read_file(f"zip://{DATA / 'rivers_ksr.zip'}").to_crs(4326)
    riv_hem = gpd.read_file(f"zip://{DATA / 'rivers_hemavati.zip'}").to_crs(4326)
    R = {"inputs": {"hemavathy_watershed_km2": round(gpd.GeoSeries([ws_hem], crs=4326).to_crs(32643).area[0] / 1e6),
                    "krs_watershed_km2": round(gpd.GeoSeries([ws_krs], crs=4326).to_crs(32643).area[0] / 1e6),
                    "river_reaches_krs": len(riv_krs), "river_reaches_hemavathy": len(riv_hem),
                    "max_stream_order": int(riv_krs.sorder.max())}}

    # Hemavathy dam = where the Hemavathy river network leaves its watershed (lowest point of the
    # highest-order reach closest to the boundary)
    net_h = RiverNet(riv_hem)
    hem_dam = hemavathy_outlet(riv_hem, ws_hem)
    net_k = RiverNet(riv_krs)
    R["inputs"].update({"hemavathy_outlet_lonlat": [round(hem_dam[0], 3), round(hem_dam[1], 3)],
                        "network_components_krs": int(net_k.n_components),
                        "network_components_hemavathy": int(net_h.n_components),
                        "krs_dam_to_network_km": round(net_k.nearest(*KRS_DAM)[1], 2)})
    print(R["inputs"])

    hc = cells_for(ws_hem, cell_ids)
    kc = cells_for(ws_krs, cell_ids)
    hc["straight_km"] = km(hc.lon, hc.lat, *hem_dam)
    hc["flow_km"] = net_h.flow_km(hc, hem_dam)
    kc["straight_km"] = km(kc.lon, kc.lat, *KRS_DAM)
    kc["flow_km"] = net_k.flow_km(kc, KRS_DAM)
    # KRS cells split by the Hemavathy watershed: rain there is caught by Hemavathy dam first
    kc["hem_frac"] = [box(lo - .125, la - .125, lo + .125, la + .125).intersection(ws_hem).area / 0.0625
                      for lo, la in zip(kc.lon, kc.lat)]
    free = kc.assign(frac=(kc.frac - kc.hem_frac).clip(lower=0))
    free = free[free.frac > 0.01].copy()
    R["cells"] = {"hemavathy": hc.round(3).to_dict("records"), "krs": kc.round(3).to_dict("records")}
    print(f"cells: hemavathy {len(hc)}, krs {len(kc)}, krs free-draining {len(free)}")
    print(kc[["cell_id", "straight_km", "flow_km", "hem_frac"]].round(1).to_string())

    # ---- 1. same model on both dams
    q_h, q_k = df["hemavathy_inflow"], df["krs_inflow"]
    rain_h = band_rain(df, hc, "straight_km")
    rain_k = band_rain(df, kc, "straight_km")
    # per-grid-cell routing (own delay + multiplier for every cell), see cell_response.py
    from cell_response import cell_model, free_cells
    cm_h = cell_model("hemavathy", df, hc, q_h, [])
    cm_k = cell_model("krs", df, free_cells(kc), q_k, ["harangi", "hemavathy"])
    routed_h, routed_k = cm_h["routed"], cm_k["routed"]
    rain_only = ["routed_rain_total", "routed_rain_wetfast"]
    Xh0, Xk0 = features(rain_h, df), features(rain_k, df, upstream=["harangi", "hemavathy"])
    Xh = Xh0.join(routed_h)
    Xk_rain = features(rain_k, df).join(routed_k[rain_only])
    Xk_full = Xk0.join(routed_k)
    daily, preds = {}, {}
    for key, X, q in [("hemavathy|Rain only", Xh, q_h),
                      ("hemavathy|Rain + past inflow", Xh.join(ar_feats(q_h)), q_h),
                      ("krs|Rain only", Xk_rain, q_k),
                      ("krs|Rain + upstream releases", Xk_full, q_k),
                      ("krs|Rain + upstream + past inflow", Xk_full.join(ar_feats(q_k)), q_k)]:
        daily[key], preds[key] = run_daily(X, q)
        print(key, daily[key]["all"], "| monsoon NSE", daily[key]["monsoon"]["NSE"])
    base_preds = {}
    for dam, q in [("hemavathy", q_h), ("krs", q_k)]:
        for k, (v, p) in baselines(q).items():
            daily[f"{dam}|{k}"] = {"all": v}
            base_preds[f"{dam}|{k}"] = p
    R["daily"] = daily
    R["baseline_peaks"] = {k: peaks(q_h if k.startswith("hem") else q_k, p) for k, p in base_preds.items()}
    hyd = {}
    for dam, q, keys in [("hemavathy", q_h, ["hemavathy|Rain only", "hemavathy|Rain + past inflow"]),
                         ("krs", q_k, ["krs|Rain + upstream releases", "krs|Rain + upstream + past inflow"])]:
        idx = preds[keys[0]].index
        idx = idx[idx.month.isin([6, 7, 8, 9, 10])]
        hyd[dam] = {"dates": [d.strftime("%Y-%m-%d") for d in idx],
                    "observed": [round(float(v)) for v in q[idx].fillna(0)]}
        for k in keys:
            hyd[dam][k.split("|")[1]] = [round(float(v)) for v in preds[k].reindex(idx).fillna(0)]
    R["hydrograph"] = hyd
    R["peaks"] = {k: peaks(q_h if k.startswith("hem") else q_k, p) for k, p in preds.items()}
    for k, v in R["peaks"].items():
        print("peaks", k, v)
    R["test_volume_ratio"] = {
        "hemavathy_inflow_cv": round(float(q_h[q_h.index.year >= 2020].std() / q_h[q_h.index.year >= 2020].mean()), 2),
        "krs_inflow_cv": round(float(q_k[q_k.index.year >= 2020].std() / q_k[q_k.index.year >= 2020].mean()), 2)}
    R["lag"] = {"hemavathy": lag_curve(rain_h, q_h), "krs_rain_only": lag_curve(rain_k, q_k)}
    print(json.dumps({k: {b: {x: y for x, y in v.items() if x != "curve"} for b, v in d.items()} for k, d in R["lag"].items()}))

    # ---- 2. total Jun-Sep inflow
    R["monsoon_inflow"] = {"hemavathy": monsoon_inflow(df, rain_h["basin"], "hemavathy"),
                           "krs": monsoon_inflow(df, rain_k["basin"], "krs", upstream=["harangi", "hemavathy"])}
    for dam, d in R["monsoon_inflow"].items():
        print(dam, "mean", d["mean_total_tmc"], "range", d["min_tmc"], d["max_tmc"])
        for r in d["by_issue"]:
            print(" ", r["issue_date"], "seen", r["already_observed_pct"], "clim", r["rmse_climatology_tmc"],
                  "climpct", r["climatology_median_abs_pct_err"], r["climatology_p80_abs_pct_err"],
                  {k[:14]: (v["median_abs_pct_err"], v["p80_abs_pct_err"], v["median_bias_pct"])
                   for k, v in r.items() if isinstance(v, dict)})

    R["lead_time"] = {"hemavathy": lead_time(rain_h, df, q_h, extra=routed_h),
                      "krs": lead_time(rain_k, df, q_k, ["harangi", "hemavathy"], extra=routed_k)}

    # ---- 3. river network experiments (test NSE, same protocol)
    exps = {}
    rain_hf = band_rain(df, hc, "flow_km")
    rain_kf = band_rain(df, kc, "flow_km")
    rain_free = band_rain(df, free, "flow_km")
    w = hc.set_index("cell_id").frac
    rain_reg = (df[w.index] * w).sum(axis=1) / w.sum()
    up = ["harangi", "hemavathy"]
    for key, X, q in [
        ("hemavathy|Straight-line bands", Xh0, q_h),
        ("hemavathy|River-distance bands", features(rain_hf, df), q_h),
        ("krs|Straight-line bands (current)", Xk0, q_k),
        ("krs|River-distance bands", features(rain_kf, df, up), q_k),
        ("krs|River-distance bands, free-draining catchment only", features(rain_free, df, up), q_k),
        ("krs|... + Hemavathy-catchment rain as its own input", features(rain_free, df, up, rain_reg), q_k),
        ("krs (rain only)|Straight-line bands", features(rain_k, df), q_k),
        ("krs (rain only)|River-distance bands", features(rain_kf, df), q_k),
    ]:
        m, p = run_daily(X, q)
        exps[key] = {"all": m["all"], "monsoon": m["monsoon"], "peaks": peaks(q, p)}
        print("river", key, m["all"]["NSE"], m["monsoon"]["NSE"], exps[key]["peaks"])
    R["river_experiments"] = exps
    R["lag"]["krs_rain_only_river_bands"] = lag_curve(rain_kf, q_k)
    R["lag"]["hemavathy_river_bands"] = lag_curve(rain_hf, q_h)
    print(json.dumps({k: {b: {x: y for x, y in v.items() if x != "curve"} for b, v in R["lag"][k].items()}
                      for k in ["krs_rain_only_river_bands", "hemavathy_river_bands"]}))
    (OUT / "dam_results.json").write_text(json.dumps(R, default=float))


if __name__ == "__main__":
    main()
