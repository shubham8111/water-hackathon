"""Forecasts issued on heavy-rain days, and the inflow volume of the next 15 days. Both dams.

1. Monsoon total (1 Jun - 30 Sep inflow, TMC) issued on the regular dates (1st and 15th) AND on every
   heavy-rain day. A heavy-rain day is the last day of a spell of days whose catchment-average rain
   is in the top 5% of June-September days (threshold from 1991-2019). Because these dates differ
   every year, one ridge model per date is replaced by a local model: for an issue day, a ridge is
   fitted on the other years' days within +-7 days of it (leave-one-year-out). On the regular dates
   this matches the per-date models of dam_analysis.py. Each heavy-rain update is compared with the
   last scheduled forecast, and with an update made on an ordinary day after the same wait.
2. Next 1, 7 and 15 days: total inflow (TMC) over the 15 days after the issue day, from today's state
   (recent inflow and rain, soil wetness, soil moisture, storage, upstream releases for KRS) with and
   without a rain forecast (actual rain of those days, a perfect forecast). Gradient boosting,
   fit on 1991-2019, tested on 2020-2023 issue days in June-September.
Writes outputs/event_results.json.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dam_analysis import (DATA, ISSUES, KRS_DAM, OUT, ROOT, TEST_END, band_rain, cells_for, hemavathy_outlet, km,
                          metrics)
from inflow_model import api
from soil import SOIL_ZIP, load_terraclimate, soil_daily

HEAVY_Q = 0.95
WINDOW = 7
H = 15


def to_tmc(q):
    return q * 86400 / 1e9  # cusec-days -> TMC (thousand million cubic feet)


def heavy_days(rain):
    """Last day of each heavy-rain spell (consecutive top-5% days), June to 15 September."""
    jjas = rain[rain.index.month.isin([6, 7, 8, 9])]
    thr = float(jjas[jjas.index.year <= 2019].quantile(HEAVY_Q))
    h = jjas[jjas >= thr]
    spell = (h.index.to_series().diff().dt.days.fillna(99) > 1).cumsum().values
    g = h.groupby(spell)
    ev = pd.DataFrame({"t": g.apply(lambda s: s.index[-1]), "start": g.apply(lambda s: s.index[0]),
                       "rain_mm": g.sum(), "days": g.size()})
    return ev[ev.t.dt.strftime("%m%d") <= "0915"].reset_index(drop=True), thr


def last_scheduled(t):
    return max(pd.Timestamp(t.year, m, d) for m, d in ISSUES if pd.Timestamp(t.year, m, d) <= t)


# ---------------------------------------------------------------- 1. monsoon total, any issue day
def season_rows(df, rain, dam, upstream, years):
    q = to_tmc(df[f"{dam}_inflow"].interpolate(limit=5).fillna(0))
    l7, r30, r3 = q.rolling(7).sum(), rain.rolling(30).sum(), rain.rolling(3).sum()
    rows = []
    for y in years:
        j1, s30 = pd.Timestamp(y, 6, 1), pd.Timestamp(y, 9, 30)
        days = pd.date_range(j1, pd.Timestamp(y, 9, 15))
        r = pd.DataFrame({"year": y, "t": days, "dos": np.arange(len(days)), "total": q[j1:s30].sum(),
                          "sofar": q[j1:s30].cumsum()[days].values, "last7": l7[days].values,
                          "rain_sofar": rain[j1:s30].cumsum()[days].values, "rain_last30": r30[days].values,
                          "rain_last3": r3[days].values, "rain_may": rain[pd.Timestamp(y, 5, 1):pd.Timestamp(y, 5, 31)].sum(),
                          "storage_pct": df[f"{dam}_pct"][days].values})
        for u in upstream:
            r[f"{u}_pct"] = df[f"{u}_pct"][days].values
        rows.append(r)
    R = pd.concat(rows, ignore_index=True)
    for c in R.columns[R.isna().any()]:
        R[c] = R[c].fillna(R.groupby("dos")[c].transform("median"))
    return R


def local_forecast(R, pts, feats):
    """Leave-one-year-out ridge on the other years' days within +-WINDOW days of each issue day."""
    out = np.zeros(len(pts))
    for k, (_, p) in enumerate(pts.iterrows()):
        tr = R[(R.year != p.year) & ((R.dos - p.dos).abs() <= WINDOW)]
        m = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(tr[feats], tr.total - tr.sofar)
        out[k] = p.sofar + max(m.predict(p[feats].to_frame().T.astype(float))[0], 0)
    return out


def pct_err(p, a):
    return np.abs(p - a) / a * 100


def season_events(df, rain, dam, upstream, years=range(1991, 2026)):
    R = season_rows(df, rain, dam, upstream, years)
    feats = ["sofar", "last7", "rain_sofar", "rain_last30", "rain_last3", "rain_may", "storage_pct"] + [f"{u}_pct" for u in upstream]
    R["forecast"] = local_forecast(R, R, feats)  # every day: the forecast as it would move through the season
    R["err"] = pct_err(R.forecast, R.total)
    ev, thr = heavy_days(rain)
    ev = ev[ev.t.dt.year.isin(list(years))]
    E = ev.merge(R[["t", "dos", "total", "forecast", "err"]], on="t")
    E["prev_t"] = E.t.map(last_scheduled)
    E = E.merge(R[["t", "forecast", "err"]].rename(columns={"t": "prev_t", "forecast": "prev_forecast", "err": "prev_err"}), on="prev_t")
    E["wait"] = (E.t - E.prev_t).dt.days
    E = E[E.wait > 0].copy()  # a heavy-rain day that is itself a regular date adds nothing
    E["gain"] = E.prev_err - E.err

    # control: updates on ordinary days after the same wait since the last scheduled date
    R["prev_t"] = R.t.map(last_scheduled)
    R["wait"] = (R.t - R.prev_t).dt.days
    C = R[~R.t.isin(set(ev.t)) & (R.wait > 0)].merge(
        R[["t", "err"]].rename(columns={"t": "prev_t", "err": "prev_err"}), on="prev_t")
    C["gain"] = C.prev_err - C.err
    wait_w = E.wait.value_counts(normalize=True)
    ctrl_gain = float(sum(C[C.wait == w].gain.median() * s for w, s in wait_w.items() if (C.wait == w).any()))
    ctrl_better = float(sum((C[C.wait == w].gain > 0).mean() * s for w, s in wait_w.items() if (C.wait == w).any()))

    month = {6: "June", 7: "July", 8: "August", 9: "September"}
    by_month = [{"month": month[m], "events": int(len(g)), "before": round(float(g.prev_err.median()), 1),
                 "after": round(float(g.err.median()), 1), "better_share": round(float((g.gain > 0).mean()), 2)}
                for m, g in E.groupby(E.t.dt.month)]
    regular = R[[(t.month, t.day) in ISSUES for t in R.t]]
    summary = {"threshold_mm": round(thr, 1), "events": int(len(E)), "per_year": round(len(E) / len(list(years)), 1),
               "median_wait_days": float(E.wait.median()),
               "err_last_scheduled_median": round(float(E.prev_err.median()), 1),
               "err_updated_median": round(float(E.err.median()), 1),
               "err_last_scheduled_p80": round(float(E.prev_err.quantile(.8)), 1),
               "err_updated_p80": round(float(E.err.quantile(.8)), 1),
               "better_share": round(float((E.gain > 0).mean()), 2),
               "median_gain_points": round(float(E.gain.median()), 1),
               "ordinary_day_median_gain_points": round(ctrl_gain, 1),
               "ordinary_day_better_share": round(ctrl_better, 2),
               "regular_p80": {f"{t:%d %b}": round(float(g.err.quantile(.8)), 1) for t, g in
                               regular.groupby(regular.t.dt.strftime("%m-%d").map(lambda s: pd.Timestamp(f"2000-{s}")))}}
    print(dam, "season events", {k: v for k, v in summary.items() if k != "regular_p80"})
    print(" ", by_month, summary["regular_p80"])
    traj = {int(y): {"actual": round(float(g.total.iloc[0]), 1),
                     "forecast": [round(float(v), 1) for v in g.forecast],
                     "rain": [round(float(v), 1) for v in rain[g.t.iloc[0]:g.t.iloc[-1]]],
                     "events": [{"dos": int(e.dos), "date": f"{e.t:%d %b}", "rain_mm": round(float(e.rain_mm)),
                                 "forecast": round(float(e.forecast), 1), "prev": round(float(e.prev_forecast), 1),
                                 "prev_date": f"{e.prev_t:%d %b}"} for e in E[E.t.dt.year == y].itertuples()]}
            for y, g in R.groupby("year")}
    return {"summary": summary, "by_month": by_month, "trajectory": traj}


# ---------------------------------------------------------------- 2. next 15 days
def next15_frame(df, rain, dam, upstream, soil):
    qc = df[f"{dam}_inflow"]
    qf = qc.interpolate(limit=5)
    fut = sum(qf.shift(-k) for k in range(1, H + 1))
    X = pd.DataFrame(index=df.index)
    X["inflow_today"] = np.log1p(qf)
    X["inflow_last7"] = np.log1p(qf.rolling(7).mean())
    X["inflow_last15"] = np.log1p(qf.rolling(15).mean())
    for w in [1, 3, 7, 14, 30]:
        X[f"rain_last{w}"] = rain["basin"].rolling(w).sum()
    for b in ["near", "mid", "far"]:
        X[f"rain_{b}_last3"] = rain[b].rolling(3).sum()
    X["wetness"] = api(rain["basin"], 0.97)
    wy = df.index.year - (df.index.month < 6)
    X["rain_since_jun1"] = rain["basin"].groupby(wy).cumsum()
    X["storage_pct"] = df[f"{dam}_pct"]
    for u in upstream:
        X[f"{u}_out_today"] = df[f"{u}_outflow"]
        X[f"{u}_out_last7"] = df[f"{u}_outflow"].rolling(7).mean()
        X[f"{u}_pct"] = df[f"{u}_pct"]
    X["doy"] = df.index.dayofyear
    X = X.join(soil)
    F = pd.DataFrame(index=df.index)  # rain forecast: actual rain of the next 15 days (perfect)
    nxt = lambda s, a, b: sum(s.shift(-k) for k in range(a, b + 1))
    F["rain_next_1_5"], F["rain_next_6_10"], F["rain_next_11_15"] = (nxt(rain["basin"], 1, 5), nxt(rain["basin"], 6, 10),
                                                                      nxt(rain["basin"], 11, 15))
    for b in ["near", "mid", "far"]:
        F[f"rain_{b}_next15"] = nxt(rain[b], 1, H)
    F7 = pd.DataFrame({"rain_next_1_7": nxt(rain["basin"], 1, 7)})
    return X, F, F7, fut


def gbm():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)


def next15(df, rain, dam, upstream, soil, events):
    X, F, F7, fut = next15_frame(df, rain, dam, upstream, soil)
    vol = to_tmc(fut)
    yrs, mon = df.index.year, df.index.month.isin([6, 7, 8, 9])
    ok = mon & X.notna().all(axis=1) & F.notna().all(axis=1) & vol.notna() & (vol > 0)
    tr, te = ok & (yrs <= 2019), ok & (yrs >= 2020) & (yrs <= TEST_END)
    y = np.log1p(fut)
    doy = df.index.dayofyear
    clim = fut[tr].groupby(doy[tr]).mean().rolling(15, center=True, min_periods=1).mean()
    preds = {"Average year": to_tmc(pd.Series(doy.map(clim).values, index=df.index)[te]),
             "Same as last 15 days": to_tmc(df[f"{dam}_inflow"].interpolate(limit=5).rolling(H).sum())[te]}
    for name, Xv in [("Today's state, no rain forecast", X), ("+ next 7 days of rain", X.join(F7)),
                     ("+ next 15 days of rain", X.join(F))]:
        m = gbm().fit(Xv[tr], y[tr])
        smear = np.mean(np.exp(y[tr] - m.predict(Xv[tr])))
        preds[name] = pd.Series(to_tmc(np.expm1(m.predict(Xv[te])) * smear), index=df.index[te])
    a = vol[te]
    is_ev = a.index.isin(events)
    res = {}
    for name, p in preds.items():
        e = pct_err(p, a)
        res[name] = {"median_pct": round(float(e.median()), 1), "p80_pct": round(float(e.quantile(.8)), 1),
                     "within_25": round(float((e <= 25).mean()), 2), "NSE": metrics(a, p)["NSE"],
                     "heavy_median_pct": round(float(e[is_ev].median()), 1),
                     "heavy_p80_pct": round(float(e[is_ev].quantile(.8)), 1),
                     "by_month": {m: round(float(g.median()), 1) for m, g in e.groupby(e.index.month)}}
        print(" ", dam, "next15", name, res[name])
    series = {int(yr): {"dates": [f"{t:%Y-%m-%d}" for t in g.index], "actual": [round(float(v), 2) for v in g],
                        **{n: [round(float(v), 2) for v in preds[n][g.index]] for n in preds},
                        "heavy": [f"{t:%Y-%m-%d}" for t in g.index if t in set(events)]}
              for yr, g in a.groupby(a.index.year)}
    return {"models": res, "series": series, "n_test_days": int(te.sum()), "n_heavy_test_days": int(is_ev.sum()),
            "mean_volume_tmc": round(float(a.mean()), 2)}


HORIZONS = [1, 7, 15]


def horizons(df, rain, dam, upstream, soil):
    """Inflow volume over the next 1, 7 and 15 days: average year, today's state only, and today's
    state + a (perfect) rain forecast for the same days. Same inputs, fit and test years as next15."""
    X = next15_frame(df, rain, dam, upstream, soil)[0]
    qf = df[f"{dam}_inflow"].interpolate(limit=5)
    nxt = lambda s, a, b: sum(s.shift(-k) for k in range(a, b + 1))
    yrs, mon, doy = df.index.year, df.index.month.isin([6, 7, 8, 9]), df.index.dayofyear
    out = {}
    for h in HORIZONS:
        fut = nxt(qf, 1, h)
        cuts = [(1, 1)] if h == 1 else [(1, 3), (4, 7)] if h == 7 else [(1, 5), (6, 10), (11, 15)]
        F = pd.DataFrame({f"rain_next_{a}_{b}": nxt(rain["basin"], a, b) for a, b in cuts})
        for b in ["near", "mid", "far"]:
            F[f"rain_{b}_next{h}"] = nxt(rain[b], 1, h)
        vol = to_tmc(fut)
        ok = mon & X.notna().all(axis=1) & F.notna().all(axis=1) & vol.notna() & (vol > 0)
        tr, te = ok & (yrs <= 2019), ok & (yrs >= 2020) & (yrs <= TEST_END)
        y = np.log1p(fut)
        clim = fut[tr].groupby(doy[tr]).mean().rolling(15, center=True, min_periods=1).mean()
        preds = {"Average year": to_tmc(pd.Series(doy.map(clim).values, index=df.index)[te])}
        for name, Xv in [("Today's state only", X), ("+ rain forecast for those days", X.join(F))]:
            m = gbm().fit(Xv[tr], y[tr])
            smear = np.mean(np.exp(y[tr] - m.predict(Xv[tr])))
            preds[name] = pd.Series(to_tmc(np.expm1(m.predict(Xv[te])) * smear), index=df.index[te])
        a = vol[te]
        out[h] = {"mean_volume_tmc": round(float(a.mean()), 2), "models": {}}
        for name, p in preds.items():
            e = pct_err(p, a)
            out[h]["models"][name] = {"median_pct": round(float(e.median()), 1), "p80_pct": round(float(e.quantile(.8)), 1),
                                      "within_25": round(float((e <= 25).mean()), 2), "NSE": metrics(a, p)["NSE"]}
            print(" ", dam, f"next {h} days", name, out[h]["models"][name])
    return out


def main():
    df = pd.read_parquet(DATA / "processed" / "daily.parquet")
    cell_ids = [c for c in df.columns if c.startswith("g_")]
    ws_krs = gpd.read_file(ROOT / "watershed_KSR" / "watershed.shp").to_crs(4326).union_all()
    ws_hem = gpd.read_file(f"zip://{DATA / 'watershed_hemavati.zip'}").to_crs(4326).union_all()
    riv_hem = gpd.read_file(f"zip://{DATA / 'rivers_hemavati.zip'}").to_crs(4326)
    hem_dam = hemavathy_outlet(riv_hem, ws_hem)
    kc, hc = cells_for(ws_krs, cell_ids), cells_for(ws_hem, cell_ids)
    kc["straight_km"] = km(kc.lon, kc.lat, *KRS_DAM)
    hc["straight_km"] = km(hc.lon, hc.lat, *hem_dam)
    R = {}
    for dam, cells, up in [("krs", kc, ["harangi", "hemavathy"]), ("hemavathy", hc, [])]:
        rain = band_rain(df, cells, "straight_km")
        soil = soil_daily(load_terraclimate(DATA / SOIL_ZIP[dam])[0], df.index)
        se = season_events(df, rain["basin"], dam, up)
        ev, _ = heavy_days(rain["basin"])
        R[dam] = {"season": se, "next15": next15(df, rain, dam, up, soil, ev.t), "horizons": horizons(df, rain, dam, up, soil)}
    R["test_years"] = [2020, TEST_END]
    (OUT / "event_results.json").write_text(json.dumps(R, default=float))


if __name__ == "__main__":
    main()
