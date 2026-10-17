"""Replay every past monsoon: an ensemble forecast of the June-September inflow (both dams).

Ensemble streamflow prediction (ESP), as used by operational river forecasters: on an issue date the
inflow already measured and today's state (last week's inflow, rain so far, May rain, last 30 days'
rain, storage) are fixed, and the rest of the season's rain is replaced, one by one, by the rest-of-
season rain of every other year on record. A ridge model learned with the actual rest-of-season rain
turns each replay into a seasonal total, plus that model's own error drawn from held-out years. The
spread of the replays is the range of plausible outcomes; its median is the point forecast.
Everything is leave-one-year-out: the year being forecast is never used for the model, its errors or
the replayed rain. Scores: median and 4-in-5 % error (vs the single ridge forecast of dam_analysis),
how often the 80% range contains the actual total, and the skill of the "dry season" probability
(total in the lowest third of years) over always saying 1 in 3 (Brier skill score).
Writes outputs/esp_results.json.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dam_analysis import DATA, ISSUES, KRS_DAM, OUT, ROOT, band_rain, cells_for, hemavathy_outlet, km

YEARS = range(1991, 2026)


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=5.0))


def season_table(df, rain, dam, upstream):
    q = df[f"{dam}_inflow"].interpolate(limit=5).fillna(0) * 86400 / 1e9  # TMC/day
    rows = []
    for y in YEARS:
        j1, s30 = pd.Timestamp(y, 6, 1), pd.Timestamp(y, 9, 30)
        for m, d in ISSUES:
            t = pd.Timestamp(y, m, d)
            r = {"year": y, "issue": f"{d} {t:%b}", "total": q[j1:s30].sum(), "sofar": q[j1:t].sum(),
                 "last7": q[t - pd.Timedelta(days=6):t].sum(), "rain_sofar": rain[j1:t].sum(),
                 "rain_last30": rain[t - pd.Timedelta(days=29):t].sum(),
                 "rain_may": rain[pd.Timestamp(y, 5, 1):pd.Timestamp(y, 5, 31)].sum(),
                 "storage_pct": df.at[t, f"{dam}_pct"], "rain_rest": rain[t + pd.Timedelta(days=1):s30].sum()}
            for u in upstream:
                r[f"{u}_pct"] = df.at[t, f"{u}_pct"]
            rows.append(r)
    R = pd.DataFrame(rows)
    num = R.columns.drop(["year", "issue"])
    R[num] = R[num].fillna(R.groupby("issue")[num].transform("median"))
    return R


def esp(h, base, i):
    """Ensemble for row i of one issue date's table h (other years only)."""
    tr = h.index[h.index != i]
    feats = base + ["rain_rest"]
    m = ridge().fit(h.loc[tr, feats], h.total[tr] - h.sofar[tr])
    # the model's own error, from held-out years inside the training years
    res = []
    for j in tr:
        tj = tr[tr != j]
        mj = ridge().fit(h.loc[tj, feats], h.total[tj] - h.sofar[tj])
        res.append((h.total[j] - h.sofar[j]) - mj.predict(h.loc[[j], feats])[0])
    X = pd.concat([h.loc[[i], base]] * len(tr), ignore_index=True)
    X["rain_rest"] = h.rain_rest[tr].values  # every other year's rest-of-season rain
    central = h.sofar[i] + np.maximum(m.predict(X), 0)
    full = h.sofar[i] + np.maximum(np.add.outer(m.predict(X), np.array(res)).ravel(), 0)
    single = ridge().fit(h.loc[tr, base], h.total[tr] - h.sofar[tr])
    return {"esp": float(np.median(central)), "lo": float(np.quantile(full, .1)), "hi": float(np.quantile(full, .9)),
            "dry_thr": float(h.total[tr].quantile(1 / 3)), "p_dry": float(np.mean(full < h.total[tr].quantile(1 / 3))),
            "ridge": float(h.sofar[i] + max(single.predict(h.loc[[i], base])[0], 0))}


def run(df, rain, dam, upstream):
    R = season_table(df, rain, dam, upstream)
    base = ["sofar", "last7", "rain_sofar", "rain_last30", "rain_may", "storage_pct"] + [f"{u}_pct" for u in upstream]
    out, years = [], {}
    for iss in [f"{d} {pd.Timestamp(2001, m, 1):%b}" for m, d in ISSUES]:
        h = R[R.issue == iss].reset_index(drop=True)
        E = pd.DataFrame([esp(h, base, i) for i in range(len(h))])
        a = h.total.values
        err = lambda p: np.abs(p - a) / a * 100
        dry = (a < E.dry_thr).astype(float)
        brier, brier_clim = np.mean((E.p_dry - dry) ** 2), np.mean((1 / 3 - dry) ** 2)
        r = {"issue": iss, "ridge_median": round(float(np.median(err(E.ridge))), 1), "ridge_p80": round(float(np.quantile(err(E.ridge), .8)), 1),
             "esp_median": round(float(np.median(err(E.esp))), 1), "esp_p80": round(float(np.quantile(err(E.esp), .8)), 1),
             "range_covers": round(float(np.mean((a >= E.lo) & (a <= E.hi))), 2),
             "range_width_pct": round(float(np.median((E.hi - E.lo) / E.esp * 100)), 0),
             "dry_brier_skill": round(float(1 - brier / brier_clim), 2),
             "dry_caught": f"{int(((E.p_dry >= .5) & (dry == 1)).sum())}/{int(dry.sum())}",
             "dry_false_alarms": int(((E.p_dry >= .5) & (dry == 0)).sum())}
        print(dam, r, flush=True)
        out.append(r)
        for k, y in enumerate(h.year):
            years.setdefault(int(y), {"actual": round(float(a[k]), 1), "issues": []})["issues"].append(
                {"issue": iss, "esp": round(float(E.esp[k]), 1), "lo": round(float(E.lo[k]), 1), "hi": round(float(E.hi[k]), 1),
                 "p_dry": round(float(E.p_dry[k]), 2), "ridge": round(float(E.ridge[k]), 1)})
    return {"by_issue": out, "years": years}


def main():
    df = pd.read_parquet(DATA / "processed" / "daily.parquet")
    cell_ids = [c for c in df.columns if c.startswith("g_")]
    ws_krs = gpd.read_file(ROOT / "watershed_KSR" / "watershed.shp").to_crs(4326).union_all()
    ws_hem = gpd.read_file(f"zip://{DATA / 'watershed_hemavati.zip'}").to_crs(4326).union_all()
    riv_hem = gpd.read_file(f"zip://{DATA / 'rivers_hemavati.zip'}").to_crs(4326)
    kc, hc = cells_for(ws_krs, cell_ids), cells_for(ws_hem, cell_ids)
    kc["straight_km"] = km(kc.lon, kc.lat, *KRS_DAM)
    hc["straight_km"] = km(hc.lon, hc.lat, *hemavathy_outlet(riv_hem, ws_hem))
    R = {"krs": run(df, band_rain(df, kc, "straight_km")["basin"], "krs", ["harangi", "hemavathy"]),
         "hemavathy": run(df, band_rain(df, hc, "straight_km")["basin"], "hemavathy", [])}
    (OUT / "esp_results.json").write_text(json.dumps(R))


if __name__ == "__main__":
    main()
