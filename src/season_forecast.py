"""Step 3: forecast 30-Sep storage from information available on an earlier forecast date.

One row per (year, forecast date). Target = storage change from forecast date to 30 Sep,
added back to current storage and capped at capacity. Evaluated with leave-one-year-out CV.
Targets: KRS + Harangi + Hemavathy combined ("system") and KRS alone.
Writes outputs/season_results.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs"
DAMS = ["krs", "harangi", "hemavathy"]
ISSUE_DATES = [(6, 1), (6, 15), (7, 1), (7, 15), (8, 1), (8, 15), (9, 1), (9, 15)]
TMC = 86400 / 1e9


def make_rows(df, res_full):
    rows = []
    for year in range(1991, 2027):
        end = pd.Timestamp(year, 9, 30)
        if end not in res_full.index:
            continue
        jun1 = pd.Timestamp(year, 6, 1)
        for m, d in ISSUE_DATES:
            t = pd.Timestamp(year, m, d)
            r = {"year": year, "issue": f"{t:%d %b}", "issue_doy": t.dayofyear}
            for dam in DAMS:
                r[f"{dam}_storage_now"] = res_full.at[t, f"{dam}_storage"]
                r[f"{dam}_storage_end"] = res_full.at[end, f"{dam}_storage"]
                r[f"{dam}_inflow_sofar"] = res_full.loc[jun1:t, f"{dam}_inflow"].sum() * TMC
                r[f"{dam}_inflow_last7"] = res_full.loc[t - pd.Timedelta(days=6):t, f"{dam}_inflow"].sum() * TMC
            r["krs_outflow_sofar"] = res_full.loc[jun1:t, "krs_outflow"].sum() * TMC
            r["krs_outflow_last7"] = res_full.loc[t - pd.Timedelta(days=6):t, "krs_outflow"].sum() * TMC
            r["system_storage_now"] = sum(r[f"{d}_storage_now"] for d in DAMS)
            r["system_storage_end"] = sum(r[f"{d}_storage_end"] for d in DAMS)
            r["system_inflow_sofar"] = sum(r[f"{d}_inflow_sofar"] for d in DAMS)
            if t in df.index:
                r["rain_sofar"] = df.loc[jun1:t, "rain_basin"].sum()
                r["rain_last14"] = df.loc[t - pd.Timedelta(days=13):t, "rain_basin"].sum()
                r["rain_may"] = df.loc[pd.Timestamp(year, 5, 1):pd.Timestamp(year, 5, 31), "rain_basin"].sum()
                r["rain_far_sofar"] = df.loc[jun1:t, "rain_far"].sum()
            rows.append(r)
    return pd.DataFrame(rows)


STORAGE_ONLY = ["{s}_storage_now", "system_inflow_sofar", "krs_inflow_sofar",
                "krs_inflow_last7", "krs_outflow_sofar", "krs_outflow_last7"]
WITH_RAIN = STORAGE_ONLY + ["rain_sofar", "rain_last14", "rain_may", "rain_far_sofar"]


def loyo(rows, target, feats, model_fn, capacity):
    """Leave-one-year-out predictions of 30-Sep storage."""
    pred = pd.Series(np.nan, index=rows.index)
    y = rows[f"{target}_storage_end"] - rows[f"{target}_storage_now"]
    for yr in rows.year.unique():
        tr, te = rows.year != yr, rows.year == yr
        if model_fn is None:  # persistence + mean change for this issue date
            mean_change = y[tr].groupby(rows.loc[tr, "issue"]).mean()
            p = rows.loc[te, "issue"].map(mean_change).values
        elif model_fn == "clim":
            p = rows.loc[tr, f"{target}_storage_end"].groupby(rows.loc[tr, "year"]).first().mean() \
                - rows.loc[te, f"{target}_storage_now"].values
        else:  # one model per issue date
            p = np.empty(te.sum())
            for k, iss in enumerate(rows.loc[te, "issue"]):
                tri = tr & (rows.issue == iss)
                m = model_fn().fit(rows.loc[tri, feats], y[tri])
                p[k] = m.predict(rows.loc[te & (rows.issue == iss), feats])[0]
        pred[te] = np.clip(rows.loc[te, f"{target}_storage_now"].values + p, 0, capacity)
    return pred


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "daily.parquet")
    res_full = pd.read_parquet(ROOT / "data" / "processed" / "reservoirs_full.parquet")
    for dam in DAMS:
        res_full[f"{dam}_inflow"] = res_full[f"{dam}_inflow"].fillna(0)
        res_full[f"{dam}_outflow"] = res_full[f"{dam}_outflow"].fillna(0)
        res_full[f"{dam}_storage"] = res_full[f"{dam}_storage"].interpolate(limit_area="inside")
    rows = make_rows(df, res_full)
    hist = rows[rows.year <= 2025].dropna().reset_index(drop=True)
    capacity = {d: float(res_full[f"{d}_storage"].max()) for d in DAMS}
    capacity["system"] = sum(capacity.values())

    ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    gbr = lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=2,
                                                min_samples_leaf=5, random_state=0)
    out = {"issue_dates": [f"{pd.Timestamp(2000, m, d):%d %b}" for m, d in ISSUE_DATES],
           "capacity": {k: round(v, 2) for k, v in capacity.items()}}
    for target in ["system", "krs"]:
        so = [f.format(s=target) for f in STORAGE_ONLY]
        wr = [f.format(s=target) for f in WITH_RAIN]
        models = {
            "Climatology (avg 30-Sep storage)": ("clim", None),
            "Current storage + avg change": (None, None),
            "Ridge: storage + flows": (ridge, so),
            "Ridge: storage + flows + rainfall": (ridge, wr),
            "Gradient boosting: storage + flows + rainfall": (gbr, wr),
        }
        actual = hist[f"{target}_storage_end"]
        tres = {"by_issue": {}, "years": {}}
        preds = {}
        for name, (fn, feats) in models.items():
            preds[name] = loyo(hist, target, feats, fn, capacity[target])
            err = (preds[name] - actual)
            g = err.groupby(hist.issue, sort=False)
            tres["by_issue"][name] = {
                "rmse": [round(float(np.sqrt((e ** 2).mean())), 2) for _, e in g],
                "mae": [round(float(e.abs().mean()), 2) for _, e in g],
                "p10": [round(float(e.quantile(0.1)), 2) for _, e in g],
                "p90": [round(float(e.quantile(0.9)), 2) for _, e in g],
            }
            print(target, name, tres["by_issue"][name]["rmse"])
        best = "Ridge: storage + flows + rainfall"
        err = preds[best] - actual
        for yr in sorted(hist.year.unique()):
            m = hist.year == yr
            tres["years"][int(yr)] = {
                "actual": round(float(actual[m].iloc[0]), 2),
                "now": [round(float(v), 2) for v in hist.loc[m, f"{target}_storage_now"]],
                "pred": [round(float(v), 2) for v in preds[best][m]],
                "pred_clim": [round(float(v), 2) for v in preds["Current storage + avg change"][m]],
                "rain_sofar": [round(float(v)) for v in hist.loc[m, "rain_sofar"]],
            }
        # 80% interval from LOYO residuals per issue date
        q = err.groupby(hist.issue, sort=False).quantile([0.1, 0.9]).unstack()
        tres["interval"] = {"p10": [round(float(v), 2) for v in q[0.1]], "p90": [round(float(v), 2) for v in q[0.9]]}

        # 2026 out-of-sample with the storage-only model (no 2026 rainfall available)
        r26 = rows[rows.year == 2026]
        if len(r26):
            p26 = []
            for _, r in r26.iterrows():
                h = hist[hist.issue == r.issue]
                m = ridge().fit(h[so], h[f"{target}_storage_end"] - h[f"{target}_storage_now"])
                p26.append(float(np.clip(r[f"{target}_storage_now"] + m.predict(r[so].to_frame().T.astype(float))[0], 0, capacity[target])))
            e_so = preds["Ridge: storage + flows"] - actual
            q26 = e_so.groupby(hist.issue, sort=False).quantile([0.1, 0.9]).unstack()
            tres["y2026"] = {"actual": round(float(r26[f"{target}_storage_end"].iloc[0]), 2),
                             "now": [round(float(v), 2) for v in r26[f"{target}_storage_now"]],
                             "pred": [round(float(v), 2) for v in p26],
                             "p10": [round(float(v), 2) for v in q26[0.1]],
                             "p90": [round(float(v), 2) for v in q26[0.9]]}
            print(target, "2026 actual", tres["y2026"]["actual"], "pred", tres["y2026"]["pred"])
        out[target] = tres
    (REPORT / "season_results.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
