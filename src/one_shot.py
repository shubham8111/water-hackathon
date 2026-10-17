"""One-shot end-of-monsoon forecast: a single forecast issued once, scored per issue date.

For each issue date (leave-one-year-out over 1991-2025):
  - 30-Sep KRS storage: error, 80% interval width and coverage
  - "Will KRS be >= 90% full on 30 Sep?": logistic probability, Brier skill vs the base rate,
    accuracy at p=0.5
Writes outputs/one_shot_results.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from season_forecast import DAMS, ISSUE_DATES, make_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CAP = 49.45
FULL = 0.9 * CAP
FEATS = ["krs_storage_now", "system_storage_now", "system_inflow_sofar", "krs_inflow_sofar",
         "krs_inflow_last7", "krs_outflow_sofar", "rain_sofar", "rain_may"]


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "daily.parquet")
    res = pd.read_parquet(ROOT / "data" / "processed" / "reservoirs_full.parquet")
    for d in DAMS:
        res[f"{d}_inflow"] = res[f"{d}_inflow"].fillna(0)
        res[f"{d}_outflow"] = res[f"{d}_outflow"].fillna(0)
        res[f"{d}_storage"] = res[f"{d}_storage"].interpolate(limit_area="inside")
    rows = make_rows(df, res)
    hist = rows[rows.year <= 2025].dropna(subset=FEATS + ["krs_storage_end"]).reset_index(drop=True)
    hist["full"] = (hist.krs_storage_end >= FULL).astype(int)

    out = []
    for iss in hist.issue.unique():
        h = hist[hist.issue == iss].reset_index(drop=True)
        pred, prob, clim_p = np.zeros(len(h)), np.zeros(len(h)), np.zeros(len(h))
        for i in range(len(h)):
            tr = h.index != i
            y = h.krs_storage_end - h.krs_storage_now
            m = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(h.loc[tr, FEATS], y[tr])
            pred[i] = np.clip(h.at[i, "krs_storage_now"] + m.predict(h.loc[[i], FEATS])[0], 0, CAP)
            clim_p[i] = h.loc[tr, "full"].mean()
            if h.loc[tr, "full"].nunique() == 2:
                c = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)).fit(h.loc[tr, FEATS], h.loc[tr, "full"])
                prob[i] = c.predict_proba(h.loc[[i], FEATS])[0, 1]
            else:
                prob[i] = clim_p[i]
        err = pred - h.krs_storage_end
        q10, q90 = np.quantile(err, [0.1, 0.9])
        lo, hi = np.clip(pred - q90, 0, CAP), np.clip(pred - q10, 0, CAP)
        brier = np.mean((prob - h.full) ** 2)
        brier_clim = np.mean((clim_p - h.full) ** 2)
        r = {"issue_date": iss, "years": len(h),
             "rmse_tmc": round(float(np.sqrt(np.mean(err ** 2))), 2),
             "mae_tmc": round(float(np.mean(np.abs(err))), 2),
             "within_5tmc": round(float(np.mean(np.abs(err) <= 5)), 3),
             "interval_width_tmc": round(float(np.median(hi - lo)), 1),
             "coverage_80": round(float(np.mean((h.krs_storage_end >= lo) & (h.krs_storage_end <= hi))), 3),
             "full_years": int(h.full.sum()),
             "fill_accuracy": round(float(np.mean((prob >= 0.5) == h.full)), 3),
             "fill_brier_skill": round(float(1 - brier / brier_clim), 3),
             "dry_years_flagged": f"{int(((prob < 0.5) & (h.full == 0)).sum())}/{int((h.full == 0).sum())}"}
        out.append(r)
        print(r)
    (OUT / "one_shot_results.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
