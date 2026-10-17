"""How many days ahead can we predict KRS inflow peaks, and how confidently?

For lead h days: predict inflow on day t+h using only what is known at the end of day t.
Two variants:
  known-only    rain, upstream releases and KRS inflow up to day t
  + rain fcst   also the actual rain over t+1..t+h (a perfect rain forecast: upper bound
                on what an IMD forecast could add)
Train 1991-2014, 80% interval from 2015-2019 log-residual quantiles, test 2020-2025.
Writes outputs/lead_time_results.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from inflow_model import build_features, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LEADS = [1, 2, 3, 5, 7]


def model():
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=0)


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "daily.parquet")
    feats, ar = build_features(df)
    base = feats.join(ar)
    peaks = json.loads((OUT / "peak_results.json").read_text())
    p95 = peaks["thresholds"]["p95"]
    events = pd.to_datetime([e["date"] for e in peaks["events"]])
    years = df.index.year
    rows, examples = [], {}
    for h in LEADS:
        y = np.log1p(df["krs_inflow"].shift(-h))
        fut = pd.DataFrame(index=df.index)
        for b in ["basin", "near", "mid", "far"]:
            fut[f"fut_rain_{b}"] = df[f"rain_{b}"].rolling(h).sum().shift(-h)
        fut["fut_rain_max_day"] = df["rain_basin"].rolling(h).max().shift(-h)
        for variant, X in [("Known on the day", base), ("+ perfect rain forecast", base.join(fut))]:
            ok = X.notna().all(axis=1) & y.notna()
            tr, va, te = ok & (years <= 2014), ok & (years >= 2015) & (years <= 2019), ok & (years >= 2020)
            m = model().fit(X[tr], y[tr])
            cap = lambda z: np.clip(z, 0, y[tr].max())
            res_va = y[va] - cap(m.predict(X[va]))
            smear = np.mean(np.exp(res_va))
            q10, q90 = np.quantile(res_va, [0.1, 0.9])
            z = cap(m.predict(X[te]))
            pred = pd.Series(np.expm1(z) * smear, index=X.index[te])
            lo = pd.Series(np.expm1(z + q10), index=pred.index)
            hi = pd.Series(np.expm1(z + q90), index=pred.index)
            obs = df["krs_inflow"].shift(-h)[te]
            # index by target day (t+h)
            tgt = pred.index + pd.Timedelta(days=h)
            pred.index = lo.index = hi.index = obs.index = tgt
            ev = [t for t in events if t in pred.index]
            pe = np.array([(pred[t] - obs[t]) / obs[t] * 100 for t in ev])
            o, p = obs >= p95, pred >= p95
            hits, miss, fa = int((o & p).sum()), int((o & ~p).sum()), int((~o & p).sum())
            cover = float(((obs >= lo) & (obs <= hi)).mean())
            cover_pk = float(np.mean([(lo[t] <= obs[t] <= hi[t]) for t in ev]))
            width_pk = float(np.median([(hi[t] - lo[t]) / obs[t] * 100 for t in ev]))
            r = {"lead_days": h, "variant": variant, **{k: float(v) for k, v in metrics(obs, pred).items()},
                 "peak_median_err_pct": round(float(np.median(pe)), 1),
                 "peak_within_25pct": round(float(np.mean(np.abs(pe) <= 25)), 3),
                 "POD": round(hits / max(hits + miss, 1), 3), "FAR": round(fa / max(hits + fa, 1), 3),
                 "coverage_80": round(cover, 3), "peak_coverage_80": round(cover_pk, 3),
                 "peak_interval_width_pct": round(width_pk, 0)}
            rows.append(r)
            print(r)
            # example: the biggest test event, window around it
            big = max(ev, key=lambda t: obs[t])
            win = pd.date_range(big - pd.Timedelta(days=15), big + pd.Timedelta(days=15))
            examples[f"{h}|{variant}"] = [
                {"date": d.strftime("%Y-%m-%d"), "observed": round(float(obs.get(d, np.nan))) if d in obs.index else None,
                 "pred": round(float(pred[d])) if d in pred.index else None,
                 "lo": round(float(lo[d])) if d in lo.index else None,
                 "hi": round(float(hi[d])) if d in hi.index else None} for d in win]
    (OUT / "lead_time_results.json").write_text(json.dumps({"rows": rows, "examples": examples, "p95": p95}))


if __name__ == "__main__":
    main()
