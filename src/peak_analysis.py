"""Peak analysis: how well do the daily models capture high-inflow events on 2020-2025?

Peak events = independent local maxima of observed inflow above the training-period 95th
percentile, at least 7 days apart. For each event we compare the model's highest value within
+/-3 days (magnitude and timing). We also score threshold exceedance day by day (POD, FAR, CSI).
Writes outputs/peak_results.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WINDOW = 3
MIN_SEP = 7
RISE = 5000  # cusecs/day jump that counts as a sharp rise
MODELS = {"gb": "Rain + upstream", "gb_ar": "+ past KRS inflow", "persistence": "Same as yesterday"}


def find_events(obs, threshold):
    is_peak = (obs >= threshold) & (obs == obs.rolling(2 * MIN_SEP + 1, center=True, min_periods=1).max())
    return obs.index[is_peak]


def contingency(obs, pred, thr):
    o, p = obs >= thr, pred >= thr
    hits, misses, fa = int((o & p).sum()), int((o & ~p).sum()), int((~o & p).sum())
    return {"threshold": thr, "observed_days": int(o.sum()), "hits": hits, "misses": misses, "false_alarms": fa,
            "POD": round(hits / max(hits + misses, 1), 3), "FAR": round(fa / max(hits + fa, 1), 3),
            "CSI": round(hits / max(hits + misses + fa, 1), 3)}


def main():
    inflow = json.loads((OUT / "inflow_results.json").read_text())
    h = inflow["hydrograph"]
    df = pd.DataFrame({"observed": h["observed"], "gb": h["Gradient boosting (rain + upstream)"],
                       "gb_ar": h["Gradient boosting (+ past KRS inflow)"]},
                      index=pd.to_datetime(h["dates"]), dtype=float)
    df["persistence"] = df["observed"].shift(1)
    daily = pd.read_parquet(ROOT / "data" / "processed" / "daily.parquet")
    train = daily.loc[:"2019", "krs_inflow"]
    p95, p99 = float(train.quantile(0.95)), float(train.quantile(0.99))

    obs = df["observed"]
    events = find_events(obs, p95)
    rows = []
    for t in events:
        win = df.loc[t - pd.Timedelta(days=WINDOW):t + pd.Timedelta(days=WINDOW)]
        r = {"date": t.strftime("%Y-%m-%d"), "observed": round(obs[t])}
        for k in MODELS:
            s = win[k].dropna()
            if s.empty:
                continue
            r[k] = round(float(s.max()))
            r[f"{k}_sameday"] = None if pd.isna(df.at[t, k]) else round(float(df.at[t, k]))
            r[f"{k}_sameday_err_pct"] = None if pd.isna(df.at[t, k]) else round((df.at[t, k] - obs[t]) / obs[t] * 100, 1)
            r[f"{k}_err_pct"] = round((s.max() - obs[t]) / obs[t] * 100, 1)
            r[f"{k}_lag_days"] = int((s.idxmax() - t).days)
        rows.append(r)
    ev = pd.DataFrame(rows)

    summary = []
    for k, label in MODELS.items():
        e = ev[f"{k}_sameday_err_pct"].dropna()
        lag = ev[f"{k}_lag_days"]
        # rise capture: on days inflow jumps by >= RISE cusecs, share of the jump the model predicted
        d = df[["observed", k]].dropna()
        prev = df["observed"].shift(1).reindex(d.index)
        jump = d.observed - prev
        sel = jump >= RISE
        capture = ((d[k] - prev)[sel] / jump[sel]).clip(-1, 2)
        big = ev[ev.observed >= p99]
        summary.append({
            "model": label, "events": len(ev),
            "median_peak_error_pct": round(float(e.median()), 1),
            "mean_abs_peak_error_pct": round(float(e.abs().mean()), 1),
            "within_25pct": round(float((e.abs() <= 25).mean()), 3),
            "big_events": len(big),
            "big_median_peak_error_pct": round(float(big[f"{k}_sameday_err_pct"].median()), 1) if len(big) else None,
            "rise_days": int(sel.sum()),
            "median_rise_captured_pct": round(float(capture.median() * 100), 0),
            "window_mean_abs_error_pct": round(float(ev[f"{k}_err_pct"].abs().mean()), 1),
            "on_time_share": round(float((lag == 0).mean()), 3),
            "mean_timing_days": round(float(lag.mean()), 2),
        })

    cont = []
    for thr_name, thr in [("95th pct", p95), ("99th pct", p99)]:
        for k, label in MODELS.items():
            m = df[["observed", k]].dropna()
            cont.append({"model": label, "level": thr_name, **contingency(m.observed, m[k], round(thr))})

    # flow-duration view: error by observed flow band
    bands = [0, 2000, 5000, 10000, 20000, 40000, np.inf]
    labels = ["<2k", "2-5k", "5-10k", "10-20k", "20-40k", ">40k"]
    df["band"] = pd.cut(df.observed, bands, labels=labels, right=False)
    by_band = []
    for k, label in MODELS.items():
        for b in labels:
            m = df[df.band == b][["observed", k]].dropna()
            if len(m):
                by_band.append({"model": label, "flow_band": b, "days": len(m),
                                "bias_pct": round(float((m[k].sum() - m.observed.sum()) / m.observed.sum() * 100), 1),
                                "mape_pct": round(float((abs(m[k] - m.observed) / m.observed.clip(lower=1)).median() * 100), 1)})

    res = {"thresholds": {"p95": round(p95), "p99": round(p99)}, "window_days": WINDOW, "rise_cusecs": RISE,
           "events": rows, "summary": summary, "contingency": cont, "by_band": by_band}
    (OUT / "peak_results.json").write_text(json.dumps(res))
    print("thresholds", res["thresholds"], "events", len(ev))
    for s in summary:
        print(s)
    for c in cont:
        print(c["model"], c["level"], "POD", c["POD"], "FAR", c["FAR"], "CSI", c["CSI"])
    for b in by_band:
        print(b)


if __name__ == "__main__":
    main()
