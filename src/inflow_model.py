"""Step 2: daily KRS inflow model.

1. Lag-response model: positive linear model on raw inflow -> cusecs of KRS inflow produced by
   1 mm of rain (per distance band) or 1 cusec of upstream release, k days earlier.
2. Predictive models (log1p target): climatology, persistence, gradient boosting;
   with and without past KRS inflow. Time split: train 1991-2014, val 2015-2019, test 2020-2025.
Writes outputs/inflow_results.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs"
MAX_LAG = 10
BANDS = ["near", "mid", "far"]
UPSTREAM = ["harangi", "hemavathy"]


def api(series, k=0.9):
    out = np.zeros(len(series))
    acc = 0.0
    for t, r in enumerate(series.fillna(0).values):
        acc = acc * k + r
        out[t] = acc
    return out


def build_features(df):
    f = pd.DataFrame(index=df.index)
    for b in BANDS:
        for k in range(8):
            f[f"rain_{b}_lag{k}"] = df[f"rain_{b}"].shift(k)
    for w in [3, 7, 14, 30, 60]:
        f[f"rain_sum_{w}d"] = df["rain_basin"].rolling(w).sum()
    f["api_0.9"] = api(df["rain_basin"], 0.9)
    f["api_0.97"] = api(df["rain_basin"], 0.97)
    wy = df.index.year - (df.index.month < 6)
    f["rain_since_jun1"] = df["rain_basin"].groupby(wy).cumsum()
    for u in UPSTREAM:
        for k in range(6):
            f[f"{u}_out_lag{k}"] = df[f"{u}_outflow"].shift(k)
        f[f"{u}_pct"] = df[f"{u}_pct"]
    doy = df.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    ar = pd.DataFrame(index=df.index)
    for k in range(1, 8):
        ar[f"krs_in_lag{k}"] = np.log1p(df["krs_inflow"].shift(k))
    ar["krs_in_mean7"] = np.log1p(df["krs_inflow"].shift(1).rolling(7).mean())
    return f, ar


def metrics(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    nse = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    rmse = np.sqrt(np.mean((obs - sim) ** 2))
    mae = np.mean(np.abs(obs - sim))
    # seasonal volume error (Jun-Sep total), useful for the storage question
    return {"NSE": round(nse, 3), "KGE": round(kge, 3), "RMSE": round(rmse, 0),
            "MAE": round(mae, 0), "bias_pct": round((beta - 1) * 100, 1)}


def lag_response(df):
    """Non-negative linear model on monsoon days: inflow = sum_k a_k*rain_band(t-k) + b_k*release(t-k)."""
    X = pd.DataFrame(index=df.index)
    for b in BANDS:
        for k in range(MAX_LAG + 1):
            X[f"rain_{b}|{k}"] = df[f"rain_{b}"].shift(k)
    for u in UPSTREAM:
        for k in range(MAX_LAG + 1):
            X[f"{u}|{k}"] = df[f"{u}_outflow"].shift(k)
    # wetness interaction: the same rain yields more runoff once the soil is wet
    wet = api(df["rain_basin"], 0.97)
    wet = wet / np.nanpercentile(wet, 90)
    for b in BANDS:
        for k in range(MAX_LAG + 1):
            X[f"rainwet_{b}|{k}"] = X[f"rain_{b}|{k}"] * np.clip(wet, 0, 2)
    y = df["krs_inflow"]
    m = df.index.month.isin([6, 7, 8, 9, 10]) & X.notna().all(axis=1) & y.notna()
    model = Ridge(alpha=1.0, positive=True).fit(X[m], y[m])
    coef = pd.Series(model.coef_, index=X.columns)
    out = {}
    for name in [f"rain_{b}" for b in BANDS] + UPSTREAM:
        out[name] = [round(float(coef[f"{name}|{k}"]), 3) for k in range(MAX_LAG + 1)]
    for b in BANDS:
        out[f"rainwet_{b}"] = [round(float(coef[f"rainwet_{b}|{k}"]), 3) for k in range(MAX_LAG + 1)]
    pred = model.predict(X[m])
    r2 = 1 - np.sum((y[m] - pred) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
    summary = {}
    for name in UPSTREAM:
        c = np.array(out[name])
        summary[name] = {"total_gain": round(float(c.sum()), 2),
                         "peak_lag_days": int(c.argmax()),
                         "mean_lag_days": round(float((c * np.arange(len(c))).sum() / max(c.sum(), 1e-9)), 1)}
    for b in BANDS:
        c = np.array(out[f"rain_{b}"]) + np.array(out[f"rainwet_{b}"])
        summary[f"rain_{b}"] = {"total_cusecs_per_mm_wet": round(float(c.sum()), 1),
                                "peak_lag_days": int(c.argmax()),
                                "mean_lag_days": round(float((c * np.arange(len(c))).sum() / max(c.sum(), 1e-9)), 1)}
    return {"coefficients": out, "summary": summary, "r2_monsoon": round(float(r2), 3)}


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "daily.parquet")
    feats, ar = build_features(df)
    y = np.log1p(df["krs_inflow"])
    valid = feats.notna().all(axis=1) & ar.notna().all(axis=1) & y.notna()
    years = df.index.year
    split = {"train": (years <= 2014) & valid, "val": (years >= 2015) & (years <= 2019) & valid,
             "test": (years >= 2020) & valid}

    clim = df.loc[split["train"], "krs_inflow"].groupby(df.index[split["train"]].dayofyear).mean()
    clim = clim.rolling(15, center=True, min_periods=1).mean()
    preds = {"Climatology": df.index.dayofyear.map(clim).values,
             "Persistence (yesterday)": df["krs_inflow"].shift(1).values}

    variants = {
        "Gradient boosting (rain + upstream)": (HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1.0, random_state=0), feats),
        "Gradient boosting (+ past KRS inflow)": (HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1.0, random_state=0), feats.join(ar)),
    }
    trainval = split["train"] | split["val"]
    models = {}
    for name, (model, X) in variants.items():
        model.fit(X[split["train"]], y[split["train"]])
        val_nse = metrics(df.loc[split["val"], "krs_inflow"],
                          np.expm1(np.clip(model.predict(X[split["val"]]), 0, y.max())))["NSE"]
        print(f"{name}: val NSE {val_nse}")
        model.fit(X[trainval], y[trainval])  # refit on train+val before testing
        # cap log-predictions to the observed range, then Duan smearing for log back-transform bias
        cap = lambda z: np.clip(z, 0, y[trainval].max())
        smear = np.mean(np.exp(y[trainval] - cap(model.predict(X[trainval]))))
        p = np.full(len(df), np.nan)
        p[valid.values] = np.expm1(cap(model.predict(X[valid]))) * smear
        preds[name] = np.clip(p, 0, None)
        models[name] = (model, cap, smear)

    res = {"metrics": {}, "monsoon_metrics": {}, "season_volume": {}}
    test = split["test"].values
    mon = test & df.index.month.isin([6, 7, 8, 9])
    obs = df["krs_inflow"].values
    for name, p in preds.items():
        res["metrics"][name] = metrics(obs[test], p[test])
        res["monsoon_metrics"][name] = metrics(obs[mon], p[mon])
        # Jun-Sep total inflow per test year, TMC
        vol = pd.DataFrame({"obs": obs, "pred": p}, index=df.index)[mon]
        vol = vol.groupby(vol.index.year).sum() * 86400 / 1e9
        res["season_volume"][name] = {int(k): [round(a, 2), round(b, 2)] for k, (a, b) in vol.iterrows()}
        print(name, res["metrics"][name], "| monsoon", res["monsoon_metrics"][name])

    td = df.index[test]
    res["hydrograph"] = {"dates": [d.strftime("%Y-%m-%d") for d in td],
                         "observed": [None if np.isnan(v) else round(float(v)) for v in obs[test]]}
    for name in ["Climatology", "Gradient boosting (rain + upstream)", "Gradient boosting (+ past KRS inflow)"]:
        res["hydrograph"][name] = [None if np.isnan(v) else round(float(v)) for v in preds[name][test]]

    # grouped permutation-free importance: drop-group NSE loss on test for the main GB model
    gb, cap, smear = models["Gradient boosting (rain + upstream)"]
    X = feats
    predict = lambda Xs: np.expm1(cap(gb.predict(Xs))) * smear
    base = metrics(obs[test], predict(X[test]))["NSE"]
    groups = {"Rain (near band)": [c for c in X if c.startswith("rain_near")],
              "Rain (mid band)": [c for c in X if c.startswith("rain_mid")],
              "Rain (far band)": [c for c in X if c.startswith("rain_far")],
              "Rain totals 3-60d": [c for c in X if c.startswith("rain_sum")],
              "Soil wetness (API, since Jun 1)": [c for c in X if c.startswith("api") or c == "rain_since_jun1"],
              "Harangi release": [c for c in X if c.startswith("harangi")],
              "Hemavathy release": [c for c in X if c.startswith("hemavathy")],
              "Season (day of year)": ["doy_sin", "doy_cos"]}
    rng = np.random.default_rng(0)
    imp = {}
    for g, cols in groups.items():
        losses = []
        for _ in range(3):
            Xp = X[test].copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[perm]
            losses.append(base - metrics(obs[test], predict(Xp))["NSE"])
        imp[g] = round(float(np.mean(losses)), 3)
    res["importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    res["lag_response"] = lag_response(df)
    print("importance", res["importance"])
    print("lag summary", json.dumps(res["lag_response"]["summary"], indent=1))
    (REPORT / "inflow_results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
