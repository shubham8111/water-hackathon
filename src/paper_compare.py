"""Hemavathy only: our daily inflow model vs Chandre Gowda (2025), "Inflow prediction and modelling for a
seasonal flow reservoir", Water-Energy Nexus 8, 281-286 (doi:10.1016/j.wen.2025.08.003).

The paper's approach is rebuilt on our data:
  1. Catchment rain = area-weighted mean (the paper uses Thiessen weights for 4 gauges; we weight the
     IMD cells by their area inside the catchment).
  2. One formula per month, June-November, from rain on the day and the 2 days before (Kirpich time
     of concentration ~12 h -> 2-day lag): linear, polynomial (degree 2), logarithmic, exponential.
  3. Neural network: back-propagation, 2 inputs (2-day lag of streamflow), 14 tanh hidden neurons,
     linear output; days split at random 70% train / 15% validation / 15% test.
     Two readings of the inputs, since the paper does not list them exactly:
       "flows from the formula": polynomial-generated flow on the day and the day before (rain only)
       "measured flows": measured inflow 1 and 2 days before
  4. Scores: NSE, KGE, R2, RMSE, MAE (m3/s).
Each approach is scored two ways:
  paper test: random 15% of June-November days, 1991-2023 (neighbouring days are in training)
  our test:   all June-November days of 2020-2023, trained on 1991-2019 only (unseen years)
Our model is the report's daily model (distance-band rain, per-cell routed rain, soil moisture,
optionally last week's inflow), gradient boosting on log inflow.
Writes outputs/paper_compare.json.
"""
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from cell_response import cell_model
from dam_analysis import DATA, OUT, ROOT, ar_feats, band_rain, cells_for, features, hemavathy_outlet, km
from soil import SOIL_ZIP, load_terraclimate, soil_daily

warnings.filterwarnings("ignore", category=ConvergenceWarning)

M3S = 0.0283168  # cusecs -> m3/s
MONTHS = [6, 7, 8, 9, 10, 11]
MONTH_NAME = {6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}
PAPER = {  # reported in the paper: Tables 1-7 (NSE per month by formula; final neural network)
    "formulas_nse": {"Linear": [0.56, 0.53, 0.86, 0.48, 0.22, 0.27], "Polynomial": [0.60, 0.68, 0.87, 0.61, 0.46, 0.57],
                     "Logarithmic": [0.65, 0.53, 0.86, 0.48, 0.21, 0.26], "Exponential": [0.65, 0.50, 0.85, 0.46, 0.20, 0.27]},
    "nn": {"NSE": 0.890, "KGE": 0.927, "R2": 0.904, "RMSE": 1.481, "MAE": 0.939},
    "catchment_km2": 5410, "gauges": 4, "years": 40}


def scores(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    r = np.corrcoef(o, s)[0, 1]
    return {"NSE": round(float(1 - ((o - s) ** 2).sum() / ((o - o.mean()) ** 2).sum()), 3),
            "KGE": round(float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)), 3),
            "R2": round(float(r * r), 3), "RMSE": round(float(np.sqrt(((o - s) ** 2).mean()) * M3S), 1),
            "MAE": round(float(np.abs(o - s).mean() * M3S), 1), "bias_pct": round(float((s.mean() / o.mean() - 1) * 100), 1)}


class Formula:
    """The paper's four deterministic forms, on rain lags 0-2."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y):
        if self.kind == "Exponential":  # y = b0 * exp(b.x)  ->  log y linear in x
            self.m = LinearRegression().fit(X, np.log(np.maximum(y, 1)))
        else:
            self.m = self._pipe().fit(self._x(X), y)
        return self

    def _pipe(self):
        return make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()) if self.kind == "Polynomial" else LinearRegression()

    def _x(self, X):
        return np.log1p(X) if self.kind == "Logarithmic" else X

    def predict(self, X):
        if self.kind == "Exponential":
            return np.exp(np.clip(self.m.predict(X), 0, 12))
        return np.maximum(self.m.predict(self._x(X)), 0)


def nn(X, y, Xv, yv):
    """2-14-1 back-propagation network; best of a few random starts on the validation set (the paper
    optimised its network by Monte Carlo simulation)."""
    best = None
    for seed in range(5):
        net = MLPRegressor(hidden_layer_sizes=(14,), activation="tanh", solver="lbfgs", max_iter=2000, alpha=1e-3, random_state=seed)
        m = TransformedTargetRegressor(make_pipeline(StandardScaler(), net), transformer=StandardScaler()).fit(X, y)
        e = ((m.predict(Xv) - yv) ** 2).mean()
        if best is None or e < best[0]:
            best = (e, m)
    return best[1]


def main():
    df = pd.read_parquet(DATA / "processed" / "daily.parquet")
    df = df[df.index.year <= 2023]
    cell_ids = [c for c in df.columns if c.startswith("g_")]
    ws_hem = gpd.read_file(f"zip://{DATA / 'watershed_hemavati.zip'}").to_crs(4326).union_all()
    riv_hem = gpd.read_file(f"zip://{DATA / 'rivers_hemavati.zip'}").to_crs(4326)
    hem_dam = hemavathy_outlet(riv_hem, ws_hem)
    hc = cells_for(ws_hem, cell_ids)
    hc["straight_km"] = km(hc.lon, hc.lat, *hem_dam)
    q = df["hemavathy_inflow"]
    rain = band_rain(df, hc, "straight_km")

    P = pd.DataFrame({f"rain_lag{k}": rain["basin"].shift(k) for k in range(3)})
    yrs, mon = df.index.year, df.index.month
    ok = P.notna().all(axis=1) & q.notna() & q.shift(1).notna() & q.shift(2).notna() & mon.isin(MONTHS)
    rng = np.random.default_rng(0)
    idx = df.index[ok]
    split = pd.Series(rng.choice(["train", "val", "test"], size=len(idx), p=[.70, .15, .15]), index=idx)
    protocols = {"paper": {"train": split.index[split == "train"], "val": split.index[split == "val"], "test": split.index[split == "test"]},
                 "ours": {"train": idx[(idx.year <= 2014)], "val": idx[(idx.year >= 2015) & (idx.year <= 2019)], "test": idx[idx.year >= 2020]}}
    R = {"paper_reported": PAPER, "n_days": int(len(idx)), "protocols": {}}

    # our model (same inputs as the report's daily model), trained on all months
    routed = cell_model("hemavathy", df, hc, q, [])["routed"]
    soil = soil_daily(load_terraclimate(DATA / SOIL_ZIP["hemavathy"])[0], df.index)
    Xo = features(rain, df).join(routed).join(soil)
    Xa = Xo.join(ar_feats(q))
    allok = lambda X: X.notna().all(axis=1) & q.notna()

    for name, pr in protocols.items():
        tr, va, te = pr["train"], pr["val"], pr["test"]
        out = {"formulas": {}, "models": {}}
        # 1. one formula per month, rain lags 0-2
        gen = pd.Series(np.nan, index=df.index)  # polynomial-generated flows, for the network
        for kind in ["Linear", "Polynomial", "Logarithmic", "Exponential"]:
            nse_m = []
            for m in MONTHS:
                trm, tem = tr[tr.month == m], te[te.month == m]
                f = Formula(kind).fit(P.loc[trm].values, q[trm].values)
                nse_m.append(scores(q[tem], f.predict(P.loc[tem].values))["NSE"])
                if kind == "Polynomial":
                    every = df.index[P.notna().all(axis=1) & (mon == m)]
                    gen[every] = f.predict(P.loc[every].values)
            out["formulas"][kind] = nse_m
        # 2. the network, two readings of its inputs
        G = pd.DataFrame({"gen_t": gen, "gen_t1": gen.shift(1)})
        Q = pd.DataFrame({"q_t1": q.shift(1), "q_t2": q.shift(2)})
        preds = {}
        for label, X in [("Paper method: network on flows from the formula", G),
                         ("Paper method: network on measured flows (1-2 days before)", Q)]:
            t, v, s = [i[X.loc[i].notna().all(axis=1)] for i in (tr, va, te)]
            m = nn(X.loc[t].values, q[t].values, X.loc[v].values, q[v].values)
            preds[label] = pd.Series(np.maximum(m.predict(X.loc[s].values), 0), index=s)
        preds["Same as yesterday"] = q.shift(1)[te]
        # 3. our model: fit on train + val (as in the report), cap + Duan smearing on log inflow
        for label, X in [("Our model: rain + soil (no inflow readings)", Xo), ("Our model: + last week's inflow", Xa)]:
            fit_i = df.index.isin(tr.union(va)) | (~mon.isin(MONTHS) & (yrs <= (2023 if name == "paper" else 2019)))
            fit_i = fit_i & allok(X) & ~df.index.isin(te)
            y = np.log1p(q)
            g = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=30,
                                              l2_regularization=1.0, random_state=0).fit(X[fit_i], y[fit_i])
            cap = lambda z: np.clip(z, 0, y[fit_i].max())
            smear = np.mean(np.exp(y[fit_i] - cap(g.predict(X[fit_i]))))
            s = te[allok(X)[te]]
            preds[label] = pd.Series(np.expm1(cap(g.predict(X.loc[s]))) * smear, index=s)
        for label, p in preds.items():
            out["models"][label] = {**scores(q[p.index], p), "by_month": {MONTH_NAME[m]: scores(q[g.index], g)["NSE"] for m, g in p.groupby(p.index.month)},
                                    "n_test_days": int(len(p))}
            print(name, label, {k: v for k, v in out["models"][label].items() if k != "by_month"}, out["models"][label]["by_month"])
        print(name, "formulas NSE", out["formulas"])
        R["protocols"][name] = out
    (OUT / "paper_compare.json").write_text(json.dumps(R, default=float))


if __name__ == "__main__":
    main()
