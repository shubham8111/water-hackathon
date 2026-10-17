"""Per-grid-cell rainfall response: how much inflow each IMD cell's rain produces, and how late.

For each dam, a non-negative linear model of daily monsoon inflow (Jun-Nov) on every cell's
wetness-scaled rain VOLUME (mm x km2 of the cell inside the catchment) at lag blocks 0, 1, 2, 3-4,
5-7 and 8-14 days, plus upstream releases for KRS. For KRS only the part of each cell outside the
Hemavathy catchment counts: rain above Hemavathy dam reaches KRS only through its (measured) releases.
Inputs are standardised so the ridge penalty shrinks poorly supported cells.
From each cell's coefficients:
  multiplier  share of the rain falling on that cell that reaches the dam (runoff share), and the
              extra inflow per mm of rain on the cell (cusec-days)
  delay       response-weighted mean lag (days) and the day of peak response
The cell responses are then fed to the gradient-boosting daily model as "routed rain" features
and compared, on 2020-2025, with the current distance-band features.
Writes outputs/cell_response.json.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from sklearn.linear_model import Ridge

from dam_analysis import (KRS_DAM, ar_feats, band_rain, cells_for, features, hemavathy_outlet,
                               km, metrics, peaks, run_daily)
from inflow_model import api

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
BLOCKS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 14)]
CENTRE = np.array([(a + b) / 2 for a, b in BLOCKS])
MONSOON = [6, 7, 8, 9, 10, 11]
CUSEC_DAY_M3 = 86400 * 0.0283168


def wetness(rain_basin):
    w = api(rain_basin, 0.97)
    return pd.Series(np.clip(w / np.nanpercentile(w, 90), 0, 2), index=rain_basin.index)


def cell_area_km2(c):
    return (0.25 * 111.32 * np.cos(np.radians(c.lat))) * (0.25 * 110.57) * c.frac


def design(df, cells, wet, upstream):
    """Lag-block design matrix: wetness-scaled cell rain volume + upstream releases + slow baseflow."""
    X = {}
    for _, c in cells.iterrows():
        cid = c.cell_id
        eff = df[cid] * cell_area_km2(c) * (0.25 + wet)  # mm*km2; more runoff on wet soil
        for a, b in BLOCKS:
            X[f"{cid}|{a}"] = eff.shift(a) if a == b else eff.rolling(b - a + 1).mean().shift(a)
    for u in upstream:
        for k in range(4):
            X[f"{u}|{k}"] = df[f"{u}_outflow"].shift(k)
    X["baseflow_api"] = pd.Series(api(df["basin_rain"], 0.99), index=df.index)
    return pd.DataFrame(X, index=df.index)


class ScaledRidge:
    """Positive ridge on standardised inputs; coef_ reported back in original units."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.s = X.std().replace(0, 1)
        self.r = Ridge(alpha=self.alpha, positive=True).fit(X / self.s, y)
        self.coef_ = self.r.coef_ / self.s.values
        return self

    def predict(self, X):
        return self.r.predict(X / self.s)


def fit(X, q, mask, alpha):
    m = mask & X.notna().all(axis=1) & q.notna()
    return ScaledRidge(alpha).fit(X[m], q[m])


def nse_on(model, X, q, mask):
    m = mask & X.notna().all(axis=1) & q.notna()
    return metrics(q[m], model.predict(X[m]))["NSE"]


def cell_model(dam, df, cells, q, upstream):
    """Fit the per-cell response; return per-cell multipliers/delays and routed-rain features."""
    w = cells.set_index("cell_id").frac
    df = df.copy()
    df["basin_rain"] = (df[w.index] * w).sum(axis=1) / w.sum()
    wet = wetness(df["basin_rain"])
    X = design(df, cells, wet, upstream)
    yrs, mon = df.index.year, df.index.month.isin(MONSOON)

    # alpha by validation (train <=2014, validate 2015-2019), then refit on <=2019 for the test
    scores = {a: nse_on(fit(X, q, mon & (yrs <= 2014), a), X, q, mon & (yrs >= 2015) & (yrs <= 2019))
              for a in [1, 10, 100, 300, 1000, 3000, 10000]}
    alpha = max(scores, key=scores.get)
    model = fit(X, q, mon & (yrs <= 2019), alpha)
    coef = pd.Series(model.coef_, index=X.columns)
    lin_test = nse_on(model, X, q, mon & (yrs >= 2020))
    print(dam, "alpha", alpha, {k: round(v, 3) for k, v in scores.items()}, "linear test NSE (Jun-Nov)", lin_test)

    wet_typ = float(0.25 + wet[mon].median())
    rows = []
    for _, c in cells.iterrows():
        b = np.array([coef[f"{c.cell_id}|{a}"] for a, _ in BLOCKS])
        area = cell_area_km2(c)
        # coefficients are cusec-days per (mm*km2): x 2446.6 m3 / 1000 m3 = share of rain volume
        share = b.sum() * wet_typ * CUSEC_DAY_M3 / 1000
        g = b.sum() * wet_typ * area  # cusec-days per mm on this cell
        per_lag = np.concatenate([np.full(bb - a + 1, v / (bb - a + 1)) for (a, bb), v in zip(BLOCKS, b)]) * wet_typ * area
        rows.append({"cell_id": c.cell_id, "lon": c.lon, "lat": c.lat, "frac": round(float(c.frac), 3),
                     "area_km2": round(float(area)),
                     "gain_cusec_days_per_mm": round(float(g), 1),
                     "runoff_share_pct": round(float(share * 100), 1),
                     "delay_days": round(float((b * CENTRE).sum() / b.sum()), 1) if b.sum() > 0 else None,
                     "peak_day": int(per_lag.argmax()) if b.sum() > 0 else None,
                     "curve": [round(float(v), 1) for v in per_lag],
                     "full_frac": round(float(c.get("full_frac", c.frac)), 3)})
    up = {u: {"gain": round(float(sum(coef[f"{u}|{k}"] for k in range(4))), 2),
              "curve": [round(float(coef[f"{u}|{k}"]), 3) for k in range(4)]} for u in upstream}

    # does the per-cell routing help the daily forecast? (same protocol as the existing models)
    contrib = pd.DataFrame({cid: sum(X[f"{cid}|{a}"] * coef[f"{cid}|{a}"] for a, _ in BLOCKS) for cid in cells.cell_id})
    share = pd.Series({r["cell_id"]: r["runoff_share_pct"] for r in rows})
    total_share = float((share * pd.Series({r["cell_id"]: r["area_km2"] for r in rows})).sum()
                        / sum(r["area_km2"] for r in rows))
    routed = pd.DataFrame(index=df.index)
    routed["routed_rain_total"] = contrib.sum(axis=1)
    routed["routed_rain_wetfast"] = sum(X[f"{cid}|0"] * coef[f"{cid}|0"] + X[f"{cid}|1"] * coef[f"{cid}|1"] for cid in cells.cell_id)
    routed["linear_model"] = model.predict(X.fillna(0))
    return {"alpha": alpha, "alpha_scores": scores, "linear_test_nse_monsoon": lin_test,
            "catchment_runoff_share_pct": round(total_share, 1), "typical_wetness_factor": round(wet_typ, 2),
            "cells": rows, "upstream": up, "routed": routed}


def free_cells(kc):
    """KRS cells reduced to the part outside the Hemavathy catchment (needs a hem_frac column)."""
    f = kc.assign(full_frac=kc.frac, frac=(kc.frac - kc.hem_frac).clip(lower=0))
    return f[f.frac > 0.02].reset_index(drop=True)


def analyse(dam, df, cells, q, upstream, outlet, ws, rivers, full_cells):
    cm = cell_model(dam, df, cells, q, upstream)
    routed = cm.pop("routed")
    rain = band_rain(df, full_cells, "straight_km")
    base = features(rain, df, upstream)
    rawcells = pd.DataFrame({f"{cid}_lag{k}": df[cid].shift(k) for cid in cells.cell_id for k in range(6)})
    comp = {}
    for name, Xg in [("Current: distance bands", base),
                     ("+ per-cell routed rain", base.join(routed)),
                     ("+ raw rain of every cell (lags 0-5)", base.join(rawcells)),
                     ("Current + past inflow", base.join(ar_feats(q))),
                     ("+ per-cell routed rain + past inflow", base.join(routed).join(ar_feats(q)))]:
        m, p = run_daily(Xg, q)
        comp[name] = {"NSE": m["all"]["NSE"], "monsoon_NSE": m["monsoon"]["NSE"], "KGE": m["all"]["KGE"], **peaks(q, p)}
        print(" ", dam, name, comp[name])

    simp = lambda g: [[round(x, 4), round(y, 4)] for x, y in g.simplify(0.004).coords]
    geo = {"outline": simp(ws.exterior), "dam": [round(outlet[0], 4), round(outlet[1], 4)],
           "rivers": [{"o": int(o), "c": simp(g)} for g, o in zip(rivers.geometry, rivers.sorder) if o >= 2]}
    return {**cm, "daily_comparison": comp, "geo": geo}


def main():
    df = pd.read_parquet(DATA / "processed" / "daily.parquet")
    cell_ids = [c for c in df.columns if c.startswith("g_")]
    ws_krs = gpd.read_file(ROOT / "watershed_KSR" / "watershed.shp").to_crs(4326).union_all()
    ws_hem = gpd.read_file(f"zip://{DATA / 'watershed_hemavati.zip'}").to_crs(4326).union_all()
    riv_krs = gpd.read_file(f"zip://{DATA / 'rivers_ksr.zip'}").to_crs(4326)
    riv_hem = gpd.read_file(f"zip://{DATA / 'rivers_hemavati.zip'}").to_crs(4326)
    hem_dam = hemavathy_outlet(riv_hem, ws_hem)
    kc, hc = cells_for(ws_krs, cell_ids), cells_for(ws_hem, cell_ids)
    kc["straight_km"] = km(kc.lon, kc.lat, *KRS_DAM)
    hc["straight_km"] = km(hc.lon, hc.lat, *hem_dam)
    kc["hem_frac"] = [box(lo - .125, la - .125, lo + .125, la + .125).intersection(ws_hem).area / 0.0625
                      for lo, la in zip(kc.lon, kc.lat)]
    # KRS: only the part of each cell that drains freely to KRS (outside the Hemavathy catchment)
    kfree = free_cells(kc)
    R = {"krs": analyse("krs", df, kfree, df["krs_inflow"], ["harangi", "hemavathy"], KRS_DAM, ws_krs, riv_krs, kc),
         "hemavathy": analyse("hemavathy", df, hc, df["hemavathy_inflow"], [], hem_dam, ws_hem, riv_hem, hc)}
    R["krs"]["held_by_hemavathy"] = [{"cell_id": c.cell_id, "lon": c.lon, "lat": c.lat, "hem_frac": round(float(c.hem_frac), 3)}
                                     for _, c in kc.iterrows() if c.hem_frac > 0.02]
    for dam, r in R.items():
        c = pd.DataFrame(r["cells"]).sort_values("gain_cusec_days_per_mm", ascending=False)
        print(dam, r["upstream"], "catchment runoff share", r["catchment_runoff_share_pct"])
        print(c[["cell_id", "area_km2", "gain_cusec_days_per_mm", "runoff_share_pct", "delay_days", "peak_day"]].to_string())
    (OUT / "cell_response.json").write_text(json.dumps(R, default=float))


if __name__ == "__main__":
    main()
