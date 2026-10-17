"""Does TerraClimate soil moisture (K-WRIS sub-watershed exports) improve the forecasts?

Inputs: data/ksr-soil-data.zip, data/hemavati-soil_data.zip. Monthly 1991-2023 per sub-watershed:
precipitation, runoff, reference and actual evapotranspiration, soil moisture, climate water deficit.
Sub-watersheds are averaged with equal weight (areas are not given); annual-only and duplicate
files are skipped. Only the last COMPLETE month before a forecast is used (no peeking ahead).

Tests, each with and without soil inputs on the same years:
  1. daily inflow model (final setup of dam_analysis.py), trained 1991-2019, tested 2020-2023
  2. monsoon-total forecast, leave-one-year-out over 1991-2023
Plus checks: TerraClimate rain vs IMD rain, soil moisture vs our rain-memory soil-wetness index,
TerraClimate runoff vs measured inflow. Writes outputs/soil_results.json.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from cell_response import cell_model, free_cells
from dam_analysis import (KRS_DAM, band_rain, cells_for, features, hemavathy_outlet, km, metrics, monsoon_inflow,
                          peaks, run_daily)
from inflow_model import api
from soil import load_terraclimate, soil_daily

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
TMC_PER_MM_KM2 = 1000 / 2.8317e7


def compare_daily(dam, X, q, soil):
    q23 = q.where(q.index.year <= 2023)
    out = {}
    for name, Xv in [("Without soil data", X.where(soil.notna().all(axis=1))), ("With soil data", X.join(soil))]:
        m, p = run_daily(Xv, q23)
        by_year = {int(y): metrics(q[g.index], g)["NSE"] for y, g in p.groupby(p.index.year)}
        out[name] = {"NSE": m["all"]["NSE"], "monsoon_NSE": m["monsoon"]["NSE"], "KGE": m["all"]["KGE"],
                     "bias_pct": m["all"]["bias_pct"], "nse_by_year": by_year, **peaks(q, p)}
        print(" ", dam, "daily", name, out[name])
    return out


def compare_monsoon(dam, df, rain, tc, upstream):
    def soil_at(year, t):
        prev = (t.to_period("M") - 1).to_timestamp()
        return {"sm": tc.sm.get(prev, np.nan), "sm_may": tc.sm.get(pd.Timestamp(year, 5, 1), np.nan),
                "cwd": tc.cwd.get(prev, np.nan)}
    res = {}
    for name, extra in [("Without soil data", None), ("With soil data", soil_at)]:
        r = monsoon_inflow(df, rain, dam, upstream, extra=extra, years=range(1991, 2024))
        res[name] = [{"issue": x["issue_date"], "median": x["No rain forecast"]["median_abs_pct_err"],
                      "p80": x["No rain forecast"]["p80_abs_pct_err"]} for x in r["by_issue"]]
    for a, b in zip(*res.values()):
        print(" ", dam, "monsoon", a["issue"], "typical", a["median"], "->", b["median"], "| 4-in-5", a["p80"], "->", b["p80"])
    return res


def checks(dam, df, rain, tc, area_km2):
    m_rain = rain.resample("MS").sum()
    wet = pd.Series(api(rain, 0.97), index=rain.index).resample("MS").mean()
    j = pd.DataFrame({"imd": m_rain, "tc_p": tc.precip, "api": wet, "sm": tc.sm}).dropna()
    anom = lambda s: s - s.groupby(s.index.month).transform("mean")
    q = df[f"{dam}_inflow"] * 86400 / 1e9
    seas = lambda s: (lambda m: m.groupby(m.index.year).sum())(s[s.index.month.isin([6, 7, 8, 9])])
    obs = seas(q.resample("MS").sum())
    tc_ro = seas(tc.runoff * area_km2 * TMC_PER_MM_KM2)
    jj = pd.concat([obs.rename("obs"), tc_ro.rename("tc")], axis=1).dropna()
    c = {"rain_corr_monthly": round(float(j.imd.corr(j.tc_p)), 2),
         "rain_ratio_tc_to_imd": round(float(j.tc_p.sum() / j.imd.sum()), 2),
         "soil_vs_wetness_index_corr": round(float(j.api.corr(j.sm)), 2),
         "soil_vs_wetness_index_corr_anomaly": round(float(anom(j.api).corr(anom(j.sm))), 2),
         "runoff_vs_inflow_corr_monsoon_total": round(float(jj.obs.corr(jj.tc)), 2),
         "runoff_to_inflow_ratio": round(float(jj.tc.sum() / jj.obs.sum()), 2),
         "monthly": [{"month": f"{t:%Y-%m}", "sm": round(float(r.sm), 1), "wetness": round(float(r.api), 1),
                      "rain": round(float(r.imd), 1)} for t, r in j.iterrows()],
         "monsoon": [{"year": int(y), "inflow": round(float(r.obs), 1), "tc_runoff": round(float(r.tc), 1)} for y, r in jj.iterrows()]}
    print(" ", dam, {k: v for k, v in c.items() if not isinstance(v, list)})
    return c


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
    kc["hem_frac"] = [box(lo - .125, la - .125, lo + .125, la + .125).intersection(ws_hem).area / 0.0625
                      for lo, la in zip(kc.lon, kc.lat)]
    area = {"krs": gpd.GeoSeries([ws_krs], crs=4326).to_crs(32643).area[0] / 1e6,
            "hemavathy": gpd.GeoSeries([ws_hem], crs=4326).to_crs(32643).area[0] / 1e6}

    R = {}
    for dam, zname, cells, q, up in [("hemavathy", "hemavati-soil_data.zip", hc, df["hemavathy_inflow"], []),
                                     ("krs", "ksr-soil-data.zip", kc, df["krs_inflow"], ["harangi", "hemavathy"])]:
        tc, codes = load_terraclimate(DATA / zname)
        print(dam, "sub-watersheds used:", codes, tc.index.min().date(), tc.index.max().date())
        rain = band_rain(df, cells, "straight_km")
        routed = cell_model(dam, df, free_cells(cells) if dam == "krs" else cells, q, up)["routed"]
        X = features(rain, df, up).join(routed)
        soil = soil_daily(tc, df.index)
        R[dam] = {"sub_watersheds": codes, "months": [f"{tc.index.min():%Y-%m}", f"{tc.index.max():%Y-%m}"],
                  "checks": checks(dam, df, rain["basin"], tc, area[dam]),
                  "daily": compare_daily(dam, X, q, soil),
                  "monsoon_total": compare_monsoon(dam, df, rain["basin"], tc, up)}
    (OUT / "soil_results.json").write_text(json.dumps(R, default=float))


if __name__ == "__main__":
    main()
