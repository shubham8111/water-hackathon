"""Export tidy JSON row files for the dashboard (outputs/dash/*.json)."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
DASH = OUT / "dash"
TMC = 86400 / 1e9


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def dump(name, rows):
    (DASH / f"{name}.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(name, len(rows))


def main():
    DASH.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC / "daily.parquet")
    res = pd.read_parquet(PROC / "reservoirs_full.parquet")
    cells = pd.read_csv(PROC / "grid_cells.csv")
    inflow = json.loads((OUT / "inflow_results.json").read_text())
    season = json.loads((OUT / "season_results.json").read_text())

    fixes = json.loads((OUT / "data_fixes.json").read_text())
    dump("fixes", [{"id": i + 1, "dataset": f["dataset"], "issue": f["issue"], "action": f["action"],
                    "count": f["count"],
                    "detail": ", ".join(f["detail"]) if isinstance(f["detail"], list) else (f["detail"] or "")}
                   for i, f in enumerate(fixes)])

    yrs = range(1991, 2026)
    annual = []
    for y in yrs:
        d = df[df.index.year == y]
        jjas = d[d.index.month.isin([6, 7, 8, 9])]
        sep30 = pd.Timestamp(y, 9, 30)
        annual.append({
            "year": y,
            "rain_mm": r(d.rain_basin.sum(), 0), "rain_near_mm": r(d.rain_near.sum(), 0),
            "rain_mid_mm": r(d.rain_mid.sum(), 0), "rain_far_mm": r(d.rain_far.sum(), 0),
            "monsoon_rain_mm": r(jjas.rain_basin.sum(), 0),
            "krs_monsoon_inflow_tmc": r(jjas.krs_inflow.sum() * TMC, 1),
            "krs_sep30_tmc": r(res.at[sep30, "krs_storage"], 1),
            "system_sep30_tmc": r(sum(res.at[sep30, f"{d_}_storage"] for d_ in ["krs", "harangi", "hemavathy"]), 1),
        })
    dump("annual", annual)

    m = df.groupby(df.index.month)
    monthly_rain = df.rain_basin.groupby([df.index.year, df.index.month]).sum().groupby(level=1).mean()
    dump("monthly", [{"month": int(k), "rain_mm": r(monthly_rain[k], 0),
                      "krs_inflow_cusecs": r(m.krs_inflow.mean()[k], 0),
                      "harangi_outflow_cusecs": r(m.harangi_outflow.mean()[k], 0),
                      "hemavathy_outflow_cusecs": r(m.hemavathy_outflow.mean()[k], 0)} for k in range(1, 13)])

    yearly_cell = df[cells.cell_id].groupby(df.index.year).sum().mean()
    dump("cells", [{"cell_id": c.cell_id, "lon": c.lon, "lat": c.lat, "area_inside_pct": r(c.frac * 100, 0),
                    "dist_km": r(c.dist_km, 0), "band": c.band, "rain_mm_yr": r(yearly_cell[c.cell_id], 0)}
                   for c in cells.itertuples()])

    ws = gpd.read_file(PROC / "watershed.geojson").union_all().simplify(0.005)
    polys = [ws] if ws.geom_type == "Polygon" else list(ws.geoms)
    big = max(polys, key=lambda p: p.area)
    dump("watershed", [{"pt": i, "lon": r(x, 4), "lat": r(y, 4)} for i, (x, y) in enumerate(big.exterior.coords)])

    rows = []
    for scope, key in [("Test 2020-2025, all days", "metrics"), ("Test 2020-2025, Jun-Sep", "monsoon_metrics")]:
        for name, mm in inflow[key].items():
            rows.append({"model": name, "scope": scope, **{k: r(v, 3) for k, v in mm.items()}})
    dump("inflow_metrics", rows)

    h = inflow["hydrograph"]
    dump("hydrograph", [{"date": d, "observed": h["observed"][i],
                         "gb": h["Gradient boosting (rain + upstream)"][i],
                         "gb_ar": h["Gradient boosting (+ past KRS inflow)"][i],
                         "climatology": h["Climatology"][i]} for i, d in enumerate(h["dates"])])

    lr = inflow["lag_response"]["coefficients"]
    lag_rows = []
    for b, label in [("near", "Rain, near band"), ("mid", "Rain, mid band"), ("far", "Rain, far band (Ghats)")]:
        for k in range(len(lr[f"rain_{b}"])):
            lag_rows.append({"series": label, "kind": "rain", "lag_days": k,
                             "dry_soil": r(lr[f"rain_{b}"][k], 1),
                             "wet_soil": r(lr[f"rain_{b}"][k] + lr[f"rainwet_{b}"][k], 1)})
    for u, label in [("harangi", "Harangi release"), ("hemavathy", "Hemavathy release")]:
        for k, v in enumerate(lr[u]):
            lag_rows.append({"series": label, "kind": "release", "lag_days": k, "dry_soil": r(v, 3), "wet_soil": r(v, 3)})
    dump("lag_response", lag_rows)

    dump("importance", [{"feature_group": k, "nse_drop": v} for k, v in inflow["importance"].items()])

    issues = season["issue_dates"]
    skill, traj = [], []
    for target, label in [("system", "All 3 dams"), ("krs", "KRS only")]:
        t = season[target]
        cap = season["capacity"][target]
        clip = lambda v: r(min(max(v, 0), cap))
        for model, vals in t["by_issue"].items():
            for i, iss in enumerate(issues):
                skill.append({"target": label, "model": model, "issue_order": i, "issue_date": iss,
                              "rmse_tmc": vals["rmse"][i], "mae_tmc": vals["mae"][i]})
        for y, v in t["years"].items():
            for i, iss in enumerate(issues):
                traj.append({"target": label, "year": int(y), "issue_order": i, "issue_date": iss,
                             "storage_now_tmc": v["now"][i], "forecast_tmc": v["pred"][i],
                             "low_tmc": clip(v["pred"][i] + t["interval"]["p10"][i]),
                             "high_tmc": clip(v["pred"][i] + t["interval"]["p90"][i]),
                             "baseline_tmc": v["pred_clim"][i], "actual_sep30_tmc": v["actual"],
                             "model": "Ridge: storage + flows + rainfall"})
        if "y2026" in t:
            v = t["y2026"]
            for i, iss in enumerate(issues):
                traj.append({"target": label, "year": 2026, "issue_order": i, "issue_date": iss,
                             "storage_now_tmc": v["now"][i], "forecast_tmc": v["pred"][i],
                             "low_tmc": clip(v["pred"][i] + v["p10"][i]), "high_tmc": clip(v["pred"][i] + v["p90"][i]),
                             "baseline_tmc": None, "actual_sep30_tmc": v["actual"],
                             "model": "Ridge: storage + flows (no 2026 rainfall)"})
    dump("season_skill", skill)
    dump("season_years", traj)


if __name__ == "__main__":
    main()
