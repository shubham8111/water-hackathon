"""Step 1: clean reservoir + IMD gridded rainfall data for the KRS watershed.

Outputs
  data/processed/daily.parquet   one row per day (1991-2025): reservoirs + rainfall
  data/processed/grid_cells.csv  watershed grid cells with weights / distance band
  outputs/data_fixes.json        every fix applied, for the dashboard
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "data" / "processed"
REPORT = ROOT / "outputs"

RESERVOIRS = {
    "krs": "KR_reservior.xlsx",
    "hemavathy": "Hemavathy_reserviour.xlsx",
    "harangi": "Harangi_reserviour.xlsx",
}
KRS_DAM = (76.575, 12.422)  # lon, lat
DESIGN_CAPACITY_TMC = {"krs": 49.45, "hemavathy": 37.10, "harangi": 8.50}
CUSEC_DAY_TO_TMC = 86400 / 1e9  # 1 cusec for one day = 86400 ft3 = 8.64e-5 TMC
MAX_INTERP_GAP = 3
COLS = {
    "% of Storage as on this day": "pct",
    "Gross Storage (TMC)": "storage",
    "Inflow (Cusecs)": "inflow",
    "OutFlow (Cusecs)": "outflow",
}

# IMD 0.25 deg grid
NLON, NLAT = 135, 129
LONS = 66.5 + 0.25 * np.arange(NLON)
LATS = 6.5 + 0.25 * np.arange(NLAT)

fixes = []


def log_fix(dataset, issue, action, count, detail=None):
    fixes.append({"dataset": dataset, "issue": issue, "action": action,
                  "count": int(count), "detail": detail})


def load_reservoir(name, fname):
    raw = pd.read_excel(DATA / fname, header=None)
    header_row = raw.index[raw[0].astype(str).str.strip() == "Date"][0]
    if header_row:
        log_fix(name, f"{header_row} title/blank rows above the header",
                "Detected header row and skipped preamble", header_row)
    df = raw.iloc[header_row + 1:].copy()
    df.columns = raw.iloc[header_row].astype(str).str.strip()
    df = df.rename(columns=COLS)

    df["date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%d-%m-%Y", errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        log_fix(name, "Unparseable dates", "Dropped rows", bad_dates)
    df = df.dropna(subset=["date"])
    for c in COLS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dups = df["date"].duplicated().sum()
    if dups:
        log_fix(name, "Duplicate dates", "Kept first occurrence", dups)
    df = df.drop_duplicates("date").set_index("date").sort_index()[list(COLS.values())]

    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing_days = full.difference(df.index)
    if len(missing_days):
        log_fix(name, "Missing calendar days", "Re-indexed to a continuous daily series",
                len(missing_days), [d.strftime("%Y-%m-%d") for d in missing_days[:10]])
    df = df.reindex(full)

    neg = (df[["inflow", "outflow"]] < 0).sum().sum()
    if neg:
        log_fix(name, "Negative flows", "Set to NaN", neg)
        df[["inflow", "outflow"]] = df[["inflow", "outflow"]].mask(df[["inflow", "outflow"]] < 0)

    # A storage jump is a reading error when it is far larger than inflow-outflow can explain
    # and reverses the next day (spike).
    ds = df["storage"].diff()
    spike = (ds.abs() > 3) & (ds.shift(-1).abs() > 3) & (np.sign(ds) != np.sign(ds.shift(-1)))
    if spike.sum():
        log_fix(name, "Single-day storage spikes (>3 TMC jump that reverses next day)",
                "Set to NaN and interpolated", spike.sum(),
                [d.strftime("%Y-%m-%d") for d in df.index[spike][:10]])
        df.loc[spike, ["storage", "pct"]] = np.nan

    over = df["storage"] > DESIGN_CAPACITY_TMC[name] * 1.01
    if over.sum():
        log_fix(name, f"Storage above design capacity ({DESIGN_CAPACITY_TMC[name]} TMC), up to {df['pct'].max():.0f}% full",
                "Set to NaN and interpolated", over.sum(),
                [d.strftime("%Y-%m-%d") for d in df.index[over][:10]])
        df.loc[over, ["storage", "pct"]] = np.nan

    # Zero inflow in the middle of the monsoon (Jun-Sep) while upstream rain is falling is
    # a recording gap, not a real zero.
    monsoon = df.index.month.isin([6, 7, 8, 9])
    zero_monsoon = monsoon & (df["inflow"] == 0)
    if zero_monsoon.sum():
        log_fix(name, "Zero inflow reported during Jun-Sep",
                "Treated as missing", zero_monsoon.sum(),
                [d.strftime("%Y-%m-%d") for d in df.index[zero_monsoon][:10]])
        df.loc[zero_monsoon, "inflow"] = np.nan
    zero_dry = (~monsoon & (df["inflow"] == 0)).sum()
    if zero_dry:
        log_fix(name, "Zero inflow outside monsoon", "Kept (plausible dry-season value)", zero_dry)

    for c in COLS.values():
        n_na = df[c].isna().sum()
        if not n_na:
            continue
        filled = df[c].interpolate(limit=MAX_INTERP_GAP, limit_area="inside")
        n_filled = n_na - filled.isna().sum()
        log_fix(name, f"Missing values in '{c}'",
                f"Linear interpolation for gaps <= {MAX_INTERP_GAP} days; longer gaps left NaN",
                n_na, f"{n_filled} filled, {filled.isna().sum()} left missing")
        df[c] = filled

    # Mass-balance check (reported only): change in storage vs. (inflow - outflow)
    resid = df["storage"].diff() - (df["inflow"] - df["outflow"]).shift(0) * CUSEC_DAY_TO_TMC
    big = resid.abs() > 1.0
    log_fix(name, "Water-balance residual > 1 TMC/day (storage change vs inflow - outflow)",
            "Flagged only (likely evaporation/diversions or rounding); not modified", big.sum())

    df.columns = [f"{name}_{c}" for c in df.columns]
    return df


def watershed_cells():
    ws = gpd.read_file(ROOT / "watershed_KSR" / "watershed.shp").to_crs(4326)
    poly = ws.union_all()
    minx, miny, maxx, maxy = poly.bounds
    rows = []
    for j, lat in enumerate(LATS):
        if lat < miny - 0.25 or lat > maxy + 0.25:
            continue
        for i, lon in enumerate(LONS):
            if lon < minx - 0.25 or lon > maxx + 0.25:
                continue
            cell = box(lon - 0.125, lat - 0.125, lon + 0.125, lat + 0.125)
            frac = cell.intersection(poly).area / cell.area
            if frac > 0.01:
                rows.append({"i": i, "j": j, "lon": lon, "lat": lat, "frac": frac})
    cells = pd.DataFrame(rows)
    # approximate km distance to KRS dam
    dx = (cells.lon - KRS_DAM[0]) * 111.32 * np.cos(np.radians(KRS_DAM[1]))
    dy = (cells.lat - KRS_DAM[1]) * 110.57
    cells["dist_km"] = np.hypot(dx, dy)
    cells["band"] = pd.qcut(cells.dist_km, 3, labels=["near", "mid", "far"]).astype(str)
    cells["cell_id"] = [f"g_{la:.2f}_{lo:.2f}" for la, lo in zip(cells.lat, cells.lon)]
    return cells, ws


def load_rainfall(cells):
    frames = []
    for year in range(1991, 2026):
        f = DATA / "IMD_Gridded_1991_2025" / f"Rainfall_ind{year}_rfp25.grd"
        arr = np.fromfile(f, dtype="<f4")
        ndays = arr.size // (NLON * NLAT)
        expected = 366 if pd.Timestamp(year, 12, 31).dayofyear == 366 else 365
        if ndays != expected:
            log_fix("rainfall", f"{year} file has {ndays} days, expected {expected}", "Truncated/aligned", 1)
        arr = arr[: ndays * NLON * NLAT].reshape(ndays, NLAT, NLON)
        sub = arr[:, cells.j.values, cells.i.values]
        frames.append(pd.DataFrame(sub, columns=cells.cell_id,
                                   index=pd.date_range(f"{year}-01-01", periods=ndays, freq="D")))
    rain = pd.concat(frames)
    n_missing = int((rain < -900).sum().sum())
    if n_missing:
        log_fix("rainfall", "IMD missing-value code -999 inside watershed", "Set to NaN, then filled with mean of other cells that day", n_missing)
    rain = rain.mask(rain < -900)
    if rain.isna().any().any():
        rain = rain.T.fillna(rain.mean(axis=1)).T
    neg = int((rain < 0).sum().sum())
    if neg:
        log_fix("rainfall", "Negative rainfall", "Clipped to 0", neg)
        rain = rain.clip(lower=0)
    return rain


def main():
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    res = pd.concat([load_reservoir(k, v) for k, v in RESERVOIRS.items()], axis=1)

    cells, ws = watershed_cells()
    log_fix("rainfall", f"Grid cells overlapping KRS watershed: {len(cells)}",
            "Weighted each cell by the fraction of its area inside the watershed", len(cells))
    rain = load_rainfall(cells)
    w = cells.set_index("cell_id").frac
    rain["rain_basin"] = (rain[w.index] * w).sum(axis=1) / w.sum()
    for band in ["near", "mid", "far"]:
        ids = cells.loc[cells.band == band, "cell_id"]
        rain[f"rain_{band}"] = (rain[ids] * w[ids]).sum(axis=1) / w[ids].sum()

    df = res.join(rain, how="inner")
    log_fix("merged", "Reservoir data runs to 2026-10-17 but rainfall ends 2025-12-31",
            "Modelling window restricted to 1991-01-01 .. 2025-12-31",
            len(res) - len(df))
    df.index.name = "date"
    df.to_parquet(OUT / "daily.parquet")
    res.to_parquet(OUT / "reservoirs_full.parquet")
    cells.to_csv(OUT / "grid_cells.csv", index=False)
    ws.to_file(OUT / "watershed.geojson", driver="GeoJSON")
    (REPORT / "data_fixes.json").write_text(json.dumps(fixes, indent=2, default=str))
    print(f"daily rows: {len(df)}, cells: {len(cells)}")
    for f in fixes:
        print(f"[{f['dataset']}] {f['issue']}: {f['count']} -> {f['action']}")


if __name__ == "__main__":
    main()
