"""TerraClimate monthly soil data (K-WRIS sub-watershed exports) as daily model inputs.

Sub-watersheds are averaged with equal weight (areas are not given); annual-only and duplicate
files are skipped. Only the last COMPLETE month before a day is used (no peeking ahead).
"""
import io
import re
import zipfile

import pandas as pd

COLS = {"Precipitation accumulation(mm)": "precip", "Runoff(mm)": "runoff", "Reference evapotranspiration(mm)": "et0",
        "Actual evapotranspiration(mm)": "aet", "Soil moisture(mm)": "sm", "Climate water deficit(mm)": "cwd"}


def load_terraclimate(zpath):
    frames, seen = [], set()
    with zipfile.ZipFile(zpath) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".xlsx")):
            raw = pd.read_excel(io.BytesIO(z.read(name)), header=None)
            code = re.search(r"Watershed : (\w+)", str(raw.iat[0, 0])).group(1)
            hdr = raw.index[raw[0].astype(str).str.strip() == "SNo"][0]
            t = raw.iloc[hdr + 1:].copy()
            t.columns = raw.iloc[hdr].astype(str).str.strip()
            if code in seen or not t["Date"].astype(str).str.contains("/").all():
                continue  # duplicate sub-watershed, or annual-only file
            seen.add(code)
            t["month"] = pd.to_datetime(t["Date"].astype(str), format="%m/%Y")
            t = t.set_index("month")[list(COLS)].rename(columns=COLS).astype(float)
            frames.append(t.assign(ws=code))
    allws = pd.concat(frames)
    return allws.drop(columns="ws").groupby(level=0).mean().sort_index(), sorted(seen)


def soil_daily(tc, index, train_end=2019):
    """Daily features from the last complete month: soil moisture, its anomaly, deficit, actual ET."""
    tr = tc[tc.index.year <= train_end]
    clim = tr.sm.groupby(tr.index.month).mean()
    prev = (index.to_period("M") - 1).to_timestamp()
    f = pd.DataFrame(index=index)
    f["soil_moisture_prev_month"] = tc.sm.reindex(prev).values
    f["soil_moisture_anomaly"] = f.soil_moisture_prev_month - prev.month.map(clim).values
    f["water_deficit_prev_month"] = tc.cwd.reindex(prev).values
    f["actual_et_prev_month"] = tc.aet.reindex(prev).values
    return f



SOIL_ZIP = {"krs": "ksr-soil-data.zip", "hemavathy": "hemavati-soil_data.zip"}
