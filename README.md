# KRS inflow and end-of-monsoon storage forecast

WELL Labs Water Data Hackathon, Theme 2 (reservoir storage assessment using inflow forecasts), Question 3:

> Can we use past rainfall and reservoir storage data to forecast end-of-monsoon reservoir storage? How early can engineers make a reasonably informed estimate?

We answer it for **Krishnaraja Sagar (KRS)** on the Cauvery, with 30 September as the end of the monsoon. We use IMD 0.25° gridded rainfall over the KRS watershed and daily records for KRS and the two upstream dams, Harangi and Hemavathy, for 1991–2025.

**Report:** [`outputs/KRS_inflow_report.html`](outputs/KRS_inflow_report.html) is a single self-contained file. Download it and open it in any browser; it works offline. It has six tabs: data exploration, every data fix, the daily inflow model, the 30-Sep storage forecast, peak prediction, and the full method for reproducibility.

## Key results

All scores are on data the models never saw: 2020–2025 for daily inflow, leave-one-year-out over 1991–2025 for the season forecast.

**Daily inflow**
- Using only rainfall and upstream releases, the model scores Nash–Sutcliffe efficiency **0.90**. "Same as yesterday" scores 0.86. Adding the last week of KRS inflow raises the model to 0.93.
- Upstream releases reach KRS within **0–2 days**. Rain effects peak after about **2 days** and last about a week.
- The western (Ghats) grid cells produce about 3× more inflow per mm of rain than cells near the dam.
- Soil wetness (rain over the past weeks) is the most important input.

**Peaks**
- On days when inflow jumps by 5,000+ cusecs, the model predicts **87% of the jump on the same day**. "Same as yesterday" catches 0%.
- **1 day ahead:** peak size about 13% low (median), 65% of peaks within ±25%, 77% of high-flow days detected.
- **2 days ahead:** still says whether a peak is coming, but underestimates its size by about a third.
- **3 days or more:** needs a rainfall forecast.

**30-Sep storage, by forecast date**

| Forecast made on | Typical error (RMSE, TMC) | "Will KRS be ≥90% full?" correct |
|---|---|---|
| 1 Jun | 11.6 (no better than the long-term average) | 54% |
| 1 Aug | 9.3 | 69% |
| 15 Aug | 6.6 | 77% |
| 1 Sep | 4.0 | 83% |
| 15 Sep | 2.4 | 89% |

**A reasonably informed estimate is possible from mid-August.** Before that, the outcome depends on rain that has not fallen yet. KRS design capacity is 49.45 TMC.

## Repository layout

```
src/
  prepare.py           # cleaning: reservoirs + IMD rainfall clipped to the watershed
  inflow_model.py      # daily inflow models, lag-response (amount and delay) model
  season_forecast.py   # 30-Sep storage forecast by forecast date
  peak_analysis.py     # peak timing, size and high-flow detection
  lead_time.py         # predicting inflow 1-7 days ahead, with 80% intervals
  one_shot.py          # single end-of-monsoon forecast and "will KRS fill?" probability
  export_dashboard.py  # tidy tables for the report
  build_report.py      # builds outputs/KRS_inflow_report.html
  report/              # report page template, peak and approach tabs
  vendor/              # d3 v7.9.0 (embedded so the report works offline)
outputs/
  KRS_inflow_report.html
requirements.txt
```

## Data (not included)

The input data is not in this repository. To rerun, place it as follows:

| Path | Content | Source |
|---|---|---|
| `data/KR_reservior.xlsx`, `data/Hemavathy_reserviour.xlsx`, `data/Harangi_reserviour.xlsx` | Daily % full, gross storage (TMC), inflow and outflow (cusecs). Columns: `Date` (DD-MM-YYYY), `% of Storage as on this day`, `Gross Storage (TMC)`, `Inflow (Cusecs)`, `OutFlow (Cusecs)` | K-WRIS reservoir records |
| `data/IMD_Gridded_1991_2025/Rainfall_ind<year>_rfp25.grd` | IMD 0.25° daily gridded rainfall, one binary file per year | IMD Pune |
| `watershed_KSR/watershed.shp` (+ `.shx`, `.dbf`, `.prj`) | KRS catchment polygon | India-WRIS hydrological layers |

## Reproduce

Python 3.12. From the repository root:

```bash
uv venv .venv -p 3.12
uv pip install -p .venv -r requirements.txt

.venv/bin/python src/prepare.py          # -> data/processed/, outputs/data_fixes.json
.venv/bin/python src/inflow_model.py     # -> outputs/inflow_results.json
.venv/bin/python src/season_forecast.py  # -> outputs/season_results.json
.venv/bin/python src/peak_analysis.py    # -> outputs/peak_results.json
.venv/bin/python src/export_dashboard.py # -> outputs/dash/*.json
.venv/bin/python src/build_report.py     # -> outputs/KRS_inflow_report.html

# extra analyses (run from src/ so they can import the other scripts)
cd src && ../.venv/bin/python lead_time.py && ../.venv/bin/python one_shot.py
```

Random seeds are fixed, so a rerun gives the same numbers. Plain `pip install -r requirements.txt` in any Python 3.12 environment works too.

## Method in brief

1. **Cleaning.** We fix header offsets, missing days, storage above design capacity, single-day storage spikes and zero monsoon inflow, and interpolate gaps of 3 days or less. Water-balance mismatches are flagged but not changed. Every fix is listed in the report.
2. **Rainfall.** We keep the 24 IMD cells that overlap the watershed, weight each by the share of its area inside, and group them into near, mid and far bands by distance to the dam.
3. **Daily inflow.** Gradient boosting on log inflow. Inputs are rain by band over the last 7 days, 3–60 day rain totals, soil wetness, Harangi and Hemavathy releases over the last 5 days, and time of year. Training uses a time-based split: train 1991–2014, validate 2015–2019, test 2020–2025.
4. **Amount and delay.** A non-negative linear model of inflow on rain and upstream releases at delays of 0–10 days.
5. **30-Sep storage.** For each forecast date, a model predicts the storage change from that date to 30 Sep. Inputs are storage, inflow and releases so far, and rain so far. It is scored with leave-one-year-out cross-validation.

The **Approach & reproducibility** tab of the report has the full details and the limitations.

## Limitations

- Releases from KRS are operator decisions (irrigation, interstate water-sharing obligations) and cannot be predicted from rainfall.
- Upstream canal diversions are not in the data.
- Forecasts made before August need a seasonal monsoon outlook to add skill.
- Distance bands are only a proxy for how long water takes to reach the dam. A DEM-based sub-catchment split, and evapotranspiration data, are natural next steps.

## Contributing

Fork the repository, make your changes on a branch, and open a pull request against `main`. Please don't commit the input data.
