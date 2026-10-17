# Monsoon inflow outlook: KRS and Hemavathy

WELL Labs Water Data Hackathon, Theme 2 (reservoir storage assessment using inflow forecasts), Question 3:

> Can we use past rainfall and reservoir storage data to forecast end-of-monsoon reservoir storage? How early can engineers make a reasonably informed estimate?

We answer it for two Cauvery dams, **Krishnaraja Sagar (KRS)** and **Hemavathy**, by forecasting the **total June–September inflow**. We use IMD 0.25° gridded daily rainfall over each catchment, daily dam records for KRS, Hemavathy and Harangi (1991–2025), and TerraClimate monthly soil moisture (1991–2023).

## Final results

- [`outputs/final_report.html`](outputs/final_report.html): one page, both dams, the three questions and their answers.
- [`outputs/presentation.html`](outputs/presentation.html): 12 slides for the hackathon (← → to move).

| Question | Answer |
|---|---|
| 1. How much water will the monsoon bring? | June–September inflow decides storage: KRS ends September ≥90% full in 19 of 21 years above 86 TMC of inflow. We forecast it with an 80% likely range by replaying every past monsoon's remaining rain. |
| 2. How early can engineers know? | From **15 August** at both dams (4 years in 5 within ±20%; NSE 0.90 KRS, 0.93 Hemavathy), or 1 August with a 30-day rain forecast (NSE 0.91 / 0.94). Before July NSE is about 0, no better than assuming an average year. A dry season is flagged by 15 August (10 of 12 dry years at KRS, 11 of 12 at Hemavathy). |
| 3. How much water will arrive in the next day, 7 days and 15 days? | Typical error (NSE), KRS / Hemavathy: **next day ±19% (0.88) / ±20% (0.91)** from today's data alone; **next 7 days ±22% (0.93) / ±27% (0.76)** and **next 15 days ±19% (0.93) / ±22% (0.88)** with a rain forecast for those days (without one: 7 days ±39% (0.72) / ±48% (0.59); 15 days ±48% (0.44) / ±44% (0.36)). |

## Detailed reports

Two self-contained, chart-led reports. Each is a single file that works offline: download it and open it in any browser. GitHub does not render HTML.

- [`outputs/KRS_insights.html`](outputs/KRS_insights.html)
- [`outputs/Hemavathy_insights.html`](outputs/Hemavathy_insights.html)

Each report covers:
1. **The monsoon, year by year.** Rainfall, inflow, outflow and 30-Sep storage for every year, with the same dry years highlighted throughout, and how these four relate to each other.
2. **Forecasting the monsoon inflow.** Forecast error by the date the forecast is made, with and without a rain forecast, forecast vs actual for each year, and extra forecasts issued on heavy-rain days.
3. **The next 15 days.** How much water reaches the dam over the next 15 days, from today's state with and without a rain forecast.
4. **How the catchment responds.** How much inflow rain produces and after what delay, as a map of each grid cell's multiplier and delay, plus the daily model against observations.
5. **Flood warning.** How many days ahead high flows can be predicted.
6. **Supporting detail.** Data fixes and the method.
7. **Compared with a published study (Hemavathy report only).** Our model vs Chandre Gowda (2025), scored both the paper's way and ours.

## Key results

Results are on data the models never saw. The monsoon total is scored by leaving out one year at a time over 1991–2025; the daily model is tested on 2020–2023, because the soil-moisture data ends in 2023.

**Monsoon-total forecast.** A forecast counts as reliable when it is within ±20% in 4 years out of 5.

| Forecast made on | KRS, typical error | KRS, 4 years in 5 within | Hemavathy, typical error | Hemavathy, 4 years in 5 within |
|---|---|---|---|---|
| Assume an average year | 37% | — | 26% | — |
| 1 Jul | 33% | 52% | 26% | 44% |
| 1 Aug | 17% | 28% | 16% | 29% |
| 1 Aug + 30-day rain forecast* | 11% | **16%** | 7% | **13%** |
| **15 Aug** | **11%** | **19%** | **7%** | **15%** |
| 1 Sep | 7% | 14% | 5% | 9% |

\*Actual rainfall used as a perfect forecast, so this is an upper bound on what a real forecast could add.

- **A reasonably informed estimate is possible from 15 August**, or from 1 August if a 30-day rain forecast is available.
- **Before July the model is no better than assuming an average year.** A median of about 55% of the season's inflow arrives after 1 August.
- **Monsoon inflow varies hugely from year to year:** KRS 41–252 TMC, Hemavathy 25–115 TMC.

**Replaying past monsoons (ensemble forecast).** On each forecast date the inflow so far and today's state are fixed, and the rest of the season is replayed with every other year's rain. The middle of the replays is slightly more accurate than the single forecast at most dates (KRS on 15 Aug: 4-in-5 error 19% → 18%; 15 Jun: 58% → 46%), and the replays give an 80% likely range (it contained the actual total 69–89% of the time) and the chance of a dry season.

**Heavy-rain days.** Besides the 1st and 15th of each month, the monsoon-total forecast is also re-issued on every heavy-rain day: catchment-average rain in the top 5% of monsoon days, about 3 a year.
- The updated forecast is typically closer than the last scheduled one: KRS ±21% → ±15%, Hemavathy ±18% → ±11%.
- **But an update on an ordinary day after the same wait improves just as often (57–59% of the time).** On the day of the storm its water has not reached the dam yet, and most of the uncertainty is rain still to come. In June an update does not help at all.

**Next 15 days** (test years 2020–2023, June–September)

| Inflow over the next 15 days | KRS typical error | KRS NSE | Hemavathy typical error | Hemavathy NSE |
|---|---|---|---|---|
| Assume an average year | 59% | 0.21 | 49% | 0.30 |
| Today's state, no rain forecast | 48% | 0.44 | 44% | 0.36 |
| + 7-day rain forecast* | 32% | 0.81 | 31% | 0.74 |
| + 15-day rain forecast* | **19%** | **0.93** | **22%** | **0.88** |

\*Actual rainfall used as a perfect forecast (best case).

- A rain forecast is what makes the 15-day outlook work; today's state alone is only a little better than assuming an average year.
- June is hardest (typical error about ±40% even with the rain forecast), because the dry catchment's response to rain varies a lot.

**Compared with a published Hemavathi model.** C. Chandre Gowda (2025), *Inflow prediction and modelling for a seasonal flow reservoir*, Water-Energy Nexus 8, 281–286, reports daily NSE 0.89 / KGE 0.93 for a neural network on lagged flows, tested on random days. We rebuilt its method on our data and scored everything two ways:

| June–November daily inflow (NSE) | Random test days (paper's test) | Unseen years 2020–2023 (our test) |
|---|---|---|
| Paper, as reported | 0.89 | — |
| Paper's method on our data (network on formula flows / on measured flows) | 0.72 / 0.83 | 0.34 / 0.85 |
| "Same as yesterday" | 0.82 | 0.84 |
| Our model, rain + soil only | **0.91** | 0.77 |
| Our model + last week's inflow | **0.92** | **0.90** |

- Random test days flatter every model, because the days around each test day are in training. On unseen years our model with last week's inflow stays at 0.90.
- Weak spot it exposes: without inflow readings our model over-predicts unseen years by about 29%, so its KGE is only 0.60.

**Year by year**
- **Dry years compound:** KRS gets 20% less rain, 55% less inflow, and ends September 64% full instead of 94%.
- **KRS ends September at least 90% full in 19 of 21 years with more than 86 TMC of monsoon inflow.** Below that it does so in only 3 of 14 years.
- **Each extra 100 mm of rain** adds about 17 TMC of inflow at KRS and 9 TMC at Hemavathy.

**Daily inflow and flood warning**
- **The daily model fits well.** On the Nash–Sutcliffe score (NSE: how much of the day-to-day rise and fall the model gets right; 1 = perfect, 0 = no better than always guessing the seasonal average), KRS scores 0.91 from rain, upstream releases and soil moisture, and Hemavathy 0.85 from rain and soil moisture alone.
- **Soil moisture helps the daily model, Hemavathy most:** NSE 0.81 → 0.85 at Hemavathy, 0.90 → 0.91 at KRS. Hemavathy gains more because its inflow comes straight from rain on its own catchment; KRS already gets most of its information from the upstream releases. Soil moisture does not improve the monsoon-total forecast.
- **Upstream releases reach KRS within a day.** Rain takes 1–2 days to arrive and keeps flowing for about a week.
- **Every rain cell has its own multiplier and delay.** 6 of 18 KRS cells and 5 of 9 Hemavathy cells barely affect inflow.
- **Learning a separate response for every cell helps a little:** NSE rises from 0.88 to 0.90 at KRS and from 0.80 to 0.81 at Hemavathy (2020–2023). At KRS it under-predicts flood peaks more, so the gain is in overall fit, not peaks.
- **High-flow warning:** 1 day ahead catches 88% (KRS) and 90% (Hemavathy) of high-flow days. At 3 days ahead this drops to 59% and 65%; with a rain forecast it is 82% and 77%.

## Repository layout

```
src/
  prepare.py            # cleaning: reservoirs + IMD rainfall clipped to the KRS watershed (logs every fix)
  inflow_model.py       # KRS daily inflow model, lag response, input importance
  dam_analysis.py       # both dams: daily model, monsoon-total forecast, lead time, river-network tests
  cell_response.py      # per-grid-cell multiplier and delay ("routed rain"), maps
  soil.py               # TerraClimate monthly soil data -> daily model inputs
  soil_test.py          # with vs without soil data, daily model and monsoon total
  event_forecast.py     # monsoon total re-issued on heavy-rain days; next-15-day inflow
  paper_compare.py      # Hemavathy: the published model's method rebuilt on our data, both test setups
  esp_forecast.py       # monsoon total by replaying every past monsoon's remaining rain (range + dry-season odds)
  build_final.py        # outputs/final_report.html and outputs/presentation.html
  build_dam_reports.py  # builds outputs/KRS_insights.html and outputs/Hemavathy_insights.html
  peak_analysis.py      # KRS peak events (thresholds used by the analyses)
  season_forecast.py, one_shot.py, lead_time.py   # earlier KRS-only analyses (30-Sep storage)
  vendor/               # d3 v7.9.0 (embedded so the reports work offline)
outputs/
  KRS_insights.html
  Hemavathy_insights.html
```

## Data (not included)

The input data is not in this repository. To rerun, place it as follows:

| Path | Content | Source |
|---|---|---|
| `data/KR_reservior.xlsx`, `data/Hemavathy_reserviour.xlsx`, `data/Harangi_reserviour.xlsx` | Daily % full, gross storage (TMC), inflow and outflow (cusecs). Columns: `Date` (DD-MM-YYYY), `% of Storage as on this day`, `Gross Storage (TMC)`, `Inflow (Cusecs)`, `OutFlow (Cusecs)` | K-WRIS reservoir records |
| `data/IMD_Gridded_1991_2025/Rainfall_ind<year>_rfp25.grd` | IMD 0.25° daily gridded rainfall, one binary file per year | IMD Pune |
| `watershed_KSR/watershed.shp` (+ `.shx`, `.dbf`, `.prj`) | KRS catchment polygon | India-WRIS hydrological layers |
| `data/watershed_hemavati.zip` | Hemavathy catchment polygon (zipped shapefile) | as above |
| `data/rivers_ksr.zip`, `data/rivers_hemavati.zip` | River flowlines with stream order (zipped shapefiles) | as above |
| `data/ksr-soil-data.zip`, `data/hemavati-soil_data.zip` | TerraClimate monthly time series per sub-watershed (soil moisture, runoff, evapotranspiration, water deficit), 1991–2023 | K-WRIS |

## Reproduce

Python 3.12. From the repository root:

```bash
uv venv .venv -p 3.12
uv pip install -p .venv -r requirements.txt

cd src
../.venv/bin/python prepare.py            # -> data/processed/, outputs/data_fixes.json
../.venv/bin/python inflow_model.py       # -> outputs/inflow_results.json
../.venv/bin/python peak_analysis.py      # -> outputs/peak_results.json
../.venv/bin/python dam_analysis.py       # -> outputs/dam_results.json
../.venv/bin/python cell_response.py      # -> outputs/cell_response.json
../.venv/bin/python soil_test.py          # -> outputs/soil_results.json
../.venv/bin/python event_forecast.py     # -> outputs/event_results.json
../.venv/bin/python paper_compare.py      # -> outputs/paper_compare.json
../.venv/bin/python esp_forecast.py       # -> outputs/esp_results.json
../.venv/bin/python build_final.py        # -> outputs/final_report.html, outputs/presentation.html
../.venv/bin/python build_dam_reports.py  # -> outputs/KRS_insights.html, outputs/Hemavathy_insights.html
```

Random seeds are fixed, so a rerun gives the same numbers. Plain `pip install -r requirements.txt` in any Python 3.12 environment works too.

## Method in brief

1. **Cleaning.** All fixes are listed in each report.
   - Fixes: header offsets, missing days, storage above design capacity, single-day storage spikes and zero monsoon inflow. Gaps of 3 days or less are interpolated.
   - **Outflow recorded in the wrong units:** from 10 July to 18 September 2019, all three dams report decimal "outflows" such as 5.1 cusecs during the August 2019 flood. These days are replaced by the water-balance outflow (inflow minus change in storage). Harangi has the same problem in July 2004.
   - Water-balance mismatches, and June 2012 (533 mm of rain but very little inflow), are flagged but not changed.
2. **Rainfall.** IMD cells overlapping each catchment, each weighted by its share of area inside the catchment.
3. **Per-cell response.** A non-negative linear model learns, for every cell, how much of its rain reaches the dam on each of the next 14 days. Wetter soil yields more inflow from the same rain. For KRS, rain above Hemavathy dam is left out: it reaches KRS only through Hemavathy's releases, which are already an input. The output ("routed rain") feeds the daily model.
4. **Daily inflow.** Gradient boosting on log inflow. Inputs are routed rain, rain lags by distance band, rain totals, soil wetness, last month's soil moisture, upstream releases (KRS only) and time of year. Trained on 1991–2019, tested on 2020–2023.
5. **Monsoon total.** The forecast is the inflow already measured since 1 June plus the predicted remaining inflow. A ridge regression is fitted for each forecast date, scored by leaving one year out at a time. The rain-forecast variants add the actual rain of the following 7–30 days, which makes them a best case. Heavy-rain-day forecasts use a local version: for any issue day, a ridge fitted on the other years' days within ±7 days of it.
6. **Next 15 days.** Gradient boosting on today's state (recent inflow and rain, soil wetness, soil moisture, storage, upstream releases for KRS), with and without the actual rain of the next 7 or 15 days. Trained on 1991–2019, tested on 2020–2023.

## Limitations

- Releases are operator decisions (irrigation, interstate water-sharing obligations); the models take upstream releases as known.
- The Harangi catchment boundary is not available, so rain above Harangi dam cannot be separated from rain that drains straight to KRS.
- Small edge cells have unreliable per-cell multipliers; the reports fade them out.
- The rain-forecast results use actual rainfall, so they overstate what a real forecast can deliver.
- TerraClimate soil moisture is published with a long delay; real-time use needs a near-real-time soil-moisture product. Sub-watersheds are averaged with equal weight because their areas are not given.

## Contributing

Fork the repository, make your changes on a branch, and open a pull request against `main`. Please don't commit the input data.
