# Air Quality Level Prediction — Pakistan (3-day AQI forecast)

Predicts the **daily AQI category** (Good → Hazardous) and **numeric AQI** for the next **1, 2 and 3 days**
for Pakistani cities, with a Streamlit dashboard, alerts, city risk ranking and explainability.

> ⚠️ **Current status: pipeline verified on SYNTHETIC data.** The sandbox could not download the Kaggle datasets
> (login required). Drop the real CSVs into `data/raw/` and run `python -m src.train` — everything below re-runs on them.
> The numbers in "Results" are from synthetic data and must **not** be reported as real performance.

## Quick start
```bash
pip install -r requirements.txt
# optional: copy Kaggle CSVs into data/raw/  (otherwise synthetic demo data is used)
python -m src.train          # cleans data, trains, evaluates, saves model + forecast CSV
streamlit run app.py         # dashboard
```

## Project layout
| Path | Purpose |
|---|---|
| `src/data.py` | Loads raw CSVs (auto-detects column names, hourly→daily), derives AQI from PM2.5/PM10, synthetic fallback |
| `src/features.py` | AQI/category maths + feature pipeline (shared by train, forecast, app) |
| `src/models.py` | RandomForest / HistGradientBoosting / XGBoost candidates |
| `src/train.py` | Time-based split, model selection, evaluation, saving, explainability |
| `src/forecast.py` | Forecast table + city risk ranking |
| `app.py` | Streamlit app (loads `models/aqi_model.joblib`) |
| `models/aqi_model.joblib` | Saved bundle: models per horizon + feature list + city list + metrics (this *is* the preprocessing contract) |
| `outputs/forecast.csv` | `city, base_date, date, horizon_days, predicted_aqi, predicted_category, prob_unhealthy_or_worse, alert` |
| `outputs/city_risk_ranking.csv` | Cities ranked by mean predicted AQI |
| `reports/metrics.json` | Test metrics + top features |

## Dataset
Expected: any of the 4 Kaggle datasets in the brief (one or many CSVs in `data/raw/`). Columns are matched by
name (city/location, date/datetime, PM2.5, PM10, AQI, temperature, humidity, wind speed, pressure). If a file has
no city column, the file name is used as the city. If there is no AQI column it is computed from PM2.5/PM10
with US-EPA breakpoints (piecewise-linear approximation); daily AQI = max of the two. Hourly data is averaged to daily.

## Preprocessing
Daily reindexing per city → short gaps (≤5 days) interpolated → longer gaps left missing and the affected rows dropped
for training → impossible PM values (<0 or >1000) set to NaN.

## Features
AQI lags (1,2,3,7), rolling mean (3,7,14) and std (7), 1-day change; PM2.5/PM10 and weather
(temp, humidity, wind, pressure) current + lag-1 + 3-day mean + 1-day change; day-of-year sin/cos, day-of-week;
city one-hot.

## Models and why
Direct multi-horizon strategy: one classifier (category) and one regressor (AQI) **per lead time** — avoids error
accumulation of recursive forecasting. Candidates: Random Forest, Histogram Gradient Boosting, XGBoost; the best per
horizon is chosen on a **validation** slice (macro-F1 / RMSE), refit on train+val, and scored on a final **test** slice.
Split is chronological 70/15/15 (no shuffling). Tree models were chosen for tabular lag features: robust, fast,
little tuning, and they give feature importance. LSTM / Temporal CNN are left as a polish-phase experiment.

## Evaluation (synthetic data — placeholder numbers)
| Lead time | Accuracy | Macro-F1 | AQI RMSE | AQI MAE | Persistence acc. | Persistence RMSE |
|---|---|---|---|---|---|---|
| +1 day | 0.629 | 0.458 | 33.5 | 22.6 | 0.607 | 37.9 |
| +2 days | 0.554 | 0.368 | 40.3 | 27.6 | 0.534 | 47.3 |
| +3 days | 0.562 | 0.385 | 41.8 | 29.2 | 0.488 | 51.3 |

Alert detection (Unhealthy or worse), +1 day: precision 0.84, recall 0.79.
The model beats the "tomorrow = today" baseline on accuracy and RMSE at all horizons, but on +1 day macro-F1 it is
slightly *below* persistence — a point to fix on real data (class balancing / thresholds).

## Limitations
- Real-data results pending (see status note above).
- Weather features are *current* observations, not weather forecasts; adding forecast weather would help +2/+3 days.
- AQI is derived from PM only (no NO2/SO2/O3/CO) if the dataset lacks an AQI column.
- Rare classes (Good, Hazardous) have few samples → low per-class F1; macro-F1 is the honest metric.
- Single-station/city-level averages; no spatial interaction between cities.
- Forecast uses the last *available* day, so stale data gives a stale forecast.

## Next (polish phase)
Real-data run and tuning, TimeSeriesSplit CV, LSTM/TCN comparison, SHAP, probability calibration, weather-forecast
features, scheduled data refresh.
