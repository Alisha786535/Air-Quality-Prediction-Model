"""AQI maths + the feature-engineering pipeline shared by training, forecasting and the app."""
import numpy as np
import pandas as pd

CATS = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
        "Unhealthy", "Very Unhealthy", "Hazardous"]
CAT_COLORS = ["#2e9e4f", "#e6c200", "#f28c28", "#d63a3a", "#8a3ab9", "#7e0023"]
ALERT_LEVEL = 3          # category index >= 3 (Unhealthy) triggers an alert
HORIZONS = [1, 2, 3]     # days ahead
WEATHER = ["temp", "humidity", "wind_speed", "pressure"]

# US-EPA style piecewise-linear breakpoints (concentration -> AQI), interpolated linearly.
_PM25 = ([0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4], [0, 50, 100, 150, 200, 300, 400, 500])
_PM10 = ([0, 54, 154, 254, 354, 424, 504, 604], [0, 50, 100, 150, 200, 300, 400, 500])


def pm25_to_aqi(c):
    return np.interp(c, *_PM25)


def pm10_to_aqi(c):
    return np.interp(c, *_PM10)


def aqi_to_cat(aqi):
    """AQI value -> category index 0..5 (NaN stays NaN)."""
    a = np.asarray(aqi, dtype=float)
    idx = np.digitize(a, [50, 100, 150, 200, 300], right=True).astype(float)
    idx[np.isnan(a)] = np.nan
    return idx


def _city_frame(g, city, base_cols):
    g = g.sort_values("date").set_index("date")
    g = g[~g.index.duplicated()].reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    g[base_cols] = g[base_cols].interpolate(limit=5, limit_direction="both")  # short gaps only
    a = g["aqi"]
    f = pd.DataFrame(index=g.index)
    f["city"], f["aqi_now"] = city, a
    for l in (1, 2, 3, 7):
        f[f"aqi_lag{l}"] = a.shift(l)
    for w in (3, 7, 14):
        f[f"aqi_roll{w}"] = a.rolling(w, min_periods=max(2, w // 2)).mean()
    f["aqi_std7"] = a.rolling(7, min_periods=4).std()
    f["aqi_diff1"] = a.diff()
    for c in [c for c in base_cols if c != "aqi"]:
        f[c] = g[c]
        f[f"{c}_lag1"] = g[c].shift(1)
        f[f"{c}_roll3"] = g[c].rolling(3, min_periods=2).mean()
        if c in WEATHER:
            f[f"{c}_chg1"] = g[c].diff()
    doy = g.index.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["dow"] = g.index.dayofweek
    for h in HORIZONS:                                   # direct multi-horizon targets
        f[f"target_aqi_t{h}"] = a.shift(-h)
        f[f"target_cat_t{h}"] = aqi_to_cat(a.shift(-h).values)
    f.index.name = "date"
    return f.reset_index()


def build_features(daily, base_cols, cities):
    """daily: columns city,date,aqi,<base_cols>. Returns one row per city-day with features + targets.
    The last row of every city has NaN targets -> that is the row we forecast from."""
    parts = [_city_frame(g, c, base_cols) for c, g in daily.groupby("city")]
    f = pd.concat(parts, ignore_index=True)
    for c in cities:                                      # fixed one-hot so train/serve match
        f[f"city_{c}"] = (f["city"] == c).astype(int)
    return f


def feature_columns(f):
    return [c for c in f.columns if c not in ("city", "date") and not c.startswith("target_")]
