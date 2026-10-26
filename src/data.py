"""Loading / cleaning real Kaggle CSVs (auto column detection) and a synthetic fallback generator."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .features import pm25_to_aqi, pm10_to_aqi

ALIASES = {  # canonical -> candidate names (normalised: lowercase, alphanumerics only), priority order
    "city": ["city", "location", "station", "site"],
    "date": ["datetime", "date", "timestamp", "time"],
    "pm25": ["pm25"], "pm10": ["pm10"],
    "aqi": ["aqi", "airqualityindex"],
    "temp": ["temperature", "temp", "t2m"],
    "humidity": ["relativehumidity", "humidity", "rh"],
    "wind_speed": ["windspeed", "wind", "ws"],
    "pressure": ["surfacepressure", "pressure", "msl"],
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def standardize(df, default_city=None):
    cols = {_norm(c): c for c in df.columns}
    out = {}
    for canon, names in ALIASES.items():
        for n in names:
            hit = next((orig for k, orig in cols.items() if k == n or (len(n) >= 4 and k.startswith(n))), None)
            if hit is not None and hit not in out.values():
                out[canon] = hit
                break
    d = pd.DataFrame({k: df[v] for k, v in out.items()})
    if "city" not in d:
        d["city"] = default_city or "Unknown"
    if "date" not in d:
        raise ValueError(f"No date/time column found. Columns: {list(df.columns)}")
    return d


def load_raw(raw_dir):
    """Read every CSV in raw_dir -> tidy DAILY frame (city, date, aqi, pm25, pm10, weather...)."""
    frames = []
    for p in sorted(Path(raw_dir).glob("*.csv")):
        frames.append(standardize(pd.read_csv(p), default_city=p.stem.replace("_", " ").title()))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.floor("D")
    d = d.dropna(subset=["date"])
    d["city"] = d["city"].astype(str).str.strip().str.title()
    num = [c for c in d.columns if c not in ("city", "date")]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    for c in ("pm25", "pm10"):                              # sensor glitches
        if c in d:
            d.loc[(d[c] < 0) | (d[c] > 1000), c] = np.nan
    d = d.groupby(["city", "date"], as_index=False)[num].mean()   # hourly -> daily mean
    return finalize(d)


def finalize(d):
    """Derive AQI from PM if the dataset has none, drop empty columns, drop days with no AQI."""
    if "aqi" not in d or d["aqi"].isna().all():
        parts = [pm25_to_aqi(d["pm25"]) if "pm25" in d else None,
                 pm10_to_aqi(d["pm10"]) if "pm10" in d else None]
        parts = [p for p in parts if p is not None]
        if not parts:
            raise ValueError("Need an 'aqi', 'pm25' or 'pm10' column.")
        d["aqi"] = np.nanmax(np.vstack(parts), axis=0)
    d = d.dropna(subset=["aqi"])
    d = d.dropna(axis=1, how="all")
    return d.sort_values(["city", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------- synthetic fallback
_CITY = {  # base PM2.5, winter amplitude, mean temp, temp amp, base humidity
    "Lahore": (75, 1.0, 24, 12, 55), "Faisalabad": (70, 0.9, 25, 11, 55),
    "Multan": (65, 0.8, 26, 12, 50), "Peshawar": (60, 0.8, 22, 11, 50),
    "Karachi": (45, 0.25, 28, 5, 65), "Islamabad": (35, 0.6, 21, 10, 55),
}


def make_synthetic(start="2021-01-01", end="2024-12-31", seed=42):
    """Realistic-looking stand-in (winter smog, wind/humidity effects, pollution episodes, gaps).
    ONLY for testing the pipeline when the Kaggle files are not available."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n = len(days)
    season = np.cos(2 * np.pi * (days.dayofyear.values - 15) / 365.25)   # +1 = mid-January

    def ar1(phi, sd):
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal(0, sd)
        return x

    rows = []
    for city, (base, wamp, tm, ta, hb) in _CITY.items():
        temp = tm - ta * season + ar1(0.8, 1.5)
        hum = np.clip(hb + 15 * season + ar1(0.8, 5), 10, 100)
        wind = np.exp(np.log(2.5) + ar1(0.6, 0.35))
        pres = 1013 + 6 * season + ar1(0.85, 1.2)
        episode = np.zeros(n)
        for t in range(1, n):
            episode[t] = 0.8 * episode[t - 1] + (rng.random() < 0.012) * rng.uniform(0.5, 1.2)
        mu = np.log(base) + 0.9 * wamp * season - 0.3 * np.log(wind / 2.5) + 0.006 * (hum - hb)
        pm25 = np.clip(np.exp(mu + ar1(0.7, 0.27) + episode), 3, 450)
        pm10 = pm25 * rng.uniform(1.6, 2.2, n) + rng.normal(0, 8, n)
        df = pd.DataFrame(dict(city=city, date=days, pm25=pm25, pm10=pm10.clip(5),
                               temp=temp, humidity=hum, wind_speed=wind, pressure=pres))
        for c in ("pm25", "pm10", "temp", "humidity", "wind_speed", "pressure"):   # random missingness
            df.loc[rng.random(n) < 0.03, c] = np.nan
        g = rng.integers(0, n - 12)                                                 # one sensor outage
        df.loc[g:g + rng.integers(4, 10), ["pm25", "pm10"]] = np.nan
        rows.append(df)
    return finalize(pd.concat(rows, ignore_index=True))
