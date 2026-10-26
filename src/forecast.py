"""Turn the trained bundle + latest data into the N-day forecast table."""
import numpy as np
import pandas as pd
from .features import CATS, ALERT_LEVEL, HORIZONS, aqi_to_cat, build_features
from .models import prob_alert


def make_forecast(bundle, daily):
    fcols = bundle["feature_cols"]
    f = build_features(daily, bundle["base_cols"], bundle["cities"])
    rows = []
    for city, g in f.groupby("city"):
        g = g.dropna(subset=fcols)
        if g.empty:
            continue
        last = g.sort_values("date").iloc[[-1]]           # latest day with a complete feature vector
        X, d0 = last[fcols], last["date"].iloc[0]
        for h in HORIZONS:
            aqi = float(bundle["reg"][h].predict(X)[0])
            cat = int(bundle["clf"][h].predict(X)[0])
            pa = float(prob_alert(bundle["clf"][h], X, ALERT_LEVEL)[0])
            rows.append(dict(city=city, base_date=d0.date(), date=(d0 + pd.Timedelta(days=h)).date(),
                             horizon_days=h, predicted_aqi=round(aqi, 1), predicted_category=CATS[cat],
                             category_index=cat, prob_unhealthy_or_worse=round(pa, 3),
                             alert=cat >= ALERT_LEVEL))
    return pd.DataFrame(rows)


def risk_ranking(fc):
    r = (fc.groupby("city").agg(mean_aqi=("predicted_aqi", "mean"), peak_aqi=("predicted_aqi", "max"),
                                max_alert_prob=("prob_unhealthy_or_worse", "max"),
                                alert_days=("alert", "sum")).reset_index())
    r["risk_category"] = [CATS[int(c)] for c in aqi_to_cat(r["mean_aqi"].values)]
    r = r.sort_values(["mean_aqi"], ascending=False).reset_index(drop=True)
    r.insert(0, "rank", r.index + 1)
    return r.round(2)
