"""Streamlit dashboard:  streamlit run app.py   (run `python -m src.train` first)"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.features import ALERT_LEVEL, CAT_COLORS, CATS
from src.forecast import make_forecast, risk_ranking

st.set_page_config(page_title="Pakistan AQI Forecast", page_icon="🌫️", layout="wide")

ADVICE = {
    0: "Air quality is good. Enjoy outdoor activities.",
    1: "Acceptable. Unusually sensitive people should limit prolonged outdoor exertion.",
    2: "Children, elderly and people with heart/lung conditions should reduce prolonged outdoor activity.",
    3: "Everyone may be affected. Wear an N95 mask outdoors, limit exertion, keep windows closed.",
    4: "Health alert. Avoid outdoor activity, use air purifiers, schools/authorities should consider restrictions.",
    5: "Emergency conditions. Stay indoors, seal windows, seek medical help if symptomatic.",
}


@st.cache_resource
def load_all():
    bundle = joblib.load(ROOT / "models/aqi_model.joblib")
    daily = pd.read_csv(ROOT / "data/processed/daily.csv", parse_dates=["date"])
    return bundle, daily, make_forecast(bundle, daily)


try:
    bundle, daily, fc = load_all()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

st.title("🌫️ Pakistan Air Quality Forecast")
if bundle["data_source"] == "synthetic":
    st.warning("Model trained on SYNTHETIC demo data. Put the Kaggle CSVs in data/raw and re-run training for real forecasts.")

city = st.sidebar.selectbox("City", bundle["cities"])
hist_days = st.sidebar.slider("History shown (days)", 14, 180, 60)
st.sidebar.caption(f"Alert threshold: **{CATS[ALERT_LEVEL]}** or worse")

cf = fc[fc.city == city].sort_values("horizon_days")
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🏙️ City risk ranking", "🔍 Model insights"])

with tab1:
    worst = int(cf.category_index.max())
    msg = f"**{city}**: {CATS[worst]} air expected within the next {len(cf)} days. {ADVICE[worst]}"
    (st.error if worst >= ALERT_LEVEL else st.warning if worst == 2 else st.success)(
        ("🚨 ALERT — " if worst >= ALERT_LEVEL else "") + msg)

    cols = st.columns(len(cf))
    for col, (_, r) in zip(cols, cf.iterrows()):
        col.metric(f"{r.date:%a %d %b} (+{r.horizon_days}d)", f"AQI {r.predicted_aqi:.0f}", r.predicted_category,
                   delta_color="off")
        col.caption(f"P(Unhealthy+) = {r.prob_unhealthy_or_worse:.0%}")

    h = daily[daily.city == city].tail(hist_days)[["date", "aqi"]].assign(kind="Observed")
    f = cf[["date", "predicted_aqi"]].rename(columns={"predicted_aqi": "aqi"}).assign(kind="Forecast")
    f["date"] = pd.to_datetime(f["date"])
    last = h.tail(1).assign(kind="Forecast")                      # connect the lines
    plot = pd.concat([h, last, f])
    dom = ["Observed", "Forecast"]
    line = alt.Chart(plot).mark_line(point=True).encode(
        x=alt.X("date:T", title=None), y=alt.Y("aqi:Q", title="AQI"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=dom, range=["#3b6fb6", "#d63a3a"]), title=None),
        strokeDash=alt.StrokeDash("kind:N", legend=None, scale=alt.Scale(domain=dom, range=[[1, 0], [5, 4]])))
    rule = alt.Chart(pd.DataFrame({"y": [150]})).mark_rule(color="#d63a3a", strokeDash=[2, 3]).encode(y="y:Q")
    st.altair_chart((line + rule).properties(height=330), width="stretch")
    st.caption("Dotted red line = AQI 150 (Unhealthy threshold).")
    st.dataframe(cf[["date", "predicted_aqi", "predicted_category", "prob_unhealthy_or_worse", "alert"]],
                 hide_index=True, width="stretch")
    st.download_button("⬇️ Download all-city forecast CSV", fc.drop(columns="category_index").to_csv(index=False),
                       "forecast.csv", "text/csv")

with tab2:
    rk = risk_ranking(fc)
    st.caption("Ranked by mean predicted AQI over the forecast window.")
    colour = alt.Scale(domain=CATS, range=CAT_COLORS)
    st.altair_chart(alt.Chart(rk).mark_bar().encode(
        y=alt.Y("city:N", sort="-x", title=None), x=alt.X("mean_aqi:Q", title="Mean predicted AQI"),
        color=alt.Color("risk_category:N", scale=colour, title="Risk"),
        tooltip=list(rk.columns)).properties(height=280), width="stretch")
    st.dataframe(rk, hide_index=True, width="stretch")

with tab3:
    c1, c2 = st.columns(2)
    imp = pd.Series(bundle["importance"]).rename("importance").reset_index().rename(columns={"index": "feature"})
    c1.subheader("What drives next-day category?")
    c1.altair_chart(alt.Chart(imp).mark_bar().encode(
        y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("importance:Q", title="Permutation importance (Δ macro-F1)")),
        width="stretch")
    c2.subheader("Test-set performance")
    m = pd.DataFrame(bundle["metrics"])
    c2.dataframe(m[["horizon_days", "classifier", "accuracy", "macro_f1", "aqi_rmse", "aqi_mae",
                    "baseline_accuracy", "baseline_aqi_rmse"]], hide_index=True)
    c2.caption(f"Chronological split: test starts {bundle['split']['test_start']}. "
               "'baseline' = persistence (tomorrow = today).")
