from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import streamlit as st

import joblib

from chpp.data_sources.fred import load_hpi_series, summarize_changes
from chpp.data_sources.geo import default_payload_for_region, get_regions
from chpp.predict import predict_many, predict_one
from chpp.train import train_model

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "models" / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "reports" / "metrics.json"


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    model, _ = train_model()
    return model


def api_predict_batch(base_url: str, payloads: list[dict]) -> list[dict] | None:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/predict-batch",
            json={"items": payloads},
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("predictions")
    except Exception:
        return None


def local_predict_batch(payloads: list[dict]) -> list[dict]:
    model = load_model()
    preds = predict_many(model, payloads)
    return [
        {
            "prediction_hundreds_k": pred,
            "prediction_usd": round(pred * 100_000, 2),
        }
        for pred in preds
    ]


def local_predict_one(payload: dict) -> float:
    model = load_model()
    return predict_one(model, payload)


def build_color_scale(values):
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return [(0, 116, 217) for _ in values]
    colors = []
    for value in values:
        ratio = (value - min_val) / (max_val - min_val)
        r = int(30 + 200 * ratio)
        g = int(70 + 140 * (1 - ratio))
        b = int(200 - 120 * ratio)
        colors.append([r, g, b, 160])
    return colors


st.set_page_config(page_title="Housing Analytics Dashboard", layout="wide")
st.title("California Housing Analytics Dashboard")
st.caption("Interactive map, neighborhood comparisons, and market trends.")

regions = get_regions()

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("API Base URL", value="http://localhost:8000")
    use_api = st.toggle("Use API for predictions", value=True)
    st.divider()
    st.header("Region")
    selected_region_id = st.selectbox(
        "Focus region",
        options=[region.region_id for region in regions],
        format_func=lambda rid: next(r.name for r in regions if r.region_id == rid),
    )

region_df = pd.DataFrame(
    [
        {
            "region_id": region.region_id,
            "name": region.name,
            "lat": region.lat,
            "lon": region.lon,
            **region.defaults,
        }
        for region in regions
    ]
)

payloads = [default_payload_for_region(region.region_id) for region in regions]
focus_payload = default_payload_for_region(selected_region_id)

predictions = []
if use_api:
    api_result = api_predict_batch(base_url, payloads)
    if api_result:
        predictions = api_result
    else:
        st.warning("API unavailable, falling back to local model.")
        predictions = local_predict_batch(payloads)
else:
    predictions = local_predict_batch(payloads)

region_df["prediction_usd"] = [pred["prediction_usd"] for pred in predictions]

st.subheader("Market Map")
colors = build_color_scale(region_df["prediction_usd"].tolist())
region_df["color"] = colors
layer = pdk.Layer(
    "ScatterplotLayer",
    data=region_df,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius=15000,
    pickable=True,
)
view_state = pdk.ViewState(latitude=36.8, longitude=-119.4, zoom=5)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\n${prediction_usd}"}))

st.subheader("Neighborhood Comparisons")
comparison_ids = st.multiselect(
    "Select regions to compare",
    options=[region.region_id for region in regions],
    default=[selected_region_id],
    format_func=lambda rid: next(r.name for r in regions if r.region_id == rid),
)
if comparison_ids:
    comparison_payloads = [default_payload_for_region(rid) for rid in comparison_ids]
    comparison_preds = (
        api_predict_batch(base_url, comparison_payloads)
        if use_api
        else None
    )
    if not comparison_preds:
        comparison_preds = local_predict_batch(comparison_payloads)
    compare_df = pd.DataFrame(comparison_payloads)
    compare_df["region_id"] = comparison_ids
    compare_df["prediction_usd"] = [item["prediction_usd"] for item in comparison_preds]
    st.dataframe(compare_df.set_index("region_id"))
    chart = px.bar(
        compare_df,
        x="region_id",
        y="prediction_usd",
        title="Predicted median value (USD)",
    )
    st.plotly_chart(chart, use_container_width=True)

st.subheader("Market Trend Analysis")
trend_region = st.selectbox(
    "Trend region",
    options=["us", "ca"],
    format_func=lambda rid: "United States" if rid == "us" else "California",
)

trend_df = None
try:
    trend_series = load_hpi_series(region_id=trend_region)
    trend_df = trend_series.points.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"])
except Exception:
    st.warning("Trend data unavailable. Check network connectivity.")

if trend_df is not None and not trend_df.empty:
    base_pred = local_predict_one(focus_payload)
    latest_value = trend_df["value"].iloc[-1]
    trend_df["predicted_usd"] = (trend_df["value"] / latest_value) * base_pred * 100_000

    fig = px.line(
        trend_df,
        x="date",
        y=["value", "predicted_usd"],
        title="House Price Index vs. model-aligned trend",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Actionable Insights")
if trend_df is not None and not trend_df.empty:
    summary = summarize_changes(trend_df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest HPI", f"{summary.get('latest', 0):.2f}")
    col2.metric("12m change", f"{summary.get('change_12m_pct', 0):.2f}%")
    col3.metric("60m change", f"{summary.get('change_60m_pct', 0):.2f}%")

medinc_up = dict(focus_payload)
medinc_up["MedInc"] *= 1.1
medinc_down = dict(focus_payload)
medinc_down["MedInc"] *= 0.9
pred_up = local_predict_one(medinc_up)
pred_down = local_predict_one(medinc_down)
st.write(
    "Income sensitivity: a 10% increase in median income shifts predictions by "
    f"{round((pred_up - pred_down) * 100_000, 2)} USD."
)

st.caption("Note: Trend overlays are illustrative, combining HPI movements with model outputs.")
