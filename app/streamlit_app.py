from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CAL_DIR = ROOT / "artifacts" / "calibration"
METRICS = ROOT / "results" / "metrics" / "overall_metrics.csv"

st.set_page_config(page_title="GB Demand Forecast Reliability", layout="wide")
st.title("GB Day-Ahead Demand Forecast Reliability")
st.caption("Research visualisation only — not an operational dispatch tool")

if not CAL_DIR.exists() or not METRICS.exists():
    st.error("No completed research run found. Run: gbdf full-run --config configs/project.yaml")
    st.stop()

metrics = pd.read_csv(METRICS)
st.subheader("Locked-test summary")
st.dataframe(metrics, use_container_width=True, hide_index=True)

files = sorted(CAL_DIR.glob("*.csv"))
if not files:
    st.warning("No calibrated prediction files found.")
    st.stop()

labels = {p.stem: p for p in files}
model_name = st.selectbox("Model / calibration variant", list(labels))
df = pd.read_csv(labels[model_name], parse_dates=["settlement_date"])
dates = sorted(df["settlement_date"].dt.date.unique())
selected_date = st.selectbox("Target date", dates)
day = df[df["settlement_date"].dt.date == selected_date].copy()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=day["settlement_period"], y=day["national_demand_mw"], name="Actual demand"
    )
)
fig.add_trace(go.Scatter(x=day["settlement_period"], y=day["q50"], name="Median forecast"))
if day[["q05", "q95"]].notna().all().all():
    fig.add_trace(
        go.Scatter(
            x=day["settlement_period"],
            y=day["q95"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=day["settlement_period"],
            y=day["q05"],
            line=dict(width=0),
            fill="tonexty",
            name="90% interval",
            hoverinfo="skip",
        )
    )
fig.update_layout(
    xaxis_title="Settlement period",
    yaxis_title="National Demand (MW)",
    title=f"{model_name} — {selected_date}",
)
st.plotly_chart(fig, use_container_width=True)

point = day.dropna(subset=["q50"])
if len(point):
    mae = (point["national_demand_mw"] - point["q50"]).abs().mean()
else:
    mae = float("nan")
if day[["q05", "q95"]].notna().all(axis=1).any():
    interval = day.dropna(subset=["q05", "q95"])
    coverage = (
        (interval["national_demand_mw"] >= interval["q05"])
        & (interval["national_demand_mw"] <= interval["q95"])
    ).mean()
    width = (interval["q95"] - interval["q05"]).mean()
else:
    coverage = float("nan")
    width = float("nan")

c1, c2, c3 = st.columns(3)
c1.metric("Day MAE", f"{mae:,.0f} MW")
c2.metric("Day interval coverage", "—" if pd.isna(coverage) else f"{coverage:.1%}")
c3.metric("Mean interval width", "—" if pd.isna(width) else f"{width:,.0f} MW")

st.info(
    "The dashboard displays saved experiment outputs only. It does not retrain models or alter the locked test protocol."
)
