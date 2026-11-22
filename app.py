import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from teleomy.clean import (
    MACHINE_METRICS, load_clean_machines, load_clean_watch, quality_report,
)
from teleomy.data import ensure_data
from teleomy.models import TREND_METRICS, add_rolling, fit_isolation_forest, recovery_trends

st.set_page_config(page_title="Teleomy demo", layout="wide")
ensure_data()   # first launch: generate the simulated data

st.title("Teleomy demo: wearable + machine telemetry")
st.caption("All data is simulated. Wellness insights only, not medical advice.")


@st.cache_data
def load_machines():
    df = add_rolling(load_clean_machines())
    _, scored = fit_isolation_forest(df)
    return scored


@st.cache_data
def load_watch():
    watch = load_clean_watch()
    sessions = pd.read_csv("data/raw/sessions.csv", parse_dates=["date"])
    return watch, sessions, recovery_trends(watch, sessions)


@st.cache_data
def load_quality():
    return quality_report()


tab_machine, tab_recovery, tab_quality = st.tabs(
    ["Machine health", "User recovery", "Data quality"]
)

with tab_machine:
    machines = load_machines()
    machine = st.selectbox("Machine", sorted(machines["machine_id"].unique()))
    m = machines[machines["machine_id"] == machine].set_index("timestamp")

    daily = m["is_anomaly"].resample("D").mean().rename("share_flagged")
    col1, col2 = st.columns(2)
    col1.metric("Readings flagged", f"{int(m['is_anomaly'].sum())} of {len(m)}")
    col2.metric("Days with an alert (>20% flagged)", int((daily > 0.2).sum()))

    st.subheader("Share of readings flagged per day")
    chart = (
        alt.Chart(daily.reset_index())
        .mark_bar()
        .encode(
            x=alt.X("timestamp:T", title="Day"),
            y=alt.Y("share_flagged:Q", scale=alt.Scale(domain=[0, 1]), title="Share flagged"),
        )
    )
    st.altair_chart(chart)

    sensor = st.selectbox("Sensor", MACHINE_METRICS)
    st.subheader(f"{sensor} (hourly mean)")
    st.line_chart(m[sensor].resample("h").mean())
    st.caption(
        "Isolation Forest trained on the first 40 days only. Gradual drift is caught "
        "through rolling 1-hour averages, not just single spikes."
    )

with tab_recovery:
    watch, sessions, trends = load_watch()
    user = st.selectbox("User", sorted(watch["user_id"].unique()))
    metric = st.selectbox("Metric", TREND_METRICS)

    uw = watch[watch["user_id"] == user].set_index("date").sort_index()
    start = sessions[sessions["user_id"] == user]["date"].min()
    st.write(f"First therapy session: **{start.date()}**")
    st.line_chart(pd.DataFrame({
        metric: uw[metric],
        "7-day mean": uw[metric].rolling(7, min_periods=3).mean(),
    }))

    st.subheader("Trend summary (baseline = 14 days before first session)")
    st.dataframe(trends[trends["user_id"] == user].drop(columns="user_id"), hide_index=True)
    st.caption(
        "A trend is not proof that a therapy works: there is no control group, and the "
        "improvement in this demo is built into the simulated data."
    )

with tab_quality:
    st.write(
        "Two watch vendors (different field names, date formats and sleep units) and two "
        "machines are mapped to one schema. Impossible values are removed, only short gaps "
        "are filled, and every value keeps a quality flag."
    )
    report, dupes = load_quality()
    st.dataframe(report, hide_index=True)
    st.write("Duplicate rows dropped:", dupes)

    feed = Path("data/telemetry.jsonl")
    if feed.exists():
        st.subheader("Example standard JSON records")
        with feed.open() as f:
            st.json([json.loads(next(f)) for _ in range(3)])