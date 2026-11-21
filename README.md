# Teleomy demo

A prototype health-telemetry pipeline: raw data from smartwatches and IoT
machines is cleaned into one standard JSON format, analysed with ML, and shown
in a dashboard.

> All data is simulated. Outputs are wellness insights, not medical advice.

## Status

- [x] Machine sensor simulator (healthy + slowly failing machine)
- [ ] Watch data simulator (two vendor formats)
- [ ] Cleaning layer
- [ ] Anomaly detection (Isolation Forest)
- [ ] Recovery trend model
- [ ] Streamlit dashboard

## Run

    uv sync
    uv run python -m teleomy.simulate
