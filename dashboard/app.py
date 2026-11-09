"""
BlindSpot Guardian -- Streamlit dashboard shell.

Run with:  streamlit run dashboard/app.py

All five tabs call into the shared GuardianPipeline (src/pipeline.py) --
there is one implementation of detection+tracking+zones+risk+prediction+
events, used here and by src/live_demo.py, not two copies that can drift.
"""
import os
import sys

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(DASHBOARD_DIR, "..", "src"))
for p in (DASHBOARD_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st  # noqa: E402

import analytics  # noqa: E402
import monitoring  # noqa: E402
import predictive_safety  # noqa: E402
import simulator  # noqa: E402
import zone_configuration  # noqa: E402

st.set_page_config(page_title="BlindSpot Guardian", layout="wide")

st.title("BlindSpot Guardian")
st.caption(
    "AI-based predictive blind-spot monitoring -- software-only, webcam-driven. "
    "A rule-based engine, not a verified collision-prediction system; see README for limitations."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Live Monitoring", "Predictive Safety", "Zone Configuration", "Incident Analytics", "Scenario Simulator"]
)

with tab1:
    monitoring.render()
with tab2:
    predictive_safety.render()
with tab3:
    zone_configuration.render()
with tab4:
    analytics.render()
with tab5:
    simulator.render()
