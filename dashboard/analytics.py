"""
Tab 4: Incident Analytics -- real statistics computed from the SQLite event
log (data/events/events.db), populated by live monitoring runs and the
Scenario Simulator. No placeholder/hardcoded numbers: if there are no events
yet, the charts are empty and say so.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from event_manager import query_events, summary_stats
from paths import DB_PATH


def render() -> None:
    if st.button("Refresh"):
        st.rerun()

    stats = summary_stats(DB_PATH)

    if stats["total"] == 0:
        st.info("No safety-risk events recorded yet. Run Live Monitoring or the Scenario Simulator "
                "with someone passing through a danger zone to generate real data here.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total events", stats["total"])
    c2.metric("Critical", stats["critical"])
    c3.metric("High", stats["high"])
    c4.metric("Avg duration (s)", stats["avg_duration_s"] if stats["avg_duration_s"] is not None else "—")

    col_a, col_b = st.columns(2)
    with col_a:
        risk_df = pd.DataFrame({"risk_level": list(stats["by_risk"].keys()),
                                 "count": list(stats["by_risk"].values())})
        fig = px.bar(risk_df, x="risk_level", y="count", title="Events by risk level",
                     color="risk_level",
                     color_discrete_map={"SAFE": "#2ecc71", "LOW": "#2ecc71", "MODERATE": "#f1c40f",
                                          "HIGH": "#e67e22", "CRITICAL": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        zone_df = pd.DataFrame({"zone": list(stats["by_zone"].keys()),
                                 "count": list(stats["by_zone"].values())})
        fig2 = px.pie(zone_df, names="zone", values="count", title="Events by zone")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Event log")
    fc1, fc2 = st.columns(2)
    risk_filter = fc1.selectbox("Filter by risk", ["All", "CRITICAL", "HIGH", "MODERATE", "LOW", "SAFE"])
    zone_filter = fc2.selectbox("Filter by zone", ["All"] + list(stats["by_zone"].keys()))

    rows = query_events(
        DB_PATH,
        risk_level=None if risk_filter == "All" else risk_filter,
        zone=None if zone_filter == "All" else zone_filter,
    )
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig3 = px.scatter(df, x="start_time", y="risk_level", color="risk_level", hover_data=["zone", "tracking_id"],
                           title="Event timeline",
                           color_discrete_map={"SAFE": "#2ecc71", "LOW": "#2ecc71", "MODERATE": "#f1c40f",
                                                "HIGH": "#e67e22", "CRITICAL": "#e74c3c"})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No events match this filter.")
