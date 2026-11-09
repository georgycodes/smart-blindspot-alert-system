"""
Tab 4: Incident Analytics -- real statistics computed from the SQLite event
log (data/events/events.db), populated by live monitoring runs and the
Scenario Simulator. No placeholder/hardcoded numbers: if there are no events
yet, the charts are empty and say so.

Also offers incident replay for events whose source video is still present
in data/videos/ -- live webcam events have no saved recording, so they
can't be replayed, and the UI says so rather than pretending to.
"""
import os

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from event_manager import query_events, summary_stats, zone_label
from paths import CONFIG_PATH, DB_PATH, MODEL_PATH, VIDEOS_DIR

REPLAY_BUFFER_FRAMES = 15


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
    # "zone" filter options include predicted-only labels (e.g. "FRONT (predicted)"),
    # so this filters in Python against zone_label() rather than a raw SQL zone match.
    zone_filter = fc2.selectbox("Filter by zone", ["All"] + list(stats["by_zone"].keys()))

    all_rows = query_events(DB_PATH, risk_level=None if risk_filter == "All" else risk_filter)
    rows = all_rows if zone_filter == "All" else [r for r in all_rows if zone_label(r) == zone_filter]
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df["zone_display"] = [zone_label(r) for r in rows]
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig3 = px.scatter(df, x="start_time", y="risk_level", color="risk_level",
                           hover_data=["zone_display", "tracking_id"],
                           title="Event timeline",
                           color_discrete_map={"SAFE": "#2ecc71", "LOW": "#2ecc71", "MODERATE": "#f1c40f",
                                                "HIGH": "#e67e22", "CRITICAL": "#e74c3c"})
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Incident replay")
        event_ids = df["id"].tolist()
        chosen_id = st.selectbox("Event to replay", event_ids,
                                  format_func=lambda i: f"#{i} -- {df.loc[df['id'] == i, 'risk_level'].iloc[0]} "
                                                         f"in {df.loc[df['id'] == i, 'zone_display'].iloc[0]} "
                                                         f"({df.loc[df['id'] == i, 'source_video'].iloc[0]})")
        _render_replay(df[df["id"] == chosen_id].iloc[0])
    else:
        st.info("No events match this filter.")


def _render_replay(event_row) -> None:
    source_video = event_row["source_video"]
    video_path = os.path.join(VIDEOS_DIR, source_video) if source_video else None

    if not video_path or not os.path.exists(video_path):
        st.caption(
            f"No saved recording for this event's source ('{source_video}'). Live webcam runs aren't "
            f"recorded, so only events from the Scenario Simulator (an uploaded/selected video that's "
            f"still in data/videos/) can be replayed."
        )
        return

    start_frame = event_row.get("start_frame")
    end_frame = event_row.get("end_frame")
    if pd.isna(start_frame):
        st.caption("This event has no recorded frame range (it predates frame-index tracking) -- can't replay it.")
        return

    cache_key = f"replay_frames_{event_row['id']}"
    if st.button("Load replay", key=f"load_replay_{event_row['id']}"):
        with st.spinner("Re-running the pipeline over this segment..."):
            st.session_state[cache_key] = _extract_replay_frames(
                video_path, int(start_frame), int(end_frame) if not pd.isna(end_frame) else int(start_frame),
                event_row["vehicle_profile"],
            )

    frames_rgb = st.session_state.get(cache_key)
    if not frames_rgb:
        st.caption("Click 'Load replay' to re-run detection/zone/risk overlays over the frames around this event.")
        return

    idx = st.slider("Frame", 0, len(frames_rgb) - 1, min(REPLAY_BUFFER_FRAMES, len(frames_rgb) - 1),
                     key=f"replay_slider_{event_row['id']}")
    st.image(frames_rgb[idx], channels="RGB", width="stretch",
             caption=f"Replay frame {idx + 1}/{len(frames_rgb)}")


def _extract_replay_frames(video_path: str, start_frame: int, end_frame: int, profile: str) -> list:
    from pipeline import GuardianPipeline

    cap = cv2.VideoCapture(video_path)
    s = max(0, start_frame - REPLAY_BUFFER_FRAMES)
    e = end_frame + REPLAY_BUFFER_FRAMES
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)

    pipeline = GuardianPipeline(model_path=MODEL_PATH, config_path=CONFIG_PATH, profile=profile, db_path=None)
    out = []
    try:
        idx = s
        while idx <= e:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process_frame(frame, draw=True)
            out.append(cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB))
            idx += 1
    finally:
        cap.release()
        pipeline.close()
    return out
