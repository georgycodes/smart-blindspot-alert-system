"""
Tab 5: Scenario Simulator -- upload or pick a prerecorded video and run the
same GuardianPipeline over it frame by frame (no webcam needed), with
start/pause/resume/stop controls and a per-run event summary. Each run tags
its events with the video's filename so Incident Analytics can tell live
webcam runs apart from simulator runs on the same video.
"""
import os
import time

import cv2
import streamlit as st

from event_manager import query_events
from paths import CONFIG_PATH, DB_PATH, MODEL_PATH, VIDEOS_DIR
from pipeline import GuardianPipeline


def _ensure_state() -> None:
    defaults = {
        "sim_playing": False, "sim_paused": False, "sim_pipeline": None, "sim_cap": None,
        "sim_video_name": None, "sim_total_frames": 0, "sim_frame_idx": 0,
        "sim_last_frame_rgb": None, "sim_last_risk": "SAFE", "sim_speed": 1.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _existing_videos() -> list:
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    return sorted(f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".avi", ".mov")))


def _start(video_path: str, profile: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open '{video_path}'.")
        return
    st.session_state.sim_cap = cap
    st.session_state.sim_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.session_state.sim_frame_idx = 0
    st.session_state.sim_video_name = os.path.basename(video_path)
    st.session_state.sim_pipeline = GuardianPipeline(
        model_path=MODEL_PATH, config_path=CONFIG_PATH, profile=profile,
        db_path=DB_PATH, source_label=st.session_state.sim_video_name,
    )
    st.session_state.sim_playing = True
    st.session_state.sim_paused = False


def _stop() -> None:
    st.session_state.sim_playing = False
    st.session_state.sim_paused = False
    if st.session_state.sim_cap is not None:
        st.session_state.sim_cap.release()
        st.session_state.sim_cap = None
    if st.session_state.sim_pipeline is not None:
        st.session_state.sim_pipeline.close()
        st.session_state.sim_pipeline = None


@st.fragment(run_every=0.03)
def _playback_fragment(frame_slot, progress_slot) -> None:
    if not st.session_state.sim_playing or st.session_state.sim_paused:
        return

    cap = st.session_state.sim_cap
    pipeline = st.session_state.sim_pipeline

    ok, frame = cap.read()
    if not ok:
        progress_slot.success(f"Finished processing '{st.session_state.sim_video_name}'.")
        _stop()
        st.rerun()
        return

    st.session_state.sim_frame_idx += 1
    result = pipeline.process_frame(frame, draw=True)
    rgb = cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB)
    st.session_state.sim_last_frame_rgb = rgb
    st.session_state.sim_last_risk = result.overall_risk
    frame_slot.image(rgb, channels="RGB", use_container_width=True)

    total = st.session_state.sim_total_frames or 1
    progress_slot.progress(min(st.session_state.sim_frame_idx / total, 1.0),
                            text=f"Frame {st.session_state.sim_frame_idx}/{total} -- risk: {result.overall_risk}")

    time.sleep(max(0.0, 0.03 / max(st.session_state.sim_speed, 0.1)))


def render() -> None:
    _ensure_state()

    import json
    with open(CONFIG_PATH) as f:
        profiles = list(json.load(f)["profiles"].keys())

    st.caption("Upload a video or pick one already in data/videos/, then run it through the "
               "same detection+tracking+zone+risk+prediction pipeline used for live monitoring.")

    uploaded = st.file_uploader("Upload a test video", type=["mp4", "avi", "mov"],
                                 disabled=st.session_state.sim_playing)
    if uploaded is not None:
        dest = os.path.join(VIDEOS_DIR, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved to data/videos/{uploaded.name}")

    existing = _existing_videos()
    if not existing:
        st.info("No videos in data/videos/ yet -- upload one above.")
        return

    selected = st.selectbox("Test video", existing, disabled=st.session_state.sim_playing,
                             key="sim_video_select")
    profile = st.selectbox("Vehicle profile", profiles, disabled=st.session_state.sim_playing,
                            key="sim_profile_select")
    st.session_state.sim_speed = st.slider("Playback speed", 0.25, 3.0, 1.0, 0.25)

    col1, col2, col3 = st.columns(3)
    if not st.session_state.sim_playing:
        if col1.button("Start processing", type="primary"):
            _start(os.path.join(VIDEOS_DIR, selected), profile)
            st.rerun()
    else:
        if st.session_state.sim_paused:
            if col1.button("Resume"):
                st.session_state.sim_paused = False
                st.rerun()
        else:
            if col1.button("Pause"):
                st.session_state.sim_paused = True
                st.rerun()
        if col2.button("Stop"):
            _stop()
            st.rerun()

    frame_slot = st.empty()
    progress_slot = st.empty()

    if st.session_state.sim_playing:
        _playback_fragment(frame_slot, progress_slot)
    elif st.session_state.sim_last_frame_rgb is not None:
        frame_slot.image(st.session_state.sim_last_frame_rgb, channels="RGB", use_container_width=True)

    if st.session_state.sim_video_name:
        st.subheader(f"Events logged for '{st.session_state.sim_video_name}'")
        rows = query_events(DB_PATH, limit=100)
        rows = [r for r in rows if r["source_video"] == st.session_state.sim_video_name]
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True, hide_index=True)
        else:
            st.caption("No safety-risk events logged for this run yet.")
