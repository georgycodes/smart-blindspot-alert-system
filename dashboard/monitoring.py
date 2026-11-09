"""Tab 1: Live Monitoring -- webcam feed through GuardianPipeline, live."""
import json
import time

import cv2
import streamlit as st

from paths import CONFIG_PATH, DB_PATH, MODEL_PATH, SOUNDS_DIR
from pipeline import GuardianPipeline, RISK_CRITICAL, RISK_HIGH
from video_processor import open_capture


def _profiles():
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    return list(data["profiles"].keys()), data.get("active_profile", "bus")


def _ensure_state():
    defaults = {
        "monitoring": False, "pipeline": None, "cap": None, "sound": None,
        "last_frame_rgb": None, "last_people": [], "last_overall_risk": "SAFE",
        "last_fps": 0.0, "last_people_in_zone": 0, "audio_enabled": True,
        "monitor_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _start(profile: str, conf: float, camera_index: int) -> None:
    from blindspot_detection import AlertSound
    st.session_state.pipeline = GuardianPipeline(
        model_path=MODEL_PATH, config_path=CONFIG_PATH, profile=profile,
        conf_threshold=conf, db_path=DB_PATH, source_label="webcam-dashboard",
    )
    # Lower than the CLI's 1280x720 default: smaller frames mean faster YOLO
    # inference *and* a smaller image to encode/ship to the browser each tick.
    st.session_state.cap = open_capture(camera_index, width=960, height=540)
    st.session_state.sound = AlertSound(SOUNDS_DIR, "child_alert.mp3", "adult_warning.mp3")
    st.session_state.monitoring = True
    st.session_state.monitor_error = None


def _stop() -> None:
    st.session_state.monitoring = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    if st.session_state.sound is not None:
        st.session_state.sound.shutdown()
        st.session_state.sound = None
    if st.session_state.pipeline is not None:
        st.session_state.pipeline.close()
        st.session_state.pipeline = None


@st.fragment(run_every=0.1)
def _live_fragment(frame_slot, fps_slot, zone_count_slot, risk_slot) -> None:
    """Processes and displays one frame on each auto-rerun. Streamlit does
    not pile up overlapping reruns if a run takes longer than run_every --
    it waits for the current one to finish and schedules the next right
    after, so this self-paces to whatever the hardware can actually sustain.
    (The real fix for the earlier flicker/lag was JPEG + a smaller capture
    resolution below, not this timer value.)"""
    if not st.session_state.monitoring:
        return

    from blindspot_detection import AlertSound

    cap = st.session_state.cap
    pipeline = st.session_state.pipeline
    sound = st.session_state.sound

    ok, frame = cap.read()
    if not ok:
        st.session_state.monitor_error = "Frame grab failed (camera disconnected?)."
        _stop()
        st.rerun()
        return

    frame = cv2.flip(frame, 1)
    result = pipeline.process_frame(frame, draw=True)

    st.session_state.last_people = result.people
    st.session_state.last_overall_risk = result.overall_risk
    st.session_state.last_fps = result.fps
    st.session_state.last_people_in_zone = result.people_in_zone

    rgb = cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame_rgb = rgb
    # JPEG instead of the default PNG: noticeably faster to encode and much
    # smaller to ship over the local websocket, which is most of the "lag"
    # beyond the YOLO inference time itself.
    frame_slot.image(rgb, channels="RGB", width="stretch", output_format="JPEG")
    # Update inside the same fragment as the image, not as separate st.metric()
    # calls in the outer script -- those only refresh on a full-page rerun,
    # which would otherwise leave them stuck while the video itself updates.
    fps_slot.metric("FPS", f"{result.fps:.1f}")
    zone_count_slot.metric("People in danger zone", result.people_in_zone)
    risk_slot.metric("Overall risk", result.overall_risk)

    if sound is not None:
        if st.session_state.audio_enabled and result.overall_risk == RISK_CRITICAL:
            sound.set_level(AlertSound.LEVEL_CRITICAL)
        elif st.session_state.audio_enabled and result.overall_risk == RISK_HIGH:
            sound.set_level(AlertSound.LEVEL_WARNING)
        else:
            sound.set_level(AlertSound.LEVEL_NONE)


def render() -> None:
    _ensure_state()
    profiles, default_profile = _profiles()

    col_ctrl, col_view = st.columns([1, 3])

    with col_ctrl:
        st.subheader("Controls")
        profile = st.selectbox(
            "Vehicle profile", profiles,
            index=profiles.index(default_profile) if default_profile in profiles else 0,
            disabled=st.session_state.monitoring, key="monitor_profile_select",
        )
        camera_index = st.number_input("Webcam index", min_value=0, max_value=5, value=0, step=1,
                                        disabled=st.session_state.monitoring)
        conf = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
        st.session_state.audio_enabled = st.checkbox("Enable audio alerts", value=st.session_state.audio_enabled)

        if st.session_state.monitor_error:
            st.error(st.session_state.monitor_error)

        if not st.session_state.monitoring:
            if st.button("Start monitoring", type="primary"):
                try:
                    _start(profile, conf, int(camera_index))
                    st.rerun()
                except RuntimeError as exc:
                    st.session_state.monitor_error = str(exc)
                    st.rerun()
        else:
            if st.button("Stop monitoring"):
                _stop()
                st.rerun()
            if st.session_state.pipeline is not None:
                st.session_state.pipeline.conf_threshold = conf

        st.divider()
        fps_slot = st.empty()
        zone_count_slot = st.empty()
        risk_slot = st.empty()
        fps_slot.metric("FPS", f"{st.session_state.last_fps:.1f}")
        zone_count_slot.metric("People in danger zone", st.session_state.last_people_in_zone)
        risk_slot.metric("Overall risk", st.session_state.last_overall_risk)
        st.caption(f"Audio alert status: {'armed' if st.session_state.audio_enabled else 'muted'}")

    with col_view:
        frame_slot = st.empty()
        if not st.session_state.monitoring:
            if st.session_state.last_frame_rgb is not None:
                frame_slot.image(st.session_state.last_frame_rgb, channels="RGB",
                                  width="stretch", caption="Last frame (monitoring stopped)")
            else:
                frame_slot.info("Monitoring stopped. Press Start monitoring to begin.")
        else:
            _live_fragment(frame_slot, fps_slot, zone_count_slot, risk_slot)
