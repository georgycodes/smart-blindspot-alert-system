"""Tab 2: Predictive Safety -- per-person tracking/prediction/risk detail panel."""
import streamlit as st


@st.fragment(run_every=0.5)
def _live_panel() -> None:
    people = st.session_state.get("last_people", [])

    if not st.session_state.get("monitoring", False):
        st.info("Start monitoring on the Live Monitoring tab to see per-person predictive detail here.")
        return

    if not people:
        st.success("No one currently tracked.")
        return

    for p in people:
        risk_emoji = {"SAFE": "🟢", "LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(p.risk, "⚪")
        with st.container(border=True):
            st.markdown(f"**{risk_emoji} PERSON #{p.track_id:02d} — {p.risk}**")
            cols = st.columns(4)
            cols[0].metric("Zone", p.zone or "none")
            cols[1].metric("Direction", p.direction)
            cols[2].metric("Confidence", f"{p.confidence:.2f}")
            if p.predicted_zone:
                cols[3].metric("Predicted entry", f"{p.predicted_zone} in ~{p.predicted_eta_s}s")
            else:
                cols[3].metric("Predicted entry", "none / unknown")

            st.caption("Why: " + " ".join(p.reasons))


def render() -> None:
    st.caption(
        "Live per-person detail: tracked ID, which danger zone (if any) they're in, "
        "recent movement direction, and -- when there's enough tracking history -- a "
        "predicted zone entry from simple linear extrapolation, each with its reason."
    )
    _live_panel()
