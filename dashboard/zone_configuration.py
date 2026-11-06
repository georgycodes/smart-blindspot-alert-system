"""
Tab 3: Zone Configuration -- view and edit the polygon danger zones for a
vehicle profile. Coordinates are frame-fractions (0-1), so they work at any
resolution. There's no drag-and-drop canvas yet (table-based editing only) --
flagged honestly rather than faked with a prettier widget that doesn't save.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from paths import CONFIG_PATH

PREVIEW_W, PREVIEW_H = 640, 360


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _save_config(data: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _draw_preview(zones: list) -> np.ndarray:
    base = st.session_state.get("last_frame_rgb")
    if base is not None:
        img = base.copy()
        img = _resize(img, PREVIEW_W, PREVIEW_H)
    else:
        img = np.full((PREVIEW_H, PREVIEW_W, 3), 30, dtype=np.uint8)
        for x in range(0, PREVIEW_W, 40):
            img[:, x] = (60, 60, 60)
        for y in range(0, PREVIEW_H, 40):
            img[y, :] = (60, 60, 60)

    import cv2
    for z in zones:
        if not z.get("active", False):
            continue
        pts = np.array([[int(px * PREVIEW_W), int(py * PREVIEW_H)] for px, py in z["polygon"]], dtype=np.int32)
        color = tuple(z.get("color", [255, 0, 0]))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, z["name"], tuple(pts[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    return cv2.resize(img, (w, h))


def render() -> None:
    st.caption(
        "Zones are polygons defined as frame-fraction coordinates (0.0-1.0), so one definition "
        "works at any camera resolution. Edit the table below, then Save. The preview uses the "
        "most recent webcam frame if the Live Monitoring tab has captured one, otherwise a blank grid."
    )

    data = _load_config()
    profile_names = list(data["profiles"].keys())
    profile_name = st.selectbox("Vehicle profile to edit", profile_names,
                                 index=profile_names.index(data.get("active_profile", "bus")))
    profile = data["profiles"][profile_name]

    if not profile.get("validated", False):
        st.warning(f"Profile '{profile_name}' is marked unvalidated -- zones are placeholders, "
                   f"not calibrated against real vehicle footage.")

    st.image(_draw_preview(profile["zones"]), caption="Zone preview", use_container_width=True)

    zone_names = [z["name"] for z in profile["zones"]]
    if not zone_names:
        st.info("This profile has no zones defined yet.")
        return

    selected = st.selectbox("Zone to edit", zone_names)
    zone = next(z for z in profile["zones"] if z["name"] == selected)

    col1, col2 = st.columns(2)
    with col1:
        zone["active"] = st.checkbox("Active (used for detection)", value=zone.get("active", False))
    with col2:
        st.caption(zone.get("note", ""))

    df = pd.DataFrame(zone["polygon"], columns=["x_frac", "y_frac"])
    edited = st.data_editor(df, num_rows="dynamic", key=f"editor_{profile_name}_{selected}")

    if st.button("Save profile"):
        try:
            points = [[float(r["x_frac"]), float(r["y_frac"])] for _, r in edited.iterrows()]
            if len(points) < 3:
                st.error("A zone polygon needs at least 3 points.")
            else:
                zone["polygon"] = points
                _save_config(data)
                st.success(f"Saved '{selected}' in profile '{profile_name}'. "
                           f"Restart monitoring to pick up the change (zones load once at start).")
        except (ValueError, KeyError) as exc:
            st.error(f"Could not save: {exc}")
