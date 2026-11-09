"""
Tab 3: Zone Configuration -- visual, click-based polygon editor for a
vehicle profile's danger zones. Click points directly on the preview image
to draw a zone boundary (not full drag-and-drop yet, but a real visual
editor rather than typing raw coordinates into a table). Coordinates are
stored as frame-fractions (0-1), so one definition works at any resolution.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from paths import CONFIG_PATH

PREVIEW_W, PREVIEW_H = 640, 360


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _save_config(data: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _base_canvas() -> np.ndarray:
    base = st.session_state.get("last_frame_rgb")
    if base is not None:
        import cv2
        return cv2.resize(base, (PREVIEW_W, PREVIEW_H))
    img = np.full((PREVIEW_H, PREVIEW_W, 3), 30, dtype=np.uint8)
    for x in range(0, PREVIEW_W, 40):
        img[:, x] = (60, 60, 60)
    for y in range(0, PREVIEW_H, 40):
        img[y, :] = (60, 60, 60)
    return img


def _draw_preview(zones: list, editing_zone_name: str, draft_points: list) -> np.ndarray:
    import cv2
    img = _base_canvas().copy()

    for z in zones:
        if not z.get("active", False) or z["name"] == editing_zone_name:
            continue
        pts = np.array([[int(px * PREVIEW_W), int(py * PREVIEW_H)] for px, py in z["polygon"]], dtype=np.int32)
        color = tuple(z.get("color", [255, 0, 0]))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, dst=img)
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, z["name"], tuple(pts[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if draft_points:
        pts = np.array([[int(px * PREVIEW_W), int(py * PREVIEW_H)] for px, py in draft_points], dtype=np.int32)
        for p in pts:
            cv2.circle(img, tuple(p), 5, (255, 255, 0), -1)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], len(pts) >= 3, (255, 255, 0), 2)
        if len(pts) >= 3:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (255, 255, 0))
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)

    return img


def render() -> None:
    st.caption(
        "Click directly on the preview image to place the zone's corner points, in order around "
        "its boundary. Points are stored as frame-fraction coordinates (0.0-1.0), so one definition "
        "works at any camera resolution. The preview uses the most recent webcam frame if the Live "
        "Monitoring tab has captured one, otherwise a blank grid."
    )

    data = _load_config()
    profile_names = list(data["profiles"].keys())
    profile_name = st.selectbox("Vehicle profile to edit", profile_names,
                                 index=profile_names.index(data.get("active_profile", "bus")),
                                 key="zonecfg_profile_select")
    profile = data["profiles"][profile_name]

    zone_names = [z["name"] for z in profile["zones"]]
    if not zone_names:
        st.info("This profile has no zones defined yet.")
        return

    selected = st.selectbox("Zone to edit", zone_names, key="zonecfg_zone_select")
    zone = next(z for z in profile["zones"] if z["name"] == selected)

    draft_key = f"zonecfg_draft_{profile_name}_{selected}"
    click_key = f"zonecfg_lastclick_{profile_name}_{selected}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = [list(p) for p in zone["polygon"]]
    if click_key not in st.session_state:
        st.session_state[click_key] = None

    col1, col2 = st.columns(2)
    with col1:
        zone["active"] = st.checkbox("Active (used for detection)", value=zone.get("active", False))
    with col2:
        st.caption(zone.get("note", ""))

    preview = _draw_preview(profile["zones"], selected, st.session_state[draft_key])
    coords = streamlit_image_coordinates(preview, key=f"click_{profile_name}_{selected}")

    if coords is not None:
        click = (coords["x"], coords["y"])
        if click != st.session_state[click_key]:
            st.session_state[click_key] = click
            x_frac = round(coords["x"] / PREVIEW_W, 4)
            y_frac = round(coords["y"] / PREVIEW_H, 4)
            st.session_state[draft_key].append([x_frac, y_frac])
            st.rerun()

    n_points = len(st.session_state[draft_key])
    st.caption(f"{n_points} point(s) placed" + ("" if n_points >= 3 else " -- need at least 3 to save."))

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    if bcol1.button("Undo last point"):
        if st.session_state[draft_key]:
            st.session_state[draft_key].pop()
            st.rerun()
    if bcol2.button("Clear points"):
        st.session_state[draft_key] = []
        st.rerun()
    if bcol3.button("Reset to saved"):
        st.session_state[draft_key] = [list(p) for p in zone["polygon"]]
        st.rerun()
    if bcol4.button("Save zone polygon", type="primary"):
        if n_points < 3:
            st.error("A zone polygon needs at least 3 points.")
        else:
            zone["polygon"] = st.session_state[draft_key]
            _save_config(data)
            st.success(f"Saved '{selected}' in profile '{profile_name}'. "
                       f"Restart monitoring to pick up the change (zones load once at start).")

    if st.session_state[draft_key]:
        st.dataframe(pd.DataFrame(st.session_state[draft_key], columns=["x_frac", "y_frac"]),
                     use_container_width=True, hide_index=True)
