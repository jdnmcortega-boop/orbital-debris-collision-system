"""Streamlit tab for documented historical debris-to-rocket collisions."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.historical_rocket_collision import load_events, summarize_events


EVENT_FILE = PROJECT_ROOT / "data" / "historical_rocket_debris_collisions.csv"


def render_historical_rocket_tab():
    st.header("🚀 Historical Debris → Rocket Collisions")
    st.caption(
        "Documented historical cases where orbital debris collided with a rocket body. "
        "These events are validation/reference data and are not injected into the live forecast."
    )

    if not EVENT_FILE.exists():
        st.warning(f"Historical event file not found: {EVENT_FILE}")
        return

    try:
        events = load_events(EVENT_FILE)
    except Exception as exc:
        st.error(f"Could not load historical rocket-collision data: {exc}")
        return

    summary = summarize_events(events)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documented events", summary["events"])
    c2.metric("Rocket-body targets", summary["rocket_body_targets"])
    c3.metric("Known altitudes", summary["known_altitude_events"])
    c4.metric("Known relative speeds", summary["known_relative_velocity_events"])

    st.subheader("Historical event table")
    display = events.copy()
    if "DATE_UTC" in display.columns:
        display["DATE_UTC"] = display["DATE_UTC"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    preferred = [
        "EVENT_ID", "DATE_UTC", "TARGET_NAME", "TARGET_NORAD_ID",
        "PROJECTILE_NAME", "PROJECTILE_NORAD_ID", "ALTITUDE_KM",
        "TARGET_MASS_KG", "PROJECTILE_MASS_KG", "RELATIVE_VELOCITY_KM_S",
        "IMPACT_ENERGY_MJ", "ENERGY_PER_TARGET_MASS_J_PER_G",
    ]
    columns = [c for c in preferred if c in display.columns]
    st.dataframe(display[columns], width="stretch", hide_index=True)

    st.subheader("Calculated collision quantities")
    st.markdown(
        "**Impact energy:** ½ × projectile mass × relative velocity². "
        "The calculator also reports energy per gram of target mass. "
        "These are physics-derived comparison metrics, not a fragmentation prediction."
    )

    st.download_button(
        "📥 Download calculated historical table",
        data=events.to_csv(index=False).encode("utf-8"),
        file_name="historical_rocket_debris_calculated.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Research note", expanded=False):
        st.write(
            "The historical dataset intentionally starts with documented debris-to-rocket-body "
            "collisions rather than mixing satellite-to-debris and satellite-to-satellite events. "
            "The 17 January 2005 Thor Burner 2A / CZ-4 debris event is a key documented example. "
            "Additional events should only be added when the target is a rocket body and the event "
            "is supported by a reliable source."
        )
