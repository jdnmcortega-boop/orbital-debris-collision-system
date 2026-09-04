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
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documented events", summary["events"])
    c2.metric("Rocket-body targets", summary["rocket_body_targets"])
    c3.metric("Known altitudes", summary["known_altitude_events"])
    c4.metric("Known relative speeds", summary["known_relative_velocity_events"])
    c5.metric("Known RCS values", summary["known_rcs_events"])

    st.subheader("Historical event table")
    display = events.copy()
    if "DATE_UTC" in display.columns:
        display["DATE_UTC"] = display["DATE_UTC"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    preferred = [
        "EVENT_ID", "DATE_UTC", "TARGET_NAME", "TARGET_NORAD_ID",
        "PROJECTILE_NAME", "PROJECTILE_NORAD_ID", "ALTITUDE_KM",
        "TARGET_MASS_KG", "DEBRIS_RCS_CM2", "RELATIVE_VELOCITY_KM_S",
        "CATALOGUED_FRAGMENTS", "IMPACT_ENERGY_MJ",
    ]
    columns = [c for c in preferred if c in display.columns]
    st.dataframe(display[columns], width="stretch", hide_index=True)

    st.subheader("Source-data integrity")
    st.info(
        "The NASA source reports the debris fragment's radar cross-section as 600 cm² "
        "and the collision speed as just under 6 km/s. It does not provide a debris mass. "
        "Therefore the software does not estimate debris mass from RCS and does not report "
        "a fabricated impact-energy value."
    )

    st.subheader("Reproducible calculations")
    st.markdown(
        "When historical mass and relative speed are both available, the module can calculate "
        "kinetic energy using **E = ½mv²**. For this event, the required debris mass is not "
        "published in the selected NASA source, so the energy field correctly remains unavailable. "
        "If state vectors are supplied for the same epoch and coordinate frame, the module can "
        "also calculate relative distance, relative velocity, and closing speed."
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
            "The 17 January 2005 Thor Burner 2A / CZ-4 debris event is a documented example. "
            "Additional events should only be added when the target is a rocket body and the event "
            "is supported by a reliable source."
        )
