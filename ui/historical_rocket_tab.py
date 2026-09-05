"""Streamlit tab for documented historical debris-to-rocket collisions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.historical_rocket_collision import load_events, summarize_events
from modules.historical_replay import merge_tle_archives, propagate_satellite, select_element_set


EVENT_FILE = PROJECT_ROOT / "data" / "historical_rocket_debris_collisions.csv"
RECONSTRUCTION_DIR = PROJECT_ROOT / "results" / "historical_rocket_2005"
RECONSTRUCTION_FILE = RECONSTRUCTION_DIR / "pre_event_window_1min.csv"
TARGET_TLE = PROJECT_ROOT / "data" / "historical_tle" / "dmsp_7219_event_2005-01-17.tle"
PROJECTILE_TLE = PROJECT_ROOT / "data" / "historical_tle" / "cz4_debris_26207_event_2005-01-17.tle"

EVENT_TIME = datetime(2005, 1, 17, 2, 14, tzinfo=timezone.utc)
WINDOW_START = EVENT_TIME - timedelta(hours=6)
TARGET_NORAD = 7219
PROJECTILE_NORAD = 26207
NASA_REFERENCE_ALTITUDE_KM = 885.0
NASA_REPORTED_SPEED_TEXT = "just under 6 km/s"


def _reconstruction_from_csv() -> pd.DataFrame | None:
    if not RECONSTRUCTION_FILE.exists():
        return None
    try:
        df = pd.read_csv(RECONSTRUCTION_FILE)
        if {"TIME_UTC", "SEPARATION_KM", "RELATIVE_SPEED_KM_S"}.issubset(df.columns):
            return df
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def run_2005_reconstruction() -> pd.DataFrame:
    """Reconstruct the six-hour pre-event window with archived TLEs and SGP4."""
    archives = merge_tle_archives([TARGET_TLE, PROJECTILE_TLE])

    rows = []
    t = WINDOW_START
    while t <= EVENT_TIME:
        states = {}
        for norad in (TARGET_NORAD, PROJECTILE_NORAD):
            selected = select_element_set(archives[norad], t)
            if selected is None:
                raise RuntimeError(
                    f"No archived TLE available for NORAD {norad} at {t.isoformat()}"
                )
            epoch, name, line1, sat = selected
            position, velocity = propagate_satellite(sat, t)
            states[norad] = (name, epoch, line1, position, velocity)

        target = states[TARGET_NORAD]
        projectile = states[PROJECTILE_NORAD]
        dr = projectile[3] - target[3]
        dv = projectile[4] - target[4]

        rows.append(
            {
                "TIME_UTC": t.isoformat(),
                "TARGET_NORAD": TARGET_NORAD,
                "PROJECTILE_NORAD": PROJECTILE_NORAD,
                "TARGET_TLE_EPOCH": target[1].isoformat(),
                "PROJECTILE_TLE_EPOCH": projectile[1].isoformat(),
                "SEPARATION_KM": float(np.linalg.norm(dr)),
                "RELATIVE_SPEED_KM_S": float(np.linalg.norm(dv)),
                "IS_VALIDATION_EVENT_TIME": int(t == EVENT_TIME),
            }
        )
        t += timedelta(minutes=1)

    df = pd.DataFrame(rows)
    RECONSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RECONSTRUCTION_FILE, index=False)
    return df


def historical_reconstruction_figure(df: pd.DataFrame):
    plot = df.copy()
    plot["TIME_UTC"] = pd.to_datetime(plot["TIME_UTC"], utc=True)
    plot["SEPARATION_KM"] = pd.to_numeric(plot["SEPARATION_KM"], errors="coerce")

    event_rows = plot[plot["IS_VALIDATION_EVENT_TIME"].astype(int) == 1]
    event_time = event_rows.iloc[0]["TIME_UTC"] if not event_rows.empty else EVENT_TIME
    closest_idx = plot["SEPARATION_KM"].idxmin()
    closest = plot.loc[closest_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot["TIME_UTC"],
            y=plot["SEPARATION_KM"],
            mode="lines",
            name="SGP4 separation",
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>%{y:.3f} km<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[closest["TIME_UTC"]],
            y=[closest["SEPARATION_KM"]],
            mode="markers",
            name="Closest sampled separation",
            hovertemplate="Closest sampled<br>%{x|%H:%M} UTC<br>%{y:.3f} km<extra></extra>",
        )
    )
    fig.add_vline(
        x=event_time,
        line_dash="dash",
        annotation_text="Reported event time",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Six-hour pre-event SGP4 reconstruction",
        xaxis_title="Time (UTC)",
        yaxis_title="Separation (km)",
        height=500,
        margin=dict(l=50, r=30, t=70, b=50),
        hovermode="x unified",
    )
    return fig


def render_reconstruction_section():
    st.subheader("🔬 Reproducible 2005 SGP4 reconstruction")
    st.caption(
        "Historical orbital reconstruction of the 17 January 2005 DMSP 5B F5 / CZ-4B debris event. "
        "Archived TLEs are propagated with SGP4 across the six hours before the documented event time."
    )

    if not TARGET_TLE.exists() or not PROJECTILE_TLE.exists():
        st.error(
            "The archived TLE inputs for the 2005 reconstruction are unavailable. "
            "Expected files are data/historical_tle/dmsp_7219_event_2005-01-17.tle and "
            "data/historical_tle/cz4_debris_26207_event_2005-01-17.tle."
        )
        return

    cached = _reconstruction_from_csv()
    run = st.button(
        "🔄 Run / refresh 2005 SGP4 reconstruction",
        key="rocket_2005_reconstruction",
        type="primary",
        width="stretch",
    )

    if run or cached is None:
        try:
            with st.spinner("Propagating the archived TLEs across the six-hour pre-event window…"):
                cached = run_2005_reconstruction()
            if run:
                st.success(f"Reconstruction complete: {len(cached):,} one-minute snapshots.")
        except Exception as exc:
            st.error(f"2005 SGP4 reconstruction failed: {exc}")
            return

    df = cached.copy()
    df["TIME_UTC"] = pd.to_datetime(df["TIME_UTC"], utc=True)
    df["SEPARATION_KM"] = pd.to_numeric(df["SEPARATION_KM"], errors="coerce")
    df["RELATIVE_SPEED_KM_S"] = pd.to_numeric(df["RELATIVE_SPEED_KM_S"], errors="coerce")

    event_rows = df[df["IS_VALIDATION_EVENT_TIME"].astype(int) == 1]
    if event_rows.empty:
        st.warning("The reconstruction file does not contain the documented validation timestamp.")
        return

    event_row = event_rows.iloc[0]
    closest = df.loc[df["SEPARATION_KM"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validation epoch", EVENT_TIME.strftime("%Y-%m-%d %H:%M UTC"))
    c2.metric("SGP4 separation", f"{float(event_row['SEPARATION_KM']):.3f} km")
    c3.metric("Relative speed", f"{float(event_row['RELATIVE_SPEED_KM_S']):.3f} km/s")
    c4.metric("Closest sampled", f"{float(closest['SEPARATION_KM']):.3f} km")

    st.plotly_chart(historical_reconstruction_figure(df), width="stretch")

    st.subheader("Reconstruction results")
    result_table = pd.DataFrame(
        [
            ["Event", "DMSP 5B F5 / CZ-4B debris"],
            ["Event / validation time", EVENT_TIME.strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["Target NORAD", TARGET_NORAD],
            ["Projectile NORAD", PROJECTILE_NORAD],
            ["SGP4 separation at event", f"{float(event_row['SEPARATION_KM']):.6f} km"],
            ["SGP4 relative speed at event", f"{float(event_row['RELATIVE_SPEED_KM_S']):.6f} km/s"],
            ["Closest sampled separation", f"{float(closest['SEPARATION_KM']):.6f} km"],
            ["Closest sampled time", closest["TIME_UTC"].strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["NASA reference altitude", f"{NASA_REFERENCE_ALTITUDE_KM:.0f} km"],
            ["NASA reported speed", NASA_REPORTED_SPEED_TEXT],
            ["Reconstruction window", f"{WINDOW_START.strftime('%H:%M UTC')}–{EVENT_TIME.strftime('%H:%M UTC')}"],
            ["Snapshots", f"{len(df):,} (1-minute cadence)"],
        ],
        columns=["Metric", "Value"],
    )
    st.dataframe(result_table, width="stretch", hide_index=True)

    st.info(
        "Interpretation: this is a historical orbital reconstruction and validation exercise, "
        "not a physical recreation of the collision. The SGP4-propagated separation is evaluated "
        "against the documented event timestamp; it should not be presented as the exact physical "
        "collision point. The current reconstruction therefore reports the computed orbital state "
        "without inventing an impact location or impact energy."
    )

    with st.expander("Archived inputs and calculation method"):
        st.write(
            "**Target:** THOR BURNER 2A R/B (NORAD 7219).  "
            "**Debris:** CZ-4B DEB (NORAD 26207)."
        )
        st.write(
            "The module selects the latest archived TLE whose epoch is available for each one-minute "
            "timestamp, propagates both objects with SGP4, then computes the Euclidean separation "
            "and relative velocity from the propagated state vectors."
        )
        st.code(
            "results/historical_rocket_2005/pre_event_window_1min.csv",
            language="text",
        )

    st.download_button(
        "📥 Download 2005 SGP4 reconstruction CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="historical_rocket_2005_pre_event_window_1min.csv",
        mime="text/csv",
        width="stretch",
    )


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

    render_reconstruction_section()

    st.subheader("Reproducible calculations")
    st.markdown(
        "For the 17 January 2005 event, the software now provides a reproducible SGP4 orbital-state "
        "reconstruction using archived pre-event TLEs. It reports separation and relative speed at "
        "the documented validation timestamp and the closest sampled separation in the six-hour "
        "window. Because the selected NASA source does not publish debris mass, no impact-energy "
        "value is inferred from RCS or other unsupported assumptions."
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
            "The SGP4 reconstruction shown above is a validation/reference workflow using archived "
            "orbital data; it is not a physical collision simulation. Additional events should only "
            "be added when the target is a rocket body and the event is supported by a reliable source."
        )
