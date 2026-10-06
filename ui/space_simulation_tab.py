"""3-D historical space simulation + conceptual ADR mission flow for ORION-X.

The historical portion is data-driven: archived event/TLE data -> historical
replay -> SGP4 -> conjunction/TCA/miss distance/relative velocity ->
uncertainty/collision probability -> Monte Carlo/QAE.

The ADR portion is explicitly a counterfactual concept layered on the
historical event. It visualizes rendezvous, inspection, net capture,
electrostatic retention, stabilization, towing/deorbit and reentry. It does
not claim that the historical event actually had an ORION-X capture vehicle.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.historical_replay import (
    merge_tle_archives,
    propagate_satellite,
    select_element_set,
)
from ui.historical_live_dashboard import (
    execute_historical_validation,
    find_event_archives,
    format_probability,
    load_historical_catalog,
    qae_mc_figure,
    utc_datetime,
)

EARTH_RADIUS_KM = 6378.137


def _earth_trace():
    """Create a simple 3-D Earth sphere under the orbital paths."""
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(-np.pi / 2, np.pi / 2, 24)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.cos(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.cos(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.sin(v))

    return go.Surface(
        x=x,
        y=y,
        z=z,
        name="Earth",
        showscale=False,
        opacity=0.55,
        hoverinfo="skip",
    )


def _propagate_path(sat, times):
    positions = []
    velocities = []
    for t in times:
        pos, vel = propagate_satellite(sat, t)
        positions.append(pos)
        velocities.append(vel)
    return np.asarray(positions), np.asarray(velocities)


def _safe_unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _adr_paths(target_positions, target_velocities, target_times):
    """Build a clearly conceptual rendezvous/capture/deorbit visualization.

    The rendezvous is a visualized approach path, not a solved impulsive
    maneuver. The post-capture path is a controlled-deorbit concept.
    """
    capture_idx = len(target_positions) // 2
    target_at_capture = target_positions[capture_idx]
    velocity_at_capture = target_velocities[capture_idx]

    radial = _safe_unit(target_at_capture)
    orbit_normal = _safe_unit(np.cross(target_at_capture, velocity_at_capture))
    if np.linalg.norm(orbit_normal) < 1e-6:
        orbit_normal = np.array([0.0, 0.0, 1.0])
    approach_direction = _safe_unit(np.cross(orbit_normal, radial))

    # Exaggerated visual starting point so the rendezvous vehicle is visible.
    tug_start = target_positions[0] + 300.0 * approach_direction
    rendezvous_n = max(capture_idx, 12)
    rendezvous_t = np.linspace(0.0, 1.0, rendezvous_n)
    tug_rendezvous = np.array([
        tug_start * (1 - a) + target_positions[capture_idx] * a
        for a in rendezvous_t
    ])

    # Hold the capture point briefly, then represent controlled radial
    # lowering. This is intentionally a conceptual mission graphic.
    deorbit_start = target_positions[capture_idx]
    deorbit_end = radial * (EARTH_RADIUS_KM + 120.0)
    deorbit_n = 45
    deorbit_t = np.linspace(0.0, 1.0, deorbit_n)
    deorbit_path = np.array([
        deorbit_start * (1 - a) + deorbit_end * a
        for a in deorbit_t
    ])

    # A short reentry corridor from the 120-km endpoint to the atmosphere.
    reentry_end = radial * (EARTH_RADIUS_KM + 5.0)
    reentry_t = np.linspace(0.0, 1.0, 18)
    reentry_path = np.array([
        deorbit_end * (1 - a) + reentry_end * a
        for a in reentry_t
    ])

    return {
        "capture_idx": capture_idx,
        "tug_rendezvous": tug_rendezvous,
        "deorbit_path": deorbit_path,
        "reentry_path": reentry_path,
        "capture_point": target_at_capture,
    }


def _space_scene(
    archives,
    norad_a,
    norad_b,
    target_norad,
    snapshot_time,
    tca_time,
    target_name,
    other_name,
    selected_day,
):
    """Render Earth + historical orbits + conceptual ORION-X ADR sequence."""
    selected_a = select_element_set(archives[norad_a], snapshot_time)
    selected_b = select_element_set(archives[norad_b], snapshot_time)
    if selected_a is None or selected_b is None:
        raise RuntimeError("No historical TLE at or before the selected replay snapshot.")

    _, name_a, _, sat_a = selected_a
    _, name_b, _, sat_b = selected_b

    # One hour around the forecast TCA gives a readable 3-D orbital segment.
    start = tca_time - timedelta(minutes=60)
    end = tca_time + timedelta(minutes=60)
    times = pd.date_range(start=start, end=end, periods=121).to_pydatetime()

    pa, va = _propagate_path(sat_a, times)
    pb, vb = _propagate_path(sat_b, times)

    if target_norad == norad_a:
        target_positions, target_velocities = pa, va
    else:
        target_positions, target_velocities = pb, vb

    distances = np.linalg.norm(pa - pb, axis=1)
    closest = int(np.argmin(distances))
    adr = _adr_paths(target_positions, target_velocities, times)

    fig = go.Figure()
    fig.add_trace(_earth_trace())

    fig.add_trace(go.Scatter3d(
        x=pa[:, 0], y=pa[:, 1], z=pa[:, 2],
        mode="lines", name=name_a,
        line=dict(width=5),
    ))
    fig.add_trace(go.Scatter3d(
        x=pb[:, 0], y=pb[:, 1], z=pb[:, 2],
        mode="lines", name=name_b,
        line=dict(width=5),
    ))

    fig.add_trace(go.Scatter3d(
        x=[pa[closest, 0], pb[closest, 0]],
        y=[pa[closest, 1], pb[closest, 1]],
        z=[pa[closest, 2], pb[closest, 2]],
        mode="lines+markers",
        name="Closest approach / TCA",
        line=dict(width=7, dash="dash"),
        marker=dict(size=5),
    ))

    capture_point = adr["capture_point"]
    fig.add_trace(go.Scatter3d(
        x=adr["tug_rendezvous"][:, 0],
        y=adr["tug_rendezvous"][:, 1],
        z=adr["tug_rendezvous"][:, 2],
        mode="lines+markers",
        name="ORION-X rendezvous path",
        line=dict(width=7, dash="dot"),
        marker=dict(size=3),
    ))

    fig.add_trace(go.Scatter3d(
        x=[capture_point[0]],
        y=[capture_point[1]],
        z=[capture_point[2]],
        mode="markers+text",
        name="Target inspection / capture point",
        marker=dict(size=11, symbol="diamond"),
        text=["Inspect → Net capture → Electrostatic retention"],
        textposition="top center",
    ))

    # Stylized capture envelope; exaggerated only so it remains visible beside Earth.
    capture_radius = 18.0
    theta = np.linspace(0, 2 * np.pi, 50)
    fig.add_trace(go.Scatter3d(
        x=capture_point[0] + capture_radius * np.cos(theta),
        y=capture_point[1] + capture_radius * np.sin(theta),
        z=np.full_like(theta, capture_point[2]),
        mode="lines",
        name="Electrostatic net envelope (visual scale)",
        line=dict(width=6),
    ))

    fig.add_trace(go.Scatter3d(
        x=adr["deorbit_path"][:, 0],
        y=adr["deorbit_path"][:, 1],
        z=adr["deorbit_path"][:, 2],
        mode="lines",
        name="Controlled deorbit concept",
        line=dict(width=6, dash="dash"),
    ))

    fig.add_trace(go.Scatter3d(
        x=adr["reentry_path"][:, 0],
        y=adr["reentry_path"][:, 1],
        z=adr["reentry_path"][:, 2],
        mode="lines",
        name="Atmospheric reentry corridor",
        line=dict(width=7, dash="longdash"),
    ))

    fig.add_trace(go.Scatter3d(
        x=[capture_point[0]],
        y=[capture_point[1]],
        z=[capture_point[2]],
        mode="text",
        text=["CAPTURE"],
        textposition="middle center",
        name="Capture marker",
        showlegend=False,
    ))

    all_points = np.vstack([pa, pb, adr["tug_rendezvous"], adr["deorbit_path"], adr["reentry_path"]])
    max_range = max(float(np.max(np.abs(all_points))), EARTH_RADIUS_KM + 500.0)
    # Keep Earth visible while preserving orbital geometry.
    axis_range = [-max_range, max_range]

    fig.update_layout(
        height=760,
        title=(
            f"3-D historical orbit + ORION-X ADR concept — "
            f"T-{float(selected_day):.1f} days"
        ),
        scene=dict(
            xaxis_title="ECI X (km)",
            yaxis_title="ECI Y (km)",
            zaxis_title="ECI Z (km)",
            xaxis=dict(range=axis_range),
            yaxis=dict(range=axis_range),
            zaxis=dict(range=axis_range),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", y=-0.03),
    )

    return fig, float(distances[closest]), name_a, name_b, adr


def _mission_stage_table(target_name, target_type):
    stages = [
        (1, "Get orbital data", "Historical archived orbital/TLE data"),
        (2, "Validate / clean data", "Existing historical replay validation"),
        (3, "Select correct TLE / epoch", "Historical snapshot TLE at or before state"),
        (4, "Propagate with SGP4", "Historical 3-D propagation"),
        (5, "Create common 30-day time grid", "Existing replay horizon"),
        (6, "Compare every object pair", "Existing conjunction workflow"),
        (7, "Find close approaches", "Conjunction screening"),
        (8, "Find TCA / miss distance / velocity", "Existing encounter metrics"),
        (9, "Model position uncertainty", "Existing uncertainty model"),
        (10, "Calculate collision probability", "Analytic collision probability"),
        (11, "Monte Carlo estimation", "Classical MC estimate"),
        (12, "Quantum amplitude estimation", "QAE estimate"),
        (13, "Compare MC vs QAE", "Existing QAE/MC comparison"),
        (14, "Classify / rank risk", "Risk level + priority"),
        (15, "Identify object / debris type", target_type),
        (16, "Send secure warning", "QKD/classical warning workflow"),
        (17, "Select high-priority target", target_name),
        (18, "Rendezvous / approach", "Conceptual ORION-X approach path"),
        (19, "Inspect target", "Pre-capture inspection point"),
        (20, "Deploy capture system", "Electrostatic-net deployment"),
        (21, "Net capture", "Mechanical enclosure of target"),
        (22, "Electrostatic retention", "Secondary retention/stabilization"),
        (23, "Stabilize / detumble", "Conceptual post-capture stabilization"),
        (24, "Attach towing / deorbit system", "Towing interface"),
        (25, "Controlled deorbit", "Conceptual lower-orbit trajectory"),
        (26, "Atmospheric reentry", "Conceptual reentry corridor"),
    ]
    return pd.DataFrame(stages, columns=["Step", "Mission step", "ORION-X implementation / visualization"])


def render_space_simulation_tab():
    st.header("🌌 Space Simulation — Historical Event + ADR Mission")
    st.caption(
        "Historical orbital reconstruction is kept separate from the ground prototype. "
        "This tab uses the existing Historical Replay/Validation method and then visualizes "
        "how ORION-X could conceptually perform active debris removal on a high-priority target."
    )

    st.info(
        "The orbital event is real and replayed from historical data. The ORION-X rendezvous, "
        "capture, towing, deorbit, and reentry sequence is a counterfactual engineering scenario "
        "layered on that event; it is not claimed to have happened during the historical event."
    )

    catalog = load_historical_catalog()
    if catalog.empty:
        st.error("Historical event catalog is unavailable.")
        return

    available = catalog[catalog["AVAILABLE_FOR_REPLAY"] == True].copy()
    if available.empty:
        st.warning("No historical event with an archived TLE dataset is currently available for replay.")
        return

    labels = []
    label_to_index = {}
    for idx, row in available.iterrows():
        label = (
            f"{row['event_name']} — {row['event_time_utc']} "
            f"(NORAD {int(row['object_a_norad'])} / {int(row['object_b_norad'])})"
        )
        labels.append(label)
        label_to_index[label] = idx

    selected = st.selectbox("Historical event", labels, key="space_sim_event")
    row = available.loc[label_to_index[selected]]
    archives_paths = find_event_archives(str(row["event_id"]))

    default_target = 1 if "debris" in str(row["object_b"]).lower() or "r/b" in str(row["object_b"]).lower() else 0
    target_choice = st.selectbox(
        "High-priority ADR target candidate",
        [str(row["object_a"]), str(row["object_b"])],
        index=default_target,
        key="space_adr_target",
    )
    target_norad = int(row["object_a_norad"]) if target_choice == str(row["object_a"]) else int(row["object_b_norad"])
    other_name = str(row["object_b"]) if target_choice == str(row["object_a"]) else str(row["object_a"])

    target_type = str(row["event_type"]).replace("-", " ").title()
    st.caption(
        f"Selected target: **{target_choice}** (NORAD {target_norad}). "
        f"Historical event classification: **{target_type}**."
    )

    st.subheader("1–16 — Existing orbital-risk workflow")
    st.markdown(
        "**Orbital data → validation → correct TLE/epoch → SGP4 → common time grid → "
        "pair comparison → conjunction → TCA/miss distance/relative velocity → uncertainty → "
        "collision probability → Monte Carlo → QAE → MC/QAE comparison → risk ranking → "
        "object identification → secure warning**"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        rewind_days = st.number_input("Rewind (days)", 1, 90, 30, 1, key="space_rewind")
    with c2:
        snapshot_step_hours = st.selectbox("Replay step", [6, 12, 24], index=2, key="space_step")
    with c3:
        forecast_step_minutes = st.selectbox(
            "Forecast step", [30, 60, 120], index=0, key="space_forecast_step"
        )

    c1, c2 = st.columns(2)
    with c1:
        qae_eval_qubits = st.slider("QAE evaluation qubits", 4, 10, 6, key="space_qae_qubits")
    with c2:
        qae_shots = st.slider("QAE shots", 50, 500, 200, 50, key="space_qae_shots")

    current_key = (
        str(row["event_id"]),
        int(rewind_days),
        int(snapshot_step_hours),
        int(forecast_step_minutes),
        int(qae_eval_qubits),
        int(qae_shots),
    )

    if st.button("▶ Run historical space + ADR scenario", type="primary", width="stretch"):
        try:
            with st.spinner("Running the existing historical replay/validation method…"):
                result = execute_historical_validation(
                    str(row["event_id"]),
                    str(row["event_time_utc"]),
                    int(row["object_a_norad"]),
                    int(row["object_b_norad"]),
                    tuple(str(p) for p in archives_paths),
                    int(rewind_days),
                    int(snapshot_step_hours),
                    int(forecast_step_minutes),
                    int(qae_eval_qubits),
                    int(qae_shots),
                )
            st.session_state["space_sim_result"] = result
            st.session_state["space_sim_key"] = current_key
            st.success(f"Historical reconstruction complete: {len(result)} snapshots.")
        except Exception as exc:
            st.error(f"Historical reconstruction failed: {exc}")
            return

    result = st.session_state.get("space_sim_result")
    if result is None or st.session_state.get("space_sim_key") != current_key:
        st.warning("Run the reconstruction above to generate the 3-D space scenario.")
        return

    result = result.copy()
    result["DAYS_BEFORE_EVENT"] = pd.to_numeric(result["DAYS_BEFORE_EVENT"], errors="coerce")

    st.subheader("Historical event")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Event", str(row["event_id"]))
    c2.metric("Target", target_choice)
    c3.metric("Other object", other_name)
    c4.metric("Event time", str(row["event_time_utc"]).replace("T", " ").replace("Z", " UTC"))

    selected_day = st.slider(
        "Select the historical prediction state",
        min_value=float(result["DAYS_BEFORE_EVENT"].min()),
        max_value=float(result["DAYS_BEFORE_EVENT"].max()),
        value=float(result["DAYS_BEFORE_EVENT"].max()),
        step=1.0,
        key="space_timeline",
    )
    nearest = result.iloc[(result["DAYS_BEFORE_EVENT"] - selected_day).abs().argsort()[:1]].iloc[0]

    st.subheader("3-D Earth + historical orbit + ORION-X capture/deorbit concept")
    snapshot_time = utc_datetime(str(nearest["SNAPSHOT_TIME"]))
    tca_time = utc_datetime(str(nearest["FORECAST_TCA"]))
    archives = merge_tle_archives(Path(p) for p in archives_paths)

    try:
        fig, visual_miss, name_a, name_b, adr = _space_scene(
            archives,
            int(row["object_a_norad"]),
            int(row["object_b_norad"]),
            target_norad,
            snapshot_time,
            tca_time,
            target_choice,
            other_name,
            float(nearest["DAYS_BEFORE_EVENT"]),
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.error(f"3-D historical/ADR reconstruction failed: {exc}")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Replay state", f"T-{float(nearest['DAYS_BEFORE_EVENT']):.1f} d")
    c2.metric("Forecast TCA", str(nearest["FORECAST_TCA"]).replace("T", " ").replace("Z", " UTC"))
    c3.metric("Miss distance", f"{float(nearest['FORECAST_MISS_DISTANCE_KM']):.6f} km")
    c4.metric("Relative velocity", f"{float(nearest['FORECAST_RELATIVE_VELOCITY_KM_S']):.3f} km/s")
    c5.metric("Analytic P(collision)", format_probability(nearest["ANALYTIC_PC"]))

    st.caption(
        "Earth is shown as the central reference body. The two orbital paths are historical "
        "SGP4 reconstructions. The ORION-X rendezvous/capture/deorbit path is a conceptual "
        "mission overlay and is not a solved low-thrust/impulsive orbital maneuver."
    )

    st.subheader("17–26 — Active debris removal sequence")
    st.markdown(
        "**17 Select target → 18 Rendezvous/approach → 19 Inspect → 20 Deploy capture system → "
        "21 Net capture → 22 Electrostatic retention → 23 Stabilize/detumble → "
        "24 Attach towing/deorbit system → 25 Controlled deorbit → 26 Atmospheric reentry**"
    )

    mission_df = _mission_stage_table(target_choice, target_type)
    st.dataframe(mission_df, width="stretch", height=520, hide_index=True)

    st.subheader("Risk estimation: Monte Carlo vs QAE")
    st.plotly_chart(qae_mc_figure(result), width="stretch")

    st.subheader("Historical replay results")
    st.dataframe(
        result[
            [
                "SNAPSHOT_TIME",
                "DAYS_BEFORE_EVENT",
                "FORECAST_TCA",
                "FORECAST_MISS_DISTANCE_KM",
                "FORECAST_RELATIVE_VELOCITY_KM_S",
                "FORECAST_PROXIMITY_ALERT_LEVEL",
                "ANALYTIC_PC",
                "ANALYTIC_RISK_LEVEL",
                "QAE_ESTIMATE",
                "MC_ESTIMATE",
                "QAE_EVAL_QUBITS_USED",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("What the 3-D ADR overlay means"):
        st.markdown(
            "- **Historical paths:** directly reconstructed from the archived event TLEs and SGP4.\\n"
            "- **ORION-X rendezvous:** conceptual approach to the selected high-priority target.\\n"
            "- **Inspection/capture marker:** where the target is inspected before deployment.\\n"
            "- **Electrostatic net:** shown as a visible capture envelope; its size is visually exaggerated.\\n"
            "- **Stabilize/detumble:** represented as the post-capture transition before towing.\\n"
            "- **Controlled deorbit:** conceptual lowering toward a reentry corridor, not a validated maneuver solution.\\n"
            "- **Atmospheric reentry:** endpoint of the disposal concept near the atmosphere."
        )
