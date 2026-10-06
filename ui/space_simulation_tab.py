"""Historical space-domain simulation tab for ORION-X.

This tab is intentionally separate from the physical prototype simulation.
It uses the existing leakage-safe Historical Replay/Validation method and
archived events/TLEs; it does not use air-table or prototype capture physics.
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

from modules.historical_replay import merge_tle_archives, propagate_satellite, select_element_set
from ui.historical_live_dashboard import (
    execute_historical_validation,
    find_event_archives,
    format_probability,
    load_historical_catalog,
    qae_mc_figure,
    utc_datetime,
)


def _encounter_3d(archives, norad_a, norad_b, snapshot_time, tca_time):
    """Render an SGP4-propagated 3-D encounter around the forecast TCA."""
    selected_a = select_element_set(archives[norad_a], snapshot_time)
    selected_b = select_element_set(archives[norad_b], snapshot_time)
    if selected_a is None or selected_b is None:
        raise RuntimeError("No historical TLE at or before the selected replay snapshot.")

    _, name_a, _, sat_a = selected_a
    _, name_b, _, sat_b = selected_b

    start = tca_time - timedelta(minutes=30)
    end = tca_time + timedelta(minutes=30)
    times = pd.date_range(start=start, end=end, periods=61).to_pydatetime()

    pa, pb = [], []
    for t in times:
        a_pos, _ = propagate_satellite(sat_a, t)
        b_pos, _ = propagate_satellite(sat_b, t)
        pa.append(a_pos)
        pb.append(b_pos)

    pa = np.asarray(pa)
    pb = np.asarray(pb)
    distances = np.linalg.norm(pa - pb, axis=1)
    closest = int(np.argmin(distances))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pa[:, 0], y=pa[:, 1], z=pa[:, 2],
        mode="lines", name=name_a,
        line=dict(width=6),
    ))
    fig.add_trace(go.Scatter3d(
        x=pb[:, 0], y=pb[:, 1], z=pb[:, 2],
        mode="lines", name=name_b,
        line=dict(width=6),
    ))
    fig.add_trace(go.Scatter3d(
        x=[pa[closest, 0], pb[closest, 0]],
        y=[pa[closest, 1], pb[closest, 1]],
        z=[pa[closest, 2], pb[closest, 2]],
        mode="lines+markers", name="Closest-approach separation",
        line=dict(width=8, dash="dash"), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter3d(
        x=[pa[closest, 0]], y=[pa[closest, 1]], z=[pa[closest, 2]],
        mode="markers+text", marker=dict(size=8),
        text=["Forecast TCA"], textposition="top center",
        name="Forecast TCA",
    ))
    fig.update_layout(
        height=650,
        title="Historical SGP4 3-D encounter reconstruction",
        scene=dict(
            xaxis_title="ECI X (km)",
            yaxis_title="ECI Y (km)",
            zaxis_title="ECI Z (km)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(orientation="h", y=-0.02),
    )
    return fig, float(distances[closest]), name_a, name_b


def render_space_simulation_tab():
    st.header("🌌 Space Simulation — Historical Event Reconstruction")
    st.caption(
        "Separate space-domain simulation: documented past orbital-debris events are replayed "
        "with the existing Historical Replay/Validation method and archived orbital data."
    )

    st.info(
        "This tab does not use the air-bearing table, alloy-sheet prototype motion, or prototype "
        "capture logic. It reconstructs a real historical encounter with archived TLEs, SGP4 "
        "propagation, collision-probability estimation, QAE, and matched-budget Monte Carlo."
    )

    catalog = load_historical_catalog()
    if catalog.empty:
        st.error("Historical event catalog is unavailable.")
        return

    available = catalog[catalog["AVAILABLE_FOR_REPLAY"] == True].copy()
    if available.empty:
        st.warning("No historical event with an archived TLE/3LE dataset is currently available for replay.")
        return

    labels = []
    label_to_index = {}
    for idx, row in available.iterrows():
        label = f"{row['event_name']} — {row['event_time_utc']} (NORAD {int(row['object_a_norad'])} / {int(row['object_b_norad'])})"
        labels.append(label)
        label_to_index[label] = idx

    selected = st.selectbox("Historical event", labels, key="space_sim_event")
    row = available.loc[label_to_index[selected]]
    archives_paths = find_event_archives(str(row["event_id"]))

    st.subheader("Historical replay controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        rewind_days = st.number_input("Rewind (days)", 1, 90, 30, 1, key="space_rewind")
    with c2:
        snapshot_step_hours = st.selectbox("Replay step", [6, 12, 24], index=2, key="space_step")
    with c3:
        forecast_step_minutes = st.selectbox("Forecast step", [30, 60, 120], index=0, key="space_forecast_step")

    c1, c2 = st.columns(2)
    with c1:
        qae_eval_qubits = st.slider("QAE evaluation qubits", 4, 10, 6, key="space_qae_qubits")
    with c2:
        qae_shots = st.slider("QAE shots", 50, 500, 200, 50, key="space_qae_shots")

    current_key = (
        str(row["event_id"]), int(rewind_days), int(snapshot_step_hours),
        int(forecast_step_minutes), int(qae_eval_qubits), int(qae_shots),
    )

    if st.button("▶ Run historical space reconstruction", type="primary", width="stretch"):
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
        st.warning("Run the reconstruction above to generate the space-domain simulation.")
        return

    result = result.copy()
    result["DAYS_BEFORE_EVENT"] = pd.to_numeric(result["DAYS_BEFORE_EVENT"], errors="coerce")

    st.subheader("Historical event")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Event", str(row["event_id"]))
    c2.metric("Object A", str(row["object_a"]))
    c3.metric("Object B", str(row["object_b"]))
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

    st.subheader("3-D orbital encounter")
    snapshot_time = utc_datetime(str(nearest["SNAPSHOT_TIME"]))
    tca_time = utc_datetime(str(nearest["FORECAST_TCA"]))
    archives = merge_tle_archives(Path(p) for p in archives_paths)

    try:
        fig, visual_miss, name_a, name_b = _encounter_3d(
            archives,
            int(row["object_a_norad"]),
            int(row["object_b_norad"]),
            snapshot_time,
            tca_time,
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.error(f"3-D SGP4 encounter reconstruction failed: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Replay state", f"T-{float(nearest['DAYS_BEFORE_EVENT']):.1f} d")
    c2.metric("Forecast TCA", str(nearest["FORECAST_TCA"]).replace("T", " ").replace("Z", " UTC"))
    c3.metric("Forecast miss distance", f"{float(nearest['FORECAST_MISS_DISTANCE_KM']):.6f} km")
    c4.metric("Analytic P(collision)", format_probability(nearest["ANALYTIC_PC"]))

    st.caption(
        f"The 3-D scene uses the archived element sets available at the selected replay state "
        f"and propagates {name_a} and {name_b} with SGP4 around the forecast TCA. "
        "It is a historical orbital reconstruction, not a fabricated collision animation."
    )

    st.subheader("Existing QAE vs Monte Carlo historical method")
    st.plotly_chart(qae_mc_figure(result), width="stretch")

    st.dataframe(
        result[[
            "SNAPSHOT_TIME", "DAYS_BEFORE_EVENT", "FORECAST_TCA",
            "FORECAST_MISS_DISTANCE_KM", "FORECAST_RELATIVE_VELOCITY_KM_S",
            "FORECAST_PROXIMITY_ALERT_LEVEL", "ANALYTIC_PC", "ANALYTIC_RISK_LEVEL",
            "QAE_ESTIMATE", "MC_ESTIMATE", "QAE_EVAL_QUBITS_USED",
        ]],
        width="stretch",
        hide_index=True,
    )
