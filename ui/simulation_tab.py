"""Interactive ground-test simulation for the ORION-X debris-capture prototype.

This module is intentionally a scaled 2-D air-bearing-table demonstration.
It does not claim to reproduce orbital dynamics. It connects the existing
QAE risk-estimation workflow to a simple prototype capture-control model.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import qae


def _risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold * 0.5:
        return "MEDIUM"
    return "LOW"


def _qae_risk_probability(miss_distance_cm: float, sigma_cm: float, radius_cm: float,
                          eval_qubits: int, shots: int) -> dict:
    miss_km = float(miss_distance_cm) / 100000.0
    sigma_km = float(sigma_cm) / 100000.0
    radius_km = float(radius_cm) / 100000.0

    analytic = qae.analytic_collision_probability(
        miss_km,
        sigma_km=sigma_km,
        hard_body_radius_km=radius_km,
    )
    qae_result = qae.run_qae(
        analytic,
        num_eval_qubits=int(eval_qubits),
        shots=int(shots),
    )
    mc_samples = max(1000, min(200000, int(qae_result["oracle_calls"])))
    mc_result = qae.run_classical_mc(
        analytic,
        n_samples=mc_samples,
        seed=42,
    )

    return {
        "Analytic probability": analytic,
        "QAE estimate": qae_result["qae_estimate"],
        "QAE error": qae_result["qae_error"],
        "MC estimate": mc_result["mc_estimate"],
        "MC error": mc_result["mc_error"],
        "QAE oracle calls": qae_result["oracle_calls"],
        "QAE runtime (s)": qae_result["runtime_sec"],
    }


def _capture_frames(initial_distance_cm: float, speed_cm_s: float,
                    net_radius_cm: float, electrostatic_range_cm: float,
                    dt: float = 0.05):
    speed = max(float(speed_cm_s), 0.01)
    total_time = max(1.0, (float(initial_distance_cm) + 8.0) / speed)
    times = np.arange(0.0, total_time + dt, dt)

    distances = np.maximum(float(initial_distance_cm) - speed * times, 0.0)
    electrostatic_on = distances <= float(electrostatic_range_cm)
    net_deployed = distances <= max(float(net_radius_cm) * 1.8, 1.0)
    captured = distances <= float(net_radius_cm)

    capture_index = np.where(captured)[0]
    if len(capture_index):
        end = int(capture_index[0])
        times = times[: end + 1]
        distances = distances[: end + 1]
        electrostatic_on = electrostatic_on[: end + 1]
        net_deployed = net_deployed[: end + 1]
        captured = captured[: end + 1]

    return times, distances, electrostatic_on, net_deployed, captured


def _capture_figure(times, distances, electrostatic_on, net_deployed, captured):
    x_cube = np.zeros_like(times)
    x_debris = distances
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_cube,
            y=np.zeros_like(times),
            mode="markers",
            marker=dict(size=22, symbol="square"),
            name="CubeSat",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_debris,
            y=np.zeros_like(times),
            mode="markers",
            marker=dict(size=18, symbol="diamond"),
            name="Alloy-sheet debris",
        )
    )

    frames = []
    for i in range(len(times)):
        status = "CAPTURED" if captured[i] else (
            "ELECTROSTATIC ACTIVE" if electrostatic_on[i] else (
                "NET DEPLOYED" if net_deployed[i] else "APPROACH"
            )
        )
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Scatter(x=[0], y=[0]),
                    go.Scatter(x=[float(distances[i])], y=[0]),
                ],
                layout=go.Layout(
                    title=f"Ground capture simulation — {status} | t = {times[i]:.2f} s"
                ),
            )
        )

    fig.frames = frames
    fig.update_layout(
        height=430,
        xaxis=dict(
            title="Relative position on air-bearing table (cm)",
            range=[-8, max(10, float(np.max(distances)) + 8)],
            zeroline=True,
        ),
        yaxis=dict(
            title="Lateral position (cm)",
            range=[-8, 8],
            zeroline=True,
        ),
        title="Ground capture simulation — APPROACH",
        margin=dict(l=30, r=30, t=60, b=30),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.14,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Time: "},
                "steps": [
                    {
                        "label": f"{times[i]:.1f}s",
                        "method": "animate",
                        "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for i in range(0, len(times), max(1, len(times) // 20))
                ],
            }
        ],
    )
    return fig


def render_simulation_tab():
    st.header("🛰️ CubeSat Debris-Capture Simulation")
    st.caption(
        "Scaled ground demonstration: ORION-X risk estimation → target selection → "
        "classical capture logic → net + electrostatic capture on an air-bearing table."
    )

    st.info(
        "This is a scaled laboratory simulation, not an orbital-dynamics model. "
        "The QAE stage demonstrates the quantum risk-estimation component, while "
        "the capture stage models the proposed prototype hardware."
    )

    st.subheader("1. QAE risk-estimation setup")
    st.caption(
        "Use centimetres for the air-bearing experiment. The values are converted "
        "to kilometres internally so the existing ORION-X QAE probability model is reused."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_count = st.number_input("Debris targets", min_value=1, max_value=3, value=2, step=1)
    with c2:
        risk_threshold = st.number_input(
            "High-risk threshold",
            min_value=0.001,
            max_value=0.50,
            value=0.05,
            step=0.005,
            format="%.3f",
        )
    with c3:
        eval_qubits = st.selectbox("QAE evaluation qubits", [5, 6, 7, 8, 9], index=2)
    with c4:
        shots = st.selectbox("QAE shots", [100, 200, 500], index=1)

    defaults = [
        ("Debris A", 3.0, 0.5, 1.0),
        ("Debris B", 7.0, 0.5, 1.0),
        ("Debris C", 12.0, 0.5, 1.0),
    ]

    target_rows = []
    for i in range(int(target_count)):
        name, default_distance, default_sigma, default_radius = defaults[i]
        a, b, c, d = st.columns(4)
        with a:
            target_name = st.text_input("Target", value=name, key=f"sim_name_{i}")
        with b:
            distance = st.number_input(
                "Miss distance (cm)",
                min_value=0.01,
                max_value=100.0,
                value=default_distance,
                step=0.5,
                key=f"sim_distance_{i}",
            )
        with c:
            sigma = st.number_input(
                "Position uncertainty (cm)",
                min_value=0.01,
                max_value=20.0,
                value=default_sigma,
                step=0.1,
                key=f"sim_sigma_{i}",
            )
        with d:
            radius = st.number_input(
                "Hard-body radius (cm)",
                min_value=0.01,
                max_value=10.0,
                value=default_radius,
                step=0.1,
                key=f"sim_radius_{i}",
            )
        target_rows.append((target_name, distance, sigma, radius))

    if st.button("⚛️ Run QAE risk assessment", type="primary", width="stretch"):
        results = []
        with st.spinner("Running QAE risk estimates for the selected targets…"):
            for name, distance, sigma, radius in target_rows:
                try:
                    result = _qae_risk_probability(
                        distance, sigma, radius, eval_qubits, shots
                    )
                    results.append(
                        {
                            "Target": name,
                            "Miss distance (cm)": distance,
                            "Analytic risk": result["Analytic probability"],
                            "QAE risk": result["QAE estimate"],
                            "QAE error": result["QAE error"],
                            "MC risk": result["MC estimate"],
                            "Risk level": _risk_level(result["QAE estimate"], risk_threshold),
                        }
                    )
                except Exception as exc:
                    st.error(f"QAE failed for {name}: {exc}")
                    return
        st.session_state["simulation_results"] = pd.DataFrame(results)

    results_df = st.session_state.get("simulation_results")
    if results_df is None or results_df.empty:
        st.info("Run the QAE assessment to generate target-risk results.")
        return

    st.subheader("2. Risk assessment and target selection")
    st.dataframe(
        results_df.style.format(
            {
                "Analytic risk": "{:.6%}",
                "QAE risk": "{:.6%}",
                "QAE error": "{:.6%}",
                "MC risk": "{:.6%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    ranked = results_df.sort_values("QAE risk", ascending=False).reset_index(drop=True)
    selected_name = st.selectbox(
        "Target for physical capture demonstration",
        ranked["Target"].tolist(),
        index=0,
        help="This selection is separate from QAE. QAE supplies the risk estimate; the control logic selects a target using that result and the physical test conditions.",
    )
    selected = ranked[ranked["Target"] == selected_name].iloc[0]

    if selected["Risk level"] == "HIGH":
        st.warning(
            f"{selected_name} has the highest estimated encounter risk in this run "
            f"({selected['QAE risk']:.4%})."
        )
    elif selected["Risk level"] == "MEDIUM":
        st.info(
            f"{selected_name} is the highest-ranked target in this run, "
            f"with an estimated risk of {selected['QAE risk']:.4%}."
        )
    else:
        st.success(
            f"{selected_name} is the highest-ranked target in this run, "
            f"with an estimated risk of {selected['QAE risk']:.4%}."
        )

    st.subheader("3. Air-bearing-table capture model")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        initial_distance = st.number_input(
            "Initial separation (cm)",
            min_value=5.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            key="capture_initial_distance",
        )
    with c2:
        approach_speed = st.number_input(
            "Approach speed (cm/s)",
            min_value=0.5,
            max_value=20.0,
            value=3.0,
            step=0.5,
            key="capture_speed",
        )
    with c3:
        net_radius = st.number_input(
            "Net capture radius (cm)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="capture_net_radius",
        )
    with c4:
        electrostatic_range = st.number_input(
            "Electrostatic activation range (cm)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            key="capture_electrostatic_range",
        )

    start_capture = st.button(
        "🚀 Run capture simulation",
        type="secondary",
        width="stretch",
    )

    if start_capture:
        times, distances, electrostatic_on, net_deployed, captured = _capture_frames(
            initial_distance,
            approach_speed,
            net_radius,
            electrostatic_range,
        )
        st.session_state["capture_data"] = (
            times, distances, electrostatic_on, net_deployed, captured
        )

    capture_data = st.session_state.get("capture_data")
    if capture_data is None:
        st.info("Configure the air-bearing parameters and run the capture simulation.")
        return

    times, distances, electrostatic_on, net_deployed, captured = capture_data

    final_captured = bool(captured[-1])
    electrostatic_time = (
        float(times[np.where(electrostatic_on)[0][0]]) if np.any(electrostatic_on) else None
    )
    net_time = (
        float(times[np.where(net_deployed)[0][0]]) if np.any(net_deployed) else None
    )
    capture_time = (
        float(times[np.where(captured)[0][0]]) if final_captured else None
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected target", selected_name)
    m2.metric("QAE risk", f"{float(selected['QAE risk']):.4%}")
    m3.metric("Electrostatic activation", f"{electrostatic_time:.2f} s" if electrostatic_time is not None else "Not reached")
    m4.metric("Capture status", "CAPTURED" if final_captured else "NOT CAPTURED")

    st.plotly_chart(
        _capture_figure(
            times, distances, electrostatic_on, net_deployed, captured
        ),
        width="stretch",
    )

    if final_captured:
        st.success(
            f"Capture condition reached at t = {capture_time:.2f} s. "
            "The simulated sequence demonstrates the control path from risk assessment "
            "to net deployment and electrostatic capture."
        )
    else:
        st.warning(
            "The target did not enter the configured net-capture radius during this run. "
            "Adjust the approach speed, initial separation, or capture radius and rerun."
        )

    with st.expander("Method trace"):
        trace = pd.DataFrame(
            {
                "Time (s)": times,
                "Separation (cm)": distances,
                "Electrostatic active": electrostatic_on,
                "Net deployed": net_deployed,
                "Captured": captured,
            }
        )
        st.dataframe(trace, width="stretch", height=260, hide_index=True)
