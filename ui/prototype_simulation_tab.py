"""3-D scaled prototype simulation for the ORION-X debris-capture hardware."""

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


def _make_frames(table_length, table_width, debris_x, debris_y, debris_vx, debris_vy,
                 catcher_x, catcher_y, detection_range, deploy_range,
                 electrostatic_range, net_radius, dt=0.08):
    half_l, half_w = table_length / 2.0, table_width / 2.0
    times = np.arange(0.0, 12.0 + dt, dt)
    x, y = float(debris_x), float(debris_y)
    vx, vy = float(debris_vx), float(debris_vy)
    rows = []
    captured_at = None

    for t in times:
        if captured_at is not None:
            x, y = catcher_x, catcher_y
        else:
            x += vx * dt
            y += vy * dt
            if abs(x) > half_l:
                x = np.clip(x, -half_l, half_l)
                vx *= -1
            if abs(y) > half_w:
                y = np.clip(y, -half_w, half_w)
                vy *= -1

            distance = math.hypot(x - catcher_x, y - catcher_y)
            detected = distance <= detection_range
            net_deployed = detected and distance <= deploy_range
            electrostatic_on = net_deployed and distance <= electrostatic_range
            if electrostatic_on and distance <= net_radius:
                captured_at = float(t)
                x, y = catcher_x, catcher_y

        distance = math.hypot(x - catcher_x, y - catcher_y)
        detected = distance <= detection_range
        net_deployed = detected and distance <= deploy_range
        electrostatic_on = net_deployed and distance <= electrostatic_range
        captured = captured_at is not None
        rows.append((t, x, y, detected, net_deployed, electrostatic_on, captured, distance))

    return rows


def _figure(rows, table_length, table_width, catcher_x, catcher_y, net_radius):
    half_l, half_w = table_length / 2.0, table_width / 2.0
    camera = np.array([catcher_x, catcher_y, 2.8])
    satellite_z, net_z = 1.45, 0.10

    table_x = [-half_l, half_l, half_l, -half_l, -half_l]
    table_y = [-half_w, -half_w, half_w, half_w, -half_w]
    table_z = [0, 0, 0, 0, 0]

    def traces(row):
        _, dx, dy, detected, net_deployed, electrostatic_on, captured, _ = row
        net_r = net_radius if net_deployed else max(net_radius * 0.35, 0.4)
        if electrostatic_on:
            net_r *= 1.12

        theta = np.linspace(0, 2 * np.pi, 40)
        ring_x = catcher_x + net_r * np.cos(theta)
        ring_y = catcher_y + net_r * np.sin(theta)
        ring_z = np.full_like(theta, net_z)

        net_lines = []
        for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
            net_lines.extend([
                (catcher_x + 0.30 * np.cos(angle), catcher_y + 0.30 * np.sin(angle), satellite_z - 0.20),
                (catcher_x + net_r * np.cos(angle), catcher_y + net_r * np.sin(angle), net_z),
            ])

        status = "CAPTURED" if captured else (
            "ELECTROSTATIC ACTIVE" if electrostatic_on else (
                "NET DEPLOYED" if net_deployed else (
                    "DETECTED" if detected else "SEARCHING"
                )
            )
        )

        return [
            go.Scatter3d(x=table_x, y=table_y, z=table_z, mode="lines",
                         name="Air-bearing table", line=dict(width=7)),
            go.Scatter3d(x=[catcher_x], y=[catcher_y], z=[satellite_z],
                         mode="markers+text", marker=dict(size=15, symbol="square"),
                         text=["Scaled satellite / catcher"], textposition="top center",
                         name="Scaled satellite / catcher"),
            go.Scatter3d(x=[catcher_x], y=[catcher_y], z=[satellite_z + 0.30],
                         mode="markers+text", marker=dict(size=8, symbol="circle"),
                         text=["Camera"], textposition="top center",
                         name="Camera tracking"),
            go.Scatter3d(x=ring_x, y=ring_y, z=ring_z, mode="lines",
                         name="Electrostatic net", line=dict(width=6)),
            go.Scatter3d(
                x=[p[0] for p in net_lines], y=[p[1] for p in net_lines],
                z=[p[2] for p in net_lines], mode="lines",
                name="Net supports", line=dict(width=3)
            ),
            go.Scatter3d(x=[dx], y=[dy], z=[0.10], mode="markers+text",
                         marker=dict(size=13, symbol="diamond"),
                         text=["Alloy-sheet debris"], textposition="bottom center",
                         name="Alloy-sheet debris"),
            go.Scatter3d(x=[camera[0], dx], y=[camera[1], dy], z=[camera[2], 0.10],
                         mode="lines", name="Camera tracking line",
                         line=dict(width=3, dash="dash")),
            go.Scatter3d(x=[catcher_x], y=[catcher_y], z=[2.55], mode="text",
                         text=[status], textposition="middle center",
                         name="System status"),
        ]

    fig = go.Figure(data=traces(rows[0]))
    fig.frames = [
        go.Frame(
            name=str(i),
            data=traces(row),
            layout=go.Layout(
                title=(
                    "3-D prototype capture — "
                    + ("CAPTURED" if row[6] else
                       "ELECTROSTATIC ACTIVE" if row[5] else
                       "NET DEPLOYED" if row[4] else
                       "DETECTED" if row[3] else "SEARCHING")
                    + f" | t = {row[0]:.2f} s"
                )
            ),
        )
        for i, row in enumerate(rows)
    ]

    fig.update_layout(
        height=650,
        title="3-D prototype capture — SEARCHING",
        scene=dict(
            xaxis_title="Air-bearing table X (cm)",
            yaxis_title="Air-bearing table Y (cm)",
            zaxis_title="Prototype height (cm)",
            xaxis=dict(range=[-half_l - 3, half_l + 3]),
            yaxis=dict(range=[-half_w - 3, half_w + 3]),
            zaxis=dict(range=[0, 3.3]),
            aspectmode="manual",
            aspectratio=dict(x=1.35, y=1.0, z=0.55),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", y=-0.02),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=1.08,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 70, "redraw": True},
                                  "transition": {"duration": 0}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            currentvalue={"prefix": "Time: "},
            steps=[
                dict(label=f"{rows[i][0]:.1f}s", method="animate",
                     args=[[str(i)], {"mode": "immediate",
                                      "frame": {"duration": 0, "redraw": True}}])
                for i in range(0, len(rows), max(1, len(rows) // 24))
            ],
        )],
    )
    return fig


def render_prototype_simulation_tab():
    st.header("🧪 Prototype Simulation — 3-D Air-Bearing Table")
    st.caption(
        "Scaled laboratory model of the proposed orbital debris-capture hardware. "
        "The air-bearing table provides controlled low-friction motion; the target is a flat alloy-sheet debris model."
    )
    st.info(
        "This tab represents the ground prototype only. The scaled satellite/catcher is above "
        "the air-bearing table, the alloy sheet slides on the table, and the camera tracks the "
        "target before the electrostatic net is deployed."
    )

    st.subheader("Prototype arrangement")
    st.markdown(
        "**Camera tracking** → **alloy-sheet detection** → **scaled satellite/catcher** → "
        "**electrostatic net deployment** → **capture and retention**"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        table_length = st.number_input("Table length (cm)", 40.0, 120.0, 70.0, 5.0)
    with c2:
        table_width = st.number_input("Table width (cm)", 30.0, 100.0, 50.0, 5.0)
    with c3:
        net_radius = st.number_input("Net capture radius (cm)", 2.0, 15.0, 6.0, 0.5)

    st.subheader("Alloy-sheet debris motion")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        debris_x = st.number_input("Initial X (cm)", -30.0, 30.0, -24.0, 1.0)
    with c2:
        debris_y = st.number_input("Initial Y (cm)", -20.0, 20.0, -12.0, 1.0)
    with c3:
        debris_vx = st.number_input("Velocity X (cm/s)", -8.0, 8.0, 3.0, 0.5)
    with c4:
        debris_vy = st.number_input("Velocity Y (cm/s)", -8.0, 8.0, 1.5, 0.5)

    st.subheader("Scaled catcher + camera")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        catcher_x = st.number_input("Catcher X (cm)", -30.0, 30.0, 12.0, 1.0)
    with c2:
        catcher_y = st.number_input("Catcher Y (cm)", -20.0, 20.0, 5.0, 1.0)
    with c3:
        detection_range = st.number_input("Camera detection range (cm)", 5.0, 50.0, 32.0, 1.0)
    with c4:
        deploy_range = st.number_input("Net deployment range (cm)", 2.0, 30.0, 18.0, 1.0)

    electrostatic_range = st.number_input(
        "Electrostatic activation range (cm)", 1.0, 20.0, 9.0, 0.5
    )

    if st.button("▶ Run 3-D prototype simulation", type="primary", width="stretch"):
        rows = _make_frames(
            table_length, table_width, debris_x, debris_y, debris_vx, debris_vy,
            catcher_x, catcher_y, detection_range, deploy_range,
            electrostatic_range, net_radius,
        )
        st.session_state["prototype_simulation_rows"] = rows
        st.session_state["prototype_simulation_params"] = (
            table_length, table_width, catcher_x, catcher_y, net_radius
        )

    rows = st.session_state.get("prototype_simulation_rows")
    if rows is None:
        st.info("Set the prototype parameters, then run the 3-D simulation.")
        return

    table_length, table_width, catcher_x, catcher_y, net_radius = st.session_state[
        "prototype_simulation_params"
    ]
    first_detected = next((r for r in rows if r[3]), None)
    first_net = next((r for r in rows if r[4]), None)
    first_electro = next((r for r in rows if r[5]), None)
    captured = rows[-1][6]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Camera detection", f"T+{first_detected[0]:.2f} s" if first_detected else "Not detected")
    m2.metric("Net deployment", f"T+{first_net[0]:.2f} s" if first_net else "Not deployed")
    m3.metric("Electrostatic activation", f"T+{first_electro[0]:.2f} s" if first_electro else "Not activated")
    m4.metric("Capture status", "CAPTURED" if captured else "NOT CAPTURED")

    st.plotly_chart(
        _figure(rows, table_length, table_width, catcher_x, catcher_y, net_radius),
        width="stretch",
    )

    if captured:
        st.success("The alloy-sheet target entered the scaled electrostatic-net capture envelope.")
    else:
        st.warning("The alloy-sheet target did not enter the capture envelope during this run.")

    with st.expander("Prototype test trace"):
        trace = pd.DataFrame([
            {
                "Time (s)": r[0],
                "Debris X (cm)": r[1],
                "Debris Y (cm)": r[2],
                "Camera detected": r[3],
                "Net deployed": r[4],
                "Electrostatic active": r[5],
                "Captured": r[6],
                "Distance to catcher (cm)": r[7],
            }
            for r in rows
        ])
        st.dataframe(trace, width="stretch", height=280, hide_index=True)
