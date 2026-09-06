"""Animated interactive 3-D orbital tracker for ORION-X."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import data_loader
from modules import sgp4_propagation

EARTH_RADIUS_KM = 6378.137


def _is_debris(name: str) -> bool:
    text = str(name).upper()
    return "DEB" in text or "DEBRIS" in text or "ROCKET BODY" in text


def _earth_surface():
    phi = np.linspace(0, np.pi, 42)
    theta = np.linspace(0, 2 * np.pi, 84)
    x = EARTH_RADIUS_KM * np.outer(np.sin(phi), np.cos(theta))
    y = EARTH_RADIUS_KM * np.outer(np.sin(phi), np.sin(theta))
    z = EARTH_RADIUS_KM * np.outer(np.cos(phi), np.ones_like(theta))
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros_like(x),
        colorscale=[[0, "#162033"], [1, "#162033"]],
        showscale=False, opacity=0.82, hoverinfo="skip",
        name="Earth", showlegend=False,
    )


def _ring(axis: str):
    t = np.linspace(0, 2 * np.pi, 180)
    r = EARTH_RADIUS_KM * 1.002
    if axis == "equator":
        x, y, z = r * np.cos(t), r * np.sin(t), np.zeros_like(t)
    else:
        x, y, z = np.zeros_like(t), r * np.cos(t), r * np.sin(t)
    return go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(width=1, color="#64748b"),
        hoverinfo="skip", showlegend=False,
    )


def _propagate_animation(data, duration_minutes=30, step_minutes=1):
    now = datetime.now(timezone.utc)
    steps = int(duration_minutes / step_minutes)
    times = [now + timedelta(minutes=step_minutes * i) for i in range(steps + 1)]
    return sgp4_propagation.propagate_all(data, times=times, verbose=False)


def build_animated_globe(current_positions, trajectory, selected_norad=None):
    current = current_positions.copy()
    current["NORAD_CAT_ID"] = current["NORAD_CAT_ID"].astype(str)
    current["OBJECT_NAME"] = current["OBJECT_NAME"].astype(str)

    trajectory = trajectory.copy()
    trajectory["NORAD_CAT_ID"] = trajectory["NORAD_CAT_ID"].astype(str)
    trajectory["TIME"] = pd.to_datetime(trajectory["TIME"], utc=True)
    trajectory = trajectory.sort_values(["TIME", "NORAD_CAT_ID"])

    ids = list(current["NORAD_CAT_ID"])
    names = dict(zip(current["NORAD_CAT_ID"], current["OBJECT_NAME"]))
    debris = {norad: _is_debris(name) for norad, name in names.items()}

    # Keep the path visible for the selected object while all markers animate.
    selected_path = trajectory[trajectory["NORAD_CAT_ID"] == str(selected_norad)]
    if selected_path.empty:
        selected_path = trajectory.iloc[0:0]

    fig = go.Figure()
    fig.add_trace(_earth_surface())
    fig.add_trace(_ring("equator"))
    fig.add_trace(_ring("meridian"))

    for is_debris, label, symbol in [
        (False, "Satellites", "circle"),
        (True, "Debris", "diamond"),
    ]:
        part = current[[debris[i] == is_debris for i in ids]].copy()
        fig.add_trace(go.Scatter3d(
            x=part["X_KM"], y=part["Y_KM"], z=part["Z_KM"],
            mode="markers",
            name=label,
            marker=dict(size=6 if not is_debris else 5, symbol=symbol),
            customdata=part[["NORAD_CAT_ID", "OBJECT_NAME"]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[1]}</b>"
                "<br>NORAD: %{customdata[0]}"
                "<br>X: %{x:.1f} km"
                "<br>Y: %{y:.1f} km"
                "<br>Z: %{z:.1f} km"
                "<extra></extra>"
            ),
        ))

    if not selected_path.empty:
        fig.add_trace(go.Scatter3d(
            x=selected_path["X_KM"], y=selected_path["Y_KM"], z=selected_path["Z_KM"],
            mode="lines",
            name=f"Orbit — {names.get(str(selected_norad), selected_norad)}",
            line=dict(width=5),
            hovertemplate="Orbit path<extra></extra>",
        ))

    # Animation frames update the satellite/debris marker traces only.
    frames = []
    for time_value, group in trajectory.groupby("TIME", sort=True):
        lookup = group.set_index("NORAD_CAT_ID")
        frame_data = []
        for is_debris, label, symbol in [
            (False, "Satellites", "circle"),
            (True, "Debris", "diamond"),
        ]:
            part_ids = [i for i in ids if debris[i] == is_debris and i in lookup.index]
            frame_data.append(go.Scatter3d(
                x=[lookup.loc[i, "X_KM"] for i in part_ids],
                y=[lookup.loc[i, "Y_KM"] for i in part_ids],
                z=[lookup.loc[i, "Z_KM"] for i in part_ids],
                customdata=[[i, names[i]] for i in part_ids],
                mode="markers",
                marker=dict(size=6 if not is_debris else 5, symbol=symbol),
                name=label,
            ))
        frames.append(go.Frame(name=pd.Timestamp(time_value).isoformat(), data=frame_data, traces=[3, 4]))

    fig.frames = frames
    frame_duration = 120
    fig.update_layout(
        title="Live 3-D LEO Orbital Tracker",
        height=720,
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01),
        scene=dict(
            xaxis=dict(title="X (km)", showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(title="Y (km)", showgrid=False, zeroline=False, showbackground=False),
            zaxis=dict(title="Z (km)", showgrid=False, zeroline=False, showbackground=False),
            aspectmode="data", bgcolor="#050914",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": True,
            "x": 0.01, "y": 1.08,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.15, "y": 0.01, "len": 0.80,
            "currentvalue": {"prefix": "Time: "},
            "steps": [
                {"label": pd.Timestamp(t).strftime("%H:%M:%S"), "method": "animate", "args": [[pd.Timestamp(t).isoformat()], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for t in trajectory["TIME"].drop_duplicates().sort_values()
            ],
        }],
    )
    return fig


def render_live_orbit_animation():
    st.subheader("🌍 Live Interactive Orbital Tracker")
    st.caption("The markers move through SGP4-propagated positions. Rotate/zoom the globe, hover an object, and select an object to display its orbital path.")

    orbital = data_loader.load_orbital_data()
    if orbital is None or orbital.empty:
        st.warning("No current orbital dataset is available.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        duration = st.selectbox("Animation duration", [10, 20, 30, 60], index=2, format_func=lambda x: f"{x} minutes")
    with c2:
        step = st.selectbox("Animation step", [1, 2, 5], index=0, format_func=lambda x: f"{x} minute")
    with c3:
        refresh = st.button("🔄 Refresh current positions", type="primary", width="stretch")

    if refresh or "orbit_anim_positions" not in st.session_state or "orbit_anim_trajectory" not in st.session_state:
        with st.spinner("Propagating live orbital positions with SGP4…"):
            positions, failed_now = sgp4_propagation.propagate_all_now(orbital, verbose=False)
            trajectory, failed_path = _propagate_animation(orbital, int(duration), int(step))
        st.session_state["orbit_anim_positions"] = positions
        st.session_state["orbit_anim_trajectory"] = trajectory
        st.session_state["orbit_anim_failed"] = failed_now + failed_path
        st.session_state["orbit_anim_settings"] = (duration, step)

    positions = st.session_state.get("orbit_anim_positions")
    trajectory = st.session_state.get("orbit_anim_trajectory")
    if positions is None or positions.empty or trajectory is None or trajectory.empty:
        st.warning("No live orbital states were produced.")
        return

    options = positions[["NORAD_CAT_ID", "OBJECT_NAME"]].drop_duplicates().sort_values("OBJECT_NAME")
    labels = [f"{r.OBJECT_NAME} — NORAD {r.NORAD_CAT_ID}" for r in options.itertuples()]
    mapping = {label: str(r.NORAD_CAT_ID) for label, r in zip(labels, options.itertuples())}
    selected_label = st.selectbox("Select satellite/debris to highlight its orbit", labels)
    selected_norad = mapping[selected_label]

    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked objects", int(positions["NORAD_CAT_ID"].nunique()))
    c2.metric("Animation frames", int(trajectory["TIME"].nunique()))
    c3.metric("Selected object", selected_label.split(" — NORAD ")[0][:28])

    fig = build_animated_globe(positions, trajectory, selected_norad)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})

    failed = st.session_state.get("orbit_anim_failed", [])
    if failed:
        st.caption(f"Propagation failures: {len(failed)} object/time-series jobs.")

    st.info("Scientific note: the moving markers and orbit line are SGP4 propagation visualizations. They are not, by themselves, collision predictions or ground-track predictions.")
