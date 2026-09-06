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
    # Higher-resolution sphere gives the Earth a smoother 3-D appearance.
    phi = np.linspace(0, np.pi, 60)
    theta = np.linspace(0, 2 * np.pi, 120)
    x = EARTH_RADIUS_KM * np.outer(np.sin(phi), np.cos(theta))
    y = EARTH_RADIUS_KM * np.outer(np.sin(phi), np.sin(theta))
    z = EARTH_RADIUS_KM * np.outer(np.cos(phi), np.ones_like(theta))
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros_like(x),
        colorscale=[[0, "#101827"], [1, "#101827"]],
        showscale=False, opacity=0.96, hoverinfo="skip",
        name="Earth", showlegend=False,
        lighting=dict(ambient=0.65, diffuse=0.8, specular=0.25, roughness=0.75),
        lightposition=dict(x=10000, y=5000, z=12000),
    )


def _ring(axis: str):
    t = np.linspace(0, 2 * np.pi, 220)
    r = EARTH_RADIUS_KM * 1.003
    if axis == "equator":
        x, y, z = r * np.cos(t), r * np.sin(t), np.zeros_like(t)
    else:
        x, y, z = np.zeros_like(t), r * np.cos(t), r * np.sin(t)
    return go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(width=1, color="#334155"),
        hoverinfo="skip", showlegend=False,
    )


def _propagate_animation(data, duration_minutes=360, step_minutes=5):
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

    selected_path = trajectory[trajectory["NORAD_CAT_ID"] == str(selected_norad)]
    if selected_path.empty:
        selected_path = trajectory.iloc[0:0]

    fig = go.Figure()
    fig.add_trace(_earth_surface())
    fig.add_trace(_ring("equator"))
    fig.add_trace(_ring("meridian"))

    # Keep objects deliberately small so the scale of LEO is visually believable.
    # Satellites use a compact sphere-like marker; debris uses a smaller diamond.
    for is_debris, label, symbol, size in [
        (False, "Satellites", "circle", 3.2),
        (True, "Debris", "diamond", 2.2),
    ]:
        part = current[[debris[i] == is_debris for i in ids]].copy()
        fig.add_trace(go.Scatter3d(
            x=part["X_KM"], y=part["Y_KM"], z=part["Z_KM"],
            mode="markers",
            name=label,
            marker=dict(size=size, symbol=symbol, opacity=0.92),
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
            line=dict(width=3),
            hovertemplate="Orbit path<extra></extra>",
        ))

    frames = []
    for time_value, group in trajectory.groupby("TIME", sort=True):
        lookup = group.set_index("NORAD_CAT_ID")
        frame_data = []
        for is_debris, label, symbol, size in [
            (False, "Satellites", "circle", 3.2),
            (True, "Debris", "diamond", 2.2),
        ]:
            part_ids = [i for i in ids if debris[i] == is_debris and i in lookup.index]
            frame_data.append(go.Scatter3d(
                x=[lookup.loc[i, "X_KM"] for i in part_ids],
                y=[lookup.loc[i, "Y_KM"] for i in part_ids],
                z=[lookup.loc[i, "Z_KM"] for i in part_ids],
                customdata=[[i, names[i]] for i in part_ids],
                mode="markers",
                marker=dict(size=size, symbol=symbol, opacity=0.92),
                name=label,
            ))
        frames.append(go.Frame(
            name=pd.Timestamp(time_value).isoformat(),
            data=frame_data,
            traces=[3, 4],
        ))

    fig.frames = frames
    frame_duration = 100
    fig.update_layout(
        title="Live 3-D LEO Orbital Tracker",
        height=760,
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="#080d17",
        plot_bgcolor="#080d17",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01),
        scene=dict(
            xaxis=dict(title="X (km)", showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(title="Y (km)", showgrid=False, zeroline=False, showbackground=False),
            zaxis=dict(title="Z (km)", showgrid=False, zeroline=False, showbackground=False),
            aspectmode="data", bgcolor="#030711",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.10)),
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
            "currentvalue": {"prefix": "UTC: "},
            "steps": [
                {
                    "label": pd.Timestamp(t).strftime("%m-%d %H:%M"),
                    "method": "animate",
                    "args": [[pd.Timestamp(t).isoformat()], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                }
                for t in trajectory["TIME"].drop_duplicates().sort_values()
            ],
        }],
    )
    return fig


def render_live_orbit_animation():
    st.subheader("🌍 Live Interactive Orbital Tracker")
    st.caption("Small 3-D satellite/debris markers move through SGP4-propagated positions while the selected object shows its orbital path. Rotate, zoom, hover, or use the timeline.")

    orbital = data_loader.load_orbital_data()
    if orbital is None or orbital.empty:
        st.warning("No current orbital dataset is available.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        duration = st.selectbox(
            "Animation duration",
            [60, 180, 360, 720, 1440],
            index=2,
            format_func=lambda x: f"{x // 60} hour" if x % 60 == 0 and x // 60 == 1 else (f"{x // 60} hours" if x % 60 == 0 else f"{x} minutes"),
        )
    with c2:
        step = st.selectbox(
            "Animation step",
            [1, 5, 10, 15, 30],
            index=1,
            format_func=lambda x: f"{x} minute" if x == 1 else f"{x} minutes",
        )
    with c3:
        refresh = st.button("🔄 Refresh current positions", type="primary", width="stretch")

    requested_settings = (int(duration), int(step))
    cached_settings = st.session_state.get("orbit_anim_settings")
    needs_propagation = (
        refresh
        or "orbit_anim_positions" not in st.session_state
        or "orbit_anim_trajectory" not in st.session_state
        or cached_settings != requested_settings
    )

    if needs_propagation:
        with st.spinner(f"Propagating {duration} minutes of future orbital motion with SGP4…"):
            positions, failed_now = sgp4_propagation.propagate_all_now(orbital, verbose=False)
            trajectory, failed_path = _propagate_animation(orbital, int(duration), int(step))
        st.session_state["orbit_anim_positions"] = positions
        st.session_state["orbit_anim_trajectory"] = trajectory
        st.session_state["orbit_anim_failed"] = failed_now + failed_path
        st.session_state["orbit_anim_settings"] = requested_settings

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
    c3.metric("Forecast span", f"{duration / 60:g} h")

    fig = build_animated_globe(positions, trajectory, selected_norad)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})

    failed = st.session_state.get("orbit_anim_failed", [])
    if failed:
        st.caption(f"Propagation failures: {len(failed)} object/time-series jobs.")

    st.info("Scientific note: the moving markers and orbit line are SGP4 propagation visualizations. The small marker shapes are visual representations of satellites/debris, not scale-accurate physical models. They are not, by themselves, collision predictions or ground-track predictions.")
