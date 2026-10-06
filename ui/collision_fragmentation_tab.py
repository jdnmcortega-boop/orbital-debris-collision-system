"""ORION-X collision and fragmentation simulation tab.

This module provides a research/demo fragmentation model for two colliding
objects. It is intentionally labeled as an estimate: real breakup outcomes
depend on geometry, materials, impact angle, structural configuration, and
energy partitioning.

The UI connects collision energetics to a visual fragment cloud and a
post-collision capture-priority workflow.
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

from modules import data_loader

EARTH_RADIUS_KM = 6378.137


def _load_objects():
    try:
        df = data_loader.load_orbital_data()
    except Exception:
        return pd.DataFrame()
    if df is None:
        return pd.DataFrame()
    return df.copy()


def _name_column(df):
    return next((c for c in ["OBJECT_NAME", "NAME", "SATNAME"] if c in df.columns), None)


def _id_column(df):
    return next((c for c in ["NORAD_CAT_ID", "NORAD_ID", "OBJECT_ID"] if c in df.columns), None)


def _collision_energy(m1_kg, m2_kg, relative_velocity_km_s):
    v = float(relative_velocity_km_s) * 1000.0
    mu = (m1_kg * m2_kg) / max(m1_kg + m2_kg, 1e-12)
    energy_j = 0.5 * mu * v * v
    return mu, energy_j


def _fragment_count_estimate(total_mass_kg, energy_j, threshold_kg):
    """Estimate fragments above a mass threshold using a calibrated power-law demo model.

    The model is a transparent statistical approximation for visualization,
    not a replacement for NASA/ESA breakup-model implementation.
    """
    threshold = max(float(threshold_kg), 1e-9)
    total_mass = max(float(total_mass_kg), 1e-6)

    # Dimensionless severity. 1e6 J is used only as a stable visualization scale.
    severity = max(float(energy_j) / 1e6, 1e-6)

    # More energetic collisions and lower thresholds produce more fragments.
    raw = 6.0 * (severity ** 0.35) * ((total_mass / threshold) ** 0.22)
    mass_limited = max(total_mass / threshold, 1.0)
    return int(np.clip(round(raw), 1, min(20000, int(max(1.0, mass_limited)))))


def _fragment_distribution(n_fragments, total_mass_kg, seed=42):
    rng = np.random.default_rng(seed)
    n = max(int(n_fragments), 1)

    # Log-normal sizes produce many small fragments and fewer large fragments.
    diam_m = np.exp(rng.normal(np.log(0.035), 1.0, n))
    diam_m = np.clip(diam_m, 0.001, 0.75)

    density = rng.uniform(1800.0, 7800.0, n)
    masses = density * (np.pi / 6.0) * diam_m ** 3

    # Normalize only for the visual population so displayed masses remain
    # consistent with the available collision mass budget.
    scale = min(1.0, total_mass_kg / max(masses.sum(), 1e-9))
    masses *= scale

    return diam_m, masses


def _fragment_cloud(n_fragments, relative_velocity_km_s, seed=42):
    rng = np.random.default_rng(seed)
    n = max(int(n_fragments), 1)

    # Visualized fragment velocities are perturbations around the collision
    # center; they are not intended as a high-fidelity ejecta solver.
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    speeds = np.clip(rng.lognormal(np.log(max(relative_velocity_km_s * 0.02, 0.01)), 0.55, n), 0.005, 0.5)
    velocities = directions * speeds[:, None]

    # Initial positions are tightly clustered around the collision point.
    positions = rng.normal(0.0, 8.0, size=(n, 3))
    return positions, velocities


def _fragment_scene(cloud_positions, fragment_diam_m, target_a, target_b):
    fig = go.Figure()

    # Reference Earth sphere.
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(-np.pi / 2, np.pi / 2, 20)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.cos(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.cos(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.sin(v))
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, name="Earth", showscale=False,
        opacity=0.35, hoverinfo="skip"
    ))

    # Put the collision point outside Earth so the fragment cloud is visible.
    offset = np.array([EARTH_RADIUS_KM + 700.0, 0.0, 0.0])
    scale = 25.0
    points = offset + cloud_positions * scale

    sizes = np.clip(4.0 + 16.0 * np.sqrt(np.maximum(fragment_diam_m, 0.001) / 0.05), 4, 22)
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        name="Generated fragments",
        marker=dict(size=sizes, opacity=0.75),
        customdata=np.column_stack([fragment_diam_m * 1000.0]),
        hovertemplate="Fragment size: %{customdata[0]:.1f} mm<extra></extra>",
    ))

    fig.add_trace(go.Scatter3d(
        x=[offset[0] - 500, offset[0]],
        y=[0, 0], z=[0, 0],
        mode="lines+markers",
        name=f"{target_a} → collision",
        line=dict(width=6),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter3d(
        x=[offset[0] + 500, offset[0]],
        y=[0, 0], z=[0, 0],
        mode="lines+markers",
        name=f"{target_b} → collision",
        line=dict(width=6),
        marker=dict(size=5),
    ))

    fig.update_layout(
        height=700,
        title="3-D Collision → Fragmentation Cloud",
        scene=dict(
            xaxis_title="ECI X (km)",
            yaxis_title="ECI Y (km)",
            zaxis_title="ECI Z (km)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", y=-0.03),
    )
    return fig


def _priority_table(diam_m, masses, velocities):
    speed = np.linalg.norm(velocities, axis=1)
    size_score = np.clip(diam_m / 0.1, 0, 1)
    speed_score = np.clip(speed / 0.5, 0, 1)
    mass_score = np.clip(np.log10(np.maximum(masses, 1e-9) + 1) / 3, 0, 1)
    priority = 0.45 * size_score + 0.35 * speed_score + 0.20 * mass_score

    df = pd.DataFrame({
        "Fragment": [f"F-{i+1:03d}" for i in range(len(diam_m))],
        "Diameter (mm)": diam_m * 1000,
        "Mass (g)": masses * 1000,
        "Relative speed (km/s)": speed,
        "Capture priority score": priority,
    })
    return df.sort_values("Capture priority score", ascending=False).reset_index(drop=True)


def render_collision_fragmentation_tab():
    st.header("💥 Collision & Fragmentation")
    st.caption(
        "Estimate the fragmentation produced when two orbital objects collide, "
        "visualize the resulting fragment cloud, and rank fragments for ORION-X capture."
    )

    st.warning(
        "Research/demo model: fragment counts are statistical estimates, not exact predictions. "
        "A real breakup depends on impact geometry, materials, structural design, and energy partitioning."
    )

    objects = _load_objects()
    name_col = _name_column(objects)
    id_col = _id_column(objects)

    st.subheader("1. Select the two colliding objects")

    if not objects.empty and name_col:
        names = objects[name_col].astype(str).drop_duplicates().tolist()
        names = sorted(names)[:5000]
        if len(names) >= 2:
            a_name = st.selectbox("Debris / Object A", names, index=0, key="frag_object_a")
            b_options = [n for n in names if n != a_name]
            b_name = st.selectbox("Debris / Object B", b_options, index=min(1, len(b_options)-1), key="frag_object_b")
        else:
            a_name, b_name = "Object A", "Object B"
    else:
        a_name, b_name = "Object A", "Object B"

    c1, c2, c3 = st.columns(3)
    with c1:
        m1 = st.number_input("Object A mass (kg)", min_value=0.01, value=500.0, step=10.0, key="frag_m1")
    with c2:
        m2 = st.number_input("Object B mass (kg)", min_value=0.01, value=500.0, step=10.0, key="frag_m2")
    with c3:
        vrel = st.number_input("Relative collision velocity (km/s)", min_value=0.01, value=10.0, step=0.5, key="frag_vrel")

    c1, c2, c3 = st.columns(3)
    with c1:
        threshold_g = st.selectbox("Count fragments above", [1.0, 10.0, 100.0, 1000.0], index=0, key="frag_threshold")
    with c2:
        seed = st.number_input("Simulation seed", min_value=0, value=42, step=1, key="frag_seed")
    with c3:
        fragment_display = st.slider("Fragments displayed", 25, 1000, 250, 25, key="frag_display")

    mu, energy = _collision_energy(m1, m2, vrel)
    threshold_kg = threshold_g / 1000.0
    estimated_count = _fragment_count_estimate(m1 + m2, energy, threshold_kg)

    st.subheader("2. Collision energetics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total mass", f"{m1 + m2:,.1f} kg")
    c2.metric("Reduced mass", f"{mu:,.1f} kg")
    c3.metric("Collision energy", f"{energy/1e9:,.2f} GJ")
    c4.metric("Estimated fragments", f"{estimated_count:,}")

    st.latex(r"E_c = \frac{1}{2}\mu v_{rel}^2,\qquad \mu=\frac{m_1m_2}{m_1+m_2}")

    st.subheader("3. Fragmentation estimate")
    st.write(
        f"Estimated number of fragments with mass above {threshold_g:g} g: "
        f"**{estimated_count:,}**."
    )

    diam_m, masses = _fragment_distribution(estimated_count, m1 + m2, seed=int(seed))
    display_n = min(int(fragment_display), estimated_count)
    positions, velocities = _fragment_cloud(display_n, vrel, seed=int(seed))
    diam_display = diam_m[:display_n]
    masses_display = masses[:display_n]

    # Approximate size bins for an immediately readable result.
    bins = [
        (">100 mm", diam_display >= 0.1),
        ("10–100 mm", (diam_display >= 0.01) & (diam_display < 0.1)),
        ("1–10 mm", (diam_display >= 0.001) & (diam_display < 0.01)),
        ("<1 mm", diam_display < 0.001),
    ]
    size_summary = pd.DataFrame({
        "Size class": [x[0] for x in bins],
        "Displayed fragments": [int(x[1].sum()) for x in bins],
    })
    st.dataframe(size_summary, width="stretch", hide_index=True)

    st.subheader("4. 3-D collision and fragment cloud")
    fig = _fragment_scene(positions, diam_display, a_name, b_name)
    st.plotly_chart(fig, width="stretch")

    st.subheader("5. Post-collision risk and capture priority")
    priority = _priority_table(diam_display, masses_display, velocities)
    top = priority.head(10).copy()
    st.dataframe(top, width="stretch", hide_index=True)

    top_score = float(priority.iloc[0]["Capture priority score"]) if not priority.empty else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Displayed fragments", f"{display_n:,}")
    c2.metric("Highest priority score", f"{top_score:.3f}")
    c3.metric("Recommended action", "Track → screen → capture")

    st.markdown(
        "**ORION-X workflow:** Collision detection → collision energy → fragmentation estimate → "
        "generate fragment population → propagate/screen new fragments → rank collision risk → "
        "select high-priority debris → rendezvous → inspect → net capture → electrostatic retention → "
        "stabilize → tow/deorbit."
    )

    with st.expander("Model assumptions"):
        st.markdown(
            "- The fragment count uses a transparent statistical power-law approximation for the software demo.\n"
            "- Fragment sizes are sampled from a log-normal distribution to represent a population dominated by smaller pieces.\n"
            "- The displayed velocity vectors are visualization perturbations, not a validated ejecta-dynamics solver.\n"
            "- For a research paper, cite and implement an established breakup model and validate its outputs against published collision experiments or fragmentation data.\n"
            "- The 3-D Earth and collision geometry are visualization coordinates; they are not a claim of physical scale."
        )
