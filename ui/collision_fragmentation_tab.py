"""ORION-X collision and fragmentation simulation tab.

NASA SSBM-based event-specific breakup model for educational/research use.
The NASA Standard Breakup Model is semi-empirical: it estimates a fragment
population statistically; it does not reproduce every individual fragment.
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
    return pd.DataFrame() if df is None else df.copy()


def _name_column(df):
    return next((c for c in ["OBJECT_NAME", "NAME", "SATNAME"] if c in df.columns), None)


def _collision_energy(m1_kg, m2_kg, relative_velocity_km_s):
    v = float(relative_velocity_km_s) * 1000.0
    mu = (m1_kg * m2_kg) / max(m1_kg + m2_kg, 1e-12)
    return mu, 0.5 * mu * v * v


def _projectile_energy_per_target_mass_j_g(m_projectile_kg, v_km_s, target_mass_kg):
    # E_p = 0.5*m_p*v^2 / m_t, expressed in J/g.
    return 0.5 * m_projectile_kg * (v_km_s * 1000.0) ** 2 / max(target_mass_kg * 1000.0, 1e-12)


def _ssbm_fragmented_mass(m_target_kg, m_projectile_kg, v_km_s, catastrophic):
    """Core NASA SSBM mass-partition relationship used for collisions.

    The published relationship is proportional to impact velocity squared and
    is capped by the available parent masses in practical implementations.
    """
    if not catastrophic:
        # Non-catastrophic collisions fragment only a fraction of each parent.
        target_fraction = min(0.5, 0.1 * v_km_s ** 2)
        projectile_fraction = min(1.0, 0.1 * v_km_s ** 2)
        return target_fraction * m_target_kg, projectile_fraction * m_projectile_kg

    # Catastrophic breakup: both parent populations are treated as fragmented.
    return m_target_kg, m_projectile_kg


def _lc_from_mass_and_density(mass_kg, density_kg_m3):
    # Equivalent-volume characteristic length. Used when no measured geometry
    # is supplied; this is a proxy, not a literal spacecraft dimension.
    volume = max(mass_kg, 1e-9) / max(density_kg_m3, 1.0)
    return (6.0 * volume / math.pi) ** (1.0 / 3.0)


def _area_from_lc(lc_m):
    # NASA SSBM area/characteristic-length relationship.
    if lc_m < 0.00167:
        return 0.540424 * lc_m ** 2
    return 0.556945 * lc_m ** 2.0047077


def _ssbm_number_above_lc(fragmented_mass_kg, lc_m):
    # Cumulative SSBM fragment-number relationship.
    # Guard against an unphysical count below one.
    n = 0.1 * max(fragmented_mass_kg, 1e-9) ** 0.75 * max(lc_m, 1e-6) ** (-1.71)
    return max(1, int(round(n)))


def _sample_lc_population(n, lc_parent_m, min_lc_m=0.001):
    """Sample a finite population consistent with a cumulative power law.

    N(>Lc) ~ Lc^-1.71. Inverse-transform sampling is used between a minimum
    characteristic length and the parent characteristic length.
    """
    rng = np.random.default_rng()
    n = max(int(n), 1)
    lo = min(max(min_lc_m, 1e-5), max(lc_parent_m * 0.999, 1.01e-5))
    hi = max(lc_parent_m, lo * 1.01)
    alpha = 1.71

    u = rng.random(n)
    lo_pow = lo ** (-alpha)
    hi_pow = hi ** (-alpha)
    lc = (lo_pow - u * (lo_pow - hi_pow)) ** (-1.0 / alpha)
    return np.clip(lc, lo, hi)


def _material_density(material):
    return {
        "Aluminum": 2700.0,
        "Steel": 7850.0,
        "Composite / plastic": 1500.0,
        "Mixed spacecraft materials": 3000.0,
    }[material]


def _cloud_from_fragments(lc_m, masses_kg, vrel_km_s, impact_angle_deg, seed):
    rng = np.random.default_rng(seed)
    n = len(lc_m)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # SSBM-style statistical delta-V visualization: larger A/M pieces receive
    # larger perturbations. This is a distributional model, not CFD/FEA.
    area = np.array([_area_from_lc(x) for x in lc_m])
    area_mass = area / np.maximum(masses_kg, 1e-12)
    chi = np.log10(np.maximum(area_mass, 1e-12))
    mu_log10_dv = 0.9 * chi + 2.90
    log10_dv = rng.normal(mu_log10_dv, 0.4)
    dv = np.clip(10.0 ** log10_dv, 0.001, max(0.5, vrel_km_s))

    # Bias the cloud toward the collision plane according to impact angle.
    theta = math.radians(float(impact_angle_deg))
    directions[:, 2] *= max(0.15, abs(math.cos(theta)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    velocities = directions * dv[:, None]
    positions = rng.normal(0.0, 3.0, size=(n, 3))
    return positions, velocities, area_mass


def _fragment_table(lc_m, masses_kg, densities, velocities, area_mass):
    speed = np.linalg.norm(velocities, axis=1)
    df = pd.DataFrame({
        "Fragment": [f"F-{i+1:04d}" for i in range(len(lc_m))],
        "Characteristic length (mm)": lc_m * 1000.0,
        "Mass (g)": masses_kg * 1000.0,
        "Density (kg/m³)": densities,
        "Area-to-mass (m²/kg)": area_mass,
        "ΔV (m/s)": speed * 1000.0,
    })
    df["Capture priority"] = (
        0.45 * np.clip(df["Characteristic length (mm)"] / 100.0, 0, 1)
        + 0.35 * np.clip(df["ΔV (m/s)"] / 500.0, 0, 1)
        + 0.20 * np.clip(np.log10(np.maximum(df["Mass (g)"], 1e-9) + 1) / 3, 0, 1)
    )
    return df.sort_values("Capture priority", ascending=False).reset_index(drop=True)


def _fragment_scene(positions, lc_m, target_a, target_b):
    fig = go.Figure()
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(-np.pi / 2, np.pi / 2, 20)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.cos(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.cos(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.sin(v))
    fig.add_trace(go.Surface(x=x, y=y, z=z, name="Earth", showscale=False, opacity=0.35, hoverinfo="skip"))

    offset = np.array([EARTH_RADIUS_KM + 700.0, 0.0, 0.0])
    points = offset + positions * 25.0
    sizes = np.clip(4.0 + 16.0 * np.sqrt(np.maximum(lc_m, 0.001) / 0.05), 4, 22)
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
        name="SSBM fragment population", marker=dict(size=sizes, opacity=0.75),
        customdata=np.column_stack([lc_m * 1000.0]),
        hovertemplate="Characteristic length: %{customdata[0]:.2f} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[offset[0] - 500, offset[0]], y=[0, 0], z=[0, 0],
        mode="lines+markers", name=f"{target_a} → collision", line=dict(width=6), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter3d(
        x=[offset[0] + 500, offset[0]], y=[0, 0], z=[0, 0],
        mode="lines+markers", name=f"{target_b} → collision", line=dict(width=6), marker=dict(size=5),
    ))
    fig.update_layout(
        height=700, title="3-D NASA SSBM-Based Fragment Cloud",
        scene=dict(xaxis_title="ECI X (km)", yaxis_title="ECI Y (km)", zaxis_title="ECI Z (km)", aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0), legend=dict(orientation="h", y=-0.03),
    )
    return fig


def render_collision_fragmentation_tab():
    st.header("💥 Collision & Fragmentation")
    st.caption(
        "NASA SSBM-based event-specific breakup model: collision energy, geometry, "
        "material/structure assumptions, fragment size, area-to-mass ratio, and ΔV."
    )

    st.warning(
        "This is a NASA Standard Breakup Model (SSBM)-based statistical estimate, not an exact "
        "reconstruction of every physical fragment. Detailed spacecraft CAD, material layup, "
        "impact geometry, and validated structural models would be required for higher-fidelity prediction."
    )

    objects = _load_objects()
    name_col = _name_column(objects)
    st.subheader("1. Collision event and parent-object properties")

    if not objects.empty and name_col:
        names = sorted(objects[name_col].astype(str).drop_duplicates().tolist())[:5000]
        if len(names) >= 2:
            a_name = st.selectbox("Debris / Object A", names, key="frag_object_a")
            b_options = [n for n in names if n != a_name]
            b_name = st.selectbox("Debris / Object B", b_options, key="frag_object_b")
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

    st.subheader("2. Geometry, material, and structural assumptions")
    c1, c2, c3 = st.columns(3)
    with c1:
        geometry = st.selectbox(
            "Parent geometry",
            ["Equivalent-volume body", "Box / spacecraft bus", "Cylinder / rocket body", "Panel-dominated structure"],
            key="frag_geometry",
        )
        lc_input = st.number_input(
            "Characteristic length (m)",
            min_value=0.01, value=2.0, step=0.1, key="frag_lc",
            help="Use a measured/engineering characteristic length when available.",
        )
    with c2:
        material = st.selectbox(
            "Dominant material",
            ["Aluminum", "Steel", "Composite / plastic", "Mixed spacecraft materials"],
            key="frag_material",
        )
        density = st.number_input(
            "Effective bulk density (kg/m³)",
            min_value=100.0, value=float(_material_density(material)), step=100.0, key="frag_density",
            help="Adjust when a documented material mixture or object-specific value is available.",
        )
    with c3:
        structure = st.selectbox(
            "Structural configuration",
            ["Intact / integrated", "Spacecraft bus + panels", "Tank / rocket-body dominated", "Highly fragmented / damaged"],
            key="frag_structure",
        )
        impact_angle = st.slider("Impact angle relative to reference axis (deg)", 0, 90, 45, 5, key="frag_angle")

    c1, c2, c3 = st.columns(3)
    with c1:
        energy_partition = st.slider(
            "Energy-to-breakup partition (%)", 1, 100, 30, 1,
            help="Research parameter for the fraction of collision energy assigned to breakup/ejecta in this event model.",
        )
    with c2:
        min_lc_mm = st.selectbox("Minimum modeled characteristic length (mm)", [1.0, 5.0, 10.0, 20.0], index=0, key="frag_min_lc")
    with c3:
        seed = st.number_input("Simulation seed", min_value=0, value=42, step=1, key="frag_seed")

    # Structural configuration affects the effective breakup mass, while the
    # actual SSBM collision regime is determined by impact energy per target mass.
    structure_factor = {
        "Intact / integrated": 1.00,
        "Spacecraft bus + panels": 0.95,
        "Tank / rocket-body dominated": 0.90,
        "Highly fragmented / damaged": 0.75,
    }[structure]
    geometry_factor = {
        "Equivalent-volume body": 1.00,
        "Box / spacecraft bus": 0.95,
        "Cylinder / rocket body": 0.92,
        "Panel-dominated structure": 0.80,
    }[geometry]

    mu, energy = _collision_energy(m1, m2, vrel)
    catastrophic_metric = _projectile_energy_per_target_mass_j_g(m2, vrel, m1)
    catastrophic = catastrophic_metric > 40.0
    fragmented_target, fragmented_projectile = _ssbm_fragmented_mass(m1, m2, vrel, catastrophic)
    effective_fragmented_mass = (
        (fragmented_target + fragmented_projectile)
        * structure_factor
        * geometry_factor
        * (energy_partition / 100.0)
    )

    # Use the user-supplied geometry when available. The density is used to
    # calculate a representative mass/geometry consistency scale.
    lc = max(float(lc_input), 0.01)
    equivalent_lc = _lc_from_mass_and_density(effective_fragmented_mass, density)
    effective_lc = max(min(lc, max(equivalent_lc * 4.0, 0.01)), 0.01)

    n_est = _ssbm_number_above_lc(effective_fragmented_mass, effective_lc)
    n_est = int(np.clip(n_est, 1, 20000))
    display_n = min(n_est, 1000)

    st.subheader("3. SSBM collision regime and energetics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total mass", f"{m1 + m2:,.1f} kg")
    c2.metric("Collision energy", f"{energy / 1e9:,.2f} GJ")
    c3.metric("Eₚ / target mass", f"{catastrophic_metric:,.1f} J/g")
    c4.metric("Regime", "Catastrophic" if catastrophic else "Non-catastrophic")

    st.latex(r"E_c=\frac{1}{2}\mu v_{rel}^2,\qquad E_p=\frac{0.5m_pv^2}{m_t}")
    st.info(
        "NASA SSBM collision screening uses 40 J/g as the catastrophic-breakup threshold. "
        "The model then estimates a statistical fragment population; it does not resolve individual structural failure."
    )

    st.subheader("4. Fragment population")
    rng = np.random.default_rng(int(seed))
    # Reproducible population without relying on global RNG state.
    lc_values = _sample_lc_population(display_n, effective_lc, float(min_lc_mm) / 1000.0)
    area_values = np.array([_area_from_lc(x) for x in lc_values])

    # Allocate the modeled fragmented mass using area/mass-informed weights,
    # then enforce conservation of the modeled mass budget.
    raw_masses = np.maximum(area_values * density, 1e-12)
    masses = raw_masses / raw_masses.sum() * max(effective_fragmented_mass, 1e-9)
    densities = np.full(display_n, density)

    positions, velocities, area_mass = _cloud_from_fragments(
        lc_values, masses, vrel, impact_angle, int(seed)
    )

    total_displayed_mass = float(masses.sum())
    st.write(
        f"SSBM cumulative estimate above the selected characteristic length: **{n_est:,} fragments**. "
        f"The visualization displays **{display_n:,}** representative fragments."
    )

    metrics = st.columns(4)
    metrics[0].metric("Modeled fragmented mass", f"{effective_fragmented_mass:,.2f} kg")
    metrics[1].metric("Displayed mass", f"{total_displayed_mass:,.2f} kg")
    metrics[2].metric("Mass-conservation error", f"{abs(total_displayed_mass-effective_fragmented_mass):.2e} kg")
    metrics[3].metric("Parent characteristic length", f"{effective_lc:.3f} m")

    bins = [
        (">100 mm", lc_values >= 0.1),
        ("10–100 mm", (lc_values >= 0.01) & (lc_values < 0.1)),
        ("1–10 mm", (lc_values >= 0.001) & (lc_values < 0.01)),
        ("<1 mm", lc_values < 0.001),
    ]
    size_summary = pd.DataFrame({
        "Size class": [x[0] for x in bins],
        "Displayed fragments": [int(x[1].sum()) for x in bins],
    })
    st.dataframe(size_summary, width="stretch", hide_index=True)

    st.subheader("5. 3-D SSBM fragment cloud")
    st.plotly_chart(_fragment_scene(positions, lc_values, a_name, b_name), width="stretch")

    st.subheader("6. Fragment properties and ORION-X priority")
    priority = _fragment_table(lc_values, masses, densities, velocities, area_mass)
    st.dataframe(priority.head(20), width="stretch", hide_index=True)

    top_score = float(priority.iloc[0]["Capture priority"]) if not priority.empty else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Displayed fragments", f"{display_n:,}")
    c2.metric("Highest capture-priority score", f"{top_score:.3f}")
    c3.metric("Next ORION-X step", "Track → screen → capture")

    st.markdown(
        "**ORION-X workflow:** collision detection → event-specific breakup estimate → "
        "fragment size/A-M/ΔV population → propagate/screen fragments → recalculate collision risk → "
        "prioritize high-risk objects → inspect → net capture → electrostatic retention → tow/deorbit."
    )

    with st.expander("Model basis and limitations"):
        st.markdown(
            "- NASA Standard Breakup Model (SSBM) relationships are semi-empirical and statistical; this tab does not claim exact individual-fragment prediction.\n"
            "- Geometry, material density, structural configuration, impact angle, and energy partition are exposed as event parameters. They modify the effective breakup population; they are not substitutes for detailed finite-element structural modeling.\n"
            "- Characteristic length, fragment count, area-to-mass ratio, and ΔV are modeled statistically.\n"
            "- TLE/orbital data alone do not provide CAD geometry, material layup, or structural design; use documented engineering data when available.\n"
            "- For publication-grade validation, compare aggregate fragment-size, mass, area-to-mass, and ΔV distributions against laboratory/on-orbit breakup datasets."
        )
