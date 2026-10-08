"""ORION-X collision and fragmentation tab.

Physics-informed, NASA SSBM-based statistical breakup model.
Important: TLE/orbital records do not contain spacecraft mass, CAD geometry,
material layup, or structural design, so those engineering properties remain
explicit inputs rather than being invented from an object name.
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
SSBM_MIN_LC_M = 0.001


def _load_objects():
    try:
        df = data_loader.load_orbital_data()
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame() if df is None else df.copy()


def _name_column(df):
    return next((c for c in ["OBJECT_NAME", "NAME", "SATNAME"] if c in df.columns), None)


def _id_column(df):
    return next((c for c in ["NORAD_CAT_ID", "NORAD_ID", "OBJECT_ID"] if c in df.columns), None)


def _object_row(df, name, name_col):
    if df.empty or not name_col:
        return None
    rows = df[df[name_col].astype(str) == str(name)]
    return rows.iloc[0] if not rows.empty else None


def _safe_float(row, column, default=np.nan):
    try:
        value = float(row[column])
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _kepler_epoch_velocity_km_s(row):
    """Approximate the inertial velocity vector at the record epoch.

    The orbital CSV contains classical mean elements, so this converts the
    elements into a two-body ECI state at epoch. It is intentionally labeled
    an epoch estimate: exact TCA relative velocity still requires SGP4.
    """
    if row is None:
        return None

    mm = _safe_float(row, "MEAN_MOTION")
    e = _safe_float(row, "ECCENTRICITY")
    inc = _safe_float(row, "INCLINATION")
    raan = _safe_float(row, "RA_OF_ASC_NODE")
    argp = _safe_float(row, "ARG_OF_PERICENTER")
    M_deg = _safe_float(row, "MEAN_ANOMALY")

    if not all(np.isfinite(x) for x in [mm, e, inc, raan, argp, M_deg]):
        return None
    if mm <= 0 or e < 0 or e >= 1:
        return None

    mu = 398600.4418  # km^3/s^2
    n = mm * 2.0 * math.pi / 86400.0
    a = (mu / (n * n)) ** (1.0 / 3.0)

    # Solve Kepler's equation M = E - e sin(E).
    M = math.radians(M_deg) % (2.0 * math.pi)
    E = M
    for _ in range(15):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        step = f / max(fp, 1e-12)
        E -= step
        if abs(step) < 1e-12:
            break

    r = a * (1.0 - e * math.cos(E))
    if r <= 0:
        return None

    vp = math.sqrt(mu * a) / r
    vx_p = -vp * math.sin(E)
    vy_p = vp * math.sqrt(1.0 - e * e) * math.cos(E)

    O = math.radians(raan)
    i = math.radians(inc)
    w = math.radians(argp)

    # R3(Omega) R1(i) R3(omega), applied to perifocal velocity.
    cO, sO = math.cos(O), math.sin(O)
    ci, si = math.cos(i), math.sin(i)
    cw, sw = math.cos(w), math.sin(w)

    R11 = cO * cw - sO * sw * ci
    R12 = -cO * sw - sO * cw * ci
    R21 = sO * cw + cO * sw * ci
    R22 = -sO * sw + cO * cw * ci
    R31 = sw * si
    R32 = cw * si

    return np.array([
        R11 * vx_p + R12 * vy_p,
        R21 * vx_p + R22 * vy_p,
        R31 * vx_p + R32 * vy_p,
    ])


def _pair_relative_velocity_estimate(row_a, row_b):
    va_vec = _kepler_epoch_velocity_km_s(row_a)
    vb_vec = _kepler_epoch_velocity_km_s(row_b)

    if va_vec is not None and vb_vec is not None:
        # Pair-specific relative velocity: |v_A - v_B|.
        return max(0.01, float(np.linalg.norm(va_vec - vb_vec))), "two-body epoch state"

    # Fallback when an orbital element is unavailable.
    va = _safe_float(row_a, "MEAN_MOTION")
    vb = _safe_float(row_b, "MEAN_MOTION")
    if np.isfinite(va) and np.isfinite(vb):
        va_km_s = (398600.4418 * (va * 2.0 * math.pi / 86400.0)) ** (1.0 / 3.0)
        vb_km_s = (398600.4418 * (vb * 2.0 * math.pi / 86400.0)) ** (1.0 / 3.0)
        return max(0.01, abs(va_km_s - vb_km_s)), "mean-motion fallback"

    return np.nan, "unavailable"


def _collision_energy(m1_kg, m2_kg, relative_velocity_km_s):
    v = float(relative_velocity_km_s) * 1000.0
    mu = (m1_kg * m2_kg) / max(m1_kg + m2_kg, 1e-12)
    return mu, 0.5 * mu * v * v


def _projectile_energy_per_target_mass_j_g(m_projectile_kg, v_km_s, target_mass_kg):
    return (
        0.5 * m_projectile_kg * (v_km_s * 1000.0) ** 2
        / max(target_mass_kg * 1000.0, 1e-12)
    )


def _ssbm_collision_mass(m_target_kg, m_projectile_kg, v_km_s, catastrophic):
    """Return SSBM collisional mass M in kg.

    For catastrophic collisions M is the sum of both parent masses.
    For non-catastrophic collisions, the SSBM collision power law uses
    projectile mass multiplied by impact velocity squared (km/s)^2.
    """
    if catastrophic:
        return max(m_target_kg + m_projectile_kg, 1e-9)
    return max(m_projectile_kg * v_km_s ** 2, 1e-9)


def _characteristic_length(dimensions_m):
    # NASA SSBM convention: average of the three maximum orthogonal
    # projected dimensions.
    return float(np.mean(np.asarray(dimensions_m, dtype=float)))


def _area_from_lc(lc_m):
    if lc_m < 0.00167:
        return 0.540424 * lc_m ** 2
    return 0.556945 * lc_m ** 2.0047077


def _ssbm_number_above_lc(collisional_mass_kg, lc_m, scale_factor=1.0):
    """Cumulative SSBM collision count N(>=Lc)."""
    lc = max(float(lc_m), 1e-6)
    n = (
        float(scale_factor)
        * 0.1
        * max(float(collisional_mass_kg), 1e-9) ** 0.75
        * lc ** (-1.71)
    )
    return max(0, int(math.floor(n)))


def _sample_lc_population(n, lc_parent_m, min_lc_m, seed):
    """Inverse-sample a finite population from the SSBM cumulative power law."""
    rng = np.random.default_rng(int(seed))
    n = max(int(n), 1)
    lo = max(float(min_lc_m), 1e-5)
    hi = max(float(lc_parent_m), lo * 1.001)

    # The cumulative law is proportional to Lc^-alpha.
    alpha = 1.71
    lo_pow = lo ** (-alpha)
    hi_pow = hi ** (-alpha)
    u = rng.random(n)
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
    rng = np.random.default_rng(int(seed))
    n = len(lc_m)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    area = np.array([_area_from_lc(x) for x in lc_m])
    area_mass = area / np.maximum(masses_kg, 1e-12)

    # NASA SSBM-style empirical log10(delta-V) relationship.
    chi = np.log10(np.maximum(area_mass, 1e-12))
    mu_log10_dv = 0.9 * chi + 2.90
    log10_dv = rng.normal(mu_log10_dv, 0.4)
    dv = np.clip(10.0 ** log10_dv, 0.001, max(0.5, float(vrel_km_s)))

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
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, name="Earth", showscale=False, opacity=0.35, hoverinfo="skip"
    ))

    offset = np.array([EARTH_RADIUS_KM + 700.0, 0.0, 0.0])
    points = offset + positions * 25.0
    sizes = np.clip(
        4.0 + 16.0 * np.sqrt(np.maximum(lc_m, 0.001) / 0.05), 4, 22
    )
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers", name="SSBM fragment population",
        marker=dict(size=sizes, opacity=0.75),
        customdata=np.column_stack([lc_m * 1000.0]),
        hovertemplate="Characteristic length: %{customdata[0]:.2f} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[offset[0] - 500, offset[0]], y=[0, 0], z=[0, 0],
        mode="lines+markers", name=f"{target_a} → collision",
        line=dict(width=6), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter3d(
        x=[offset[0] + 500, offset[0]], y=[0, 0], z=[0, 0],
        mode="lines+markers", name=f"{target_b} → collision",
        line=dict(width=6), marker=dict(size=5),
    ))
    fig.update_layout(
        height=700,
        title="3-D NASA SSBM-Based Fragment Cloud",
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


def render_collision_fragmentation_tab():
    st.header("💥 Collision & Fragmentation")
    st.caption(
        "NASA SSBM-based breakup estimate using pair-specific orbital-element kinematics "
        "plus explicit mass, geometry, material, structural, and impact assumptions."
    )

    st.warning(
        "Important: orbital/TLE data do not contain reliable spacecraft mass, CAD geometry, "
        "material layup, or structural design. Those engineering properties must be supplied "
        "from documented sources. The SSBM result is a statistical fragment population, not "
        "an exact prediction of every physical fragment."
    )

    objects = _load_objects()
    name_col = _name_column(objects)
    id_col = _id_column(objects)

    st.subheader("1. Select the colliding pair")

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

    row_a = _object_row(objects, a_name, name_col)
    row_b = _object_row(objects, b_name, name_col)
    pair_v, pair_v_source = _pair_relative_velocity_estimate(row_a, row_b)

    pair_id_a = _safe_float(row_a, id_col) if row_a is not None and id_col else np.nan
    pair_id_b = _safe_float(row_b, id_col) if row_b is not None and id_col else np.nan

    st.caption(
        f"Selected pair: {a_name} + {b_name} | "
        f"IDs: {pair_id_a if np.isfinite(pair_id_a) else 'n/a'} / "
        f"{pair_id_b if np.isfinite(pair_id_b) else 'n/a'}"
    )

    st.subheader("2. Parent mass and collision velocity")

    c1, c2, c3 = st.columns(3)
    with c1:
        m1 = st.number_input(
            f"{a_name} mass (kg)", min_value=0.01, value=500.0, step=10.0, key=f"frag_m1_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}"
        )
    with c2:
        m2 = st.number_input(
            f"{b_name} mass (kg)", min_value=0.01, value=500.0, step=10.0, key=f"frag_m2_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}"
        )
    with c3:
        default_v = float(pair_v) if np.isfinite(pair_v) else 10.0
        vrel = st.number_input(
            "Relative collision velocity (km/s)",
            min_value=0.01,
            value=round(default_v, 2),
            step=0.5,
            key=f"frag_vrel_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}",
            help="Auto-filled from the selected pair's orbital-element state vectors. Exact TCA relative velocity requires SGP4 state propagation.",
        )

    if np.isfinite(pair_v):
        st.info(
            f"Pair-specific relative velocity at the orbital-record epoch: {pair_v:.3f} km/s "
            f"({pair_v_source}). Exact TCA relative velocity still requires SGP4 propagation. "
            "Change the input manually if you have a validated TCA value."
        )

    st.subheader("3. Geometry, material, and structural design")

    c1, c2, c3 = st.columns(3)
    with c1:
        geometry = st.selectbox(
            "Parent geometry",
            ["Box / spacecraft bus", "Cylinder / rocket body", "Panel-dominated structure", "Other / equivalent body"],
            key=f"frag_geometry_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}",
        )
        dim_x = st.number_input("Maximum dimension X (m)", min_value=0.01, value=2.0, step=0.1, key=f"frag_dim_x_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}")
    with c2:
        dim_y = st.number_input("Maximum dimension Y (m)", min_value=0.01, value=2.0, step=0.1, key=f"frag_dim_y_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}")
        dim_z = st.number_input("Maximum dimension Z (m)", min_value=0.01, value=2.0, step=0.1, key=f"frag_dim_z_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}")
    with c3:
        material = st.selectbox(
            "Dominant material",
            ["Aluminum", "Steel", "Composite / plastic", "Mixed spacecraft materials"],
            key=f"frag_material_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}",
        )
        density = st.number_input(
            "Effective bulk density (kg/m³)",
            min_value=100.0,
            value=float(_material_density(material)),
            step=100.0,
            key=f"frag_density_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}",
        )

    lc_parent = _characteristic_length([dim_x, dim_y, dim_z])
    st.metric("Parent characteristic length Lc", f"{lc_parent:.3f} m")

    c1, c2, c3 = st.columns(3)
    with c1:
        structure = st.selectbox(
            "Structural configuration",
            ["Intact / integrated", "Bus + panels", "Tank / rocket-body dominated", "Previously damaged"],
            key=f"frag_structure_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}",
        )
    with c2:
        impact_angle = st.slider(
            "Impact angle relative to reference axis (deg)",
            0, 90, 45, 5, key=f"frag_angle_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}"
        )
    with c3:
        energy_partition = st.slider(
            "Breakup/ejecta mass fraction (%)",
            1, 100, 100, 1, key=f"frag_energy_partition_{int(pair_id_a) if np.isfinite(pair_id_a) else a_name}_{int(pair_id_b) if np.isfinite(pair_id_b) else b_name}",
            help="Research sensitivity parameter for how much of the SSBM collisional mass is represented in the modeled breakup cloud. It is not a measured material constant.",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        min_lc_mm = st.selectbox(
            "Minimum modeled fragment size (mm)",
            [1.0, 2.0, 5.0, 10.0, 20.0],
            index=0,
            key="frag_min_lc",
        )
    with c2:
        scale_factor = st.number_input(
            "SSBM empirical scale S",
            min_value=0.10,
            max_value=5.00,
            value=1.00,
            step=0.05,
            key="frag_ssbm_scale",
            help="SSBM rescaling factor. Keep at 1.0 for the standard model unless you have validation data supporting another value.",
        )
    with c3:
        seed = st.number_input("Simulation seed", min_value=0, value=42, step=1, key="frag_seed")

    structure_factor = {
        "Intact / integrated": 1.00,
        "Bus + panels": 0.95,
        "Tank / rocket-body dominated": 0.90,
        "Previously damaged": 0.75,
    }[structure]

    geometry_factor = {
        "Box / spacecraft bus": 1.00,
        "Cylinder / rocket body": 0.92,
        "Panel-dominated structure": 0.80,
        "Other / equivalent body": 1.00,
    }[geometry]

    # Determine the catastrophic regime using the SSBM 40 J/g criterion.
    # Use the larger parent as target and the smaller as projectile.
    target_mass = max(m1, m2)
    projectile_mass = min(m1, m2)
    mu, energy = _collision_energy(m1, m2, vrel)
    catastrophic_metric = _projectile_energy_per_target_mass_j_g(
        projectile_mass, vrel, target_mass
    )
    catastrophic = catastrophic_metric >= 40.0

    collisional_mass = _ssbm_collision_mass(
        target_mass, projectile_mass, vrel, catastrophic
    )

    # Structural/geometry factors are explicitly sensitivity modifiers, while
    # the SSBM collision mass remains visible so the result is auditable.
    modeled_mass = (
        collisional_mass
        * structure_factor
        * geometry_factor
        * (energy_partition / 100.0)
    )

    min_lc_m = float(min_lc_mm) / 1000.0
    n_est = _ssbm_number_above_lc(
        modeled_mass, min_lc_m, scale_factor=float(scale_factor)
    )
    # Never let a valid breakup collapse to an arbitrary two-fragment display.
    # Keep the full model population up to a UI-safe ceiling.
    n_display = int(np.clip(n_est, 10, 5000))

    st.subheader("4. SSBM collision regime and energetics")
    st.caption(
        f"Pair input used in this run: {a_name} + {b_name} | "
        f"relative velocity = {vrel:.3f} km/s | velocity source = {pair_v_source}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total parent mass", f"{m1 + m2:,.1f} kg")
    c2.metric("Collision energy", f"{energy / 1e9:,.2f} GJ")
    c3.metric("Eₚ / target mass", f"{catastrophic_metric:,.1f} J/g")
    c4.metric("Regime", "Catastrophic" if catastrophic else "Non-catastrophic")

    st.latex(r"E_c=\frac{1}{2}\mu v_{rel}^2,\qquad E_p=\frac{0.5m_pv^2}{m_t}")
    st.info(
        "SSBM defines a collision as catastrophic when impact kinetic energy per target "
        "mass reaches 40 J/g. For collisions, the cumulative fragment count follows "
        "N(≥Lc) = S·0.1·M^0.75·Lc^-1.71."
    )

    st.subheader("5. Fragment population")

    lc_values = _sample_lc_population(
        n_display,
        lc_parent,
        min_lc_m,
        int(seed),
    )

    area_values = np.array([_area_from_lc(x) for x in lc_values])

    # Material density affects mass assigned to a geometric fragment. Normalize
    # to the modeled breakup mass so the displayed population conserves mass.
    raw_masses = np.maximum(area_values * density, 1e-12)
    masses = raw_masses / raw_masses.sum() * max(modeled_mass, 1e-9)
    densities = np.full(n_display, density)

    positions, velocities, area_mass = _cloud_from_fragments(
        lc_values, masses, vrel, impact_angle, int(seed)
    )

    total_displayed_mass = float(masses.sum())
    st.write(
        f"**SSBM cumulative estimate:** {n_est:,} fragments ≥ {min_lc_mm:g} mm. "
        f"**Rendered:** {n_display:,} representative fragments. "
        "The renderer caps the plotted sample at 5,000 for browser performance; "
        "the cumulative SSBM count above is the model estimate."
    )

    metrics = st.columns(4)
    metrics[0].metric("SSBM collisional mass M", f"{collisional_mass:,.2f} kg")
    metrics[1].metric("Modeled breakup mass", f"{modeled_mass:,.2f} kg")
    metrics[2].metric("Displayed mass", f"{total_displayed_mass:,.2f} kg")
    metrics[3].metric(
        "Mass-conservation error",
        f"{abs(total_displayed_mass - modeled_mass):.2e} kg",
    )

    bins = [
        (">100 mm", lc_values >= 0.1),
        ("10–100 mm", (lc_values >= 0.01) & (lc_values < 0.1)),
        ("1–10 mm", (lc_values >= 0.001) & (lc_values < 0.01)),
        ("<1 mm", lc_values < 0.001),
    ]
    size_summary = pd.DataFrame({
        "Size class": [x[0] for x in bins],
        "Rendered fragments": [int(x[1].sum()) for x in bins],
    })
    st.dataframe(size_summary, width="stretch", hide_index=True)

    st.subheader("6. 3-D SSBM fragment cloud")
    st.plotly_chart(
        _fragment_scene(positions, lc_values, a_name, b_name),
        width="stretch",
    )

    st.subheader("7. Fragment properties and ORION-X priority")
    priority = _fragment_table(
        lc_values, masses, densities, velocities, area_mass
    )
    st.dataframe(priority.head(25), width="stretch", hide_index=True)

    top_score = (
        float(priority.iloc[0]["Capture priority"])
        if not priority.empty else 0.0
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Rendered fragments", f"{n_display:,}")
    c2.metric("Highest capture-priority score", f"{top_score:.3f}")
    c3.metric("Next ORION-X step", "Track → screen → capture")

    st.markdown(
        "**ORION-X workflow:** collision detection → pair-specific event inputs → "
        "SSBM breakup population → fragment size/A-M/ΔV → propagate/screen fragments → "
        "recalculate collision risk → prioritize high-risk debris → inspect → capture → tow/deorbit."
    )

    with st.expander("Model basis and limitations"):
        st.markdown(
            "- NASA SSBM is a semi-empirical statistical breakup model derived from observed breakups and ground impact tests; it predicts fragment distributions, not exact individual pieces.\n"
            "- SSBM characteristic length is based on the average of three maximum orthogonal dimensions.\n"
            "- The cumulative collision power law is evaluated at the selected minimum fragment size. "
            "A 1 mm lower bound can legitimately produce a very large population, so the UI reports "
            "the full cumulative estimate and only renders a capped representative sample.\n"
            "- Pair selection now changes the automatically estimated relative orbital speed when mean-motion data are available. Exact pair-specific TCA velocity still requires SGP4 state propagation.\n"
            "- Mass, geometry, material, and structural inputs are intentionally not fabricated from object names because TLEs do not contain those engineering properties.\n"
            "- For higher fidelity, future versions can sample directly from SOCIT/DebriSat fragment datasets rather than only the analytic SSBM distribution."
        )
