"""
ascent_trajectory.py

Synthetic launch-vehicle ascent trajectory generator.

This module generates physically grounded, modeled rocket ascent
trajectories for the rocket-launch dataset.

The trajectory is NOT recovered flight telemetry. It is a modeled
gravity-turn ascent based on the formulation described by:

Teofilatto, Carletta & Pontani (2022),
"Analytic Derivation of Ascent Trajectories and Performance of Launch
Vehicles," Applied Sciences, 12(11), 5685.

The generated trajectory can later be compared with SGP4-propagated
orbital debris positions to estimate:

    rocket-debris miss distance
    rocket-debris relative velocity
    Monte Carlo collision probability
    QAE collision probability

Input:
    data/rockets/rocket_launches.csv

Output:
    data/processed/rocket_trajectories.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ============================================================
# EARTH CONSTANTS
# ============================================================

MU_EARTH = 398600.4418       # km^3/s^2
R_EARTH = 6378.137           # km
OMEGA_EARTH = 7.2921150e-5   # rad/s


# ============================================================
# LAUNCH SITE COORDINATES
# ============================================================

LAUNCH_SITES = {
    "Cape Canaveral SLC-40, USA": (28.485, -80.577),
    "Cape Canaveral, USA": (28.485, -80.577),

    "Satish Dhawan Space Centre, India": (
        13.733,
        80.235,
    ),

    "Kourou, French Guiana": (
        5.236,
        -52.775,
    ),

    "Wallops Flight Facility, USA": (
        37.940,
        -75.466,
    ),

    "Rocket Lab LC-1, New Zealand": (
        -39.262,
        177.865,
    ),

    "Baikonur, Kazakhstan": (
        45.965,
        63.305,
    ),

    "Jiuquan, China": (
        40.958,
        100.291,
    ),
}


# ============================================================
# 1. LAUNCH AZIMUTH
# ============================================================

def launch_azimuth_deg(
    inclination_deg: float,
    launch_lat_deg: float,
) -> float:
    """
    Calculate the launch azimuth required to reach the target
    orbital inclination from the launch-site latitude.

    Formula:

        sin(psi) = cos(i) / cos(L)

    where:

        psi = launch azimuth
        i   = target inclination
        L   = launch-site latitude
    """

    L = np.radians(launch_lat_deg)
    i = np.radians(inclination_deg)

    if abs(np.cos(i)) > abs(np.cos(L)) + 1e-9:
        raise ValueError(
            f"Inclination {inclination_deg} deg is not directly "
            f"reachable from latitude {launch_lat_deg} deg."
        )

    sin_psi = np.cos(i) / np.cos(L)
    sin_psi = np.clip(sin_psi, -1.0, 1.0)

    return float(np.degrees(np.arcsin(sin_psi)))


# ============================================================
# 2. GRAVITY-TURN EQUATIONS
# ============================================================

def _gravity_turn_rhs(
    t,
    y,
    thrust_to_weight,
):
    """
    Gravity-turn equations.

    State:

        y[0] = V      relative velocity, km/s
        y[1] = gamma  flight-path angle, radians
        y[2] = h      altitude, km
        y[3] = s      downrange distance, km
    """

    V, gamma, h, s = y

    r = R_EARTH + h

    g_local = MU_EARTH / r**2

    # Prevent division by zero near liftoff.
    V_safe = max(abs(V), 1e-6)

    dVdt = (
        thrust_to_weight * g_local
        - g_local * np.sin(gamma)
    )

    dgammadt = (
        (V_safe * np.cos(gamma)) / r
        - (g_local / V_safe) * np.cos(gamma)
    )

    dhdt = V * np.sin(gamma)

    dsdt = (
        (R_EARTH / r)
        * V
        * np.cos(gamma)
    )

    return [
        dVdt,
        dgammadt,
        dhdt,
        dsdt,
    ]


# ============================================================
# 3. INTEGRATE ASCENT
# ============================================================

def _integrate_gravity_turn(
    thrust_to_weight,
    burn_time_s,
    gamma0_deg=89.5,
    n_points=400,
):
    """
    Integrate the powered gravity-turn ascent.
    """

    y0 = [
        1e-3,
        np.radians(gamma0_deg),
        0.0,
        0.0,
    ]

    t_eval = np.linspace(
        0,
        burn_time_s,
        n_points,
    )

    solution = solve_ivp(
        _gravity_turn_rhs,
        [0, burn_time_s],
        y0,
        args=(thrust_to_weight,),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(
            f"Gravity-turn integration failed: "
            f"{solution.message}"
        )

    return solution.t, solution.y


# ============================================================
# 4. SOLVE FOR THRUST-TO-WEIGHT
# ============================================================

def _solve_for_burnout_flight_path_angle(
    target_gamma_f_deg,
    thrust_to_weight,
    burn_time_s,
    target_h_km,
):
    """
    Find a thrust-to-weight ratio that produces approximately
    the requested target altitude at burnout.
    """

    def altitude_error(n):
        _, state = _integrate_gravity_turn(
            n,
            burn_time_s,
        )

        h_final = state[2, -1]

        return h_final - target_h_km

    try:
        n_solution = brentq(
            altitude_error,
            1.05,
            4.0,
            xtol=1e-3,
        )

    except ValueError:
        n_solution = thrust_to_weight

    return n_solution


# ============================================================
# 5. CIRCULAR ORBIT VELOCITY
# ============================================================

def _circular_orbital_velocity(
    h_km,
):
    """
    Circular orbital velocity at altitude h.
    """

    r = R_EARTH + h_km

    return np.sqrt(
        MU_EARTH / r
    )


# ============================================================
# 6. ASCENT STATE TO ECI
# ============================================================

def _ascent_state_to_eci(
    h_km,
    s_km,
    V_kms,
    gamma_rad,
    azimuth_deg,
    launch_lat_deg,
    launch_lon_deg,
    t_s,
    launch_gmst0_rad=0.0,
):
    """
    Convert modeled ascent state into approximate ECI
    position and velocity.
    """

    r = R_EARTH + h_km

    lat0 = np.radians(
        launch_lat_deg
    )

    lon0 = np.radians(
        launch_lon_deg
    )

    az = np.radians(
        azimuth_deg
    )

    delta = s_km / R_EARTH

    # Great-circle destination.
    lat = np.arcsin(
        np.sin(lat0) * np.cos(delta)
        +
        np.cos(lat0)
        * np.sin(delta)
        * np.cos(az)
    )

    lon = lon0 + np.arctan2(
        np.sin(az)
        * np.sin(delta)
        * np.cos(lat0),

        np.cos(delta)
        -
        np.sin(lat0)
        * np.sin(lat)
    )

    # Earth rotation.
    gmst = (
        launch_gmst0_rad
        +
        OMEGA_EARTH * t_s
    )

    lon_eci = lon + gmst

    # Position.
    x = (
        r
        * np.cos(lat)
        * np.cos(lon_eci)
    )

    y = (
        r
        * np.cos(lat)
        * np.sin(lon_eci)
    )

    z = (
        r
        * np.sin(lat)
    )

    # Velocity components.
    v_radial = (
        V_kms
        * np.sin(gamma_rad)
    )

    v_horizontal = (
        V_kms
        * np.cos(gamma_rad)
    )

    # Local ENU vectors.
    east = np.array([
        -np.sin(lon_eci),
        np.cos(lon_eci),
        0.0,
    ])

    north = np.array([
        -np.sin(lat)
        * np.cos(lon_eci),

        -np.sin(lat)
        * np.sin(lon_eci),

        np.cos(lat),
    ])

    up = np.array([
        np.cos(lat)
        * np.cos(lon_eci),

        np.cos(lat)
        * np.sin(lon_eci),

        np.sin(lat),
    ])

    v_east = (
        v_horizontal
        * np.sin(az)
    )

    v_north = (
        v_horizontal
        * np.cos(az)
    )

    v_vec = (
        v_east * east
        +
        v_north * north
        +
        v_radial * up
    )

    # Earth rotational velocity.
    omega_vec = np.array([
        0.0,
        0.0,
        OMEGA_EARTH,
    ])

    r_vec = np.array([
        x,
        y,
        z,
    ])

    v_rot = np.cross(
        omega_vec,
        r_vec,
    )

    v_eci = v_vec + v_rot

    return (
        x,
        y,
        z,
        v_eci[0],
        v_eci[1],
        v_eci[2],
    )


# ============================================================
# 7. GENERATE ONE ROCKET TRAJECTORY
# ============================================================

def generate_ascent_trajectory(
    mission_name: str,
    launch_lat_deg: float,
    launch_lon_deg: float,
    target_altitude_km: float,
    inclination_deg: float,
    launch_date=None,
    burn_time_s: float = 540.0,
    coast_and_insertion_s: float = 120.0,
    sample_interval_s: float = 5.0,
    launch_gmst0_rad: float = 0.0,
):
    """
    Generate one modeled rocket ascent trajectory.

    Output columns:

        mission_name
        launch_date
        time_from_launch_s
        x_km
        y_km
        z_km
        velocity_x_km_s
        velocity_y_km_s
        velocity_z_km_s
        altitude_km
        latitude_deg
        longitude_deg
    """

    azimuth = launch_azimuth_deg(
        inclination_deg,
        launch_lat_deg,
    )

    # Solve for approximate thrust-to-weight ratio.
    n_twr = _solve_for_burnout_flight_path_angle(
        target_gamma_f_deg=5.0,
        thrust_to_weight=1.4,
        burn_time_s=burn_time_s,
        target_h_km=target_altitude_km,
    )

    n_points = (
        int(
            burn_time_s
            // sample_interval_s
        )
        + 1
    )

    t_ascent, state = (
        _integrate_gravity_turn(
            n_twr,
            burn_time_s,
            n_points=n_points,
        )
    )

    V, gamma, h, s = state

    rows = []

    # --------------------------------------------------------
    # Powered ascent
    # --------------------------------------------------------

    for i, t in enumerate(t_ascent):

        (
            x,
            y,
            z,
            vx,
            vy,
            vz,
        ) = _ascent_state_to_eci(
            h[i],
            s[i],
            V[i],
            gamma[i],
            azimuth,
            launch_lat_deg,
            launch_lon_deg,
            t,
            launch_gmst0_rad,
        )

        radius = R_EARTH + h[i]

        lat = np.degrees(
            np.arcsin(
                np.clip(
                    z / radius,
                    -1.0,
                    1.0,
                )
            )
        )

        rows.append({
            "mission_name": mission_name,
            "launch_date": launch_date,
            "time_from_launch_s": round(
                float(t),
                1,
            ),
            "x_km": x,
            "y_km": y,
            "z_km": z,
            "velocity_x_km_s": vx,
            "velocity_y_km_s": vy,
            "velocity_z_km_s": vz,
            "altitude_km": h[i],
            "latitude_deg": lat,
            "longitude_deg": np.nan,
        })

    # --------------------------------------------------------
    # Coast / insertion phase
    # --------------------------------------------------------

    v_circ = _circular_orbital_velocity(
        target_altitude_km
    )

    n_coast = max(
        int(
            coast_and_insertion_s
            // sample_interval_s
        ),
        2,
    )

    h_last = h[-1]
    s_last = s[-1]
    V_last = V[-1]
    gamma_last = gamma[-1]

    for j in range(
        1,
        n_coast + 1,
    ):

        frac = (
            j / n_coast
        )

        t = (
            t_ascent[-1]
            +
            j * sample_interval_s
        )

        h_j = (
            h_last
            +
            frac
            * (
                target_altitude_km
                - h_last
            )
        )

        gamma_j = (
            gamma_last
            * (1 - frac)
        )

        V_j = (
            V_last
            +
            frac
            * (
                v_circ
                - V_last
            )
        )

        s_j = (
            s_last
            +
            V_j
            * sample_interval_s
            * np.cos(gamma_j)
            *
            (
                R_EARTH
                /
                (R_EARTH + h_j)
            )
        )

        s_last = s_j

        (
            x,
            y,
            z,
            vx,
            vy,
            vz,
        ) = _ascent_state_to_eci(
            h_j,
            s_j,
            V_j,
            gamma_j,
            azimuth,
            launch_lat_deg,
            launch_lon_deg,
            t,
            launch_gmst0_rad,
        )

        radius = R_EARTH + h_j

        lat = np.degrees(
            np.arcsin(
                np.clip(
                    z / radius,
                    -1.0,
                    1.0,
                )
            )
        )

        rows.append({
            "mission_name": mission_name,
            "launch_date": launch_date,
            "time_from_launch_s": round(
                float(t),
                1,
            ),
            "x_km": x,
            "y_km": y,
            "z_km": z,
            "velocity_x_km_s": vx,
            "velocity_y_km_s": vy,
            "velocity_z_km_s": vz,
            "altitude_km": h_j,
            "latitude_deg": lat,
            "longitude_deg": np.nan,
        })

    trajectory = pd.DataFrame(rows)

    trajectory["trajectory_source"] = (
        "modeled (gravity-turn analytic, "
        "Teofilatto et al. 2022)"
    )

    return trajectory


# ============================================================
# 8. GENERATE ALL ROCKET TRAJECTORIES
# ============================================================

def generate_all_trajectories(
    launch_table: pd.DataFrame,
):
    """
    Generate trajectories for every rocket in the CSV.
    """

    required_columns = [
        "launch_date",
        "mission_name",
        "rocket",
        "launch_site",
        "target_altitude_km",
        "inclination_deg",
    ]

    missing = [
        column
        for column in required_columns
        if column not in launch_table.columns
    ]

    if missing:
        raise ValueError(
            "Rocket CSV is missing required "
            f"columns: {missing}"
        )

    all_trajectories = []

    for _, row in launch_table.iterrows():

        mission = row["mission_name"]
        launch_site = row["launch_site"]

        if launch_site not in LAUNCH_SITES:

            print(
                "WARNING: no coordinates found for "
                f"launch site '{launch_site}' "
                f"(mission: {mission}) -- skipped."
            )

            continue

        launch_lat, launch_lon = (
            LAUNCH_SITES[launch_site]
        )

        try:

            trajectory = (
                generate_ascent_trajectory(
                    mission_name=mission,
                    launch_lat_deg=launch_lat,
                    launch_lon_deg=launch_lon,
                    target_altitude_km=float(
                        row["target_altitude_km"]
                    ),
                    inclination_deg=float(
                        row["inclination_deg"]
                    ),
                    launch_date=row[
                        "launch_date"
                    ],
                )
            )

            # Keep rocket information attached.
            trajectory["rocket"] = row[
                "rocket"
            ]

            trajectory["launch_site"] = (
                launch_site
            )

            trajectory["target_altitude_km"] = (
                float(
                    row["target_altitude_km"]
                )
            )

            trajectory["inclination_deg"] = (
                float(
                    row["inclination_deg"]
                )
            )

            all_trajectories.append(
                trajectory
            )

            print(
                f"[OK] {mission}: "
                f"{len(trajectory)} trajectory points"
            )

        except Exception as exc:

            print(
                f"[ERROR] {mission}: "
                f"{exc}"
            )

    if not all_trajectories:
        return pd.DataFrame()

    return pd.concat(
        all_trajectories,
        ignore_index=True,
    )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Project root.
    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    # Input CSV.
    ROCKET_FILE = (
        PROJECT_ROOT
        / "data"
        / "rockets"
        / "rocket_launches.csv"
    )

    # Output CSV.
    OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "rocket_trajectories.csv"
    )

    print("=" * 60)
    print("ROCKET ASCENT TRAJECTORY GENERATOR")
    print("=" * 60)

    print(
        f"\nLoading rocket data:\n"
        f"{ROCKET_FILE}"
    )

    if not ROCKET_FILE.exists():

        raise FileNotFoundError(
            "\nRocket CSV was not found.\n"
            f"Expected:\n{ROCKET_FILE}\n\n"
            "Create the file at:\n"
            "data/rockets/rocket_launches.csv"
        )

    # Read rocket dataset.
    launch_table = pd.read_csv(
        ROCKET_FILE
    )

    print(
        f"\nRocket launches loaded: "
        f"{len(launch_table)}"
    )

    # Generate trajectories.
    result = generate_all_trajectories(
        launch_table
    )

    if result.empty:

        raise RuntimeError(
            "No rocket trajectories were generated."
        )

    # Create output directory.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save.
    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nTrajectory points generated: "
        f"{len(result)}"
    )

    print(
        f"\nOutput file:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nTrajectories by mission:"
    )

    print(
        result.groupby(
            "mission_name"
        )
        .size()
        .to_string()
    )

    print(
        "\nAltitude range by mission:"
    )

    print(
        result.groupby(
            "mission_name"
        )["altitude_km"]
        .agg(
            [
                "min",
                "max",
            ]
        )
        .to_string()
    )

    print("\nDone.")


