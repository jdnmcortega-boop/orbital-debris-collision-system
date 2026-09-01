
"""
rocket_debris_collision.py

Rocket-Debris Conjunction and Collision-Probability Analysis.

Supports:

1. Normal modeled rocket trajectories.
2. Controlled synthetic collision-validation trajectories.

IMPORTANT:
The controlled trajectories are synthetic validation scenarios.
They are NOT real rocket launches and must NOT be presented as
observed collision events.

Controlled validation scenarios:

    CONTROL_SAFE
        > 68 km

    CONTROL_CONJUNCTION
        <= 68 km

    CONTROL_HIGH_RISK
        <= 1 km

    CONTROL_COLLISION
        <= physical collision radius

The system reports separately:

    - Minimum miss distance
    - Relative velocity
    - Conjunction status
    - Monte Carlo collision probability
    - Risk classification

Collision probability:

    Pc = collision_samples / total_Monte_Carlo_samples

The conjunction threshold and physical collision radius are
different quantities and must not be confused.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


# ============================================================
# CONFIGURATION
# ============================================================

# Conjunction screening threshold.
CONJUNCTION_THRESHOLD_KM = 68.0

# Physical collision radius.
#
# 0.01 km = 10 meters.
COLLISION_RADIUS_KM = 0.01

# Position uncertainty for normal analysis.
POSITION_UNCERTAINTY_KM = 0.10

# Position uncertainty for controlled validation.
#
# 0.001 km = 1 meter.
CONTROLLED_POSITION_UNCERTAINTY_KM = 0.001

# Number of Monte Carlo samples.
MONTE_CARLO_SAMPLES = 10000

# Reproducible random seed.
RANDOM_SEED = 42

# Hypothetical launch interval for normal analysis.
LAUNCH_INTERVAL_HOURS = 6

# Progress display interval.
PROGRESS_INTERVAL = 100


# ============================================================
# FILE PATHS
# ============================================================

NORMAL_ROCKET_TRAJECTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rocket_trajectories.csv"
)

CONTROLLED_ROCKET_TRAJECTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "controlled_rocket_trajectories.csv"
)

PROPAGATED_GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "propagated_grid.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rocket_debris_conjunctions.csv"
)


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

NORMAL_POSITION_COLUMNS = [
    "x_km",
    "y_km",
    "z_km",
]

NORMAL_VELOCITY_COLUMNS = [
    "velocity_x_km_s",
    "velocity_y_km_s",
    "velocity_z_km_s",
]

CONTROLLED_POSITION_COLUMNS = [
    "X_KM",
    "Y_KM",
    "Z_KM",
]

CONTROLLED_VELOCITY_COLUMNS = [
    "VX_KM_S",
    "VY_KM_S",
    "VZ_KM_S",
]

OBJECT_POSITION_COLUMNS = [
    "X_KM",
    "Y_KM",
    "Z_KM",
]

OBJECT_VELOCITY_COLUMNS = [
    "VX_KM_S",
    "VY_KM_S",
    "VZ_KM_S",
]


# ============================================================
# LOAD ROCKET TRAJECTORIES
# ============================================================

def load_rocket_trajectories():
    """
    Load either the controlled validation dataset or the
    normal modeled rocket trajectory dataset.

    Controlled data is preferred when available.
    """

    # --------------------------------------------------------
    # CONTROLLED VALIDATION DATASET
    # --------------------------------------------------------

    if CONTROLLED_ROCKET_TRAJECTORY_FILE.exists():

        print("\nLoading rocket trajectories:")
        print(CONTROLLED_ROCKET_TRAJECTORY_FILE)

        df = pd.read_csv(
            CONTROLLED_ROCKET_TRAJECTORY_FILE
        )

        print("\nRocket trajectory columns:")
        print(list(df.columns))

        required = [
            "TIME",
            "X_KM",
            "Y_KM",
            "Z_KM",
            "VX_KM_S",
            "VY_KM_S",
            "VZ_KM_S",
            "ROCKET",
            "SCENARIO_TYPE",
            "REFERENCE_OBJECT",
            "CONTROLLED_OFFSET_KM",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Controlled rocket trajectory CSV is missing "
                "columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing
                )
            )

        df["TIME"] = pd.to_datetime(
            df["TIME"],
            utc=True,
            errors="coerce",
        )

        for column in (
            CONTROLLED_POSITION_COLUMNS
            + CONTROLLED_VELOCITY_COLUMNS
            + ["CONTROLLED_OFFSET_KM"]
        ):
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(
            subset=[
                "TIME",
                *CONTROLLED_POSITION_COLUMNS,
                *CONTROLLED_VELOCITY_COLUMNS,
            ]
        )

        df = df.sort_values(
            [
                "SCENARIO_TYPE",
                "TIME",
            ]
        ).reset_index(drop=True)

        print(
            "\nDetected controlled collision validation dataset."
        )

        print(
            f"Controlled trajectory points: {len(df):,}"
        )

        print(
            f"Validation scenarios: "
            f"{df['SCENARIO_TYPE'].nunique()}"
        )

        print("\nValidation scenarios:")

        for scenario, group in df.groupby(
            "SCENARIO_TYPE"
        ):
            offset = float(
                group[
                    "CONTROLLED_OFFSET_KM"
                ].iloc[0]
            )

            print(
                f"  {scenario}: "
                f"{offset:g} km"
            )

        return df, True

    # --------------------------------------------------------
    # NORMAL ROCKET TRAJECTORY DATA
    # --------------------------------------------------------

    print("\nLoading rocket trajectories:")
    print(NORMAL_ROCKET_TRAJECTORY_FILE)

    if not NORMAL_ROCKET_TRAJECTORY_FILE.exists():

        raise FileNotFoundError(
            "Neither controlled nor normal rocket trajectory "
            "data was found.\n\n"
            f"Expected controlled file:\n"
            f"{CONTROLLED_ROCKET_TRAJECTORY_FILE}\n\n"
            f"Expected normal file:\n"
            f"{NORMAL_ROCKET_TRAJECTORY_FILE}"
        )

    df = pd.read_csv(
        NORMAL_ROCKET_TRAJECTORY_FILE
    )

    print("\nRocket trajectory columns:")
    print(list(df.columns))

    required = [
        "mission_name",
        "time_from_launch_s",
        "x_km",
        "y_km",
        "z_km",
        "velocity_x_km_s",
        "velocity_y_km_s",
        "velocity_z_km_s",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Normal rocket trajectory CSV is missing "
            "columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    df["time_from_launch_s"] = pd.to_numeric(
        df["time_from_launch_s"],
        errors="coerce",
    )

    for column in (
        NORMAL_POSITION_COLUMNS
        + NORMAL_VELOCITY_COLUMNS
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "mission_name",
            "time_from_launch_s",
            *NORMAL_POSITION_COLUMNS,
            *NORMAL_VELOCITY_COLUMNS,
        ]
    )

    df = df.sort_values(
        [
            "mission_name",
            "time_from_launch_s",
        ]
    ).reset_index(drop=True)

    print(
        f"Rocket trajectory points: {len(df):,}"
    )

    print(
        f"Rocket missions: "
        f"{df['mission_name'].nunique()}"
    )

    return df, False


# ============================================================
# LOAD PROPAGATED ORBITAL DATA
# ============================================================

def load_propagated_grid():

    print("\nLoading propagated orbital data:")
    print(PROPAGATED_GRID_FILE)

    if not PROPAGATED_GRID_FILE.exists():

        raise FileNotFoundError(
            f"Propagated grid not found:\n"
            f"{PROPAGATED_GRID_FILE}"
        )

    df = pd.read_csv(
        PROPAGATED_GRID_FILE
    )

    print("\nPropagated grid columns:")
    print(list(df.columns))

    required = [
        "OBJECT_NAME",
        "OBJECT_ID",
        "NORAD_CAT_ID",
        "TIME",
        "X_KM",
        "Y_KM",
        "Z_KM",
        "VX_KM_S",
        "VY_KM_S",
        "VZ_KM_S",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Propagated grid is missing columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    df["TIME"] = pd.to_datetime(
        df["TIME"],
        utc=True,
        errors="coerce",
    )

    for column in (
        OBJECT_POSITION_COLUMNS
        + OBJECT_VELOCITY_COLUMNS
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "TIME",
            *OBJECT_POSITION_COLUMNS,
            *OBJECT_VELOCITY_COLUMNS,
        ]
    )

    df = df.sort_values(
        [
            "NORAD_CAT_ID",
            "TIME",
        ]
    ).reset_index(drop=True)

    print(
        f"Valid propagated states: "
        f"{len(df):,}"
    )

    print(
        f"Tracked orbital objects: "
        f"{df['NORAD_CAT_ID'].nunique():,}"
    )

    return df


# ============================================================
# INTERPOLATE OBJECT STATES
# ============================================================

def interpolate_object_states(
    object_df,
    requested_times,
):
    """
    Interpolate orbital-object states at requested timestamps.

    Linear interpolation is used between propagated states.
    """

    object_df = (
        object_df
        .sort_values("TIME")
        .copy()
    )

    source_seconds = (
        object_df["TIME"]
        .astype("int64")
        .to_numpy(dtype=float)
        / 1e9
    )

    requested_index = (
        pd.DatetimeIndex(
            requested_times
        )
    )

    requested_seconds = (
        requested_index
        .astype("int64")
        .to_numpy(dtype=float)
        / 1e9
    )

    result = {}

    for column in (
        OBJECT_POSITION_COLUMNS
        + OBJECT_VELOCITY_COLUMNS
    ):

        values = pd.to_numeric(
            object_df[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(source_seconds)
            & np.isfinite(values)
        )

        if valid.sum() < 2:

            result[column] = np.full(
                len(requested_times),
                np.nan,
            )

            continue

        valid_times = source_seconds[
            valid
        ]

        valid_values = values[
            valid
        ]

        interpolated = np.interp(
            requested_seconds,
            valid_times,
            valid_values,
        )

        outside = (
            (requested_seconds < valid_times.min())
            |
            (requested_seconds > valid_times.max())
        )

        interpolated[
            outside
        ] = np.nan

        result[column] = interpolated

    return pd.DataFrame(
        result,
        index=requested_index,
    )


# ============================================================
# NORMAL ROCKET CLOSEST APPROACH
# ============================================================

def find_normal_closest_approach(
    rocket_df,
    object_df,
    launch_time,
):
    """
    Find the closest approach between a normal modeled rocket
    trajectory and one propagated orbital object.
    """

    rocket = rocket_df.copy()

    rocket["ABSOLUTE_TIME"] = (
        pd.Timestamp(launch_time)
        + pd.to_timedelta(
            rocket[
                "time_from_launch_s"
            ],
            unit="s",
        )
    )

    rocket_times = pd.DatetimeIndex(
        rocket["ABSOLUTE_TIME"]
    )

    object_interp = interpolate_object_states(
        object_df,
        rocket_times,
    )

    valid = (
        np.isfinite(
            object_interp[
                OBJECT_POSITION_COLUMNS
            ].to_numpy()
        ).all(axis=1)
        &
        np.isfinite(
            object_interp[
                OBJECT_VELOCITY_COLUMNS
            ].to_numpy()
        ).all(axis=1)
    )

    if not valid.any():
        return None

    rocket_positions = (
        rocket.loc[
            valid,
            NORMAL_POSITION_COLUMNS,
        ]
        .to_numpy(dtype=float)
    )

    rocket_velocities = (
        rocket.loc[
            valid,
            NORMAL_VELOCITY_COLUMNS,
        ]
        .to_numpy(dtype=float)
    )

    object_positions = (
        object_interp.loc[
            valid,
            OBJECT_POSITION_COLUMNS,
        ]
        .to_numpy(dtype=float)
    )

    object_velocities = (
        object_interp.loc[
            valid,
            OBJECT_VELOCITY_COLUMNS,
        ]
        .to_numpy(dtype=float)
    )

    relative_positions = (
        rocket_positions
        - object_positions
    )

    distances = np.linalg.norm(
        relative_positions,
        axis=1,
    )

    minimum_index = int(
        np.argmin(distances)
    )

    miss_distance = float(
        distances[minimum_index]
    )

    relative_velocity_vector = (
        rocket_velocities[minimum_index]
        - object_velocities[minimum_index]
    )

    relative_velocity = float(
        np.linalg.norm(
            relative_velocity_vector
        )
    )

    relative_position = (
        relative_positions[
            minimum_index
        ]
    )

    if miss_distance > 0:

        line_of_sight = (
            relative_position
            / miss_distance
        )

        closing_speed = float(
            -np.dot(
                relative_velocity_vector,
                line_of_sight,
            )
        )

    else:

        closing_speed = relative_velocity

    valid_indices = np.flatnonzero(
        valid
    )

    original_index = (
        valid_indices[
            minimum_index
        ]
    )

    closest_time = pd.Timestamp(
        rocket.iloc[
            original_index
        ]["ABSOLUTE_TIME"]
    )

    return {
        "TCA": closest_time,
        "MISS_DISTANCE_KM": miss_distance,
        "RELATIVE_VELOCITY_KM_S": relative_velocity,
        "CLOSING_SPEED_KM_S": closing_speed,
    }


# ============================================================
# CONTROLLED CLOSEST APPROACH
# ============================================================

def find_controlled_closest_approach(
    rocket_df,
    object_df,
):
    """
    Find closest approach for a controlled validation
    scenario.

    Controlled trajectory timestamps are already absolute,
    so no launch-time shift is applied.
    """

    rocket_times = pd.DatetimeIndex(
        rocket_df["TIME"]
    )

    object_interp = interpolate_object_states(
        object_df,
        rocket_times,
    )

    valid = (
        np.isfinite(
            object_interp[
                OBJECT_POSITION_COLUMNS
            ].to_numpy()
        ).all(axis=1)
        &
        np.isfinite(
            object_interp[
                OBJECT_VELOCITY_COLUMNS
            ].to_numpy()
        ).all(axis=1)
    )

    if not valid.any():
        return None

    rocket_positions = (
        rocket_df.loc[
            valid,
            CONTROLLED_POSITION_COLUMNS,
        ]
        .to_numpy(dtype=float)
    )

    rocket_velocities = (
        rocket_df.loc[
            valid,
            CONTROLLED_VELOCITY_COLUMNS,
        ]
        .to_numpy(dtype=float)
    )

    object_positions = (
        object_interp.loc[
            valid,
            OBJECT_POSITION_COLUMNS,
        ]
        .to_numpy(dtype=float)
    )

    object_velocities = (
        object_interp.loc[
            valid,
            OBJECT_VELOCITY_COLUMNS,
        ]
        .to_numpy(dtype=float)
    )

    relative_positions = (
        rocket_positions
        - object_positions
    )

    distances = np.linalg.norm(
        relative_positions,
        axis=1,
    )

    minimum_index = int(
        np.argmin(distances)
    )

    miss_distance = float(
        distances[minimum_index]
    )

    relative_velocity_vector = (
        rocket_velocities[minimum_index]
        - object_velocities[minimum_index]
    )

    relative_velocity = float(
        np.linalg.norm(
            relative_velocity_vector
        )
    )

    relative_position = (
        relative_positions[
            minimum_index
        ]
    )

    if miss_distance > 0:

        line_of_sight = (
            relative_position
            / miss_distance
        )

        closing_speed = float(
            -np.dot(
                relative_velocity_vector,
                line_of_sight,
            )
        )

    else:

        closing_speed = relative_velocity

    valid_indices = np.flatnonzero(
        valid
    )

    original_index = (
        valid_indices[
            minimum_index
        ]
    )

    closest_time = pd.Timestamp(
        rocket_df.iloc[
            original_index
        ]["TIME"]
    )

    return {
        "TCA": closest_time,
        "MISS_DISTANCE_KM": miss_distance,
        "RELATIVE_VELOCITY_KM_S": relative_velocity,
        "CLOSING_SPEED_KM_S": closing_speed,
    }


# ============================================================
# MONTE CARLO COLLISION PROBABILITY
# ============================================================

def monte_carlo_collision_probability(
    miss_distance_km,
    collision_radius_km,
    uncertainty_km,
    samples,
    rng,
):
    """
    Estimate collision probability using Monte Carlo sampling.

    The nominal relative-position vector is placed along the
    x-axis because the uncertainty distribution is isotropic.

    Collision condition:

        ||nominal_relative_position + uncertainty|| <= radius

    Collision probability:

        Pc = collision_count / total_samples
    """

    miss_distance_km = float(
        miss_distance_km
    )

    if not np.isfinite(
        miss_distance_km
    ):
        return 0.0, 0

    if samples <= 0:
        raise ValueError(
            "Monte Carlo sample count must be positive."
        )

    if uncertainty_km <= 0:
        raise ValueError(
            "Position uncertainty must be positive."
        )

    # --------------------------------------------------------
    # FAST DETERMINISTIC REJECTION
    # --------------------------------------------------------

    # If the nominal miss distance is sufficiently far away
    # from the collision radius, the probability is effectively
    # zero for this prototype.
    if (
        miss_distance_km
        > collision_radius_km
        + 8.0 * uncertainty_km
    ):
        return 0.0, 0

    nominal = np.array(
        [
            miss_distance_km,
            0.0,
            0.0,
        ]
    )

    # --------------------------------------------------------
    # MONTE CARLO SAMPLING
    # --------------------------------------------------------

    noise = rng.normal(
        loc=0.0,
        scale=uncertainty_km,
        size=(
            samples,
            3,
        ),
    )

    simulated_positions = (
        nominal
        + noise
    )

    simulated_distances = np.linalg.norm(
        simulated_positions,
        axis=1,
    )

    collision_count = int(
        np.count_nonzero(
            simulated_distances
            <= collision_radius_km
        )
    )

    probability = (
        collision_count
        / samples
    )

    return (
        float(probability),
        collision_count,
    )


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(
    probability,
    miss_distance_km,
):
    """
    Classify the event using both collision probability and
    conjunction distance.

    CONTROLLED VALIDATION LEVELS
    ----------------------------

    CRITICAL:
        Physical collision region.

    HIGH:
        Very close conjunction:
        <= 1 km

    MEDIUM:
        Conjunction within screening threshold:
        <= 68 km

    LOW:
        Outside conjunction threshold:
        > 68 km

    Collision probability is retained as a separate quantitative
    measure.

    For a physical collision or non-zero Monte Carlo collision
    probability, CRITICAL takes precedence.

    These categories are research-prototype classifications and
    are NOT official operational aerospace thresholds.
    """

    probability = float(
        probability
    )

    miss_distance_km = float(
        miss_distance_km
    )

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if (
        miss_distance_km
        <= COLLISION_RADIUS_KM
    ):
        return "CRITICAL"

    if probability > 0:
        return "CRITICAL"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if miss_distance_km <= 1.0:
        return "HIGH"

    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if (
        miss_distance_km
        <= CONJUNCTION_THRESHOLD_KM
    ):
        return "MEDIUM"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "LOW"


# ============================================================
# ANALYZE CONTROLLED SCENARIO
# ============================================================

def analyze_controlled_scenario(
    scenario_name,
    rocket_df,
    object_df,
    rng,
):
    """
    Analyze one controlled validation scenario.

    The controlled rocket is matched against its reference
    propagated orbital object.
    """

    reference_object = str(
        rocket_df[
            "REFERENCE_OBJECT"
        ].iloc[0]
    )

    controlled_offset = float(
        rocket_df[
            "CONTROLLED_OFFSET_KM"
        ].iloc[0]
    )

    closest = find_controlled_closest_approach(
        rocket_df,
        object_df,
    )

    if closest is None:
        return None

    miss_distance = closest[
        "MISS_DISTANCE_KM"
    ]

    probability, collision_count = (
        monte_carlo_collision_probability(
            miss_distance_km=miss_distance,
            collision_radius_km=(
                COLLISION_RADIUS_KM
            ),
            uncertainty_km=(
                CONTROLLED_POSITION_UNCERTAINTY_KM
            ),
            samples=MONTE_CARLO_SAMPLES,
            rng=rng,
        )
    )

    risk_level = classify_risk(
        probability,
        miss_distance,
    )

    conjunction = (
        miss_distance
        <= CONJUNCTION_THRESHOLD_KM
    )

    return {
        "ROCKET": scenario_name,
        "OBJECT_NAME": reference_object,
        "SCENARIO_TYPE": scenario_name,
        "CONTROLLED_OFFSET_KM": controlled_offset,
        "LAUNCH_TIME": rocket_df[
            "TIME"
        ].min(),
        "TCA": closest["TCA"],
        "MISS_DISTANCE_KM": miss_distance,
        "RELATIVE_VELOCITY_KM_S": closest[
            "RELATIVE_VELOCITY_KM_S"
        ],
        "CLOSING_SPEED_KM_S": closest[
            "CLOSING_SPEED_KM_S"
        ],
        "CONJUNCTION": bool(
            conjunction
        ),
        "CONJUNCTION_THRESHOLD_KM": (
            CONJUNCTION_THRESHOLD_KM
        ),
        "COLLISION_RADIUS_KM": (
            COLLISION_RADIUS_KM
        ),
        "POSITION_UNCERTAINTY_KM": (
            CONTROLLED_POSITION_UNCERTAINTY_KM
        ),
        "MONTE_CARLO_SAMPLES": (
            MONTE_CARLO_SAMPLES
        ),
        "COLLISION_COUNT": collision_count,
        "COLLISION_PROBABILITY": probability,
        "RISK_LEVEL": risk_level,
        "VALIDATION_DATASET": True,
    }


# ============================================================
# ANALYZE NORMAL SCENARIO
# ============================================================

def analyze_normal_scenario(
    rocket_name,
    rocket_df,
    object_name,
    object_df,
    launch_time,
    rng,
):
    """
    Analyze one normal rocket/debris scenario.
    """

    closest = find_normal_closest_approach(
        rocket_df,
        object_df,
        launch_time,
    )

    if closest is None:
        return None

    miss_distance = closest[
        "MISS_DISTANCE_KM"
    ]

    conjunction = (
        miss_distance
        <= CONJUNCTION_THRESHOLD_KM
    )

    probability, collision_count = (
        monte_carlo_collision_probability(
            miss_distance_km=miss_distance,
            collision_radius_km=(
                COLLISION_RADIUS_KM
            ),
            uncertainty_km=(
                POSITION_UNCERTAINTY_KM
            ),
            samples=MONTE_CARLO_SAMPLES,
            rng=rng,
        )
    )

    risk_level = classify_risk(
        probability,
        miss_distance,
    )

    return {
        "ROCKET": rocket_name,
        "OBJECT_NAME": object_name,
        "SCENARIO_TYPE": "NORMAL",
        "CONTROLLED_OFFSET_KM": np.nan,
        "LAUNCH_TIME": pd.Timestamp(
            launch_time
        ),
        "TCA": closest["TCA"],
        "MISS_DISTANCE_KM": miss_distance,
        "RELATIVE_VELOCITY_KM_S": closest[
            "RELATIVE_VELOCITY_KM_S"
        ],
        "CLOSING_SPEED_KM_S": closest[
            "CLOSING_SPEED_KM_S"
        ],
        "CONJUNCTION": bool(
            conjunction
        ),
        "CONJUNCTION_THRESHOLD_KM": (
            CONJUNCTION_THRESHOLD_KM
        ),
        "COLLISION_RADIUS_KM": (
            COLLISION_RADIUS_KM
        ),
        "POSITION_UNCERTAINTY_KM": (
            POSITION_UNCERTAINTY_KM
        ),
        "MONTE_CARLO_SAMPLES": (
            MONTE_CARLO_SAMPLES
        ),
        "COLLISION_COUNT": collision_count,
        "COLLISION_PROBABILITY": probability,
        "RISK_LEVEL": risk_level,
        "VALIDATION_DATASET": False,
    }


# ============================================================
# CONTROLLED ANALYSIS
# ============================================================

def run_controlled_analysis(
    rocket_data,
    propagated,
):
    """
    Run analysis for all controlled validation cases.
    """

    print()
    print("=" * 60)
    print("CONTROLLED COLLISION VALIDATION")
    print("=" * 60)

    object_groups = {
        str(name): group.copy()
        for name, group
        in propagated.groupby(
            "OBJECT_NAME"
        )
    }

    results = []

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    scenarios = [
        scenario
        for scenario
        in rocket_data[
            "SCENARIO_TYPE"
        ]
        .dropna()
        .unique()
    ]

    # --------------------------------------------------------
    # Analyze each scenario.
    # --------------------------------------------------------

    for scenario_name in scenarios:

        scenario_df = rocket_data[
            rocket_data[
                "SCENARIO_TYPE"
            ]
            == scenario_name
        ].copy()

        reference_object = str(
            scenario_df[
                "REFERENCE_OBJECT"
            ].iloc[0]
        )

        if (
            reference_object
            not in object_groups
        ):

            print(
                f"\n[WARNING] Reference object not found: "
                f"{reference_object}"
            )

            continue

        object_df = object_groups[
            reference_object
        ]

        print()
        print(
            f"Analyzing {scenario_name}..."
        )

        result = analyze_controlled_scenario(
            scenario_name=scenario_name,
            rocket_df=scenario_df,
            object_df=object_df,
            rng=rng,
        )

        if result is None:

            print(
                "  [WARNING] No valid closest approach."
            )

            continue

        results.append(result)

        print(
            f"  Controlled offset: "
            f"{result['CONTROLLED_OFFSET_KM']:.6f} km"
        )

        print(
            f"  Minimum separation: "
            f"{result['MISS_DISTANCE_KM']:.6f} km"
        )

        print(
            f"  Relative velocity: "
            f"{result['RELATIVE_VELOCITY_KM_S']:.6f} km/s"
        )

        print(
            f"  Conjunction: "
            f"{result['CONJUNCTION']}"
        )

        print(
            f"  Collision samples: "
            f"{result['COLLISION_COUNT']}/"
            f"{result['MONTE_CARLO_SAMPLES']}"
        )

        print(
            f"  Collision probability: "
            f"{result['COLLISION_PROBABILITY']:.8f}"
        )

        print(
            f"  Risk level: "
            f"{result['RISK_LEVEL']}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# NORMAL ANALYSIS
# ============================================================

def run_normal_analysis(
    rocket_data,
    propagated,
):
    """
    Run the normal rocket/debris analysis.
    """

    observation_start = propagated[
        "TIME"
    ].min()

    observation_end = propagated[
        "TIME"
    ].max()

    print("\nObservation period:")
    print(
        f"  Start: {observation_start}"
    )
    print(
        f"  End:   {observation_end}"
    )

    launch_times = pd.date_range(
        start=observation_start,
        end=observation_end,
        freq=f"{LAUNCH_INTERVAL_HOURS}h",
    )

    print("\nHypothetical launch scenarios:")
    print(
        f"  Interval: every "
        f"{LAUNCH_INTERVAL_HOURS} hours"
    )

    print(
        f"  Scenarios: "
        f"{len(launch_times)}"
    )

    rocket_groups = {
        name: group.copy()
        for name, group
        in rocket_data.groupby(
            "mission_name"
        )
    }

    object_groups = {
        int(norad_id): group.copy()
        for norad_id, group
        in propagated.groupby(
            "NORAD_CAT_ID"
        )
    }

    object_names = {
        norad_id: str(
            group[
                "OBJECT_NAME"
            ].iloc[0]
        )
        for norad_id, group
        in object_groups.items()
    }

    total_scenarios = (
        len(rocket_groups)
        * len(object_groups)
        * len(launch_times)
    )

    print(
        f"\nTotal analysis scenarios: "
        f"{total_scenarios}"
    )

    print("\nStarting analysis...")

    results = []

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    counter = 0

    for rocket_name, rocket_df in (
        rocket_groups.items()
    ):

        for launch_time in launch_times:

            for norad_id, object_df in (
                object_groups.items()
            ):

                counter += 1

                result = analyze_normal_scenario(
                    rocket_name=rocket_name,
                    rocket_df=rocket_df,
                    object_name=object_names[
                        norad_id
                    ],
                    object_df=object_df,
                    launch_time=launch_time,
                    rng=rng,
                )

                if result is not None:

                    result[
                        "NORAD_CAT_ID"
                    ] = norad_id

                    results.append(
                        result
                    )

                if (
                    counter % PROGRESS_INTERVAL == 0
                    or counter == total_scenarios
                ):

                    print(
                        f"Analyzing "
                        f"{counter}/"
                        f"{total_scenarios}: "
                        f"{rocket_name} | "
                        f"{launch_time}"
                    )

    return pd.DataFrame(
        results
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results_df,
    controlled=False,
):
    """
    Print analysis results and validation summary.
    """

    if results_df.empty:

        print(
            "\nNo valid rocket/object comparisons."
        )

        return

    results_df = (
        results_df
        .sort_values(
            [
                "MISS_DISTANCE_KM",
                "COLLISION_PROBABILITY",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 60)

    if controlled:
        print(
            "CONTROLLED VALIDATION RESULT"
        )
    else:
        print("RESULT")

    print("=" * 60)

    total = len(
        results_df
    )

    conjunction_count = int(
        results_df[
            "CONJUNCTION"
        ].sum()
    )

    collision_events = int(
        (
            results_df[
                "COLLISION_COUNT"
            ]
            > 0
        ).sum()
    )

    nonzero_probability = int(
        (
            results_df[
                "COLLISION_PROBABILITY"
            ]
            > 0
        ).sum()
    )

    print(
        f"\nTotal scenarios: {total}"
    )

    print(
        f"Within "
        f"{CONJUNCTION_THRESHOLD_KM:.1f} km: "
        f"{conjunction_count}"
    )

    print(
        f"Outside threshold: "
        f"{total - conjunction_count}"
    )

    print(
        f"Non-zero probability events: "
        f"{nonzero_probability}"
    )

    print(
        f"Events with Monte Carlo collision samples: "
        f"{collision_events}"
    )

    # --------------------------------------------------------
    # Detailed results
    # --------------------------------------------------------

    display_columns = [
        "ROCKET",
        "OBJECT_NAME",
        "SCENARIO_TYPE",
        "CONTROLLED_OFFSET_KM",
        "MISS_DISTANCE_KM",
        "RELATIVE_VELOCITY_KM_S",
        "CONJUNCTION",
        "COLLISION_PROBABILITY",
        "COLLISION_COUNT",
        "RISK_LEVEL",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results_df.columns
    ]

    print(
        "\nScenario results:"
    )

    print(
        results_df[
            available_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Risk counts
    # --------------------------------------------------------

    print(
        "\nRisk counts:"
    )

    print(
        results_df[
            "RISK_LEVEL"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Collision probability summary
    # --------------------------------------------------------

    print(
        "\nCollision probability summary:"
    )

    print(
        results_df[
            "COLLISION_PROBABILITY"
        ]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # Non-zero events
    # --------------------------------------------------------

    nonzero = results_df[
        results_df[
            "COLLISION_PROBABILITY"
        ]
        > 0
    ]

    print(
        "\nNon-zero collision probability events:"
    )

    if nonzero.empty:

        print("None")

    else:

        print(
            nonzero[
                available_columns
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Closest event
    # --------------------------------------------------------

    closest = results_df.iloc[0]

    print(
        "\nClosest overall approach:"
    )

    print(
        f"  Rocket: "
        f"{closest['ROCKET']}"
    )

    print(
        f"  Object: "
        f"{closest['OBJECT_NAME']}"
    )

    print(
        f"  Scenario: "
        f"{closest['SCENARIO_TYPE']}"
    )

    print(
        f"  TCA: "
        f"{closest['TCA']}"
    )

    print(
        f"  Miss distance: "
        f"{closest['MISS_DISTANCE_KM']:.6f} km"
    )

    print(
        f"  Relative velocity: "
        f"{closest['RELATIVE_VELOCITY_KM_S']:.6f} km/s"
    )

    print(
        f"  Collision probability: "
        f"{closest['COLLISION_PROBABILITY']:.8f}"
    )

    print(
        f"  Collision samples: "
        f"{closest['COLLISION_COUNT']}/"
        f"{closest['MONTE_CARLO_SAMPLES']}"
    )

    print(
        f"  Risk level: "
        f"{closest['RISK_LEVEL']}"
    )

    print(
        f"\nConjunction threshold: "
        f"{CONJUNCTION_THRESHOLD_KM} km"
    )

    print(
        f"Physical collision radius: "
        f"{COLLISION_RADIUS_KM} km "
        f"({COLLISION_RADIUS_KM * 1000:.1f} m)"
    )

    if controlled:

        print(
            f"Validation uncertainty: "
            f"{CONTROLLED_POSITION_UNCERTAINTY_KM} km "
            f"({CONTROLLED_POSITION_UNCERTAINTY_KM * 1000:.1f} m)"
        )

    else:

        print(
            f"Position uncertainty: "
            f"{POSITION_UNCERTAINTY_KM} km "
            f"({POSITION_UNCERTAINTY_KM * 1000:.1f} m)"
        )

    print(
        f"Monte Carlo samples: "
        f"{MONTE_CARLO_SAMPLES}"
    )

    # --------------------------------------------------------
    # Controlled validation expectation
    # --------------------------------------------------------

    if controlled:

        print()
        print(
            "Expected controlled validation classification:"
        )

        print(
            "  CONTROL_SAFE       -> LOW"
        )

        print(
            "  CONTROL_CONJUNCTION -> MEDIUM"
        )

        print(
            "  CONTROL_HIGH_RISK  -> HIGH"
        )

        print(
            "  CONTROL_COLLISION  -> CRITICAL"
        )

        # Check actual classifications.
        expected = {
            "CONTROL_SAFE": "LOW",
            "CONTROL_CONJUNCTION": "MEDIUM",
            "CONTROL_HIGH_RISK": "HIGH",
            "CONTROL_COLLISION": "CRITICAL",
        }

        validation_passed = True

        for scenario_name, expected_risk in (
            expected.items()
        ):

            rows = results_df[
                results_df[
                    "SCENARIO_TYPE"
                ]
                == scenario_name
            ]

            if rows.empty:

                validation_passed = False

                print(
                    f"  [FAIL] "
                    f"{scenario_name}: "
                    f"scenario missing"
                )

                continue

            actual_risk = str(
                rows[
                    "RISK_LEVEL"
                ].iloc[0]
            )

            if actual_risk == expected_risk:

                print(
                    f"  [PASS] "
                    f"{scenario_name}: "
                    f"{actual_risk}"
                )

            else:

                validation_passed = False

                print(
                    f"  [FAIL] "
                    f"{scenario_name}: "
                    f"expected {expected_risk}, "
                    f"got {actual_risk}"
                )

        print()

        if validation_passed:

            print(
                "CONTROLLED VALIDATION: PASSED"
            )

        else:

            print(
                "CONTROLLED VALIDATION: FAILED"
            )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis():

    print("=" * 60)

    print(
        "ROCKET-DEBRIS COLLISION "
        "PROBABILITY ANALYSIS"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Load rocket data.
    # --------------------------------------------------------

    rocket_data, controlled = (
        load_rocket_trajectories()
    )

    # --------------------------------------------------------
    # Load propagated orbital data.
    # --------------------------------------------------------

    propagated = (
        load_propagated_grid()
    )

    # --------------------------------------------------------
    # Run appropriate analysis.
    # --------------------------------------------------------

    if controlled:

        results_df = run_controlled_analysis(
            rocket_data,
            propagated,
        )

    else:

        results_df = run_normal_analysis(
            rocket_data,
            propagated,
        )

    # --------------------------------------------------------
    # Check results.
    # --------------------------------------------------------

    if results_df.empty:

        print(
            "\nNo results were generated."
        )

        return results_df

    # --------------------------------------------------------
    # Sort results.
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            [
                "MISS_DISTANCE_KM",
                "COLLISION_PROBABILITY",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Create output directory.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print_results(
        results_df,
        controlled=controlled,
    )

    # --------------------------------------------------------
    # Output location.
    # --------------------------------------------------------

    print()
    print(
        "Output file:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("Done.")

    return results_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_analysis()
