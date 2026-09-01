import numpy as np
import pandas as pd

import config


# ============================================================
# MONTE CARLO COLLISION-PROBABILITY THRESHOLDS
# ============================================================

RISK_THRESHOLD_HIGH = getattr(
    config,
    "RISK_THRESHOLD_HIGH",
    1e-4
)

RISK_THRESHOLD_MEDIUM = getattr(
    config,
    "RISK_THRESHOLD_MEDIUM",
    1e-6
)


def classify_risk(probability):
    """
    Classify collision risk using ONLY the Monte Carlo
    collision probability.

    This is the actual probability-based risk classification.
    """

    probability = float(probability)

    if probability >= RISK_THRESHOLD_HIGH:
        return "HIGH"

    elif probability >= RISK_THRESHOLD_MEDIUM:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# REFERENCE VELOCITY
# ============================================================

REFERENCE_VELOCITY_KM_S = getattr(
    config,
    "REFERENCE_VELOCITY_KM_S",
    10.0
)


# ============================================================
# VELOCITY SEVERITY
# ============================================================

def severity_factor(relative_velocity_km_s):
    """
    Estimate encounter severity from relative velocity.

    Kinetic energy scales approximately with velocity squared,
    therefore higher relative velocity increases consequence
    severity.

    IMPORTANT:
    This is NOT collision probability.
    """

    velocity = max(
        float(relative_velocity_km_s),
        0.0
    )

    return (
        velocity /
        REFERENCE_VELOCITY_KM_S
    ) ** 2


# ============================================================
# ORBITAL GEOMETRY FACTOR
# ============================================================

def orbital_geometry_factor(
    miss_distance_km,
    altitude_difference_km,
    inclination_difference_deg,
):
    """
    Calculate an orbital-geometry relevance factor.

    Smaller miss distance, smaller altitude separation, and
    smaller inclination separation produce a higher factor.

    Range:
        approximately 0 to 1

    IMPORTANT:
    This is a ranking factor, NOT collision probability.
    """

    miss_distance_km = max(
        abs(float(miss_distance_km)),
        0.0
    )

    altitude_difference_km = max(
        abs(float(altitude_difference_km)),
        0.0
    )

    inclination_difference_deg = max(
        abs(float(inclination_difference_deg)),
        0.0
    )

    # --------------------------------------------------------
    # MISS DISTANCE
    # --------------------------------------------------------

    distance_factor = np.exp(
        -miss_distance_km / 25.0
    )

    # --------------------------------------------------------
    # ALTITUDE SEPARATION
    # --------------------------------------------------------

    altitude_factor = np.exp(
        -altitude_difference_km / 25.0
    )

    # --------------------------------------------------------
    # INCLINATION SEPARATION
    # --------------------------------------------------------

    inclination_factor = np.exp(
        -inclination_difference_deg / 15.0
    )

    # --------------------------------------------------------
    # WEIGHTED GEOMETRY
    # --------------------------------------------------------

    geometry_factor = (
        0.50 * distance_factor
        + 0.30 * altitude_factor
        + 0.20 * inclination_factor
    )

    return float(
        np.clip(
            geometry_factor,
            0.0,
            1.0
        )
    )


# ============================================================
# DISTANCE FACTOR
# ============================================================

def distance_factor(miss_distance_km):
    """
    Convert miss distance into a bounded ranking factor.

    Smaller miss distance -> higher factor.

    This is deliberately stronger than the old uncertainty
    component because physical separation should be one of
    the primary indicators of conjunction severity.
    """

    distance = max(
        abs(float(miss_distance_km)),
        0.0
    )

    return float(
        1.0 /
        (
            1.0 +
            distance / 10.0
        )
    )


# ============================================================
# UNCERTAINTY FACTOR
# ============================================================

def uncertainty_factor(
    miss_distance_km,
    sigma_a_km,
    sigma_b_km,
):
    """
    Estimate how much the combined positional uncertainty
    overlaps the nominal miss distance.

    IMPORTANT FIX:

    Large uncertainty is NOT automatically considered high risk.

    The previous implementation used:

        combined_sigma / miss_distance

    and then capped it at 1.

    This caused an object such as DIWATA 2B to receive a maximum
    uncertainty contribution simply because one uncertainty value
    was very large.

    Instead, uncertainty is now treated as a moderate modifier
    of the physical conjunction geometry.

    The factor is bounded between 0 and 1.
    """

    miss_distance = max(
        abs(float(miss_distance_km)),
        0.001
    )

    sigma_a = max(
        float(sigma_a_km),
        0.0
    )

    sigma_b = max(
        float(sigma_b_km),
        0.0
    )

    combined_sigma = np.sqrt(
        sigma_a ** 2 +
        sigma_b ** 2
    )

    # Ratio between uncertainty and separation.
    ratio = combined_sigma / miss_distance

    # Convert to a smooth bounded factor.
    #
    # This approaches 1 when uncertainty is comparable to or
    # larger than the miss distance, but does not allow huge
    # uncertainty to dominate the entire risk score.
    factor = (
        ratio /
        (1.0 + ratio)
    )

    return float(
        np.clip(
            factor,
            0.0,
            1.0
        )
    )


# ============================================================
# COMPOSITE RISK SCORE
# ============================================================

def composite_risk_score(
    probability,
    relative_velocity_km_s,
    miss_distance_km,
    sigma_a_km,
    sigma_b_km,
    altitude_difference_km,
    inclination_difference_deg,
):
    """
    Calculate a composite conjunction-priority score.

    The score combines:

        1. Monte Carlo collision probability
        2. Miss distance
        3. Orbital geometry
        4. Relative velocity
        5. Positional uncertainty

    IMPORTANT:

    This score is a PRIORITIZATION / RANKING SCORE.

    It is NOT collision probability.

    Physical conjunction geometry is deliberately given more
    importance than uncertainty so that a very uncertain object
    that is hundreds of kilometers away is not incorrectly
    classified as a medium-risk conjunction.
    """

    probability = max(
        float(probability),
        0.0
    )

    # ========================================================
    # 1. COLLISION PROBABILITY
    # ========================================================

    if RISK_THRESHOLD_HIGH > 0:

        probability_factor = min(
            probability /
            RISK_THRESHOLD_HIGH,
            1.0
        )

    else:

        probability_factor = 0.0

    # ========================================================
    # 2. MISS DISTANCE
    # ========================================================

    dist_factor = distance_factor(
        miss_distance_km
    )

    # ========================================================
    # 3. ORBITAL GEOMETRY
    # ========================================================

    geometry_factor = orbital_geometry_factor(
        miss_distance_km,
        altitude_difference_km,
        inclination_difference_deg,
    )

    # ========================================================
    # 4. RELATIVE VELOCITY
    # ========================================================

    velocity_raw = severity_factor(
        relative_velocity_km_s
    )

    # Cap the velocity contribution so extremely high
    # relative velocities cannot dominate the score.

    velocity_factor = min(
        velocity_raw / 2.0,
        1.0
    )

    # ========================================================
    # 5. POSITIONAL UNCERTAINTY
    # ========================================================

    uncertainty = uncertainty_factor(
        miss_distance_km,
        sigma_a_km,
        sigma_b_km,
    )

    # ========================================================
    # 6. FINAL SCORE
    # ========================================================
    #
    # Physical geometry receives the greatest influence.
    #
    # Probability:
    #     30%
    #
    # Miss distance:
    #     30%
    #
    # Orbital geometry:
    #     20%
    #
    # Relative velocity:
    #     15%
    #
    # Uncertainty:
    #      5%
    #
    # This prevents large uncertainty values from artificially
    # producing MEDIUM risk for distant conjunctions.
    #

    score = (
        0.30 * probability_factor
        + 0.30 * dist_factor
        + 0.20 * geometry_factor
        + 0.15 * velocity_factor
        + 0.05 * uncertainty
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# COMPOSITE RISK CLASSIFICATION
# ============================================================

def classify_composite_risk(
    score,
    miss_distance_km,
    altitude_difference_km,
):
    """
    Classify the composite conjunction-priority score.

    IMPORTANT FIX:

    Composite MEDIUM/HIGH classification now requires the
    objects to be physically close enough to represent a
    meaningful conjunction.

    This prevents cases such as:

        DIWATA 2B
        miss distance = 45.45 km
        altitude difference = 518.81 km
        uncertainty = 215 km

    from becoming MEDIUM simply because uncertainty is large.

    Classification:

        HIGH:
            score >= 0.70 AND physically close

        MEDIUM:
            score >= 0.40 AND physically close

        LOW:
            otherwise
    """

    score = float(score)

    miss_distance = abs(
        float(miss_distance_km)
    )

    altitude_difference = abs(
        float(altitude_difference_km)
    )

    # --------------------------------------------------------
    # PHYSICAL-CONJUNCTION GATES
    # --------------------------------------------------------
    #
    # These are ranking gates, not collision-probability
    # thresholds.
    #
    # A very large altitude separation should prevent an event
    # from being promoted simply because of uncertainty.
    #

    CLOSE_MISS_DISTANCE_KM = 25.0
    CLOSE_ALTITUDE_DIFFERENCE_KM = 50.0

    # HIGH requires especially close geometry.
    HIGH_MISS_DISTANCE_KM = 10.0
    HIGH_ALTITUDE_DIFFERENCE_KM = 25.0

    high_geometry = (
        miss_distance <= HIGH_MISS_DISTANCE_KM
        and
        altitude_difference <= HIGH_ALTITUDE_DIFFERENCE_KM
    )

    medium_geometry = (
        miss_distance <= CLOSE_MISS_DISTANCE_KM
        and
        altitude_difference <= CLOSE_ALTITUDE_DIFFERENCE_KM
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if (
        score >= 0.70
        and high_geometry
    ):
        return "HIGH"

    elif (
        score >= 0.40
        and medium_geometry
    ):
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# OPERATOR / COUNTRY LOOKUP
# ============================================================

OPERATOR_LOOKUP = {
    "FENGYUN": ("China", "CNSA"),
    "IRIDIUM": ("USA", "Iridium Communications"),
    "COSMOS": ("Russia", "Roscosmos"),
    "CALSPHERE": ("USA", "US Air Force"),
    "LCS": ("USA", "US Air Force"),
    "TEMPSAT": ("USA", "US Air Force"),
    "RIGIDSPHERE": ("USA", "US Air Force"),
    "OPS 5712": ("USA", "US Air Force"),
    "SURCAL": ("USA", "US Air Force"),
    "LES-5": ("USA", "MIT Lincoln Laboratory"),
    "OSCAR": ("USA", "AMSAT"),
    "LUSAT": ("Argentina", "AMSAT-LU"),
    "STARLETTE": ("France", "CNES"),
    "LAGEOS": ("USA", "NASA"),
    "PHASE 3B": ("Germany", "AMSAT-DL"),
    "UOSAT": ("UK", "Surrey Satellite Technology"),
    "AJISAI": ("Japan", "JAXA"),
    "TDRS": ("USA", "NASA"),
    "ETALON": ("Russia", "Roscosmos"),
    "FLTSATCOM": ("USA", "US Navy"),

    # Expanded dataset
    "YAOGAN": ("China", "CNSA"),
    "SKYNET": ("UK", "UK Ministry of Defence"),
    "ASTRA": ("Luxembourg", "SES"),
    "COSMO-SKYMED": ("Italy", "ASI"),
    "RADARSAT": ("Canada", "Canadian Space Agency / MDA"),
    "NAVSTAR": ("USA", "US Space Force (GPS)"),
    "HORIZONS": ("USA", "Intelsat"),
    "THURAYA": ("UAE", "Thuraya Telecommunications"),
    "THOR": ("Norway", "Telenor Satellite"),
    "AMC-": ("USA", "SES Americom"),
    "DIRECTV": ("USA", "DIRECTV"),
    "ICO": ("USA", "ICO Global Communications"),
    "VINASAT": ("Vietnam", "VNPT"),
    "STAR ONE": ("Brazil", "Embratel Star One"),
    "CARTOSAT": ("India", "ISRO"),
    "CUTE-1.7": ("Japan", "Tokyo Institute of Technology"),
    "CANX": ("Canada", "University of Toronto (UTIAS/SFL)"),
    "SEEDS": ("Japan", "Nihon University"),
    "AMOS": ("Israel", "Spacecom"),
    "GALAXY": ("USA", "Intelsat"),
    "YUBILEINY": ("Russia", "Roscosmos"),
    "ZHONGXING": ("China", "China Satcom"),
    "FGRST": ("USA", "NASA"),
    "TURKSAT": ("Turkey", "Turksat"),
    "ECHOSTAR": ("USA", "EchoStar"),
    "SUPERBIRD": ("Japan", "SKY Perfect JSAT"),
    "INMARSAT": ("UK", "Inmarsat"),
    "HUANJING": ("China", "CNSA"),
    "GEOEYE": ("USA", "GeoEye"),
    "NIMIQ": ("Canada", "Telesat"),
    "THEOS": ("Thailand", "GISTDA"),
    "SHIJIAN": ("China", "CNSA"),
    "SHIYAN": ("China", "CNSA"),
    "CHUANGXIN": ("China", "CNSA"),
    "EUTELSAT": ("France", "Eutelsat"),
    "GOSAT": ("Japan", "JAXA"),
    "STARS": ("Japan", "Kagawa University"),
    "KKS-1": ("Japan", "Kyushu Institute of Technology"),
}


def lookup_operator(object_name):
    """
    Identify country and operator using the object-name prefix.
    """

    if pd.isna(object_name):
        return "Unknown", "Unknown"

    name = str(object_name).strip().upper()

    for prefix, (country, operator) in OPERATOR_LOOKUP.items():

        if name.startswith(prefix):
            return country, operator

    return "Unknown", "Unknown"


# ============================================================
# PREDICTION GENERATION
# ============================================================

def build_predictions(mc_results_df):
    """
    Take Monte Carlo results and generate:

        - Monte Carlo collision probability
        - probability-based risk level
        - orbital geometry factor
        - composite risk score
        - composite risk level
        - country/operator attribution
    """

    df = mc_results_df.copy()

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "COLLISION_PROBABILITY_MC",
        "RELATIVE_VELOCITY_KM_S",
        "MISS_DISTANCE_KM",
        "SIGMA_A_KM",
        "SIGMA_B_KM",
        "ALTITUDE_DIFFERENCE_KM",
        "INCLINATION_DIFFERENCE_DEG",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns in Monte Carlo results: "
            + ", ".join(missing_columns)
        )

    # ========================================================
    # ACTUAL MONTE CARLO RISK LEVEL
    # ========================================================

    df["RISK_LEVEL"] = (
        df["COLLISION_PROBABILITY_MC"]
        .fillna(0.0)
        .apply(classify_risk)
    )

    # ========================================================
    # ORBITAL GEOMETRY FACTOR
    # ========================================================

    df["ORBITAL_GEOMETRY_FACTOR"] = df.apply(
        lambda r: orbital_geometry_factor(
            r["MISS_DISTANCE_KM"],
            r["ALTITUDE_DIFFERENCE_KM"],
            r["INCLINATION_DIFFERENCE_DEG"],
        ),
        axis=1,
    )

    # ========================================================
    # COMPOSITE RISK SCORE
    # ========================================================

    df["COMPOSITE_RISK_SCORE"] = df.apply(
        lambda r: composite_risk_score(
            probability=r["COLLISION_PROBABILITY_MC"],
            relative_velocity_km_s=r[
                "RELATIVE_VELOCITY_KM_S"
            ],
            miss_distance_km=r[
                "MISS_DISTANCE_KM"
            ],
            sigma_a_km=r[
                "SIGMA_A_KM"
            ],
            sigma_b_km=r[
                "SIGMA_B_KM"
            ],
            altitude_difference_km=r[
                "ALTITUDE_DIFFERENCE_KM"
            ],
            inclination_difference_deg=r[
                "INCLINATION_DIFFERENCE_DEG"
            ],
        ),
        axis=1,
    )

    # ========================================================
    # COMPOSITE RISK LEVEL
    # ========================================================

    df["COMPOSITE_RISK_LEVEL"] = df.apply(
        lambda r: classify_composite_risk(
            r["COMPOSITE_RISK_SCORE"],
            r["MISS_DISTANCE_KM"],
            r["ALTITUDE_DIFFERENCE_KM"],
        ),
        axis=1,
    )

    # ========================================================
    # COUNTRY / OPERATOR INFORMATION
    # ========================================================

    country_a = []
    operator_a = []
    country_b = []
    operator_b = []

    for _, row in df.iterrows():

        ca, oa = lookup_operator(
            row["OBJECT_A"]
        )

        cb, ob = lookup_operator(
            row["OBJECT_B"]
        )

        country_a.append(ca)
        operator_a.append(oa)

        country_b.append(cb)
        operator_b.append(ob)

    df["COUNTRY_A"] = country_a
    df["OPERATOR_A"] = operator_a

    df["COUNTRY_B"] = country_b
    df["OPERATOR_B"] = operator_b

    # ========================================================
    # SORT BY COMPOSITE SCORE
    # ========================================================

    return (
        df
        .sort_values(
            "COMPOSITE_RISK_SCORE",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# WARNING MESSAGE
# ============================================================

def format_warning_message(row):
    """
    Generate a simulated collision-warning message.
    """

    return (
        "=== COLLISION WARNING ===\n"
        f"Object 1: {row['OBJECT_A']} "
        f"(NORAD {row['NORAD_A']}) "
        f"- {row['OPERATOR_A']} "
        f"({row['COUNTRY_A']})\n"

        f"Object 2: {row['OBJECT_B']} "
        f"(NORAD {row['NORAD_B']}) "
        f"- {row['OPERATOR_B']} "
        f"({row['COUNTRY_B']})\n"

        f"Predicted TCA: {row['TCA']}\n"

        f"Miss distance: "
        f"{row['MISS_DISTANCE_KM']:.3f} km\n"

        f"Altitude difference: "
        f"{row['ALTITUDE_DIFFERENCE_KM']:.3f} km\n"

        f"Inclination difference: "
        f"{row['INCLINATION_DIFFERENCE_DEG']:.3f} deg\n"

        f"Relative velocity: "
        f"{row['RELATIVE_VELOCITY_KM_S']:.3f} km/s\n"

        f"Collision probability (MC): "
        f"{row['COLLISION_PROBABILITY_MC']:.6e}\n"

        f"Monte Carlo 95% upper probability: "
        f"{row['MC_UPPER_95_PROBABILITY']:.6e}\n"

        f"Orbital geometry factor: "
        f"{row['ORBITAL_GEOMETRY_FACTOR']:.4f}\n"

        f"Composite risk score: "
        f"{row['COMPOSITE_RISK_SCORE']:.4f}\n"

        f"Risk level: "
        f"{row['COMPOSITE_RISK_LEVEL']}\n"

        "=========================\n"
    )


# ============================================================
# WARNING GENERATION
# ============================================================

def generate_warnings(
    predictions_df,
    min_risk="MEDIUM"
):
    """
    Generate warnings for events at or above the selected
    composite-risk level.
    """

    order = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
    }

    min_risk = str(
        min_risk
    ).upper()

    if min_risk not in order:
        raise ValueError(
            "min_risk must be LOW, MEDIUM, or HIGH."
        )

    threshold = order[min_risk]

    flagged = predictions_df[
        predictions_df[
            "COMPOSITE_RISK_LEVEL"
        ]
        .map(order)
        >= threshold
    ]

    return [
        format_warning_message(row)
        for _, row in flagged.iterrows()
    ]


# ============================================================
# RUN AND SAVE
# ============================================================

def run_and_save():
    """
    Load Monte Carlo results, generate predictions,
    classify risks, and save predictions.csv.
    """

    config.ensure_dirs()

    mc_path = (
        config.RESULTS_DIR /
        "monte_carlo_results.csv"
    )

    if not mc_path.exists():

        print(
            f"No Monte Carlo results found at "
            f"{mc_path}. Run monte_carlo.py first."
        )

        return None

    # ========================================================
    # LOAD MONTE CARLO RESULTS
    # ========================================================

    mc_results = pd.read_csv(
        mc_path,
        parse_dates=["TCA"]
    )

    # ========================================================
    # BUILD PREDICTIONS
    # ========================================================

    predictions = build_predictions(
        mc_results
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    output_path = (
        config.RESULTS_DIR /
        "predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False
    )

    print(
        f"Predictions written: {output_path}"
    )

    # ========================================================
    # MONTE CARLO RISK COUNTS
    # ========================================================

    print(
        "\nRisk level counts "
        "(Monte Carlo collision probability):"
    )

    print(
        predictions[
            "RISK_LEVEL"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # COMPOSITE RISK COUNTS
    # ========================================================

    print(
        "\nComposite risk level counts "
        "(probability + geometry + velocity + "
        "uncertainty + distance):"
    )

    print(
        predictions[
            "COMPOSITE_RISK_LEVEL"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # TOP RISK EVENTS
    # ========================================================

    print(
        "\nTop risk events "
        "(ranked by composite score):"
    )

    display_columns = [
        "OBJECT_A",
        "OBJECT_B",
        "MISS_DISTANCE_KM",
        "ALTITUDE_DIFFERENCE_KM",
        "INCLINATION_DIFFERENCE_DEG",
        "RELATIVE_VELOCITY_KM_S",
        "SIGMA_A_KM",
        "SIGMA_B_KM",
        "ORBITAL_GEOMETRY_FACTOR",
        "COLLISION_PROBABILITY_MC",
        "COMPOSITE_RISK_SCORE",
        "RISK_LEVEL",
        "COMPOSITE_RISK_LEVEL",
    ]

    print(
        predictions[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # GENERATE WARNINGS
    # ========================================================

    warnings = generate_warnings(
        predictions,
        min_risk="MEDIUM"
    )

    warnings_path = (
        config.RESULTS_DIR /
        "warnings.txt"
    )

    with open(
        warnings_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(warnings)
            if warnings
            else "No MEDIUM/HIGH risk events detected.\n"
        )

    print(
        f"\nWarnings written: "
        f"{warnings_path} "
        f"({len(warnings)} message(s))"
    )

    if warnings:

        print(
            "\n" +
            warnings[0]
        )

    return predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_and_save()