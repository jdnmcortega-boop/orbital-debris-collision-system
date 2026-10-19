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


def probability_model_diagnostics(sigma_a_km=None, sigma_b_km=None):
    """Report whether configured Pc thresholds are reachable under the model.

    This is a mathematical diagnostic, not a probability estimate. It uses the
    same isotropic 2-D Gaussian assumption and hard-body radius as qae.py.
    Missing object-specific sigmas fall back to the configured assumed sigma.
    """
    radius = float(getattr(config, "HARD_BODY_RADIUS_KM", 0.02))
    fallback_sigma = float(getattr(config, "POSITION_UNCERTAINTY_KM", 10.0))

    if sigma_a_km is None or sigma_b_km is None:
        sigma_a = sigma_b = fallback_sigma
        source = "CONFIGURED_ASSUMPTION"
    else:
        sigma_a = float(sigma_a_km)
        sigma_b = float(sigma_b_km)
        source = "ROW_OBJECT_SIGMAS"

    if (
        not np.isfinite(radius) or radius <= 0
        or not np.isfinite(sigma_a) or sigma_a <= 0
        or not np.isfinite(sigma_b) or sigma_b <= 0
    ):
        return {
            "PC_MAX_AT_ZERO_MISS": np.nan,
            "PC_MEDIUM_THRESHOLD_REACHABLE": False,
            "PC_HIGH_THRESHOLD_REACHABLE": False,
            "PC_MODEL_DIAGNOSTIC": "INVALID_RADIUS_OR_UNCERTAINTY",
            "PC_UNCERTAINTY_SOURCE": source,
        }

    combined_sigma = np.sqrt(sigma_a**2 + sigma_b**2)
    max_pc = float(-np.expm1(-(radius**2) / (2.0 * combined_sigma**2)))
    medium_reachable = max_pc >= RISK_THRESHOLD_MEDIUM
    high_reachable = max_pc >= RISK_THRESHOLD_HIGH

    if not medium_reachable and not high_reachable:
        status = "BOTH_MEDIUM_AND_HIGH_THRESHOLDS_UNREACHABLE"
    elif not high_reachable:
        status = "HIGH_THRESHOLD_UNREACHABLE"
    else:
        status = "THRESHOLDS_REACHABLE_AT_ZERO_MISS_ONLY_OR_ABOVE"

    return {
        "PC_MAX_AT_ZERO_MISS": max_pc,
        "PC_MEDIUM_THRESHOLD_REACHABLE": bool(medium_reachable),
        "PC_HIGH_THRESHOLD_REACHABLE": bool(high_reachable),
        "PC_MODEL_DIAGNOSTIC": status,
        "PC_UNCERTAINTY_SOURCE": source,
    }


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
    """Score conjunction response priority, not collision probability.

    Probability is accepted for backward compatibility but deliberately does
    not enter this score. Collision probability and response priority are
    separate outputs: the former is Pc-only; the latter ranks close geometry,
    relative speed, and uncertainty for operator review.
    """
    dist_factor = distance_factor(miss_distance_km)
    geometry_factor = orbital_geometry_factor(
        miss_distance_km,
        altitude_difference_km,
        inclination_difference_deg,
    )
    velocity_raw = severity_factor(relative_velocity_km_s)
    velocity_factor = float(np.clip(velocity_raw / 2.0, 0.0, 1.0))
    uncertainty = uncertainty_factor(
        miss_distance_km,
        sigma_a_km,
        sigma_b_km,
    )
    score = (
        0.40 * dist_factor
        + 0.30 * geometry_factor
        + 0.20 * velocity_factor
        + 0.10 * uncertainty
    )
    return float(np.clip(score, 0.0, 1.0))


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

def build_predictions(results_df):
    """Build separate probability-based collision risk and response priority."""
    df = results_df.copy()
    probability_column = (
        "ANALYTIC_PC" if "ANALYTIC_PC" in df.columns
        else "COLLISION_PROBABILITY_MC" if "COLLISION_PROBABILITY_MC" in df.columns
        else None
    )
    required_columns = [
        "RELATIVE_VELOCITY_KM_S", "MISS_DISTANCE_KM",
        "SIGMA_A_KM", "SIGMA_B_KM",
        "ALTITUDE_DIFFERENCE_KM", "INCLINATION_DIFFERENCE_DEG",
    ]
    if probability_column is None:
        raise ValueError("Results need ANALYTIC_PC or COLLISION_PROBABILITY_MC.")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError("Missing required prediction columns: " + ", ".join(missing))

    df[probability_column] = pd.to_numeric(df[probability_column], errors="coerce")
    df["PROBABILITY_SOURCE"] = (
        "Analytical Pc (assumed uncertainty model)" if probability_column == "ANALYTIC_PC"
        else "Monte Carlo validation estimate"
    )
    df["COLLISION_RISK_LEVEL"] = df[probability_column].apply(
        lambda p: classify_risk(p) if pd.notna(p) and np.isfinite(float(p)) else "UNAVAILABLE"
    )
    # Backward-compatible alias. This field always means Pc-threshold risk.
    df["RISK_LEVEL"] = df["COLLISION_RISK_LEVEL"]

    # Explicitly expose when the assumed uncertainty makes configured Pc
    # categories mathematically unreachable. Do not silently relabel events
    # HIGH to compensate; the priority classifier remains a separate output.
    diagnostics = []
    for _, row in df.iterrows():
        sigma_a = row.get("SIGMA_A_KM")
        sigma_b = row.get("SIGMA_B_KM")
        if pd.isna(sigma_a) or pd.isna(sigma_b):
            sigma_a = sigma_b = None
        diagnostics.append(probability_model_diagnostics(sigma_a, sigma_b))
    diagnostic_df = pd.DataFrame(diagnostics, index=df.index)
    for column in diagnostic_df.columns:
        df[column] = diagnostic_df[column]

    df["ORBITAL_GEOMETRY_FACTOR"] = df.apply(
        lambda r: orbital_geometry_factor(
            r["MISS_DISTANCE_KM"], r["ALTITUDE_DIFFERENCE_KM"],
            r["INCLINATION_DIFFERENCE_DEG"],
        ) if all(pd.notna(r.get(c)) for c in [
            "MISS_DISTANCE_KM", "ALTITUDE_DIFFERENCE_KM", "INCLINATION_DIFFERENCE_DEG"
        ]) else np.nan,
        axis=1,
    )

    def score_row(r):
        needed = [
            "RELATIVE_VELOCITY_KM_S", "MISS_DISTANCE_KM",
            "SIGMA_A_KM", "SIGMA_B_KM", "ALTITUDE_DIFFERENCE_KM",
            "INCLINATION_DIFFERENCE_DEG",
        ]
        if not all(pd.notna(r.get(c)) for c in needed):
            return np.nan
        return composite_risk_score(
            probability=r[probability_column],
            relative_velocity_km_s=r["RELATIVE_VELOCITY_KM_S"],
            miss_distance_km=r["MISS_DISTANCE_KM"],
            sigma_a_km=r["SIGMA_A_KM"],
            sigma_b_km=r["SIGMA_B_KM"],
            altitude_difference_km=r["ALTITUDE_DIFFERENCE_KM"],
            inclination_difference_deg=r["INCLINATION_DIFFERENCE_DEG"],
        )

    df["ORION_X_PRIORITY_SCORE"] = df.apply(score_row, axis=1)
    df["ORION_X_PRIORITY_LEVEL"] = df.apply(
        lambda r: classify_composite_risk(
            r["ORION_X_PRIORITY_SCORE"], r["MISS_DISTANCE_KM"],
            r["ALTITUDE_DIFFERENCE_KM"],
        ) if pd.notna(r["ORION_X_PRIORITY_SCORE"]) else "UNAVAILABLE",
        axis=1,
    )
    # Compatibility aliases for existing UI/warning code. These now refer to
    # response priority, not collision probability; the explicit Pc label above
    # remains the only collision-risk classification.
    df["COMPOSITE_RISK_SCORE"] = df["ORION_X_PRIORITY_SCORE"]
    df["COMPOSITE_RISK_LEVEL"] = df["ORION_X_PRIORITY_LEVEL"]
    df["PRIORITY_BASIS"] = (
        "Miss distance, orbital geometry, relative velocity, and assumed uncertainty; not Pc"
    )

    for suffix in ("A", "B"):
        countries, operators = [], []
        names = df["OBJECT_" + suffix] if "OBJECT_" + suffix in df.columns else [""] * len(df)
        for name in names:
            country, operator = lookup_operator(name)
            countries.append(country)
            operators.append(operator)
        df["COUNTRY_" + suffix] = countries
        df["OPERATOR_" + suffix] = operators

    return df.sort_values(
        ["ORION_X_PRIORITY_SCORE", "ANALYTIC_PC"] if "ANALYTIC_PC" in df.columns
        else ["ORION_X_PRIORITY_SCORE"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


# ============================================================
# WARNING MESSAGE
# ============================================================

def format_warning_message(row):
    """Generate a warning that keeps Pc risk separate from response priority."""
    probability = row.get("ANALYTIC_PC", row.get("COLLISION_PROBABILITY_MC", np.nan))
    probability_source = row.get("PROBABILITY_SOURCE", "modeled probability")
    mc_upper = row.get("MC_UPPER_95_PROBABILITY", np.nan)
    return (
        "=== ORION-X CONJUNCTION REVIEW ===\n"
        f"Object 1: {row.get('OBJECT_A', 'Unknown')} (NORAD {row.get('NORAD_A', 'N/A')}) - "
        f"{row.get('OPERATOR_A', 'Unknown')} ({row.get('COUNTRY_A', 'Unknown')})\n"
        f"Object 2: {row.get('OBJECT_B', 'Unknown')} (NORAD {row.get('NORAD_B', 'N/A')}) - "
        f"{row.get('OPERATOR_B', 'Unknown')} ({row.get('COUNTRY_B', 'Unknown')})\n"
        f"Refined TCA: {row.get('TCA', 'N/A')}\n"
        f"Refinement method: {row.get('TCA_REFINEMENT_METHOD', 'not recorded')}\n"
        f"Refined miss distance: {float(row.get('MISS_DISTANCE_KM', np.nan)):.6f} km\n"
        f"Sampled miss distance: {float(row.get('SAMPLED_MISS_DISTANCE_KM', np.nan)):.6f} km\n"
        f"Altitude difference (semimajor-axis proxy): {float(row.get('ALTITUDE_DIFFERENCE_KM', np.nan)):.3f} km\n"
        f"Inclination difference: {float(row.get('INCLINATION_DIFFERENCE_DEG', np.nan)):.3f} deg\n"
        f"Relative velocity: {float(row.get('RELATIVE_VELOCITY_KM_S', np.nan)):.3f} km/s\n"
        f"Collision probability ({probability_source}): {float(probability):.6e}\n"
        f"Collision risk (Pc threshold): {row.get('COLLISION_RISK_LEVEL', row.get('RISK_LEVEL', 'UNAVAILABLE'))}\n"
        f"ORION-X response priority score: {float(row.get('ORION_X_PRIORITY_SCORE', row.get('COMPOSITE_RISK_SCORE', np.nan))):.4f}\n"
        f"ORION-X response priority: {row.get('ORION_X_PRIORITY_LEVEL', row.get('COMPOSITE_RISK_LEVEL', 'UNAVAILABLE'))}\n"
        f"Monte Carlo 95% upper probability (if available): {float(mc_upper):.6e}\n"
        "Priority is not a collision probability and does not establish that a collision will occur.\n"
        "=================================\n"
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
    """Prefer analytical Pc results; fall back to MC results for older workflows."""
    config.ensure_dirs()
    analytic_path = config.RESULTS_DIR / "analytic_pc_results.csv"
    mc_path = config.RESULTS_DIR / "monte_carlo_results.csv"
    input_path = analytic_path if analytic_path.exists() else mc_path
    if not input_path.exists():
        print(f"No analytical or Monte Carlo result file found in {config.RESULTS_DIR}.")
        return None

    results = pd.read_csv(input_path)
    predictions = build_predictions(results)
    output_path = config.RESULTS_DIR / "predictions.csv"
    predictions.to_csv(output_path, index=False)

    display_columns = [
        "OBJECT_A", "OBJECT_B", "TCA", "MISS_DISTANCE_KM",
        "ALTITUDE_DIFFERENCE_KM", "RELATIVE_VELOCITY_KM_S",
        "SIGMA_A_KM", "SIGMA_B_KM", "ANALYTIC_PC",
        "COLLISION_PROBABILITY_MC", "PROBABILITY_SOURCE",
        "COMPOSITE_RISK_SCORE", "RISK_LEVEL", "COMPOSITE_RISK_LEVEL",
    ]
    display_columns = [c for c in display_columns if c in predictions.columns]
    if display_columns:
        print(predictions[display_columns].head(10).to_string(index=False))

    warnings = generate_warnings(predictions, min_risk="MEDIUM")
    warnings_path = config.RESULTS_DIR / "warnings.txt"
    with open(warnings_path, "w", encoding="utf-8") as f:
        f.write("\n".join(warnings) if warnings else "No MEDIUM/HIGH risk events detected.\n")
    print(f"Predictions written: {output_path}")
    print(f"Warnings written: {warnings_path} ({len(warnings)} message(s))")
    if warnings:
        print("\n" + warnings[0])
    return predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_and_save()