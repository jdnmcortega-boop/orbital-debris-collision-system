import pandas as pd

import config


# ============================================================
# RISK THRESHOLDS
# ============================================================
# Based on commonly cited operational conjunction-assessment practice:
# Pc >= 1e-4 is a widely used "red" / maneuver-consideration threshold
# (e.g. ISS and NASA CARA operational guidance); Pc >= 1e-6 is a common
# "worth tracking" threshold below which events are usually deprioritized.
# These are reasonable, citable defaults for a research project — state
# them explicitly in your methodology rather than treating them as fact.

RISK_THRESHOLD_HIGH = getattr(config, "RISK_THRESHOLD_HIGH", 1e-4)
RISK_THRESHOLD_MEDIUM = getattr(config, "RISK_THRESHOLD_MEDIUM", 1e-6)


def classify_risk(probability):
    if probability >= RISK_THRESHOLD_HIGH:
        return "HIGH"
    elif probability >= RISK_THRESHOLD_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


# ============================================================
# OPERATOR / COUNTRY LOOKUP
# ============================================================
# Public-record operator/country attribution for objects commonly found
# in this dataset (matched by name prefix). Not exhaustive — anything
# not listed falls back to "Unknown". Verify against Space-Track/UCS
# Satellite Database before citing specific attributions in your paper.

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
    # --- Added for the expanded 105-object dataset ---
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
    for prefix, (country, operator) in OPERATOR_LOOKUP.items():
        if object_name.upper().startswith(prefix):
            return country, operator
    return "Unknown", "Unknown"


# ============================================================
# PREDICTION / WARNING GENERATION
# ============================================================

def build_predictions(mc_results_df):
    """
    Take Monte Carlo results (with COLLISION_PROBABILITY_MC) and add
    risk classification + operator/country attribution for both objects.
    """
    df = mc_results_df.copy()

    df["RISK_LEVEL"] = df["COLLISION_PROBABILITY_MC"].apply(classify_risk)

    country_a, operator_a, country_b, operator_b = [], [], [], []
    for _, row in df.iterrows():
        ca, oa = lookup_operator(row["OBJECT_A"])
        cb, ob = lookup_operator(row["OBJECT_B"])
        country_a.append(ca)
        operator_a.append(oa)
        country_b.append(cb)
        operator_b.append(ob)

    df["COUNTRY_A"] = country_a
    df["OPERATOR_A"] = operator_a
    df["COUNTRY_B"] = country_b
    df["OPERATOR_B"] = operator_b

    return df.sort_values("COLLISION_PROBABILITY_MC", ascending=False).reset_index(drop=True)


def format_warning_message(row):
    """Build a simulated collision-warning message for one high/medium-risk event."""
    return (
        "=== COLLISION WARNING ===\n"
        f"Object 1: {row['OBJECT_A']} (NORAD {row['NORAD_A']}) "
        f"- {row['OPERATOR_A']} ({row['COUNTRY_A']})\n"
        f"Object 2: {row['OBJECT_B']} (NORAD {row['NORAD_B']}) "
        f"- {row['OPERATOR_B']} ({row['COUNTRY_B']})\n"
        f"Predicted TCA: {row['TCA']}\n"
        f"Miss distance: {row['MISS_DISTANCE_KM']:.3f} km\n"
        f"Relative velocity: {row['RELATIVE_VELOCITY_KM_S']:.3f} km/s\n"
        f"Collision probability (MC): {row['COLLISION_PROBABILITY_MC']:.6e}\n"
        f"Risk level: {row['RISK_LEVEL']}\n"
        "=========================\n"
    )


def generate_warnings(predictions_df, min_risk="MEDIUM"):
    """Generate warning messages for events at or above the given risk level."""
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    threshold = order[min_risk]

    flagged = predictions_df[predictions_df["RISK_LEVEL"].map(order) >= threshold]
    return [format_warning_message(row) for _, row in flagged.iterrows()]


def run_and_save():
    config.ensure_dirs()

    mc_path = config.RESULTS_DIR / "monte_carlo_results.csv"
    if not mc_path.exists():
        print(f"No Monte Carlo results found at {mc_path}. Run monte_carlo.py first.")
        return None

    mc_results = pd.read_csv(mc_path, parse_dates=["TCA"])
    predictions = build_predictions(mc_results)

    output_path = config.RESULTS_DIR / "predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Predictions written: {output_path}")

    print("\nRisk level counts:")
    print(predictions["RISK_LEVEL"].value_counts().to_string())

    warnings = generate_warnings(predictions, min_risk="MEDIUM")
    warnings_path = config.RESULTS_DIR / "warnings.txt"
    with open(warnings_path, "w") as f:
        f.write("\n".join(warnings) if warnings else "No MEDIUM/HIGH risk events detected.\n")
    print(f"\nWarnings written: {warnings_path} ({len(warnings)} message(s))")

    if warnings:
        print("\n" + warnings[0])

    return predictions


if __name__ == "__main__":
    run_and_save()