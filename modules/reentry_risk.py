"""
Reentry-likelihood classification for objects involved in a conjunction.

SCOPE NOTE (state this in your methodology):
Modeling how collision fragments' orbits change requires a fragmentation/
breakup model (e.g. NASA's standard breakup model) — out of scope here.
Instead, this module classifies each object's EXISTING perigee altitude
(computed from its own orbital elements, independent of any collision)
into a qualitative reentry-likelihood category, based on well-established
orbital-decay behavior: perigee altitude is the dominant factor in how
quickly atmospheric drag causes decay. This tells you whether the objects
involved in a conjunction are already on a decaying trajectory — useful
context for a collision event, not a simulation of the collision's
physical aftermath.
"""

import numpy as np
import pandas as pd


GM_EARTH_KM3_S2 = 398600.4418   # standard gravitational parameter of Earth
EARTH_RADIUS_KM = 6378.137      # WGS84 equatorial radius


# ============================================================
# OBJECT TYPE CLASSIFICATION
# ============================================================

def classify_object_type(object_name):
    """SATELLITE (intact spacecraft) vs DEBRIS, from standard TLE naming."""
    return "DEBRIS" if "DEB" in object_name.upper() else "SATELLITE"


def classify_collision_type(name_a, name_b):
    type_a = classify_object_type(name_a)
    type_b = classify_object_type(name_b)
    pair = sorted([type_a, type_b])
    return f"{pair[0]}-{pair[1]}"


# ============================================================
# PERIGEE ALTITUDE (from orbital elements, independent of propagation)
# ============================================================

def perigee_altitude_km(mean_motion_rev_per_day, eccentricity):
    """
    Compute perigee altitude from mean motion and eccentricity via Kepler's
    third law. mean_motion is in revolutions/day (as given in the raw CSV).
    """
    n_rad_s = float(mean_motion_rev_per_day) * 2.0 * np.pi / 86400.0
    semi_major_axis_km = (GM_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)
    perigee_radius_km = semi_major_axis_km * (1.0 - float(eccentricity))
    return perigee_radius_km - EARTH_RADIUS_KM


# ============================================================
# REENTRY LIKELIHOOD CLASSIFICATION
# ============================================================
# Qualitative bands based on widely-cited debris-literature behavior
# (e.g. NASA ODPO / ESA Space Debris Office reporting): decay timescale
# rises sharply with perigee altitude. These are illustrative bands, not
# precise predictions — cite general orbital-decay literature for exact
# figures if your report needs them.

REENTRY_BANDS = [
    (200,  "Imminent (days-weeks)"),
    (300,  "Very high (weeks-months)"),
    (450,  "High (months-few years)"),
    (600,  "Moderate (years-decades)"),
    (1000, "Low (decades-centuries)"),
    (float("inf"), "Negligible (centuries+, effectively stable)"),
]


def classify_reentry_likelihood(perigee_km):
    for threshold, label in REENTRY_BANDS:
        if perigee_km < threshold:
            return label
    return REENTRY_BANDS[-1][1]


# ============================================================
# PIPELINE INTEGRATION
# ============================================================

def build_reentry_analysis(conjunctions_or_predictions_df, orbital_data_df):
    """
    Add object-type, collision-type, perigee altitude, and reentry
    likelihood columns for both objects in each conjunction/prediction row.

    orbital_data_df: the raw loaded orbital elements (from data_loader),
    used to look up MEAN_MOTION/ECCENTRICITY by NORAD_CAT_ID.
    """
    df = conjunctions_or_predictions_df.copy()

    elements_by_norad = orbital_data_df.set_index("NORAD_CAT_ID")

    def lookup_perigee(norad_id):
        row = elements_by_norad.loc[norad_id]
        return perigee_altitude_km(row["MEAN_MOTION"], row["ECCENTRICITY"])

    df["PERIGEE_A_KM"] = df["NORAD_A"].apply(lookup_perigee)
    df["PERIGEE_B_KM"] = df["NORAD_B"].apply(lookup_perigee)

    df["OBJECT_A_TYPE"] = df["OBJECT_A"].apply(classify_object_type)
    df["OBJECT_B_TYPE"] = df["OBJECT_B"].apply(classify_object_type)
    df["COLLISION_TYPE"] = df.apply(
        lambda r: classify_collision_type(r["OBJECT_A"], r["OBJECT_B"]), axis=1
    )

    df["REENTRY_LIKELIHOOD_A"] = df["PERIGEE_A_KM"].apply(classify_reentry_likelihood)
    df["REENTRY_LIKELIHOOD_B"] = df["PERIGEE_B_KM"].apply(classify_reentry_likelihood)

    return df


def run_and_save():
    import config
    from modules import data_loader

    config.ensure_dirs()

    conj_path = config.CONJUNCTIONS_FILE
    if not conj_path.exists():
        print(f"No conjunctions file found at {conj_path}. Run conjunction_detection first.")
        return None

    conjunctions = pd.read_csv(conj_path)
    orbital_data = data_loader.load_orbital_data()

    analysis = build_reentry_analysis(conjunctions, orbital_data)

    output_path = config.RESULTS_DIR / "reentry_analysis.csv"
    analysis.to_csv(output_path, index=False)

    print(f"\nResults written: {output_path}")
    print("\nCollision type counts:")
    print(analysis["COLLISION_TYPE"].value_counts().to_string())

    print(analysis[["OBJECT_A", "OBJECT_A_TYPE", "REENTRY_LIKELIHOOD_A",
                     "OBJECT_B", "OBJECT_B_TYPE", "REENTRY_LIKELIHOOD_B",
                     "COLLISION_TYPE"]].to_string(index=False))

    return analysis


if __name__ == "__main__":
    run_and_save()