"""
Standalone analysis (NOT part of the main pipeline): quantifies how orbital
parameters (altitude, inclination, eccentricity) relate to collision risk,
providing the statistical backing for objective 1.2 — "based on orbital
parameters such as altitude, inclination, and eccentricity."

Miss distance and collision probability are outputs of SGP4 propagation,
which itself takes orbital elements as input — altitude/inclination/
eccentricity influence Pc THROUGH their effect on relative geometry, not as
separate direct inputs to the probability formula. This script makes that
relationship explicit and measurable via correlation and regression,
rather than leaving it as an implicit, unverified claim.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

import config
from modules import data_loader
from modules import orbital_mechanics as om
from modules.qae import analytic_collision_probability


def build_orbital_features(orbital_data_df):
    """Per-object altitude (mean of apogee/perigee), inclination, eccentricity."""
    rows = []
    for _, row in orbital_data_df.iterrows():
        apogee, perigee = om.apogee_perigee_altitude_km(row["MEAN_MOTION"], row["ECCENTRICITY"])
        rows.append({
            "NORAD_CAT_ID": row["NORAD_CAT_ID"],
            "ALTITUDE_KM": (apogee + perigee) / 2.0,
            "INCLINATION_DEG": row["INCLINATION"],
            "ECCENTRICITY": row["ECCENTRICITY"],
        })
    return pd.DataFrame(rows)


def build_analysis_table(conjunctions_df, orbital_features_df):
    """Join per-object orbital features onto each conjunction pair and derive
    pairwise differences, plus the analytic collision probability."""
    features = orbital_features_df.set_index("NORAD_CAT_ID")

    rows = []
    for _, row in conjunctions_df.iterrows():
        if row["NORAD_A"] not in features.index or row["NORAD_B"] not in features.index:
            continue

        a = features.loc[row["NORAD_A"]]
        b = features.loc[row["NORAD_B"]]

        analytic_pc = analytic_collision_probability(row["MISS_DISTANCE_KM"])

        rows.append({
            "OBJECT_A": row["OBJECT_A"],
            "OBJECT_B": row["OBJECT_B"],
            "MISS_DISTANCE_KM": row["MISS_DISTANCE_KM"],
            "ANALYTIC_PC": analytic_pc,
            "ALTITUDE_DIFF_KM": abs(a["ALTITUDE_KM"] - b["ALTITUDE_KM"]),
            "INCLINATION_DIFF_DEG": abs(a["INCLINATION_DEG"] - b["INCLINATION_DEG"]),
            "MEAN_ECCENTRICITY": (a["ECCENTRICITY"] + b["ECCENTRICITY"]) / 2.0,
        })

    return pd.DataFrame(rows)


def compute_correlations(analysis_df):
    """Pearson correlation between each orbital parameter and miss distance / Pc."""
    predictors = ["ALTITUDE_DIFF_KM", "INCLINATION_DIFF_DEG", "MEAN_ECCENTRICITY"]
    targets = ["MISS_DISTANCE_KM", "ANALYTIC_PC"]

    results = []
    for predictor in predictors:
        for target in targets:
            x = analysis_df[predictor].values
            y = analysis_df[target].values

            if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
                r, p = float("nan"), float("nan")
            else:
                r, p = stats.pearsonr(x, y)

            results.append({
                "predictor": predictor,
                "target": target,
                "pearson_r": r,
                "p_value": p,
                "significant_at_0.05": (p < 0.05) if not np.isnan(p) else None,
            })

    return pd.DataFrame(results)


def fit_regression(analysis_df, target="MISS_DISTANCE_KM"):
    """Multiple linear regression: target ~ altitude_diff + inclination_diff + eccentricity."""
    predictors = ["ALTITUDE_DIFF_KM", "INCLINATION_DIFF_DEG", "MEAN_ECCENTRICITY"]
    X = analysis_df[predictors].values
    y = analysis_df[target].values

    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)

    return {
        "target": target,
        "r_squared": r_squared,
        "coefficients": dict(zip(predictors, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "n_samples": len(analysis_df),
    }


def run_and_save():
    config.ensure_dirs()

    conj_path = config.CONJUNCTIONS_FILE
    if not conj_path.exists():
        print(f"No conjunctions file found at {conj_path}. Run conjunction_detection first.")
        return None

    conjunctions = pd.read_csv(conj_path)
    if len(conjunctions) < 3:
        print(f"Only {len(conjunctions)} conjunction(s) found — need at least 3 for "
              f"a meaningful correlation/regression analysis. Loosen SCREENING_DISTANCE_KM "
              f"in config.py to get more candidate pairs.")
        return None

    orbital_data = data_loader.load_orbital_data()
    orbital_features = build_orbital_features(orbital_data)
    analysis_df = build_analysis_table(conjunctions, orbital_features)

    if len(analysis_df) < 3:
        print("Too few matched pairs after joining orbital features — cannot proceed.")
        return None

    correlations = compute_correlations(analysis_df)
    regression_miss = fit_regression(analysis_df, target="MISS_DISTANCE_KM")
    regression_pc = fit_regression(analysis_df, target="ANALYTIC_PC")

    print(f"Analyzed {len(analysis_df)} conjunction pairs.\n")

    print("=== Correlations (orbital parameter vs miss distance / Pc) ===")
    print(correlations.to_string(index=False))

    print(f"\n=== Regression: MISS_DISTANCE_KM ~ orbital parameters ===")
    print(f"R^2 = {regression_miss['r_squared']:.4f}")
    for k, v in regression_miss["coefficients"].items():
        print(f"  {k}: {v:.4f}")

    print(f"\n=== Regression: ANALYTIC_PC ~ orbital parameters ===")
    print(f"R^2 = {regression_pc['r_squared']:.4f}")
    for k, v in regression_pc["coefficients"].items():
        print(f"  {k}: {v:.6e}")

    analysis_df.to_csv(config.RESULTS_DIR / "orbital_parameter_analysis.csv", index=False)
    correlations.to_csv(config.RESULTS_DIR / "orbital_parameter_correlations.csv", index=False)

    import json
    with open(config.RESULTS_DIR / "orbital_parameter_regression.json", "w") as f:
        json.dump({"miss_distance_model": regression_miss, "pc_model": regression_pc},
                   f, indent=2)

    print(f"\nResults written to {config.RESULTS_DIR}")

    return analysis_df, correlations, regression_miss, regression_pc


if __name__ == "__main__":
    run_and_save()