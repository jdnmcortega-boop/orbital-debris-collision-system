import numpy as np
import pandas as pd

import config


def estimate_collision_probability(
    pos_a, pos_b,
    sigma_km=None,
    n_samples=None,
    hard_body_radius_km=None,
):
    """
    Monte Carlo collision probability for one conjunction pair at TCA.

    pos_a, pos_b: nominal [X, Y, Z] positions (km) at closest approach.
    sigma_km: 1-sigma position uncertainty per axis, applied independently
              to each object (simple isotropic Gaussian model — real ops use
              full covariance matrices, but this is a reasonable first model).
    Returns: (probability, samples_within_radius, n_samples)
    """
    sigma_km = sigma_km or getattr(config, "POSITION_UNCERTAINTY_KM", 1.0)
    n_samples = n_samples or config.MC_SAMPLES
    hard_body_radius_km = hard_body_radius_km or config.HARD_BODY_RADIUS_KM

    pos_a = np.array(pos_a, dtype=float)
    pos_b = np.array(pos_b, dtype=float)

    # Independent Gaussian position noise for each object, each axis
    noise_a = np.random.normal(0, sigma_km, size=(n_samples, 3))
    noise_b = np.random.normal(0, sigma_km, size=(n_samples, 3))

    samples_a = pos_a + noise_a
    samples_b = pos_b + noise_b

    distances = np.linalg.norm(samples_a - samples_b, axis=1)
    hits = np.sum(distances < hard_body_radius_km)

    probability = hits / n_samples
    return probability, hits, n_samples


def get_position_at_tca(propagated_df, norad_id, tca):
    """Look up an object's propagated X/Y/Z at (or nearest to) TCA."""
    obj_rows = propagated_df[propagated_df["NORAD_CAT_ID"] == norad_id].copy()
    obj_rows["TIME"] = pd.to_datetime(obj_rows["TIME"])
    tca = pd.to_datetime(tca)

    idx = (obj_rows["TIME"] - tca).abs().idxmin()
    row = obj_rows.loc[idx]
    return [row["X_KM"], row["Y_KM"], row["Z_KM"]]


def run_monte_carlo(conjunctions_df, propagated_df, verbose=True):
    """
    Run Monte Carlo collision probability estimation for every row in
    conjunctions_df. Returns a copy with a COLLISION_PROBABILITY_MC column.
    """
    results = conjunctions_df.copy()
    probabilities = []

    for _, row in conjunctions_df.iterrows():
        pos_a = get_position_at_tca(propagated_df, row["NORAD_A"], row["TCA"])
        pos_b = get_position_at_tca(propagated_df, row["NORAD_B"], row["TCA"])

        prob, hits, n = estimate_collision_probability(pos_a, pos_b)
        probabilities.append(prob)

        if verbose:
            print(f"[MC] {row['OBJECT_A']} vs {row['OBJECT_B']}: "
                  f"P={prob:.6f} ({hits}/{n} samples)")

    results["COLLISION_PROBABILITY_MC"] = probabilities
    return results


def run_and_save(output_path=None):
    output_path = output_path or (config.RESULTS_DIR / "monte_carlo_results.csv")
    config.ensure_dirs()

    conjunctions_df = pd.read_csv(config.CONJUNCTIONS_FILE, parse_dates=["TCA"])
    propagated_df = pd.read_csv(config.PROPAGATED_GRID_FILE, parse_dates=["TIME"])

    if len(conjunctions_df) == 0:
        print("No conjunctions to process — run conjunction_detection first, "
              "or lower SCREENING_DISTANCE_KM.")
        return None

    results = run_monte_carlo(conjunctions_df, propagated_df)
    results.to_csv(output_path, index=False)

    print(f"\nResults written: {output_path}")
    print(results[["OBJECT_A", "OBJECT_B", "MISS_DISTANCE_KM",
                    "COLLISION_PROBABILITY_MC"]].to_string(index=False))

    return results


if __name__ == "__main__":
    run_and_save()