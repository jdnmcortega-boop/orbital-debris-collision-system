import numpy as np
import pandas as pd

import config
from modules.uncertainty_model import get_pair_sigmas


def estimate_collision_probability(
    pos_a, pos_b,
    sigma_a_km=None,
    sigma_b_km=None,
    n_samples=None,
    hard_body_radius_km=None,
):
    """
    Monte Carlo collision probability for one conjunction pair at TCA.

    pos_a, pos_b: nominal [X, Y, Z] positions (km) at closest approach.
    sigma_a_km, sigma_b_km: 1-sigma position uncertainty per axis for EACH
        object independently (age-scaled per object via uncertainty_model.py
        by default — objects with older TLEs get larger uncertainty, rather
        than one fixed value applied to every object regardless of how
        current its tracking data is).
    Returns: (probability, samples_within_radius, n_samples)
    """
    default_sigma = getattr(config, "POSITION_UNCERTAINTY_KM", 1.0)
    sigma_a_km = sigma_a_km if sigma_a_km is not None else default_sigma
    sigma_b_km = sigma_b_km if sigma_b_km is not None else default_sigma
    n_samples = n_samples or config.MC_SAMPLES
    hard_body_radius_km = hard_body_radius_km or config.HARD_BODY_RADIUS_KM

    pos_a = np.array(pos_a, dtype=float)
    pos_b = np.array(pos_b, dtype=float)

    # Independent Gaussian position noise for each object, each with its
    # OWN sigma (age-scaled), each axis
    noise_a = np.random.normal(0, sigma_a_km, size=(n_samples, 3))
    noise_b = np.random.normal(0, sigma_b_km, size=(n_samples, 3))

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


def run_monte_carlo(conjunctions_df, propagated_df, orbital_data_df=None, verbose=True):
    """
    Run Monte Carlo collision probability estimation for every row in
    conjunctions_df. Returns a copy with COLLISION_PROBABILITY_MC,
    SIGMA_A_KM, and SIGMA_B_KM columns.

    If orbital_data_df is provided, per-object age-scaled uncertainty is
    used (each object's own TLE epoch age at TCA). If not provided, falls
    back to the fixed config.POSITION_UNCERTAINTY_KM for both objects.
    """
    results = conjunctions_df.copy()
    probabilities = []
    sigma_a_list, sigma_b_list = [], []

    orbital_data_indexed = None
    if orbital_data_df is not None:
        orbital_data_indexed = orbital_data_df.set_index("NORAD_CAT_ID")

    for _, row in conjunctions_df.iterrows():
        pos_a = get_position_at_tca(propagated_df, row["NORAD_A"], row["TCA"])
        pos_b = get_position_at_tca(propagated_df, row["NORAD_B"], row["TCA"])

        if orbital_data_indexed is not None:
            sigma_a, sigma_b = get_pair_sigmas(row, orbital_data_indexed)
        else:
            default_sigma = getattr(config, "POSITION_UNCERTAINTY_KM", 1.0)
            sigma_a, sigma_b = default_sigma, default_sigma

        prob, hits, n = estimate_collision_probability(
            pos_a, pos_b, sigma_a_km=sigma_a, sigma_b_km=sigma_b
        )
        probabilities.append(prob)
        sigma_a_list.append(sigma_a)
        sigma_b_list.append(sigma_b)

        if verbose:
            print(f"[MC] {row['OBJECT_A']} vs {row['OBJECT_B']}: "
                  f"P={prob:.6f} ({hits}/{n} samples, "
                  f"sigma_a={sigma_a:.2f}km, sigma_b={sigma_b:.2f}km)")

    results["SIGMA_A_KM"] = sigma_a_list
    results["SIGMA_B_KM"] = sigma_b_list
    results["COLLISION_PROBABILITY_MC"] = probabilities
    return results


def run_and_save(output_path=None):
    output_path = output_path or (config.RESULTS_DIR / "monte_carlo_results.csv")
    config.ensure_dirs()

    from modules import data_loader

    conjunctions_df = pd.read_csv(config.CONJUNCTIONS_FILE, parse_dates=["TCA"])
    propagated_df = pd.read_csv(config.PROPAGATED_GRID_FILE, parse_dates=["TIME"])

    if len(conjunctions_df) == 0:
        print("No conjunctions to process — run conjunction_detection first, "
              "or lower SCREENING_DISTANCE_KM.")
        return None

    try:
        orbital_data_df = data_loader.load_orbital_data()
    except Exception as e:
        print(f"Could not load orbital data for age-scaled uncertainty "
              f"({e}) — falling back to fixed uncertainty for all objects.")
        orbital_data_df = None

    results = run_monte_carlo(conjunctions_df, propagated_df, orbital_data_df)
    results.to_csv(output_path, index=False)

    print(f"\nResults written: {output_path}")
    print(results[["OBJECT_A", "OBJECT_B", "MISS_DISTANCE_KM", "SIGMA_A_KM",
                    "SIGMA_B_KM", "COLLISION_PROBABILITY_MC"]].to_string(index=False))

    return results


if __name__ == "__main__":
    run_and_save()