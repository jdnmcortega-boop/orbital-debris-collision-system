import numpy as np
import pandas as pd

import config
from modules.uncertainty_model import get_pair_sigmas


def _validate_probability_inputs(
    sigma_a_km,
    sigma_b_km,
    n_samples,
    hard_body_radius_km,
):
    if sigma_a_km < 0 or sigma_b_km < 0:
        raise ValueError("Position uncertainty cannot be negative.")
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    if hard_body_radius_km <= 0:
        raise ValueError("HARD_BODY_RADIUS_KM must be greater than zero.")


def _wilson_upper_for_zero_hits(n_samples, alpha=0.05):
    return float(1.0 - alpha ** (1.0 / n_samples))


def _normal_confidence_interval(estimate, standard_error, z=1.96):
    low = max(0.0, float(estimate - z * standard_error))
    high = min(1.0, float(estimate + z * standard_error))
    return low, high


def estimate_collision_probability_bruteforce(
    pos_a,
    pos_b,
    sigma_a_km=None,
    sigma_b_km=None,
    n_samples=None,
    hard_body_radius_km=None,
    random_seed=None,
):
    """Reference 3-D brute-force estimator retained for comparison only."""
    default_sigma = float(getattr(config, "POSITION_UNCERTAINTY_KM", 1.0))
    sigma_a_km = default_sigma if sigma_a_km is None else float(sigma_a_km)
    sigma_b_km = default_sigma if sigma_b_km is None else float(sigma_b_km)
    n_samples = int(config.MC_SAMPLES if n_samples is None else n_samples)
    hard_body_radius_km = float(
        config.HARD_BODY_RADIUS_KM if hard_body_radius_km is None else hard_body_radius_km
    )

    _validate_probability_inputs(
        sigma_a_km, sigma_b_km, n_samples, hard_body_radius_km
    )

    pos_a = np.asarray(pos_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float)
    if pos_a.shape != (3,) or pos_b.shape != (3,):
        raise ValueError("pos_a and pos_b must each contain exactly three coordinates [X, Y, Z].")

    rng = np.random.default_rng(random_seed)
    noise_a = rng.normal(0.0, sigma_a_km, size=(n_samples, 3))
    noise_b = rng.normal(0.0, sigma_b_km, size=(n_samples, 3))
    distances = np.linalg.norm((pos_a + noise_a) - (pos_b + noise_b), axis=1)
    hits = int(np.count_nonzero(distances <= hard_body_radius_km))
    probability = hits / float(n_samples)
    upper = _wilson_upper_for_zero_hits(n_samples) if hits == 0 else min(
        1.0,
        probability + 1.96 * np.sqrt(probability * (1.0 - probability) / n_samples),
    )
    return probability, hits, n_samples, upper


def estimate_collision_probability(
    pos_a,
    pos_b,
    sigma_a_km=None,
    sigma_b_km=None,
    n_samples=None,
    hard_body_radius_km=None,
    random_seed=None,
):
    """
    Estimate collision probability with encounter-plane importance sampling.

    The analytic/QAE probability model in this project is a 2-D encounter
    plane model with independent isotropic Gaussian relative-position
    uncertainty. Direct brute-force 3-D sampling is inefficient at
    probabilities near 1e-6 because almost all samples miss the hard-body
    region. This estimator instead samples uniformly inside the hard-body
    disk, then applies the exact Gaussian likelihood weight.

    The estimator is unbiased for the same 2-D probability model used by
    QAE, and its uncertainty is reported from the variance of the weighted
    samples. No arbitrary probability inflation or threshold change is used.

    Returns:
        probability
        collision_hits (NaN; not a Bernoulli hit count under importance
            sampling)
        n_samples
        upper_95_probability
        ci_low
        effective_sample_size
    """
    default_sigma = float(getattr(config, "POSITION_UNCERTAINTY_KM", 1.0))
    sigma_a_km = default_sigma if sigma_a_km is None else float(sigma_a_km)
    sigma_b_km = default_sigma if sigma_b_km is None else float(sigma_b_km)
    n_samples = int(config.MC_SAMPLES if n_samples is None else n_samples)
    hard_body_radius_km = float(
        config.HARD_BODY_RADIUS_KM if hard_body_radius_km is None else hard_body_radius_km
    )

    _validate_probability_inputs(
        sigma_a_km, sigma_b_km, n_samples, hard_body_radius_km
    )

    pos_a = np.asarray(pos_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float)
    if pos_a.shape != (3,) or pos_b.shape != (3,):
        raise ValueError("pos_a and pos_b must each contain exactly three coordinates [X, Y, Z].")

    relative_vector = pos_a - pos_b
    miss_distance_km = float(np.linalg.norm(relative_vector))
    combined_sigma = float(np.sqrt(sigma_a_km ** 2 + sigma_b_km ** 2))

    if combined_sigma <= 0:
        probability = 1.0 if miss_distance_km <= hard_body_radius_km else 0.0
        return probability, np.nan, n_samples, probability, probability, float(n_samples)

    # In the isotropic encounter-plane model, only the magnitude of the
    # nominal relative displacement matters. Put that displacement on x.
    d = miss_distance_km
    radius = hard_body_radius_km
    disk_area = np.pi * radius ** 2

    rng = np.random.default_rng(random_seed)

    # Uniform disk proposal centered on the collision point.
    radial = radius * np.sqrt(rng.random(n_samples))
    angle = 2.0 * np.pi * rng.random(n_samples)
    x = radial * np.cos(angle)
    y = radial * np.sin(angle)

    # Target density: N([d, 0], combined_sigma^2 I_2).
    squared_distance = (x - d) ** 2 + y ** 2
    normalization = 1.0 / (2.0 * np.pi * combined_sigma ** 2)
    target_density = normalization * np.exp(
        -squared_distance / (2.0 * combined_sigma ** 2)
    )

    # Proposal density is uniform over the disk.
    weights = target_density * disk_area

    estimate = float(np.mean(weights))

    if len(weights) > 1:
        sample_variance = float(np.var(weights, ddof=1))
        standard_error = float(np.sqrt(sample_variance / n_samples))
    else:
        standard_error = 0.0

    ci_low, ci_high = _normal_confidence_interval(
        estimate,
        standard_error,
    )

    sum_weights = float(np.sum(weights))
    sum_squared_weights = float(np.sum(weights ** 2))
    if sum_squared_weights > 0:
        effective_sample_size = (sum_weights ** 2) / sum_squared_weights
    else:
        effective_sample_size = 0.0

    estimate = float(np.clip(estimate, 0.0, 1.0))
    ci_low = float(np.clip(ci_low, 0.0, 1.0))
    ci_high = float(np.clip(ci_high, 0.0, 1.0))

    return (
        estimate,
        np.nan,
        n_samples,
        ci_high,
        ci_low,
        float(effective_sample_size),
    )


def get_position_at_tca(propagated_df, norad_id, tca):
    """Look up the propagated X/Y/Z position nearest to TCA."""
    obj_rows = propagated_df[
        propagated_df["NORAD_CAT_ID"] == norad_id
    ].copy()

    if obj_rows.empty:
        raise ValueError(f"No propagated position found for NORAD {norad_id}.")

    obj_rows["TIME"] = pd.to_datetime(obj_rows["TIME"], utc=True)
    tca = pd.to_datetime(tca, utc=True)
    idx = (obj_rows["TIME"] - tca).abs().idxmin()
    row = obj_rows.loc[idx]

    return np.array(
        [float(row["X_KM"]), float(row["Y_KM"]), float(row["Z_KM"])],
        dtype=float,
    )


def calculate_orbital_geometry(row, orbital_data_indexed):
    """Calculate inclination and approximate altitude differences."""
    if orbital_data_indexed is None:
        return np.nan, np.nan

    try:
        inclination_a = float(orbital_data_indexed.loc[row["NORAD_A"], "INCLINATION"])
        inclination_b = float(orbital_data_indexed.loc[row["NORAD_B"], "INCLINATION"])
        inclination_difference = abs(inclination_a - inclination_b)
        inclination_difference = min(inclination_difference, 180.0 - inclination_difference)

        earth_mu = 398600.4418
        earth_radius = 6378.137
        mean_motion_a = float(orbital_data_indexed.loc[row["NORAD_A"], "MEAN_MOTION"])
        mean_motion_b = float(orbital_data_indexed.loc[row["NORAD_B"], "MEAN_MOTION"])
        if mean_motion_a <= 0 or mean_motion_b <= 0:
            return inclination_difference, np.nan

        n_a = mean_motion_a * 2.0 * np.pi / 86400.0
        n_b = mean_motion_b * 2.0 * np.pi / 86400.0
        semi_major_axis_a = (earth_mu / (n_a ** 2)) ** (1.0 / 3.0)
        semi_major_axis_b = (earth_mu / (n_b ** 2)) ** (1.0 / 3.0)
        altitude_difference = abs(
            (semi_major_axis_a - earth_radius) - (semi_major_axis_b - earth_radius)
        )

        return inclination_difference, altitude_difference
    except (KeyError, ValueError, TypeError):
        return np.nan, np.nan


def run_monte_carlo(conjunctions_df, propagated_df, orbital_data_df=None, verbose=True):
    """Run rare-event importance-sampling MC for every conjunction."""
    results = conjunctions_df.copy()

    probabilities = []
    hits_list = []
    samples_list = []
    upper_95_list = []
    ci_low_list = []
    ess_list = []
    method_list = []
    sigma_a_list = []
    sigma_b_list = []
    inclination_difference_list = []
    altitude_difference_list = []

    orbital_data_indexed = None
    if orbital_data_df is not None:
        orbital_data_indexed = orbital_data_df.set_index("NORAD_CAT_ID")

    for row_number, (_, row) in enumerate(conjunctions_df.iterrows()):
        pos_a = get_position_at_tca(propagated_df, row["NORAD_A"], row["TCA"])
        pos_b = get_position_at_tca(propagated_df, row["NORAD_B"], row["TCA"])

        if orbital_data_indexed is not None:
            sigma_a, sigma_b = get_pair_sigmas(row, orbital_data_indexed)
        else:
            default_sigma = float(getattr(config, "POSITION_UNCERTAINTY_KM", 1.0))
            sigma_a, sigma_b = default_sigma, default_sigma

        sigma_a = float(sigma_a)
        sigma_b = float(sigma_b)

        (
            probability,
            hits,
            n,
            upper_95_probability,
            ci_low,
            effective_sample_size,
        ) = estimate_collision_probability(
            pos_a,
            pos_b,
            sigma_a_km=sigma_a,
            sigma_b_km=sigma_b,
            random_seed=100000 + row_number,
        )

        inclination_difference, altitude_difference = calculate_orbital_geometry(
            row,
            orbital_data_indexed,
        )

        probabilities.append(probability)
        hits_list.append(hits)
        samples_list.append(n)
        upper_95_list.append(upper_95_probability)
        ci_low_list.append(ci_low)
        ess_list.append(effective_sample_size)
        method_list.append(config.MC_METHOD)
        sigma_a_list.append(sigma_a)
        sigma_b_list.append(sigma_b)
        inclination_difference_list.append(inclination_difference)
        altitude_difference_list.append(altitude_difference)

        if verbose:
            print(
                f"[MC-IS] {row['OBJECT_A']} vs {row['OBJECT_B']}: "
                f"P={probability:.6e}, "
                f"95%CI=[{ci_low:.6e}, {upper_95_probability:.6e}], "
                f"N={n}, ESS={effective_sample_size:.0f}, "
                f"sigma_a={sigma_a:.2f}km, sigma_b={sigma_b:.2f}km, "
                f"delta_i={inclination_difference:.2f}deg, "
                f"delta_alt={altitude_difference:.2f}km"
            )

    results["SIGMA_A_KM"] = sigma_a_list
    results["SIGMA_B_KM"] = sigma_b_list
    results["MC_COLLISION_HITS"] = hits_list
    results["MC_SAMPLES"] = samples_list
    results["COLLISION_PROBABILITY_MC"] = probabilities
    results["MC_UPPER_95_PROBABILITY"] = upper_95_list
    results["MC_CI_LOW"] = ci_low_list
    results["MC_CI_HIGH"] = upper_95_list
    results["MC_EFFECTIVE_SAMPLE_SIZE"] = ess_list
    results["MC_METHOD"] = method_list
    results["INCLINATION_DIFFERENCE_DEG"] = inclination_difference_list
    results["ALTITUDE_DIFFERENCE_KM"] = altitude_difference_list

    return results


def run_and_save(output_path=None):
    output_path = output_path or (config.RESULTS_DIR / "monte_carlo_results.csv")
    config.ensure_dirs()

    from modules import data_loader

    conjunctions_df = pd.read_csv(
        config.CONJUNCTIONS_FILE,
        parse_dates=["TCA"],
    )
    propagated_df = pd.read_csv(
        config.PROPAGATED_GRID_FILE,
        parse_dates=["TIME"],
    )

    if conjunctions_df.empty:
        print(
            "No conjunctions to process - run conjunction_detection first, "
            "or lower SCREENING_DISTANCE_KM."
        )
        return None

    try:
        orbital_data_df = data_loader.load_orbital_data()
    except Exception as exc:
        print(
            "Could not load orbital data for age-scaled uncertainty/orbital "
            f"geometry ({exc}) - falling back to fixed uncertainty."
        )
        orbital_data_df = None

    results = run_monte_carlo(
        conjunctions_df,
        propagated_df,
        orbital_data_df,
    )

    results.to_csv(output_path, index=False)

    print(f"\nResults written: {output_path}")
    print(
        results[
            [
                "OBJECT_A",
                "OBJECT_B",
                "MISS_DISTANCE_KM",
                "RELATIVE_VELOCITY_KM_S",
                "SIGMA_A_KM",
                "SIGMA_B_KM",
                "MC_METHOD",
                "MC_SAMPLES",
                "MC_EFFECTIVE_SAMPLE_SIZE",
                "COLLISION_PROBABILITY_MC",
                "MC_CI_LOW",
                "MC_CI_HIGH",
                "INCLINATION_DIFFERENCE_DEG",
                "ALTITUDE_DIFFERENCE_KM",
            ]
        ].to_string(index=False)
    )

    return results


if __name__ == "__main__":
    run_and_save()
