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
    relative_velocity_vector=None,
):
    """
    Estimate collision probability with encounter-plane importance sampling.

    The analytic/QAE probability model in this project is a 2-D encounter
    plane model with independent isotropic Gaussian relative-position
    uncertainty. Direct brute-force 3-D sampling is inefficient at
    probabilities near 1e-6 because almost all samples miss the hard-body
    region. This estimator instead samples uniformly inside the hard-body
    disk for rare events and applies the exact Gaussian likelihood weight.
    For non-rare events it samples directly from the relative 2-D Gaussian,
    which handles narrow distributions near the collision disk efficiently.

    The hybrid estimator targets the same 2-D isotropic Gaussian model used
    by the project's QAE comparison. No arbitrary probability inflation or
    threshold change is used. This is a research model, not a substitute for
    full covariance-based operational conjunction assessment.

    Returns:
        probability
        collision_hits (NaN; not a Bernoulli hit count under importance
            sampling)
        n_samples
        upper_95_probability
        ci_low
        effective_sample_size
        log10_probability (retains rare-event scale when float probability underflows)
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
    # Project the relative position into the plane perpendicular to relative
    # velocity. This removes residual along-track separation caused by a
    # discrete propagation grid when the sampled timestamp is near, but not
    # exactly at, TCA. Positions must be in the same inertial frame and km.
    if relative_velocity_vector is not None:
        relative_velocity_vector = np.asarray(relative_velocity_vector, dtype=float)
        if relative_velocity_vector.shape != (3,) or not np.all(np.isfinite(relative_velocity_vector)):
            raise ValueError("relative_velocity_vector must be a finite 3-vector in km/s.")
        speed = float(np.linalg.norm(relative_velocity_vector))
        if speed > 0.0:
            v_hat = relative_velocity_vector / speed
            relative_vector = relative_vector - np.dot(relative_vector, v_hat) * v_hat
    miss_distance_km = float(np.linalg.norm(relative_vector))
    combined_sigma = float(np.sqrt(sigma_a_km ** 2 + sigma_b_km ** 2))

    if combined_sigma <= 0:
        probability = 1.0 if miss_distance_km <= hard_body_radius_km else 0.0
        log10_probability = float('-inf') if probability == 0.0 else 0.0
        return probability, np.nan, n_samples, probability, probability, float(n_samples), log10_probability

    # In the isotropic encounter-plane model, only the magnitude of the
    # nominal relative displacement matters. Put that displacement on x.
    d = miss_distance_km
    radius = hard_body_radius_km
    disk_area = np.pi * radius ** 2

    rng = np.random.default_rng(random_seed)

    # For non-rare encounters, sample from the actual relative Gaussian.
    # This avoids the failure of a uniform-disk proposal when sigma is tiny:
    # almost none of the disk proposal points land in the narrow Gaussian peak.
    # Rare events use importance sampling below.
    if d <= radius + 3.0 * combined_sigma:
        x_direct = d + rng.normal(0.0, combined_sigma, size=n_samples)
        y_direct = rng.normal(0.0, combined_sigma, size=n_samples)
        hit_count = int(np.count_nonzero(x_direct ** 2 + y_direct ** 2 <= radius ** 2))
        probability = hit_count / float(n_samples)
        if hit_count == 0:
            ci_low = 0.0
            ci_high = _wilson_upper_for_zero_hits(n_samples)
        else:
            standard_error = float(np.sqrt(probability * (1.0 - probability) / n_samples))
            ci_low, ci_high = _normal_confidence_interval(probability, standard_error)
        log10_probability = float(np.log10(probability)) if probability > 0 else float('-inf')
        return (
            probability,
            hit_count,
            n_samples,
            ci_high,
            ci_low,
            float(n_samples),
            log10_probability,
        )

    # Uniform disk proposal centered on the collision point.
    radial = radius * np.sqrt(rng.random(n_samples))
    angle = 2.0 * np.pi * rng.random(n_samples)
    x = radial * np.cos(angle)
    y = radial * np.sin(angle)

    # Compute importance weights in log space. Directly evaluating exp(-d^2 /
    # (2*sigma^2)) underflows to exact zero for distant encounters even when
    # the mathematical probability is positive. Log space preserves the
    # probability scale for diagnostics without artificially increasing it.
    squared_distance = (x - d) ** 2 + y ** 2
    log_weights = (
        np.log(disk_area)
        - np.log(2.0 * np.pi * combined_sigma ** 2)
        - squared_distance / (2.0 * combined_sigma ** 2)
    )
    max_log_weight = float(np.max(log_weights))
    scaled_weights = np.exp(log_weights - max_log_weight)
    scaled_mean = float(np.mean(scaled_weights))
    log_estimate = max_log_weight + np.log(scaled_mean)
    log10_probability = float(log_estimate / np.log(10.0))

    # Convert back to ordinary probability only when representable as a
    # positive float. A zero here means numerical underflow, not zero hits.
    min_log_float = float(np.log(np.nextafter(0.0, 1.0)))
    estimate = float(np.exp(log_estimate)) if log_estimate >= min_log_float else 0.0

    scaled_se = 0.0
    if len(scaled_weights) > 1:
        scaled_se = float(np.sqrt(np.var(scaled_weights, ddof=1) / n_samples))
    if scaled_mean > 0.0:
        relative_se = scaled_se / scaled_mean
        standard_error = estimate * relative_se
    else:
        standard_error = 0.0

    ci_low, ci_high = _normal_confidence_interval(estimate, standard_error)

    sum_weights_scaled = float(np.sum(scaled_weights))
    sum_squared_weights_scaled = float(np.sum(scaled_weights ** 2))
    effective_sample_size = (
        (sum_weights_scaled ** 2) / sum_squared_weights_scaled
        if sum_squared_weights_scaled > 0 else 0.0
    )

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
        log10_probability,
    )


def get_state_at_tca(propagated_df, norad_id, tca):
    """Return position (km) and velocity (km/s) nearest to the requested TCA."""
    obj_rows = propagated_df[
        propagated_df["NORAD_CAT_ID"] == norad_id
    ].copy()

    if obj_rows.empty:
        raise ValueError(f"No propagated position found for NORAD {norad_id}.")

    obj_rows["TIME"] = pd.to_datetime(obj_rows["TIME"], utc=True)
    tca = pd.to_datetime(tca, utc=True)
    idx = (obj_rows["TIME"] - tca).abs().idxmin()
    row = obj_rows.loc[idx]

    position = np.array(
        [float(row["X_KM"]), float(row["Y_KM"]), float(row["Z_KM"])],
        dtype=float,
    )
    velocity_columns = ["VX_KM_S", "VY_KM_S", "VZ_KM_S"]
    if all(col in obj_rows.columns for col in velocity_columns):
        velocity = np.array([float(row[col]) for col in velocity_columns], dtype=float)
    else:
        velocity = None
    return position, velocity


def get_position_at_tca(propagated_df, norad_id, tca):
    """Backward-compatible position-only helper."""
    position, _ = get_state_at_tca(propagated_df, norad_id, tca)
    return position


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
    log10_probability_list = []
    underflow_list = []
    sigma_a_list = []
    sigma_b_list = []
    inclination_difference_list = []
    altitude_difference_list = []

    orbital_data_indexed = None
    if orbital_data_df is not None:
        orbital_data_indexed = orbital_data_df.set_index("NORAD_CAT_ID")

    for row_number, (_, row) in enumerate(conjunctions_df.iterrows()):
        pos_a, vel_a = get_state_at_tca(propagated_df, row["NORAD_A"], row["TCA"])
        pos_b, vel_b = get_state_at_tca(propagated_df, row["NORAD_B"], row["TCA"])
        relative_velocity_vector = (
            vel_a - vel_b if vel_a is not None and vel_b is not None else None
        )

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
            log10_probability,
        ) = estimate_collision_probability(
            pos_a,
            pos_b,
            sigma_a_km=sigma_a,
            sigma_b_km=sigma_b,
            random_seed=100000 + row_number,
            relative_velocity_vector=relative_velocity_vector,
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
        log10_probability_list.append(log10_probability)
        underflow_list.append(probability == 0.0 and np.isfinite(log10_probability))
        sigma_a_list.append(sigma_a)
        sigma_b_list.append(sigma_b)
        inclination_difference_list.append(inclination_difference)
        altitude_difference_list.append(altitude_difference)

        if verbose:
            print(
                f"[MC-HYBRID] {row['OBJECT_A']} vs {row['OBJECT_B']}: "
                f"P={probability:.6e}, log10(P)={log10_probability:.3f}, "
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
    results["MC_LOG10_PROBABILITY"] = log10_probability_list
    results["MC_PROBABILITY_UNDERFLOW"] = underflow_list
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
                "MC_LOG10_PROBABILITY",
                "MC_PROBABILITY_UNDERFLOW",
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
