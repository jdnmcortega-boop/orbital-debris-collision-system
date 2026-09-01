import numpy as np
import pandas as pd

import config
from modules.uncertainty_model import get_pair_sigmas


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
    Estimate collision probability using Monte Carlo sampling.

    Each object's position is independently perturbed using a
    3-dimensional Gaussian uncertainty model.

    Collision occurs when the sampled relative distance is less
    than the combined hard-body radius.

    Returns:
        probability
        hits
        n_samples
        upper_95_probability
    """

    default_sigma = getattr(
        config,
        "POSITION_UNCERTAINTY_KM",
        1.0
    )

    sigma_a_km = (
        float(sigma_a_km)
        if sigma_a_km is not None
        else float(default_sigma)
    )

    sigma_b_km = (
        float(sigma_b_km)
        if sigma_b_km is not None
        else float(default_sigma)
    )

    n_samples = (
        int(n_samples)
        if n_samples is not None
        else int(config.MC_SAMPLES)
    )

    hard_body_radius_km = (
        float(hard_body_radius_km)
        if hard_body_radius_km is not None
        else float(config.HARD_BODY_RADIUS_KM)
    )

    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")

    if sigma_a_km < 0 or sigma_b_km < 0:
        raise ValueError("Position uncertainty cannot be negative.")

    if hard_body_radius_km <= 0:
        raise ValueError(
            "HARD_BODY_RADIUS_KM must be greater than zero."
        )

    pos_a = np.asarray(pos_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float)

    if pos_a.shape != (3,) or pos_b.shape != (3,):
        raise ValueError(
            "pos_a and pos_b must each contain exactly "
            "three coordinates [X, Y, Z]."
        )

    # ------------------------------------------------------------
    # RANDOM NUMBER GENERATOR
    # ------------------------------------------------------------

    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------
    # INDEPENDENT POSITION UNCERTAINTY
    # ------------------------------------------------------------

    noise_a = rng.normal(
        loc=0.0,
        scale=sigma_a_km,
        size=(n_samples, 3)
    )

    noise_b = rng.normal(
        loc=0.0,
        scale=sigma_b_km,
        size=(n_samples, 3)
    )

    samples_a = pos_a + noise_a
    samples_b = pos_b + noise_b

    # ------------------------------------------------------------
    # RELATIVE DISTANCE
    # ------------------------------------------------------------

    relative_vectors = samples_a - samples_b

    distances = np.linalg.norm(
        relative_vectors,
        axis=1
    )

    # ------------------------------------------------------------
    # COLLISION CONDITION
    # ------------------------------------------------------------

    hits = int(
        np.count_nonzero(
            distances <= hard_body_radius_km
        )
    )

    probability = (
        hits / float(n_samples)
    )

    # ------------------------------------------------------------
    # 95% UPPER CONFIDENCE BOUND
    # ------------------------------------------------------------
    #
    # For zero observed collisions, the probability estimate is
    # exactly zero, but that does NOT mean the true probability
    # is mathematically zero.
    #
    # For a binomial experiment with zero successes, a simple
    # one-sided 95% upper bound is:
    #
    #       P_upper = 1 - alpha^(1/N)
    #
    # where alpha = 0.05.
    #
    # This gives the probability level below which the true
    # collision probability is expected to lie with 95% confidence
    # under the binomial sampling model.
    #

    alpha = 0.05

    if hits == 0:

        upper_95_probability = (
            1.0
            - alpha ** (1.0 / n_samples)
        )

    else:

        # Normal approximation for non-zero hit counts.
        standard_error = np.sqrt(
            probability
            * (1.0 - probability)
            / n_samples
        )

        upper_95_probability = min(
            1.0,
            probability
            + 1.96 * standard_error
        )

    return (
        probability,
        hits,
        n_samples,
        upper_95_probability
    )


def get_position_at_tca(
    propagated_df,
    norad_id,
    tca
):
    """
    Look up an object's propagated X/Y/Z position
    at the propagation time nearest to TCA.
    """

    obj_rows = propagated_df[
        propagated_df["NORAD_CAT_ID"] == norad_id
    ].copy()

    if obj_rows.empty:
        raise ValueError(
            f"No propagated position found for NORAD {norad_id}."
        )

    obj_rows["TIME"] = pd.to_datetime(
        obj_rows["TIME"],
        utc=True
    )

    tca = pd.to_datetime(
        tca,
        utc=True
    )

    idx = (
        obj_rows["TIME"] - tca
    ).abs().idxmin()

    row = obj_rows.loc[idx]

    return np.array(
        [
            float(row["X_KM"]),
            float(row["Y_KM"]),
            float(row["Z_KM"])
        ],
        dtype=float
    )


def calculate_orbital_geometry(
    row,
    orbital_data_indexed
):
    """
    Calculate orbital geometry variables for a conjunction.

    Returns:
        inclination_difference_deg,
        altitude_difference_km
    """

    if orbital_data_indexed is None:
        return np.nan, np.nan

    try:

        # ========================================================
        # INCLINATION DIFFERENCE
        # ========================================================

        inclination_a = float(
            orbital_data_indexed.loc[
                row["NORAD_A"],
                "INCLINATION"
            ]
        )

        inclination_b = float(
            orbital_data_indexed.loc[
                row["NORAD_B"],
                "INCLINATION"
            ]
        )

        inclination_difference = abs(
            inclination_a - inclination_b
        )

        # Use the smaller angular separation.
        inclination_difference = min(
            inclination_difference,
            180.0 - inclination_difference
        )

        # ========================================================
        # ALTITUDE DIFFERENCE
        # ========================================================
        #
        # Kepler's third law:
        #
        #       n = sqrt(mu / a^3)
        #
        # Therefore:
        #
        #       a = (mu / n^2)^(1/3)
        #
        # Mean motion is supplied in revolutions/day.
        #

        EARTH_MU_KM3_S2 = 398600.4418
        EARTH_RADIUS_KM = 6378.137

        mean_motion_a = float(
            orbital_data_indexed.loc[
                row["NORAD_A"],
                "MEAN_MOTION"
            ]
        )

        mean_motion_b = float(
            orbital_data_indexed.loc[
                row["NORAD_B"],
                "MEAN_MOTION"
            ]
        )

        if mean_motion_a <= 0 or mean_motion_b <= 0:
            return (
                inclination_difference,
                np.nan
            )

        # Convert revolutions/day to radians/second.

        mean_motion_a_rad_s = (
            mean_motion_a
            * 2.0
            * np.pi
            / 86400.0
        )

        mean_motion_b_rad_s = (
            mean_motion_b
            * 2.0
            * np.pi
            / 86400.0
        )

        # Semi-major axis.

        semi_major_axis_a = (
            EARTH_MU_KM3_S2
            / (
                mean_motion_a_rad_s ** 2
            )
        ) ** (1.0 / 3.0)

        semi_major_axis_b = (
            EARTH_MU_KM3_S2
            / (
                mean_motion_b_rad_s ** 2
            )
        ) ** (1.0 / 3.0)

        # Approximate orbital altitude.

        altitude_a = (
            semi_major_axis_a
            - EARTH_RADIUS_KM
        )

        altitude_b = (
            semi_major_axis_b
            - EARTH_RADIUS_KM
        )

        altitude_difference = abs(
            altitude_a - altitude_b
        )

        return (
            inclination_difference,
            altitude_difference
        )

    except (
        KeyError,
        ValueError,
        TypeError
    ):

        return np.nan, np.nan


def run_monte_carlo(
    conjunctions_df,
    propagated_df,
    orbital_data_df=None,
    verbose=True
):
    """
    Run Monte Carlo collision-probability estimation
    for every conjunction.

    Output columns include:

        COLLISION_PROBABILITY_MC
        MC_COLLISION_HITS
        MC_SAMPLES
        MC_UPPER_95_PROBABILITY
        SIGMA_A_KM
        SIGMA_B_KM
        INCLINATION_DIFFERENCE_DEG
        ALTITUDE_DIFFERENCE_KM
    """

    results = conjunctions_df.copy()

    probabilities = []
    hits_list = []
    samples_list = []
    upper_95_list = []

    sigma_a_list = []
    sigma_b_list = []

    inclination_difference_list = []
    altitude_difference_list = []

    orbital_data_indexed = None

    if orbital_data_df is not None:

        orbital_data_indexed = (
            orbital_data_df
            .set_index("NORAD_CAT_ID")
        )

    # ============================================================
    # PROCESS EACH CONJUNCTION
    # ============================================================

    for _, row in conjunctions_df.iterrows():

        # --------------------------------------------------------
        # NOMINAL POSITIONS AT TCA
        # --------------------------------------------------------

        pos_a = get_position_at_tca(
            propagated_df,
            row["NORAD_A"],
            row["TCA"]
        )

        pos_b = get_position_at_tca(
            propagated_df,
            row["NORAD_B"],
            row["TCA"]
        )

        # --------------------------------------------------------
        # AGE-SCALED POSITION UNCERTAINTY
        # --------------------------------------------------------

        if orbital_data_indexed is not None:

            sigma_a, sigma_b = get_pair_sigmas(
                row,
                orbital_data_indexed
            )

        else:

            default_sigma = getattr(
                config,
                "POSITION_UNCERTAINTY_KM",
                1.0
            )

            sigma_a = default_sigma
            sigma_b = default_sigma

        sigma_a = float(sigma_a)
        sigma_b = float(sigma_b)

        # --------------------------------------------------------
        # MONTE CARLO COLLISION PROBABILITY
        # --------------------------------------------------------

        (
            probability,
            hits,
            n,
            upper_95_probability
        ) = estimate_collision_probability(
            pos_a,
            pos_b,
            sigma_a_km=sigma_a,
            sigma_b_km=sigma_b
        )

        probabilities.append(
            probability
        )

        hits_list.append(
            hits
        )

        samples_list.append(
            n
        )

        upper_95_list.append(
            upper_95_probability
        )

        sigma_a_list.append(
            sigma_a
        )

        sigma_b_list.append(
            sigma_b
        )

        # --------------------------------------------------------
        # ORBITAL GEOMETRY
        # --------------------------------------------------------

        (
            inclination_difference,
            altitude_difference
        ) = calculate_orbital_geometry(
            row,
            orbital_data_indexed
        )

        inclination_difference_list.append(
            inclination_difference
        )

        altitude_difference_list.append(
            altitude_difference
        )

        # --------------------------------------------------------
        # CONSOLE OUTPUT
        # --------------------------------------------------------

        if verbose:

            print(
                f"[MC] "
                f"{row['OBJECT_A']} vs "
                f"{row['OBJECT_B']}: "
                f"P={probability:.6e} "
                f"({hits}/{n} samples), "
                f"P95_upper={upper_95_probability:.6e}, "
                f"sigma_a={sigma_a:.2f}km, "
                f"sigma_b={sigma_b:.2f}km, "
                f"delta_i={inclination_difference:.2f}deg, "
                f"delta_alt={altitude_difference:.2f}km"
            )

    # ============================================================
    # SAVE RESULTS TO DATAFRAME
    # ============================================================

    results["SIGMA_A_KM"] = (
        sigma_a_list
    )

    results["SIGMA_B_KM"] = (
        sigma_b_list
    )

    results["MC_COLLISION_HITS"] = (
        hits_list
    )

    results["MC_SAMPLES"] = (
        samples_list
    )

    results["COLLISION_PROBABILITY_MC"] = (
        probabilities
    )

    results["MC_UPPER_95_PROBABILITY"] = (
        upper_95_list
    )

    results["INCLINATION_DIFFERENCE_DEG"] = (
        inclination_difference_list
    )

    results["ALTITUDE_DIFFERENCE_KM"] = (
        altitude_difference_list
    )

    return results


def run_and_save(
    output_path=None
):

    output_path = (
        output_path
        or (
            config.RESULTS_DIR
            / "monte_carlo_results.csv"
        )
    )

    config.ensure_dirs()

    from modules import data_loader

    # ============================================================
    # LOAD CONJUNCTIONS
    # ============================================================

    conjunctions_df = pd.read_csv(
        config.CONJUNCTIONS_FILE,
        parse_dates=["TCA"]
    )

    # ============================================================
    # LOAD PROPAGATED POSITIONS
    # ============================================================

    propagated_df = pd.read_csv(
        config.PROPAGATED_GRID_FILE,
        parse_dates=["TIME"]
    )

    if len(conjunctions_df) == 0:

        print(
            "No conjunctions to process - "
            "run conjunction_detection first, "
            "or lower SCREENING_DISTANCE_KM."
        )

        return None

    # ============================================================
    # LOAD ORBITAL DATA
    # ============================================================

    try:

        orbital_data_df = (
            data_loader.load_orbital_data()
        )

    except Exception as e:

        print(
            "Could not load orbital data for "
            f"age-scaled uncertainty/orbital geometry "
            f"({e}) - falling back to fixed uncertainty."
        )

        orbital_data_df = None

    # ============================================================
    # RUN MONTE CARLO
    # ============================================================

    results = run_monte_carlo(
        conjunctions_df,
        propagated_df,
        orbital_data_df
    )

    # ============================================================
    # SAVE CSV
    # ============================================================

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults written: {output_path}"
    )

    print()

    print(
        results[
            [
                "OBJECT_A",
                "OBJECT_B",
                "MISS_DISTANCE_KM",
                "RELATIVE_VELOCITY_KM_S",
                "SIGMA_A_KM",
                "SIGMA_B_KM",
                "INCLINATION_DIFFERENCE_DEG",
                "ALTITUDE_DIFFERENCE_KM",
                "MC_COLLISION_HITS",
                "MC_SAMPLES",
                "COLLISION_PROBABILITY_MC",
                "MC_UPPER_95_PROBABILITY"
            ]
        ].to_string(index=False)
    )

    return results


if __name__ == "__main__":
    run_and_save()