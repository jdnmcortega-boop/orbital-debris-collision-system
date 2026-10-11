"""Analytical encounter-plane collision probability for screened conjunctions.

The model assumes independent isotropic Gaussian position errors because the
public TLE input does not include a full state covariance. Its sigma values are
explicit modeling assumptions, not measured covariance. This module records
the maximum probability possible under the same assumed sigma (at zero miss
distance), so a HIGH threshold that the uncertainty model cannot possibly
reach is visible rather than silently appearing as a classification bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ncx2

import config
from modules.qae import analytic_collision_probability
from modules.uncertainty_model import get_pair_sigmas

EARTH_MU_KM3_S2 = 398600.4418
SECONDS_PER_DAY = 86400.0


def _semi_major_axis_km(mean_motion_rev_day):
    """Approximate semimajor axis from mean motion (two-body Kepler model)."""
    n = float(mean_motion_rev_day) * 2.0 * np.pi / SECONDS_PER_DAY
    if not np.isfinite(n) or n <= 0.0:
        return np.nan
    return float((EARTH_MU_KM3_S2 / (n * n)) ** (1.0 / 3.0))


def _pc_max_at_zero_miss(combined_sigma_km, hard_body_radius_km):
    """Maximum Pc possible for the assumed 2-D isotropic Gaussian model."""
    sigma = float(combined_sigma_km)
    radius = float(hard_body_radius_km)
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(radius) or radius <= 0.0:
        return np.nan
    # At zero miss distance, the 2-D central chi-square CDF has this closed form.
    return float(-np.expm1(-(radius * radius) / (2.0 * sigma * sigma)))


def build_analytic_pc_results(conjunctions, orbital_data_df):
    """Enrich refined conjunctions and calculate analytical Pc for each pair."""
    if conjunctions is None or conjunctions.empty:
        return pd.DataFrame()

    required_conjunction = {
        "OBJECT_A", "NORAD_A", "OBJECT_B", "NORAD_B", "TCA",
        "DAYS_TO_TCA", "FORECAST_HORIZON_DAYS", "MISS_DISTANCE_KM",
        "RELATIVE_VELOCITY_KM_S",
    }
    missing = sorted(required_conjunction - set(conjunctions.columns))
    if missing:
        raise ValueError("Conjunction data missing required columns: " + ", ".join(missing))
    if orbital_data_df is None or orbital_data_df.empty:
        raise ValueError("Orbital catalog is empty; cannot model position uncertainty.")

    catalog = orbital_data_df.copy()
    catalog["NORAD_CAT_ID"] = pd.to_numeric(catalog["NORAD_CAT_ID"], errors="coerce")
    catalog = catalog.dropna(subset=["NORAD_CAT_ID"]).copy()
    catalog["NORAD_CAT_ID"] = catalog["NORAD_CAT_ID"].astype("int64")
    catalog = catalog.drop_duplicates("NORAD_CAT_ID", keep="last").set_index("NORAD_CAT_ID")
    for column in ("EPOCH", "MEAN_MOTION", "INCLINATION"):
        if column not in catalog.columns:
            raise ValueError(f"Orbital catalog missing required column: {column}")

    hard_body_radius_km = float(getattr(config, "HARD_BODY_RADIUS_KM", 0.02))
    high_threshold = float(getattr(config, "RISK_THRESHOLD_HIGH", 1e-4))
    rows = []

    for _, event in conjunctions.iterrows():
        row = event.to_dict()
        try:
            norad_a = int(event["NORAD_A"])
            norad_b = int(event["NORAD_B"])
            if norad_a not in catalog.index or norad_b not in catalog.index:
                raise KeyError("One or both NORAD IDs are absent from the orbital catalog.")

            sigma_a, sigma_b = get_pair_sigmas(event, catalog)
            sigma_a, sigma_b = float(sigma_a), float(sigma_b)
            if (
                not np.isfinite(sigma_a) or not np.isfinite(sigma_b)
                or sigma_a <= 0.0 or sigma_b <= 0.0
            ):
                raise ValueError("Modeled positional uncertainty is invalid.")

            miss = float(event["MISS_DISTANCE_KM"])
            if not np.isfinite(miss) or miss < 0.0:
                raise ValueError("Refined miss distance is invalid.")

            combined_sigma = float(np.hypot(sigma_a, sigma_b))
            pc = analytic_collision_probability(
                miss,
                sigma_a_km=sigma_a,
                sigma_b_km=sigma_b,
                hard_body_radius_km=hard_body_radius_km,
            )
            # Keep a log diagnostic for underflow; it must not be used to inflate Pc.
            noncentrality = (miss / combined_sigma) ** 2
            radius_term = (hard_body_radius_km / combined_sigma) ** 2
            log_pc = float(ncx2.logcdf(radius_term, df=2, nc=noncentrality) / np.log(10.0))
            pc_max = _pc_max_at_zero_miss(combined_sigma, hard_body_radius_km)

            # Sensitivity-only scenarios show how Pc depends on the assumed
            # uncertainty scale. They do not replace the baseline probability
            # and are never used to assign the official collision-risk label.
            pc_half_sigma = analytic_collision_probability(
                miss,
                sigma_a_km=sigma_a * 0.5,
                sigma_b_km=sigma_b * 0.5,
                hard_body_radius_km=hard_body_radius_km,
            )
            pc_double_sigma = analytic_collision_probability(
                miss,
                sigma_a_km=sigma_a * 2.0,
                sigma_b_km=sigma_b * 2.0,
                hard_body_radius_km=hard_body_radius_km,
            )

            row["SIGMA_A_KM"] = sigma_a
            row["SIGMA_B_KM"] = sigma_b
            row["COMBINED_SIGMA_KM"] = combined_sigma
            row["ANALYTIC_PC"] = float(pc)
            row["LOG10_ANALYTIC_PC"] = log_pc
            row["ANALYTIC_PC_MAX_AT_ZERO_MISS"] = pc_max
            row["PC_SENSITIVITY_HALF_SIGMA"] = float(pc_half_sigma)
            row["PC_SENSITIVITY_DOUBLE_SIGMA"] = float(pc_double_sigma)
            row["PC_SENSITIVITY_NOTE"] = (
                "Illustrative sensitivity only; sigma scaled by 0.5x/2x, "
                "not measured covariance and not used for classification"
            )
            row["HIGH_PC_THRESHOLD"] = high_threshold
            row["HIGH_THRESHOLD_REACHABLE_UNDER_SIGMA"] = bool(
                np.isfinite(pc_max) and pc_max >= high_threshold
            )
            row["PROBABILITY_METHOD"] = "2D isotropic analytical encounter-plane model"
            row["UNCERTAINTY_SOURCE"] = (
                "Assumed age-scaled isotropic sigma; not measured covariance"
            )
            row["PROBABILITY_STATUS"] = (
                "MODELED_ESTIMATE"
                if pc_max >= high_threshold
                else "MODELED_ESTIMATE_HIGH_THRESHOLD_UNREACHABLE_UNDER_ASSUMED_SIGMA"
            )

            a = catalog.loc[norad_a]
            b = catalog.loc[norad_b]
            axis_a = _semi_major_axis_km(a["MEAN_MOTION"])
            axis_b = _semi_major_axis_km(b["MEAN_MOTION"])
            row["ALTITUDE_DIFFERENCE_KM"] = (
                abs(axis_a - axis_b)
                if np.isfinite(axis_a) and np.isfinite(axis_b)
                else np.nan
            )
            inclination_a = float(a["INCLINATION"])
            inclination_b = float(b["INCLINATION"])
            inclination_difference = abs(inclination_a - inclination_b)
            row["INCLINATION_DIFFERENCE_DEG"] = min(
                inclination_difference, 180.0 - inclination_difference
            )
        except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            row["SIGMA_A_KM"] = np.nan
            row["SIGMA_B_KM"] = np.nan
            row["COMBINED_SIGMA_KM"] = np.nan
            row["ANALYTIC_PC"] = np.nan
            row["LOG10_ANALYTIC_PC"] = np.nan
            row["ANALYTIC_PC_MAX_AT_ZERO_MISS"] = np.nan
            row["PC_SENSITIVITY_HALF_SIGMA"] = np.nan
            row["PC_SENSITIVITY_DOUBLE_SIGMA"] = np.nan
            row["PC_SENSITIVITY_NOTE"] = "Unavailable"
            row["HIGH_PC_THRESHOLD"] = high_threshold
            row["HIGH_THRESHOLD_REACHABLE_UNDER_SIGMA"] = False
            row["PROBABILITY_METHOD"] = "2D isotropic analytical encounter-plane model"
            row["UNCERTAINTY_SOURCE"] = "Unavailable"
            row["PROBABILITY_STATUS"] = f"NOT_CALCULATED: {exc}"
            row["ALTITUDE_DIFFERENCE_KM"] = np.nan
            row["INCLINATION_DIFFERENCE_DEG"] = np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    result["ANALYTIC_PC"] = pd.to_numeric(result["ANALYTIC_PC"], errors="coerce")
    return result
