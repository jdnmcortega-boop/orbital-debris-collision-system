"""Analytical encounter-plane collision probability for live conjunctions.

The output is a research estimate based on an isotropic, age-scaled position
uncertainty model. TLEs do not provide covariance matrices, so these sigmas
must not be described as measured covariance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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


def build_analytic_pc_results(conjunctions, orbital_data_df):
    """Enrich screened conjunctions and calculate analytical Pc for each pair."""
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
            if not np.isfinite(sigma_a) or not np.isfinite(sigma_b) or sigma_a <= 0 or sigma_b <= 0:
                raise ValueError("Modeled positional uncertainty is invalid.")

            miss = float(event["MISS_DISTANCE_KM"])
            if not np.isfinite(miss) or miss < 0:
                raise ValueError("Miss distance is invalid.")

            pc = analytic_collision_probability(
                miss,
                sigma_a_km=sigma_a,
                sigma_b_km=sigma_b,
                hard_body_radius_km=float(getattr(config, "HARD_BODY_RADIUS_KM", 0.02)),
            )
            row["SIGMA_A_KM"] = sigma_a
            row["SIGMA_B_KM"] = sigma_b
            row["ANALYTIC_PC"] = float(pc)
            row["PROBABILITY_METHOD"] = "2D isotropic analytical encounter-plane model"
            row["UNCERTAINTY_SOURCE"] = "Assumed age-scaled isotropic sigma; not measured covariance"
            row["PROBABILITY_STATUS"] = "MODELED_ESTIMATE"

            a = catalog.loc[norad_a]
            b = catalog.loc[norad_b]
            axis_a = _semi_major_axis_km(a["MEAN_MOTION"])
            axis_b = _semi_major_axis_km(b["MEAN_MOTION"])
            row["ALTITUDE_DIFFERENCE_KM"] = abs(axis_a - axis_b) if np.isfinite(axis_a) and np.isfinite(axis_b) else np.nan
            row["INCLINATION_DIFFERENCE_DEG"] = abs(float(a["INCLINATION"]) - float(b["INCLINATION"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            row["SIGMA_A_KM"] = np.nan
            row["SIGMA_B_KM"] = np.nan
            row["ANALYTIC_PC"] = np.nan
            row["PROBABILITY_METHOD"] = "2D isotropic analytical encounter-plane model"
            row["UNCERTAINTY_SOURCE"] = "Unavailable"
            row["PROBABILITY_STATUS"] = f"NOT_CALCULATED: {exc}"
            row["ALTITUDE_DIFFERENCE_KM"] = np.nan
            row["INCLINATION_DIFFERENCE_DEG"] = np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    result["ANALYTIC_PC"] = pd.to_numeric(result["ANALYTIC_PC"], errors="coerce")
    return result
