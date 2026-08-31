"""
Age-scaled position uncertainty: instead of a single fixed sigma applied
to every object, each object's uncertainty grows with how much time has
elapsed since its own TLE epoch, at the actual moment of closest approach
(TCA) — not just at the start of the propagation window. This reflects a
real, well-documented property of SGP4 (accuracy degrades with time since
epoch) rather than an arbitrary knob, directly addressing the isotropic-
fixed-sigma limitation already flagged in the statistical treatment doc.
"""

import pandas as pd

import config


# Linear growth model: sigma(t) = sigma_base + growth_rate * days_since_epoch.
# These are illustrative, citable-order-of-magnitude values (real SGP4 error
# growth is regime- and object-dependent) — state them explicitly as a
# modeling choice in your methodology, not as a precisely validated figure.
SIGMA_BASE_KM = 1.0
GROWTH_RATE_KM_PER_DAY = 1.5


def age_scaled_sigma(epoch_str, reference_time, sigma_base=None, growth_rate=None):
    """
    Position uncertainty (km) for one object, scaled by time since its TLE
    epoch as of `reference_time` (typically the conjunction's TCA).
    """
    sigma_base = sigma_base if sigma_base is not None else SIGMA_BASE_KM
    growth_rate = growth_rate if growth_rate is not None else GROWTH_RATE_KM_PER_DAY

    epoch = pd.to_datetime(epoch_str, utc=True)
    reference_time = pd.Timestamp(reference_time)
    if reference_time.tzinfo is None:
        reference_time = reference_time.tz_localize("UTC")

    days_since_epoch = max(0.0, (reference_time - epoch).total_seconds() / 86400.0)
    return sigma_base + growth_rate * days_since_epoch


def combined_sigma(sigma_a_km, sigma_b_km):
    """
    Combined relative-position uncertainty for two INDEPENDENT objects with
    (possibly different) individual sigmas. Generalizes the earlier fixed
    sqrt(2)*sigma case (which assumed sigma_a == sigma_b) to the age-scaled,
    generally-unequal case.
    """
    return (sigma_a_km ** 2 + sigma_b_km ** 2) ** 0.5


def get_pair_sigmas(row, orbital_data_indexed):
    """
    Look up each object's age-scaled sigma at TCA, given a conjunction row
    (with NORAD_A, NORAD_B, TCA) and orbital_data indexed by NORAD_CAT_ID.
    """
    epoch_a = orbital_data_indexed.loc[row["NORAD_A"], "EPOCH"]
    epoch_b = orbital_data_indexed.loc[row["NORAD_B"], "EPOCH"]

    sigma_a = age_scaled_sigma(epoch_a, row["TCA"])
    sigma_b = age_scaled_sigma(epoch_b, row["TCA"])

    return sigma_a, sigma_b