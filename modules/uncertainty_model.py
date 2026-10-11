"""
Configurable age-scaled uncertainty model for ORION-X.

IMPORTANT SCIENTIFIC LIMITATION:
Public TLEs do not contain object-specific state covariance. Unless validated
covariance or historical residual calibration is supplied, this module produces
a scenario estimate, not an operationally validated collision probability.

Environment variables:
  ORION_SIGMA_BASE_KM             Base 1-sigma position error in km (default 1.0)
  ORION_SIGMA_GROWTH_KM_PER_DAY   Assumed 1-sigma growth in km/day (default 1.5)

These defaults preserve the previous model for reproducibility. They must not be
described as empirically calibrated unless residual validation has actually
been performed on archived, time-separated orbital solutions.
"""
import os
import pandas as pd

# Explicitly configurable assumptions. Keep values finite and positive/nonnegative.
def _env_float(name, default, minimum):
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else float(default)
    except (TypeError, ValueError):
        value = float(default)
    if not pd.notna(value) or value < minimum or value == float("inf"):
        value = float(default)
    return value


SIGMA_BASE_KM = _env_float("ORION_SIGMA_BASE_KM", 1.0, 0.001)
GROWTH_RATE_KM_PER_DAY = _env_float("ORION_SIGMA_GROWTH_KM_PER_DAY", 1.5, 0.0)
UNCERTAINTY_MODEL_NAME = "AGE_SCALED_ISOTROPIC_ASSUMPTION"
UNCERTAINTY_MODEL_VALIDATED = False
UNCERTAINTY_MODEL_NOTE = (
    "Assumed age-scaled isotropic sigma; not object-specific covariance. "
    "Defaults are illustrative and not empirically calibrated."
)


def age_scaled_sigma(epoch_str, reference_time, sigma_base=None, growth_rate=None):
    """Return assumed 1-sigma position uncertainty (km) at reference_time."""
    sigma_base = sigma_base if sigma_base is not None else SIGMA_BASE_KM
    growth_rate = growth_rate if growth_rate is not None else GROWTH_RATE_KM_PER_DAY
    sigma_base = float(sigma_base)
    growth_rate = float(growth_rate)
    if not pd.notna(sigma_base) or sigma_base <= 0:
        raise ValueError("sigma_base must be finite and positive.")
    if not pd.notna(growth_rate) or growth_rate < 0:
        raise ValueError("growth_rate must be finite and non-negative.")

    epoch = pd.to_datetime(epoch_str, utc=True)
    reference_time = pd.Timestamp(reference_time)
    if reference_time.tzinfo is None:
        reference_time = reference_time.tz_localize("UTC")
    days_since_epoch = max(0.0, (reference_time - epoch).total_seconds() / 86400.0)
    return sigma_base + growth_rate * days_since_epoch


def combined_sigma(sigma_a_km, sigma_b_km):
    """Combine independent scalar 1-sigma assumptions in quadrature (km)."""
    return (float(sigma_a_km) ** 2 + float(sigma_b_km) ** 2) ** 0.5


def get_pair_sigmas(row, orbital_data_indexed):
    """Return each object's assumed sigma at TCA using its own TLE epoch."""
    epoch_a = orbital_data_indexed.loc[row["NORAD_A"], "EPOCH"]
    epoch_b = orbital_data_indexed.loc[row["NORAD_B"], "EPOCH"]
    sigma_a = age_scaled_sigma(epoch_a, row["TCA"])
    sigma_b = age_scaled_sigma(epoch_b, row["TCA"])
    return sigma_a, sigma_b
