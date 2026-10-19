import numpy as np

from modules.prediction import (
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    probability_model_diagnostics,
)


def test_configured_uncertainty_reports_unreachable_probability_thresholds():
    diagnostic = probability_model_diagnostics()

    radius = 0.02
    sigma = 10.0
    expected_max = -np.expm1(-(radius**2) / (4.0 * sigma**2))

    assert np.isclose(diagnostic["PC_MAX_AT_ZERO_MISS"], expected_max)
    assert diagnostic["PC_MEDIUM_THRESHOLD_REACHABLE"] is False
    assert diagnostic["PC_HIGH_THRESHOLD_REACHABLE"] is False
    assert diagnostic["PC_MODEL_DIAGNOSTIC"] == (
        "BOTH_MEDIUM_AND_HIGH_THRESHOLDS_UNREACHABLE"
    )


def test_diagnostics_reflect_lower_explicit_uncertainty_without_changing_thresholds():
    diagnostic = probability_model_diagnostics(0.5, 0.5)

    assert diagnostic["PC_MAX_AT_ZERO_MISS"] > RISK_THRESHOLD_HIGH
    assert diagnostic["PC_MEDIUM_THRESHOLD_REACHABLE"] is True
    assert diagnostic["PC_HIGH_THRESHOLD_REACHABLE"] is True
    assert diagnostic["PC_UNCERTAINTY_SOURCE"] == "ROW_OBJECT_SIGMAS"
