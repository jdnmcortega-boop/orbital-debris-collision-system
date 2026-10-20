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


def test_historical_validation_probability_audit_columns_are_explicit():
    # QAE estimates the same Pc target; it is not an independent orbit predictor.
    from pathlib import Path

    source = Path("modules/historical_validation.py").read_text(encoding="utf-8")
    for column in (
        '"ASSUMED_POSITION_SIGMA_PER_OBJECT_KM"',
        '"COMBINED_ISOTROPIC_SIGMA_KM"',
        '"HARD_BODY_RADIUS_KM"',
        '"QAE_RISK_LEVEL"',
        '"QAE_MINUS_ANALYTIC_PC"',
        '"QAE_ABS_ERROR_FROM_ANALYTIC_PC"',
        '"MC_CI_CONTAINS_ANALYTIC_PC"',
    ):
        assert column in source
