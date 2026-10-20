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


def test_probability_estimator_summary_separates_pre_event_rows_and_estimator_noise():
    import pandas as pd

    from modules.audit_historical_validation import summarize_probability_estimators

    frame = pd.DataFrame({
        "PREDICTION_TIME_IS_PRE_EVENT": [True, True, False],
        "ANALYTIC_RISK_LEVEL": ["LOW", "LOW", "LOW"],
        "QAE_RISK_LEVEL": ["LOW", "LOW", "LOW"],
        "MC_RISK_LEVEL": ["LOW", "LOW", "MEDIUM"],
        "QAE_ABS_ERROR_FROM_ANALYTIC_PC": [1e-8, 3e-8, 2e-8],
        "MC_CI_CONTAINS_ANALYTIC_PC": [True, False, True],
    })

    summary = summarize_probability_estimators(frame)

    assert summary["all_rows"]["analytic_risk_counts"] == {"LOW": 3}
    assert summary["pre_event_rows"]["row_count"] == 2
    assert summary["pre_event_rows"]["mc_risk_counts"] == {"LOW": 2}
    assert summary["pre_event_rows"]["qae_absolute_error_from_analytic_pc"]["max"] == 3e-8
    assert summary["pre_event_rows"]["mc_ci_contains_analytic_pc"]["contains_analytic_pc"] == 1
    assert summary["pre_event_rows"]["mc_ci_contains_analytic_pc"]["fraction"] == 0.5


def test_proximity_and_probability_disagreement_is_reported_without_relabeling():
    import pandas as pd

    from modules.audit_historical_validation import summarize_proximity_probability_disagreement

    frame = pd.DataFrame({
        "PREDICTION_TIME_IS_PRE_EVENT": [True, True, True, False],
        "FORECAST_PROXIMITY_ALERT_LEVEL": ["HIGH", "MEDIUM", "LOW", "HIGH"],
        "ANALYTIC_RISK_LEVEL": ["LOW", "LOW", "LOW", "HIGH"],
    })

    summary = summarize_proximity_probability_disagreement(frame)

    assert summary["pre_event_row_count"] == 3
    assert summary["high_proximity_with_low_analytic_probability_count"] == 1
    assert summary["medium_or_high_proximity_with_low_analytic_probability_count"] == 2
    assert summary["proximity_vs_analytic_risk_cross_tab"]["HIGH"]["LOW"] == 1
    assert "not proof that the event is safe" in summary["interpretation"]
