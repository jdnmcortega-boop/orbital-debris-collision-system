import pandas as pd

from modules.audit_historical_validation import summarize_collision_geometry


def test_collision_geometry_inside_hard_body_radius():
    df = pd.DataFrame({
        "FORECAST_MISS_DISTANCE_KM": [0.01],
        "HARD_BODY_RADIUS_KM": [0.02],
    })

    result = summarize_collision_geometry(df)

    assert result["available"] is True
    assert result["rows_with_forecast_miss_within_hard_body_radius"] == 1
    assert result["minimum_miss_to_radius_ratio"] == 0.5


def test_collision_geometry_outside_hard_body_radius():
    df = pd.DataFrame({
        "FORECAST_MISS_DISTANCE_KM": [2.479],
        "HARD_BODY_RADIUS_KM": [0.02],
    })

    result = summarize_collision_geometry(df)

    assert result["available"] is True
    assert result["rows_with_forecast_miss_within_hard_body_radius"] == 0
    assert result["minimum_miss_to_radius_ratio"] > 100


def test_collision_geometry_missing_radius_fails_closed():
    df = pd.DataFrame({
        "FORECAST_MISS_DISTANCE_KM": [0.01],
    })

    result = summarize_collision_geometry(df)

    assert result["available"] is False
    assert "HARD_BODY_RADIUS_KM" in result["missing_columns"]
