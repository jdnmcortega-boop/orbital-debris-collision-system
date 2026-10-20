"""Tests for the historical operational-availability gate."""
from pathlib import Path

import pandas as pd

from modules.provenance_gate import assess_operational_availability


def sample_validation():
    return pd.DataFrame([{
        "SNAPSHOT_TIME": "2009-01-01T00:00:00Z",
        "NORAD_A": 100,
        "TLE_EPOCH_A": "2008-12-31T00:00:00Z",
        "NORAD_B": 200,
        "TLE_EPOCH_B": "2008-12-31T12:00:00Z",
    }])


def test_missing_availability_proof_is_not_verified(tmp_path: Path):
    rows, summary = assess_operational_availability(
        sample_validation(),
        {"status": "NOT_VERIFIED", "record_level_publication_timestamps_available": False},
        tmp_path,
    )
    assert summary["status"] == "NOT_VERIFIED"
    assert summary["eligible_rows"] == 0
    assert rows.loc[0, "OPERATIONAL_AVAILABILITY_STATUS"] == "NOT_VERIFIED"


def test_future_publication_makes_row_ineligible(tmp_path: Path):
    availability = tmp_path / "availability.csv"
    pd.DataFrame([
        {"NORAD_CAT_ID": 100, "TLE_EPOCH": "2008-12-31T00:00:00Z", "AVAILABLE_AT_UTC": "2009-01-01T00:01:00Z"},
        {"NORAD_CAT_ID": 200, "TLE_EPOCH": "2008-12-31T12:00:00Z", "AVAILABLE_AT_UTC": "2008-12-31T13:00:00Z"},
    ]).to_csv(availability, index=False)
    rows, summary = assess_operational_availability(
        sample_validation(),
        {
            "status": "VERIFIED",
            "record_level_publication_timestamps_available": True,
            "record_level_availability_file": "availability.csv",
        },
        tmp_path,
    )
    assert summary["status"] == "NOT_VERIFIED"
    assert rows.loc[0, "OPERATIONAL_AVAILABILITY_STATUS"] == "INELIGIBLE_FUTURE_AVAILABLE"


def test_both_records_available_before_snapshot_are_verified(tmp_path: Path):
    availability = tmp_path / "availability.csv"
    pd.DataFrame([
        {"NORAD_CAT_ID": 100, "TLE_EPOCH": "2008-12-31T00:00:00Z", "AVAILABLE_AT_UTC": "2008-12-31T01:00:00Z"},
        {"NORAD_CAT_ID": 200, "TLE_EPOCH": "2008-12-31T12:00:00Z", "AVAILABLE_AT_UTC": "2008-12-31T13:00:00Z"},
    ]).to_csv(availability, index=False)
    rows, summary = assess_operational_availability(
        sample_validation(),
        {
            "status": "VERIFIED",
            "record_level_publication_timestamps_available": True,
            "record_level_availability_file": "availability.csv",
        },
        tmp_path,
    )
    assert summary["status"] == "VERIFIED_FOR_ALL_ROWS"
    assert rows.loc[0, "OPERATIONAL_AVAILABILITY_STATUS"] == "VERIFIED_AVAILABLE_BY_SNAPSHOT"


def test_missing_one_exact_record_match_is_labeled_clearly(tmp_path: Path):
    availability = tmp_path / "availability.csv"
    pd.DataFrame([
        {"NORAD_CAT_ID": 100, "TLE_EPOCH": "2008-12-31T00:00:00Z", "AVAILABLE_AT_UTC": "2008-12-31T01:00:00Z"},
    ]).to_csv(availability, index=False)
    rows, summary = assess_operational_availability(
        sample_validation(),
        {
            "status": "NOT_VERIFIED",
            "record_level_publication_timestamps_available": False,
            "record_level_availability_file": "availability.csv",
        },
        tmp_path,
    )
    diagnostic = summary["spacetrack_creation_date_diagnostic"]
    assert diagnostic["rows_missing_one_or_both_exact_record_matches"] == 1
    assert rows.loc[0, "SPACETRACK_CREATION_DATE_MATCH_STATUS"] == (
        "MISSING_ONE_OR_BOTH_EXACT_RECORD_MATCHES"
    )
    assert summary["status"] == "NOT_VERIFIED"


def test_spacetrack_matches_are_diagnostic_only_when_manifest_unverified(tmp_path: Path):
    availability = tmp_path / "availability.csv"
    pd.DataFrame([
        {"NORAD_CAT_ID": 100, "TLE_EPOCH": "2008-12-31T00:00:00Z", "AVAILABLE_AT_UTC": "2008-12-31T01:00:00Z"},
        {"NORAD_CAT_ID": 200, "TLE_EPOCH": "2008-12-31T12:00:00Z", "AVAILABLE_AT_UTC": "2008-12-31T13:00:00Z"},
    ]).to_csv(availability, index=False)
    rows, summary = assess_operational_availability(
        sample_validation(),
        {
            "status": "NOT_VERIFIED",
            "record_level_publication_timestamps_available": False,
            "record_level_availability_file": "availability.csv",
        },
        tmp_path,
    )
    diagnostic = summary["spacetrack_creation_date_diagnostic"]
    assert diagnostic["status"] == "DIAGNOSTIC_ONLY_NOT_VERIFIED"
    assert diagnostic["rows_with_both_exact_record_matches"] == 1
    assert diagnostic["rows_both_creation_dates_on_or_before_snapshot"] == 1
    assert summary["status"] == "NOT_VERIFIED"
    assert rows.loc[0, "OPERATIONAL_AVAILABILITY_STATUS"] == "NOT_VERIFIED"
    assert rows.loc[0, "SPACETRACK_CREATION_DATE_MATCH_STATUS"] == (
        "BOTH_CREATION_DATES_ON_OR_BEFORE_SNAPSHOT_NOT_AVAILABILITY_PROOF"
    )
