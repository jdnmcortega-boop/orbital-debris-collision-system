from pathlib import Path

import pandas as pd
import pytest

from modules.audit_historical_event_catalog import audit_event_catalog


def write_catalog(path: Path):
    pd.DataFrame([{
        "event_id": "TEST_EVENT",
        "event_name": "Test event",
        "event_time_utc": "2000-01-01T00:00:00Z",
        "object_a_norad": "100",
        "object_b_norad": "200",
        "confirmation": "confirmed",
        "status_for_backtest": "candidate",
    }]).to_csv(path, index=False)


def test_event_without_archive_is_not_runnable(tmp_path: Path):
    catalog = tmp_path / "events.csv"
    root = tmp_path / "historical"
    root.mkdir()
    write_catalog(catalog)
    report = audit_event_catalog(catalog, root)
    assert report.loc[0, "READINESS"] == "NEEDS_ARCHIVED_TLE_DATA"
    assert report.loc[0, "ARCHIVE_FILE_COUNT"] == 0


def test_event_with_archive_is_only_marked_for_content_check(tmp_path: Path):
    catalog = tmp_path / "events.csv"
    root = tmp_path / "historical"
    event_dir = root / "test_event"
    event_dir.mkdir(parents=True)
    write_catalog(catalog)
    (event_dir / "sample.tle").write_text("placeholder, not validated", encoding="utf-8")
    report = audit_event_catalog(catalog, root)
    assert report.loc[0, "READINESS"] == "ARCHIVE_FILE_PRESENT_REQUIRES_CONTENT_CHECK"


def test_invalid_catalog_columns_fail_closed(tmp_path: Path):
    catalog = tmp_path / "events.csv"
    catalog.write_text("event_id,event_name\nX,Y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        audit_event_catalog(catalog, tmp_path)
