from pathlib import Path

import pandas as pd
import pytest

from modules.prepare_spacetrack_provenance import prepare_pair_availability


def test_combines_two_object_exports_and_deduplicates_earliest_creation(tmp_path: Path):
    iridium = tmp_path / "iridium.csv"
    cosmos = tmp_path / "cosmos.csv"
    output = tmp_path / "record_level_availability.csv"
    report_path = tmp_path / "report.json"

    pd.DataFrame([
        {"NORAD_CAT_ID": 24946, "EPOCH": "2009-01-10T00:00:00Z", "CREATION_DATE": "2009-01-10T01:00:00Z"},
        {"NORAD_CAT_ID": 24946, "EPOCH": "2009-01-10T00:00:00Z", "CREATION_DATE": "2009-01-10T00:45:00Z"},
    ]).to_csv(iridium, index=False)
    pd.DataFrame([
        {"NORAD_CAT_ID": 22675, "EPOCH": "2009-01-10T12:00:00Z", "CREATION_DATE": "2009-01-10T12:30:00Z"},
    ]).to_csv(cosmos, index=False)

    result, report = prepare_pair_availability(
        [iridium, cosmos], output, report_path
    )

    assert len(result) == 2
    assert set(result["NORAD_CAT_ID"]) == {24946, 22675}
    row = result[result["NORAD_CAT_ID"] == 24946].iloc[0]
    assert row["AVAILABLE_AT_UTC"] == "2009-01-10T00:45:00Z"
    assert report["status"] == "PREPARED_NOT_VERIFIED"
    assert report_path.is_file()


def test_rejects_missing_required_object(tmp_path: Path):
    only_iridium = tmp_path / "iridium.csv"
    pd.DataFrame([
        {"NORAD_CAT_ID": 24946, "EPOCH": "2009-01-10T00:00:00Z", "CREATION_DATE": "2009-01-10T01:00:00Z"},
    ]).to_csv(only_iridium, index=False)

    with pytest.raises(ValueError, match="Missing required NORAD IDs"):
        prepare_pair_availability(
            [only_iridium], tmp_path / "availability.csv"
        )


def test_rejects_missing_columns(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("NORAD_CAT_ID,EPOCH\n24946,2009-01-10T00:00:00Z\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CREATION_DATE"):
        prepare_pair_availability([bad], tmp_path / "availability.csv")
