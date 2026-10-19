from pathlib import Path

import pandas as pd
import pytest

from modules.build_tle_availability import build_availability


def test_build_availability_uses_earliest_creation_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "gp_history.csv"
    output = tmp_path / "availability.csv"
    pd.DataFrame(
        [
            {"NORAD_CAT_ID": 23606, "EPOCH": "1996-07-23T09:25:32Z", "CREATION_DATE": "1996-07-23T10:00:00Z"},
            {"NORAD_CAT_ID": 23606, "EPOCH": "1996-07-23T09:25:32Z", "CREATION_DATE": "1996-07-23T09:50:00Z"},
            {"NORAD_CAT_ID": 18208, "EPOCH": "1996-07-23T09:30:00Z", "CREATION_DATE": "1996-07-23T09:55:00Z"},
        ]
    ).to_csv(source, index=False)

    result = build_availability(source, output)

    assert list(result.columns) == ["NORAD_CAT_ID", "TLE_EPOCH", "AVAILABLE_AT_UTC"]
    assert len(result) == 2
    cerise = result[result["NORAD_CAT_ID"] == 23606].iloc[0]
    assert cerise["AVAILABLE_AT_UTC"] == "1996-07-23T09:50:00Z"
    assert output.is_file()


def test_build_availability_rejects_missing_source_columns(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text("NORAD_CAT_ID,EPOCH\n23606,1996-07-23T09:25:32Z\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CREATION_DATE"):
        build_availability(source, tmp_path / "out.csv")


def test_build_availability_rejects_noninteger_catalog_id(tmp_path: Path) -> None:
    source = tmp_path / "bad_id.csv"
    source.write_text(
        "NORAD_CAT_ID,EPOCH,CREATION_DATE\n23606.5,1996-07-23T09:25:32Z,1996-07-23T09:50:00Z\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integer"):
        build_availability(source, tmp_path / "out.csv")
