"""Build a record-level TLE availability table from Space-Track GP_HISTORY CSV.

Input must preserve the source NORAD_CAT_ID, EPOCH, and CREATION_DATE fields.
CREATION_DATE is treated as the source publication/availability timestamp; it
must not be replaced with the element epoch or the local download time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"NORAD_CAT_ID", "EPOCH", "CREATION_DATE"}


def build_availability(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    source = pd.read_csv(input_csv)
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    rows = source[["NORAD_CAT_ID", "EPOCH", "CREATION_DATE"]].copy()
    rows["NORAD_CAT_ID"] = pd.to_numeric(rows["NORAD_CAT_ID"], errors="raise")
    if rows["NORAD_CAT_ID"].isna().any() or (rows["NORAD_CAT_ID"] <= 0).any():
        raise ValueError("NORAD_CAT_ID must contain positive integer catalog IDs.")
    if ((rows["NORAD_CAT_ID"] % 1) != 0).any():
        raise ValueError("NORAD_CAT_ID must contain positive integer catalog IDs.")
    rows["NORAD_CAT_ID"] = rows["NORAD_CAT_ID"].astype("int64")

    rows["TLE_EPOCH"] = pd.to_datetime(rows["EPOCH"], utc=True, errors="raise", format="mixed")
    rows["AVAILABLE_AT_UTC"] = pd.to_datetime(
        rows["CREATION_DATE"], utc=True, errors="raise", format="mixed"
    )
    if rows[["TLE_EPOCH", "AVAILABLE_AT_UTC"]].isna().any().any():
        raise ValueError("EPOCH and CREATION_DATE must not be empty.")

    result = (
        rows.groupby(["NORAD_CAT_ID", "TLE_EPOCH"], as_index=False)["AVAILABLE_AT_UTC"]
        .min()
        .sort_values(["NORAD_CAT_ID", "TLE_EPOCH"])
        .reset_index(drop=True)
    )
    result["TLE_EPOCH"] = result["TLE_EPOCH"].map(lambda value: value.isoformat().replace("+00:00", "Z"))
    result["AVAILABLE_AT_UTC"] = result["AVAILABLE_AT_UTC"].map(
        lambda value: value.isoformat().replace("+00:00", "Z")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Space-Track GP_HISTORY CSV export")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output columns: NORAD_CAT_ID,TLE_EPOCH,AVAILABLE_AT_UTC")
    args = parser.parse_args()
    result = build_availability(args.input_csv, args.output_csv)
    print(f"Availability rows written: {len(result)}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
