"""Combine Space-Track GP_HISTORY CSV exports for a historical event.

This creates a record-level table keyed by NORAD_CAT_ID + TLE epoch.
CREATION_DATE is the Space-Track record creation/publication timestamp; it
does not prove when an operator received the record or that a matching record
was actually used by the replay. Keep source CSVs and query/retrieval notes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"NORAD_CAT_ID", "EPOCH", "CREATION_DATE"}


def prepare_pair_availability(
    input_csvs: list[Path],
    output_csv: Path,
    report_json: Path | None = None,
    expected_norad_ids: tuple[int, ...] = (24946, 22675),
) -> tuple[pd.DataFrame, dict]:
    if not input_csvs:
        raise ValueError("Provide at least one Space-Track GP_HISTORY CSV.")

    frames: list[pd.DataFrame] = []
    source_reports = []
    for path in input_csvs:
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        source = pd.read_csv(path, low_memory=False)
        missing = sorted(REQUIRED_COLUMNS - set(source.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        frame = source[["NORAD_CAT_ID", "EPOCH", "CREATION_DATE"]].copy()
        frame["NORAD_CAT_ID"] = pd.to_numeric(frame["NORAD_CAT_ID"], errors="raise")
        if frame["NORAD_CAT_ID"].isna().any() or (frame["NORAD_CAT_ID"] <= 0).any():
            raise ValueError(f"{path}: NORAD_CAT_ID must contain positive integer IDs.")
        if ((frame["NORAD_CAT_ID"] % 1) != 0).any():
            raise ValueError(f"{path}: NORAD_CAT_ID must contain integer IDs.")
        frame["NORAD_CAT_ID"] = frame["NORAD_CAT_ID"].astype("int64")
        frame["TLE_EPOCH"] = pd.to_datetime(frame["EPOCH"], utc=True, errors="raise", format="mixed")
        frame["AVAILABLE_AT_UTC"] = pd.to_datetime(
            frame["CREATION_DATE"], utc=True, errors="raise", format="mixed"
        )
        if frame[["TLE_EPOCH", "AVAILABLE_AT_UTC"]].isna().any().any():
            raise ValueError(f"{path}: EPOCH and CREATION_DATE must not be empty.")

        frames.append(frame[["NORAD_CAT_ID", "TLE_EPOCH", "AVAILABLE_AT_UTC"]])
        source_reports.append({
            "file": str(path),
            "rows": int(len(frame)),
            "norad_ids": sorted(int(v) for v in frame["NORAD_CAT_ID"].unique()),
        })

    combined = pd.concat(frames, ignore_index=True)
    observed = set(int(v) for v in combined["NORAD_CAT_ID"].unique())
    missing_ids = sorted(set(expected_norad_ids) - observed)
    if missing_ids:
        raise ValueError(
            f"Missing required NORAD IDs {missing_ids}; observed IDs: {sorted(observed)}. "
            "Do not mark the event provenance verified until both objects are represented."
        )

    result = (
        combined.groupby(["NORAD_CAT_ID", "TLE_EPOCH"], as_index=False)["AVAILABLE_AT_UTC"]
        .min()
        .sort_values(["NORAD_CAT_ID", "TLE_EPOCH"])
        .reset_index(drop=True)
    )
    result["TLE_EPOCH"] = result["TLE_EPOCH"].map(
        lambda value: value.isoformat().replace("+00:00", "Z")
    )
    result["AVAILABLE_AT_UTC"] = result["AVAILABLE_AT_UTC"].map(
        lambda value: value.isoformat().replace("+00:00", "Z")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    report = {
        "status": "PREPARED_NOT_VERIFIED",
        "source": "Space-Track GP_HISTORY CSV export(s)",
        "source_timestamp_field": "CREATION_DATE",
        "timestamp_interpretation": (
            "Space-Track record creation/publication timestamp; not proof of operator receipt "
            "or independent proof that this exact element record was used in the replay."
        ),
        "expected_norad_ids": list(expected_norad_ids),
        "observed_norad_ids": sorted(observed),
        "rows_after_deduplication": int(len(result)),
        "objects": {},
        "input_files": source_reports,
        "availability_csv": str(output_csv),
        "verification_requirements": [
            "Retain original source CSV exports and document each query and retrieval time.",
            "Confirm selected NORAD_CAT_ID + TLE_EPOCH records match the exact TLEs used in the replay.",
            "Keep provenance status NOT_VERIFIED until the evidence and limitations have been reviewed.",
        ],
    }
    for norad in expected_norad_ids:
        obj = result[result["NORAD_CAT_ID"] == norad]
        if obj.empty:
            continue
        epochs = pd.to_datetime(obj["TLE_EPOCH"], utc=True)
        created = pd.to_datetime(obj["AVAILABLE_AT_UTC"], utc=True)
        report["objects"][str(norad)] = {
            "rows": int(len(obj)),
            "earliest_tle_epoch_utc": epochs.min().isoformat().replace("+00:00", "Z"),
            "latest_tle_epoch_utc": epochs.max().isoformat().replace("+00:00", "Z"),
            "earliest_creation_date_utc": created.min().isoformat().replace("+00:00", "Z"),
            "latest_creation_date_utc": created.max().isoformat().replace("+00:00", "Z"),
        }

    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, action="append", required=True,
                        help="Space-Track GP_HISTORY CSV; repeat once per downloaded object history.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--expected-norad-ids", type=int, nargs="+", default=[24946, 22675])
    args = parser.parse_args()
    result, report = prepare_pair_availability(
        args.input_csv, args.output_csv, args.report_json, tuple(args.expected_norad_ids)
    )
    print(json.dumps({
        "status": report["status"],
        "rows_after_deduplication": len(result),
        "observed_norad_ids": report["observed_norad_ids"],
        "availability_csv": str(args.output_csv),
        "report_json": str(args.report_json) if args.report_json else None,
    }, indent=2))


if __name__ == "__main__":
    main()
