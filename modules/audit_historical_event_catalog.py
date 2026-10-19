"""Audit the historical-event catalog without inventing missing orbital archives.

This is a readiness inventory, not a validation run. An event is runnable only
when its catalog metadata is valid and its event-specific directory contains
at least one plausible archived TLE/2LE/3LE file. Actual NORAD coverage and
epoch coverage are checked by the replay engine when the event is run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "event_id", "event_name", "event_time_utc", "object_a_norad",
    "object_b_norad", "confirmation", "status_for_backtest",
}
ARCHIVE_SUFFIXES = {".tle", ".2le", ".3le", ".txt"}


def audit_event_catalog(catalog_path: Path, historical_root: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path, dtype=str).fillna("")
    missing = sorted(REQUIRED_COLUMNS - set(catalog.columns))
    if missing:
        raise ValueError(f"Event catalog is missing required columns: {', '.join(missing)}")

    rows = []
    for _, event in catalog.iterrows():
        event_id = event["event_id"].strip()
        event_dir = historical_root / event_id.lower()
        archives = sorted(
            p for p in event_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in ARCHIVE_SUFFIXES
        ) if event_dir.is_dir() else []
        try:
            event_time = pd.Timestamp(event["event_time_utc"])
            time_valid = not pd.isna(event_time)
        except (TypeError, ValueError):
            time_valid = False

        try:
            norad_a = int(event["object_a_norad"])
            norad_b = int(event["object_b_norad"])
            norad_valid = norad_a > 0 and norad_b > 0 and norad_a != norad_b
        except (TypeError, ValueError):
            norad_valid = False

        confirmed = event["confirmation"].strip().lower() == "confirmed"
        if not event_id or not time_valid or not norad_valid:
            readiness = "INVALID_CATALOG_METADATA"
        elif not confirmed:
            readiness = "NOT_CONFIRMED_DO_NOT_USE_AS_PRIMARY"
        elif not archives:
            readiness = "NEEDS_ARCHIVED_TLE_DATA"
        else:
            readiness = "ARCHIVE_FILE_PRESENT_REQUIRES_CONTENT_CHECK"

        rows.append({
            "EVENT_ID": event_id,
            "EVENT_NAME": event["event_name"],
            "EVENT_TIME_UTC": event["event_time_utc"],
            "NORAD_A": event["object_a_norad"],
            "NORAD_B": event["object_b_norad"],
            "CONFIRMATION": event["confirmation"],
            "CATALOG_STATUS": event["status_for_backtest"],
            "ARCHIVE_DIRECTORY": str(event_dir.as_posix()),
            "ARCHIVE_FILE_COUNT": len(archives),
            "ARCHIVE_FILES": ";".join(str(p.as_posix()) for p in archives),
            "READINESS": readiness,
            "NOTE": (
                "File presence does not verify object IDs, historical epoch coverage, "
                "or publication/availability timestamps."
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/historical_events/events.csv"))
    parser.add_argument("--historical-root", type=Path, default=Path("data/historical_events"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/historical/event_catalog_readiness.csv"))
    args = parser.parse_args()

    report = audit_event_catalog(args.catalog, args.historical_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_csv, index=False)
    print(report[["EVENT_ID", "ARCHIVE_FILE_COUNT", "READINESS"]].to_string(index=False))
    print(f"\nReadiness report: {args.output_csv}")
    print("This inventory does not certify archive contents or operational data availability.")


if __name__ == "__main__":
    main()
