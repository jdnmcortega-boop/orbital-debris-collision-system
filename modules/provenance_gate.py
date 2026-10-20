"""Operational-availability gate for historical TLE replay outputs.

TLE epoch is not publication time. This gate only marks a row operationally
eligible when an explicit, sourced, record-level availability table proves
both selected TLEs were available no later than the replay snapshot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


AVAILABILITY_COLUMNS = {"NORAD_CAT_ID", "TLE_EPOCH", "AVAILABLE_AT_UTC"}


def assess_operational_availability(validation: pd.DataFrame, manifest: dict, repo_root: Path) -> tuple[pd.DataFrame, dict]:
    """Return conservative availability status plus a non-verifying Space-Track diagnostic."""
    rows = validation.copy()
    rows["OPERATIONAL_AVAILABILITY_STATUS"] = "NOT_VERIFIED"
    rows["OPERATIONAL_AVAILABILITY_REASON"] = (
        "No verified record-level first-availability timestamps supplied."
    )

    availability_rel = manifest.get("record_level_availability_file")
    availability_path = (repo_root / availability_rel).resolve() if availability_rel else None
    rows["SPACETRACK_CREATION_DATE_MATCH_STATUS"] = "NO_AVAILABILITY_TABLE"
    rows["SPACETRACK_CREATION_DATE_A_UTC"] = None
    rows["SPACETRACK_CREATION_DATE_B_UTC"] = None
    diagnostic_summary = {
        "status": "NOT_RUN",
        "availability_file": str(availability_rel) if availability_rel else None,
        "rows_with_both_exact_record_matches": 0,
        "rows_both_creation_dates_on_or_before_snapshot": 0,
        "rows_with_creation_date_after_snapshot": 0,
        "rows_missing_one_or_both_exact_record_matches": int(len(rows)),
        "interpretation": (
            "Space-Track CREATION_DATE is a source-record timestamp, not proof of "
            "operator receipt, first availability, or that the record was used in the replay. "
            "This diagnostic does not verify operational availability or predictive skill."
        ),
    }

    # This table is diagnostic only. Never use CREATION_DATE alone to upgrade the manifest.
    if availability_path is not None and availability_path.is_file():
        diagnostic = pd.read_csv(availability_path)
        missing = sorted(AVAILABILITY_COLUMNS - set(diagnostic.columns))
        if missing:
            raise ValueError(f"Availability file missing required columns: {missing}")
        diagnostic["NORAD_CAT_ID"] = pd.to_numeric(
            diagnostic["NORAD_CAT_ID"], errors="raise"
        ).astype(int)
        diagnostic["TLE_EPOCH"] = pd.to_datetime(
            diagnostic["TLE_EPOCH"], utc=True, errors="raise", format="ISO8601"
        )
        diagnostic["AVAILABLE_AT_UTC"] = pd.to_datetime(
            diagnostic["AVAILABLE_AT_UTC"], utc=True, errors="raise", format="ISO8601"
        )
        if diagnostic.duplicated(["NORAD_CAT_ID", "TLE_EPOCH"]).any():
            raise ValueError("Availability file contains duplicate NORAD_CAT_ID + TLE_EPOCH keys.")
        lookup_diagnostic = {
            (int(r.NORAD_CAT_ID), r.TLE_EPOCH): r.AVAILABLE_AT_UTC
            for r in diagnostic.itertuples(index=False)
        }
        parsed_rows = rows.copy()
        for column in ("SNAPSHOT_TIME", "TLE_EPOCH_A", "TLE_EPOCH_B"):
            parsed_rows[column] = pd.to_datetime(
                parsed_rows[column], utc=True, errors="raise", format="ISO8601"
            )
        statuses, timestamp_a, timestamp_b = [], [], []
        for r in parsed_rows.itertuples(index=False):
            time_a = lookup_diagnostic.get((int(r.NORAD_A), r.TLE_EPOCH_A))
            time_b = lookup_diagnostic.get((int(r.NORAD_B), r.TLE_EPOCH_B))
            timestamp_a.append(time_a.isoformat() if time_a is not None else None)
            timestamp_b.append(time_b.isoformat() if time_b is not None else None)
            if time_a is None or time_b is None:
                statuses.append("NO_EXACT_RECORD_MATCH_FOR_BOTH")
            elif time_a > r.SNAPSHOT_TIME or time_b > r.SNAPSHOT_TIME:
                statuses.append("AT_LEAST_ONE_CREATION_DATE_AFTER_SNAPSHOT")
            else:
                statuses.append(
                    "BOTH_CREATION_DATES_ON_OR_BEFORE_SNAPSHOT_NOT_AVAILABILITY_PROOF"
                )
        rows["SPACETRACK_CREATION_DATE_MATCH_STATUS"] = statuses
        rows["SPACETRACK_CREATION_DATE_A_UTC"] = timestamp_a
        rows["SPACETRACK_CREATION_DATE_B_UTC"] = timestamp_b
        counts = pd.Series(statuses).value_counts()
        after = int(counts.get("AT_LEAST_ONE_CREATION_DATE_AFTER_SNAPSHOT", 0))
        before = int(counts.get(
            "BOTH_CREATION_DATES_ON_OR_BEFORE_SNAPSHOT_NOT_AVAILABILITY_PROOF", 0
        ))
        missing_matches = int(counts.get("NO_EXACT_RECORD_MATCH_FOR_BOTH", 0))
        diagnostic_summary = {
            "status": "DIAGNOSTIC_ONLY_NOT_VERIFIED",
            "availability_file": str(availability_rel),
            "rows_with_both_exact_record_matches": after + before,
            "rows_both_creation_dates_on_or_before_snapshot": before,
            "rows_with_creation_date_after_snapshot": after,
            "rows_missing_one_or_both_exact_record_matches": missing_matches,
            "interpretation": (
                "Space-Track CREATION_DATE is a source-record timestamp, not proof of "
                "operator receipt, first availability, or that the record was used in the replay. "
                "This diagnostic does not verify operational availability or predictive skill."
            ),
        }

    eligible_manifest = (
        str(manifest.get("status", "")).upper() == "VERIFIED"
        and manifest.get("record_level_publication_timestamps_available") is True
        and bool(availability_rel)
    )
    if not eligible_manifest or availability_path is None or not availability_path.is_file():
        return rows, {
            "status": "NOT_VERIFIED",
            "eligible_rows": 0,
            "ineligible_rows": int(len(rows)),
            "availability_file": str(availability_rel) if availability_rel else None,
            "reason": (
                "Verified manifest status, timestamp declaration, and an existing "
                "record-level availability file are all required."
            ),
            "interpretation": (
                "TLE epoch is not proof of publication or operator availability. "
                "Do not describe this replay as operationally validated."
            ),
            "spacetrack_creation_date_diagnostic": diagnostic_summary,
        }

    # Only a manifest backed by independently reviewed availability evidence reaches here.
    availability = pd.read_csv(availability_path)
    missing = sorted(AVAILABILITY_COLUMNS - set(availability.columns))
    if missing:
        raise ValueError(f"Availability file missing required columns: {missing}")
    availability["NORAD_CAT_ID"] = pd.to_numeric(
        availability["NORAD_CAT_ID"], errors="raise"
    ).astype(int)
    availability["TLE_EPOCH"] = pd.to_datetime(
        availability["TLE_EPOCH"], utc=True, errors="raise", format="ISO8601"
    )
    availability["AVAILABLE_AT_UTC"] = pd.to_datetime(
        availability["AVAILABLE_AT_UTC"], utc=True, errors="raise", format="ISO8601"
    )
    if availability.duplicated(["NORAD_CAT_ID", "TLE_EPOCH"]).any():
        raise ValueError("Availability file contains duplicate NORAD_CAT_ID + TLE_EPOCH keys.")
    lookup = {
        (int(r.NORAD_CAT_ID), r.TLE_EPOCH): r.AVAILABLE_AT_UTC
        for r in availability.itertuples(index=False)
    }
    parsed_rows = rows.copy()
    for column in ("SNAPSHOT_TIME", "TLE_EPOCH_A", "TLE_EPOCH_B"):
        parsed_rows[column] = pd.to_datetime(
            parsed_rows[column], utc=True, errors="raise", format="ISO8601"
        )
    statuses, reasons = [], []
    for r in parsed_rows.itertuples(index=False):
        pair = (
            (int(r.NORAD_A), r.TLE_EPOCH_A),
            (int(r.NORAD_B), r.TLE_EPOCH_B),
        )
        times = [lookup.get(key) for key in pair]
        if any(t is None for t in times):
            statuses.append("NOT_VERIFIED")
            reasons.append("One or both selected TLE records lack a matching availability timestamp.")
        elif any(t > r.SNAPSHOT_TIME for t in times):
            statuses.append("INELIGIBLE_FUTURE_AVAILABLE")
            reasons.append("At least one selected TLE became available after the replay snapshot.")
        else:
            statuses.append("VERIFIED_AVAILABLE_BY_SNAPSHOT")
            reasons.append("Both selected TLE records have documented availability at or before the snapshot.")
    rows["OPERATIONAL_AVAILABILITY_STATUS"] = statuses
    rows["OPERATIONAL_AVAILABILITY_REASON"] = reasons
    eligible_count = int((rows["OPERATIONAL_AVAILABILITY_STATUS"] == "VERIFIED_AVAILABLE_BY_SNAPSHOT").sum())
    status = "VERIFIED_FOR_ALL_ROWS" if eligible_count == len(rows) else "PARTIALLY_VERIFIED" if eligible_count else "NOT_VERIFIED"
    return rows, {
        "status": status,
        "eligible_rows": eligible_count,
        "ineligible_rows": int(len(rows) - eligible_count),
        "availability_file": str(availability_rel),
        "interpretation": (
            "Eligibility verifies data availability only; it does not by itself establish "
            "forecast skill or collision-probability calibration."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-rows-csv", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    manifest = json.loads(args.provenance_manifest.read_text(encoding="utf-8"))
    validation = pd.read_csv(args.validation_csv)
    annotated, summary = assess_operational_availability(validation, manifest, args.repo_root)
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    audit["operational_availability_gate"] = summary
    audit["validation_interpretation"]["predictive_skill_claim_status"] = (
        "NOT_ESTABLISHED" if summary["status"] != "VERIFIED_FOR_ALL_ROWS" else "NOT_ESTABLISHED_PENDING_INDEPENDENT_BASELINE"
    )
    audit["validation_interpretation"]["reasons"].append(
        "Record-level TLE availability must be verified independently of element epoch; availability eligibility alone does not establish predictive skill."
    )
    args.audit_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    annotated.to_csv(args.audit_rows_csv, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
