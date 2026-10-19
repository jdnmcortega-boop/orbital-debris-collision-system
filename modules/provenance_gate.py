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
    """Return annotated validation rows and a conservative provenance summary."""
    rows = validation.copy()
    rows["OPERATIONAL_AVAILABILITY_STATUS"] = "NOT_VERIFIED"
    rows["OPERATIONAL_AVAILABILITY_REASON"] = (
        "No verified record-level first-availability timestamps supplied."
    )

    availability_rel = manifest.get("record_level_availability_file")
    eligible_manifest = (
        str(manifest.get("status", "")).upper() == "VERIFIED"
        and manifest.get("record_level_publication_timestamps_available") is True
        and bool(availability_rel)
    )
    availability_path = (repo_root / availability_rel).resolve() if availability_rel else None

    if not eligible_manifest or availability_path is None or not availability_path.is_file():
        return rows, {
            "status": "NOT_VERIFIED",
            "eligible_rows": 0,
            "ineligible_rows": int(len(rows)),
            "availability_file": str(availability_rel) if availability_rel else None,
            "reason": "Verified manifest status, timestamp declaration, and an existing record-level availability file are all required.",
            "interpretation": "TLE epoch is not proof of publication or operator availability. Do not describe this replay as operationally validated.",
        }

    availability = pd.read_csv(availability_path)
    missing = sorted(AVAILABILITY_COLUMNS - set(availability.columns))
    if missing:
        raise ValueError(f"Availability file missing required columns: {missing}")

    availability["NORAD_CAT_ID"] = pd.to_numeric(availability["NORAD_CAT_ID"], errors="raise").astype(int)
    availability["TLE_EPOCH"] = pd.to_datetime(availability["TLE_EPOCH"], utc=True, errors="raise", format="ISO8601")
    availability["AVAILABLE_AT_UTC"] = pd.to_datetime(availability["AVAILABLE_AT_UTC"], utc=True, errors="raise", format="ISO8601")
    if availability.duplicated(["NORAD_CAT_ID", "TLE_EPOCH"]).any():
        raise ValueError("Availability file contains duplicate NORAD_CAT_ID + TLE_EPOCH keys.")

    lookup = {
        (int(r.NORAD_CAT_ID), r.TLE_EPOCH): r.AVAILABLE_AT_UTC
        for r in availability.itertuples(index=False)
    }
    rows["SNAPSHOT_TIME"] = pd.to_datetime(rows["SNAPSHOT_TIME"], utc=True, errors="raise", format="ISO8601")
    rows["TLE_EPOCH_A"] = pd.to_datetime(rows["TLE_EPOCH_A"], utc=True, errors="raise", format="ISO8601")
    rows["TLE_EPOCH_B"] = pd.to_datetime(rows["TLE_EPOCH_B"], utc=True, errors="raise", format="ISO8601")

    statuses = []
    reasons = []
    for r in rows.itertuples(index=False):
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
        "interpretation": "Eligibility verifies data availability only; it does not by itself establish forecast skill or collision-probability calibration.",
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
