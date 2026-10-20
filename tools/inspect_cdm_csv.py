"""Inspect a Space-Track / CCSDS CDM CSV for ORION-X import compatibility.

Usage:
    python tools/inspect_cdm_csv.py "C:/path/to/cdm_public.csv"
    python tools/inspect_cdm_csv.py "C:/path/to/cdm_public.csv" --rows 3

Writes cdm_diagnostic.txt beside the input file. Inspection is local only;
the CSV is not uploaded or transmitted by this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

KEYWORDS = (
    "TCA", "PC", "PROB", "SAT_1", "SAT_2", "SAT1", "SAT2",
    "OBJECT1", "OBJECT2", "NORAD", "CATALOG", "DESIGNATOR",
    "CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N",
    "COV", "SIGMA", "UNCERTAINT", "MISS_DISTANCE", "MIN_RNG",
    "RELATIVE_SPEED",
)
COVARIANCE_KEYS = ("CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N")


def normalize_column(value: object) -> str:
    """Normalize a column name the same way as the CDM parser."""
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def supported_covariance_column(columns, sat: int, key: str):
    """Return a column recognized by modules.cdm_covariance, if present."""
    normalized = {normalize_column(c): c for c in columns}
    candidates = (
        f"SAT{sat}_{key}", f"SAT_{sat}_{key}",
        f"SAT{sat}_COVARIANCE_{key}", f"SAT_{sat}_COVARIANCE_{key}",
        f"OBJECT{sat}_{key}", f"OBJECT_{sat}_{key}",
        f"OBJECT{sat}_COVARIANCE_{key}", f"OBJECT_{sat}_COVARIANCE_{key}",
        f"COVARIANCE_{sat}_{key}", f"{key}_{sat}",
    )
    return next((normalized[c] for c in candidates if c in normalized), None)


def build_diagnostic(frame: pd.DataFrame, filename: str, rows: int = 3) -> str:
    """Build a diagnostic that distinguishes absent fields from unsupported names."""
    lines: list[str] = [
        "ORION-X CDM CSV DIAGNOSTIC",
        f"File: {Path(filename).name}",
        f"Rows: {len(frame)}",
        f"Columns: {len(frame.columns)}",
        "",
        "=== ALL COLUMN NAMES ===",
    ]
    lines.extend(f"{i + 1}. {col}" for i, col in enumerate(frame.columns))

    selected = [
        col for col in frame.columns
        if any(key in normalize_column(col) for key in KEYWORDS)
    ]
    lines.extend(["", "=== RELEVANT COLUMN TYPES / NON-EMPTY COUNTS ==="])
    if selected:
        lines.extend(
            f"{col}: dtype={frame[col].dtype}; non_empty={int(frame[col].notna().sum())}/{len(frame)}"
            for col in selected
        )
    else:
        lines.append("No columns matched common CDM identifiers, probability, or uncertainty names.")

    lines.extend(["", f"=== FIRST {max(rows, 0)} SAMPLE ROW(S), RELEVANT COLUMNS ==="])
    if selected and rows > 0:
        lines.append(frame[selected].head(rows).to_string(index=False, max_cols=100, max_colwidth=60))
    else:
        lines.append("(No sample rows requested or no relevant columns found.)")

    normalized = {normalize_column(c): c for c in frame.columns}
    lines.extend(["", "=== QUICK FIELD CHECK ==="])
    checks = {
        "TCA": ("TCA", "TIME_OF_CLOSEST_APPROACH"),
        "reported Pc": ("PC", "COLLISION_PROBABILITY", "PROBABILITY_OF_COLLISION"),
        "SAT 1 object ID": ("SAT_1_ID", "SAT_1_OBJECT_DESIGNATOR", "SAT_1_NORAD_ID", "SAT1_ID"),
        "SAT 2 object ID": ("SAT_2_ID", "SAT_2_OBJECT_DESIGNATOR", "SAT_2_NORAD_ID", "SAT2_ID"),
    }
    for label, candidates in checks.items():
        found = next((normalized[c] for c in candidates if c in normalized), None)
        lines.append(f"{label}: {found if found is not None else 'NOT FOUND'}")

    lines.extend(["", "=== POSITION COVARIANCE FIELD CHECK ==="])
    supported_count = 0
    for sat in (1, 2):
        found_for_object = []
        for key in COVARIANCE_KEYS:
            col = supported_covariance_column(frame.columns, sat, key)
            if col is not None:
                supported_count += 1
                populated = int(frame[col].notna().sum())
                found_for_object.append(f"{key} -> {col} (non-empty {populated}/{len(frame)})")
        lines.append(
            f"Object {sat}: " + ("; ".join(found_for_object) if found_for_object else "NO SUPPORTED RTN POSITION-COVARIANCE FIELDS")
        )

    covariance_like = [
        col for col in frame.columns
        if any(token in normalize_column(col) for token in ("COV", "CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N", "SIGMA", "UNCERTAINT"))
    ]
    lines.append(f"Parser-supported covariance component columns found: {supported_count}/12")
    if covariance_like:
        lines.append("All covariance/uncertainty-like column names:")
        lines.extend(f"- {col}" for col in covariance_like)
    else:
        lines.append("No covariance/uncertainty-like column names detected at all.")

    if supported_count == 0 and covariance_like:
        lines.append(
            "DIAGNOSIS: uncertainty-like fields exist, but none match the parser's supported RTN component aliases. "
            "This suggests a possible parser/schema compatibility issue; inspect the listed names before changing the parser."
        )
    elif supported_count == 0:
        lines.append(
            "DIAGNOSIS: this CSV has no recognized covariance component fields and no covariance/uncertainty-like "
            "column names. It appears to be a Pc/reference export rather than a covariance-bearing flattened CDM. "
            "A reported Pc cannot reconstruct the missing covariance."
        )
    elif supported_count < 12:
        lines.append(
            "DIAGNOSIS: some covariance components are present, but the expected six components for both objects "
            "are incomplete. Check per-column non-empty counts and each object's validation status."
        )
    else:
        lines.append(
            "DIAGNOSIS: all 12 expected component columns are recognized by name. This does not prove values are "
            "valid; run the ORION-X row-level covariance validation for completeness and positive-semidefinite checks."
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the CDM CSV downloaded from Space-Track")
    parser.add_argument("--rows", type=int, default=3, help="Number of sample rows to include (default: 3)")
    args = parser.parse_args()

    path = Path(args.csv_path).expanduser()
    if not path.is_file():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 2

    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"ERROR: Could not read CSV: {exc}", file=sys.stderr)
        return 2

    report = build_diagnostic(frame, path.name, args.rows)
    output = path.with_name("cdm_diagnostic.txt")
    try:
        output.write_text(report, encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: Could not write report: {exc}", file=sys.stderr)
        return 2

    print("\n".join(report.splitlines()[:8]))
    print("")
    print(f"Full diagnostic saved to: {output}")
    print("Share the covariance-field check and diagnosis sections here; you can omit the local file path.")
    return 0


