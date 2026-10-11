"""Inspect a Space-Track / CCSDS CDM CSV for ORION-X import compatibility.

Usage:
    python tools/inspect_cdm_csv.py "C:/path/to/cdm_public.csv"
    python tools/inspect_cdm_csv.py "C:/path/to/cdm_public.csv" --rows 3

Writes a compact cdm_diagnostic.txt beside the input file. It does not upload
or transmit the CSV; all inspection happens locally.
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
    "COVARIANCE", "MISS_DISTANCE", "MIN_RNG", "RELATIVE_SPEED",
)


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

    lines: list[str] = []
    lines.append("ORION-X CDM CSV DIAGNOSTIC")
    lines.append(f"File: {path.name}")
    lines.append(f"Rows: {len(frame)}")
    lines.append(f"Columns: {len(frame.columns)}")
    lines.append("")
    lines.append("=== ALL COLUMN NAMES ===")
    lines.extend(f"{i + 1}. {col}" for i, col in enumerate(frame.columns))

    selected = [
        col for col in frame.columns
        if any(key in re.sub(r"[^A-Z0-9_]", "_", str(col).upper()) for key in KEYWORDS)
    ]
    lines.append("")
    lines.append("=== RELEVANT COLUMN TYPES ===")
    if selected:
        lines.extend(f"{col}: {frame[col].dtype}" for col in selected)
    else:
        lines.append("No columns matched the usual CDM identifiers, TCA, Pc, or covariance names.")

    lines.append("")
    lines.append(f"=== FIRST {max(args.rows, 0)} SAMPLE ROW(S), RELEVANT COLUMNS ===")
    if selected and args.rows > 0:
        lines.append(frame[selected].head(args.rows).to_string(index=False, max_cols=80, max_colwidth=45))
    else:
        lines.append("(No sample rows requested or no relevant columns found.)")

    normalized = {re.sub(r"[^A-Z0-9]+", "_", str(c).upper()).strip("_"): c for c in frame.columns}
    lines.append("")
    lines.append("=== QUICK FIELD CHECK ===")
    checks = {
        "TCA": ("TCA", "TIME_OF_CLOSEST_APPROACH"),
        "reported Pc": ("PC", "COLLISION_PROBABILITY", "PROBABILITY_OF_COLLISION"),
        "SAT 1 object ID": ("SAT_1_ID", "SAT_1_OBJECT_DESIGNATOR", "SAT_1_NORAD_ID", "SAT1_ID"),
        "SAT 2 object ID": ("SAT_2_ID", "SAT_2_OBJECT_DESIGNATOR", "SAT_2_NORAD_ID", "SAT2_ID"),
    }
    for label, candidates in checks.items():
        found = next((normalized[c] for c in candidates if c in normalized), None)
        lines.append(f"{label}: {found if found is not None else 'NOT FOUND'}")

    output = path.with_name("cdm_diagnostic.txt")
    try:
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: Could not write report: {exc}", file=sys.stderr)
        return 2

    print("\n".join(lines[:8]))
    print("")
    print(f"Full diagnostic saved to: {output}")
    print("Open cdm_diagnostic.txt and share its contents here. You can omit any personal file paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
