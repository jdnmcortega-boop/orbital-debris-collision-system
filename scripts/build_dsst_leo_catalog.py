#!/usr/bin/env python3
"""Build a reproducible LEO-only GP/OMM catalog for the DSST experiment.

This downloads CelesTrak's ACTIVE GP group at most once per two-hour cache
window, preserves the original payload, and derives a transparent LEO filter.
It does NOT convert SGP4 mean elements into DSST initial states.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dsst_experiment"
RAW = OUT / "raw"
RAW_CSV = RAW / "celestrak_active_gp.csv"
MANIFEST = OUT / "source_manifest.json"
CATALOG = OUT / "objects_10000.csv"
AUDIT = OUT / "catalog_audit.csv"
SOURCE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=ACTIVE&FORMAT=CSV"
CACHE_SECONDS = 2 * 60 * 60
EARTH_MU_KM3_S2 = 398600.4418
EARTH_MEAN_RADIUS_KM = 6371.0088
LEO_APOGEE_LIMIT_KM = 2000.0

FIELDS = [
    "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "EPOCH",
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
    "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
    "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS_KM", "PERIGEE_ALTITUDE_KM", "APOGEE_ALTITUDE_KM",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_float(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing {key}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {key}")
    return number


def fetch_source() -> tuple[bytes, str, bool]:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        age = (utc_now().timestamp() - RAW_CSV.stat().st_mtime)
        if age < CACHE_SECONDS:
            stamp = datetime.fromtimestamp(RAW_CSV.stat().st_mtime, timezone.utc).isoformat()
            return RAW_CSV.read_bytes(), stamp, True

    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "ORION-X-DSST-Research/1.0 (catalog acquisition; cached)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"CelesTrak returned HTTP {response.status}")
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"CelesTrak HTTP {exc.code}; request stopped without automatic retry. {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not retrieve CelesTrak GP data: {exc}") from exc

    # Validate before replacing the cache.
    decoded = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames or "NORAD_CAT_ID" not in reader.fieldnames:
        raise RuntimeError("CelesTrak response did not contain the expected OMM CSV header.")
    rows = list(reader)
    if not rows:
        raise RuntimeError("CelesTrak returned an empty ACTIVE GP catalog.")
    RAW_CSV.write_bytes(payload)
    retrieved = utc_now().isoformat()
    return payload, retrieved, False


def build(limit: int = 10000) -> dict:
    payload, retrieved_at, cache_used = fetch_source()
    text = payload.decode("utf-8-sig")
    source_rows = list(csv.DictReader(io.StringIO(text)))
    seen: set[str] = set()
    selected: list[dict] = []
    audit = {
        "source_rows": len(source_rows),
        "duplicate_catalog_ids": 0,
        "invalid_or_incomplete_rows": 0,
        "outside_leo_orbit": 0,
        "valid_leo_rows": 0,
    }
    for row in source_rows:
        cat_id = str(row.get("NORAD_CAT_ID", "")).strip()
        if not cat_id or cat_id in seen:
            audit["duplicate_catalog_ids"] += 1
            continue
        seen.add(cat_id)
        try:
            mean_motion = parse_float(row, "MEAN_MOTION")
            eccentricity = parse_float(row, "ECCENTRICITY")
            if mean_motion <= 0 or not 0 <= eccentricity < 1:
                raise ValueError("invalid orbital range")
            mean_motion_rad_s = mean_motion * 2.0 * math.pi / 86400.0
            semi_major = (EARTH_MU_KM3_S2 / (mean_motion_rad_s ** 2)) ** (1.0 / 3.0)
            perigee = semi_major * (1.0 - eccentricity) - EARTH_MEAN_RADIUS_KM
            apogee = semi_major * (1.0 + eccentricity) - EARTH_MEAN_RADIUS_KM
            if perigee <= 0 or apogee >= LEO_APOGEE_LIMIT_KM:
                audit["outside_leo_orbit"] += 1
                continue
            normalized = {key: (row.get(key) or "").strip() for key in FIELDS[:13]}
            normalized.update({
                "SEMIMAJOR_AXIS_KM": f"{semi_major:.6f}",
                "PERIGEE_ALTITUDE_KM": f"{perigee:.6f}",
                "APOGEE_ALTITUDE_KM": f"{apogee:.6f}",
            })
            selected.append(normalized)
        except (TypeError, ValueError, OverflowError):
            audit["invalid_or_incomplete_rows"] += 1

    audit["valid_leo_rows"] = len(selected)
    selected.sort(key=lambda item: int(item["NORAD_CAT_ID"]) if item["NORAD_CAT_ID"].isdigit() else item["NORAD_CAT_ID"])
    selected = selected[:limit]

    OUT.mkdir(parents=True, exist_ok=True)
    with CATALOG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)

    with AUDIT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows({"metric": key, "value": value} for key, value in audit.items())

    manifest = {
        "dataset": "ORION-X DSST experimental LEO GP catalog",
        "source": "CelesTrak ACTIVE GP group",
        "source_url": SOURCE_URL,
        "source_format": "CCSDS OMM-compatible CSV",
        "retrieved_at_utc": retrieved_at,
        "cache_used": cache_used,
        "cache_policy_hours": 2,
        "filter": {
            "orbit_class": "LEO",
            "perigee_altitude_km_exclusive_min": 0,
            "apogee_altitude_km_exclusive_max": LEO_APOGEE_LIMIT_KM,
            "altitude_reference": "Earth mean radius",
            "earth_mu_km3_s2": EARTH_MU_KM3_S2,
            "earth_mean_radius_km": EARTH_MEAN_RADIUS_KM,
            "derivation": "two-body semimajor axis derived from mean motion; apsides derived using eccentricity",
        },
        "requested_object_limit": limit,
        "selected_object_count": len(selected),
        "source_audit": audit,
        "important_limitation": (
            "CelesTrak GP OMM elements use the SGP4 mean-element theory. "
            "This catalog is not a DSST-ready osculating-state dataset. "
            "A separately validated initialization/conversion step is required."
        ),
        "raw_file": str(RAW_CSV.relative_to(ROOT)),
        "normalized_file": str(CATALOG.relative_to(ROOT)),
        "audit_file": str(AUDIT.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10000, help="Maximum number of LEO objects to retain (default: 10000).")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        result = build(args.limit)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if result["selected_object_count"] < args.limit:
        print(
            f"WARNING: only {result['selected_object_count']} qualifying LEO objects "
            f"were available; the dataset was not padded to {args.limit}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
