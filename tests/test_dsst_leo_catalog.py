"""Offline tests for the DSST LEO catalog preparation script."""
from __future__ import annotations

import csv
import io
import json
import math

from scripts import build_dsst_leo_catalog as builder


def _mean_motion_for_altitude(altitude_km: float) -> float:
    radius = builder.EARTH_MEAN_RADIUS_KM + altitude_km
    return math.sqrt(builder.EARTH_MU_KM3_S2 / radius**3) * 86400.0 / (2.0 * math.pi)


def _row(cat_id: str, altitude_km: float = 400.0) -> dict[str, str]:
    return {
        "OBJECT_NAME": f"TEST OBJECT {cat_id}",
        "OBJECT_ID": "2020-001A",
        "NORAD_CAT_ID": cat_id,
        "EPOCH": "2026-10-30T00:00:00.000000",
        "MEAN_MOTION": str(_mean_motion_for_altitude(altitude_km)),
        "ECCENTRICITY": "0.001",
        "INCLINATION": "51.6",
        "RA_OF_ASC_NODE": "10",
        "ARG_OF_PERICENTER": "20",
        "MEAN_ANOMALY": "30",
        "BSTAR": "0.00001",
        "MEAN_MOTION_DOT": "0",
        "MEAN_MOTION_DDOT": "0",
    }


def _csv_payload(rows: list[dict[str, str]]) -> bytes:
    fields = list(rows[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def test_build_keeps_only_valid_leo_and_writes_audit(tmp_path, monkeypatch):
    rows = [
        _row("10001", 400),
        _row("10002", 2100),  # outside LEO apogee limit
        _row("10003", 500),
        _row("10001", 400),  # duplicate catalog ID
        {**_row("10004"), "ECCENTRICITY": "not-a-number"},
    ]
    payload = _csv_payload(rows)

    monkeypatch.setattr(builder, "OUT", tmp_path)
    monkeypatch.setattr(builder, "RAW", tmp_path / "raw")
    monkeypatch.setattr(builder, "RAW_CSV", tmp_path / "raw" / "source.csv")
    monkeypatch.setattr(builder, "MANIFEST", tmp_path / "source_manifest.json")
    monkeypatch.setattr(builder, "CATALOG", tmp_path / "objects_10000.csv")
    monkeypatch.setattr(builder, "AUDIT", tmp_path / "catalog_audit.csv")
    monkeypatch.setattr(
        builder,
        "fetch_source",
        lambda: (payload, "2026-10-30T00:00:00+00:00", False),
    )

    result = builder.build(limit=10000)

    with (tmp_path / "objects_10000.csv").open(newline="", encoding="utf-8") as handle:
        catalog = list(csv.DictReader(handle))
    assert [row["NORAD_CAT_ID"] for row in catalog] == ["10001", "10003"]
    assert result["selected_object_count"] == 2
    assert result["source_audit"]["source_rows"] == 5
    assert result["source_audit"]["duplicate_catalog_ids"] == 1
    assert result["source_audit"]["outside_leo_orbit"] == 1
    assert result["source_audit"]["invalid_or_incomplete_rows"] == 1

    manifest = json.loads((tmp_path / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["filter"]["orbit_class"] == "LEO"
    assert manifest["selected_object_count"] == 2


def test_build_respects_requested_limit(tmp_path, monkeypatch):
    payload = _csv_payload([_row("10011"), _row("10012"), _row("10013")])
    monkeypatch.setattr(builder, "OUT", tmp_path)
    monkeypatch.setattr(builder, "RAW", tmp_path / "raw")
    monkeypatch.setattr(builder, "RAW_CSV", tmp_path / "raw" / "source.csv")
    monkeypatch.setattr(builder, "MANIFEST", tmp_path / "source_manifest.json")
    monkeypatch.setattr(builder, "CATALOG", tmp_path / "objects_10000.csv")
    monkeypatch.setattr(builder, "AUDIT", tmp_path / "catalog_audit.csv")
    monkeypatch.setattr(
        builder,
        "fetch_source",
        lambda: (payload, "2026-10-30T00:00:00+00:00", False),
    )

    result = builder.build(limit=2)
    assert result["selected_object_count"] == 2


def test_invalid_mean_motion_is_rejected():
    row = _row("10021")
    row["MEAN_MOTION"] = "0"
    try:
        builder.parse_float(row, "MEAN_MOTION")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid mean motion to be rejected by range validation")
