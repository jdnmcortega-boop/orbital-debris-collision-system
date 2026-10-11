"""Import and validate covariance data from flattened CCSDS/Space-Track CDM CSV files.

This module does NOT calculate Pc. It establishes whether a CDM row can be
matched to an ORION-X conjunction and whether each object's 3x3 RTN position
covariance block is present, finite, symmetric by construction, and positive
semidefinite. A validated RTN block is not by itself sufficient to replace the
current Pc model: the full encounter-plane transformation, frame/time handling,
hard-body assumptions, and a verified CDM convention must also be implemented.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

COV_KEYS = ("CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N")
DEFAULT_TCA_TOLERANCE_MINUTES = 10.0


def _norm(value):
    return re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_")


def _find_column(columns, candidates):
    lookup = {_norm(c): c for c in columns}
    for candidate in candidates:
        if _norm(candidate) in lookup:
            return lookup[_norm(candidate)]
    return None


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _covariance_column(columns, sat, key):
    candidates = [
        f"SAT{sat}_{key}",
        f"OBJECT{sat}_{key}",
        f"OBJECT_{sat}_{key}",
        f"{key}_{sat}",
    ]
    return _find_column(columns, candidates)


def _object_id(row, columns, sat):
    candidates = [
        f"SAT{sat}_OBJECT_DESIGNATOR",
        f"SAT{sat}_OBJECT",
        f"OBJECT{sat}_OBJECT_DESIGNATOR",
        f"OBJECT{sat}_NORAD_CAT_ID",
        f"SAT{sat}_NORAD_CAT_ID",
        f"SAT{sat}_NORAD_ID",
    ]
    col = _find_column(columns, candidates)
    if col is None:
        return None
    value = row.get(col)
    if pd.isna(value):
        return None
    # Do not guess that an international designator is a NORAD catalog ID.
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text.upper()


def _object_covariance(row, columns, sat):
    values = {}
    missing = []
    for key in COV_KEYS:
        col = _covariance_column(columns, sat, key)
        if col is None:
            missing.append(key)
            values[key] = np.nan
        else:
            values[key] = _number(row.get(col))
            if not np.isfinite(values[key]):
                missing.append(key)
    if missing:
        return None, "MISSING_OR_INVALID_COVARIANCE_FIELDS:" + ",".join(missing)

    # CCSDS CDM CR_R, CT_R, CT_T, CN_R, CN_T, CN_N are RTN position
    # covariance entries in m^2. Convert the 3x3 position block to km^2.
    m2 = np.array([
        [values["CR_R"], values["CT_R"], values["CN_R"]],
        [values["CT_R"], values["CT_T"], values["CN_T"]],
        [values["CN_R"], values["CN_T"], values["CN_N"]],
    ], dtype=float)
    km2 = m2 / 1_000_000.0
    if not np.all(np.isfinite(km2)):
        return None, "NONFINITE_COVARIANCE"
    eigenvalues = np.linalg.eigvalsh(km2)
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-15)
    if float(eigenvalues.min()) < -1e-9 * scale:
        return None, "POSITION_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE"
    return {
        "matrix_km2": km2,
        "sigma_r_km": float(np.sqrt(max(km2[0, 0], 0.0))),
        "sigma_t_km": float(np.sqrt(max(km2[1, 1], 0.0))),
        "sigma_n_km": float(np.sqrt(max(km2[2, 2], 0.0))),
        "min_eigenvalue_km2": float(eigenvalues.min()),
    }, "VALID_POSITION_COVARIANCE_BLOCK"


def load_cdm_csv(source):
    """Load a flattened CDM CSV from a path or file-like upload."""
    if isinstance(source, (str, Path)):
        frame = pd.read_csv(source)
    else:
        frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("The CDM CSV contains no records.")
    tca_col = _find_column(frame.columns, ["TCA", "TIME_OF_CLOSEST_APPROACH"])
    if tca_col is None:
        raise ValueError("No TCA column found. Expected a flattened CDM CSV with a TCA field.")
    frame = frame.copy()
    frame["_CDM_TCA"] = pd.to_datetime(frame[tca_col], utc=True, errors="coerce")
    frame["_CDM_OBJECT1"] = frame.apply(lambda r: _object_id(r, frame.columns, 1), axis=1)
    frame["_CDM_OBJECT2"] = frame.apply(lambda r: _object_id(r, frame.columns, 2), axis=1)
    frame["_CDM_SOURCE_ROW"] = np.arange(len(frame)) + 1
    return frame


def validate_cdm_rows(cdm_frame):
    """Return one auditable validation row per CDM row."""
    reports = []
    for _, row in cdm_frame.iterrows():
        cov_a, status_a = _object_covariance(row, cdm_frame.columns, 1)
        cov_b, status_b = _object_covariance(row, cdm_frame.columns, 2)
        tca = row.get("_CDM_TCA")
        reports.append({
            "CDM_SOURCE_ROW": int(row.get("_CDM_SOURCE_ROW", len(reports) + 1)),
            "CDM_TCA": tca.isoformat() if pd.notna(tca) else None,
            "CDM_OBJECT1_ID": row.get("_CDM_OBJECT1"),
            "CDM_OBJECT2_ID": row.get("_CDM_OBJECT2"),
            "OBJECT1_COVARIANCE_STATUS": status_a,
            "OBJECT2_COVARIANCE_STATUS": status_b,
            "OBJECT1_SIGMA_R_KM": cov_a["sigma_r_km"] if cov_a else np.nan,
            "OBJECT1_SIGMA_T_KM": cov_a["sigma_t_km"] if cov_a else np.nan,
            "OBJECT1_SIGMA_N_KM": cov_a["sigma_n_km"] if cov_a else np.nan,
            "OBJECT2_SIGMA_R_KM": cov_b["sigma_r_km"] if cov_b else np.nan,
            "OBJECT2_SIGMA_T_KM": cov_b["sigma_t_km"] if cov_b else np.nan,
            "OBJECT2_SIGMA_N_KM": cov_b["sigma_n_km"] if cov_b else np.nan,
            "COVARIANCE_VALIDATION_STATUS": (
                "BOTH_POSITION_BLOCKS_VALID" if cov_a and cov_b
                else "PARTIAL_COVARIANCE" if cov_a or cov_b
                else "NO_VALID_POSITION_COVARIANCE"
            ),
            "NOTE": (
                "Position block only; not yet used for Pc. Confirm units/frame and "
                "implement encounter-plane covariance transformation before probability use."
            ),
        })
    return pd.DataFrame(reports)


def match_cdms_to_conjunctions(cdm_frame, conjunctions, tolerance_minutes=DEFAULT_TCA_TOLERANCE_MINUTES):
    """Match by unordered NORAD IDs (or designator strings) and nearest TCA."""
    if conjunctions is None or conjunctions.empty:
        return pd.DataFrame()
    required = {"NORAD_A", "NORAD_B", "TCA"}
    if not required.issubset(conjunctions.columns):
        raise ValueError("Conjunctions need NORAD_A, NORAD_B, and TCA columns.")

    cdm = cdm_frame.copy()
    report = validate_cdm_rows(cdm)
    cdm = cdm.merge(report, on="CDM_SOURCE_ROW", how="left")
    cdm["_PAIR"] = cdm.apply(
        lambda r: tuple(sorted([str(r.get("_CDM_OBJECT1_ID") or ""), str(r.get("_CDM_OBJECT2_ID") or "")])),
        axis=1,
    )
    cdm = cdm[cdm["_CDM_TCA"].notna()].copy()
    cdm["CDM_TCA"] = cdm["_CDM_TCA"]

    events = conjunctions.copy()
    events["_EVENT_TCA"] = pd.to_datetime(events["TCA"], utc=True, errors="coerce")
    events["_PAIR"] = events.apply(
        lambda r: tuple(sorted([str(int(r["NORAD_A"])), str(int(r["NORAD_B"]))])),
        axis=1,
    )

    matches = []
    tolerance_seconds = float(tolerance_minutes) * 60.0
    for _, event in events.iterrows():
        candidates = cdm[cdm["_PAIR"] == event["_PAIR"]].copy()
        if candidates.empty or pd.isna(event["_EVENT_TCA"]):
            matches.append({
                "NORAD_A": event["NORAD_A"], "NORAD_B": event["NORAD_B"],
                "EVENT_TCA": event["TCA"], "CDM_MATCH_STATUS": "NO_OBJECT_PAIR_MATCH",
                "CDM_TCA": None, "CDM_SOURCE_ROW": None,
                "COVARIANCE_VALIDATION_STATUS": "NOT_AVAILABLE",
            })
            continue
        candidates["_TCA_DELTA_SECONDS"] = (
            candidates["_CDM_TCA"] - event["_EVENT_TCA"]
        ).dt.total_seconds().abs()
        best = candidates.sort_values("_TCA_DELTA_SECONDS").iloc[0]
        within = float(best["_TCA_DELTA_SECONDS"]) <= tolerance_seconds
        matches.append({
            "NORAD_A": event["NORAD_A"], "NORAD_B": event["NORAD_B"],
            "EVENT_TCA": event["TCA"],
            "CDM_MATCH_STATUS": "MATCHED_WITHIN_TOLERANCE" if within else "PAIR_MATCH_TCA_OUTSIDE_TOLERANCE",
            "CDM_TCA": best["_CDM_TCA"].isoformat(),
            "CDM_TCA_DELTA_MINUTES": float(best["_TCA_DELTA_SECONDS"]) / 60.0,
            "CDM_SOURCE_ROW": int(best["_CDM_SOURCE_ROW"]),
            "COVARIANCE_VALIDATION_STATUS": best["COVARIANCE_VALIDATION_STATUS"],
            "OBJECT1_COVARIANCE_STATUS": best["OBJECT1_COVARIANCE_STATUS"],
            "OBJECT2_COVARIANCE_STATUS": best["OBJECT2_COVARIANCE_STATUS"],
        })
    return pd.DataFrame(matches)
