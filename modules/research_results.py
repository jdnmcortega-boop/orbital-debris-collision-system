"""Build current-run research objective and hypothesis artifacts after live refresh.

These artifacts summarize only outputs from the completed live pipeline. They do
not manufacture ground truth or rerun fixed statistical benchmark experiments.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _col(frame: pd.DataFrame, *names):
    lookup = {str(c).upper(): c for c in frame.columns}
    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]
    return None


def _finite_stats(frame: pd.DataFrame, *names):
    column = _col(frame, *names)
    if column is None:
        return {"column": None, "count": 0, "min": None, "median": None, "max": None}
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "column": str(column),
        "count": int(len(values)),
        "min": float(values.min()) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def _risk_counts(frame: pd.DataFrame):
    column = _col(frame, "COLLISION_RISK_LEVEL", "RISK_LEVEL", "COMPOSITE_RISK_LEVEL")
    if column is None or frame.empty:
        return {}
    return {str(k).upper(): int(v) for k, v in frame[column].fillna("UNKNOWN").astype(str).str.upper().value_counts().items()}


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")


def build_live_research_artifacts(fresh_df, propagated, conjunctions, analytic_results=None,
                                  predictions=None, qae_results=None, fp_results=None,
                                  reentry_results=None, failed_objects=None):
    """Write live summary, objective evidence, hypothesis status, and Markdown report."""
    config.ensure_dirs()
    out = config.RESULTS_DIR
    now = datetime.now(timezone.utc).isoformat()
    conjunctions = conjunctions if isinstance(conjunctions, pd.DataFrame) else _read_csv(config.CONJUNCTIONS_FILE)
    analytic_results = analytic_results if isinstance(analytic_results, pd.DataFrame) else _read_csv(out / "analytic_pc_results.csv")
    predictions = predictions if isinstance(predictions, pd.DataFrame) else _read_csv(out / "predictions.csv")
    qae_results = qae_results if isinstance(qae_results, pd.DataFrame) else _read_csv(out / "qae_comparison.csv")
    fp_results = fp_results if isinstance(fp_results, pd.DataFrame) else _read_csv(out / "false_positive_analysis.csv")
    reentry_results = reentry_results if isinstance(reentry_results, pd.DataFrame) else _read_csv(out / "reentry_analysis.csv")
    quality = _read_json(out / "data_quality_report.json")
    classical = _read_json(out / "classical_security_results.json")
    qkd_data = _read_json(out / "qkd_results.json")

    object_count = int(fresh_df["NORAD_CAT_ID"].nunique()) if fresh_df is not None and not fresh_df.empty and "NORAD_CAT_ID" in fresh_df else 0
    latest_epoch = None
    if fresh_df is not None and "EPOCH" in fresh_df and not fresh_df.empty:
        latest_epoch = pd.to_datetime(fresh_df["EPOCH"], utc=True, errors="coerce").max()
        latest_epoch = latest_epoch.isoformat() if pd.notna(latest_epoch) else None

    miss_stats = _finite_stats(conjunctions, "MISS_DISTANCE_KM")
    velocity_stats = _finite_stats(conjunctions, "RELATIVE_VELOCITY_KM_S")
    pc_stats = _finite_stats(analytic_results, "ANALYTIC_PC")
    pc_col = _col(analytic_results, "ANALYTIC_PC")
    max_pc_col = _col(analytic_results, "ANALYTIC_PC_MAX_AT_ZERO_MISS")
    source_col = _col(predictions, "PROBABILITY_SOURCE")
    risk_col = _col(predictions, "COLLISION_RISK_LEVEL", "RISK_LEVEL")
    priority_col = _col(predictions, "ORION_X_PRIORITY_LEVEL", "COMPOSITE_RISK_LEVEL")
    priority_counts = {}
    if priority_col is not None:
        priority_counts = {str(k).upper(): int(v) for k, v in predictions[priority_col].fillna("UNKNOWN").astype(str).str.upper().value_counts().items()}

    live_summary = {
        "report_type": "live_run",
        "generated_at_utc": now,
        "forecast_start_utc": getattr(config, "GRID_START", datetime.now(timezone.utc)).isoformat(),
        "forecast_horizon_days": int(getattr(config, "FORECAST_HORIZON_DAYS", 30)),
        "orbital_objects": object_count,
        "latest_orbital_epoch_utc": latest_epoch,
        "propagated_state_rows": int(len(propagated)) if propagated is not None else 0,
        "failed_object_count": int(len(failed_objects)) if failed_objects is not None else 0,
        "conjunction_count": int(len(conjunctions)),
        "analytical_pc_rows": int(len(analytic_results)),
        "prediction_rows": int(len(predictions)),
        "qae_rows": int(len(qae_results)),
        "false_positive_analysis_rows": int(len(fp_results)),
        "reentry_analysis_rows": int(len(reentry_results)),
        "miss_distance_km": miss_stats,
        "relative_velocity_km_s": velocity_stats,
        "analytical_pc": pc_stats,
        "collision_probability_risk_counts": _risk_counts(predictions),
        "response_priority_counts": priority_counts,
        "probability_source_counts": (
            {str(k): int(v) for k, v in predictions[source_col].fillna("UNKNOWN").astype(str).value_counts().items()}
            if source_col is not None else {}
        ),
        "classical_security_result_present": bool(classical),
        "qkd_result_present": bool(qkd_data),
        "data_quality_report_present": bool(quality),
        "monte_carlo_rows": int(len(_read_csv(out / "monte_carlo_results.csv"))),
        "note": "This summary is generated after the live pipeline completes. Fixed benchmark experiments and historical CDM validation are not silently relabeled as live-run results."
    }

    objectives = {
        "report_type": "live_run_objective_evidence",
        "generated_at_utc": now,
        "run": live_summary,
        "objectives": [
            {
                "objective": 1,
                "topic": "Estimate collision probability using relative motion, orbital information, modeled uncertainty, and collision-region probability.",
                "current_run_evidence": {
                    "conjunction_count": len(conjunctions),
                    "miss_distance_km": miss_stats,
                    "relative_velocity_km_s": velocity_stats,
                    "analytical_pc": pc_stats,
                    "modeled_uncertainty_fields_available": bool(_col(analytic_results, "SIGMA_A_KM") and _col(analytic_results, "SIGMA_B_KM")),
                    "maximum_pc_at_zero_miss_stats": _finite_stats(analytic_results, "ANALYTIC_PC_MAX_AT_ZERO_MISS"),
                    "monte_carlo_rows": live_summary["monte_carlo_rows"]
                },
                "limitation": "The analytical model uses modeled uncertainty unless valid object-specific covariance is imported and correctly transformed. Live Monte Carlo is not run by default; a missing MC table is not a zero-probability result."
            },
            {
                "objective": 2,
                "topic": "Evaluate conjunction detection.",
                "current_run_evidence": {
                    "objects": object_count,
                    "propagated_state_rows": live_summary["propagated_state_rows"],
                    "detected_conjunctions": len(conjunctions),
                    "tca_column_present": _col(conjunctions, "TCA") is not None,
                    "miss_distance_km": miss_stats
                },
                "limitation": "Precision, recall, false-positive rate, and false-negative rate require a verified reference event set for the same time window. They are not inferred from event counts alone."
            },
            {
                "objective": 3,
                "topic": "Compare QAE with classical Monte Carlo.",
                "current_run_evidence": {
                    "qae_rows": len(qae_results),
                    "monte_carlo_rows": live_summary["monte_carlo_rows"],
                    "analytical_pc_rows": len(analytic_results)
                },
                "limitation": "A live paired significance test is not run unless paired current-run QAE and Monte Carlo estimates are available. The fixed benchmark hypothesis test is retained separately and is not overwritten by this refresh."
            },
            {
                "objective": 4,
                "topic": "Compare QKD and classical secure transmission.",
                "current_run_evidence": {
                    "classical_security_output_present": bool(classical),
                    "qkd_output_present": bool(qkd_data),
                    "classical_result_keys": sorted(classical.keys()) if isinstance(classical, dict) else [],
                    "qkd_result_keys": sorted(qkd_data.keys()) if isinstance(qkd_data, dict) else []
                },
                "limitation": "The refresh writes per-run security outputs, but a single run is not a statistical benchmark. The fixed multi-trial benchmark and its hypothesis test are kept distinct."
            },
            {
                "objective": 5,
                "topic": "Classify conjunctions by collision probability and response priority.",
                "current_run_evidence": {
                    "collision_probability_risk_counts": live_summary["collision_probability_risk_counts"],
                    "response_priority_counts": priority_counts,
                    "probability_source_counts": live_summary["probability_source_counts"]
                },
                "limitation": "Collision-probability classification and response-priority ranking are separate metrics. A lack of HIGH labels is not proof that no real conjunction is dangerous."
            },
            {
                "objective": 6,
                "topic": "Prioritize debris for further assessment and possible removal.",
                "current_run_evidence": {
                    "reentry_analysis_rows": len(reentry_results),
                    "debris_involved_events": int(sum(
                        (conjunctions.get("OBJECT_A", pd.Series(dtype=str)).fillna("").astype(str).str.upper().str.contains(r"DEB|DEBRIS|FRAGMENT|ROCKET BODY|R/B", regex=True))
                        | (conjunctions.get("OBJECT_B", pd.Series(dtype=str)).fillna("").astype(str).str.upper().str.contains(r"DEB|DEBRIS|FRAGMENT|ROCKET BODY|R/B", regex=True))
                    )) if not conjunctions.empty else 0
                },
                "limitation": "This is a screening list, not an authorization or recommendation to remove a specific object. Verify nonfunctional debris status, consequences, uncertainty, and removal feasibility; never classify an active spacecraft as a removal target."
            }
        ]
    }

    # Hypotheses are evaluated only when the relevant evidence exists. The live
    # refresh deliberately does not fabricate reference truth or extra trials.
    cdm_files = [
        out / "orion_x_historical_cdm_comparison.csv",
        out / "cdm_comparison.csv",
        out / "historical_cdm_comparison.csv",
    ]
    cdm_path = next((p for p in cdm_files if p.exists()), None)
    cdm = _read_csv(cdm_path) if cdm_path else pd.DataFrame()
    cdm_status_col = _col(cdm, "CDM_MATCH_STATUS")
    cdm_match_count = int(cdm[cdm_status_col].astype(str).eq("MATCHED_WITHIN_TOLERANCE").sum()) if cdm_status_col else 0

    hypotheses = {
        "report_type": "live_run_hypothesis_status",
        "generated_at_utc": now,
        "alpha": 0.05,
        "hypotheses": [
            {
                "id": "H01",
                "null_hypothesis": "ORION-X does not meet the predefined conjunction-detection and classification accuracy criteria.",
                "status": "NOT_TESTED_IN_THIS_REFRESH",
                "decision": "INCONCLUSIVE",
                "evidence": {
                    "current_run_conjunctions": len(conjunctions),
                    "cdm_comparison_file_present": cdm_path is not None,
                    "cdm_matches_within_tolerance": cdm_match_count if cdm_path is not None else None
                },
                "reason": "A current conjunction count alone cannot measure accuracy. A matched, verified reference set with false-positive and false-negative accounting is required."
            },
            {
                "id": "H02",
                "null_hypothesis": "There is no statistically significant difference between QAE and Monte Carlo estimation errors.",
                "status": "NOT_TESTED_IN_THIS_REFRESH" if live_summary["monte_carlo_rows"] == 0 or len(qae_results) == 0 else "PAIRED_DATA_AVAILABLE_REQUIRES_TEST",
                "decision": "INCONCLUSIVE",
                "evidence": {
                    "current_run_qae_rows": len(qae_results),
                    "current_run_monte_carlo_rows": live_summary["monte_carlo_rows"],
                    "paired_test_performed": False
                },
                "reason": "The live pipeline intentionally skips Monte Carlo by default. The fixed benchmark test is separate and must not be described as a fresh live-run test."
            },
            {
                "id": "H03",
                "null_hypothesis": "There is no statistically significant difference between QKD and classical transmission under the specified test conditions.",
                "status": "PER_RUN_OUTPUTS_GENERATED_BUT_NO_NEW_MULTI_TRIAL_TEST",
                "decision": "INCONCLUSIVE_FOR_THIS_REFRESH",
                "evidence": {
                    "classical_security_output_present": bool(classical),
                    "qkd_output_present": bool(qkd_data),
                    "new_fisher_or_other_hypothesis_test_performed": False
                },
                "reason": "A refresh creates current security output files, but does not perform a multi-trial paired or independent-sample experiment. Preserve the fixed benchmark test as a separately labeled result."
            }
        ]
    }

    # Generate a human-readable report from the exact JSON artifacts written here.
    lines = [
        "# ORION-X Live Research Results",
        "",
        f"**Generated (UTC):** {now}",
        "",
        "> This file is generated automatically after a successful live refresh. It reports current-run output only. Fixed benchmarks and historical CDM tests are not presented as new live measurements.",
        "",
        "## Current run summary",
        "",
        f"- Orbital objects: {object_count}",
        f"- Latest orbital element epoch: {latest_epoch or 'Unavailable'}",
        f"- Propagated state rows: {live_summary['propagated_state_rows']:,}",
        f"- Failed objects: {live_summary['failed_object_count']}",
        f"- Detected conjunctions: {len(conjunctions)}",
        f"- Analytical Pc rows: {len(analytic_results)}",
        f"- Prediction rows: {len(predictions)}",
        f"- QAE rows: {len(qae_results)}",
        f"- Monte Carlo rows: {live_summary['monte_carlo_rows']} (the live pipeline skips MC by default)",
        "",
        "## Objective evidence",
        ""
    ]
    for item in objectives["objectives"]:
        ev = item["current_run_evidence"]
        lines.extend([
            f"### Objective {item['objective']}",
            "",
            item["topic"],
            "",
            "**Current-run evidence:**",
            "",
        ])
        lines.extend([f"- {key.replace('_', ' ')}: {value}" for key, value in ev.items()])
        lines.extend(["", f"**Limitation:** {item['limitation']}", ""])
    lines.extend(["## Hypothesis status", ""])
    for item in hypotheses["hypotheses"]:
        lines.extend([
            f"### {item['id']} — {item['decision']}",
            "",
            f"**Null hypothesis:** {item['null_hypothesis']}",
            "",
            f"**Status:** {item['status']}",
            "",
            item["reason"],
            "",
        ])
    lines.extend([
        "## Interpretation guardrails",
        "",
        "- A zero Monte Carlo hit count is not proof that true Pc equals zero.",
        "- No current-run H01 decision is made without verified reference events.",
        "- H02 is not tested on live data unless paired live QAE and Monte Carlo results exist.",
        "- H03 is not statistically retested from one security run; use multi-trial benchmark data.",
        "- HIGH response priority is not the same as HIGH collision probability.",
        "- Debris screening candidates require independent status and feasibility checks before any removal recommendation.",
        ""
    ])

    _write_json(out / "live_run_summary.json", live_summary)
    _write_json(out / "objective_results_live.json", objectives)
    _write_json(out / "hypothesis_results_live.json", hypotheses)
    report_text = "\n".join(lines)
    (out / "research_results_live.md").write_text(report_text, encoding="utf-8")
    # Keep the established report path current too, so the GitHub report link
    # reflects the most recent successful live refresh.
    (out / "results_by_research_objectives.md").write_text(report_text, encoding="utf-8")

    return {
        "live_run_summary": str(out / "live_run_summary.json"),
        "objective_results": str(out / "objective_results_live.json"),
        "hypothesis_results": str(out / "hypothesis_results_live.json"),
        "research_report": str(out / "research_results_live.md"),
        "objectives_report": str(out / "results_by_research_objectives.md"),
        "generated_at_utc": now,
    }
