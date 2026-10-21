"""Audit the historical-validation CSV without changing forecast outputs.

This checks timestamp consistency and TLE epoch freshness. TLE epoch is only a
proxy for data freshness; it does not prove when an element set was published.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_probability_estimators(df: pd.DataFrame) -> dict:
    """Summarize analytic/QAE/MC agreement without treating estimators as independent predictors."""
    pre_mask = df.get(
        "PREDICTION_TIME_IS_PRE_EVENT",
        pd.Series([True] * len(df), index=df.index),
    ).astype(str).str.lower().isin(["true", "1"])
    pre = df.loc[pre_mask]

    def risk_counts(frame: pd.DataFrame, column: str) -> dict:
        if column not in frame.columns:
            return {}
        return {
            str(level): int(count)
            for level, count in frame[column].fillna("MISSING").value_counts().items()
        }

    def error_summary(frame: pd.DataFrame) -> dict:
        column = "QAE_ABS_ERROR_FROM_ANALYTIC_PC"
        if column not in frame.columns:
            return {"available": False, "count": 0, "median": None, "max": None}
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return {
            "available": True,
            "count": int(len(values)),
            "median": float(values.median()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }

    coverage_column = "MC_CI_CONTAINS_ANALYTIC_PC"

    def coverage_summary(frame: pd.DataFrame) -> dict:
        if coverage_column not in frame.columns:
            return {"available": False, "count": 0, "contains_analytic_pc": 0, "fraction": None}
        values = frame[coverage_column].astype(str).str.lower().isin(["true", "1"])
        valid = frame[coverage_column].notna()
        count = int(valid.sum())
        covered = int((values & valid).sum())
        return {
            "available": True,
            "count": count,
            "contains_analytic_pc": covered,
            "fraction": float(covered / count) if count else None,
        }

    return {
        "interpretation": (
            "QAE estimates the analytic Pc target; Monte Carlo estimates that same target "
            "with finite-sample noise. Agreement here does not validate the orbit forecast "
            "or the assumed uncertainty model."
        ),
        "all_rows": {
            "analytic_risk_counts": risk_counts(df, "ANALYTIC_RISK_LEVEL"),
            "qae_risk_counts": risk_counts(df, "QAE_RISK_LEVEL"),
            "mc_risk_counts": risk_counts(df, "MC_RISK_LEVEL"),
            "qae_absolute_error_from_analytic_pc": error_summary(df),
            "mc_ci_contains_analytic_pc": coverage_summary(df),
        },
        "pre_event_rows": {
            "row_count": int(len(pre)),
            "analytic_risk_counts": risk_counts(pre, "ANALYTIC_RISK_LEVEL"),
            "qae_risk_counts": risk_counts(pre, "QAE_RISK_LEVEL"),
            "mc_risk_counts": risk_counts(pre, "MC_RISK_LEVEL"),
            "qae_absolute_error_from_analytic_pc": error_summary(pre),
            "mc_ci_contains_analytic_pc": coverage_summary(pre),
        },
    }



def summarize_proximity_probability_disagreement(df: pd.DataFrame) -> dict:
    """Report proximity alerts whose analytical Pc label is lower, without relabeling risk."""
    pre_mask = df.get(
        "PREDICTION_TIME_IS_PRE_EVENT",
        pd.Series([True] * len(df), index=df.index),
    ).astype(str).str.lower().isin(["true", "1"])
    pre = df.loc[pre_mask]

    proximity = pre.get(
        "FORECAST_PROXIMITY_ALERT_LEVEL",
        pd.Series(["MISSING"] * len(pre), index=pre.index),
    ).fillna("MISSING").astype(str).str.upper()
    probability = pre.get(
        "ANALYTIC_RISK_LEVEL",
        pd.Series(["MISSING"] * len(pre), index=pre.index),
    ).fillna("MISSING").astype(str).str.upper()

    high_proximity_low_pc = int(((proximity == "HIGH") & (probability == "LOW")).sum())
    medium_or_high_proximity_low_pc = int(
        (proximity.isin(["MEDIUM", "HIGH"]) & (probability == "LOW")).sum()
    )
    cross_tab = pd.crosstab(
        proximity.rename("proximity_alert_level"),
        probability.rename("analytic_probability_risk_level"),
        dropna=False,
    )
    return {
        "pre_event_row_count": int(len(pre)),
        "high_proximity_with_low_analytic_probability_count": high_proximity_low_pc,
        "medium_or_high_proximity_with_low_analytic_probability_count": medium_or_high_proximity_low_pc,
        "proximity_vs_analytic_risk_cross_tab": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in cross_tab.iterrows()
        },
        "interpretation": (
            "Proximity alerts and analytical collision-probability labels measure different things. "
            "A HIGH proximity alert with LOW analytical Pc is a model/threshold diagnostic, not proof "
            "that the event is safe and not a basis for silently changing the Pc label."
        ),
    }

def summarize_collision_geometry(df: pd.DataFrame) -> dict:
    """Compare forecast miss distances with the configured hard-body radius."""
    required = {"FORECAST_MISS_DISTANCE_KM", "HARD_BODY_RADIUS_KM"}
    missing = sorted(required - set(df.columns))

    if missing:
        return {
            "available": False,
            "missing_columns": missing,
            "interpretation": (
                "Collision geometry cannot be evaluated without both "
                "forecast miss distance and hard-body radius."
            ),
        }

    miss = pd.to_numeric(df["FORECAST_MISS_DISTANCE_KM"], errors="coerce")
    radius = pd.to_numeric(df["HARD_BODY_RADIUS_KM"], errors="coerce")

    valid = (
        miss.notna()
        & radius.notna()
        & (miss >= 0)
        & (radius > 0)
        & np.isfinite(miss)
        & np.isfinite(radius)
    )

    checked = int(valid.sum())
    within = valid & (miss <= radius)
    ratios = miss[valid] / radius[valid]

    return {
        "available": True,
        "rows_checked": checked,
        "rows_with_invalid_or_missing_geometry": int((~valid).sum()),
        "rows_with_forecast_miss_within_hard_body_radius": int(within.sum()),
        "minimum_forecast_miss_distance_km": (
            float(miss[valid].min()) if checked else None
        ),
        "minimum_miss_to_radius_ratio": (
            float(ratios.min()) if checked else None
        ),
        "interpretation": (
            "This compares reconstructed forecast geometry with the configured "
            "hard-body radius. A miss within the radius is a geometric consistency "
            "check, not proof of prospective predictive skill. A miss outside the "
            "radius means the reconstructed trajectory does not reproduce the "
            "collision geometry for that row under this criterion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tca-offset-hours", type=float, default=1.0)
    parser.add_argument("--provenance-manifest", type=Path, default=None)
    args = parser.parse_args()

    provenance_path = args.provenance_manifest
    provenance = {}
    if provenance_path is not None:
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Provenance manifest not found: {provenance_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    df = pd.read_csv(args.input)
    required = {
        "SNAPSHOT_TIME", "DAYS_BEFORE_EVENT", "TLE_EPOCH_A", "TLE_EPOCH_B",
        "FORECAST_TCA", "ACTUAL_EVENT_TIME", "FORECAST_TCA_OFFSET_FROM_EVENT_HOURS",
        "FORECAST_MISS_DISTANCE_KM", "FORECAST_PROXIMITY_ALERT_LEVEL",
        "COARSE_BASELINE_TCA", "COARSE_BASELINE_TCA_OFFSET_FROM_EVENT_HOURS",
        "COARSE_BASELINE_MISS_DISTANCE_KM",
        "ANALYTIC_RISK_LEVEL", "PREDICTION_TIME_IS_PRE_EVENT",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Validation CSV missing required columns: {missing}")

    for col in ("SNAPSHOT_TIME", "TLE_EPOCH_A", "TLE_EPOCH_B", "FORECAST_TCA", "COARSE_BASELINE_TCA", "ACTUAL_EVENT_TIME"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="raise", format="ISO8601")

    df["TLE_AGE_A_DAYS"] = (df["SNAPSHOT_TIME"] - df["TLE_EPOCH_A"]).dt.total_seconds() / 86400
    df["TLE_AGE_B_DAYS"] = (df["SNAPSHOT_TIME"] - df["TLE_EPOCH_B"]).dt.total_seconds() / 86400
    df["TLE_EPOCHS_NOT_AFTER_SNAPSHOT"] = (
        (df["TLE_AGE_A_DAYS"] >= 0) & (df["TLE_AGE_B_DAYS"] >= 0)
    )
    df["TCA_OFFSET_RECALCULATED_HOURS"] = (
        (df["FORECAST_TCA"] - df["ACTUAL_EVENT_TIME"]).dt.total_seconds() / 3600
    )
    df["TCA_OFFSET_COLUMN_MATCHES"] = np.isclose(
        df["TCA_OFFSET_RECALCULATED_HOURS"],
        pd.to_numeric(df["FORECAST_TCA_OFFSET_FROM_EVENT_HOURS"]),
        atol=1e-5,
        rtol=0,
    )
    pre = df[df["PREDICTION_TIME_IS_PRE_EVENT"].astype(str).str.lower().isin(["true", "1"])]
    alerts = pre[pre["FORECAST_PROXIMITY_ALERT_LEVEL"].isin(["HIGH", "MEDIUM"])]

    # Descriptive forecast diagnostics only. These summaries describe this replay;
    # they are not a standalone measure of predictive skill because the known event
    # time is evaluation ground truth, not independent evidence of historical TLE availability.
    proximity_counts = {
        str(level): int(count)
        for level, count in pre["FORECAST_PROXIMITY_ALERT_LEVEL"].value_counts(dropna=False).items()
    }
    miss_distance = pd.to_numeric(pre["FORECAST_MISS_DISTANCE_KM"], errors="coerce").dropna()
    tca_abs_offset_hours = pre["TCA_OFFSET_RECALCULATED_HOURS"].abs().dropna()
    if len(pre) >= 2:
        miss_distance_days_spearman = float(
            pre[["DAYS_BEFORE_EVENT", "FORECAST_MISS_DISTANCE_KM"]]
            .dropna()
            .rank(method="average")
            .corr()
            .iloc[0, 1]
        )
    else:
        miss_distance_days_spearman = None

    baseline_offset = pd.to_numeric(pre["COARSE_BASELINE_TCA_OFFSET_FROM_EVENT_HOURS"], errors="coerce").abs()
    refined_offset = pd.to_numeric(pre["TCA_OFFSET_RECALCULATED_HOURS"], errors="coerce").abs()
    baseline_miss = pd.to_numeric(pre["COARSE_BASELINE_MISS_DISTANCE_KM"], errors="coerce")
    refined_miss = pd.to_numeric(pre["FORECAST_MISS_DISTANCE_KM"], errors="coerce")
    valid_pairs = pd.concat(
        [baseline_offset.rename("baseline_offset"), refined_offset.rename("refined_offset"),
         baseline_miss.rename("baseline_miss"), refined_miss.rename("refined_miss")],
        axis=1,
    ).dropna()

    probability_diagnostics = {}
    diagnostic_columns = {
        "pc_max": "PC_MAX_AT_ZERO_MISS_UNDER_ASSUMED_SIGMA",
        "high_threshold": "HIGH_PC_THRESHOLD",
        "high_reachable": "HIGH_THRESHOLD_REACHABLE_UNDER_ASSUMED_SIGMA",
    }
    if all(column in df.columns for column in diagnostic_columns.values()):
        pc_max_values = pd.to_numeric(
            df[diagnostic_columns["pc_max"]], errors="coerce"
        ).dropna()
        high_threshold_values = pd.to_numeric(
            df[diagnostic_columns["high_threshold"]], errors="coerce"
        ).dropna()
        high_reachable_values = (
            df[diagnostic_columns["high_reachable"]]
            .astype(str).str.lower().isin(["true", "1"])
        )
        probability_diagnostics = {
            "model": "ISOTROPIC_2D_GAUSSIAN_ASSUMPTION",
            "uncertainty_interpretation": (
                "Configured/assumed sigma, not verified object-specific covariance."
            ),
            "max_pc_at_zero_miss_min": (
                float(pc_max_values.min()) if len(pc_max_values) else None
            ),
            "max_pc_at_zero_miss_max": (
                float(pc_max_values.max()) if len(pc_max_values) else None
            ),
            "high_pc_threshold": (
                float(high_threshold_values.iloc[0])
                if len(high_threshold_values) else None
            ),
            "high_threshold_reachable_rows": int(high_reachable_values.sum()),
            "high_threshold_unreachable_rows": int((~high_reachable_values).sum()),
            "interpretation": (
                "Reachability is a mathematical diagnostic under the assumed model; "
                "it does not establish empirical probability calibration and must not "
                "be used to force a desired risk label."
            ),
        }

    report = {
        "collision_geometry_diagnostics": summarize_collision_geometry(df),
        "collision_probability_model_diagnostics": probability_diagnostics,
        "probability_estimator_diagnostics": summarize_probability_estimators(df),
        "proximity_probability_disagreement": summarize_proximity_probability_disagreement(df),
        "coarse_grid_baseline": {
            "definition": "Same frozen TLE pair, SGP4 propagator, and forecast window; minimum separation sampled on a fixed UTC-epoch-aligned grid without sub-grid optimization. Grid samples are not anchored to the event-aligned snapshot.",
            "comparison_scope": "Numerical refinement diagnostic only; not an independent orbital-prediction baseline.",
            "pre_event_rows_compared": int(len(valid_pairs)),
            "median_absolute_tca_error_hours_coarse": float(baseline_offset.median()) if len(baseline_offset.dropna()) else None,
            "median_absolute_tca_error_hours_refined": float(refined_offset.median()) if len(refined_offset.dropna()) else None,
            "rows_refinement_reduced_absolute_tca_error": int((valid_pairs["refined_offset"] < valid_pairs["baseline_offset"]).sum()),
            "rows_refinement_increased_absolute_tca_error": int((valid_pairs["refined_offset"] > valid_pairs["baseline_offset"]).sum()),
            "median_miss_distance_km_coarse": float(baseline_miss.median()) if len(baseline_miss.dropna()) else None,
            "median_miss_distance_km_refined": float(refined_miss.median()) if len(refined_miss.dropna()) else None,
            "rows_refinement_reduced_miss_distance": int((valid_pairs["refined_miss"] < valid_pairs["baseline_miss"]).sum()),
        },
        "validation_interpretation": {
            "study_design": "RETROSPECTIVE_SINGLE_EVENT_DIAGNOSTIC",
            "predictive_skill_claim_status": "NOT_ESTABLISHED",
            "reasons": [
                "Only one known collision event is evaluated.",
                "TLE element epochs do not prove when each record became available to an operator.",
                "The forecast horizon includes the known event time, so a close forecast TCA is descriptive and does not alone establish operational advance-warning skill.",
                "The coarse-grid comparison is a numerical-refinement diagnostic, not an independent predictor baseline."
            ],
            "pre_event_rows_only_for_lead_time_claims": True,
            "event_time_row_excluded_from_pre_event_counts": True
        },
        "input_csv": str(args.input),
        "tle_data_provenance": {
            "manifest_path": str(provenance_path) if provenance_path is not None else None,
            "status": provenance.get("status", "NOT_ASSESSED"),
            "source_url": provenance.get("source_url"),
            "archive_retrieved_at_utc": provenance.get("archive_retrieved_at_utc"),
            "original_archive_timestamp_utc": provenance.get("original_archive_timestamp_utc"),
            "record_level_publication_timestamps_available": provenance.get("record_level_publication_timestamps_available"),
            "interpretation": "TLE epoch checks establish only that element epochs are not later than each snapshot; they do not establish when those records became available to an operator.",
        },
        "row_count": int(len(df)),
        "event_time_utc": df["ACTUAL_EVENT_TIME"].iloc[0].isoformat() if len(df) else None,
        "all_tle_epochs_not_after_snapshot": bool(df["TLE_EPOCHS_NOT_AFTER_SNAPSHOT"].all()),
        "future_epoch_row_count": int((~df["TLE_EPOCHS_NOT_AFTER_SNAPSHOT"]).sum()),
        "max_tle_age_a_days": float(df["TLE_AGE_A_DAYS"].max()),
        "max_tle_age_b_days": float(df["TLE_AGE_B_DAYS"].max()),
        "min_tle_age_a_days": float(df["TLE_AGE_A_DAYS"].min()),
        "min_tle_age_b_days": float(df["TLE_AGE_B_DAYS"].min()),
        "tca_offset_column_matches_recalculation": bool(df["TCA_OFFSET_COLUMN_MATCHES"].all()),
        "tca_offset_mismatch_rows": int((~df["TCA_OFFSET_COLUMN_MATCHES"]).sum()),
        "rows_with_tca_within_configured_offset_hours": int(
            (df["TCA_OFFSET_RECALCULATED_HOURS"].abs() <= args.max_tca_offset_hours).sum()
        ),
        "configured_tca_offset_tolerance_hours": args.max_tca_offset_hours,
        "pre_event_snapshot_count": int(len(pre)),
        "pre_event_medium_or_high_proximity_alert_count": int(len(alerts)),
        "pre_event_proximity_alert_counts": proximity_counts,
        "pre_event_miss_distance_km_min": float(miss_distance.min()) if len(miss_distance) else None,
        "pre_event_miss_distance_km_median": float(miss_distance.median()) if len(miss_distance) else None,
        "pre_event_miss_distance_km_max": float(miss_distance.max()) if len(miss_distance) else None,
        "pre_event_abs_tca_offset_hours_median": float(tca_abs_offset_hours.median()) if len(tca_abs_offset_hours) else None,
        "miss_distance_vs_days_before_event_spearman": miss_distance_days_spearman,
        "first_high_proximity_alert_days_before_event": (
            float(pre.loc[pre["FORECAST_PROXIMITY_ALERT_LEVEL"] == "HIGH", "DAYS_BEFORE_EVENT"].max())
            if (pre["FORECAST_PROXIMITY_ALERT_LEVEL"] == "HIGH").any() else None
        ),
        "first_medium_proximity_alert_days_before_event": (
            float(pre.loc[pre["FORECAST_PROXIMITY_ALERT_LEVEL"] == "MEDIUM", "DAYS_BEFORE_EVENT"].max())
            if (pre["FORECAST_PROXIMITY_ALERT_LEVEL"] == "MEDIUM").any() else None
        ),
        "pre_event_analytic_medium_or_high_warning_count": int(
            pre["ANALYTIC_RISK_LEVEL"].isin(["MEDIUM", "HIGH"]).sum()
        ),
        "scientific_limitations": [
            "The TLE provenance manifest records whether source and archive-availability evidence is documented. A TLE epoch at or before a snapshot does not prove the TLE was publicly available by that snapshot; publication/archive timestamps are needed.",
            "A forecast TCA near the known event time is not sufficient by itself to prove predictive skill.",
            "The coarse-grid comparison measures only the effect of sub-grid numerical refinement; it is not an independent predictive baseline. Grid samples are UTC-epoch-aligned to avoid making the event timestamp a guaranteed sample simply because snapshots share its time of day.",
            "Miss-distance summaries and their rank correlation are descriptive for this single event, not a validated skill score or independent baseline comparison.",
            "Collision probability uses the configured model assumptions and should not be treated as empirically validated without object-specific covariance.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    detail_path = args.output.with_name(args.output.stem + "_rows.csv")
    df.to_csv(detail_path, index=False)

    print("Historical validation audit")
    print(json.dumps(report, indent=2))
    print(f"Audit JSON: {args.output}")
    print(f"Audit rows: {detail_path}")


if __name__ == "__main__":
    main()
