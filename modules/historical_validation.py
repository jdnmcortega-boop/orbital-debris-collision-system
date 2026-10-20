"""Historical validation of the Iridium 33 / Cosmos 2251 encounter.

This module is deliberately separate from the normal/live pipeline.
It replays the historical event using only archived TLE information that
would have been available at each replay timestamp.

For each daily snapshot from T-30 days through T-0:

1. Select the newest historical TLE whose epoch is not newer than the snapshot.
2. Freeze that TLE pair as the information state available at that time.
3. Propagate that frozen pair across a fixed forecast horizon independent of the event time.
4. Find the predicted closest approach from that information state.
5. Calculate the model collision probability.
6. Estimate the same probability with QAE and matched-budget Monte Carlo.
7. Keep deterministic probability risk separate from MC sampling noise.
8. Keep geometry/proximity forecasting separate from probability risk.

The historical event time is used only for evaluation and ground-truth labels.
The forecast horizon is fixed independently of the event time. The known event
state is never used to construct the forecast.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import config

from .historical_replay import (
    merge_tle_archives,
    propagate_satellite,
    select_element_set,
)
from .qae import (
    analytic_collision_probability,
    run_classical_mc,
    run_qae,
)
from .prediction import classify_risk


@dataclass(frozen=True)
class HistoricalValidationEvent:
    event_id: str
    event_time_utc: datetime
    object_a_norad: int
    object_b_norad: int
    ground_truth_outcome: str = "CONFIRMED_COLLISION"


def utc_datetime(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def forecast_closest_approach(
    sat_a,
    sat_b,
    start_time: datetime,
    end_time: datetime,
    step_minutes: int = 30,
):
    """Forecast closest approach using only the TLE pair known at start_time.

    Sample a fixed UTC grid, identify every sampled local minimum, and refine
    each local-minimum bracket. Refining only the globally smallest grid sample
    can miss a narrow conjunction between grid points when a different,
    broader encounter has a smaller sampled distance.
    """
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    if end_time < start_time:
        raise ValueError("end_time must not be earlier than start_time")

    step = timedelta(minutes=step_minutes)
    step_seconds = step_minutes * 60
    first_tick = math.ceil(start_time.timestamp() / step_seconds) * step_seconds
    last_tick = math.floor(end_time.timestamp() / step_seconds) * step_seconds
    grid_times = []
    tick = first_tick
    while tick <= last_tick:
        grid_times.append(datetime.fromtimestamp(tick, tz=timezone.utc))
        tick += step_seconds

    # Include forecast-window boundaries as boundaries only; do not insert the
    # known collision time or any event-derived timestamp into the sample grid.
    sample_times = [start_time]
    sample_times.extend(t for t in grid_times if start_time < t < end_time)
    if end_time > start_time:
        sample_times.append(end_time)

    distances = []
    velocities = []
    for sample_time in sample_times:
        pos_a, vel_a = propagate_satellite(sat_a, sample_time)
        pos_b, vel_b = propagate_satellite(sat_b, sample_time)
        distances.append(float(np.linalg.norm(pos_a - pos_b)))
        velocities.append(float(np.linalg.norm(vel_a - vel_b)))

    best_index = int(np.argmin(distances))
    best_time = sample_times[best_index]
    best_distance = distances[best_index]
    best_relative_velocity = velocities[best_index]

    # Refine every sampled local minimum, not just the globally smallest grid
    # sample. This is important for short, high-relative-speed encounters.
    candidate_indices = []
    for i in range(1, len(sample_times) - 1):
        if distances[i] <= distances[i - 1] and distances[i] <= distances[i + 1]:
            candidate_indices.append(i)

    # Also refine edge minima over their available one-sided intervals.
    if len(sample_times) >= 2:
        if distances[0] <= distances[1]:
            candidate_indices.append(0)
        if distances[-1] <= distances[-2]:
            candidate_indices.append(len(sample_times) - 1)

    for i in candidate_indices:
        lower_time = sample_times[max(0, i - 1)]
        upper_time = sample_times[min(len(sample_times) - 1, i + 1)]
        if upper_time <= lower_time:
            continue

        def distance_at_offset(offset_seconds: float) -> float:
            candidate_time = lower_time + timedelta(seconds=float(offset_seconds))
            pos_a, _ = propagate_satellite(sat_a, candidate_time)
            pos_b, _ = propagate_satellite(sat_b, candidate_time)
            return float(np.linalg.norm(pos_a - pos_b))

        interval_seconds = (upper_time - lower_time).total_seconds()
        refined = minimize_scalar(
            distance_at_offset,
            bounds=(0.0, interval_seconds),
            method="bounded",
            options={"xatol": 0.1},
        )
        if refined.success and np.isfinite(refined.fun) and float(refined.fun) < best_distance:
            best_time = lower_time + timedelta(seconds=float(refined.x))
            best_distance = float(refined.fun)
            _, vel_a = propagate_satellite(sat_a, best_time)
            _, vel_b = propagate_satellite(sat_b, best_time)
            best_relative_velocity = float(np.linalg.norm(vel_a - vel_b))

    return best_time, best_distance, best_relative_velocity


def forecast_closest_approach_coarse_grid(
    sat_a,
    sat_b,
    start_time: datetime,
    end_time: datetime,
    step_minutes: int = 30,
):
    """Reference baseline: select the minimum separation on the fixed coarse grid.

    This is a numerical-resolution baseline, not an independent orbit predictor:
    it uses the same frozen TLE pair, SGP4 propagator, and forecast window as the
    refined method, but does not optimize between grid points.
    """
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    if end_time < start_time:
        raise ValueError("end_time must not be earlier than start_time")

    step = timedelta(minutes=step_minutes)
    step_seconds = step_minutes * 60
    # Use the same fixed UTC epoch grid as the refined forecast. Do not inject
    # start/end timestamps into the coarse grid: doing so reintroduces snapshot
    # phase alignment and can make the known event time a guaranteed sample.
    first_tick = math.ceil(start_time.timestamp() / step_seconds) * step_seconds
    last_tick = math.floor(end_time.timestamp() / step_seconds) * step_seconds
    current = datetime.fromtimestamp(first_tick, tz=timezone.utc)
    last_grid_time = datetime.fromtimestamp(last_tick, tz=timezone.utc)
    best_time = current
    best_distance = float("inf")
    best_relative_velocity = float("nan")

    while current <= last_grid_time:
        pos_a, vel_a = propagate_satellite(sat_a, current)
        pos_b, vel_b = propagate_satellite(sat_b, current)
        distance = float(np.linalg.norm(pos_a - pos_b))
        if distance < best_distance:
            best_time = current
            best_distance = distance
            best_relative_velocity = float(np.linalg.norm(vel_a - vel_b))
        current += step

    return best_time, best_distance, best_relative_velocity


def forecast_proximity_alert_level(miss_distance_km: float) -> str:
    """Classify forecast proximity separately from collision probability."""
    distance = abs(float(miss_distance_km))

    if distance <= 10.0:
        return "HIGH"
    if distance <= 25.0:
        return "MEDIUM"
    return "LOW"


def adaptive_qae_qubits(
    probability: float,
    requested_qubits: int,
    max_qubits: int = 14,
) -> int:
    """Choose enough QAE evaluation qubits to resolve very small amplitudes."""
    probability = float(np.clip(probability, 0.0, 1.0))
    m = max(1, int(requested_qubits))
    max_qubits = max(m, int(max_qubits))

    if probability <= 0.0:
        return m

    while m < max_qubits:
        first_bin = np.sin(np.pi / (2 ** m)) ** 2
        if probability >= first_bin:
            break
        m += 1

    return m


def run_historical_validation(
    event: HistoricalValidationEvent,
    archives,
    rewind_days: int = 30,
    snapshot_step_hours: int = 24,
    forecast_step_minutes: int = 30,
    forecast_horizon_days: int = 35,
    qae_eval_qubits: int = 6,
    qae_shots: int = 200,
    sigma_km: float | None = None,
    hard_body_radius_km: float | None = None,
):
    """Run the leakage-safe historical validation experiment."""
    if rewind_days < 0:
        raise ValueError("rewind_days must be non-negative")
    if snapshot_step_hours <= 0:
        raise ValueError("snapshot_step_hours must be positive")
    if forecast_horizon_days <= 0:
        raise ValueError("forecast_horizon_days must be positive")

    for norad in (event.object_a_norad, event.object_b_norad):
        if norad not in archives:
            raise KeyError(f"NORAD {norad} is missing from the historical archive")

    start = event.event_time_utc - timedelta(days=rewind_days)
    snapshot_step = timedelta(hours=snapshot_step_hours)

    rows = []
    snapshot = start
    qae_upgrades = 0

    while snapshot <= event.event_time_utc:
        selected_a = select_element_set(archives[event.object_a_norad], snapshot)
        selected_b = select_element_set(archives[event.object_b_norad], snapshot)

        if selected_a is None or selected_b is None:
            raise RuntimeError(
                f"Historical archive does not contain a TLE at or before {snapshot.isoformat()} "
                "for both objects."
            )

        epoch_a, name_a, _, sat_a = selected_a
        epoch_b, name_b, _, sat_b = selected_b

        forecast_tca, forecast_miss, forecast_velocity = forecast_closest_approach(
            sat_a,
            sat_b,
            snapshot,
            snapshot + timedelta(days=forecast_horizon_days),
            step_minutes=forecast_step_minutes,
        )

        # Same-data coarse-grid reference baseline. This isolates the value of
        # sub-grid TCA refinement; it is not an independent predictive benchmark.
        baseline_tca, baseline_miss, baseline_velocity = forecast_closest_approach_coarse_grid(
            sat_a,
            sat_b,
            snapshot,
            snapshot + timedelta(days=forecast_horizon_days),
            step_minutes=forecast_step_minutes,
        )

        analytic_pc = analytic_collision_probability(
            forecast_miss,
            sigma_km=sigma_km,
            hard_body_radius_km=hard_body_radius_km,
        )

        effective_qae_qubits = adaptive_qae_qubits(
            analytic_pc,
            requested_qubits=qae_eval_qubits,
            max_qubits=14,
        )

        if effective_qae_qubits > qae_eval_qubits:
            qae_upgrades += 1

        qae = run_qae(
            analytic_pc,
            num_eval_qubits=effective_qae_qubits,
            shots=qae_shots,
        )

        mc = run_classical_mc(
            analytic_pc,
            n_samples=qae["oracle_calls"],
            seed=42,
        )

        lead_time_days = max(
            (forecast_tca - snapshot).total_seconds() / 86400.0,
            0.0,
        )

        proximity_alert = forecast_proximity_alert_level(forecast_miss)
        analytic_risk = classify_risk(analytic_pc)
        mc_risk = classify_risk(mc["mc_estimate"])

        # Diagnostic only: under this isotropic 2-D Gaussian model, Pc is
        # maximized at zero miss distance. Do not use this value to inflate Pc.
        effective_sigma = (
            float(sigma_km) if sigma_km is not None
            else float(getattr(config, "POSITION_UNCERTAINTY_KM", 1.0))
        )
        effective_radius = (
            float(hard_body_radius_km) if hard_body_radius_km is not None
            else float(getattr(config, "HARD_BODY_RADIUS_KM", 0.02))
        )
        combined_sigma = np.sqrt(2.0) * effective_sigma
        pc_max_zero_miss = (
            float(-np.expm1(-(effective_radius ** 2) / (2.0 * combined_sigma ** 2)))
            if effective_sigma > 0 and effective_radius > 0 else float("nan")
        )
        high_threshold = float(getattr(config, "RISK_THRESHOLD_HIGH", 1e-4))
        high_threshold_reachable = bool(
            np.isfinite(pc_max_zero_miss) and pc_max_zero_miss >= high_threshold
        )
        ground_truth = str(event.ground_truth_outcome or "UNKNOWN").strip().upper()

        # Operational probability risk is deliberately based on the analytic
        # model probability, not a single noisy rare-event MC realization.
        # MC remains a matched-budget validation benchmark.
        rows.append(
            {
                "EVENT_ID": event.event_id,
                "GROUND_TRUTH_OUTCOME": ground_truth,
                "PREDICTION_TIME_IS_PRE_EVENT": bool(snapshot < event.event_time_utc),
                "SNAPSHOT_TIME": snapshot.isoformat(),
                "DAYS_BEFORE_EVENT": (event.event_time_utc - snapshot).total_seconds() / 86400.0,
                "NORAD_A": event.object_a_norad,
                "OBJECT_A": name_a,
                "TLE_EPOCH_A": epoch_a.isoformat(),
                "NORAD_B": event.object_b_norad,
                "OBJECT_B": name_b,
                "TLE_EPOCH_B": epoch_b.isoformat(),
                "FORECAST_HORIZON_DAYS": int(forecast_horizon_days),
                "FORECAST_TCA_GRID_STEP_MINUTES": int(forecast_step_minutes),
                "FORECAST_TCA_REFINEMENT_METHOD": "bounded_scalar_minimization_within_one_grid_step",
                "FORECAST_TCA": forecast_tca.isoformat(),
                "COARSE_BASELINE_TCA": baseline_tca.isoformat(),
                "COARSE_BASELINE_TCA_OFFSET_FROM_EVENT_HOURS": (
                    baseline_tca - event.event_time_utc
                ).total_seconds() / 3600.0,
                "COARSE_BASELINE_MISS_DISTANCE_KM": baseline_miss,
                "COARSE_BASELINE_RELATIVE_VELOCITY_KM_S": baseline_velocity,
                "REFINEMENT_ABS_TCA_OFFSET_IMPROVEMENT_SECONDS": (
                    abs((baseline_tca - event.event_time_utc).total_seconds())
                    - abs((forecast_tca - event.event_time_utc).total_seconds())
                ),
                "REFINEMENT_MISS_DISTANCE_CHANGE_KM": baseline_miss - forecast_miss,
                "FORECAST_TCA_OFFSET_FROM_EVENT_HOURS": (
                    forecast_tca - event.event_time_utc
                ).total_seconds() / 3600.0,
                "FORECAST_TCA_AT_OR_BEFORE_EVENT": bool(
                    forecast_tca <= event.event_time_utc
                ),
                "FORECAST_LEAD_TIME_DAYS": lead_time_days,
                "FORECAST_MISS_DISTANCE_KM": forecast_miss,
                "FORECAST_RELATIVE_VELOCITY_KM_S": forecast_velocity,
                "FORECAST_PROXIMITY_ALERT_LEVEL": proximity_alert,
                "ASSUMED_POSITION_SIGMA_PER_OBJECT_KM": effective_sigma,
                "COMBINED_ISOTROPIC_SIGMA_KM": float(combined_sigma),
                "HARD_BODY_RADIUS_KM": effective_radius,
                "ANALYTIC_PC": analytic_pc,
                "ANALYTIC_RISK_LEVEL": analytic_risk,
                # QAE estimates the same analytic probability target; expose its
                # risk label and error so it is not mistaken for an independent
                # orbit-risk model.
                "QAE_RISK_LEVEL": classify_risk(qae["qae_estimate"]),
                "QAE_MINUS_ANALYTIC_PC": float(qae["qae_estimate"]) - float(analytic_pc),
                "QAE_ABS_ERROR_FROM_ANALYTIC_PC": abs(
                    float(qae["qae_estimate"]) - float(analytic_pc)
                ),
                "MC_CI_CONTAINS_ANALYTIC_PC": bool(
                    float(mc["ci_low"]) <= float(analytic_pc) <= float(mc["ci_high"])
                ),
                "PC_MAX_AT_ZERO_MISS_UNDER_ASSUMED_SIGMA": pc_max_zero_miss,
                "HIGH_PC_THRESHOLD": high_threshold,
                "HIGH_THRESHOLD_REACHABLE_UNDER_ASSUMED_SIGMA": high_threshold_reachable,
                "MODEL_DIAGNOSTIC": (
                    "HIGH threshold unreachable under assumed sigma; validate covariance or document sensitivity analysis"
                    if not high_threshold_reachable else "HIGH threshold reachable under assumed sigma"
                ),
                "QAE_EVAL_QUBITS_REQUESTED": int(qae_eval_qubits),
                "QAE_EVAL_QUBITS_USED": int(effective_qae_qubits),
                "QAE_ESTIMATOR": qae["estimator"],
                "QAE_ESTIMATE": qae["qae_estimate"],
                "QAE_ERROR": qae["qae_error"],
                "QAE_ORACLE_CALLS": qae["oracle_calls"],
                "QAE_RUNTIME_SEC": qae["runtime_sec"],
                "MC_ESTIMATE": mc["mc_estimate"],
                "MC_ERROR": mc["mc_error"],
                "MC_SAMPLES": mc["n_samples"],
                "MC_HITS": mc["hits"],
                "MC_CI_LOW": mc["ci_low"],
                "MC_CI_HIGH": mc["ci_high"],
                "MC_RUNTIME_SEC": mc["runtime_sec"],
                "MC_RISK_LEVEL": mc_risk,
                # Operational/backward-compatible risk is deterministic
                # analytic probability risk, not MC sampling noise.
                "RISK_LEVEL": analytic_risk,
                "ACTUAL_EVENT_TIME": event.event_time_utc.isoformat(),
                "ACTUAL_EVENT": int(snapshot == event.event_time_utc),
                "ACTUAL_COLLISION_AT_EVENT_BOUNDARY": bool(
                    ground_truth == "CONFIRMED_COLLISION" and snapshot == event.event_time_utc
                ),
                "PRE_EVENT_PC_WARNING": bool(
                    snapshot < event.event_time_utc and analytic_risk in {"MEDIUM", "HIGH"}
                ),
                "PRE_EVENT_PROXIMITY_WARNING": bool(
                    snapshot < event.event_time_utc and proximity_alert in {"MEDIUM", "HIGH"}
                ),
            }
        )

        snapshot += snapshot_step

    output = pd.DataFrame(rows)

    # Warning lead times must be measured strictly before the event. The T-0
    # boundary is ground truth/evaluation, not a pre-event prediction.
    pre_event_output = output[output["PREDICTION_TIME_IS_PRE_EVENT"]]
    high_alert = pre_event_output[pre_event_output["FORECAST_PROXIMITY_ALERT_LEVEL"] == "HIGH"]
    medium_alert = pre_event_output[pre_event_output["FORECAST_PROXIMITY_ALERT_LEVEL"] == "MEDIUM"]
    high_probability = pre_event_output[pre_event_output["ANALYTIC_RISK_LEVEL"] == "HIGH"]
    medium_probability = pre_event_output[pre_event_output["ANALYTIC_RISK_LEVEL"] == "MEDIUM"]

    output.attrs["first_high_forecast_alert_days"] = (
        float(high_alert["DAYS_BEFORE_EVENT"].max()) if not high_alert.empty else None
    )
    output.attrs["first_medium_forecast_alert_days"] = (
        float(medium_alert["DAYS_BEFORE_EVENT"].max()) if not medium_alert.empty else None
    )
    output.attrs["first_high_probability_warning_days"] = (
        float(high_probability["DAYS_BEFORE_EVENT"].max()) if not high_probability.empty else None
    )
    output.attrs["first_medium_probability_warning_days"] = (
        float(medium_probability["DAYS_BEFORE_EVENT"].max()) if not medium_probability.empty else None
    )
    output.attrs["qae_qubit_upgrades"] = int(qae_upgrades)
    pre_event = output[output["PREDICTION_TIME_IS_PRE_EVENT"]]
    output.attrs["ground_truth_outcome"] = str(event.ground_truth_outcome or "UNKNOWN").upper()
    output.attrs["pre_event_pc_warning_count"] = int(pre_event["PRE_EVENT_PC_WARNING"].sum())
    output.attrs["pre_event_proximity_warning_count"] = int(pre_event["PRE_EVENT_PROXIMITY_WARNING"].sum())
    output.attrs["high_threshold_reachable"] = bool(
        output["HIGH_THRESHOLD_REACHABLE_UNDER_ASSUMED_SIGMA"].all()
    )

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run leakage-safe historical validation for Iridium 33 / Cosmos 2251"
    )
    parser.add_argument(
        "--tle",
        nargs="+",
        type=Path,
        required=True,
        help="Historical 2LE/3LE files from CelesTrak",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-time", default="2009-02-10T16:56:00Z")
    parser.add_argument("--rewind-days", type=int, default=30)
    parser.add_argument("--snapshot-step-hours", type=int, default=24)
    parser.add_argument("--forecast-step-minutes", type=int, default=30)
    parser.add_argument(
        "--forecast-horizon-days",
        type=int,
        default=35,
        help="Fixed forecast horizon, independent of the known event time",
    )
    parser.add_argument("--qae-eval-qubits", type=int, default=6)
    parser.add_argument("--qae-shots", type=int, default=200)
    parser.add_argument("--sigma-km", type=float, default=None)
    parser.add_argument("--hard-body-radius-km", type=float, default=None)
    args = parser.parse_args()

    event = HistoricalValidationEvent(
        event_id="iridium33_cosmos2251_2009",
        event_time_utc=utc_datetime(args.event_time),
        object_a_norad=24946,
        object_b_norad=22675,
    )

    archives = merge_tle_archives(args.tle)

    result = run_historical_validation(
        event,
        archives,
        rewind_days=args.rewind_days,
        snapshot_step_hours=args.snapshot_step_hours,
        forecast_step_minutes=args.forecast_step_minutes,
        forecast_horizon_days=args.forecast_horizon_days,
        qae_eval_qubits=args.qae_eval_qubits,
        qae_shots=args.qae_shots,
        sigma_km=args.sigma_km,
        hard_body_radius_km=args.hard_body_radius_km,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Historical validation rows: {len(result)}")
    print(f"Output: {args.output}")
    print()
    print(
        result[
            [
                "SNAPSHOT_TIME",
                "DAYS_BEFORE_EVENT",
                "FORECAST_HORIZON_DAYS",
                "FORECAST_TCA",
                "FORECAST_LEAD_TIME_DAYS",
                "FORECAST_MISS_DISTANCE_KM",
                "FORECAST_PROXIMITY_ALERT_LEVEL",
                "ANALYTIC_PC",
                "ANALYTIC_RISK_LEVEL",
                "QAE_EVAL_QUBITS_USED",
                "QAE_ESTIMATE",
                "MC_ESTIMATE",
                "RISK_LEVEL",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "First HIGH forecast-proximity alert lead time: "
        f"{result.attrs.get('first_high_forecast_alert_days')} days"
    )
    print(
        "First MEDIUM forecast-proximity alert lead time: "
        f"{result.attrs.get('first_medium_forecast_alert_days')} days"
    )
    print(
        "First HIGH probability warning lead time (analytic): "
        f"{result.attrs.get('first_high_probability_warning_days')} days"
    )
    print(
        "First MEDIUM probability warning lead time (analytic): "
        f"{result.attrs.get('first_medium_probability_warning_days')} days"
    )
    print(
        "QAE qubit upgrades for small probabilities: "
        f"{result.attrs.get('qae_qubit_upgrades', 0)}"
    )


if __name__ == "__main__":
    main()
