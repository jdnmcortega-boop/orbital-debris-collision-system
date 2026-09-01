"""Historical validation of the Iridium 33 / Cosmos 2251 encounter.

This module is deliberately separate from the normal/live pipeline.
It replays the historical event using only archived TLE information that
would have been available at each replay timestamp.

For each daily snapshot from T-30 days through T-0:

1. Select the newest historical TLE whose epoch is not newer than the snapshot.
2. Propagate both objects from that snapshot toward the historical event time.
3. Find the predicted closest approach from that information state.
4. Calculate analytic collision probability from the predicted miss distance.
5. Estimate the same probability with QAE and matched-budget Monte Carlo.
6. Classify risk using the existing probability thresholds.
7. Record the forecast and the eventual historical outcome.

Important: the known collision state is NEVER used to construct a prediction.
The historical event time is used only as the evaluation boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Forecast the closest approach using only the two selected TLEs.

    The same TLE pair selected at ``start_time`` is propagated forward. This
    intentionally represents a forecast made from the information state at
    that timestamp rather than quietly switching to future historical TLEs.
    """
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")

    if end_time < start_time:
        raise ValueError("end_time must not be earlier than start_time")

    best_time = start_time
    best_distance = float("inf")
    best_relative_velocity = float("nan")

    current = start_time
    step = timedelta(minutes=step_minutes)

    while current <= end_time:
        pos_a, vel_a = propagate_satellite(sat_a, current)
        pos_b, vel_b = propagate_satellite(sat_b, current)

        relative_position = pos_a - pos_b
        relative_velocity = vel_a - vel_b
        distance = float(np.linalg.norm(relative_position))

        if distance < best_distance:
            best_distance = distance
            best_time = current
            best_relative_velocity = float(np.linalg.norm(relative_velocity))

        current += step

    # Always evaluate the exact event boundary even when it is not aligned
    # with the forecast step.
    if current - step != end_time:
        pos_a, vel_a = propagate_satellite(sat_a, end_time)
        pos_b, vel_b = propagate_satellite(sat_b, end_time)
        distance = float(np.linalg.norm(pos_a - pos_b))
        if distance < best_distance:
            best_distance = distance
            best_time = end_time
            best_relative_velocity = float(np.linalg.norm(vel_a - vel_b))

    return best_time, best_distance, best_relative_velocity


def run_historical_validation(
    event: HistoricalValidationEvent,
    archives,
    rewind_days: int = 30,
    snapshot_step_hours: int = 24,
    forecast_step_minutes: int = 30,
    qae_eval_qubits: int = 6,
    qae_shots: int = 200,
    sigma_km: float | None = None,
    hard_body_radius_km: float | None = None,
):
    """Run the complete leakage-safe historical validation experiment."""
    if rewind_days < 0:
        raise ValueError("rewind_days must be non-negative")
    if snapshot_step_hours <= 0:
        raise ValueError("snapshot_step_hours must be positive")

    for norad in (event.object_a_norad, event.object_b_norad):
        if norad not in archives:
            raise KeyError(f"NORAD {norad} is missing from the historical archive")

    start = event.event_time_utc - timedelta(days=rewind_days)
    snapshot_step = timedelta(hours=snapshot_step_hours)

    rows = []
    snapshot = start

    while snapshot <= event.event_time_utc:
        selected_a = select_element_set(archives[event.object_a_norad], snapshot)
        selected_b = select_element_set(archives[event.object_b_norad], snapshot)

        if selected_a is None or selected_b is None:
            raise RuntimeError(
                f"Historical archive does not contain a TLE at or before {snapshot.isoformat()} "
                "for both objects."
            )

        epoch_a, name_a, line1_a, sat_a = selected_a
        epoch_b, name_b, line1_b, sat_b = selected_b

        forecast_tca, forecast_miss, forecast_velocity = forecast_closest_approach(
            sat_a,
            sat_b,
            snapshot,
            event.event_time_utc,
            step_minutes=forecast_step_minutes,
        )

        analytic_pc = analytic_collision_probability(
            forecast_miss,
            sigma_km=sigma_km,
            hard_body_radius_km=hard_body_radius_km,
        )

        qae = run_qae(
            analytic_pc,
            num_eval_qubits=qae_eval_qubits,
            shots=qae_shots,
        )

        mc = run_classical_mc(
            analytic_pc,
            n_samples=qae["oracle_calls"],
            seed=42,
        )

        rows.append(
            {
                "EVENT_ID": event.event_id,
                "SNAPSHOT_TIME": snapshot.isoformat(),
                "DAYS_BEFORE_EVENT": (event.event_time_utc - snapshot).total_seconds() / 86400.0,
                "NORAD_A": event.object_a_norad,
                "OBJECT_A": name_a,
                "TLE_EPOCH_A": epoch_a.isoformat(),
                "NORAD_B": event.object_b_norad,
                "OBJECT_B": name_b,
                "TLE_EPOCH_B": epoch_b.isoformat(),
                "FORECAST_TCA": forecast_tca.isoformat(),
                "FORECAST_MISS_DISTANCE_KM": forecast_miss,
                "FORECAST_RELATIVE_VELOCITY_KM_S": forecast_velocity,
                "ANALYTIC_PC": analytic_pc,
                "QAE_ESTIMATE": qae["qae_estimate"],
                "QAE_ERROR": qae["qae_error"],
                "QAE_ORACLE_CALLS": qae["oracle_calls"],
                "QAE_RUNTIME_SEC": qae["runtime_sec"],
                "MC_ESTIMATE": mc["mc_estimate"],
                "MC_ERROR": mc["mc_error"],
                "MC_SAMPLES": mc["n_samples"],
                "MC_RUNTIME_SEC": mc["runtime_sec"],
                "RISK_LEVEL": classify_risk(mc["mc_estimate"]),
                "ACTUAL_EVENT_TIME": event.event_time_utc.isoformat(),
                "ACTUAL_EVENT": int(snapshot == event.event_time_utc),
            }
        )

        snapshot += snapshot_step

    output = pd.DataFrame(rows)

    # Earliest warning point is reported from the actual calculated risk
    # results; no forced HIGH classification is applied.
    high = output[output["RISK_LEVEL"] == "HIGH"]
    medium = output[output["RISK_LEVEL"] == "MEDIUM"]

    if not high.empty:
        first_warning = float(high["DAYS_BEFORE_EVENT"].max())
        output.attrs["first_high_warning_days"] = first_warning
    else:
        output.attrs["first_high_warning_days"] = None

    if not medium.empty:
        first_medium = float(medium["DAYS_BEFORE_EVENT"].max())
        output.attrs["first_medium_warning_days"] = first_medium
    else:
        output.attrs["first_medium_warning_days"] = None

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
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--event-time",
        default="2009-02-10T16:56:00Z",
    )
    parser.add_argument("--rewind-days", type=int, default=30)
    parser.add_argument("--snapshot-step-hours", type=int, default=24)
    parser.add_argument("--forecast-step-minutes", type=int, default=30)
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
                "FORECAST_TCA",
                "FORECAST_MISS_DISTANCE_KM",
                "ANALYTIC_PC",
                "QAE_ESTIMATE",
                "MC_ESTIMATE",
                "RISK_LEVEL",
            ]
        ].to_string(index=False)
    )

    first_high = result.attrs.get("first_high_warning_days")
    first_medium = result.attrs.get("first_medium_warning_days")

    print()
    print(f"First HIGH warning lead time: {first_high} days")
    print(f"First MEDIUM warning lead time: {first_medium} days")


if __name__ == "__main__":
    main()
