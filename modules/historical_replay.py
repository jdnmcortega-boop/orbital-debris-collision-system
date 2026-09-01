"""Historical collision replay utilities.

This module builds a leakage-safe historical replay from archived GP/TLE data.
It does NOT invent orbital states and it does NOT use the known collision state
as an input to the prediction stage.

Input:
    One or more archived TLE/3LE files containing historical element sets for
    the two objects in a selected event.

Output:
    A timestamped state table covering the requested rewind/forecast window.
    Each snapshot uses the latest archived element set whose epoch is at or
    before that snapshot. This avoids propagating one stale TLE for 30 days.

The resulting state CSV is compatible with the column conventions used by
modules/conjunction_detection.py after propagation, while also retaining
historical replay metadata for later risk/QAE/Monte-Carlo evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS72, jday


@dataclass(frozen=True)
class HistoricalEvent:
    event_id: str
    event_time_utc: datetime
    object_a_norad: int
    object_b_norad: int


REQUIRED_STATE_COLUMNS = [
    "EVENT_ID",
    "SNAPSHOT_TIME",
    "DAYS_BEFORE_EVENT",
    "OBJECT_NAME",
    "NORAD_CAT_ID",
    "TLE_EPOCH",
    "X_KM",
    "Y_KM",
    "Z_KM",
    "VX_KM_S",
    "VY_KM_S",
    "VZ_KM_S",
]


def parse_tle_epoch(sat: Satrec) -> datetime:
    """Return a Satrec epoch as an aware UTC datetime."""
    year = int(sat.epochyr)
    year += 2000 if year < 57 else 1900
    day_of_year = float(sat.epochdays)
    day_int = int(day_of_year)
    fraction = day_of_year - day_int
    result = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day_int - 1,
        seconds=fraction * 86400.0,
    )
    return result


def parse_3le_file(path: Path) -> dict[int, list[tuple[datetime, str, str, Satrec]]]:
    """Parse a 2LE/3LE text archive into NORAD-indexed element histories.

    Accepted layouts:
      0 NAME
      1 TLE line 1
      2 TLE line 2

    or repeated pairs of TLE lines without a name line.
    """
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records: dict[int, list[tuple[datetime, str, str, Satrec]]] = {}

    i = 0
    while i < len(lines):
        if lines[i].startswith("1 "):
            name = "UNKNOWN"
            line1 = lines[i]
            if i + 1 >= len(lines) or not lines[i + 1].startswith("2 "):
                raise ValueError(f"Invalid TLE pair near line {i + 1} in {path}")
            line2 = lines[i + 1]
            i += 2
        else:
            if i + 2 >= len(lines):
                raise ValueError(f"Incomplete 3LE record near line {i + 1} in {path}")
            name = lines[i].strip()
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            if not line1.startswith("1 ") or not line2.startswith("2 "):
                raise ValueError(f"Invalid 3LE record near line {i + 1} in {path}")
            i += 3

        sat = Satrec.twoline2rv(line1, line2, WGS72)
        norad = int(sat.satnum)
        epoch = parse_tle_epoch(sat)
        records.setdefault(norad, []).append((epoch, name, line1, sat))

    for norad in records:
        records[norad].sort(key=lambda item: item[0])

    return records


def merge_tle_archives(paths: Iterable[Path]) -> dict[int, list[tuple[datetime, str, str, Satrec]]]:
    merged: dict[int, list[tuple[datetime, str, str, Satrec]]] = {}
    for path in paths:
        parsed = parse_3le_file(path)
        for norad, records in parsed.items():
            merged.setdefault(norad, []).extend(records)

    for norad in merged:
        # Same archive can be supplied twice; de-duplicate by epoch.
        unique = {record[0]: record for record in merged[norad]}
        merged[norad] = sorted(unique.values(), key=lambda item: item[0])

    return merged


def select_element_set(history, snapshot: datetime):
    """Select the freshest historical element set not newer than snapshot."""
    candidates = [record for record in history if record[0] <= snapshot]
    if not candidates:
        return None
    return candidates[-1]


def propagate_satellite(sat: Satrec, t: datetime):
    jd, fr = jday(
        t.year,
        t.month,
        t.day,
        t.hour,
        t.minute,
        t.second + t.microsecond / 1_000_000.0,
    )
    error, position, velocity = sat.sgp4(jd, fr)
    if error != 0:
        raise RuntimeError(f"SGP4 error code {error}")
    return np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)


def build_replay(
    event: HistoricalEvent,
    archives: dict[int, list[tuple[datetime, str, str, Satrec]]],
    rewind_days: int = 30,
    forecast_days: int = 0,
    step_hours: int = 24,
) -> pd.DataFrame:
    """Build pre-event and optional post-classification forecast snapshots."""
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    if rewind_days < 0 or forecast_days < 0:
        raise ValueError("rewind_days and forecast_days must be non-negative")

    for norad in (event.object_a_norad, event.object_b_norad):
        if norad not in archives:
            raise KeyError(
                f"NORAD {norad} is missing from the supplied historical TLE archives"
            )

    start = event.event_time_utc - timedelta(days=rewind_days)
    end = event.event_time_utc + timedelta(days=forecast_days)
    times = []
    current = start
    while current <= end:
        times.append(current)
        current += timedelta(hours=step_hours)

    rows = []
    for snapshot in times:
        for norad in (event.object_a_norad, event.object_b_norad):
            selected = select_element_set(archives[norad], snapshot)
            if selected is None:
                raise RuntimeError(
                    f"No historical TLE at or before {snapshot.isoformat()} for NORAD {norad}. "
                    "Supply an archive that reaches the 30-day rewind date."
                )

            tle_epoch, name, _, sat = selected
            position, velocity = propagate_satellite(sat, snapshot)

            rows.append({
                "EVENT_ID": event.event_id,
                "SNAPSHOT_TIME": snapshot.isoformat(),
                "DAYS_BEFORE_EVENT": (event.event_time_utc - snapshot).total_seconds() / 86400.0,
                "OBJECT_NAME": name,
                "NORAD_CAT_ID": norad,
                "TLE_EPOCH": tle_epoch.isoformat(),
                "X_KM": position[0],
                "Y_KM": position[1],
                "Z_KM": position[2],
                "VX_KM_S": velocity[0],
                "VY_KM_S": velocity[1],
                "VZ_KM_S": velocity[2],
            })

    return pd.DataFrame(rows, columns=REQUIRED_STATE_COLUMNS)


def add_pair_metrics(states: pd.DataFrame, event: HistoricalEvent) -> pd.DataFrame:
    """Calculate distance and relative velocity for every replay timestamp."""
    a = states[states["NORAD_CAT_ID"] == event.object_a_norad].set_index("SNAPSHOT_TIME")
    b = states[states["NORAD_CAT_ID"] == event.object_b_norad].set_index("SNAPSHOT_TIME")
    common = a.index.intersection(b.index)

    rows = []
    for timestamp in common:
        pa = a.loc[timestamp, ["X_KM", "Y_KM", "Z_KM"]].to_numpy(dtype=float)
        pb = b.loc[timestamp, ["X_KM", "Y_KM", "Z_KM"]].to_numpy(dtype=float)
        va = a.loc[timestamp, ["VX_KM_S", "VY_KM_S", "VZ_KM_S"]].to_numpy(dtype=float)
        vb = b.loc[timestamp, ["VX_KM_S", "VY_KM_S", "VZ_KM_S"]].to_numpy(dtype=float)

        rows.append({
            "EVENT_ID": event.event_id,
            "SNAPSHOT_TIME": timestamp,
            "DAYS_BEFORE_EVENT": float(a.loc[timestamp, "DAYS_BEFORE_EVENT"]),
            "OBJECT_A": a.loc[timestamp, "OBJECT_NAME"],
            "NORAD_A": event.object_a_norad,
            "OBJECT_B": b.loc[timestamp, "OBJECT_NAME"],
            "NORAD_B": event.object_b_norad,
            "RELATIVE_DISTANCE_KM": float(np.linalg.norm(pa - pb)),
            "RELATIVE_VELOCITY_KM_S": float(np.linalg.norm(va - vb)),
            "ACTUAL_EVENT": int(timestamp == event.event_time_utc.isoformat()),
        })

    return pd.DataFrame(rows).sort_values("SNAPSHOT_TIME").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Replay a historical orbital collision from archived TLEs")
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--event-time", required=True, help="UTC ISO timestamp, e.g. 2009-02-10T16:56:00Z")
    parser.add_argument("--norad-a", type=int, required=True)
    parser.add_argument("--norad-b", type=int, required=True)
    parser.add_argument("--tle", nargs="+", type=Path, required=True)
    parser.add_argument("--rewind-days", type=int, default=30)
    parser.add_argument("--forecast-days", type=int, default=0)
    parser.add_argument("--step-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-output", type=Path, required=True)
    args = parser.parse_args()

    event_time = datetime.fromisoformat(args.event_time.replace("Z", "+00:00"))
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    event = HistoricalEvent(args.event_id, event_time.astimezone(timezone.utc), args.norad_a, args.norad_b)

    archives = merge_tle_archives(args.tle)
    states = build_replay(
        event,
        archives,
        rewind_days=args.rewind_days,
        forecast_days=args.forecast_days,
        step_hours=args.step_hours,
    )
    pair = add_pair_metrics(states, event)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pair_output.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(args.output, index=False)
    pair.to_csv(args.pair_output, index=False)

    print(f"Historical state rows written: {len(states)}")
    print(f"Pair snapshot rows written: {len(pair)}")
    print(f"State output: {args.output}")
    print(f"Pair output: {args.pair_output}")


if __name__ == "__main__":
    main()
