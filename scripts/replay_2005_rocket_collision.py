"""Reconstruct the 17 Jan 2005 debris-to-rocket-body collision.

This script uses only archived pre-event TLEs and SGP4. The known collision
record is used as the validation timestamp, not as an input to prediction.

Run from the repository root:
    python scripts/replay_2005_rocket_collision.py

The script writes a one-minute pre-event reconstruction to
results/historical_rocket_2005/.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from modules.historical_replay import merge_tle_archives, propagate_satellite, select_element_set


ROOT = Path(__file__).resolve().parents[1]
EVENT_TIME = datetime(2005, 1, 17, 2, 14, tzinfo=timezone.utc)
WINDOW_START = EVENT_TIME - timedelta(hours=6)
WINDOW_END = EVENT_TIME

TARGET_NORAD = 7219
PROJECTILE_NORAD = 26207
TARGET_TLE = ROOT / "data" / "historical_tle" / "dmsp_7219_event_2005-01-17.tle"
PROJECTILE_TLE = ROOT / "data" / "historical_tle" / "cz4_debris_26207_event_2005-01-17.tle"
OUTPUT_DIR = ROOT / "results" / "historical_rocket_2005"


def main() -> None:
    archives = merge_tle_archives([TARGET_TLE, PROJECTILE_TLE])

    rows = []
    t = WINDOW_START
    while t <= WINDOW_END:
        states = {}
        for norad in (TARGET_NORAD, PROJECTILE_NORAD):
            selected = select_element_set(archives[norad], t)
            if selected is None:
                raise RuntimeError(f"No archived TLE available for NORAD {norad} at {t.isoformat()}")
            epoch, name, line1, sat = selected
            position, velocity = propagate_satellite(sat, t)
            states[norad] = (name, epoch, line1, position, velocity)

        target = states[TARGET_NORAD]
        projectile = states[PROJECTILE_NORAD]
        dr = projectile[3] - target[3]
        dv = projectile[4] - target[4]

        rows.append({
            "TIME_UTC": t.isoformat(),
            "TARGET_NORAD": TARGET_NORAD,
            "PROJECTILE_NORAD": PROJECTILE_NORAD,
            "TARGET_TLE_EPOCH": target[1].isoformat(),
            "PROJECTILE_TLE_EPOCH": projectile[1].isoformat(),
            "SEPARATION_KM": float(np.linalg.norm(dr)),
            "RELATIVE_SPEED_KM_S": float(np.linalg.norm(dv)),
            "IS_VALIDATION_EVENT_TIME": int(t == EVENT_TIME),
        })
        t += timedelta(minutes=1)

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "pre_event_window_1min.csv"
    df.to_csv(output, index=False)

    event_row = df.loc[df["IS_VALIDATION_EVENT_TIME"] == 1].iloc[0]
    closest = df.loc[df["SEPARATION_KM"].idxmin()]

    print("2005 DMSP 5B F5 / CZ-4 debris reconstruction")
    print("=" * 54)
    print(f"Validation timestamp:       {EVENT_TIME.isoformat()}")
    print(f"SGP4 separation at event:   {event_row['SEPARATION_KM']:.6f} km")
    print(f"SGP4 relative speed:        {event_row['RELATIVE_SPEED_KM_S']:.6f} km/s")
    print(f"Closest sampled separation: {closest['SEPARATION_KM']:.6f} km")
    print(f"Closest sampled time:       {closest['TIME_UTC']}")
    print("NASA reference altitude:    885 km")
    print("NASA reported speed:        just under 6 km/s")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
