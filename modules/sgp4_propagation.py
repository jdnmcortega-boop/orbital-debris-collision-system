from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS72, jday

import config


def datetime_to_sgp4_epoch(epoch):
    reference = datetime(1949, 12, 31, tzinfo=timezone.utc)
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    return (epoch - reference).total_seconds() / 86400.0


def create_satellite(row):
    """Build a Satrec object from one row of orbital elements."""
    satellite = Satrec()

    epoch_datetime = pd.to_datetime(row["EPOCH"], utc=True).to_pydatetime()
    epoch = datetime_to_sgp4_epoch(epoch_datetime)

    mean_motion = float(row["MEAN_MOTION"]) * 2.0 * np.pi / 1440.0
    mean_motion_dot = float(row["MEAN_MOTION_DOT"]) * 2.0 * np.pi / (1440.0 ** 2)
    mean_motion_ddot = float(row["MEAN_MOTION_DDOT"]) * 2.0 * np.pi / (1440.0 ** 3)

    satellite.sgp4init(
        WGS72,
        "i",
        int(row["NORAD_CAT_ID"]),
        epoch,
        float(row["BSTAR"]),
        mean_motion_dot,
        mean_motion_ddot,
        float(row["ECCENTRICITY"]),
        np.radians(float(row["ARG_OF_PERICENTER"])),
        np.radians(float(row["INCLINATION"])),
        np.radians(float(row["MEAN_ANOMALY"])),
        mean_motion,
        np.radians(float(row["RA_OF_ASC_NODE"])),
    )
    return satellite


def build_time_grid(start=None, duration_hours=None, step_minutes=None):
    start = config.GRID_START if start is None else start
    duration_hours = config.GRID_DURATION_HOURS if duration_hours is None else duration_hours
    step_minutes = config.GRID_STEP_MINUTES if step_minutes is None else step_minutes

    steps = int((duration_hours * 60) / step_minutes)
    return [start + timedelta(minutes=step_minutes * i) for i in range(steps + 1)]


def propagate_row_over_grid(row, times):
    """Propagate one object to every timestamp in `times`."""
    satellite = create_satellite(row)
    results = []

    for t in times:
        jd, fr = jday(
            t.year, t.month, t.day, t.hour, t.minute,
            t.second + t.microsecond / 1_000_000
        )

        error, position, velocity = satellite.sgp4(jd, fr)

        if error != 0:
            continue

        results.append({
            "OBJECT_NAME": row["OBJECT_NAME"],
            "OBJECT_ID": row["OBJECT_ID"],
            "NORAD_CAT_ID": row["NORAD_CAT_ID"],
            "SOURCE_EPOCH": pd.Timestamp(row["EPOCH"]).isoformat(),
            "FORECAST_START": pd.Timestamp(config.GRID_START).isoformat(),
            "FORECAST_HORIZON_DAYS": config.FORECAST_HORIZON_DAYS,
            "TIME": t.isoformat(),
            "X_KM": position[0],
            "Y_KM": position[1],
            "Z_KM": position[2],
            "VX_KM_S": velocity[0],
            "VY_KM_S": velocity[1],
            "VZ_KM_S": velocity[2],
        })

    return results


def propagate_all(data, times=None, verbose=True):
    """
    Propagate every selected forecast-input object across the same
    shared future time grid.
    """
    times = build_time_grid() if times is None else times

    all_results = []
    failed = []

    for _, row in data.iterrows():
        try:
            rows = propagate_row_over_grid(row, times)
            if not rows:
                raise RuntimeError("no valid propagation points")
            all_results.extend(rows)
            if verbose:
                print(f"[OK] {row['OBJECT_NAME']} -> {len(rows)} points")
        except Exception as error:
            failed.append({"OBJECT_NAME": row["OBJECT_NAME"], "ERROR": str(error)})
            if verbose:
                print(f"[FAILED] {row['OBJECT_NAME']}: {error}")

    return pd.DataFrame(all_results), failed


def propagate_and_save(data, output_path=None):
    output_path = output_path or config.PROPAGATED_GRID_FILE
    config.ensure_dirs()

    result_df, failed = propagate_all(data)

    for col in ["X_KM", "Y_KM", "Z_KM", "VX_KM_S", "VY_KM_S", "VZ_KM_S"]:
        if col in result_df.columns:
            result_df[col] = result_df[col].round(4)

    result_df.to_csv(output_path, index=False)

    print(
        f"\nForecast mode: {config.FORECAST_MODE.upper()}"
        f" | Start: {config.GRID_START.isoformat()}"
        f" | Horizon: {config.FORECAST_HORIZON_DAYS} days"
    )
    print(f"Rows written: {len(result_df)} | Failed objects: {len(failed)}")
    print(f"Output file: {output_path}")

    return result_df, failed


def propagate_all_now(data, verbose=False):
    """Compute every object's position at the real current UTC time."""
    now = datetime.now(timezone.utc)
    return propagate_all(data, times=[now], verbose=verbose)


if __name__ == "__main__":
    from modules import data_loader

    df = data_loader.load_forecast_orbital_data()
    propagate_and_save(df)
