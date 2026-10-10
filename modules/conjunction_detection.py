import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import config


def find_conjunctions(propagated_df, screening_distance_km=None):
    """
    Take the shared-grid propagation output and return future
    conjunctions whose closest approach is within the screening
    distance during the configured forecast horizon.

    The propagation grid is expected to begin at the current
    forecast epoch and extend for FORECAST_HORIZON_DAYS. For each
    object pair, the closest sampled approach is retained.
    """
    if screening_distance_km is None:
        screening_distance_km = config.SCREENING_DISTANCE_KM

    if propagated_df is None or propagated_df.empty:
        return pd.DataFrame(columns=[
            "OBJECT_A", "NORAD_A", "OBJECT_B", "NORAD_B",
            "TCA", "DAYS_TO_TCA", "FORECAST_HORIZON_DAYS",
            "FORECAST_STATUS", "MISS_DISTANCE_KM",
            "RELATIVE_VELOCITY_KM_S",
        ])

    required_columns = {
        "NORAD_CAT_ID", "OBJECT_NAME", "TIME",
        "X_KM", "Y_KM", "Z_KM",
        "VX_KM_S", "VY_KM_S", "VZ_KM_S",
    }
    missing = required_columns - set(propagated_df.columns)
    if missing:
        raise ValueError(
            f"Propagated data is missing required columns: {sorted(missing)}"
        )

    df = propagated_df.copy()
    df["TIME"] = pd.to_datetime(df["TIME"], utc=True)
    df = df.sort_values(["NORAD_CAT_ID", "TIME"])

    # Spatial indexing avoids testing every pair at every timestamp.
    # This preserves the existing sampled-grid definition of a conjunction:
    # any pair within the screening distance at a shared sampled time is retained.
    # The search is still an approximation between grid points; it is not a
    # continuous-time collision guarantee.
    df = df.dropna(subset=[
        "NORAD_CAT_ID", "X_KM", "Y_KM", "Z_KM",
        "VX_KM_S", "VY_KM_S", "VZ_KM_S", "TIME",
    ])
    object_ids = df["NORAD_CAT_ID"].dropna().unique()
    print(f"Objects: {len(object_ids)} | spatial-index screening enabled")
    print(f"Forecast horizon: {config.FORECAST_HORIZON_DAYS} days")

    best_by_pair = {}
    forecast_start = pd.Timestamp(df["TIME"].min())
    # Group by timestamp; all successful propagated objects are indexed together
    # and only pairs spatially close at that timestamp receive exact distance and
    # relative-velocity calculations.
    for timestamp, frame in df.groupby("TIME", sort=True):
        if len(frame) < 2:
            continue
        frame = frame.drop_duplicates("NORAD_CAT_ID", keep="last")
        if len(frame) < 2:
            continue

        positions = frame[["X_KM", "Y_KM", "Z_KM"]].to_numpy(dtype=float)
        pairs = cKDTree(positions).query_pairs(
            r=float(screening_distance_km),
            output_type="ndarray",
        )
        if pairs.size == 0:
            continue

        ids = frame["NORAD_CAT_ID"].to_numpy()
        names = frame["OBJECT_NAME"].fillna("").astype(str).to_numpy()
        velocities = frame[["VX_KM_S", "VY_KM_S", "VZ_KM_S"]].to_numpy(dtype=float)
        for idx_a, idx_b in pairs:
            norad_a, norad_b = ids[idx_a], ids[idx_b]
            # Stable ordering prevents the same pair being recorded in reverse.
            if str(norad_a) > str(norad_b):
                idx_a, idx_b = idx_b, idx_a
                norad_a, norad_b = norad_b, norad_a
            delta_position = positions[idx_a] - positions[idx_b]
            distance = float(np.linalg.norm(delta_position))
            key = (int(norad_a), int(norad_b))
            previous = best_by_pair.get(key)
            if previous is not None and previous["MISS_DISTANCE_KM"] <= distance:
                continue

            delta_velocity = velocities[idx_a] - velocities[idx_b]
            relative_velocity = float(np.linalg.norm(delta_velocity))
            best_by_pair[key] = {
                "OBJECT_A": names[idx_a],
                "NORAD_A": int(norad_a),
                "OBJECT_B": names[idx_b],
                "NORAD_B": int(norad_b),
                "TCA": pd.Timestamp(timestamp).isoformat(),
                "DAYS_TO_TCA": max(
                    (pd.Timestamp(timestamp) - forecast_start).total_seconds()
                    / 86400.0,
                    0.0,
                ),
                "FORECAST_HORIZON_DAYS": config.FORECAST_HORIZON_DAYS,
                "FORECAST_STATUS": "FORECASTED_CONJUNCTION",
                "MISS_DISTANCE_KM": distance,
                "RELATIVE_VELOCITY_KM_S": relative_velocity,
            }

    results = list(best_by_pair.values())

    columns = [
        "OBJECT_A", "NORAD_A", "OBJECT_B", "NORAD_B",
        "TCA", "DAYS_TO_TCA", "FORECAST_HORIZON_DAYS",
        "FORECAST_STATUS", "MISS_DISTANCE_KM",
        "RELATIVE_VELOCITY_KM_S",
    ]

    if not results:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(results, columns=columns)
        .sort_values(["DAYS_TO_TCA", "MISS_DISTANCE_KM"])
        .reset_index(drop=True)
    )


def detect_and_save(propagated_df=None, output_path=None):
    output_path = output_path or config.CONJUNCTIONS_FILE
    config.ensure_dirs()

    if propagated_df is None:
        propagated_df = pd.read_csv(
            config.PROPAGATED_GRID_FILE,
            parse_dates=["TIME"],
        )

    conjunctions = find_conjunctions(propagated_df)
    conjunctions.to_csv(output_path, index=False)

    print(
        f"\n30-day forecast conjunctions found "
        f"(< {config.SCREENING_DISTANCE_KM} km): {len(conjunctions)}"
    )
    print(f"Output file: {output_path}")

    if len(conjunctions) > 0:
        print("\nForecasted closest approaches:")
        print(conjunctions.head(10).to_string(index=False))

    return conjunctions


if __name__ == "__main__":
    detect_and_save()
