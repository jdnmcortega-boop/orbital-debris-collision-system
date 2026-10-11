"""Continuous-time conjunction screening over a sampled SGP4 state grid.

The grid is a screening aid, not the final TCA. Candidate pairs are collected
within a conservative interval-expanded radius, then each candidate's relative
trajectory is refined between adjacent samples with cubic Hermite interpolation
using the propagated positions and velocities. Events are retained only when
the refined miss distance is inside the configured screening radius.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

import config


_POSITION_COLUMNS = ["X_KM", "Y_KM", "Z_KM"]
_VELOCITY_COLUMNS = ["VX_KM_S", "VY_KM_S", "VZ_KM_S"]


def _hermite_position(r0, v0, r1, v1, duration_s, u):
    """Cubic-Hermite relative position for normalized segment time u in [0, 1]."""
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return (
        h00 * r0
        + h10 * duration_s * v0
        + h01 * r1
        + h11 * duration_s * v1
    )


def _refine_pair_tca(frame_a, frame_b, sampled_time):
    """Refine a sampled minimum over its neighboring intervals.

    Returns (TCA, miss distance km, relative speed km/s). Hermite interpolation
    uses the SGP4-propagated endpoint positions and velocities. If neighboring
    states are unavailable, the sampled state is returned as a conservative
    fallback and explicitly marked by the caller.
    """
    times_a = pd.DatetimeIndex(frame_a.index)
    times_b = pd.DatetimeIndex(frame_b.index)
    if not times_a.equals(times_b) or len(times_a) < 2:
        raise ValueError("Pair trajectories do not share a complete time grid.")

    center_idx = int(times_a.get_indexer([sampled_time])[0])
    if center_idx < 0:
        raise ValueError("Sampled TCA timestamp is absent from pair trajectory.")

    best = None
    segment_indices = []
    if center_idx > 0:
        segment_indices.append((center_idx - 1, center_idx))
    if center_idx + 1 < len(times_a):
        segment_indices.append((center_idx, center_idx + 1))

    for left_idx, right_idx in segment_indices:
        t0 = times_a[left_idx]
        t1 = times_a[right_idx]
        duration_s = (t1 - t0).total_seconds()
        if duration_s <= 0:
            continue

        a0 = frame_a.iloc[left_idx]
        a1 = frame_a.iloc[right_idx]
        b0 = frame_b.iloc[left_idx]
        b1 = frame_b.iloc[right_idx]

        r0 = a0[_POSITION_COLUMNS].to_numpy(dtype=float) - b0[_POSITION_COLUMNS].to_numpy(dtype=float)
        r1 = a1[_POSITION_COLUMNS].to_numpy(dtype=float) - b1[_POSITION_COLUMNS].to_numpy(dtype=float)
        v0 = a0[_VELOCITY_COLUMNS].to_numpy(dtype=float) - b0[_VELOCITY_COLUMNS].to_numpy(dtype=float)
        v1 = a1[_VELOCITY_COLUMNS].to_numpy(dtype=float) - b1[_VELOCITY_COLUMNS].to_numpy(dtype=float)

        def distance_squared(u):
            relative_position = _hermite_position(r0, v0, r1, v1, duration_s, float(u))
            return float(np.dot(relative_position, relative_position))

        result = minimize_scalar(
            distance_squared,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        candidates = [(0.0, distance_squared(0.0)), (1.0, distance_squared(1.0))]
        if result.success:
            candidates.append((float(result.x), float(result.fun)))
        u_best, d2_best = min(candidates, key=lambda item: item[1])
        if best is None or d2_best < best[0]:
            tca = t0 + pd.to_timedelta(u_best * duration_s, unit="s")
            relative_position = _hermite_position(r0, v0, r1, v1, duration_s, u_best)
            relative_velocity = (
                (1.0 - u_best) * v0 + u_best * v1
            )
            best = (
                d2_best,
                tca,
                float(np.linalg.norm(relative_position)),
                float(np.linalg.norm(relative_velocity)),
            )

    if best is None:
        raise ValueError("No usable interval for TCA refinement.")
    return best[1], best[2], best[3]


def find_conjunctions(propagated_df, screening_distance_km=None):
    """Find and refine close approaches within the forecast horizon."""
    if screening_distance_km is None:
        screening_distance_km = config.SCREENING_DISTANCE_KM
    screening_distance_km = float(screening_distance_km)
    if not np.isfinite(screening_distance_km) or screening_distance_km <= 0:
        raise ValueError("screening_distance_km must be a finite positive number.")

    columns = [
        "OBJECT_A", "NORAD_A", "OBJECT_B", "NORAD_B",
        "TCA", "DAYS_TO_TCA", "FORECAST_HORIZON_DAYS",
        "FORECAST_STATUS", "MISS_DISTANCE_KM",
        "RELATIVE_VELOCITY_KM_S", "SAMPLED_MISS_DISTANCE_KM",
        "TCA_REFINEMENT_METHOD",
    ]
    if propagated_df is None or propagated_df.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "NORAD_CAT_ID", "OBJECT_NAME", "TIME",
        *_POSITION_COLUMNS, *_VELOCITY_COLUMNS,
    }
    missing = required - set(propagated_df.columns)
    if missing:
        raise ValueError(
            f"Propagated data is missing required columns: {sorted(missing)}"
        )

    df = propagated_df.copy()
    df["TIME"] = pd.to_datetime(df["TIME"], utc=True, errors="coerce")
    for column in _POSITION_COLUMNS + _VELOCITY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"], errors="coerce")
    df = df.dropna(subset=["NORAD_CAT_ID", "TIME"] + _POSITION_COLUMNS + _VELOCITY_COLUMNS)
    df["NORAD_CAT_ID"] = df["NORAD_CAT_ID"].astype("int64")
    df = df.sort_values(["NORAD_CAT_ID", "TIME"]).drop_duplicates(
        ["NORAD_CAT_ID", "TIME"], keep="last"
    )
    if df.empty:
        return pd.DataFrame(columns=columns)

    object_frames = {
        int(object_id): part.set_index("TIME").sort_index()
        for object_id, part in df.groupby("NORAD_CAT_ID", sort=False)
    }
    forecast_start = df["TIME"].min()
    time_groups = list(df.groupby("TIME", sort=True))
    if len(time_groups) < 2:
        return pd.DataFrame(columns=columns)

    intervals = [
        (time_groups[i + 1][0] - time_groups[i][0]).total_seconds()
        for i in range(len(time_groups) - 1)
    ]
    positive_intervals = [value for value in intervals if value > 0]
    if not positive_intervals:
        return pd.DataFrame(columns=columns)
    half_step_s = float(np.median(positive_intervals)) / 2.0

    # Bound the maximum relative speed by twice the largest propagated object
    # speed. This expands screening enough to catch a close pass between grid
    # samples; refined TCA still must pass the original screening distance.
    speeds = np.linalg.norm(df[_VELOCITY_COLUMNS].to_numpy(dtype=float), axis=1)
    max_object_speed = float(np.nanmax(speeds)) if len(speeds) else 0.0
    candidate_radius_km = screening_distance_km + 2.0 * max_object_speed * half_step_s
    print(f"Objects: {df['NORAD_CAT_ID'].nunique()} | interval-expanded spatial screening enabled")
    print(f"Forecast horizon: {config.FORECAST_HORIZON_DAYS} days")
    print(
        f"Sample interval median: {2.0 * half_step_s:.1f} s | "
        f"candidate radius: {candidate_radius_km:.1f} km | "
        f"final refined limit: {screening_distance_km:.1f} km"
    )

    # Keep the best sampled state for each candidate pair. The expanded radius
    # is only a broad-phase filter; it does not itself create a conjunction.
    best_by_pair = {}
    for timestamp, frame in time_groups:
        if len(frame) < 2:
            continue
        frame = frame.drop_duplicates("NORAD_CAT_ID", keep="last")
        if len(frame) < 2:
            continue
        positions = frame[_POSITION_COLUMNS].to_numpy(dtype=float)
        pairs = cKDTree(positions).query_pairs(
            r=candidate_radius_km,
            output_type="ndarray",
        )
        if pairs.size == 0:
            continue
        ids = frame["NORAD_CAT_ID"].to_numpy(dtype=np.int64)
        names = frame["OBJECT_NAME"].fillna("").astype(str).to_numpy()
        velocities = frame[_VELOCITY_COLUMNS].to_numpy(dtype=float)
        for idx_a, idx_b in pairs:
            norad_a = int(ids[idx_a])
            norad_b = int(ids[idx_b])
            if norad_a == norad_b:
                continue
            if norad_a > norad_b:
                idx_a, idx_b = idx_b, idx_a
                norad_a, norad_b = norad_b, norad_a
            delta_position = positions[idx_a] - positions[idx_b]
            sampled_distance = float(np.linalg.norm(delta_position))
            key = (norad_a, norad_b)
            previous = best_by_pair.get(key)
            if previous is not None and previous["SAMPLED_MISS_DISTANCE_KM"] <= sampled_distance:
                continue
            delta_velocity = velocities[idx_a] - velocities[idx_b]
            best_by_pair[key] = {
                "OBJECT_A": names[idx_a],
                "NORAD_A": norad_a,
                "OBJECT_B": names[idx_b],
                "NORAD_B": norad_b,
                "SAMPLED_TIME": pd.Timestamp(timestamp),
                "SAMPLED_MISS_DISTANCE_KM": sampled_distance,
                "SAMPLED_RELATIVE_VELOCITY_KM_S": float(np.linalg.norm(delta_velocity)),
            }

    results = []
    for pair in best_by_pair.values():
        norad_a = pair["NORAD_A"]
        norad_b = pair["NORAD_B"]
        try:
            tca, miss_distance, relative_speed = _refine_pair_tca(
                object_frames[norad_a],
                object_frames[norad_b],
                pair["SAMPLED_TIME"],
            )
            method = "CUBIC_HERMITE_LOCAL_TCA"
        except (KeyError, ValueError, IndexError, TypeError):
            # Do not silently claim a refined result if interpolation failed.
            # The fallback is retained only when the sampled separation itself
            # satisfies the configured screening limit.
            tca = pair["SAMPLED_TIME"]
            miss_distance = pair["SAMPLED_MISS_DISTANCE_KM"]
            relative_speed = pair["SAMPLED_RELATIVE_VELOCITY_KM_S"]
            method = "SAMPLED_FALLBACK"

        if not np.isfinite(miss_distance) or miss_distance > screening_distance_km:
            continue

        results.append({
            "OBJECT_A": pair["OBJECT_A"],
            "NORAD_A": norad_a,
            "OBJECT_B": pair["OBJECT_B"],
            "NORAD_B": norad_b,
            "TCA": pd.Timestamp(tca).isoformat(),
            "DAYS_TO_TCA": max(
                (pd.Timestamp(tca) - forecast_start).total_seconds() / 86400.0,
                0.0,
            ),
            "FORECAST_HORIZON_DAYS": config.FORECAST_HORIZON_DAYS,
            "FORECAST_STATUS": "FORECASTED_CONJUNCTION",
            "MISS_DISTANCE_KM": float(miss_distance),
            "RELATIVE_VELOCITY_KM_S": float(relative_speed),
            "SAMPLED_MISS_DISTANCE_KM": float(pair["SAMPLED_MISS_DISTANCE_KM"]),
            "TCA_REFINEMENT_METHOD": method,
        })

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
        propagated_df = pd.read_csv(config.PROPAGATED_GRID_FILE, parse_dates=["TIME"])

    conjunctions = find_conjunctions(propagated_df)
    conjunctions.to_csv(output_path, index=False)
    print(
        f"\n30-day forecast conjunctions found "
        f"(< {config.SCREENING_DISTANCE_KM} km after TCA refinement): {len(conjunctions)}"
    )
    print(f"Output file: {output_path}")
    if len(conjunctions) > 0:
        print("\nForecasted closest approaches:")
        print(conjunctions.head(10).to_string(index=False))
    return conjunctions


if __name__ == "__main__":
    detect_and_save()
