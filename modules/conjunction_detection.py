from itertools import combinations

import numpy as np
import pandas as pd

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

    object_ids = df["NORAD_CAT_ID"].dropna().unique()
    n_pairs = len(object_ids) * (len(object_ids) - 1) // 2
    print(f"Objects: {len(object_ids)} | Candidate pairs: {n_pairs}")
    print(f"Forecast horizon: {config.FORECAST_HORIZON_DAYS} days")

    by_object = {
        norad: g.set_index("TIME").sort_index()
        for norad, g in df.groupby("NORAD_CAT_ID")
    }

    results = []

    for norad_a, norad_b in combinations(object_ids, 2):
        a_full = by_object[norad_a]
        b_full = by_object[norad_b]

        common_times = a_full.index.intersection(b_full.index)
        if len(common_times) == 0:
            continue

        a = a_full.loc[common_times]
        b = b_full.loc[common_times]

        dx = a["X_KM"].to_numpy() - b["X_KM"].to_numpy()
        dy = a["Y_KM"].to_numpy() - b["Y_KM"].to_numpy()
        dz = a["Z_KM"].to_numpy() - b["Z_KM"].to_numpy()
        distances = np.sqrt(dx**2 + dy**2 + dz**2)

        min_idx = int(np.argmin(distances))
        min_distance = float(distances[min_idx])

        if min_distance > screening_distance_km:
            continue

        tca = pd.Timestamp(common_times[min_idx])

        # The forecast epoch is the earliest shared timestamp for this
        # pair. Since all objects use the same propagation grid, this is
        # normally the same timestamp for every pair.
        forecast_start = pd.Timestamp(common_times[0])
        days_to_tca = max(
            (tca - forecast_start).total_seconds() / 86400.0,
            0.0,
        )

        dvx = float(a["VX_KM_S"].to_numpy()[min_idx] - b["VX_KM_S"].to_numpy()[min_idx])
        dvy = float(a["VY_KM_S"].to_numpy()[min_idx] - b["VY_KM_S"].to_numpy()[min_idx])
        dvz = float(a["VZ_KM_S"].to_numpy()[min_idx] - b["VZ_KM_S"].to_numpy()[min_idx])
        relative_velocity = float(np.sqrt(dvx**2 + dvy**2 + dvz**2))

        results.append({
            "OBJECT_A": a["OBJECT_NAME"].iloc[0],
            "NORAD_A": norad_a,
            "OBJECT_B": b["OBJECT_NAME"].iloc[0],
            "NORAD_B": norad_b,
            "TCA": tca.isoformat(),
            "DAYS_TO_TCA": round(days_to_tca, 4),
            "FORECAST_HORIZON_DAYS": config.FORECAST_HORIZON_DAYS,
            "FORECAST_STATUS": "FORECASTED_CONJUNCTION",
            "MISS_DISTANCE_KM": min_distance,
            "RELATIVE_VELOCITY_KM_S": relative_velocity,
        })

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
