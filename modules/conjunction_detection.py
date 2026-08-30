from itertools import combinations

import numpy as np
import pandas as pd

import config


def find_conjunctions(propagated_df, screening_distance_km=None):
    """
    Take the shared-grid propagation output and return a DataFrame of
    object pairs whose closest approach is under the screening distance.
    """
    screening_distance_km = screening_distance_km or config.SCREENING_DISTANCE_KM

    object_ids = propagated_df["NORAD_CAT_ID"].unique()
    n_pairs = len(object_ids) * (len(object_ids) - 1) // 2
    print(f"Objects: {len(object_ids)} | Candidate pairs: {n_pairs}")

    by_object = {
        norad: g.set_index("TIME").sort_index()
        for norad, g in propagated_df.groupby("NORAD_CAT_ID")
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

        dx = a["X_KM"].values - b["X_KM"].values
        dy = a["Y_KM"].values - b["Y_KM"].values
        dz = a["Z_KM"].values - b["Z_KM"].values
        distances = np.sqrt(dx**2 + dy**2 + dz**2)

        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]

        if min_distance > screening_distance_km:
            continue

        tca = common_times[min_idx]

        dvx = a["VX_KM_S"].values[min_idx] - b["VX_KM_S"].values[min_idx]
        dvy = a["VY_KM_S"].values[min_idx] - b["VY_KM_S"].values[min_idx]
        dvz = a["VZ_KM_S"].values[min_idx] - b["VZ_KM_S"].values[min_idx]
        relative_velocity = np.sqrt(dvx**2 + dvy**2 + dvz**2)

        results.append({
            "OBJECT_A": a["OBJECT_NAME"].iloc[0],
            "NORAD_A": norad_a,
            "OBJECT_B": b["OBJECT_NAME"].iloc[0],
            "NORAD_B": norad_b,
            "TCA": tca,
            "MISS_DISTANCE_KM": min_distance,
            "RELATIVE_VELOCITY_KM_S": relative_velocity,
        })

    if not results:
        return pd.DataFrame(columns=[
            "OBJECT_A", "NORAD_A", "OBJECT_B", "NORAD_B",
            "TCA", "MISS_DISTANCE_KM", "RELATIVE_VELOCITY_KM_S",
        ])

    return pd.DataFrame(results).sort_values("MISS_DISTANCE_KM").reset_index(drop=True)


def detect_and_save(propagated_df=None, output_path=None):
    output_path = output_path or config.CONJUNCTIONS_FILE
    config.ensure_dirs()

    if propagated_df is None:
        propagated_df = pd.read_csv(config.PROPAGATED_GRID_FILE, parse_dates=["TIME"])

    conjunctions = find_conjunctions(propagated_df)
    conjunctions.to_csv(output_path, index=False)

    print(f"\nConjunctions found (< {config.SCREENING_DISTANCE_KM} km): {len(conjunctions)}")
    print(f"Output file: {output_path}")

    if len(conjunctions) > 0:
        print("\nClosest approaches:")
        print(conjunctions.head(10).to_string(index=False))

    return conjunctions


if __name__ == "__main__":
    detect_and_save()