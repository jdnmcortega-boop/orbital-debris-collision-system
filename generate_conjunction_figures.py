"""
Generate conjunction-analysis figures from Monte Carlo results.

Figures:
1. Number of conjunction pairs vs. miss distance at TCA
2. Closing speed vs. miss distance at TCA

Input:
    results/monte_carlo_results.csv

Output:
    results/figures/conjunction_pairs_vs_miss_distance.png
    results/figures/closing_speed_vs_miss_distance.png

The calculations use the already-computed TCA values,
miss distances, and relative velocities from the
Monte Carlo conjunction results.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_FILE = RESULTS_DIR / "monte_carlo_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_conjunction_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Monte Carlo results not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "OBJECT_A",
        "OBJECT_B",
        "TCA",
        "MISS_DISTANCE_KM",
        "RELATIVE_VELOCITY_KM_S",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # Convert numerical columns safely.

    df["MISS_DISTANCE_KM"] = pd.to_numeric(
        df["MISS_DISTANCE_KM"],
        errors="coerce"
    )

    df["RELATIVE_VELOCITY_KM_S"] = pd.to_numeric(
        df["RELATIVE_VELOCITY_KM_S"],
        errors="coerce"
    )

    # Remove invalid rows.

    df = df.dropna(
        subset=[
            "MISS_DISTANCE_KM",
            "RELATIVE_VELOCITY_KM_S",
        ]
    )

    df = df[
        (df["MISS_DISTANCE_KM"] >= 0)
        &
        (df["RELATIVE_VELOCITY_KM_S"] >= 0)
    ].copy()

    return df


# ============================================================
# FIGURE 1
# NUMBER OF CONJUNCTION PAIRS VS MISS DISTANCE
# ============================================================

def generate_conjunction_distance_figure(df):

    # --------------------------------------------------------
    # Distance bins
    # --------------------------------------------------------

    bins = [
        0,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        float("inf"),
    ]

    labels = [
        "0–1",
        "1–2",
        "2–5",
        "5–10",
        "10–20",
        "20–50",
        "50–100",
        "100+",
    ]

    df["MISS_DISTANCE_BIN"] = pd.cut(
        df["MISS_DISTANCE_KM"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    counts = (
        df["MISS_DISTANCE_BIN"]
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        counts.index.astype(str),
        counts.values,
    )

    ax.set_title(
        "Number of Conjunction Pairs vs. Miss Distance at TCA"
    )

    ax.set_xlabel(
        "Miss Distance at TCA (km)"
    )

    ax.set_ylabel(
        "Number of Conjunction Pairs"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / "conjunction_pairs_vs_miss_distance.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save the underlying figure data
    # --------------------------------------------------------

    figure_data = pd.DataFrame({
        "MISS_DISTANCE_RANGE_KM": labels,
        "NUMBER_OF_CONJUNCTION_PAIRS": counts.values,
    })

    csv_output = (
        RESULTS_DIR
        / "conjunction_pairs_vs_miss_distance.csv"
    )

    figure_data.to_csv(
        csv_output,
        index=False
    )

    print(
        f"[OK] Conjunction-distance figure: {output}"
    )

    print(
        f"[OK] Figure data: {csv_output}"
    )


# ============================================================
# FIGURE 2
# CLOSING SPEED VS MISS DISTANCE
# ============================================================

def generate_closing_speed_figure(df):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        df["MISS_DISTANCE_KM"],
        df["RELATIVE_VELOCITY_KM_S"],
        alpha=0.7,
    )

    ax.set_title(
        "Closing Speed vs. Miss Distance at TCA"
    )

    ax.set_xlabel(
        "Miss Distance at TCA (km)"
    )

    ax.set_ylabel(
        "Relative / Closing Speed at TCA (km/s)"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / "closing_speed_vs_miss_distance.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save the underlying figure data
    # --------------------------------------------------------

    figure_data = df[
        [
            "OBJECT_A",
            "OBJECT_B",
            "TCA",
            "MISS_DISTANCE_KM",
            "RELATIVE_VELOCITY_KM_S",
        ]
    ].copy()

    csv_output = (
        RESULTS_DIR
        / "closing_speed_vs_miss_distance.csv"
    )

    figure_data.to_csv(
        csv_output,
        index=False
    )

    print(
        f"[OK] Closing-speed figure: {output}"
    )

    print(
        f"[OK] Figure data: {csv_output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("GENERATING CONJUNCTION FIGURES")
    print("=" * 60)

    print(
        f"\nInput: {INPUT_FILE}"
    )

    df = load_conjunction_data()

    print(
        f"Conjunction pairs loaded: {len(df)}"
    )

    if len(df) == 0:

        raise ValueError(
            "No valid conjunction data found."
        )

    print(
        f"Miss-distance range: "
        f"{df['MISS_DISTANCE_KM'].min():.6f} – "
        f"{df['MISS_DISTANCE_KM'].max():.6f} km"
    )

    print(
        f"Relative-velocity range: "
        f"{df['RELATIVE_VELOCITY_KM_S'].min():.6f} – "
        f"{df['RELATIVE_VELOCITY_KM_S'].max():.6f} km/s"
    )

    print()

    generate_conjunction_distance_figure(
        df
    )

    generate_closing_speed_figure(
        df
    )

    print(
        "\nFigure generation complete."
    )


if __name__ == "__main__":
    main()