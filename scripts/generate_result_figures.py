"""
Generate publication-ready figures for the orbital-debris collision study.

Run from the repository root:
    python scripts/generate_result_figures.py

Outputs are written to:
    results/figures/

The script reads the existing result files rather than hard-coding experiment
observations. It generates one figure for each major Results subsection:
    Figure 1 - miss-distance distribution
    Figure 2 - relative velocity versus miss distance
    Figure 3 - QAE versus matched-budget Monte Carlo error
    Figure 4 - paired QAE versus MC error distribution (statistical comparison)
    Figure 5 - QKD security/detection benchmark
    Figure 6 - communication runtime and transmission overhead

PNG and PDF copies are generated for each figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# Keep figures clean and consistent for a research paper.
FIG_DPI = 300
FONT_SIZE = 11
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save both high-resolution PNG and vector PDF copies."""
    png = FIGURES / f"{name}.png"
    pdf = FIGURES / f"{name}.pdf"
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return pd.read_csv(path)


def read_json(name: str):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_conjunction_data() -> pd.DataFrame:
    """Load conjunction observations used by the current result set.

    Preferred source is closing_speed_vs_miss_distance.csv because it already
    contains the exact miss-distance/relative-velocity pairs used by the
    analysis. If the full processed conjunction table is available and has
    the required columns, it is used instead so Figure 1 can represent all
    candidate conjunctions.
    """
    processed = ROOT / "data" / "processed" / "conjunctions.csv"
    required = {"MISS_DISTANCE_KM"}

    if processed.exists():
        df = pd.read_csv(processed)
        if required.issubset(df.columns):
            # The processed table may contain additional columns, so only
            # require miss distance here. Figure 1 can use all rows.
            return df

    return read_csv("closing_speed_vs_miss_distance.csv")


def figure_1_miss_distance_distribution() -> None:
    df = get_conjunction_data()
    distances = pd.to_numeric(df["MISS_DISTANCE_KM"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(distances, bins=12, edgecolor="black", linewidth=0.8)
    ax.axvline(distances.min(), linestyle="--", linewidth=1.2,
               label=f"Minimum = {distances.min():.2f} km")
    ax.axvline(distances.mean(), linestyle=":", linewidth=1.4,
               label=f"Mean = {distances.mean():.2f} km")
    ax.set_xlabel("Miss distance (km)")
    ax.set_ylabel("Number of candidate conjunctions")
    ax.set_title("Distribution of Candidate Conjunction Miss Distances")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "figure_1_miss_distance_distribution")


def figure_2_velocity_vs_distance() -> None:
    df = read_csv("closing_speed_vs_miss_distance.csv")
    x = pd.to_numeric(df["MISS_DISTANCE_KM"], errors="coerce")
    y = pd.to_numeric(df["RELATIVE_VELOCITY_KM_S"], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x, y, s=42, alpha=0.85, edgecolors="black", linewidths=0.4)
    ax.set_xlabel("Miss distance (km)")
    ax.set_ylabel("Relative velocity (km/s)")
    ax.set_title("Relative Velocity versus Miss Distance")
    ax.grid(alpha=0.25)

    # Annotate the closest five observations to connect the plot to the
    # highest-interest conjunctions without overcrowding the figure.
    nearest = pd.DataFrame({"x": x, "y": y}).sort_values("x").head(5)
    for idx, row in nearest.iterrows():
        ax.annotate(
            f"{row.x:.1f} km",
            (row.x, row.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    save_figure(fig, "figure_2_relative_velocity_vs_miss_distance")


def figure_3_qae_vs_mc_error() -> None:
    df = read_csv("qae_accuracy_sweep.csv")
    df["TRUE_PROBABILITY"] = pd.to_numeric(df["TRUE_PROBABILITY"])
    df["EVAL_QUBITS"] = pd.to_numeric(df["EVAL_QUBITS"])
    df["ORACLE_CALLS"] = pd.to_numeric(df["ORACLE_CALLS"])
    df["QAE_ERROR_MEAN"] = pd.to_numeric(df["QAE_ERROR_MEAN"])
    df["MC_ERROR_MEAN"] = pd.to_numeric(df["MC_ERROR_MEAN"])

    probabilities = sorted(df["TRUE_PROBABILITY"].unique(), reverse=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for p in probabilities:
        sub = df[df["TRUE_PROBABILITY"] == p].sort_values("ORACLE_CALLS")
        label = f"p = {p:g}"
        ax.plot(sub["ORACLE_CALLS"], sub["QAE_ERROR_MEAN"], marker="o",
                linewidth=1.5, markersize=3.5, label=f"QAE, {label}")
        ax.plot(sub["ORACLE_CALLS"], sub["MC_ERROR_MEAN"], marker="x",
                linestyle="--", linewidth=1.1, markersize=3.5, alpha=0.8,
                label=f"MC, {label}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matched oracle/sample budget")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("QAE versus Matched-Budget Monte Carlo Accuracy")
    ax.grid(which="both", alpha=0.2)
    ax.legend(ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, "figure_3_qae_vs_mc_error")


def load_qae_paired_trials() -> pd.DataFrame | None:
    """Locate the raw paired QAE/MC benchmark if present."""
    candidates = [
        RESULTS / "qae_accuracy_sweep_raw_trials.csv",
        RESULTS / "qae_accuracy_sweep_trials.csv",
        RESULTS / "qae_comparison_trials.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            lower = {c.lower(): c for c in df.columns}
            qae_col = lower.get("qae_error")
            mc_col = lower.get("mc_error")
            if qae_col and mc_col:
                return df.rename(columns={qae_col: "QAE_ERROR", mc_col: "MC_ERROR"})
    return None


def figure_4_statistical_error_comparison() -> None:
    raw = load_qae_paired_trials()

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    if raw is not None:
        qae = pd.to_numeric(raw["QAE_ERROR"], errors="coerce").dropna()
        mc = pd.to_numeric(raw["MC_ERROR"], errors="coerce").dropna()
        data = [qae, mc]
        ax.boxplot(data, labels=["QAE", "Monte Carlo"], showfliers=False)
        ax.set_yscale("log")
        ax.set_ylabel("Absolute error (log scale)")
        ax.set_title("Paired QAE versus Monte Carlo Error")
        ax.grid(axis="y", alpha=0.2)
    else:
        # Fallback uses the 19-qubit sweep means. This keeps the figure
        # reproducible even when raw-trial files are not committed.
        sweep = read_csv("qae_accuracy_sweep.csv")
        sub = sweep[sweep["EVAL_QUBITS"] == 19]
        qae = pd.to_numeric(sub["QAE_ERROR_MEAN"])
        mc = pd.to_numeric(sub["MC_ERROR_MEAN"])
        ax.bar([0, 1], [qae.median(), mc.median()], width=0.55)
        ax.set_xticks([0, 1], ["QAE", "Monte Carlo"])
        ax.set_yscale("log")
        ax.set_ylabel("Median mean absolute error (log scale)")
        ax.set_title("19-Qubit Median Error Comparison")
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    save_figure(fig, "figure_4_qae_mc_statistical_comparison")


def figure_5_qkd_security() -> None:
    data = read_json("benchmark_summary.json")
    lookup = {row["condition"]: row for row in data}

    labels = ["Classical", "QKD\nhonest", "QKD\nintercept-resend"]
    delivery = [
        lookup["Classical (ECDH+AES-GCM)"]["delivery_rate"] * 100,
        lookup["QKD - honest channel"]["secure_delivery_rate"] * 100,
        lookup["QKD - eavesdropped (intercept-resend)"]["secure_delivery_rate"] * 100,
    ]
    detection = [
        np.nan,
        lookup["QKD - honest channel"]["detection_rate"] * 100,
        lookup["QKD - eavesdropped (intercept-resend)"]["detection_rate"] * 100,
    ]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2, delivery, width, label="Secure delivery rate")
    ax.bar(x + width / 2, np.nan_to_num(detection, nan=0.0), width,
           label="Attack detection rate")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Communication Security Benchmark")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    # Classical detection is not applicable because classical ECDH+AES-GCM
    # does not include a QKD-style eavesdropper detector.
    ax.text(x[0] + width / 2, 5, "N/A", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_figure(fig, "figure_5_qkd_security_benchmark")


def figure_6_communication_performance() -> None:
    data = read_json("benchmark_summary.json")
    lookup = {row["condition"]: row for row in data}

    labels = ["Classical\nECDH + AES-GCM", "QKD\nhonest", "QKD\nintercept-resend"]
    runtimes_ms = [
        lookup["Classical (ECDH+AES-GCM)"]["mean_runtime_sec"] * 1000,
        lookup["QKD - honest channel"]["mean_runtime_sec"] * 1000,
        lookup["QKD - eavesdropped (intercept-resend)"]["mean_runtime_sec"] * 1000,
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, runtimes_ms, width=0.55)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean runtime (ms)")
    ax.set_title("Communication Runtime Benchmark")
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, runtimes_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_figure(fig, "figure_6_communication_runtime")


def main() -> None:
    print(f"Repository root: {ROOT}")
    print(f"Figure output directory: {FIGURES}")

    figure_1_miss_distance_distribution()
    figure_2_velocity_vs_distance()
    figure_3_qae_vs_mc_error()
    figure_4_statistical_error_comparison()
    figure_5_qkd_security()
    figure_6_communication_performance()

    print("\nAll result figures generated successfully.")


if __name__ == "__main__":
    main()
