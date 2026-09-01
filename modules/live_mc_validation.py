"""Validate live QAE against a matched-budget direct Monte Carlo baseline.

The production live Monte Carlo estimator uses encounter-plane importance
sampling, which is intentionally efficient for rare collision events. That
estimator is useful operationally, but it is not the same as conventional
Monte Carlo sampling from the underlying Gaussian uncertainty distribution.

This module adds a separate direct encounter-plane Monte Carlo experiment.
For each live QAE evaluation, the direct MC baseline receives exactly the
same nominal QAE oracle-call budget. It samples the 2-D relative-position
Gaussian directly and counts samples falling inside the hard-body collision
disk.

This gives three clearly separated quantities:

    1. ANALYTIC_PC              deterministic model reference
    2. QAE_ESTIMATE             quantum amplitude-estimation result
    3. DIRECT_MC_MATCHED        conventional MC at matched budget
    4. MC_IS_PRODUCTION         operational importance-sampling MC

No result is used to alter the probability thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config


def wilson_interval(hits: int, n_samples: int, z: float = 1.96):
    n = int(n_samples)
    x = int(hits)
    if n <= 0:
        raise ValueError("n_samples must be positive")

    phat = x / n
    z2 = z ** 2
    denominator = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denominator
    margin = (
        z
        * np.sqrt(
            phat * (1.0 - phat) / n
            + z2 / (4.0 * n ** 2)
        )
        / denominator
    )

    return (
        float(max(0.0, center - margin)),
        float(min(1.0, center + margin)),
    )


def direct_encounter_plane_mc(
    miss_distance_km: float,
    sigma_a_km: float,
    sigma_b_km: float,
    hard_body_radius_km: float,
    n_samples: int,
    seed: int,
    chunk_size: int = 250_000,
):
    """Directly sample the 2-D Gaussian encounter plane and count collisions."""
    d = abs(float(miss_distance_km))
    sigma = float(np.sqrt(float(sigma_a_km) ** 2 + float(sigma_b_km) ** 2))
    radius = float(hard_body_radius_km)
    n = int(n_samples)

    if sigma <= 0.0:
        probability = 1.0 if d <= radius else 0.0
        return {
            "estimate": probability,
            "hits": 1 if probability > 0.0 else 0,
            "samples": n,
            "ci_low": probability,
            "ci_high": probability,
        }

    if radius <= 0.0:
        raise ValueError("hard_body_radius_km must be positive")
    if n <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    hits = 0
    remaining = n

    while remaining > 0:
        current = min(remaining, int(chunk_size))
        noise = rng.normal(0.0, sigma, size=(current, 2))
        dx = noise[:, 0] - d
        dy = noise[:, 1]
        hits += int(np.count_nonzero(dx * dx + dy * dy <= radius * radius))
        remaining -= current

    estimate = hits / float(n)
    ci_low, ci_high = wilson_interval(hits, n)

    return {
        "estimate": float(estimate),
        "hits": int(hits),
        "samples": n,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def build_validation(
    live_qae_df: pd.DataFrame,
    hard_body_radius_km: float,
    seed: int = 20260902,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    required = [
        "LIVE_PRIORITY_RANK",
        "OBJECT_A",
        "OBJECT_B",
        "TCA",
        "DAYS_TO_TCA",
        "MISS_DISTANCE_KM",
        "SIGMA_A_KM",
        "SIGMA_B_KM",
        "ANALYTIC_PC",
        "QAE_ESTIMATE",
        "QAE_ERROR",
        "QAE_ORACLE_CALLS",
        "QAE_EVAL_QUBITS_USED",
        "QAE_SHOTS",
        "MC_ESTIMATE",
        "MC_SAMPLES",
        "MC_METHOD",
    ]

    missing = [column for column in required if column not in live_qae_df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in live QAE comparison: "
            + ", ".join(missing)
        )

    rows = []

    for row_number, (_, row) in enumerate(live_qae_df.iterrows()):
        matched_samples = int(row["QAE_ORACLE_CALLS"])

        direct = direct_encounter_plane_mc(
            miss_distance_km=row["MISS_DISTANCE_KM"],
            sigma_a_km=row["SIGMA_A_KM"],
            sigma_b_km=row["SIGMA_B_KM"],
            hard_body_radius_km=hard_body_radius_km,
            n_samples=matched_samples,
            seed=seed + row_number,
            chunk_size=chunk_size,
        )

        analytic_p = float(row["ANALYTIC_PC"])
        qae_p = float(row["QAE_ESTIMATE"])
        is_p = float(row["MC_ESTIMATE"])
        direct_p = float(direct["estimate"])

        rows.append(
            {
                "LIVE_PRIORITY_RANK": int(row["LIVE_PRIORITY_RANK"]),
                "OBJECT_A": row["OBJECT_A"],
                "OBJECT_B": row["OBJECT_B"],
                "TCA": row["TCA"],
                "DAYS_TO_TCA": float(row["DAYS_TO_TCA"]),
                "MISS_DISTANCE_KM": float(row["MISS_DISTANCE_KM"]),
                "SIGMA_A_KM": float(row["SIGMA_A_KM"]),
                "SIGMA_B_KM": float(row["SIGMA_B_KM"]),
                "ANALYTIC_PC": analytic_p,
                "QAE_ESTIMATE": qae_p,
                "QAE_ERROR": float(row["QAE_ERROR"]),
                "QAE_RELATIVE_ERROR_PCT": (
                    abs(qae_p - analytic_p) / analytic_p * 100.0
                    if analytic_p > 0.0
                    else np.nan
                ),
                "QAE_EVAL_QUBITS_USED": int(row["QAE_EVAL_QUBITS_USED"]),
                "QAE_SHOTS": int(row["QAE_SHOTS"]),
                "QAE_ORACLE_CALLS": matched_samples,
                "DIRECT_MC_MATCHED_ESTIMATE": direct_p,
                "DIRECT_MC_MATCHED_HITS": direct["hits"],
                "DIRECT_MC_MATCHED_SAMPLES": direct["samples"],
                "DIRECT_MC_CI_LOW": direct["ci_low"],
                "DIRECT_MC_CI_HIGH": direct["ci_high"],
                "DIRECT_MC_RELATIVE_ERROR_PCT": (
                    abs(direct_p - analytic_p) / analytic_p * 100.0
                    if analytic_p > 0.0
                    else np.nan
                ),
                "MC_IS_PRODUCTION_ESTIMATE": is_p,
                "MC_IS_PRODUCTION_SAMPLES": int(row["MC_SAMPLES"]),
                "MC_IS_PRODUCTION_METHOD": row["MC_METHOD"],
                "MC_IS_VS_ANALYTIC_RELATIVE_ERROR_PCT": (
                    abs(is_p - analytic_p) / analytic_p * 100.0
                    if analytic_p > 0.0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run matched-budget direct MC validation against live QAE"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=config.RESULTS_DIR / "live_qae_comparison.csv",
        help="Live QAE comparison CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR / "live_mc_validation.csv",
        help="Validation output CSV",
    )
    parser.add_argument(
        "--hard-body-radius-km",
        type=float,
        default=getattr(config, "HARD_BODY_RADIUS_KM", 0.02),
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Live QAE comparison file not found: {args.input}. "
            "Run modules.live_qae first."
        )

    live_qae = pd.read_csv(args.input)
    result = build_validation(
        live_qae,
        hard_body_radius_km=args.hard_body_radius_km,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Live MC validation rows: {len(result)}")
    print(f"Output: {args.output}")
    print()
    print(
        result[
            [
                "LIVE_PRIORITY_RANK",
                "OBJECT_A",
                "OBJECT_B",
                "DAYS_TO_TCA",
                "ANALYTIC_PC",
                "QAE_ESTIMATE",
                "DIRECT_MC_MATCHED_ESTIMATE",
                "DIRECT_MC_MATCHED_HITS",
                "DIRECT_MC_MATCHED_SAMPLES",
                "MC_IS_PRODUCTION_ESTIMATE",
            ]
        ].to_string(index=False)
    )

    qae_mean_error = float(result["QAE_RELATIVE_ERROR_PCT"].replace([np.inf, -np.inf], np.nan).dropna().mean())
    direct_zero_count = int((result["DIRECT_MC_MATCHED_HITS"] == 0).sum())
    print()
    print(f"Mean QAE relative error: {qae_mean_error:.2f}%")
    print(f"Direct matched-budget MC zero-hit cases: {direct_zero_count}/{len(result)}")


if __name__ == "__main__":
    main()
