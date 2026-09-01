"""Run QAE alongside the live 30-day conjunction/MC forecast.

The live pipeline already computes collision probability with the analytic
encounter-plane model and estimates it classically with importance-sampling
Monte Carlo. This module takes those same forecasted conjunctions and runs
QPE-based QAE against the same analytic probability target, producing a
single aligned comparison table.

The default execution evaluates the highest-priority conjunctions only so a
normal live run remains practical. Use --limit 0 to evaluate every forecast
conjunction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from modules.prediction import composite_risk_score
from modules.qae import analytic_collision_probability, run_qae


def adaptive_qae_qubits(
    probability: float,
    requested_qubits: int,
    max_qubits: int = 14,
) -> int:
    """Choose enough QAE evaluation qubits for a small probability."""
    probability = float(np.clip(probability, 0.0, 1.0))
    requested = max(1, int(requested_qubits))
    maximum = max(requested, int(max_qubits))

    if probability <= 0.0:
        return requested

    m = requested
    while m < maximum:
        first_bin = np.sin(np.pi / (2 ** m)) ** 2
        if probability >= first_bin:
            break
        m += 1

    return m


def required_columns():
    return [
        "OBJECT_A",
        "OBJECT_B",
        "NORAD_A",
        "NORAD_B",
        "TCA",
        "DAYS_TO_TCA",
        "FORECAST_HORIZON_DAYS",
        "MISS_DISTANCE_KM",
        "RELATIVE_VELOCITY_KM_S",
        "SIGMA_A_KM",
        "SIGMA_B_KM",
        "ALTITUDE_DIFFERENCE_KM",
        "INCLINATION_DIFFERENCE_DEG",
        "COLLISION_PROBABILITY_MC",
        "MC_CI_LOW",
        "MC_CI_HIGH",
        "COMPOSITE_RISK_SCORE",
        "COMPOSITE_RISK_LEVEL",
    ]


def build_live_qae_comparison(
    mc_results: pd.DataFrame,
    requested_qubits: int = 6,
    shots: int = 200,
    limit: int = 20,
) -> pd.DataFrame:
    """Run QAE on the prioritized live conjunctions."""
    missing = [c for c in required_columns() if c not in mc_results.columns]
    if missing:
        raise ValueError(
            "Missing required columns in Monte Carlo results: "
            + ", ".join(missing)
        )

    df = mc_results.copy()

    # Ensure the priority order is deterministic even if prediction.py has
    # not yet been run. Prefer its composite score when present.
    if "COMPOSITE_RISK_SCORE" not in df.columns:
        df["COMPOSITE_RISK_SCORE"] = df.apply(
            lambda row: composite_risk_score(
                probability=row["COLLISION_PROBABILITY_MC"],
                relative_velocity_km_s=row["RELATIVE_VELOCITY_KM_S"],
                miss_distance_km=row["MISS_DISTANCE_KM"],
                sigma_a_km=row["SIGMA_A_KM"],
                sigma_b_km=row["SIGMA_B_KM"],
                altitude_difference_km=row["ALTITUDE_DIFFERENCE_KM"],
                inclination_difference_deg=row["INCLINATION_DIFFERENCE_DEG"],
            ),
            axis=1,
        )

    ordered = df.sort_values(
        ["COMPOSITE_RISK_SCORE", "DAYS_TO_TCA"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if limit > 0:
        ordered = ordered.head(int(limit)).copy()

    rows = []

    for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
        analytic_p = analytic_collision_probability(
            row["MISS_DISTANCE_KM"],
            sigma_a_km=row["SIGMA_A_KM"],
            sigma_b_km=row["SIGMA_B_KM"],
        )

        qae_qubits = adaptive_qae_qubits(
            analytic_p,
            requested_qubits=requested_qubits,
            max_qubits=14,
        )

        qae = run_qae(
            analytic_p,
            num_eval_qubits=qae_qubits,
            shots=shots,
        )

        mc_estimate = float(row["COLLISION_PROBABILITY_MC"])
        mc_ci_low = float(row["MC_CI_LOW"])
        mc_ci_high = float(row["MC_CI_HIGH"])

        rows.append(
            {
                "LIVE_PRIORITY_RANK": rank,
                "OBJECT_A": row["OBJECT_A"],
                "NORAD_A": row["NORAD_A"],
                "OBJECT_B": row["OBJECT_B"],
                "NORAD_B": row["NORAD_B"],
                "TCA": row["TCA"],
                "DAYS_TO_TCA": row["DAYS_TO_TCA"],
                "FORECAST_HORIZON_DAYS": row["FORECAST_HORIZON_DAYS"],
                "MISS_DISTANCE_KM": row["MISS_DISTANCE_KM"],
                "RELATIVE_VELOCITY_KM_S": row["RELATIVE_VELOCITY_KM_S"],
                "SIGMA_A_KM": row["SIGMA_A_KM"],
                "SIGMA_B_KM": row["SIGMA_B_KM"],
                "ALTITUDE_DIFFERENCE_KM": row["ALTITUDE_DIFFERENCE_KM"],
                "INCLINATION_DIFFERENCE_DEG": row["INCLINATION_DIFFERENCE_DEG"],
                "COMPOSITE_RISK_SCORE": row["COMPOSITE_RISK_SCORE"],
                "COMPOSITE_RISK_LEVEL": row["COMPOSITE_RISK_LEVEL"],
                "ANALYTIC_PC": analytic_p,
                "QAE_EVAL_QUBITS_REQUESTED": int(requested_qubits),
                "QAE_EVAL_QUBITS_USED": int(qae_qubits),
                "QAE_SHOTS": int(shots),
                "QAE_ESTIMATOR": qae["estimator"],
                "QAE_ESTIMATE": qae["qae_estimate"],
                "QAE_ERROR": qae["qae_error"],
                "QAE_ORACLE_CALLS": qae["oracle_calls"],
                "QAE_RUNTIME_SEC": qae["runtime_sec"],
                "MC_ESTIMATE": mc_estimate,
                "MC_ERROR_VS_ANALYTIC": abs(mc_estimate - analytic_p),
                "MC_SAMPLES": int(row["MC_SAMPLES"]),
                "MC_EFFECTIVE_SAMPLE_SIZE": float(row["MC_EFFECTIVE_SAMPLE_SIZE"]),
                "MC_CI_LOW": mc_ci_low,
                "MC_CI_HIGH": mc_ci_high,
                "MC_METHOD": row["MC_METHOD"],
                "QAE_VS_MC_ABS_ERROR": abs(qae["qae_estimate"] - mc_estimate),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run QAE on the prioritized live 30-day forecast conjunctions"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=config.RESULTS_DIR / "monte_carlo_results.csv",
        help="Live Monte Carlo result CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR / "live_qae_comparison.csv",
        help="Output comparison CSV",
    )
    parser.add_argument(
        "--qae-eval-qubits",
        type=int,
        default=getattr(config, "QAE_EVALUATION_QUBITS", 6),
    )
    parser.add_argument("--qae-shots", type=int, default=200)
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of highest-priority conjunctions to evaluate; 0 means all",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Live Monte Carlo result file not found: {args.input}. "
            "Run modules.monte_carlo first."
        )

    mc_results = pd.read_csv(args.input)

    result = build_live_qae_comparison(
        mc_results,
        requested_qubits=args.qae_eval_qubits,
        shots=args.qae_shots,
        limit=args.limit,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Live QAE comparison rows: {len(result)}")
    print(f"Output: {args.output}")

    if result.empty:
        print("No live conjunctions were available for QAE evaluation.")
        return

    print()
    print(
        result[
            [
                "LIVE_PRIORITY_RANK",
                "OBJECT_A",
                "OBJECT_B",
                "DAYS_TO_TCA",
                "MISS_DISTANCE_KM",
                "ANALYTIC_PC",
                "QAE_ESTIMATE",
                "MC_ESTIMATE",
                "QAE_VS_MC_ABS_ERROR",
                "QAE_EVAL_QUBITS_USED",
                "QAE_ORACLE_CALLS",
                "MC_SAMPLES",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
