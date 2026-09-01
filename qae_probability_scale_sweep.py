"""
QAE vs Monte Carlo probability-scale benchmark.

Tests QAE estimation accuracy over increasingly rare probabilities.
This is a simulation benchmark showing the probability-resolution
limit of the current finite-shot QAE implementation.
"""

import numpy as np
import pandas as pd

from modules.qae import run_qae_estimation

import config


# ============================================================
# SETTINGS
# ============================================================

PROBABILITIES = [
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
]

QAE_SHOTS = getattr(
    config,
    "QAE_SHOTS",
    4096
)

MC_SAMPLES = 100000

N_TRIALS = 10

BASE_SEED = getattr(
    config,
    "RANDOM_SEED",
    42
)


# ============================================================
# MONTE CARLO
# ============================================================

def run_mc(
    true_probability,
    samples,
    seed
):
    rng = np.random.default_rng(seed)

    hits = rng.binomial(
        1,
        true_probability,
        size=samples
    ).sum()

    estimate = hits / samples

    error = abs(
        estimate - true_probability
    )

    return (
        float(estimate),
        float(error),
        int(hits)
    )


# ============================================================
# SWEEP
# ============================================================

def run_scale_sweep():

    summary_rows = []
    raw_rows = []

    for probability in PROBABILITIES:

        print(
            f"\nProbability = {probability:.1e}"
        )

        qae_errors = []
        mc_errors = []

        qae_estimates = []
        mc_estimates = []

        qae_zero_count = 0
        mc_zero_hit_count = 0

        for trial in range(N_TRIALS):

            seed = BASE_SEED + trial

            # ------------------------------------------------
            # QAE
            # ------------------------------------------------

            qae_estimate, qae_runtime, _ = (
                run_qae_estimation(
                    probability=probability,
                    shots=QAE_SHOTS,
                    seed=seed
                )
            )

            qae_error = abs(
                qae_estimate - probability
            )

            if qae_estimate == 0.0:
                qae_zero_count += 1

            # ------------------------------------------------
            # MONTE CARLO
            # ------------------------------------------------

            (
                mc_estimate,
                mc_error,
                mc_hits
            ) = run_mc(
                true_probability=probability,
                samples=MC_SAMPLES,
                seed=BASE_SEED + 1000 + trial
            )

            if mc_hits == 0:
                mc_zero_hit_count += 1

            qae_errors.append(qae_error)
            mc_errors.append(mc_error)

            qae_estimates.append(qae_estimate)
            mc_estimates.append(mc_estimate)

            raw_rows.append({
                "TRUE_PROBABILITY":
                    probability,

                "TRIAL":
                    trial,

                "QAE_ESTIMATE":
                    qae_estimate,

                "QAE_ERROR":
                    qae_error,

                "QAE_ZERO":
                    qae_estimate == 0.0,

                "QAE_RUNTIME_SEC":
                    qae_runtime,

                "MC_ESTIMATE":
                    mc_estimate,

                "MC_ERROR":
                    mc_error,

                "MC_HITS":
                    mc_hits,

                "MC_ZERO_HITS":
                    mc_hits == 0
            })

        qae_mean = float(
            np.mean(qae_errors)
        )

        mc_mean = float(
            np.mean(mc_errors)
        )

        qae_mean_estimate = float(
            np.mean(qae_estimates)
        )

        mc_mean_estimate = float(
            np.mean(mc_estimates)
        )

        qae_wins = (
            qae_mean < mc_mean
        )

        summary_rows.append({
            "TRUE_PROBABILITY":
                probability,

            "QAE_SHOTS":
                QAE_SHOTS,

            "MC_SAMPLES":
                MC_SAMPLES,

            "N_TRIALS":
                N_TRIALS,

            "QAE_MEAN_ESTIMATE":
                qae_mean_estimate,

            "MC_MEAN_ESTIMATE":
                mc_mean_estimate,

            "QAE_ERROR_MEAN":
                qae_mean,

            "MC_ERROR_MEAN":
                mc_mean,

            "QAE_ZERO_RATE":
                qae_zero_count / N_TRIALS,

            "MC_ZERO_HIT_RATE":
                mc_zero_hit_count / N_TRIALS,

            "QAE_WINS":
                qae_wins
        })

        print(
            f"QAE mean estimate = "
            f"{qae_mean_estimate:.6e}"
        )

        print(
            f"MC mean estimate  = "
            f"{mc_mean_estimate:.6e}"
        )

        print(
            f"QAE mean error    = "
            f"{qae_mean:.6e}"
        )

        print(
            f"MC mean error     = "
            f"{mc_mean:.6e}"
        )

        print(
            f"QAE zero rate     = "
            f"{qae_zero_count}/{N_TRIALS}"
        )

        print(
            f"MC zero-hit rate  = "
            f"{mc_zero_hit_count}/{N_TRIALS}"
        )

        print(
            "Winner            = "
            + (
                "QAE"
                if qae_wins
                else "MC"
            )
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(raw_rows)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    config.ensure_dirs()

    summary_df, raw_df = (
        run_scale_sweep()
    )

    output_path = (
        config.RESULTS_DIR
        / "qae_probability_scale_sweep.csv"
    )

    raw_output_path = (
        config.RESULTS_DIR
        / "qae_probability_scale_sweep_raw_trials.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False
    )

    raw_df.to_csv(
        raw_output_path,
        index=False
    )

    print(
        f"\nResults written: "
        f"{output_path}"
    )

    print(
        f"Raw trials written: "
        f"{raw_output_path}"
    )

    print(
        "\nFinal summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )