"""
Standalone benchmark (NOT part of the main pipeline): compares QAE vs
classical Monte Carlo accuracy across a range of probabilities and query
budgets that are actually resolvable by both methods.

The classical Monte Carlo sample count is matched to QAE's oracle-call
budget for every evaluation-qubit setting.

This benchmark is separate from the main orbital-debris pipeline.
"""

import pandas as pd
import numpy as np

from modules.qae import run_qae


def run_classical_mc(true_p, n_samples, seed=None):
    """
    Classical Monte Carlo using a direct binomial draw.

    This is mathematically equivalent to performing n_samples
    Bernoulli trials, but avoids allocating a huge NumPy array.
    """

    rng = np.random.default_rng(seed)

    hits = rng.binomial(
        n_samples,
        true_p
    )

    estimate = hits / n_samples

    return {
        "estimate": float(estimate),
        "error": float(abs(estimate - true_p)),
        "samples": int(n_samples),
    }


# ------------------------------------------------------------
# Probabilities spanning a range both methods can resolve
# ------------------------------------------------------------

TEST_PROBABILITIES = [
    0.5,
    0.1,
    0.05,
    0.01,
    0.005,
    0.001,
]


# ------------------------------------------------------------
# Evaluation-qubit counts
#
# QAE query budget is approximately:
#
#     shots × (2^m - 1)
#
# ------------------------------------------------------------

EVAL_QUBIT_COUNTS = [
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
]


# ------------------------------------------------------------
# Repeated trials
# ------------------------------------------------------------

N_TRIALS = 10


def run_sweep(shots=100):

    rows = []

    raw_trial_rows = []

    for true_p in TEST_PROBABILITIES:

        for m in EVAL_QUBIT_COUNTS:

            qae_errors = []

            mc_errors = []

            oracle_calls = None

            for trial in range(N_TRIALS):

                seed = 42 + trial

                # ------------------------------------------------
                # QAE
                # ------------------------------------------------

                qae_result = run_qae(
                    true_p,
                    num_eval_qubits=m,
                    shots=shots,
                )

                # ------------------------------------------------
                # MATCHED-BUDGET MONTE CARLO
                #
                # MC receives exactly the same number of
                # oracle calls available to QAE.
                # ------------------------------------------------

                mc_result = run_classical_mc(
                    true_p,
                    n_samples=qae_result["oracle_calls"],
                    seed=seed,
                )

                qae_error = qae_result["qae_error"]

                mc_error = mc_result["error"]

                qae_errors.append(
                    qae_error
                )

                mc_errors.append(
                    mc_error
                )

                oracle_calls = (
                    qae_result["oracle_calls"]
                )

                raw_trial_rows.append({
                    "TRUE_PROBABILITY":
                        true_p,

                    "EVAL_QUBITS":
                        m,

                    "TRIAL":
                        trial,

                    "QAE_ERROR":
                        qae_error,

                    "MC_ERROR":
                        mc_error,

                    "ORACLE_CALLS":
                        oracle_calls,
                })

            # ----------------------------------------------------
            # Mean error
            # ----------------------------------------------------

            qae_err_mean = float(
                np.mean(
                    qae_errors
                )
            )

            mc_err_mean = float(
                np.mean(
                    mc_errors
                )
            )

            # ----------------------------------------------------
            # Summary row
            # ----------------------------------------------------

            rows.append({
                "TRUE_PROBABILITY":
                    true_p,

                "EVAL_QUBITS":
                    m,

                "ORACLE_CALLS":
                    oracle_calls,

                "N_TRIALS":
                    N_TRIALS,

                "QAE_ERROR_MEAN":
                    qae_err_mean,

                "MC_ERROR_MEAN":
                    mc_err_mean,

                "QAE_WINS":
                    qae_err_mean < mc_err_mean,
            })

            winner = (
                "QAE"
                if qae_err_mean < mc_err_mean
                else "MC"
            )

            print(
                f"p={true_p:<8} "
                f"m={m:<3} "
                f"calls={oracle_calls:<12} "
                f"QAE_err={qae_err_mean:.6f}  "
                f"MC_err={mc_err_mean:.6f}  "
                f"({winner} wins)"
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(raw_trial_rows),
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import config

    config.ensure_dirs()

    df, raw_df = run_sweep()

    output_path = (
        config.RESULTS_DIR
        / "qae_accuracy_sweep.csv"
    )

    raw_output_path = (
        config.RESULTS_DIR
        / "qae_accuracy_sweep_raw_trials.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    raw_df.to_csv(
        raw_output_path,
        index=False,
    )

    print(
        f"\nSweep results written: "
        f"{output_path}"
    )

    print(
        f"Raw trial results written: "
        f"{raw_output_path}"
    )

    print(
        "\nMean error by evaluation-qubit count:"
    )

    print(
        df.groupby(
            "EVAL_QUBITS"
        )[
            [
                "QAE_ERROR_MEAN",
                "MC_ERROR_MEAN",
            ]
        ].mean().to_string()
    )