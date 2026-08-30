"""
Standalone benchmark (NOT part of the main pipeline): compares QAE vs
classical Monte Carlo accuracy across a range of probabilities and query
budgets that are actually resolvable by both methods, unlike the real
orbital data (which is far too small for either at feasible budgets).

This is what demonstrates QAE's theoretical quadratic query-complexity
advantage empirically for your report.
"""

import pandas as pd
import numpy as np

from modules.qae import run_qae, run_classical_mc


# Probabilities spanning a range both methods can actually resolve
TEST_PROBABILITIES = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

# Evaluation-qubit counts to sweep (controls QAE's query budget: ~2^m calls).
# Higher m -> QAE's error shrinks faster than MC's, making the crossover and
# growing gap clearly visible on the chart. Circuit size only grows to m+1
# qubits (statevector-trivial even at m=19), so this stays fast — what scales
# is oracle-call count, not simulation cost.
EVAL_QUBIT_COUNTS = [3, 5, 7, 9, 11, 13, 15, 17, 19]

# Repeated trials per (p, m) combination, averaged, to smooth single-draw noise
N_TRIALS = 10


def run_sweep(shots=100):
    rows = []
    raw_trial_rows = []  # per-trial paired data, needed for the significance test

    for true_p in TEST_PROBABILITIES:
        for m in EVAL_QUBIT_COUNTS:
            qae_errors, mc_errors = [], []
            oracle_calls = None

            for trial in range(N_TRIALS):
                qae_result = run_qae(true_p, num_eval_qubits=m, shots=shots)
                mc_result = run_classical_mc(true_p, n_samples=qae_result["oracle_calls"])

                qae_errors.append(qae_result["qae_error"])
                mc_errors.append(mc_result["mc_error"])
                oracle_calls = qae_result["oracle_calls"]

                raw_trial_rows.append({
                    "TRUE_PROBABILITY": true_p,
                    "EVAL_QUBITS": m,
                    "TRIAL": trial,
                    "QAE_ERROR": qae_result["qae_error"],
                    "MC_ERROR": mc_result["mc_error"],
                })

            qae_err_mean = float(np.mean(qae_errors))
            mc_err_mean = float(np.mean(mc_errors))

            rows.append({
                "TRUE_PROBABILITY": true_p,
                "EVAL_QUBITS": m,
                "ORACLE_CALLS": oracle_calls,
                "N_TRIALS": N_TRIALS,
                "QAE_ERROR_MEAN": qae_err_mean,
                "MC_ERROR_MEAN": mc_err_mean,
            })

            print(f"p={true_p:<8} m={m:<3} calls={oracle_calls:<8} "
                  f"QAE_err={qae_err_mean:.6f}  MC_err={mc_err_mean:.6f}  "
                  f"({'QAE' if qae_err_mean < mc_err_mean else 'MC'} wins)")

    return pd.DataFrame(rows), pd.DataFrame(raw_trial_rows)


if __name__ == "__main__":
    df, raw_df = run_sweep()

    import config
    config.ensure_dirs()

    output_path = config.RESULTS_DIR / "qae_accuracy_sweep.csv"
    df.to_csv(output_path, index=False)

    raw_output_path = config.RESULTS_DIR / "qae_accuracy_sweep_raw_trials.csv"
    raw_df.to_csv(raw_output_path, index=False)

    print(f"\nSweep results written: {output_path}")
    print(f"Raw per-trial data written: {raw_output_path}")

    print("\nMean error by evaluation-qubit count (fewer QAE queries -> higher error, both methods):")
    print(df.groupby("EVAL_QUBITS")[["QAE_ERROR_MEAN", "MC_ERROR_MEAN"]].mean().to_string())