"""
QAE vs Classical Monte Carlo query-budget benchmark.

This benchmark measures how estimation error changes as the available
oracle/sample budget increases.

IMPORTANT:
This is a simulation benchmark. The QAE implementation estimates a
probability already encoded into a quantum amplitude. It does not
construct a full quantum orbital-collision oracle.

The QAE powers are explicitly limited according to the requested
oracle budget so that the benchmark actually tests different QAE
budgets.
"""

import time
import numpy as np
import pandas as pd

from modules.qae import run_qae_estimation

import config


# ============================================================
# BENCHMARK SETTINGS
# ============================================================

TEST_PROBABILITY = 0.10

MC_BUDGETS = [
    100,
    500,
    1000,
    5000,
    10000,
    50000,
    100000
]

N_TRIALS = 10

QAE_SHOTS = getattr(
    config,
    "QAE_SHOTS",
    4096
)

RANDOM_SEED = getattr(
    config,
    "RANDOM_SEED",
    42
)


# ============================================================
# CLASSICAL MONTE CARLO
# ============================================================

def run_classical_mc(
    true_probability,
    n_samples,
    seed=None
):
    rng = np.random.default_rng(seed)

    start = time.perf_counter()

    hits = rng.binomial(
        1,
        true_probability,
        size=n_samples
    ).sum()

    runtime = (
        time.perf_counter()
        - start
    )

    estimate = hits / n_samples

    error = abs(
        estimate
        - true_probability
    )

    return {
        "estimate": float(estimate),
        "error": float(error),
        "runtime": float(runtime),
        "samples": int(n_samples)
    }


# ============================================================
# QAE BUDGET MAPPING
# ============================================================

def qae_powers_for_budget(
    budget
):
    """
    Determine which Grover powers can actually be executed
    within the requested QAE oracle budget.

    For Grover power k, the approximate number of Grover
    operator applications is:

        2^k

    Each power is repeated for QAE_SHOTS shots.

    Therefore:

        cost(k) = 2^k * QAE_SHOTS

    The returned powers are the powers that the QAE experiment
    will ACTUALLY execute.
    """

    total = 0
    powers = []

    power = 0

    while True:

        cost = (
            (2 ** power)
            * QAE_SHOTS
        )

        if total + cost > budget:
            break

        powers.append(power)

        total += cost

        power += 1

    return powers, total


# ============================================================
# BENCHMARK
# ============================================================

def run_budget_sweep():

    rows = []
    raw_rows = []

    for budget in MC_BUDGETS:

        powers, qae_budget = (
            qae_powers_for_budget(
                budget
            )
        )

        print(
            f"\nBudget={budget}"
        )

        print(
            f"QAE powers available: "
            f"{powers}"
        )

        print(
            f"Approximate QAE budget: "
            f"{qae_budget}"
        )

        # ----------------------------------------------------
        # No QAE experiment possible.
        # ----------------------------------------------------

        if not powers:

            print(
                "Budget too small for one QAE shot."
            )

            continue

        qae_errors = []
        mc_errors = []

        # ----------------------------------------------------
        # Trials
        # ----------------------------------------------------

        for trial in range(
            N_TRIALS
        ):

            seed = (
                RANDOM_SEED
                + trial
            )

            # =================================================
            # QAE
            # =================================================

            qae_estimate, qae_runtime, measurements = (
                run_qae_estimation(
                    probability=TEST_PROBABILITY,
                    shots=QAE_SHOTS,
                    seed=seed,
                    powers=powers
                )
            )

            qae_error = abs(
                qae_estimate
                - TEST_PROBABILITY
            )

            # =================================================
            # CLASSICAL MONTE CARLO
            # =================================================

            mc_result = run_classical_mc(
                true_probability=TEST_PROBABILITY,
                n_samples=budget,
                seed=(
                    RANDOM_SEED
                    + 1000
                    + trial
                )
            )

            # -------------------------------------------------
            # Store errors.
            # -------------------------------------------------

            qae_errors.append(
                qae_error
            )

            mc_errors.append(
                mc_result["error"]
            )

            # -------------------------------------------------
            # Store raw trial.
            # -------------------------------------------------

            raw_rows.append({

                "TRUE_PROBABILITY":
                    TEST_PROBABILITY,

                "REQUESTED_BUDGET":
                    budget,

                "QAE_BUDGET":
                    qae_budget,

                "QAE_POWERS":
                    str(powers),

                "TRIAL":
                    trial,

                "QAE_ESTIMATE":
                    qae_estimate,

                "QAE_ERROR":
                    qae_error,

                "QAE_RUNTIME_SEC":
                    qae_runtime,

                "MC_ESTIMATE":
                    mc_result["estimate"],

                "MC_ERROR":
                    mc_result["error"],

                "MC_SAMPLES":
                    mc_result["samples"],

                "MC_RUNTIME_SEC":
                    mc_result["runtime"]
            })

        # =====================================================
        # MEAN ERRORS
        # =====================================================

        qae_mean = float(
            np.mean(
                qae_errors
            )
        )

        mc_mean = float(
            np.mean(
                mc_errors
            )
        )

        qae_wins = (
            qae_mean
            <
            mc_mean
        )

        # =====================================================
        # SUMMARY ROW
        # =====================================================

        rows.append({

            "TRUE_PROBABILITY":
                TEST_PROBABILITY,

            "REQUESTED_BUDGET":
                budget,

            "QAE_APPROX_ORACLE_BUDGET":
                qae_budget,

            "QAE_POWERS":
                str(powers),

            "QAE_ERROR_MEAN":
                qae_mean,

            "MC_ERROR_MEAN":
                mc_mean,

            "QAE_WINS":
                qae_wins,

            "N_TRIALS":
                N_TRIALS
        })

        # =====================================================
        # CONSOLE OUTPUT
        # =====================================================

        print(
            f"QAE error = "
            f"{qae_mean:.8f}"
        )

        print(
            f"MC error  = "
            f"{mc_mean:.8f}"
        )

        print(
            "Winner    = "
            + (
                "QAE"
                if qae_wins
                else "MC"
            )
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(raw_rows)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    config.ensure_dirs()

    df, raw_df = (
        run_budget_sweep()
    )

    output_path = (
        config.RESULTS_DIR
        / "qae_query_budget_sweep.csv"
    )

    raw_output_path = (
        config.RESULTS_DIR
        / "qae_query_budget_sweep_raw_trials.csv"
    )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Save raw trials.
    # --------------------------------------------------------

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
        df.to_string(
            index=False
        )
    )
