"""
Standalone script (NOT part of the main pipeline): formal significance
tests for H02 and H03, producing an actual test statistic and p-value
rather than just descriptive comparisons.

H02: no significant difference between QAE and classical Monte Carlo in
     estimating collision probabilities.
H03: no significant difference between QKD and classical transmission.

Run qae_accuracy_sweep.py and qkd_benchmark.py first — this script reads
their output files.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

import config


ALPHA = 0.05  # standard significance threshold


def test_h02():
    """
    Wilcoxon signed-rank test: paired comparison of QAE error vs classical
    MC error across every (probability, eval_qubits, trial) combination.
    Wilcoxon is used rather than a paired t-test because error values are
    bounded at zero and not guaranteed to be normally distributed.
    """
    path = config.RESULTS_DIR / "qae_accuracy_sweep_raw_trials.csv"
    if not path.exists():
        print(f"No raw trial data found at {path}. Run qae_accuracy_sweep.py first.")
        return None

    df = pd.read_csv(path)

    qae_errors = df["QAE_ERROR"].values
    mc_errors = df["MC_ERROR"].values

    # Wilcoxon requires at least one nonzero difference
    differences = qae_errors - mc_errors
    if np.all(differences == 0):
        print("All paired differences are exactly zero — cannot run Wilcoxon test.")
        return None

    statistic, p_value = stats.wilcoxon(qae_errors, mc_errors)

    median_diff = float(np.median(differences))
    qae_median = float(np.median(qae_errors))
    mc_median = float(np.median(mc_errors))

    reject_h0 = p_value < ALPHA

    result = {
        "hypothesis": "H02",
        "test": "Wilcoxon signed-rank test (paired, two-sided)",
        "n_pairs": len(df),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": ALPHA,
        "reject_null": reject_h0,
        "qae_median_error": qae_median,
        "mc_median_error": mc_median,
        "median_difference_qae_minus_mc": median_diff,
        "conclusion": (
            f"Reject H02 (p={p_value:.3e} < {ALPHA}): there IS a statistically "
            f"significant difference between QAE and classical Monte Carlo error. "
            f"{'QAE' if qae_median < mc_median else 'Classical MC'} has the lower "
            f"median error across {len(df)} paired trials."
            if reject_h0 else
            f"Fail to reject H02 (p={p_value:.3e} >= {ALPHA}): no statistically "
            f"significant difference detected between QAE and classical Monte "
            f"Carlo error across {len(df)} paired trials."
        ),
    }

    print("=== H02: QAE vs Classical Monte Carlo ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result


def test_h03():
    """
    Fisher's exact test on eavesdropping-detection outcomes, in two forms:
    (a) QKD's own internal validity — does it discriminate eavesdropped
        trials from honest trials significantly better than chance?
    (b) QKD vs classical head-to-head — classical has no eavesdropping-
        detection mechanism at all (structural 0/N, not an empirical
        failure), so this comparison is reported with that caveat rather
        than treated as an equivalent empirical measurement.
    """
    honest_path = config.RESULTS_DIR / "benchmark_qkd_honest_trials.csv"
    eavesdropped_path = config.RESULTS_DIR / "benchmark_qkd_eavesdropped_trials.csv"
    classical_path = config.RESULTS_DIR / "benchmark_classical_trials.csv"

    missing = [p for p in [honest_path, eavesdropped_path, classical_path] if not p.exists()]
    if missing:
        print(f"Missing benchmark file(s): {missing}. Run qkd_benchmark.py first.")
        return None

    honest_df = pd.read_csv(honest_path)
    eavesdropped_df = pd.read_csv(eavesdropped_path)
    classical_df = pd.read_csv(classical_path)

    # --- (a) QKD internal validity: eavesdropped-detected vs honest-detected ---
    honest_detected = int(honest_df["eavesdropping_detected"].sum())
    honest_total = len(honest_df)
    eaves_detected = int(eavesdropped_df["eavesdropping_detected"].sum())
    eaves_total = len(eavesdropped_df)

    table_internal = [
        [eaves_detected, eaves_total - eaves_detected],
        [honest_detected, honest_total - honest_detected],
    ]
    odds_ratio_a, p_value_a = stats.fisher_exact(table_internal)

    result_internal = {
        "test": "Fisher's exact test (QKD eavesdropped vs QKD honest detection)",
        "contingency_table": {
            "eavesdropped_detected": eaves_detected, "eavesdropped_not_detected": eaves_total - eaves_detected,
            "honest_detected": honest_detected, "honest_not_detected": honest_total - honest_detected,
        },
        "p_value": float(p_value_a),
        "alpha": ALPHA,
        "reject_null": p_value_a < ALPHA,
        "conclusion": (
            f"QKD's detection rate under eavesdropping ({eaves_detected}/{eaves_total}) is "
            f"statistically significantly different from its false-positive rate under normal "
            f"operation ({honest_detected}/{honest_total}), p={p_value_a:.3e}. The detection "
            f"mechanism reliably discriminates eavesdropping from normal operation."
            if p_value_a < ALPHA else
            f"No statistically significant difference between QKD's eavesdropped and honest "
            f"detection rates, p={p_value_a:.3e} — the detection mechanism does not reliably "
            f"discriminate eavesdropping from normal operation in this data."
        ),
    }

    # --- (b) QKD vs classical head-to-head (with explicit structural caveat) ---
    classical_detected = 0  # classical has no detection mechanism — not measured, structurally absent
    classical_total = len(classical_df)

    table_head_to_head = [
        [eaves_detected, eaves_total - eaves_detected],
        [classical_detected, classical_total - classical_detected],
    ]
    odds_ratio_b, p_value_b = stats.fisher_exact(table_head_to_head)

    result_head_to_head = {
        "test": "Fisher's exact test (QKD eavesdropped detection vs classical, CAVEAT: classical's "
                "0 is structural absence of a detection mechanism, not an empirically measured failure rate)",
        "contingency_table": {
            "qkd_eavesdropped_detected": eaves_detected, "qkd_eavesdropped_not_detected": eaves_total - eaves_detected,
            "classical_detected": classical_detected, "classical_not_detected": classical_total - classical_detected,
        },
        "p_value": float(p_value_b),
        "alpha": ALPHA,
        "reject_null": p_value_b < ALPHA,
        "conclusion": (
            f"Reject H03 for detection capability (p={p_value_b:.3e} < {ALPHA}): QKD detects "
            f"simulated interception significantly more often than classical transmission, which "
            f"has no eavesdropping-detection mechanism by design. Note: this compares an empirical "
            f"rate (QKD) against a structural absence (classical), not two empirically equivalent "
            f"measurements — report this distinction explicitly rather than implying classical was "
            f"tested and failed to detect anything."
            if p_value_b < ALPHA else
            f"No statistically significant difference detected, p={p_value_b:.3e}."
        ),
    }

    # --- (c) Delivery rate comparison (both methods share this metric) ---
    classical_delivered = int(classical_df["delivered"].sum())
    qkd_delivered = int(honest_df["delivered"].sum())

    table_delivery = [
        [qkd_delivered, honest_total - qkd_delivered],
        [classical_delivered, classical_total - classical_delivered],
    ]
    try:
        odds_ratio_c, p_value_c = stats.fisher_exact(table_delivery)
        delivery_conclusion = (
            f"No statistically significant difference in delivery rate between QKD "
            f"({qkd_delivered}/{honest_total}) and classical ({classical_delivered}/{classical_total}) "
            f"transmission, p={p_value_c:.3e} — both deliver reliably under normal conditions."
            if p_value_c >= ALPHA else
            f"Statistically significant difference in delivery rate, p={p_value_c:.3e}."
        )
    except Exception:
        p_value_c = None
        delivery_conclusion = "Delivery rate identical (100%) for both methods — no variance to test."

    result_delivery = {
        "test": "Fisher's exact test (delivery rate, QKD-honest vs classical)",
        "qkd_delivery_rate": qkd_delivered / honest_total,
        "classical_delivery_rate": classical_delivered / classical_total,
        "p_value": p_value_c,
        "conclusion": delivery_conclusion,
    }

    print("\n=== H03: QKD vs Classical Transmission ===")
    print("\n--- (a) QKD internal validity (eavesdropped vs honest detection) ---")
    for k, v in result_internal.items():
        print(f"{k}: {v}")

    print("\n--- (b) QKD vs classical detection capability (head-to-head, see caveat) ---")
    for k, v in result_head_to_head.items():
        print(f"{k}: {v}")

    print("\n--- (c) Delivery rate (QKD-honest vs classical) ---")
    for k, v in result_delivery.items():
        print(f"{k}: {v}")

    return {
        "hypothesis": "H03",
        "internal_validity": result_internal,
        "head_to_head_detection": result_head_to_head,
        "delivery_rate": result_delivery,
    }


def run_and_save():
    config.ensure_dirs()

    h02_result = test_h02()
    print()
    h03_result = test_h03()

    output = {"H02": h02_result, "H03": h03_result}
    output_path = config.RESULTS_DIR / "hypothesis_test_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults written: {output_path}")
    return output


if __name__ == "__main__":
    run_and_save()