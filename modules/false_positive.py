"""
False-positive assessment: compares distance-screened close approaches
(from conjunction_detection.py) against actual collision probability, to
identify which screened pairs are real concerns vs. false alarms.

Uses the analytic collision-probability formula from qae.py rather than
the raw indicator-sampling Monte Carlo, since MC at feasible sample sizes
cannot resolve the rare-event probabilities typical of these conjunctions
(see monte_carlo.py sanity check) — the analytic formula gives a stable,
comparable value for every pair regardless of how small the true Pc is.
"""

import pandas as pd

import config
from modules.qae import analytic_collision_probability


def classify_pair(analytic_pc, threshold=None):
    threshold = threshold if threshold is not None else getattr(
        config, "RISK_THRESHOLD_MEDIUM", 1e-6
    )
    return "CONFIRMED_CONCERN" if analytic_pc >= threshold else "FALSE_POSITIVE"


def build_false_positive_analysis(conjunctions_df, sigma_km=None,
                                   hard_body_radius_km=None, threshold=None):
    """
    Take the distance-screened conjunctions and classify each as a
    confirmed concern or a false positive, based on analytic Pc.
    """
    df = conjunctions_df.copy()

    df["ANALYTIC_PC"] = df["MISS_DISTANCE_KM"].apply(
        lambda d: analytic_collision_probability(d, sigma_km, hard_body_radius_km)
    )
    df["CLASSIFICATION"] = df["ANALYTIC_PC"].apply(lambda p: classify_pair(p, threshold))

    return df.sort_values("ANALYTIC_PC", ascending=False).reset_index(drop=True)


def run_and_save():
    config.ensure_dirs()

    conj_path = config.CONJUNCTIONS_FILE
    if not conj_path.exists():
        print(f"No conjunctions file found at {conj_path}. Run conjunction_detection first.")
        return None

    conjunctions = pd.read_csv(conj_path)
    if len(conjunctions) == 0:
        print("Conjunctions file is empty — nothing to assess.")
        return None

    analysis = build_false_positive_analysis(conjunctions)

    output_path = config.RESULTS_DIR / "false_positive_analysis.csv"
    analysis.to_csv(output_path, index=False)

    total = len(analysis)
    false_positives = (analysis["CLASSIFICATION"] == "FALSE_POSITIVE").sum()
    confirmed = total - false_positives
    fp_rate = false_positives / total if total > 0 else 0.0

    print(f"\nScreened pairs: {total}")
    print(f"Confirmed concerns: {confirmed}")
    print(f"False positives:    {false_positives}")
    print(f"False-positive rate: {fp_rate:.2%}")

    print(f"\nResults written: {output_path}")
    print(analysis[["OBJECT_A", "OBJECT_B", "MISS_DISTANCE_KM",
                     "ANALYTIC_PC", "CLASSIFICATION"]].to_string(index=False))

    return analysis


if __name__ == "__main__":
    run_and_save()