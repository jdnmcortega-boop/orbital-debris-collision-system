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
    analytic_path = config.RESULTS_DIR / "analytic_pc_results.csv"
    if analytic_path.exists():
        analysis = pd.read_csv(analytic_path)
        if analysis.empty:
            print("Analytical Pc results are empty — nothing to assess.")
            return None
        if "ANALYTIC_PC" not in analysis.columns:
            raise ValueError("analytic_pc_results.csv does not contain ANALYTIC_PC.")
        analysis["CLASSIFICATION"] = analysis["ANALYTIC_PC"].apply(
            lambda p: "NOT_CALCULATED" if pd.isna(p) else classify_pair(float(p))
        )
        analysis = analysis.sort_values("ANALYTIC_PC", ascending=False, na_position="last").reset_index(drop=True)
    else:
        conj_path = config.CONJUNCTIONS_FILE
        if not conj_path.exists():
            print(f"No conjunctions file found at {conj_path}. Run conjunction_detection first.")
            return None
        conjunctions = pd.read_csv(conj_path)
        if conjunctions.empty:
            print("Conjunctions file is empty — nothing to assess.")
            return None
        analysis = build_false_positive_analysis(conjunctions)

    output_path = config.RESULTS_DIR / "false_positive_analysis.csv"
    analysis.to_csv(output_path, index=False)
    valid = analysis["CLASSIFICATION"].isin(["FALSE_POSITIVE", "CONFIRMED_CONCERN"])
    valid_count = int(valid.sum())
    false_positives = int((analysis["CLASSIFICATION"] == "FALSE_POSITIVE").sum())
    confirmed = int((analysis["CLASSIFICATION"] == "CONFIRMED_CONCERN").sum())
    fp_rate = false_positives / valid_count if valid_count else float("nan")
    print(f"Screened pairs with calculated Pc: {valid_count}")
    print(f"Modeled concerns: {confirmed}")
    print(f"Below-threshold pairs: {false_positives}")
    print(f"Below-threshold rate: {fp_rate:.2%}" if valid_count else "Below-threshold rate: unavailable")
    print(f"Results written: {output_path}")
    cols = [c for c in ["OBJECT_A", "OBJECT_B", "MISS_DISTANCE_KM", "ANALYTIC_PC", "CLASSIFICATION"] if c in analysis.columns]
    if cols:
        print(analysis[cols].to_string(index=False))
    return analysis

if __name__ == "__main__":
    run_and_save()