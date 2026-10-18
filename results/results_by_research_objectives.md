# ORION-X Research Results and Current Validation Status

**Repository:** https://github.com/jdnmcortega-boop/orbital-debris-collision-system  
**Report revision:** 18 October 2026  
**Purpose:** Map the repository's available evidence to the six research objectives and hypotheses H01–H03 without presenting old output snapshots as newly computed results.

## Critical status: the saved numerical outputs are not current

The latest repository commits focus on the CDM Historical Validation Workbench, object-ID matching, and pair/TCA diagnostics. The committed result artifacts have not been regenerated with those changes. In particular:

- `results/dashboard_summary.json` still describes an older 70-object run with 44 conjunctions and 947,614 propagated state points.
- `data/processed/conjunctions.csv` and `results/predictions.csv` contain 22 conjunction records, not 44.
- The saved conjunction TCAs are from late August to early September 2026, so they are not a current 30-day forecast as of this report date.
- The current dashboard code exposes a live-refresh action that rebuilds outputs from the saved orbital catalog, but a successful post-change refresh result is not recorded in the repository files inspected for this report.
- The CDM workbench is now designed to compare unordered object pairs and TCA within a configurable tolerance (10 minutes by default), report unmatched pairs and time differences, and check whether covariance fields are present. A workbench feature is not itself a completed validation result.

**Therefore, this report does not label the old conjunction counts as current system performance.** A fresh run and a saved export from that same run are required before current conjunction counts, risk counts, or accuracy percentages can be reported.

## Current repository changes relevant to the research

The latest code changes include:
- NORAD catalog IDs are preferred when matching CDM objects; international designators are not treated as interchangeable with NORAD IDs.
- Object-pair matching is order-independent.
- The CDM validation page reports pair overlap, TCA-range overlap, TCA deltas, event-match status, reported Pc, and covariance-field availability.
- The covariance importer distinguishes a Pc-only reference export from a file containing flattened position-covariance fields. It validates covariance blocks but explicitly does not yet use them to calculate collision probability.
- The overview separates **HIGH collision-probability events** from **HIGH ORION-X response-priority events**. A response-priority score is not a collision probability.

These changes improve diagnostics and interpretability, but they do not by themselves establish detection accuracy or validate the collision-probability model.

## Objective 1 — Estimate collision probability from relative motion, orbital parameters, uncertainty, and simulated collision states

The committed prediction and Monte Carlo artifacts are older snapshots, not current forecast results. In those files, the 22 rows in `results/monte_carlo_results.csv` each record 1,000,000 samples and zero collision hits. The corresponding empirical Monte Carlo estimate is therefore zero for each row, but zero observed hits does not prove a true probability of zero. The stored 95% upper bound is approximately (2.996\times10^{-6}) for one million trials with zero hits.

The stored `results/qae_results.csv` uses an amplified/scaled QAE estimate. Its first row reports a target probability of approximately (1.0074\times10^{-9}), a scaled QAE estimate of approximately (9.9798\times10^{-10}), and 4,096 shots. These are historical benchmark values from the committed file, not a fresh live-run measurement.

The uncertainty columns in the saved prediction table are modeled values. The repository's covariance-import code can inspect compatible covariance fields, but the code itself notes that a validated position-covariance block is not yet enough to replace the collision-probability model: frame/time handling, encounter-plane transformation, hard-body assumptions, and CDM conventions still need to be handled.

**Current finding:** The project implements the relevant calculation and diagnostic workflow. Current real-world probability accuracy remains unverified until a fresh forecast is run and compared with matched CDM records using suitable covariance data.

## Objective 2 — Evaluate conjunction detection

The old dashboard snapshot reports 70 objects, 947,614 propagated points, and 44 close approaches. The committed conjunction and prediction tables contain 22 records. Because these outputs disagree, neither count should be presented as the current, reconciled result.

The current validation workflow is designed to evaluate events using:
1. Matching object pairs by catalog IDs without depending on object order.
2. Comparing TCA values using a stated tolerance.
3. Reporting events with matching pairs but TCA outside the tolerance separately from events with no pair match.
4. Checking whether the CDM and ORION-X TCA ranges overlap.

**Current finding:** The latest code supports more auditable event matching, but a current detection-success percentage cannot yet be reported from the committed artifacts. Precision, recall, false-positive rate, and false-negative rate require a completed comparison against a verified reference set covering the same time interval.

## Objective 3 — Compare Quantum Amplitude Estimation (QAE) with classical Monte Carlo

The committed hypothesis-test artifact records a paired Wilcoxon signed-rank test with (n=540), (p=0.0170), and (alpha=0.05). It reports median absolute error of approximately (7.946\times10^{-6}) for QAE and (1.613\times10^{-4}) for Monte Carlo in that saved benchmark.

**Decision for the saved benchmark only:** Reject H02 for that benchmark because the reported paired errors differ significantly, with a lower reported median error for QAE.

This is not a result from a fresh live conjunction run. It also does not establish quantum speedup: the committed QAE implementation is simulator-based, and the stored wall-clock times include different computational paths and overhead. Do not claim that QAE is faster in operational use from these results.

## Objective 4 — Compare QKD and classical data transmission

The saved `results/benchmark_summary.json` contains 50 trials per condition:

| Stored benchmark condition | Delivery rate | Mean runtime | Other recorded result |
|---|---:|---:|---|
| Classical ECDH + AES-GCM | 100% (50/50) | 0.00428 s | 194 bytes mean public-key overhead |
| QKD, honest channel | 100% (50/50) | 0.04495 s | 0% mean QBER; 100% secure delivery |
| QKD, simulated intercept-resend | 100% transport delivery | 0.09339 s | 26.29% mean QBER; 0% secure delivery; 100% interception detection (50/50) |

These are stored simulation benchmarks and have not been re-run by the latest CDM code changes. The saved hypothesis-test artifact reports no significant delivery-rate difference between classical transmission and honest QKD ((p=1.00)). It reports a significant difference between QKD's simulated interception detection and its honest-channel false-positive rate ((p\approx1.98\times10^{-29})).

The classical method has no built-in QBER-style detection mechanism in this comparison, so its “zero detections” is not an empirically measured detector failure. The comparison is also a software simulation, not a physical quantum communication experiment.

**Decision:** Interpret H03 by metric. The saved benchmark found no delivery-rate difference, while QKD detected the simulated intercept-resend condition. Do not make one blanket claim that QKD outperforms classical transmission across all metrics.

## Objective 5 — Classify conjunctions as High, Medium, or Low

The old saved dashboard shows 44 LOW collision-probability labels and no MEDIUM or HIGH collision-probability labels. The older prediction table also labels its 22 rows LOW. These are old outputs and should not be used as the current classification distribution.

The current dashboard code explicitly separates the count of HIGH collision-probability events from the count of HIGH ORION-X response-priority events. It also warns when events cannot reach the configured HIGH probability threshold under the current assumed uncertainty values. This distinction is important: response priority is a separate ranking, not a substitute collision probability, and thresholds must not be changed merely to force HIGH labels.

**Current finding:** The latest code improves the reporting distinction, but the current distribution of risk labels is unknown until the outputs are regenerated and saved from a single run. The absence of a HIGH label must not be interpreted as proof that no high-risk conjunction exists.

## Objective 6 — Prioritize debris for removal using risk, collision probability, and consequence

No current validated debris-removal target list is present in the repository evidence reviewed for this report. A close approach alone is not sufficient to recommend removal. A defensible preliminary ranking should consider confirmed nonfunctional debris status, validated risk, repeated encounters and consequences for active spacecraft, and feasibility of capture and removal. Active operational satellites must not be classified as debris or recommended for removal.

The old dashboard's closest-pair record is a historical observation and must not be presented as a current removal recommendation.

**Current finding:** ORION-X can support preliminary screening, but a defensible current removal-priority list requires synchronized current orbital data, verified object status, current conjunction outputs, transparent consequence criteria, and removal-feasibility constraints.

## Hypothesis decisions

| Hypothesis | Evidence currently available | Responsible conclusion |
|---|---|---|
| **H01:** ORION-X could not accurately detect and classify LEO conjunctions under predefined criteria. | The committed output snapshots conflict (44 dashboard events versus 22 table rows). A newer CDM matching workflow exists, but no completed, current matched-event export is recorded in the inspected files. | **Inconclusive.** Do not reject H01 until a fresh run is evaluated against verified reference events over the same interval and false positives/negatives are counted. |
| **H02:** There is no significant difference between QAE and Monte Carlo estimates. | Saved paired benchmark: (n=540), Wilcoxon (p=0.0170<0.05), with lower reported median error for QAE. | **Reject for the saved benchmark only.** This does not establish live-run superiority or wall-clock speedup. |
| **H03:** There is no significant difference between QKD and classical transmission. | Saved benchmark: both honest-channel delivery rates are 100% ((p=1.00)); QKD detected 50/50 simulated intercept-resend trials with 0/50 honest-channel false alarms. | **Metric-specific / partial conclusion.** No delivery-rate difference was found; QKD's simulated interception-detection result is significant, but classical transmission does not have an equivalent detector in this benchmark. |

## What is needed before the report can contain current numerical results

1. Run **“Rebuild live results from orbital_data.csv”** in the current ORION-X dashboard and wait for the run to complete.
2. Save the live-refresh summary, current orbital catalog epoch, object count, propagated-state count, conjunction count, prediction count, and separate HIGH Pc / HIGH response-priority counts from that same run.
3. Export the CDM comparison using an ORION-X conjunction file covering the CDM's TCA interval. Record pair overlap, TCA-range overlap, matches within tolerance, pair matches outside tolerance, unmatched events, and reported-Pc comparisons.
4. Record whether the uploaded CDM actually contains covariance fields. If it is Pc-only, use its reported Pc as reference data only; do not describe it as covariance validation.
5. Regenerate the relevant tables from one run and update the hypothesis tests only with the data and test design that match each hypothesis.

## Repository evidence inspected

- `results/dashboard_summary.json`
- `data/processed/conjunctions.csv`
- `results/predictions.csv`
- `results/monte_carlo_results.csv`
- `results/qae_results.csv`
- `results/hypothesis_test_results.json`
- `results/benchmark_summary.json`
- `results/qkd_results.json`
- `ui/dashboard.py`
- `ui/historical_live_dashboard.py`
- `pages/2_CDM_Historical_Validation.py`
- `modules/cdm_covariance.py`

**Bottom line:** The repository has newer validation and diagnostic code, but the saved numerical outputs remain old and internally inconsistent. This revision makes that status explicit instead of relabeling the old numbers as current. A genuinely current results report must be completed from a fresh run and its matching CDM export.
