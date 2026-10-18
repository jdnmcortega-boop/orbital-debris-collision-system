# ORION-X Results by Research Objective

**Repository examined:** https://github.com/jdnmcortega-boop/orbital-debris-collision-system  
**Evidence basis:** Saved outputs currently present in the repository (including `results/dashboard_summary.json`, `results/predictions.csv`, `results/qae_results.csv`, `results/qae_accuracy_sweep.csv`, `results/hypothesis_test_results.json`, and `results/benchmark_summary.json`). These are repository snapshots, not a fresh execution of the deployed dashboard on 18 October 2026.

## Important reporting note

The saved files are not fully synchronized: `dashboard_summary.json` reports 44 conjunctions from a 70-object run, while the saved probability/prediction tables contain a smaller set of rows (the CSV snapshots show 22 conjunction records). The report below keeps these results separate rather than combining them into one unsupported total. Before submitting a final paper, rerun the pipeline once and export all tables and summaries from that same run.

## Objective 1 — Estimate collision probability from relative motion, orbital parameters, uncertainty, and simulated collision states

The software calculates conjunction geometry using propagated orbital states and reports relative velocity, miss distance, altitude difference, inclination difference, and object-specific modeled uncertainty values in the saved prediction tables. The saved data-quality report describes a separate 60-object input snapshot: 30 debris and 30 satellites; eccentricity ranged from 0.0000371 to 0.10243452, inclination from 24.969° to 107.9611°, and TLE age averaged 6.14 days (range 2.14–136.79 days).

In the saved Monte Carlo table, each listed conjunction used 1,000,000 samples and recorded zero collision hits. Therefore, the empirical estimate was (0/1{,}000{,}000 = 0) for each listed event. This does **not** prove that true collision probability is zero. With zero hits in one million independent samples, the saved one-sided 95% upper bound is approximately (2.996\times10^{-6}), which is also present in the results. The analytical model produced nonzero probabilities for the same events, generally in the very-low-probability range (for example, (8.39\times10^{-7}) for ORBCOMM FM06–SCD 2).

**Finding:** The system produces the requested inputs and probability estimates, but the available results do not establish calibrated real-world probability accuracy. The uncertainty values are modeled estimates, not verified object-specific CDM covariance for every event. The orbital-parameter report should be treated as contextual analysis, not proof that every parameter independently improves Pc accuracy.

## Objective 2 — Evaluate conjunction detection

The saved dashboard summary reports:
- Tracked objects: 70
- Propagated state points: 947,614
- Detected close approaches: 44
- Minimum miss distance: 18.1959 km
- Mean miss distance: 48.7363 km
- Collision-pair types: 19 debris–satellite, 16 satellite–satellite, and 9 debris–debris
- Probability-based classes in that snapshot: 44 LOW, 0 MEDIUM, 0 HIGH

The separate saved conjunction-distance distribution records 1 pair at 5–10 km, 1 at 10–20 km, 10 at 20–50 km, and 10 at 50–100 km. This distribution appears to correspond to the smaller saved prediction table rather than the 44-event dashboard summary, so it must not be used to overwrite the dashboard counts.

The percentage below the configured collision-probability thresholds cannot be responsibly reported as a validated detection-success percentage without a defined ground-truth set of true conjunctions and missed events. In the dashboard snapshot, all 44 detected events were labeled LOW, i.e. 100% were below the system's MEDIUM/HIGH probability thresholds; this is a classification output, **not** a sensitivity, precision, or accuracy score.

**Finding:** Close approaches were detected and minimum miss distances were reported. Detection accuracy remains unproven because the saved summary does not provide a complete independently verified list of true positives, false negatives, and false positives over the same evaluation interval.

## Objective 3 — Compare Quantum Amplitude Estimation (QAE) with classical Monte Carlo

### Saved live/conjunction comparison

In the saved `qae_results.csv`, the listed conjunctions have zero Monte Carlo hits from 1,000,000 samples, so their raw MC estimate is zero. The saved QAE implementation uses probability scaling/amplitude amplification for very small target probabilities. For the first listed row, the analytic target is (1.0074\times10^{-9}), the scaled QAE estimate is (9.9798\times10^{-10}), and its reported absolute error against that target is (9.4450\times10^{-12}). The same row records 4,096 QAE shots and about 0.916 seconds for QAE versus about 0.00112 seconds for the model execution field. These timings are from this saved run and are not evidence of a quantum speedup; the implementation runs on a simulator and includes overhead that differs from the classical sampling path.

The separate `qae_comparison.csv` snapshot shows both QAE and Monte Carlo returning zero for several extremely small probabilities, while the analytic target is nonzero. This illustrates the finite-sample limitation of raw Monte Carlo and also shows why results from different benchmark configurations must not be mixed.

### Controlled paired benchmark and H02

The repository's saved hypothesis-test file reports 540 paired trials and a two-sided Wilcoxon signed-rank test: statistic = 64,380; (p=0.0170); (alpha=0.05). It reports median absolute error of (7.9460\times10^{-6}) for QAE and (1.6129\times10^{-4}) for Monte Carlo in that benchmark. Under that benchmark's stated test, **reject H02**: the errors differ significantly, with lower median error for QAE.

**Qualification:** This statistical conclusion applies to the saved paired benchmark, not automatically to the live conjunction table. The accuracy sweep contains synthetic known probabilities, and the QAE simulator's theoretical query complexity must be distinguished from wall-clock performance. The available evidence does not show that QAE is faster on the current classical simulator.

## Objective 4 — Compare QKD and classical data transmission

The saved `benchmark_summary.json` reports 50 trials per condition:

| Metric | Classical ECDH + AES-GCM | QKD, honest channel | QKD, simulated intercept-resend |
|---|---:|---:|---:|
| Message delivery rate | 100% (50/50) | 100% (50/50) | 100% (50/50 transport; not secure acceptance) |
| Mean protocol/runtime | 0.00428 s | 0.04495 s | 0.09339 s |
| Mean communication overhead | 194 public-key bytes exchanged | 256 qubits + 575.92 classical bits | 512 qubits + 575.52 classical bits |
| Mean QBER | Not applicable | 0% | 26.29% |
| Secure delivery rate | Not measured as QKD | 100% | 0% |
| Simulated interception detection | No built-in QKD-style QBER detection mechanism | 0% false alarms in 50 honest trials | 100% (50/50) |

The separate single-run JSON records 512 transmitted qubits for an honest run, 1,024 for its intercepted run, and successful message round-trip verification. These are different run settings from the 50-trial benchmark (256 and 512 mean qubits), so the figures should be reported as separate experiments.

The saved statistical report says delivery rates did not differ between classical and honest QKD (Fisher's exact test (p=1.00)). It reports a significant difference between QKD's detection in the simulated intercept-resend condition and its honest-channel false-positive rate ((p\approx1.98\times10^{-29})). It also cautions that classical transmission's “0 detection” is the absence of a QKD-style detection mechanism by design, not a measured classical detector failure.

**Finding:** In this simulation, both methods delivered messages reliably under normal conditions. Classical transmission had lower runtime and lower reported setup overhead. QKD detected the simulated interception pattern and rejected secure delivery when the QBER was high. These are software-simulation results, not a demonstration of a physical quantum communication link or proof that QKD is universally superior.

## Objective 5 — Classify conjunctions as High, Medium, or Low

The saved dashboard snapshot classifies all 44 events as LOW under its probability-based classifier. The saved `predictions.csv` likewise labels the listed rows LOW in both probability-based and composite-priority fields. No MEDIUM or HIGH event appears in those saved outputs.

The current classification logic distinguishes collision probability from a separate composite response-priority score. That score uses miss distance, relative velocity, uncertainty, and orbital geometry; it is a prioritization aid, not collision probability. The available saved rows show modeled altitude and inclination differences, but they do not provide enough synchronized evidence to establish the exact time-to-closest-approach threshold performance for every event.

**Finding:** The system applies the LOW/MEDIUM/HIGH labels, but the current saved results do not demonstrate successful detection of known high-risk cases. All-LOW output may reflect the probability/uncertainty model and configured thresholds rather than proof that the orbital environment contains no high-risk conjunctions. This objective requires a defined threshold table and validation against independently sourced CDM/ground-truth events.

## Objective 6 — Prioritize debris for removal using risk, collision probability, and consequence

The available outputs do not justify claiming that any object is a confirmed removal target: all listed conjunctions are labeled LOW and their sampled collision counts are zero. A defensible prototype ranking can only identify **candidates for further review**, not prescribe removal. For a removal-priority list, rank confirmed nonfunctional debris using (1) validated conjunction/collision risk, (2) orbit and object-removal feasibility, and (3) consequence such as repeated encounters with active spacecraft and relative encounter speed. Do not prioritize an object solely because it appears in a close-approach pair, and do not treat an active satellite as debris.

The saved dashboard's closest pair was IRIDIUM 33 DEB (NORAD 33866) and COSMOS 2251 DEB (NORAD 33766), with a reported miss distance of 18.1959 km and relative speed of 4.4697 km/s at the reported TCA. This is a candidate for orbital-risk review, **not** a validated removal recommendation. Its miss distance is not a collision, and its TCA in the saved snapshot is historical.

**Finding:** ORION-X can support preliminary screening and ranking, but the current saved data are insufficient to establish a validated debris-removal priority list. Such a list needs synchronized current orbital data, object status confirmation, consistent event outputs, consequence criteria, and removal-feasibility constraints.

## Hypothesis decisions

| Hypothesis | Evidence in saved outputs | Decision / interpretation |
|---|---|---|
| H01: ORION-X could not accurately detect and classify LEO conjunctions under predefined criteria. | 44 events detected in the dashboard snapshot; all labeled LOW. No complete independent ground-truth event set or false-negative count is included in that summary. | **Inconclusive from the available saved results.** Do not claim H01 is rejected until detection/classification performance is tested against a verified reference set. |
| H02: No significant difference between QAE and classical Monte Carlo collision-probability estimates. | Paired benchmark (n=540), Wilcoxon (p=0.0170<0.05); reported QAE median error is lower. | **Reject H02 for the saved paired benchmark.** State that this does not demonstrate faster wall-clock execution or live operational superiority. |
| H03: No significant difference between QKD and classical data transmission. | Delivery rate 100% for both in 50 trials ((p=1.00)); QKD intercept-resend detection 50/50 and honest false alarms 0/50. Classical ECDH/AES has no equivalent QBER detection mechanism. | **Partially reject / refine H03.** No delivery-rate difference was found; a difference was found for simulated interception detection, but that comparison is not like-for-like because classical transmission lacks the same detection mechanism. If H03 is intended to cover all four metrics as one hypothesis, report separate tests per metric rather than a single blanket decision. |

## Overall conclusion

The saved repository results show that ORION-X implements a pipeline for conjunction screening, collision-probability estimation, QAE-versus-Monte-Carlo experiments, risk classification, and simulated secure-message transmission. The strongest quantitative evidence in the stored outputs is the 44-event dashboard snapshot, the million-sample Monte Carlo runs with zero observed hits and a nonzero upper confidence bound, the paired QAE benchmark with a statistically lower median error, and the QKD simulation's 50/50 detection of intercept-resend trials.

The results do **not** yet establish real-world collision-probability accuracy, validated conjunction-detection accuracy, operational quantum speedup, physical QKD security, or a final debris-removal target list. Synchronize the run outputs and validate against independent CDMs before presenting those claims.

## Repository files used

- `results/dashboard_summary.json`
- `results/data_quality_report.json`
- `results/monte_carlo_results.csv`
- `results/predictions.csv`
- `results/qae_results.csv`
- `results/qae_comparison.csv`
- `results/qae_accuracy_sweep.csv`
- `results/hypothesis_test_results.json`
- `results/benchmark_summary.json`
- `results/classical_security_results.json`
- `results/qkd_results.json`
- `results/conjunction_pairs_vs_miss_distance.csv`
