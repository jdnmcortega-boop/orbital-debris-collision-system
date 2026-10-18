# ORION-X Live Research Results

**Generated (UTC):** 2026-10-18T22:46:32.933137+00:00

> This file is generated automatically after a successful live refresh. It reports current-run output only. Fixed benchmarks and historical CDM tests are not presented as new live measurements.

## Current run summary

- Orbital objects: 150
- Latest orbital element epoch: 2026-10-09T23:33:49.876000+00:00
- Propagated state rows: 2,160,150
- Failed objects: 0
- Detected conjunctions: 808
- Analytical Pc rows: 808
- Prediction rows: 808
- QAE rows: 808
- Monte Carlo rows: 0 (the live pipeline skips MC by default)

## Objective evidence

### Objective 1

Estimate collision probability using relative motion, orbital information, modeled uncertainty, and collision-region probability.

**Current-run evidence:**

- conjunction count: 808
- miss distance km: {'column': 'MISS_DISTANCE_KM', 'count': 808, 'min': 0.7455414072067872, 'median': 39.78392091283804, 'max': 67.94431136680203}
- relative velocity km s: {'column': 'RELATIVE_VELOCITY_KM_S', 'count': 808, 'min': 0.02129990126030805, 'median': 8.54108738635285, 'max': 15.219494920608014}
- analytical pc: {'column': 'ANALYTIC_PC', 'count': 808, 'min': 1.8174914685552013e-09, 'median': 3.4189926933653355e-08, 'max': 3.7253308058050087e-07}
- modeled uncertainty fields available: True
- maximum pc at zero miss stats: {'column': 'ANALYTIC_PC_MAX_AT_ZERO_MISS', 'count': 808, 'min': 1.846706779382554e-09, 'median': 4.5828984825845e-08, 'max': 4.022369229668246e-07}
- monte carlo rows: 0

**Limitation:** The analytical model uses modeled uncertainty unless valid object-specific covariance is imported and correctly transformed. Live Monte Carlo is not run by default; a missing MC table is not a zero-probability result.

### Objective 2

Evaluate conjunction detection.

**Current-run evidence:**

- objects: 150
- propagated state rows: 2160150
- detected conjunctions: 808
- tca column present: True
- miss distance km: {'column': 'MISS_DISTANCE_KM', 'count': 808, 'min': 0.7455414072067872, 'median': 39.78392091283804, 'max': 67.94431136680203}

**Limitation:** Precision, recall, false-positive rate, and false-negative rate require a verified reference event set for the same time window. They are not inferred from event counts alone.

### Objective 3

Compare QAE with classical Monte Carlo.

**Current-run evidence:**

- qae rows: 808
- monte carlo rows: 0
- analytical pc rows: 808

**Limitation:** A live paired significance test is not run unless paired current-run QAE and Monte Carlo estimates are available. The fixed benchmark hypothesis test is retained separately and is not overwritten by this refresh.

### Objective 4

Compare QKD and classical secure transmission.

**Current-run evidence:**

- classical security output present: True
- qkd output present: True
- classical result keys: ['ciphertext_bytes', 'handshake_runtime_sec', 'method', 'plaintext_bytes', 'public_key_bytes_exchanged', 'round_trip_verified']
- qkd result keys: ['ciphertext_bytes', 'final_key_length_bits_honest', 'honest_eavesdropping_detected', 'honest_qber', 'intercepted_eavesdropping_detected', 'intercepted_qber', 'message_round_trip_verified', 'qubits_transmitted_honest', 'qubits_transmitted_intercepted']

**Limitation:** The refresh writes per-run security outputs, but a single run is not a statistical benchmark. The fixed multi-trial benchmark and its hypothesis test are kept distinct.

### Objective 5

Classify conjunctions by collision probability and response priority.

**Current-run evidence:**

- collision probability risk counts: {'LOW': 808}
- response priority counts: {'LOW': 694, 'MEDIUM': 110, 'HIGH': 4}
- probability source counts: {'Analytical Pc (assumed uncertainty model)': 808}

**Limitation:** Collision-probability classification and response-priority ranking are separate metrics. A lack of HIGH labels is not proof that no real conjunction is dangerous.

### Objective 6

Prioritize debris for further assessment and possible removal.

**Current-run evidence:**

- reentry analysis rows: 808
- debris involved events: 764

**Limitation:** This is a screening list, not an authorization or recommendation to remove a specific object. Verify nonfunctional debris status, consequences, uncertainty, and removal feasibility; never classify an active spacecraft as a removal target.

## Hypothesis status

### H01 — INCONCLUSIVE

**Null hypothesis:** ORION-X does not meet the predefined conjunction-detection and classification accuracy criteria.

**Status:** NOT_TESTED_IN_THIS_REFRESH

A current conjunction count alone cannot measure accuracy. A matched, verified reference set with false-positive and false-negative accounting is required.

### H02 — INCONCLUSIVE

**Null hypothesis:** There is no statistically significant difference between QAE and Monte Carlo estimation errors.

**Status:** NOT_TESTED_IN_THIS_REFRESH

The live pipeline intentionally skips Monte Carlo by default. The fixed benchmark test is separate and must not be described as a fresh live-run test.

### H03 — INCONCLUSIVE_FOR_THIS_REFRESH

**Null hypothesis:** There is no statistically significant difference between QKD and classical transmission under the specified test conditions.

**Status:** PER_RUN_OUTPUTS_GENERATED_BUT_NO_NEW_MULTI_TRIAL_TEST

A refresh creates current security output files, but does not perform a multi-trial paired or independent-sample experiment. Preserve the fixed benchmark test as a separately labeled result.

## Interpretation guardrails

- A zero Monte Carlo hit count is not proof that true Pc equals zero.
- No current-run H01 decision is made without verified reference events.
- H02 is not tested on live data unless paired live QAE and Monte Carlo results exist.
- H03 is not statistically retested from one security run; use multi-trial benchmark data.
- HIGH response priority is not the same as HIGH collision probability.
- Debris screening candidates require independent status and feasibility checks before any removal recommendation.
