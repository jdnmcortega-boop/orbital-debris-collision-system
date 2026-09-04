# ORION-X — Orbital Risk & Intelligence Operations Network

## Research Scope

ORION-X investigates the application of quantum and hybrid
quantum-classical computing to orbital-debris collision-risk analysis.

The current software phase focuses on four major functions:

1. Detection of potential orbital conjunctions.
2. Calculation of collision probability.
3. Prediction and classification of collision risk.
4. Simulation of secure communication of collision warnings.

## Quantum Computing

Quantum Amplitude Estimation (QAE) is investigated as a quantum
approach to probability estimation.

The QAE results will be compared with a classical Monte Carlo
baseline.

The study will distinguish theoretical quantum query complexity
from practical execution performance.

## Data

Orbital data will be obtained from publicly accessible orbital-data
sources where permitted.

Potential sources include:

- LeoLabs
- Orbital Radar
- SatelliteMap.space
- CelesTrak
- NASA
- ESA

Data obtained from external sources will be distinguished from
calculated and simulated data.

## Current Experimental Pipeline

Raw orbital data
        ↓
Data preprocessing
        ↓
Conjunction detection
        ↓
Collision probability estimation
        ↓
Monte Carlo vs QAE
        ↓
Risk prediction
        ↓
False-positive analysis
        ↓
Country/operator identification
        ↓
Secure warning communication
        ↓
Classical security vs QKD simulation

## Dashboard: PAST vs PRESENT

Run the main ORION-X dashboard with:

```text
streamlit run ui/dashboard.py
```

The dashboard now has two intentionally separate modes:

### PAST — Historical Replay / Validation

Uses confirmed historical collision events and archived TLE/3LE data. At
each replay timestamp, only an element set with epoch at or before that
timestamp may be used. The known collision is retained only as ground
truth for evaluation.

The historical experiment provides:

- 30-day rewind before the collision
- SGP4 historical propagation
- Forecast closest approach
- Relative velocity
- Collision probability
- QAE estimate
- Matched-budget Monte Carlo estimate
- Risk classification
- Replay timeline
- Warning lead-time measurement
- Comparison against the actual historical event

The current repository contains an archived Iridium 33 / Cosmos 2251
experiment under:

```text
data/historical_events/iridium33_cosmos2251/historical_tles.3le
```

### PRESENT — Live 30-Day Forecast

Uses the current orbital dataset in `data/orbital_data.csv` and the
existing current-data SGP4/conjunction/prediction pipeline. Historical
collision outcomes are not fed into this predictor.

The live mode provides:

- Current orbital objects
- Current SGP4 propagation
- Current conjunction screening
- Existing 30-day prediction results
- Current HIGH/CRITICAL forecast rows

The historical and live modes must remain separate: historical data is
for validation, while current orbital data is for present-day forecasting.

## Current Limitations

ORION-X is a research prototype and is not an operational
collision-avoidance system.

It does not control spacecraft or physically intercept orbital debris.

Physical interception and debris-removal hardware are outside the
scope of the current software phase.
