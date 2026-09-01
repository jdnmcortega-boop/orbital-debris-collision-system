from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

ORBITAL_DATA_FILE = DATA_DIR / "orbital_data.csv"
PROPAGATED_GRID_FILE = PROCESSED_DATA_DIR / "propagated_objects.csv"
CONJUNCTIONS_FILE = PROCESSED_DATA_DIR / "conjunctions.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# TIME GRID SETTINGS
# ============================================================
# All objects are propagated to the SAME future timestamps so
# their positions can be compared directly for conjunction
# screening.
#
# The forecasting horizon is now 30 days rather than the old
# 7-day window (168 hours).

FORECAST_HORIZON_DAYS = 30
GRID_START = datetime.now(timezone.utc)
GRID_DURATION_HOURS = FORECAST_HORIZON_DAYS * 24
GRID_STEP_MINUTES = 3

# ============================================================
# CONJUNCTION SCREENING
# ============================================================

SCREENING_DISTANCE_KM = 68.0

# 1-sigma isotropic position uncertainty per object, per axis.
POSITION_UNCERTAINTY_KM = 10.0

# ============================================================
# MONTE CARLO / COLLISION PROBABILITY (used later)
# ============================================================

MC_SAMPLES = 1000000
HARD_BODY_RADIUS_KM = 0.02

# ============================================================
# ENSURE OUTPUT FOLDERS EXIST
# ============================================================

def ensure_dirs():
    for d in (PROCESSED_DATA_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# QAE BENCHMARK SETTINGS
# ============================================================

QAE_BENCHMARK_PROBABILITIES = [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50
]

QAE_BENCHMARK_SAMPLES = 100000

QAE_PROBABILITY_SCALE = 1_000_000.0
QAE_EVALUATION_QUBITS = 6
