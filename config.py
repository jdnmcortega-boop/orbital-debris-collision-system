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
PROPAGATED_GRID_FILE = PROCESSED_DATA_DIR / "propagated_grid.csv"
CONJUNCTIONS_FILE = PROCESSED_DATA_DIR / "conjunctions.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# TIME GRID SETTINGS
# ============================================================
# All objects are propagated to this SAME set of timestamps so
# their positions are directly comparable for conjunction screening.

GRID_START = datetime.now(timezone.utc)   # swap for a fixed datetime for reproducible runs
GRID_DURATION_HOURS = 168
GRID_STEP_MINUTES = 1

# ============================================================
# CONJUNCTION SCREENING
# ============================================================

SCREENING_DISTANCE_KM = 68.0

# 1-sigma isotropic position uncertainty per object, per axis. TLE-derived
# uncertainty grows substantially the longer you propagate without a fresh
# update — commonly several to tens of km after a week, especially
# along-track. 1 km is unrealistically tight for 7-day-old propagation and
# makes every analytic/QAE probability underflow to exactly 0.0.
POSITION_UNCERTAINTY_KM = 10.0

# ============================================================
# MONTE CARLO / COLLISION PROBABILITY (used later)
# ============================================================

MC_SAMPLES = 10000
HARD_BODY_RADIUS_KM = 0.02  # combined radius of both objects, ~20 m default

# ============================================================
# ENSURE OUTPUT FOLDERS EXIST
# ============================================================

def ensure_dirs():
    for d in (PROCESSED_DATA_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)