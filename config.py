from pathlib import Path
from datetime import datetime, timezone
import os

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
# FORECAST SETTINGS
# ============================================================
FORECAST_MODE = os.getenv("FORECAST_MODE", "live").strip().lower()
FORECAST_HORIZON_DAYS = 30
_forecast_start_text = os.getenv("FORECAST_START_UTC", "").strip()

if FORECAST_MODE == "backtest":
    if not _forecast_start_text:
        raise ValueError(
            "FORECAST_START_UTC must be set when FORECAST_MODE=backtest. "
            "Example: 2026-08-25T00:00:00Z"
        )
    GRID_START = datetime.fromisoformat(
        _forecast_start_text.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
else:
    GRID_START = datetime.now(timezone.utc)

GRID_DURATION_HOURS = FORECAST_HORIZON_DAYS * 24
GRID_STEP_MINUTES = 3

# ============================================================
# CONJUNCTION SCREENING
# ============================================================
SCREENING_DISTANCE_KM = 68.0
POSITION_UNCERTAINTY_KM = 10.0

# ============================================================
# COLLISION PROBABILITY / MONTE CARLO
# ============================================================
# Collision probability is evaluated in the 2-D encounter plane,
# matching the noncentral-chi-square analytic model used by QAE.
# Importance sampling draws directly inside the hard-body disk and
# applies the exact Gaussian likelihood weight. This avoids the
# zero-hit problem of naive 3-D brute-force MC at ~1e-6 probabilities.
MC_SAMPLES = 1000000
HARD_BODY_RADIUS_KM = 0.02
MC_METHOD = "encounter_plane_importance_sampling"

# ============================================================
# QAE BENCHMARK SETTINGS
# ============================================================
QAE_BENCHMARK_PROBABILITIES = [0.01, 0.05, 0.10, 0.25, 0.50]
QAE_BENCHMARK_SAMPLES = 100000
QAE_PROBABILITY_SCALE = 1_000_000.0
QAE_EVALUATION_QUBITS = 6


def ensure_dirs():
    for d in (PROCESSED_DATA_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
