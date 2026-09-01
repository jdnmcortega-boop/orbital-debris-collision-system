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
#
# LIVE mode:
#   Use the latest available orbital record for each NORAD object
#   and forecast forward from the current UTC time.
#
# BACKTEST mode:
#   Use only orbital records whose EPOCH is at or before the explicit
#   historical cutoff. The forecast starts at that cutoff and runs
#   forward for 30 days. This prevents future information from
#   leaking into a historical forecast.
#
# Set with environment variables when needed:
#   $env:FORECAST_MODE="backtest"
#   $env:FORECAST_START_UTC="2026-08-25T00:00:00Z"
#

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
# MONTE CARLO / COLLISION PROBABILITY
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
