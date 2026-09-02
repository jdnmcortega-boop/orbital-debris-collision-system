"""
Main Streamlit dashboard.

The dashboard is intentionally split into two scientifically distinct modes:

    PAST    -> historical collision replay / validation
    PRESENT -> live current-data 30-day forecasting

The implementation lives in historical_live_dashboard.py so the two modes
remain isolated from one another and from the underlying scientific modules.

Run:
    streamlit run ui/dashboard.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.historical_live_dashboard import main


if __name__ == "__main__":
    main()
