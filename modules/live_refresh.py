"""Refresh current orbital data and rebuild all present-day ORION-X outputs.

Historical replay archives and fixed benchmark experiments are intentionally
left untouched. Live data is refreshed from CelesTrak and limited to the
project's existing curated NORAD catalog to avoid an accidental all-pairs
explosion from importing thousands of debris objects.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import json

import pandas as pd

import config
from modules import (
    data_loader,
    sgp4_propagation,
    conjunction_detection,
    monte_carlo,
    prediction,
    preprocessing,
    false_positive,
    qae,
    reentry_risk,
    classical_security,
    qkd,
)

# Query by the debris name prefix rather than downloading each full group.
# These NAME queries return the same curated debris families with a smaller response.
CELESTRAK_QUERIES = (
    ("FENGYUN-1C-DEBRIS", "FENGYUN 1C DEB"),
    ("IRIDIUM-33-DEBRIS", "IRIDIUM 33 DEB"),
    ("COSMOS-2251-DEBRIS", "COSMOS 2251 DEB"),
)

def _progress(callback, message):
    if callback is not None:
        callback(message)

def fetch_current_curated_orbital_data(existing_df=None, progress_callback=None):
    """Fetch current CelesTrak group CSVs and refresh only curated NORAD IDs."""
    if existing_df is None:
        existing_df = data_loader.load_orbital_data()

    if existing_df is None or existing_df.empty:
        raise ValueError("The existing curated orbital catalog is empty.")

    curated_ids = set(
        pd.to_numeric(existing_df["NORAD_CAT_ID"], errors="coerce")
        .dropna().astype("int64").tolist()
    )
    if not curated_ids:
        raise ValueError("No valid NORAD catalog IDs exist in the current dataset.")

    frames = []
    for group, name_query in CELESTRAK_QUERIES:
        _progress(progress_callback, f"Downloading current CelesTrak name query: {group}")
        # CelesTrak supports NAME queries. Request only the debris family needed
        # instead of the full GROUP response, which can be slow from hosted apps.
        query = urlencode({"NAME": name_query, "FORMAT": "CSV"})
        url = f"https://celestrak.org/NORAD/elements/gp.php?{query}"
        request = Request(
            url,
            headers={
                "User-Agent": "ORION-X-research-dashboard/1.0",
                "Accept": "text/csv,*/*",
            },
        )
        try:
            with urlopen(request, timeout=45) as response:
                payload = response.read()
        except Exception as exc:
            raise RuntimeError(
                f"Could not download current CelesTrak debris query {group}. "
                "The request timed out or the host was unreachable. "
                "Do not repeatedly click refresh; check the app deployment/network "
                "and try again later. Details: "
                f"{exc}"
            ) from exc

        if not payload.strip():
            continue
        try:
            group_df = pd.read_csv(BytesIO(payload))
        except Exception as exc:
            raise RuntimeError(
                f"CelesTrak returned unreadable CSV for {group}: {exc}"
            ) from exc

        group_df.columns = [str(column).strip().upper() for column in group_df.columns]
        if "NORAD_CAT_ID" not in group_df.columns:
            raise ValueError(
                f"CelesTrak group {group} is missing NORAD_CAT_ID."
            )
        group_df["NORAD_CAT_ID"] = pd.to_numeric(
            group_df["NORAD_CAT_ID"], errors="coerce"
        )
        group_df = group_df[
            group_df["NORAD_CAT_ID"].isin(curated_ids)
        ].copy()
        if not group_df.empty:
            frames.append(group_df)

    if not frames:
        raise RuntimeError(
            "CelesTrak returned no objects matching the project's curated NORAD IDs. "
            "The existing orbital file was not replaced."
        )

    fresh = pd.concat(frames, ignore_index=True, sort=False)
    fresh.columns = [str(column).strip().upper() for column in fresh.columns]

    # Retain the exact input schema required by the existing ORION-X pipeline.
    required = data_loader.REQUIRED_COLUMNS
    missing = [column for column in required if column not in fresh.columns]
    if missing:
        raise ValueError(
            "Current CelesTrak data is missing required fields: "
            + ", ".join(missing)
            + ". The existing orbital file was not replaced."
        )

    fresh["EPOCH"] = pd.to_datetime(fresh["EPOCH"], utc=True, errors="coerce")
    fresh = fresh.dropna(subset=["EPOCH", "NORAD_CAT_ID"])
    fresh["NORAD_CAT_ID"] = fresh["NORAD_CAT_ID"].astype("int64")

    for column in (
        "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR",
        "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
    ):
        fresh[column] = pd.to_numeric(fresh[column], errors="coerce")

    fresh = fresh.dropna(subset=[
        "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR",
        "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
    ])
    fresh = fresh[
        (fresh["MEAN_MOTION"] > 0)
        & (fresh["ECCENTRICITY"] >= 0)
        & (fresh["ECCENTRICITY"] < 1)
        & (fresh["INCLINATION"] >= 0)
        & (fresh["INCLINATION"] <= 180)
    ]
    fresh = (
        fresh.sort_values("EPOCH")
        .drop_duplicates("NORAD_CAT_ID", keep="last")
        .reset_index(drop=True)
    )

    if len(fresh) < min(5, len(curated_ids)):
        raise RuntimeError(
            f"Only {len(fresh)} curated objects could be refreshed from CelesTrak; "
            "the existing orbital file was not replaced."
        )

    # Keep the required schema first; retain extra source metadata after it.
    extra = [column for column in fresh.columns if column not in required]
    fresh = fresh[required + extra]

    # Validate with the same checks used by the normal data loader before writing.
    bad = fresh[
        (fresh["ECCENTRICITY"] < 0) | (fresh["ECCENTRICITY"] >= 1)
        | (fresh["INCLINATION"] < 0) | (fresh["INCLINATION"] > 180)
        | (fresh["MEAN_MOTION"] <= 0)
    ]
    if not bad.empty:
        raise ValueError("Fresh CelesTrak data failed orbital-element validation.")

    _progress(
        progress_callback,
        f"Validated {len(fresh)} refreshed orbital objects; replacing the stale input file",
    )
    config.ORBITAL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config.ORBITAL_DATA_FILE.with_suffix(".csv.tmp")
    fresh.to_csv(temp_path, index=False)
    temp_path.replace(config.ORBITAL_DATA_FILE)
    return fresh

def _clear_live_outputs(write_empty_warning=True):
    """Remove stale current-data outputs so old results cannot masquerade as fresh."""
    paths = [
        config.PROPAGATED_GRID_FILE,
        config.CONJUNCTIONS_FILE,
        config.RESULTS_DIR / "monte_carlo_results.csv",
        config.RESULTS_DIR / "predictions.csv",
        config.RESULTS_DIR / "qae_comparison.csv",
        config.RESULTS_DIR / "false_positive_analysis.csv",
        config.RESULTS_DIR / "reentry_analysis.csv",
        config.RESULTS_DIR / "classical_security_results.json",
        config.RESULTS_DIR / "qkd_results.json",
    ]
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Keep going; the caller reports a successful refresh only if the
            # main pipeline itself completes.
            pass

    warnings_path = config.RESULTS_DIR / "warnings.txt"
    warnings_path.write_text(
        "No current conjunctions were found inside the configured screening distance.\n",
        encoding="utf-8",
    )

def refresh_all_live_results(progress_callback=None):
    """Refresh CelesTrak inputs, then rebuild the current-data result products."""
    config.ensure_dirs()
    _progress(progress_callback, "Validating the existing curated orbital catalog")
    old_df = data_loader.load_orbital_data()

    fresh_df = fetch_current_curated_orbital_data(
        existing_df=old_df,
        progress_callback=progress_callback,
    )

    # Fresh input is now validated and safely written. Clear every derived
    # current-data output before rebuilding so a later calculation failure
    # cannot leave yesterday's results looking current.
    _clear_live_outputs(write_empty_warning=False)

    # Re-anchor this forecast at the actual refresh time, not the app boot time.
    config.GRID_START = datetime.now(timezone.utc)

    _progress(progress_callback, "Propagating refreshed objects across the 30-day grid")
    propagated, failed_objects = sgp4_propagation.propagate_and_save(fresh_df)
    if propagated is None or propagated.empty:
        raise RuntimeError(
            "The refreshed orbital data produced no valid SGP4 states. "
            "The old derived results have been left in place."
        )

    _progress(progress_callback, "Re-screening all object pairs for conjunctions")
    conjunctions = conjunction_detection.detect_and_save(propagated_df=propagated)

    _progress(progress_callback, "Refreshing data-quality and stale-element reports")
    quality_report = preprocessing.summary_report(
        fresh_df, reference_time=config.GRID_START
    )
    stale_tles = preprocessing.flag_stale_tles(
        fresh_df, reference_time=config.GRID_START
    )
    with open(config.RESULTS_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, default=str)
    stale_tles.to_csv(config.RESULTS_DIR / "stale_tles.csv", index=False)

    if conjunctions is None or conjunctions.empty:
        _clear_live_outputs()
        # An empty conjunction CSV is the new current result, not a stale prior run.
        conjunctions = pd.DataFrame(columns=[
            "OBJECT_A", "NORAD_A", "OBJECT_B", "NORAD_B", "TCA",
            "DAYS_TO_TCA", "FORECAST_HORIZON_DAYS", "FORECAST_STATUS",
            "MISS_DISTANCE_KM", "RELATIVE_VELOCITY_KM_S",
        ])
        conjunctions.to_csv(config.CONJUNCTIONS_FILE, index=False)
        _progress(
            progress_callback,
            "Refreshing warning-security outputs for the current no-conjunction state",
        )
        classical_security.run_and_save()
        qkd.run_and_save(n_qubits=512)
        _progress(
            progress_callback,
            "Refresh complete: no current pairs passed the screening threshold; stale risk outputs cleared",
        )
        return {
            "orbital_objects": int(fresh_df["NORAD_CAT_ID"].nunique()),
            "propagated_states": int(len(propagated)),
            "failed_objects": int(len(failed_objects)),
            "conjunctions": 0,
            "monte_carlo_rows": 0,
            "prediction_rows": 0,
            "data_epoch_latest_utc": pd.to_datetime(fresh_df["EPOCH"], utc=True).max().isoformat(),
            "forecast_start_utc": config.GRID_START.isoformat(),
            "status": "completed_no_conjunctions",
        }

    _progress(progress_callback, "Recalculating collision probabilities with hybrid Monte Carlo")
    mc_results = monte_carlo.run_monte_carlo(
        conjunctions,
        propagated,
        orbital_data_df=fresh_df,
        verbose=False,
    )
    mc_results.to_csv(config.RESULTS_DIR / "monte_carlo_results.csv", index=False)

    _progress(progress_callback, "Rebuilding 30-day risk classifications and warning messages")
    predictions = prediction.run_and_save()
    if predictions is None:
        raise RuntimeError("Prediction generation returned no result.")

    _progress(progress_callback, "Refreshing QAE-vs-classical comparison for current conjunctions")
    qae_results = qae.run_and_save()

    _progress(progress_callback, "Refreshing false-positive analysis")
    fp_results = false_positive.run_and_save()

    _progress(progress_callback, "Refreshing re-entry and object-type analysis")
    reentry_results = reentry_risk.run_and_save()

    _progress(progress_callback, "Refreshing classical and BB84/QKD warning-security results")
    classical_security.run_and_save()
    qkd.run_and_save(n_qubits=512)

    _progress(progress_callback, "All current/live result files have been replaced")
    return {
        "orbital_objects": int(fresh_df["NORAD_CAT_ID"].nunique()),
        "propagated_states": int(len(propagated)),
        "failed_objects": int(len(failed_objects)),
        "conjunctions": int(len(conjunctions)),
        "monte_carlo_rows": int(len(mc_results)),
        "prediction_rows": int(len(predictions)),
        "qae_rows": int(len(qae_results)) if qae_results is not None else 0,
        "false_positive_rows": int(len(fp_results)) if fp_results is not None else 0,
        "reentry_rows": int(len(reentry_results)) if reentry_results is not None else 0,
        "data_epoch_latest_utc": pd.to_datetime(fresh_df["EPOCH"], utc=True).max().isoformat(),
        "forecast_start_utc": config.GRID_START.isoformat(),
        "status": "completed",
    }
