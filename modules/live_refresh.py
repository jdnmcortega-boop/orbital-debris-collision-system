"""Rebuild present-day ORION-X outputs from a bounded orbital catalog.

Historical replay archives and fixed benchmark experiments are intentionally
left untouched. Live propagation is capped at 120 objects to limit memory use.
An explicitly uploaded CSV can opt into the CelesTrak replacement path.
The saved catalog is not overwritten during the default refresh.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import json
import importlib

import pandas as pd

import config
from modules import (
    data_loader,
    sgp4_propagation,
    conjunction_detection,
    monte_carlo,
    analytic_pc,
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

def fetch_current_curated_orbital_data(existing_df=None, progress_callback=None, uploaded_csv=None):
    """Use saved orbital_data.csv by default; optionally process an uploaded CelesTrak CSV."""
    if existing_df is None:
        existing_df = data_loader.load_orbital_data()

    if existing_df is None or existing_df.empty:
        raise ValueError("The existing curated orbital catalog is empty.")

    # Default source of truth: use every object already stored in
    # data/orbital_data.csv. This lets the hosted app rebuild forecasts even
    # when its network cannot reach CelesTrak. Only an explicitly uploaded CSV
    # opts into the external-data replacement path below.
    if uploaded_csv is None:
        catalog = existing_df.copy()
        catalog.columns = [str(column).strip().upper() for column in catalog.columns]
        required = data_loader.REQUIRED_COLUMNS
        missing = [column for column in required if column not in catalog.columns]
        if missing:
            raise ValueError(
                "The existing orbital_data.csv is missing required columns: "
                + ", ".join(missing)
            )

        catalog["EPOCH"] = pd.to_datetime(catalog["EPOCH"], utc=True, errors="coerce")
        catalog["NORAD_CAT_ID"] = pd.to_numeric(catalog["NORAD_CAT_ID"], errors="coerce")
        numeric_columns = [
            "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR",
            "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
        ]
        for column in numeric_columns:
            catalog[column] = pd.to_numeric(catalog[column], errors="coerce")

        catalog = catalog.dropna(subset=["EPOCH", "NORAD_CAT_ID"] + numeric_columns)
        catalog = catalog[
            (catalog["MEAN_MOTION"] > 0)
            & (catalog["ECCENTRICITY"] >= 0)
            & (catalog["ECCENTRICITY"] < 1)
            & (catalog["INCLINATION"] >= 0)
            & (catalog["INCLINATION"] <= 180)
        ]
        catalog["NORAD_CAT_ID"] = catalog["NORAD_CAT_ID"].astype("int64")
        catalog = (
            catalog.sort_values("EPOCH")
            .drop_duplicates("NORAD_CAT_ID", keep="last")
            .sort_values("NORAD_CAT_ID")
            .reset_index(drop=True)
        )
        if catalog.empty:
            raise ValueError("No valid orbital objects remain in orbital_data.csv.")

        # Keep the saved catalog and the live forecast workload aligned.
        # Prefer 35 objects from each debris family plus 15 non-debris objects;
        # fill any unused slots from the remaining catalog, never above 120.
        live_object_cap = 120
        if len(catalog) > live_object_cap:
            names = catalog["OBJECT_NAME"].fillna("").astype(str).str.upper()
            selected_parts = []
            for pattern in (
                r"^FENGYUN 1C DEB$",
                r"^IRIDIUM 33 DEB$",
                r"^COSMOS 2251 DEB$",
            ):
                family = catalog.loc[names.str.match(pattern, na=False)].sort_values(
                    "EPOCH", ascending=False
                ).head(35)
                if not family.empty:
                    selected_parts.append(family)

            selected = (
                pd.concat(selected_parts, ignore_index=False)
                if selected_parts
                else catalog.iloc[0:0].copy()
            )
            selected_ids = set(selected["NORAD_CAT_ID"].astype("int64").tolist())
            debris_mask = names.str.contains(
                r"\bDEB\b|DEBRIS|FRAGMENT|ROCKET BODY|\bR/B\b",
                regex=True,
                na=False,
            )
            non_debris = catalog.loc[~debris_mask].sort_values(
                "EPOCH", ascending=False
            )
            non_debris = non_debris[
                ~non_debris["NORAD_CAT_ID"].isin(selected_ids)
            ].head(15)
            selected = pd.concat([selected, non_debris], ignore_index=False)
            selected_ids = set(selected["NORAD_CAT_ID"].astype("int64").tolist())

            remaining = catalog[
                ~catalog["NORAD_CAT_ID"].isin(selected_ids)
            ].sort_values("EPOCH", ascending=False)
            slots = max(0, live_object_cap - len(selected))
            if slots:
                selected = pd.concat(
                    [selected, remaining.head(slots)], ignore_index=False
                )
            catalog = (
                selected.drop_duplicates("NORAD_CAT_ID", keep="last")
                .sort_values("NORAD_CAT_ID")
                .reset_index(drop=True)
            )

        _progress(
            progress_callback,
            f"Using {len(catalog)} validated objects from data/orbital_data.csv (live cap: {live_object_cap}); no CelesTrak download required",
        )
        extra = [column for column in catalog.columns if column not in required]
        return catalog[required + extra]

    curated_ids = set(
        pd.to_numeric(existing_df["NORAD_CAT_ID"], errors="coerce")
        .dropna().astype("int64").tolist()
    )
    if not curated_ids:
        raise ValueError("No valid NORAD catalog IDs exist in the current dataset.")

    frames = []
    download_errors = []

    if uploaded_csv is not None:
        # Manual fallback for hosted environments that cannot reach CelesTrak.
        # Accept an official GP/OMM CSV and retain the three debris families
        # used by the live forecast. The CSV is validated below before any
        # persistent input file is replaced.
        uploaded = pd.read_csv(BytesIO(uploaded_csv))
        uploaded.columns = [str(column).strip().upper() for column in uploaded.columns]
        missing = [column for column in data_loader.REQUIRED_COLUMNS if column not in uploaded.columns]
        if missing:
            raise ValueError(
                "Uploaded CSV is missing required GP fields: " + ", ".join(missing)
            )
        names = uploaded["OBJECT_NAME"].fillna("").astype(str).str.upper()
        family_patterns = (
            ("FENGYUN-1C-DEBRIS", r"FENGYUN[- ]?1C.*DEB"),
            ("IRIDIUM-33-DEBRIS", r"IRIDIUM[- ]?33.*DEB"),
            ("COSMOS-2251-DEBRIS", r"COSMOS[- ]?2251.*DEB"),
        )
        family_frames = []
        for group, pattern in family_patterns:
            family = uploaded.loc[names.str.contains(pattern, regex=True, na=False)].copy()
            if not family.empty:
                family["_SOURCE_GROUP"] = group
                family_frames.append(family)
                _progress(progress_callback, f"Read {len(family)} rows for {group} from uploaded CSV")
        if not family_frames:
            raise ValueError(
                "The uploaded CSV contains none of the configured debris families "
                "(FENGYUN-1C, IRIDIUM-33, or COSMOS-2251 debris). Upload a current "
                "CelesTrak GP CSV that includes those objects."
            )
        frames.extend(family_frames)
    else:
        for group, name_query in CELESTRAK_QUERIES:
            _progress(progress_callback, f"Downloading current CelesTrak name query: {group}")
            # Try the narrow NAME query first, then the official GROUP endpoint.
            # Hosted runners sometimes time out on one endpoint even when another
            # CelesTrak route is reachable. Parse and validate each response before
            # accepting it, so an HTML error page cannot be mistaken for orbital CSV.
            name_query_encoded = urlencode({"NAME": name_query, "FORMAT": "CSV"})
            group_query_encoded = urlencode({"GROUP": group, "FORMAT": "CSV"})
            candidates = (
                (f"https://celestrak.org/NORAD/elements/gp.php?{name_query_encoded}", "NAME query"),
                (f"https://www.celestrak.org/NORAD/elements/gp.php?{name_query_encoded}", "www NAME query"),
                (f"https://celestrak.org/NORAD/elements/gp.php?{group_query_encoded}", "GROUP query"),
            )
            group_df = None
            attempt_errors = []
            for attempt, (url, source_label) in enumerate(candidates, start=1):
                try:
                    _progress(
                        progress_callback,
                        f"Requesting {group} via {source_label} ({attempt}/{len(candidates)})",
                    )
                    request = Request(
                        url,
                        headers={
                            "User-Agent": "ORION-X-research-dashboard/1.0",
                            "Accept": "text/csv,*/*",
                        },
                    )
                    with urlopen(request, timeout=12) as response:
                        payload = response.read()
                    if not payload.strip():
                        raise RuntimeError("empty response")
                    candidate_df = pd.read_csv(BytesIO(payload))
                    candidate_df.columns = [
                        str(column).strip().upper() for column in candidate_df.columns
                    ]
                    if "NORAD_CAT_ID" not in candidate_df.columns:
                        raise RuntimeError("response is not a GP CSV (NORAD_CAT_ID missing)")
                    if candidate_df.empty:
                        raise RuntimeError("valid CSV contained no objects")
                    group_df = candidate_df
                    _progress(
                        progress_callback,
                        f"Received {len(group_df)} rows for {group} via {source_label}",
                    )
                    break
                except Exception as exc:
                    attempt_errors.append(f"{source_label}: {exc}")
                    if attempt < len(candidates):
                        _progress(
                            progress_callback,
                            f"{source_label} failed for {group}; trying the next CelesTrak route…",
                        )

            if group_df is None:
                download_errors.append(f"{group}: " + " | ".join(attempt_errors))
                _progress(
                    progress_callback,
                    f"Skipping unavailable query {group}; trying the next debris family",
                )
                continue
            group_df["NORAD_CAT_ID"] = pd.to_numeric(
                group_df["NORAD_CAT_ID"], errors="coerce"
            )
            group_df["_SOURCE_GROUP"] = group
            if not group_df.empty:
                frames.append(group_df)

    if not frames:
        details = " | ".join(download_errors) if download_errors else "No CSV rows returned."
        raise RuntimeError(
            "All configured CelesTrak debris queries failed. The existing orbital "
            "file was not replaced. Try again later or use a verified current TLE CSV. "
            f"Details: {details}"
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

    # Keep a complete copy of all validated downloaded debris for the persistent
    # catalog; the smaller selection below is only for the expensive live forecast.
    all_fresh_debris = fresh.copy()

    # Bound the expensive 30-day, 3-minute propagation to 120 objects.
    # At 14,401 timestamps per object, this can create about 1.73M state rows.
    # Keep the original time resolution while expanding the live object sample.
    live_object_cap = 120
    debris_slots_per_family = 35
    satellite_slots = 15
    selected_frames = []
    for group, _name_query in CELESTRAK_QUERIES:
        family = fresh[fresh["_SOURCE_GROUP"] == group].copy()
        if family.empty:
            continue
        family["_IS_EXISTING"] = family["NORAD_CAT_ID"].isin(curated_ids)
        family = family.sort_values(
            ["_IS_EXISTING", "EPOCH", "NORAD_CAT_ID"],
            ascending=[False, False, True],
        )
        selected_frames.append(family.head(debris_slots_per_family))

    if not selected_frames:
        raise RuntimeError(
            "CelesTrak returned no valid objects from the configured debris families; "
            "the existing orbital file was not replaced."
        )

    selected_debris = pd.concat(selected_frames, ignore_index=True, sort=False)
    selected_debris = selected_debris.drop_duplicates("NORAD_CAT_ID", keep="last")

    # Keep curated satellites/non-debris records in the persisted catalog, but
    # use only a small representative sample for the expensive live forecast.
    # Refreshed debris records replace old records with the same NORAD catalog IDs.
    old = existing_df.copy()
    old["NORAD_CAT_ID"] = pd.to_numeric(old["NORAD_CAT_ID"], errors="coerce")
    old = old.dropna(subset=["NORAD_CAT_ID"])
    old["NORAD_CAT_ID"] = old["NORAD_CAT_ID"].astype("int64")
    name_text = old["OBJECT_NAME"].fillna("").astype(str).str.upper()
    debris_mask = name_text.str.contains(
        r"\bDEB\b|DEBRIS|FRAGMENT|ROCKET BODY|\bR/B\b",
        regex=True,
        na=False,
    )
    old_satellites_full = old.loc[~debris_mask].copy()
    all_refreshed_ids = set(all_fresh_debris["NORAD_CAT_ID"].astype("int64"))
    old_satellites_full = old_satellites_full[
        ~old_satellites_full["NORAD_CAT_ID"].isin(all_refreshed_ids)
    ]
    old_satellites = old_satellites_full.sort_values(
        "EPOCH", ascending=False
    ).head(satellite_slots).copy()

    # Fill unused forecast slots with additional refreshed debris, never exceeding
    # the compute budget. The full validated debris catalog is still persisted below.
    remaining_slots = max(
        0, live_object_cap - len(selected_debris) - len(old_satellites)
    )
    if remaining_slots:
        remaining_debris = fresh[
            ~fresh["NORAD_CAT_ID"].isin(selected_debris["NORAD_CAT_ID"])
        ].sort_values(["EPOCH", "NORAD_CAT_ID"], ascending=[False, True])
        selected_debris = pd.concat(
            [selected_debris, remaining_debris.head(remaining_slots)],
            ignore_index=True,
            sort=False,
        )

    fresh = pd.concat([selected_debris, old_satellites], ignore_index=True, sort=False)
    fresh = fresh.drop(columns=["_SOURCE_GROUP", "_IS_EXISTING"], errors="ignore")
    fresh = fresh.drop_duplicates("NORAD_CAT_ID", keep="last")
    fresh = fresh.sort_values("NORAD_CAT_ID").reset_index(drop=True)

    if len(fresh) < 30:
        raise RuntimeError(
            f"Only {len(fresh)} valid objects were available after validation; "
            "the existing orbital file was not replaced. "
            + (("Download errors: " + " | ".join(download_errors)) if download_errors else "")
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
    # Persist only the bounded, validated sample actually sent to propagation.
    # This prevents a later refresh from silently restoring hundreds of objects
    # and exceeding the hosted app's memory budget.
    catalog_to_write = fresh.copy()
    catalog_to_write = catalog_to_write.drop(
        columns=["_SOURCE_GROUP", "_IS_EXISTING"], errors="ignore"
    )
    catalog_to_write = catalog_to_write.drop_duplicates(
        "NORAD_CAT_ID", keep="last"
    ).sort_values("NORAD_CAT_ID").reset_index(drop=True)
    catalog_extra = [
        column for column in catalog_to_write.columns if column not in required
    ]
    catalog_to_write = catalog_to_write[required + catalog_extra]

    config.ORBITAL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config.ORBITAL_DATA_FILE.with_suffix(".csv.tmp")
    catalog_to_write.to_csv(temp_path, index=False)
    temp_path.replace(config.ORBITAL_DATA_FILE)
    _progress(
        progress_callback,
        f"Saved {len(catalog_to_write)} catalog records; selected {len(fresh)} objects for live forecasting",
    )
    return fresh

def _clear_live_outputs(write_empty_warning=True):
    """Remove stale current-data outputs so old results cannot masquerade as fresh."""
    paths = [
        config.PROPAGATED_GRID_FILE,
        config.CONJUNCTIONS_FILE,
        config.RESULTS_DIR / "monte_carlo_results.csv",
        config.RESULTS_DIR / "analytic_pc_results.csv",
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

def refresh_all_live_results(progress_callback=None, uploaded_csv=None):
    """Refresh CelesTrak inputs, then rebuild the current-data result products."""
    config.ensure_dirs()
    _progress(progress_callback, "Validating the existing curated orbital catalog")
    old_df = data_loader.load_orbital_data()

    fresh_df = fetch_current_curated_orbital_data(
        existing_df=old_df,
        progress_callback=progress_callback,
        uploaded_csv=uploaded_csv,
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

    _progress(progress_callback, "Calculating primary analytical collision probabilities (Pc)")
    analytic_results = analytic_pc.build_analytic_pc_results(conjunctions, fresh_df)
    analytic_results.to_csv(config.RESULTS_DIR / "analytic_pc_results.csv", index=False)

    # Monte Carlo is deliberately optional: the live risk forecast must not
    # depend on a sampling estimator that may be too slow or under-resolve rare events.
    # Users can still run modules.monte_carlo separately for validation.
    _progress(progress_callback, "Reloading current prediction/QAE modules after repository updates")
    # Streamlit can retain previously imported module objects across reruns.
    # Reload these dependencies so a new live refresh does not call a stale
    # prediction.py that still requires the removed MC output file.
    importlib.invalidate_caches()
    importlib.reload(analytic_pc)
    importlib.reload(prediction)
    importlib.reload(qae)
    importlib.reload(false_positive)

    _progress(progress_callback, "Rebuilding risk classifications from analytical Pc")
    predictions = prediction.run_and_save()
    if predictions is None:
        analytic_path = config.RESULTS_DIR / "analytic_pc_results.csv"
        raise RuntimeError(
            "Prediction generation returned no result. "
            f"Expected analytical input at {analytic_path}; "
            f"exists={analytic_path.exists()}. "
            "Check logs for a stale module or missing analytical results."
        )

    _progress(progress_callback, "Refreshing QAE estimate against the analytical Pc values")
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
        "monte_carlo_rows": 0,
        "analytical_pc_rows": int(len(analytic_results)),
        "prediction_rows": int(len(predictions)),
        "qae_rows": int(len(qae_results)) if qae_results is not None else 0,
        "false_positive_rows": int(len(fp_results)) if fp_results is not None else 0,
        "reentry_rows": int(len(reentry_results)) if reentry_results is not None else 0,
        "data_epoch_latest_utc": pd.to_datetime(fresh_df["EPOCH"], utc=True).max().isoformat(),
        "forecast_start_utc": config.GRID_START.isoformat(),
        "status": "completed",
    }
