"""
Data-quality summary for the loaded orbital elements: epoch age relative
to the propagation window, and basic distribution stats. Complements
data_loader.py's structural validation with a descriptive-quality report,
useful for the "data collection/processing" section of your report.
"""

import pandas as pd

import config


def compute_epoch_age_days(orbital_data_df, reference_time=None):
    reference_time = reference_time or config.GRID_START
    epochs = pd.to_datetime(orbital_data_df["EPOCH"], utc=True)
    reference_time = pd.Timestamp(reference_time)
    if reference_time.tzinfo is None:
        reference_time = reference_time.tz_localize("UTC")
    return (reference_time - epochs).dt.total_seconds() / 86400.0


def flag_stale_tles(orbital_data_df, max_age_days=14, reference_time=None):
    """
    Objects whose TLE epoch is more than max_age_days old relative to the
    propagation window start. Older TLEs mean larger real-world position
    uncertainty than fresher ones — worth flagging, not necessarily excluding.
    """
    ages = compute_epoch_age_days(orbital_data_df, reference_time)
    stale = orbital_data_df[ages > max_age_days].copy()
    stale["EPOCH_AGE_DAYS"] = ages[ages > max_age_days]
    return stale[["OBJECT_NAME", "NORAD_CAT_ID", "EPOCH", "EPOCH_AGE_DAYS"]]


def summary_report(orbital_data_df, reference_time=None):
    ages = compute_epoch_age_days(orbital_data_df, reference_time)

    return {
        "object_count": len(orbital_data_df),
        "mean_motion_min": float(orbital_data_df["MEAN_MOTION"].min()),
        "mean_motion_max": float(orbital_data_df["MEAN_MOTION"].max()),
        "eccentricity_min": float(orbital_data_df["ECCENTRICITY"].min()),
        "eccentricity_max": float(orbital_data_df["ECCENTRICITY"].max()),
        "inclination_min": float(orbital_data_df["INCLINATION"].min()),
        "inclination_max": float(orbital_data_df["INCLINATION"].max()),
        "epoch_age_days_min": float(ages.min()),
        "epoch_age_days_max": float(ages.max()),
        "epoch_age_days_mean": float(ages.mean()),
        "debris_count": int(orbital_data_df["OBJECT_NAME"].str.upper().str.contains("DEB").sum()),
        "satellite_count": int((~orbital_data_df["OBJECT_NAME"].str.upper().str.contains("DEB")).sum()),
    }


def run_and_save():
    from modules import data_loader

    config.ensure_dirs()

    orbital_data = data_loader.load_orbital_data()

    report = summary_report(orbital_data)
    stale = flag_stale_tles(orbital_data)

    print("=== Data Quality Summary ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    if len(stale) > 0:
        print(f"\n{len(stale)} object(s) with TLE epoch >14 days before propagation window:")
        print(stale.to_string(index=False))
    else:
        print("\nNo stale TLEs flagged (all within 14 days of propagation window start).")

    import json
    output_path = config.RESULTS_DIR / "data_quality_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    stale_path = config.RESULTS_DIR / "stale_tles.csv"
    stale.to_csv(stale_path, index=False)

    print(f"\nResults written: {output_path}, {stale_path}")

    return report, stale


if __name__ == "__main__":
    run_and_save()