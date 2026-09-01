import pandas as pd

import config


REQUIRED_COLUMNS = [
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "NORAD_CAT_ID",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
]


def load_orbital_data(path=None):
    """
    Load and validate the orbital-elements CSV.

    This function preserves the original behavior: one latest input
    record per NORAD ID is returned. Forecast-aware selection is handled
    by load_forecast_orbital_data().
    """
    path = path or config.ORBITAL_DATA_FILE

    if not path.exists():
        raise FileNotFoundError(f"Orbital data file not found: {path}")

    data = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["EPOCH"] = pd.to_datetime(data["EPOCH"], utc=True, errors="coerce")
    if data["EPOCH"].isna().any():
        bad = data.loc[data["EPOCH"].isna(), "OBJECT_NAME"].tolist()
        raise ValueError(f"Rows with invalid EPOCH values: {bad}")

    bad_rows = data[
        (data["ECCENTRICITY"] < 0) | (data["ECCENTRICITY"] >= 1)
        | (data["INCLINATION"] < 0) | (data["INCLINATION"] > 180)
        | (data["MEAN_MOTION"] <= 0)
    ]
    if len(bad_rows) > 0:
        names = bad_rows["OBJECT_NAME"].tolist()
        raise ValueError(f"Rows with out-of-range orbital elements: {names}")

    return data.sort_values("EPOCH").reset_index(drop=True)


def load_forecast_orbital_data(path=None, cutoff=None):
    """
    Select the orbital state that would have been available at the
    forecast start.

    LIVE mode:
        Select the latest record for each NORAD ID from the dataset.

    BACKTEST mode:
        Discard every record after the historical cutoff, then select
        the latest record at or before the cutoff for each NORAD ID.

    This is the key leakage-prevention step for historical forecasting:
    information published after the forecast cutoff cannot be used to
    make the forecast.
    """
    data = load_orbital_data(path)

    if cutoff is None:
        cutoff = config.GRID_START

    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    if config.FORECAST_MODE == "backtest":
        eligible = data[data["EPOCH"] <= cutoff].copy()

        if eligible.empty:
            raise ValueError(
                f"No orbital records are available on or before the backtest "
                f"cutoff {cutoff.isoformat()}."
            )

        selected = (
            eligible.sort_values("EPOCH")
            .groupby("NORAD_CAT_ID", as_index=False)
            .tail(1)
        )

        print(
            f"[BACKTEST] Cutoff: {cutoff.isoformat()} | "
            f"Objects available: {selected['NORAD_CAT_ID'].nunique()}"
        )
        print("[BACKTEST] Future orbital records were excluded from the forecast input.")
    else:
        selected = (
            data.sort_values("EPOCH")
            .groupby("NORAD_CAT_ID", as_index=False)
            .tail(1)
        )

        print(
            f"[LIVE] Forecast start: {cutoff.isoformat()} | "
            f"Objects: {selected['NORAD_CAT_ID'].nunique()}"
        )

    return selected.sort_values("NORAD_CAT_ID").reset_index(drop=True)


if __name__ == "__main__":
    df = load_forecast_orbital_data()
    print(f"Loaded {len(df)} forecast input objects")
    print(df[["OBJECT_NAME", "NORAD_CAT_ID", "EPOCH"]].head())
