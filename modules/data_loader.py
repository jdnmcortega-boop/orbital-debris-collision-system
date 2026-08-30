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
    Load and validate the orbital elements CSV.

    Returns a pandas DataFrame. Raises ValueError if required
    columns are missing, or if any row fails basic sanity checks.
    """
    path = path or config.ORBITAL_DATA_FILE

    if not path.exists():
        raise FileNotFoundError(f"Orbital data file not found: {path}")

    data = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Basic sanity checks — catch obviously bad rows early rather than
    # letting sgp4init fail cryptically later.
    bad_rows = data[
        (data["ECCENTRICITY"] < 0) | (data["ECCENTRICITY"] >= 1)
        | (data["INCLINATION"] < 0) | (data["INCLINATION"] > 180)
        | (data["MEAN_MOTION"] <= 0)
    ]
    if len(bad_rows) > 0:
        names = bad_rows["OBJECT_NAME"].tolist()
        raise ValueError(f"Rows with out-of-range orbital elements: {names}")

    # Drop exact duplicate NORAD IDs, keeping the first occurrence
    before = len(data)
    data = data.drop_duplicates(subset="NORAD_CAT_ID", keep="first")
    if len(data) < before:
        print(f"[data_loader] Dropped {before - len(data)} duplicate NORAD_CAT_ID rows")

    return data.reset_index(drop=True)


if __name__ == "__main__":
    df = load_orbital_data()
    print(f"Loaded {len(df)} objects")
    print(df[["OBJECT_NAME", "NORAD_CAT_ID", "EPOCH"]].head())