"""Historical debris-to-rocket collision analysis.

This module keeps documented historical rocket-body collision events separate
from the live predictor. It calculates quantities that can be reproduced from
published event data and, when state vectors are available, derives relative
position and velocity directly.

Input CSV columns supported by the batch calculator:
    EVENT_ID,DATE_UTC,TARGET_NAME,TARGET_TYPE,TARGET_NORAD_ID,
    PROJECTILE_NAME,PROJECTILE_TYPE,PROJECTILE_NORAD_ID,ALTITUDE_KM,
    TARGET_MASS_KG,PROJECTILE_MASS_KG,RELATIVE_VELOCITY_KM_S,
    NOTES,SOURCE

The calculator does not invent missing historical measurements. Missing values
remain NaN and are reported as unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "EVENT_ID",
    "DATE_UTC",
    "TARGET_NAME",
    "TARGET_TYPE",
    "PROJECTILE_NAME",
    "PROJECTILE_TYPE",
]


def relative_state(
    target_position_km: Iterable[float],
    target_velocity_km_s: Iterable[float],
    projectile_position_km: Iterable[float],
    projectile_velocity_km_s: Iterable[float],
) -> dict:
    """Calculate relative position, miss distance and closing speed.

    Positions and velocities must use the same coordinate frame and epoch.
    """
    r_t = np.asarray(list(target_position_km), dtype=float)
    v_t = np.asarray(list(target_velocity_km_s), dtype=float)
    r_p = np.asarray(list(projectile_position_km), dtype=float)
    v_p = np.asarray(list(projectile_velocity_km_s), dtype=float)

    if any(x.shape != (3,) for x in (r_t, v_t, r_p, v_p)):
        raise ValueError("Each position/velocity vector must contain exactly 3 values.")

    dr = r_p - r_t
    dv = v_p - v_t
    distance_km = float(np.linalg.norm(dr))
    relative_speed_km_s = float(np.linalg.norm(dv))

    # Positive closing speed means the objects are moving toward one another.
    closing_speed_km_s = float(-np.dot(dr, dv) / distance_km) if distance_km > 0 else 0.0

    return {
        "RELATIVE_DX_KM": float(dr[0]),
        "RELATIVE_DY_KM": float(dr[1]),
        "RELATIVE_DZ_KM": float(dr[2]),
        "MISS_DISTANCE_KM": distance_km,
        "RELATIVE_VELOCITY_KM_S": relative_speed_km_s,
        "CLOSING_SPEED_KM_S": closing_speed_km_s,
    }


def collision_energy(projectile_mass_kg: float, relative_velocity_km_s: float) -> dict:
    """Calculate projectile kinetic energy and useful normalized measures.

    This is a simple physics calculation, not a fragmentation model. The
    projectile kinetic energy is 0.5*m*v^2. Fragment counts should not be
    inferred from this function alone.
    """
    mass = float(projectile_mass_kg)
    speed_m_s = float(relative_velocity_km_s) * 1000.0

    if mass < 0 or speed_m_s < 0:
        raise ValueError("Mass and speed must be non-negative.")

    energy_j = 0.5 * mass * speed_m_s**2
    return {
        "IMPACT_ENERGY_J": energy_j,
        "IMPACT_ENERGY_MJ": energy_j / 1e6,
        "IMPACT_ENERGY_GJ": energy_j / 1e9,
    }


def add_calculated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reproducible derived quantities to historical event rows."""
    out = df.copy()

    for column in REQUIRED_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    def energy_row(row):
        try:
            return pd.Series(
                collision_energy(
                    row["PROJECTILE_MASS_KG"],
                    row["RELATIVE_VELOCITY_KM_S"],
                )
            )
        except (TypeError, ValueError):
            return pd.Series({
                "IMPACT_ENERGY_J": np.nan,
                "IMPACT_ENERGY_MJ": np.nan,
                "IMPACT_ENERGY_GJ": np.nan,
            })

    energy = out.apply(energy_row, axis=1)
    for column in energy.columns:
        out[column] = energy[column]

    # Energy per gram of the target is a useful comparison metric for the
    # documented events, but it is not a substitute for a fragmentation model.
    target_mass_g = pd.to_numeric(out.get("TARGET_MASS_KG"), errors="coerce") * 1000.0
    out["ENERGY_PER_TARGET_MASS_J_PER_G"] = np.where(
        target_mass_g > 0,
        out["IMPACT_ENERGY_J"] / target_mass_g,
        np.nan,
    )

    out["DATE_UTC"] = pd.to_datetime(out["DATE_UTC"], utc=True, errors="coerce")
    for column in [
        "ALTITUDE_KM",
        "TARGET_MASS_KG",
        "PROJECTILE_MASS_KG",
        "RELATIVE_VELOCITY_KM_S",
        "IMPACT_ENERGY_MJ",
        "IMPACT_ENERGY_GJ",
        "ENERGY_PER_TARGET_MASS_J_PER_G",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def load_events(path: str | Path) -> pd.DataFrame:
    """Load the historical collision-event CSV and calculate derived fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Historical collision file not found: {path}")
    return add_calculated_columns(pd.read_csv(path))


def save_calculated_events(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Calculate and save the historical event table."""
    calculated = add_calculated_columns(df)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    calculated.to_csv(output, index=False)
    return calculated


def summarize_events(df: pd.DataFrame) -> dict:
    """Return compact research metrics for the historical rocket-body set."""
    calculated = add_calculated_columns(df)
    return {
        "events": int(len(calculated)),
        "rocket_body_targets": int(
            calculated["TARGET_TYPE"].astype(str).str.contains("rocket", case=False, na=False).sum()
        ),
        "known_altitude_events": int(calculated["ALTITUDE_KM"].notna().sum()),
        "known_relative_velocity_events": int(calculated["RELATIVE_VELOCITY_KM_S"].notna().sum()),
        "known_energy_events": int(calculated["IMPACT_ENERGY_MJ"].notna().sum()),
        "max_relative_velocity_km_s": (
            float(calculated["RELATIVE_VELOCITY_KM_S"].max())
            if calculated["RELATIVE_VELOCITY_KM_S"].notna().any()
            else None
        ),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "historical_rocket_debris_collisions.csv"
    output_path = project_root / "results" / "historical_rocket_debris_calculated.csv"

    events = load_events(input_path)
    save_calculated_events(events, output_path)

    print("Historical debris-to-rocket collision analysis")
    print("=" * 52)
    for key, value in summarize_events(events).items():
        print(f"{key}: {value}")
    print(f"Saved: {output_path}")
