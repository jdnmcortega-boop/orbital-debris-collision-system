"""
Converts a row of our orbital-elements CSV (CelesTrak OMM-style columns)
into standard NORAD Two-Line Element (TLE) text, including checksums.

Needed to feed the browser-side live orbit view: satellite.js (the
standard JS SGP4 library used for real-time client-side propagation,
same approach CelesTrak/n2yo use for smooth animation) expects TLE text
via its documented twoline2satrec() API, not our named-column format.
"""

from datetime import datetime
import re


def tle_checksum(line):
    """Standard TLE checksum: sum of all digits, '-' counts as 1, mod 10."""
    total = 0
    for ch in line:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def format_exp(value):
    """
    Format a float into TLE's 8-character assumed-decimal-point exponential
    notation, e.g. -0.000011606 -> '-11606-4' (meaning -0.11606 x 10^-4).
    """
    if value == 0:
        return " 00000-0"

    sign = "-" if value < 0 else " "
    value = abs(value)

    exponent = 0
    while value >= 1:
        value /= 10.0
        exponent += 1
    while value < 0.1 and value > 0:
        value *= 10.0
        exponent -= 1

    mantissa = round(value * 100000)
    if mantissa >= 100000:  # rounding pushed it up a digit
        mantissa //= 10
        exponent += 1

    exp_sign = "-" if exponent < 0 else "+"
    return f"{sign}{mantissa:05d}{exp_sign}{abs(exponent):1d}"


def epoch_to_tle_format(epoch_str):
    """Convert an ISO datetime string to TLE epoch format: YYDDD.DDDDDDDD."""
    dt = datetime.fromisoformat(epoch_str.replace("Z", ""))
    year_2digit = dt.year % 100

    day_of_year = dt.timetuple().tm_yday
    fraction_of_day = (dt.hour * 3600 + dt.minute * 60 + dt.second
                        + dt.microsecond / 1_000_000) / 86400.0

    return f"{year_2digit:02d}{day_of_year:03d}{fraction_of_day:.8f}".replace("0.", ".", 1) \
        if False else f"{year_2digit:02d}{(day_of_year + fraction_of_day):012.8f}"


def international_designator(object_id):
    """
    Parse OBJECT_ID (e.g. '1997-051C') into TLE's international designator
    fields: 2-digit year, 3-digit launch number, up to 3-char piece.
    """
    match = re.match(r"(\d{4})-(\d{3})([A-Za-z]*)", object_id)
    if not match:
        return "00000A  "  # fallback for malformed/missing IDs

    year, launch_num, piece = match.groups()
    year_2digit = int(year) % 100
    piece = (piece or "A").ljust(3)[:3]
    return f"{year_2digit:02d}{launch_num:>03s}{piece}"


def row_to_tle(row):
    """
    Build TLE line 1 and line 2 from one row of the orbital-elements CSV
    (a pandas Series or dict with the standard CelesTrak-format columns).
    """
    norad_id = int(row["NORAD_CAT_ID"])
    classification = row.get("CLASSIFICATION_TYPE", "U")
    intl_designator = international_designator(str(row["OBJECT_ID"]))
    epoch_field = epoch_to_tle_format(str(row["EPOCH"]))

    mean_motion_dot = float(row["MEAN_MOTION_DOT"]) / 2.0
    mmd_sign = "-" if mean_motion_dot < 0 else " "
    mmd_field = f"{mmd_sign}{abs(mean_motion_dot):.8f}".replace("0.", ".", 1)[:10].ljust(10)

    mean_motion_ddot_field = format_exp(float(row["MEAN_MOTION_DDOT"]) / 6.0)
    bstar_field = format_exp(float(row["BSTAR"]))

    ephemeris_type = int(row.get("EPHEMERIS_TYPE", 0))
    element_set_no = int(row.get("ELEMENT_SET_NO", 999))

    line1_body = (
        f"1 {norad_id:05d}{classification} {intl_designator} "
        f"{epoch_field} {mmd_field} {mean_motion_ddot_field} {bstar_field} "
        f"{ephemeris_type} {element_set_no:4d}"
    )
    line1 = line1_body + str(tle_checksum(line1_body))

    inclination = float(row["INCLINATION"])
    raan = float(row["RA_OF_ASC_NODE"])
    eccentricity = float(row["ECCENTRICITY"])
    ecc_field = f"{eccentricity:.7f}".split(".")[1]  # 7 digits, no leading "0."
    arg_perigee = float(row["ARG_OF_PERICENTER"])
    mean_anomaly = float(row["MEAN_ANOMALY"])
    mean_motion = float(row["MEAN_MOTION"])
    rev_at_epoch = int(row.get("REV_AT_EPOCH", 0))

    line2_body = (
        f"2 {norad_id:05d} {inclination:8.4f} {raan:8.4f} {ecc_field} "
        f"{arg_perigee:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}{rev_at_epoch:5d}"
    )
    line2 = line2_body + str(tle_checksum(line2_body))

    return line1, line2


def decode_tle_for_validation(line1, line2):
    """
    Parse our OWN generated TLE back into values, for round-trip testing.
    This is NOT a general-purpose TLE parser — just enough to validate
    row_to_tle()'s field positions are self-consistent.
    """
    return {
        "norad_id": int(line1[2:7]),
        "epoch_year": int(line1[18:20]),
        "epoch_day": float(line1[20:32]),
        "inclination": float(line2[8:16]),
        "raan": float(line2[17:25]),
        "eccentricity": float("0." + line2[26:33]),
        "arg_perigee": float(line2[34:42]),
        "mean_anomaly": float(line2[43:51]),
        "mean_motion": float(line2[52:63]),
        "checksum_linefrom modules.tle_formatter import row_to_tle1_valid": tle_checksum(line1[:-1]) == int(line1[-1]),
        "checksum_line2_valid": tle_checksum(line2[:-1]) == int(line2[-1]),
    }