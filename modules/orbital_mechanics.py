"""
Orbital mechanics utilities: converting SGP4-propagated positions into
geodetic latitude/longitude/altitude for map display.

SCOPE NOTE: SGP4 outputs position in the TEME frame (True Equator, Mean
Equinox) — a near-Earth-fixed approximation, not a precise ECI or ECEF
frame. This module rotates TEME coordinates by Greenwich Mean Sidereal
Time (GMST) to approximate an Earth-fixed frame, then converts to
lat/lon/altitude using a SPHERICAL Earth model (not WGS84 ellipsoid).
This introduces a small error (up to ~0.1-0.2 degrees of latitude) —
acceptable for visualization on a dashboard, not for precision geodesy.
State this simplification if you cite exact ground-track coordinates.
"""

import math
from datetime import datetime, timedelta, timezone


EARTH_RADIUS_KM = 6378.137  # WGS84 equatorial radius (same as reentry_risk.py)

# Fixed +8 hour offset (Philippine Standard Time doesn't observe DST, so no
# timezone database is needed). DISPLAY ONLY — never use this for SGP4/GMST
# calculations, which must stay in UTC.
UTC_PLUS_8 = timezone(timedelta(hours=8))


def to_utc8(dt):
    """Convert a UTC datetime to UTC+8, for display purposes only."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(UTC_PLUS_8)


def julian_date(dt):
    """Julian Date for a timezone-aware UTC datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    year, month = dt.year, dt.month
    day = dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0

    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + A // 4

    jd = (int(365.25 * (year + 4716)) + int(30.6001 * (month + 1))
          + day + B - 1524.5)
    return jd


def gmst_degrees(dt):
    """Greenwich Mean Sidereal Time, in degrees, via the standard Vallado formula."""
    jd = julian_date(dt)
    T = (jd - 2451545.0) / 36525.0

    gmst = (280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T**2
            - (T**3) / 38710000.0)

    return gmst % 360.0


def eci_to_geodetic(x_km, y_km, z_km, dt):
    """
    Convert a TEME/pseudo-ECI position (as output by SGP4) to geodetic
    latitude, longitude, and altitude, at the given UTC datetime.
    Uses a spherical Earth model — see module scope note above.
    """
    theta = math.radians(gmst_degrees(dt))

    # Rotate into an approximate Earth-fixed frame
    x_ecef = x_km * math.cos(theta) + y_km * math.sin(theta)
    y_ecef = -x_km * math.sin(theta) + y_km * math.cos(theta)
    z_ecef = z_km

    r = math.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)
    lat_deg = math.degrees(math.asin(z_ecef / r))
    lon_deg = math.degrees(math.atan2(y_ecef, x_ecef))
    lon_deg = ((lon_deg + 180) % 360) - 180  # wrap to [-180, 180]

    altitude_km = r - EARTH_RADIUS_KM

    return lat_deg, lon_deg, altitude_km


def velocity_magnitude_km_s(vx, vy, vz):
    return math.sqrt(vx**2 + vy**2 + vz**2)


def apogee_perigee_altitude_km(mean_motion_rev_per_day, eccentricity):
    """
    Apogee and perigee altitude from orbital elements (same Kepler's-third-
    law approach as reentry_risk.perigee_altitude_km, extended to apogee).
    """
    GM_EARTH_KM3_S2 = 398600.4418
    n_rad_s = float(mean_motion_rev_per_day) * 2.0 * math.pi / 86400.0
    semi_major_axis_km = (GM_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)

    perigee_km = semi_major_axis_km * (1.0 - float(eccentricity)) - EARTH_RADIUS_KM
    apogee_km = semi_major_axis_km * (1.0 + float(eccentricity)) - EARTH_RADIUS_KM

    return apogee_km, perigee_km