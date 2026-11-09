"""
orbital/coordinates.py

Core constants and coordinate-frame transformations for ORION's orbital
mechanics layer.

Frames implemented:
    - Keplerian orbital elements  -> Perifocal frame (PQW)
    - Perifocal frame             -> Earth-Centered Inertial (ECI / J2000-ish)
    - ECI                         -> Earth-Centered Earth-Fixed (ECEF)
    - ECEF                        -> Geodetic (lat, lon, alt) using WGS84

These are the building blocks every other ORION layer (propagation,
ground tracks, access windows, spacecraft twin) depends on, so they are
kept dependency-light (numpy only) and unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MU_EARTH = 398600.4418          # km^3 / s^2  (standard gravitational parameter)
R_EARTH_EQ = 6378.137            # km  (WGS84 equatorial radius)
R_EARTH_POLAR = 6356.752314245   # km  (WGS84 polar radius)
F_EARTH = 1.0 / 298.257223563    # WGS84 flattening
OMEGA_EARTH = 7.2921150e-5       # rad/s (Earth rotation rate, sidereal)
J2 = 1.08262668e-3               # Earth oblateness coefficient (for later use)


@dataclass
class KeplerianElements:
    """Classical (Keplerian) orbital elements."""
    a: float          # semi-major axis [km]
    e: float          # eccentricity [-]
    i: float          # inclination [rad]
    raan: float       # right ascension of ascending node [rad]
    argp: float       # argument of perigee [rad]
    nu0: float        # true anomaly at epoch [rad]
    epoch: datetime    # epoch time (UTC, timezone-aware)

    @property
    def period(self) -> float:
        """Orbital period [s] via Kepler's third law."""
        return 2.0 * np.pi * np.sqrt(self.a ** 3 / MU_EARTH)

    @property
    def mean_motion(self) -> float:
        """Mean motion n [rad/s]."""
        return np.sqrt(MU_EARTH / self.a ** 3)

    @property
    def altitude_perigee(self) -> float:
        return self.a * (1 - self.e) - R_EARTH_EQ

    @property
    def altitude_apogee(self) -> float:
        return self.a * (1 + self.e) - R_EARTH_EQ


# ---------------------------------------------------------------------------
# Keplerian -> Perifocal (PQW) -> ECI
# ---------------------------------------------------------------------------

def perifocal_state(a: float, e: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Position and velocity in the perifocal (PQW) frame for a given true
    anomaly nu, using the orbit's specific angular momentum.

    Returns (r_pqw [km], v_pqw [km/s]).
    """
    p = a * (1 - e ** 2)                      # semi-latus rectum
    r_mag = p / (1 + e * np.cos(nu))
    h = np.sqrt(MU_EARTH * p)                 # specific angular momentum

    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = (MU_EARTH / h) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    return r_pqw, v_pqw


def _rotation_313(raan: float, i: float, argp: float) -> np.ndarray:
    """
    Classical 3-1-3 Euler rotation matrix (R3(-RAAN) R1(-i) R3(-argp))
    that maps perifocal -> ECI.
    """
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(argp), np.sin(argp)

    R = np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i,
         -cos_O * sin_w - sin_O * cos_w * cos_i,
         sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i,
         -sin_O * sin_w + cos_O * cos_w * cos_i,
         -cos_O * sin_i],
        [sin_w * sin_i,
         cos_w * sin_i,
         cos_i],
    ])
    return R


def keplerian_to_eci(elements: KeplerianElements, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian elements + true anomaly to ECI position/velocity.

    Returns (r_eci [km], v_eci [km/s]).
    """
    r_pqw, v_pqw = perifocal_state(elements.a, elements.e, nu)
    R = _rotation_313(elements.raan, elements.i, elements.argp)
    return R @ r_pqw, R @ v_pqw


def eci_to_keplerian(r_eci: np.ndarray, v_eci: np.ndarray) -> KeplerianElements:
    """
    Convert an ECI state vector back to classical orbital elements.
    Useful for validating propagation and for ingesting arbitrary
    initial states (e.g. from an external ephemeris).
    """
    r = np.asarray(r_eci, dtype=float)
    v = np.asarray(v_eci, dtype=float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    n_vec = np.cross([0, 0, 1], h_vec)
    n_mag = np.linalg.norm(n_vec)

    e_vec = (1 / MU_EARTH) * ((v_mag ** 2 - MU_EARTH / r_mag) * r - np.dot(r, v) * v)
    e = np.linalg.norm(e_vec)

    energy = v_mag ** 2 / 2 - MU_EARTH / r_mag
    a = -MU_EARTH / (2 * energy)

    i = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))

    raan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0)) if n_mag > 1e-12 else 0.0
    if n_mag > 1e-12 and n_vec[1] < 0:
        raan = 2 * np.pi - raan

    if n_mag > 1e-12 and e > 1e-8:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0

    if e > 1e-8:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0

    return KeplerianElements(a=a, e=e, i=i, raan=raan, argp=argp, nu0=nu,
                              epoch=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Time: Julian Date, GMST
# ---------------------------------------------------------------------------

def julian_date(dt: datetime) -> float:
    """Julian Date for a timezone-aware UTC datetime."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    dt = dt.astimezone(timezone.utc)

    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return jd


def gmst_rad(dt: datetime) -> float:
    """
    Greenwich Mean Sidereal Time [rad] at datetime dt, via the standard
    IAU 1982 polynomial in Julian centuries since J2000.0.
    """
    jd = julian_date(dt)
    T = (jd - 2451545.0) / 36525.0
    gmst_sec = (67310.54841
                + (876600 * 3600 + 8640184.812866) * T
                + 0.093104 * T ** 2
                - 6.2e-6 * T ** 3)
    gmst_deg = (gmst_sec % 86400.0) / 240.0   # 86400 s = 360 deg -> 240 s/deg
    return np.deg2rad(gmst_deg % 360.0)


# ---------------------------------------------------------------------------
# ECI <-> ECEF <-> Geodetic
# ---------------------------------------------------------------------------

def eci_to_ecef(r_eci: np.ndarray, dt: datetime) -> np.ndarray:
    """Rotate an ECI position vector into ECEF using GMST at dt."""
    theta = gmst_rad(dt)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return R @ np.asarray(r_eci, dtype=float)


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """
    Convert ECEF position [km] to geodetic (lat [rad], lon [rad], alt [km])
    using the WGS84 ellipsoid (Bowring's iterative method).
    """
    x, y, z = r_ecef
    a = R_EARTH_EQ
    f = F_EARTH
    b = a * (1 - f)
    e2 = 1 - (b ** 2) / (a ** 2)

    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)

    lat = np.arctan2(z, p * (1 - e2))  # initial guess
    for _ in range(6):                  # converges in ~3-4 iterations
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat ** 2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z + N * e2 * sin_lat, p)

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat ** 2)
    alt = p / np.cos(lat) - N

    return lat, lon, alt


def eci_to_geodetic(r_eci: np.ndarray, dt: datetime) -> tuple[float, float, float]:
    """Convenience: ECI -> ECEF -> geodetic in one call."""
    r_ecef = eci_to_ecef(r_eci, dt)
    return ecef_to_geodetic(r_ecef)
