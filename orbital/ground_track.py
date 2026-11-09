"""
orbital/ground_track.py

Computes:
  1. Ground tracks: lat/lon subsatellite points over time.
  2. Access windows: when a satellite is visible (above a minimum
     elevation angle) from a fixed ground station, e.g. a ground-station
     antenna for downlink, or an imaging target's window.

Access windows are exactly what mission/planner.py will consume to
decide "when can I communicate" and "when can I image target X".
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

from orbital.coordinates import (
    eci_to_geodetic, eci_to_ecef, R_EARTH_EQ,
)


@dataclass
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elevation_deg: float = 10.0  # typical operational elevation mask

    @property
    def ecef(self) -> np.ndarray:
        """Ground station position in ECEF [km] (spherical-Earth approximation)."""
        lat = np.deg2rad(self.lat_deg)
        lon = np.deg2rad(self.lon_deg)
        r = R_EARTH_EQ + self.alt_km
        return r * np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])


@dataclass
class AccessWindow:
    station: str
    aos: datetime      # Acquisition of signal (rise)
    los: datetime       # Loss of signal (set)
    max_elevation_deg: float

    @property
    def duration_s(self) -> float:
        return (self.los - self.aos).total_seconds()

    def __repr__(self):
        return (f"AccessWindow({self.station}: {self.aos.isoformat()} -> "
                f"{self.los.isoformat()}, {self.duration_s:.0f}s, "
                f"max_el={self.max_elevation_deg:.1f} deg)")


def ground_track(samples: list[tuple[datetime, np.ndarray, np.ndarray]]):
    """
    Given propagator samples [(dt, r_eci, v_eci), ...], return a list of
    (dt, lat_deg, lon_deg, alt_km) subsatellite points.
    """
    track = []
    for dt, r_eci, _v in samples:
        lat, lon, alt = eci_to_geodetic(r_eci, dt)
        track.append((dt, np.rad2deg(lat), np.rad2deg(lon), alt))
    return track


def elevation_angle(sat_ecef: np.ndarray, station: GroundStation) -> float:
    """
    Elevation angle [deg] of the satellite as seen from a ground station,
    via the station's local East-North-Up (ENU) frame.
    """
    station_ecef = station.ecef
    rho = sat_ecef - station_ecef   # line-of-sight vector, ECEF

    lat = np.deg2rad(station.lat_deg)
    lon = np.deg2rad(station.lon_deg)

    # ECEF -> ENU rotation
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])
    enu = R @ rho
    east, north, up = enu
    horiz_dist = np.sqrt(east ** 2 + north ** 2)
    el = np.arctan2(up, horiz_dist)
    return np.rad2deg(el)


def compute_access_windows(samples: list[tuple[datetime, np.ndarray, np.ndarray]],
                            station: GroundStation) -> list[AccessWindow]:
    """
    Scan propagated ECI samples and return AccessWindows where the
    satellite is above `station.min_elevation_deg`.

    Note: samples should be reasonably fine-grained (e.g. 10-30s steps)
    for accurate AOS/LOS timing; this does simple threshold crossing
    without sub-sample interpolation.
    """
    windows = []
    in_view = False
    current_start = None
    current_max_el = -90.0

    for dt, r_eci, _v in samples:
        sat_ecef = eci_to_ecef(r_eci, dt)
        el = elevation_angle(sat_ecef, station)

        if el >= station.min_elevation_deg:
            if not in_view:
                in_view = True
                current_start = dt
                current_max_el = el
            else:
                current_max_el = max(current_max_el, el)
        else:
            if in_view:
                windows.append(AccessWindow(
                    station=station.name, aos=current_start, los=dt,
                    max_elevation_deg=current_max_el))
                in_view = False

    if in_view:  # still in view at the end of the sample range
        windows.append(AccessWindow(
            station=station.name, aos=current_start, los=samples[-1][0],
            max_elevation_deg=current_max_el))

    return windows


def is_eclipse(r_eci: np.ndarray, sun_vec_eci: np.ndarray) -> bool:
    """
    Simple cylindrical shadow model: True if the satellite is in Earth's
    shadow (eclipse), given the Sun direction unit vector in ECI.

    This is intentionally simple (no penumbra) — good enough for mission
    planning power-budget decisions; can be refined to a conical model later.
    """
    r = np.asarray(r_eci, dtype=float)
    sun_hat = np.asarray(sun_vec_eci, dtype=float)
    sun_hat = sun_hat / np.linalg.norm(sun_hat)

    proj = np.dot(r, sun_hat)
    if proj > 0:
        return False  # satellite is on the sunward side, can't be in cylindrical shadow

    perp = r - proj * sun_hat
    return np.linalg.norm(perp) < R_EARTH_EQ


def sun_vector_eci(dt: datetime) -> np.ndarray:
    """
    Low-precision Sun position unit vector in ECI, good to about 0.01 deg,
    from the standard low-precision solar ephemeris (Astronomical Almanac
    algorithm). Sufficient for eclipse/sunlight determination.
    """
    from orbital.coordinates import julian_date
    jd = julian_date(dt)
    n = jd - 2451545.0  # days since J2000.0

    L = np.deg2rad((280.460 + 0.9856474 * n) % 360)       # mean longitude
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360)        # mean anomaly
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 0.0000004 * n)                # obliquity

    x = np.cos(lam)
    y = np.cos(eps) * np.sin(lam)
    z = np.sin(eps) * np.sin(lam)
    return np.array([x, y, z])
