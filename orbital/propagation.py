"""
orbital/propagation.py

Two propagation methods, matching the pipeline described in the ORION
design doc ("TLE -> orbital state -> propagate orbit"):

1. TwoBodyPropagator
   Analytical Keplerian propagation: solves Kepler's equation for
   eccentric anomaly at time t, converts to true anomaly, and rotates
   into ECI. Good for understanding/deriving orbital mechanics and for
   scenarios where you define an orbit from scratch (a, e, i, RAAN, argp).

2. TLEPropagator
   Wraps the `sgp4` library (the same algorithm used operationally for
   real TLEs from Space-Track / Celestrak) to propagate a real
   satellite's Two-Line Element set. This is what you'd use to model
   an actual CubeSat's orbit realistically, including perturbations
   SGP4 accounts for (drag, J2, etc.) that a pure two-body model ignores.

Both propagators expose a common interface:
    propagate(dt: datetime) -> (r_eci [km], v_eci [km/s])
    propagate_range(start, end, step_s) -> list[(datetime, r_eci, v_eci)]

so downstream code (ground_track, spacecraft twin, mission planner)
doesn't need to care which one is in use.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from orbital.coordinates import (
    KeplerianElements, keplerian_to_eci, MU_EARTH,
)

try:
    from sgp4.api import Satrec, WGS84 as SGP4_WGS84, jday
    _HAVE_SGP4 = True
except ImportError:
    _HAVE_SGP4 = False


# ---------------------------------------------------------------------------
# Kepler's equation solver
# ---------------------------------------------------------------------------

def solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E,
    given mean anomaly M [rad] and eccentricity e, via Newton-Raphson.
    """
    M = M % (2 * np.pi)
    E = M if e < 0.8 else np.pi  # better starting guess for high e

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        dE = f / f_prime
        E -= dE
        if abs(dE) < tol:
            break
    return E


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    """Convert eccentric anomaly to true anomaly [rad]."""
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                           np.sqrt(1 - e) * np.cos(E / 2))


def true_to_mean_anomaly(nu: float, e: float) -> float:
    """Convert true anomaly to mean anomaly [rad] (inverse of the above)."""
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2),
                        np.sqrt(1 + e) * np.cos(nu / 2))
    M = E - e * np.sin(E)
    return M % (2 * np.pi)


# ---------------------------------------------------------------------------
# Two-body analytical propagator
# ---------------------------------------------------------------------------

class TwoBodyPropagator:
    """Analytical Keplerian (unperturbed two-body) orbit propagator."""

    def __init__(self, elements: KeplerianElements):
        self.elements = elements
        self.M0 = true_to_mean_anomaly(elements.nu0, elements.e)

    def true_anomaly_at(self, dt: datetime) -> float:
        dt_sec = (dt - self.elements.epoch).total_seconds()
        M = (self.M0 + self.elements.mean_motion * dt_sec) % (2 * np.pi)
        E = solve_kepler(M, self.elements.e)
        return eccentric_to_true_anomaly(E, self.elements.e)

    def propagate(self, dt: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Return (r_eci [km], v_eci [km/s]) at time dt."""
        nu = self.true_anomaly_at(dt)
        return keplerian_to_eci(self.elements, nu)

    def propagate_range(self, start: datetime, end: datetime, step_s: float):
        """Yield (datetime, r_eci, v_eci) samples from start to end."""
        t = start
        out = []
        while t <= end:
            r, v = self.propagate(t)
            out.append((t, r, v))
            t += timedelta(seconds=step_s)
        return out


# ---------------------------------------------------------------------------
# SGP4 / TLE propagator
# ---------------------------------------------------------------------------

class TLEPropagator:
    """
    Propagates a real satellite using its Two-Line Element (TLE) set via
    SGP4 — the standard algorithm behind Celestrak/Space-Track ephemeris.

    Example TLE (ISS, illustrative format only):
        line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9994"
        line2 = "2 25544  51.6416 339.9975 0007976  15.4143 344.7011 15.49560492    46"
    """

    def __init__(self, line1: str, line2: str, name: str = "SAT"):
        if not _HAVE_SGP4:
            raise ImportError("The 'sgp4' package is required for TLEPropagator "
                               "(pip install sgp4)")
        self.name = name
        self.line1 = line1
        self.line2 = line2
        self.satrec = Satrec.twoline2rv(line1, line2)

    def propagate(self, dt: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Return (r_eci [km], v_eci [km/s]) at time dt (TEME frame, treated as ECI)."""
        dt = dt.astimezone(timezone.utc)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                       dt.second + dt.microsecond * 1e-6)
        error_code, r, v = self.satrec.sgp4(jd, fr)
        if error_code != 0:
            raise RuntimeError(f"SGP4 propagation error code {error_code} at {dt}")
        return np.array(r), np.array(v)

    def propagate_range(self, start: datetime, end: datetime, step_s: float):
        t = start
        out = []
        while t <= end:
            r, v = self.propagate(t)
            out.append((t, r, v))
            t += timedelta(seconds=step_s)
        return out

    def to_keplerian_elements(self) -> KeplerianElements:
        """Approximate Keplerian elements from the TLE's mean elements (for display/comparison)."""
        sr = self.satrec
        a = (MU_EARTH / (sr.no_kozai / 60.0) ** 2) ** (1 / 3)  # no_kozai is rad/min
        return KeplerianElements(
            a=a, e=sr.ecco, i=sr.inclo, raan=sr.nodeo, argp=sr.argpo,
            nu0=eccentric_to_true_anomaly(solve_kepler(sr.mo, sr.ecco), sr.ecco),
            epoch=datetime(2000, 1, 1, tzinfo=timezone.utc)  # sgp4 uses its own epoch internally
        )
