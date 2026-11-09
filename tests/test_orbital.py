"""
tests/test_orbital.py

Sanity/regression tests for orbital/. Run with:
    python -m pytest tests/test_orbital.py -v
"""

import numpy as np
from datetime import datetime, timedelta, timezone
import pytest

from orbital.coordinates import (
    KeplerianElements, MU_EARTH, R_EARTH_EQ,
    keplerian_to_eci, eci_to_keplerian, ecef_to_geodetic, eci_to_ecef,
    julian_date, gmst_rad,
)
from orbital.propagation import (
    TwoBodyPropagator, solve_kepler, eccentric_to_true_anomaly, true_to_mean_anomaly,
)
from orbital.ground_track import GroundStation, elevation_angle, is_eclipse, sun_vector_eci


EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def circular_orbit(alt_km=500.0, i_deg=51.6, raan_deg=0.0):
    return KeplerianElements(
        a=R_EARTH_EQ + alt_km, e=0.0, i=np.deg2rad(i_deg),
        raan=np.deg2rad(raan_deg), argp=0.0, nu0=0.0, epoch=EPOCH,
    )


class TestKeplerRoundTrip:
    def test_true_mean_anomaly_roundtrip(self):
        for nu in np.linspace(0, 2 * np.pi, 20, endpoint=False):
            for e in [0.0, 0.1, 0.5, 0.8]:
                M = true_to_mean_anomaly(nu, e)
                E = solve_kepler(M, e)
                nu_back = eccentric_to_true_anomaly(E, e)
                assert np.isclose(np.sin(nu), np.sin(nu_back), atol=1e-6)
                assert np.isclose(np.cos(nu), np.cos(nu_back), atol=1e-6)

    def test_eci_keplerian_roundtrip(self):
        elements = KeplerianElements(
            a=7000.0, e=0.01, i=np.deg2rad(45.0),
            raan=np.deg2rad(30.0), argp=np.deg2rad(60.0), nu0=np.deg2rad(90.0),
            epoch=EPOCH,
        )
        r, v = keplerian_to_eci(elements, elements.nu0)
        recovered = eci_to_keplerian(r, v)
        assert np.isclose(recovered.a, elements.a, rtol=1e-6)
        assert np.isclose(recovered.e, elements.e, atol=1e-6)
        assert np.isclose(recovered.i, elements.i, atol=1e-6)


class TestPropagation:
    def test_period_matches_keplers_third_law(self):
        elements = circular_orbit(alt_km=500.0)
        expected_period = 2 * np.pi * np.sqrt(elements.a ** 3 / MU_EARTH)
        assert np.isclose(elements.period, expected_period)
        # ISS-ish 500km orbit should have ~94-95 min period
        assert 90 * 60 < elements.period < 100 * 60

    def test_full_orbit_returns_to_start(self):
        elements = circular_orbit(alt_km=500.0)
        prop = TwoBodyPropagator(elements)
        r0, v0 = prop.propagate(EPOCH)
        r1, v1 = prop.propagate(EPOCH + timedelta(seconds=elements.period))
        assert np.allclose(r0, r1, atol=1e-3)
        assert np.allclose(v0, v1, atol=1e-6)

    def test_circular_orbit_constant_radius(self):
        elements = circular_orbit(alt_km=500.0)
        prop = TwoBodyPropagator(elements)
        samples = prop.propagate_range(EPOCH, EPOCH + timedelta(seconds=elements.period), 60)
        radii = [np.linalg.norm(r) for _, r, _ in samples]
        assert np.std(radii) < 1e-6  # circular orbit -> constant |r|

    def test_vis_viva_energy_conservation(self):
        """Specific orbital energy should be constant throughout the orbit."""
        elements = KeplerianElements(a=7000, e=0.1, i=0.5, raan=0.2, argp=0.3,
                                      nu0=0.0, epoch=EPOCH)
        prop = TwoBodyPropagator(elements)
        samples = prop.propagate_range(EPOCH, EPOCH + timedelta(seconds=elements.period), 100)
        energies = [np.linalg.norm(v)**2/2 - MU_EARTH/np.linalg.norm(r) for _, r, v in samples]
        assert np.std(energies) < 1e-6


class TestCoordinates:
    def test_geodetic_roundtrip_equator(self):
        """A point on the equator at the WGS84 equatorial radius should map to lat=0."""
        r_ecef = np.array([R_EARTH_EQ, 0.0, 0.0])
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        assert np.isclose(lat, 0.0, atol=1e-6)
        assert np.isclose(lon, 0.0, atol=1e-6)
        assert np.isclose(alt, 0.0, atol=1e-3)

    def test_geodetic_roundtrip_pole(self):
        r_ecef = np.array([0.0, 0.0, R_EARTH_EQ * (1 - 1/298.257223563)])
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        assert np.isclose(np.rad2deg(lat), 90.0, atol=0.1)

    def test_gmst_increases_with_time(self):
        g1 = gmst_rad(EPOCH)
        g2 = gmst_rad(EPOCH + timedelta(hours=1))
        # GMST should advance by ~ Earth's rotation rate * 1hr (mod 2pi)
        diff = (g2 - g1) % (2 * np.pi)
        expected = 7.2921150e-5 * 3600
        assert np.isclose(diff, expected, atol=1e-3)


class TestGroundStationVisibility:
    def test_satellite_directly_overhead_has_90deg_elevation(self):
        station = GroundStation(name="test", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=10)
        sat_ecef = np.array([R_EARTH_EQ + 500.0, 0.0, 0.0])  # directly above station
        el = elevation_angle(sat_ecef, station)
        assert np.isclose(el, 90.0, atol=1e-3)

    def test_satellite_on_opposite_side_of_earth_is_below_horizon(self):
        station = GroundStation(name="test", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=10)
        sat_ecef = np.array([-(R_EARTH_EQ + 500.0), 0.0, 0.0])
        el = elevation_angle(sat_ecef, station)
        assert el < 0


class TestEclipse:
    def test_sunward_side_not_eclipsed(self):
        sun_hat = np.array([1.0, 0.0, 0.0])
        r_sunward = np.array([R_EARTH_EQ + 500, 0, 0])
        assert not is_eclipse(r_sunward, sun_hat)

    def test_antisunward_side_eclipsed(self):
        sun_hat = np.array([1.0, 0.0, 0.0])
        r_antisunward = np.array([-(R_EARTH_EQ + 500), 0, 0])
        assert is_eclipse(r_antisunward, sun_hat)

    def test_sun_vector_is_unit_length(self):
        v = sun_vector_eci(EPOCH)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
