"""
experiments/demo_orbital_layer.py

End-to-end demonstration of ORION's Layer 1 (Orbital Mechanics):

    Keplerian elements --or-- TLE
        -> propagate over time
        -> ground track
        -> access windows to a ground station
        -> eclipse/sunlight determination

Run:
    python -m experiments.demo_orbital_layer
"""

from datetime import datetime, timedelta, timezone
import numpy as np

from orbital.coordinates import KeplerianElements, MU_EARTH, R_EARTH_EQ
from orbital.propagation import TwoBodyPropagator, TLEPropagator, eccentric_to_true_anomaly, solve_kepler
from orbital.ground_track import (
    GroundStation, ground_track, compute_access_windows, is_eclipse, sun_vector_eci,
)


def demo_two_body():
    print("=" * 70)
    print("PART 1 — Two-body analytical propagation (a synthetic 523 km CubeSat orbit)")
    print("=" * 70)

    epoch = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
    alt_km = 523.0
    a = R_EARTH_EQ + alt_km

    elements = KeplerianElements(
        a=a, e=0.001, i=np.deg2rad(97.5),   # sun-synchronous-ish inclination
        raan=np.deg2rad(45.0), argp=np.deg2rad(0.0), nu0=0.0, epoch=epoch,
    )

    print(f"Semi-major axis:     {elements.a:.2f} km")
    print(f"Perigee altitude:    {elements.altitude_perigee:.2f} km")
    print(f"Apogee altitude:     {elements.altitude_apogee:.2f} km")
    print(f"Orbital period:      {elements.period / 60:.2f} min")
    print(f"Mean motion:         {elements.mean_motion * 86400 / (2*np.pi):.3f} rev/day")

    prop = TwoBodyPropagator(elements)

    # Propagate one full orbit at 30s steps
    end = epoch + timedelta(seconds=elements.period)
    samples = prop.propagate_range(epoch, end, step_s=30)
    print(f"\nPropagated {len(samples)} samples over one orbit ({elements.period/60:.1f} min)")

    r0, v0 = samples[0][1], samples[0][2]
    r_mid, v_mid = samples[len(samples)//2][1], samples[len(samples)//2][2]
    print(f"r at t=0:       {np.round(r0, 1)} km   |r|={np.linalg.norm(r0):.1f} km")
    print(f"r at t=T/2:     {np.round(r_mid, 1)} km   |r|={np.linalg.norm(r_mid):.1f} km")

    # sanity check: speed should be higher near perigee, lower near apogee (vis-viva)
    print(f"speed at t=0:   {np.linalg.norm(v0):.3f} km/s")
    print(f"speed at t=T/2: {np.linalg.norm(v_mid):.3f} km/s")

    return samples, elements


def demo_ground_track_and_access(samples):
    print("\n" + "=" * 70)
    print("PART 2 — Ground track + access windows to a ground station")
    print("=" * 70)

    track = ground_track(samples)
    print(f"First 3 ground track points (dt, lat, lon, alt):")
    for dt, lat, lon, alt in track[:3]:
        print(f"  {dt.isoformat()}  lat={lat:7.2f}  lon={lon:7.2f}  alt={alt:7.1f} km")

    # A ground station: e.g. Bengaluru
    station = GroundStation(name="Bengaluru-GS", lat_deg=12.97, lon_deg=77.59,
                             alt_km=0.9, min_elevation_deg=10.0)

    windows = compute_access_windows(samples, station)
    print(f"\nAccess windows to {station.name} (min elevation {station.min_elevation_deg} deg):")
    if not windows:
        print("  None in this orbit pass window (try a longer propagation range).")
    for w in windows:
        print(f"  {w}")

    return station, windows


def demo_eclipse(samples, epoch):
    print("\n" + "=" * 70)
    print("PART 3 — Eclipse / sunlight determination")
    print("=" * 70)

    eclipse_count = 0
    transitions = []
    prev_state = None
    for dt, r_eci, _v in samples:
        sun_hat = sun_vector_eci(dt)
        eclipsed = is_eclipse(r_eci, sun_hat)
        if eclipsed:
            eclipse_count += 1
        if prev_state is not None and eclipsed != prev_state:
            transitions.append((dt, "-> ECLIPSE" if eclipsed else "-> SUNLIGHT"))
        prev_state = eclipsed

    frac = eclipse_count / len(samples)
    print(f"Fraction of orbit in eclipse: {frac*100:.1f}%")
    print("Transitions:")
    for dt, label in transitions:
        print(f"  {dt.isoformat()}  {label}")


def demo_tle_propagation():
    print("\n" + "=" * 70)
    print("PART 4 — TLE/SGP4 propagation (illustrative ISS-like TLE)")
    print("=" * 70)
    # NOTE: illustrative TLE for demonstration — for real work, fetch a
    # current TLE from Celestrak/Space-Track for your actual target satellite.
    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9994"
    line2 = "2 25544  51.6416 339.9975 0007976  15.4143 344.7011 15.49560492    46"

    try:
        prop = TLEPropagator(line1, line2, name="ISS-ILLUSTRATIVE")
    except ImportError as exc:
        print(f"  Skipped: {exc}")
        return

    dt = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    r, v = prop.propagate(dt)
    print(f"At {dt.isoformat()}:")
    print(f"  r_eci = {np.round(r, 2)} km   |r| = {np.linalg.norm(r):.1f} km")
    print(f"  v_eci = {np.round(v, 3)} km/s  |v| = {np.linalg.norm(v):.3f} km/s")
    print("  (This is exactly the SGP4 pipeline: TLE -> orbital state -> propagate.)")


if __name__ == "__main__":
    samples, elements = demo_two_body()
    demo_ground_track_and_access(samples)
    demo_eclipse(samples, samples[0][0])
    demo_tle_propagation()
