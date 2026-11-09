"""
experiments/plot_ground_track.py

Generates a ground-track plot (lat/lon over multiple orbits) — useful
directly as a figure in your technical report / paper.

Run:
    python -m experiments.plot_ground_track
Output:
    data/ground_track.png
"""

import os
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt

from orbital.coordinates import KeplerianElements, R_EARTH_EQ
from orbital.propagation import TwoBodyPropagator
from orbital.ground_track import ground_track, GroundStation

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    epoch = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
    elements = KeplerianElements(
        a=R_EARTH_EQ + 523.0, e=0.001, i=np.deg2rad(97.5),
        raan=np.deg2rad(45.0), argp=0.0, nu0=0.0, epoch=epoch,
    )
    prop = TwoBodyPropagator(elements)

    # 3 orbits worth of ground track, so precession/coverage is visible
    n_orbits = 3
    end = epoch + timedelta(seconds=elements.period * n_orbits)
    samples = prop.propagate_range(epoch, end, step_s=30)
    track = ground_track(samples)

    lats = [p[1] for p in track]
    lons = [p[2] for p in track]

    fig, ax = plt.subplots(figsize=(12, 6))
    # Break the line where longitude wraps around, so we don't draw
    # spurious horizontal lines across the whole plot.
    lons_arr = np.array(lons)
    lats_arr = np.array(lats)
    breaks = np.where(np.abs(np.diff(lons_arr)) > 180)[0]
    start = 0
    for b in list(breaks) + [len(lons_arr) - 1]:
        ax.plot(lons_arr[start:b+1], lats_arr[start:b+1], color="tab:blue", lw=1.2)
        start = b + 1

    station = GroundStation(name="Bengaluru-GS", lat_deg=12.97, lon_deg=77.59)
    ax.scatter([station.lon_deg], [station.lat_deg], color="tab:red", s=60,
               zorder=5, label=station.name)

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"ORION — CubeSat Ground Track ({n_orbits} orbits, "
                 f"i={np.rad2deg(elements.i):.1f} deg, alt={elements.altitude_perigee:.0f} km)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    out_path = os.path.join(OUT_DIR, "ground_track.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
