"""
mission/scenario.py

Bridges Layer 1 (orbital) to Layer 3 (mission planning): builds a
TimeStep timeline with eclipse flags, and an Opportunity list (imaging
targets + ground-station comm passes) from real orbit propagation. This
is the "input" section from the ORION design doc turned into code.
"""

from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np

from orbital.coordinates import KeplerianElements, R_EARTH_EQ
from orbital.propagation import TwoBodyPropagator
from orbital.ground_track import (
    GroundStation, compute_access_windows, is_eclipse, sun_vector_eci, eci_to_geodetic,
)
from mission.activities import ActivityType, Opportunity
from mission.planner import TimeStep


def build_default_scenario(epoch: datetime, n_orbits: float = 3.0, step_s: float = 30.0,
                            imaging_targets: list[GroundStation] | None = None,
                            ground_stations: list[GroundStation] | None = None):
    """
    Propagates a representative 523km sun-sync-ish CubeSat orbit and
    returns (timeline, opportunities, samples) ready for a planner.

    imaging_targets / ground_stations use the same GroundStation
    structure (lat/lon/elevation mask) — an "imaging opportunity" is
    just a target on the ground becoming visible with adequate geometry,
    same math as a comm access window.
    """
    elements = KeplerianElements(
        a=R_EARTH_EQ + 523.0, e=0.001, i=np.deg2rad(97.5),
        raan=np.deg2rad(45.0), argp=0.0, nu0=0.0, epoch=epoch,
    )
    prop = TwoBodyPropagator(elements)
    end = epoch + timedelta(seconds=elements.period * n_orbits)
    samples = prop.propagate_range(epoch, end, step_s=step_s)

    timeline = []
    for dt, r_eci, _v in samples:
        eclipsed = is_eclipse(r_eci, sun_vector_eci(dt))
        timeline.append(TimeStep(t=dt, dt_s=step_s, in_eclipse=eclipsed))

    if ground_stations is None:
        ground_stations = [
            GroundStation(name="Bengaluru-GS", lat_deg=12.97, lon_deg=77.59, min_elevation_deg=10),
            GroundStation(name="Kiruna-GS", lat_deg=67.86, lon_deg=20.23, min_elevation_deg=10),
        ]
    if imaging_targets is None:
        # min_elevation_deg=40 approximates "target within the payload's
        # usable off-nadir imaging cone" — tighter than a comm mask since
        # imaging quality degrades faster with slant angle than RF link margin does.
        imaging_targets = [
            GroundStation(name="Target-Amazon", lat_deg=-3.4, lon_deg=-62.2, min_elevation_deg=25),
            GroundStation(name="Target-Sahara", lat_deg=23.4, lon_deg=11.0, min_elevation_deg=25),
            GroundStation(name="Target-Himalaya", lat_deg=28.0, lon_deg=84.0, min_elevation_deg=25),
            GroundStation(name="Target-Andes", lat_deg=-32.0, lon_deg=-70.0, min_elevation_deg=25),
            GroundStation(name="Target-Siberia", lat_deg=60.0, lon_deg=100.0, min_elevation_deg=25),
        ]

    opportunities = []
    for gs in ground_stations:
        for w in compute_access_windows(samples, gs):
            opportunities.append(Opportunity(kind=ActivityType.DOWNLINK, label=w.station,
                                              start=w.aos, end=w.los, priority=1.0))
    for tgt in imaging_targets:
        for w in compute_access_windows(samples, tgt):
            opportunities.append(Opportunity(kind=ActivityType.IMAGE, label=tgt.name,
                                              start=w.aos, end=w.los, priority=1.0))

    return timeline, opportunities, samples
