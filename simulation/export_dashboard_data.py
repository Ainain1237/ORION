"""
simulation/export_dashboard_data.py

Runs one full ORION simulation (orbit propagation -> planner -> digital
twin -> telemetry -> anomaly injection) and packages everything the
dashboard needs into a single JSON-serializable dict: ground track,
spacecraft vitals over time, mission timeline/activities, opportunities,
and injected anomalies.

Run standalone:
    python -m simulation.export_dashboard_data
Output:
    data/dashboard_data.json
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone

import numpy as np

from simulation.simulator import Simulator
from mission.planner import PLANNERS
from orbital.coordinates import eci_to_geodetic
from telemetry.generator import generate_telemetry
from telemetry.anomalies import inject_random_anomalies

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def build_dashboard_data(planner_name: str = "greedy", n_orbits: float = 6.0, step_s: float = 30.0,
                          n_anomalies: int = 6, seed: int = 4) -> dict:
    epoch = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
    sim = Simulator()
    result = sim.run(epoch, planner_name=planner_name, n_orbits=n_orbits, step_s=step_s)

    df = generate_telemetry(result.schedule.states)
    df = inject_random_anomalies(df, n_anomalies=n_anomalies, min_duration=12, max_duration=30,
                                  seed=seed, margin=20)

    t0 = result.timeline[0].t
    steps = []
    for i, (step, state, (dt, r_eci, _v)) in enumerate(
            zip(result.timeline, result.schedule.states, result.samples)):
        lat, lon, alt = eci_to_geodetic(r_eci, dt)
        row = df.iloc[i]
        steps.append(dict(
            i=i,
            t=dt.isoformat(),
            t_offset_s=(dt - t0).total_seconds(),
            lat=round(float(np.rad2deg(lat)), 3),
            lon=round(float(np.rad2deg(lon)), 3),
            alt_km=round(float(alt), 1),
            activity=state.activity,
            in_eclipse=bool(state.in_eclipse),
            battery_pct=round(state.battery_pct, 2),
            temperature_c=round(state.temperature_c, 2),
            memory_pct=round(state.memory_pct, 2),
            solar_w=round(state.solar_gen_w, 2),
            comm_active=bool(state.comm_active),
            fault=state.fault,
            battery_voltage=round(float(row["battery_voltage"]), 3),
            reaction_wheel_speed=round(float(row["reaction_wheel_speed"]), 1),
            is_anomaly=bool(row["is_anomaly"]),
            anomaly_type=str(row["anomaly_type"]),
        ))

    opportunities = [dict(kind=o.kind.value, label=o.label, start=o.start.isoformat(),
                           end=o.end.isoformat(), start_offset_s=(o.start - t0).total_seconds(),
                           end_offset_s=(o.end - t0).total_seconds())
                      for o in result.opportunities]

    from mission.scenario import build_default_scenario
    # re-derive station/target coordinates for map markers (cheap, avoids
    # threading GroundStation objects through the whole pipeline)
    _, _, _ = None, None, None
    from orbital.ground_track import GroundStation
    ground_stations = [
        GroundStation(name="Bengaluru-GS", lat_deg=12.97, lon_deg=77.59),
        GroundStation(name="Kiruna-GS", lat_deg=67.86, lon_deg=20.23),
    ]
    imaging_targets = [
        GroundStation(name="Target-Amazon", lat_deg=-3.4, lon_deg=-62.2),
        GroundStation(name="Target-Sahara", lat_deg=23.4, lon_deg=11.0),
        GroundStation(name="Target-Himalaya", lat_deg=28.0, lon_deg=84.0),
        GroundStation(name="Target-Andes", lat_deg=-32.0, lon_deg=-70.0),
        GroundStation(name="Target-Siberia", lat_deg=60.0, lon_deg=100.0),
    ]

    data = dict(
        meta=dict(
            mission_name="ORION-1",
            planner_name=result.planner_name,
            epoch=epoch.isoformat(),
            step_s=step_s,
            n_steps=len(steps),
            n_orbits=n_orbits,
            total_images=result.schedule.images_captured,
            total_downlink_mb=round(result.schedule.data_downlinked_mb, 1),
            total_reward=round(result.schedule.reward, 1),
            total_violations=result.schedule.violations,
        ),
        steps=steps,
        opportunities=opportunities,
        ground_stations=[dict(name=g.name, lat=g.lat_deg, lon=g.lon_deg) for g in ground_stations],
        imaging_targets=[dict(name=g.name, lat=g.lat_deg, lon=g.lon_deg) for g in imaging_targets],
    )
    return data


if __name__ == "__main__":
    data = build_dashboard_data()
    out_path = os.path.join(OUT_DIR, "dashboard_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"Saved: {out_path}  ({len(data['steps'])} steps, "
          f"{os.path.getsize(out_path)/1024:.0f} KB)")
