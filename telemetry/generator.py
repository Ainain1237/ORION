"""
telemetry/generator.py

Converts a SatelliteState history (from a simulation run) into the kind
of telemetry stream a real spacecraft would downlink: battery voltage
and current, temperature, solar power, memory usage, reaction-wheel
speed, and communication status. Adds realistic sensor noise so the
downstream anomaly detectors have to do real work, not just threshold
a clean signal.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from spacecraft.satellite import SatelliteState


# Nominal battery pack characteristics used to derive voltage/current
# from the SoC/power values the digital twin already tracks.
NOMINAL_PACK_VOLTAGE = 7.4    # V, typical 2S Li-ion CubeSat pack
VOLTAGE_SAG_PER_PCT_SOC = 0.01  # simple linear SoC->voltage relationship


def _battery_voltage(battery_pct: float, current_a: float, noise_std: float = 0.02) -> float:
    """Simple linear SoC-to-voltage model with a small IR-drop term."""
    v = NOMINAL_PACK_VOLTAGE - (100 - battery_pct) * VOLTAGE_SAG_PER_PCT_SOC
    v -= 0.05 * current_a  # internal resistance sag under load
    return v


def generate_telemetry(states: list[SatelliteState], seed: int = 7,
                        add_noise: bool = True) -> pd.DataFrame:
    """
    Build a telemetry DataFrame with columns:
        timestamp, battery_voltage, battery_current, temperature,
        solar_power, memory_usage, reaction_wheel_speed, communication_status,
        activity, in_eclipse
    """
    rng = np.random.default_rng(seed)
    rows = []

    prev_battery_wh = states[0].battery_wh if states else 0.0
    for s in states:
        # approximate current draw from the change in stored energy this step
        # (a real telemetry stream would measure this directly; here we
        # derive it from the twin's own energy bookkeeping)
        d_wh = prev_battery_wh - s.battery_wh
        prev_battery_wh = s.battery_wh
        current_a = max(0.0, d_wh) / NOMINAL_PACK_VOLTAGE  # rough Wh->Ah proxy per step

        voltage = _battery_voltage(s.battery_pct, current_a)

        # activity-coupled baseline; crc32, not hash(), because Python randomises
        # str hashes per process and the telemetry must be reproducible
        reaction_wheel_speed = 3000 + 500 * np.sin(zlib.crc32(s.activity.encode()) % 100)

        row = dict(
            timestamp=s.timestamp,
            battery_voltage=voltage,
            battery_current=current_a,
            temperature=s.temperature_c,
            solar_power=s.solar_gen_w,
            memory_usage=s.memory_pct,
            reaction_wheel_speed=reaction_wheel_speed,
            communication_status=1 if s.comm_active else 0,
            activity=s.activity,
            in_eclipse=s.in_eclipse,
            is_anomaly=False,       # set by telemetry.anomalies when injecting faults
            anomaly_type="",
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    if add_noise:
        df["battery_voltage"] += rng.normal(0, 0.01, len(df))
        df["battery_current"] += np.abs(rng.normal(0, 0.005, len(df)))
        df["temperature"] += rng.normal(0, 0.3, len(df))
        df["solar_power"] += rng.normal(0, 0.2, len(df))
        df["reaction_wheel_speed"] += rng.normal(0, 15, len(df))

    return df
