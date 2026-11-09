"""
telemetry/anomalies.py

Injects labeled anomalies into a telemetry DataFrame: overheating,
battery degradation, communication failure, abnormal power consumption,
and sensor drift. Each injected point/window is labeled (is_anomaly,
anomaly_type) so ai/anomaly_detection.py can be evaluated against
ground truth (precision/recall/F1 — Experiment 2 in the design doc).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ANOMALY_TYPES = [
    "overheating", "battery_degradation", "communication_failure",
    "abnormal_power", "sensor_drift",
]


def inject_overheating(df: pd.DataFrame, start_idx: int, duration: int,
                        magnitude_c: float = 20.0) -> pd.DataFrame:
    end_idx = min(len(df), start_idx + duration)
    ramp = np.linspace(0, magnitude_c, end_idx - start_idx)
    df.loc[start_idx:end_idx - 1, "temperature"] += ramp
    df.loc[start_idx:end_idx - 1, "is_anomaly"] = True
    df.loc[start_idx:end_idx - 1, "anomaly_type"] = "overheating"
    return df


def inject_battery_degradation(df: pd.DataFrame, start_idx: int, duration: int,
                                magnitude_v: float = 0.6) -> pd.DataFrame:
    end_idx = min(len(df), start_idx + duration)
    ramp = np.linspace(0, magnitude_v, end_idx - start_idx)
    df.loc[start_idx:end_idx - 1, "battery_voltage"] -= ramp
    df.loc[start_idx:end_idx - 1, "is_anomaly"] = True
    df.loc[start_idx:end_idx - 1, "anomaly_type"] = "battery_degradation"
    return df


def inject_communication_failure(df: pd.DataFrame, start_idx: int, duration: int) -> pd.DataFrame:
    end_idx = min(len(df), start_idx + duration)
    df.loc[start_idx:end_idx - 1, "communication_status"] = 0
    # a comm failure often coincides with an unexpected power draw spike
    # (transmitter fault drawing current without radiating properly)
    df.loc[start_idx:end_idx - 1, "battery_current"] += 0.3
    df.loc[start_idx:end_idx - 1, "is_anomaly"] = True
    df.loc[start_idx:end_idx - 1, "anomaly_type"] = "communication_failure"
    return df


def inject_abnormal_power(df: pd.DataFrame, start_idx: int, duration: int,
                           magnitude_w: float = 8.0) -> pd.DataFrame:
    end_idx = min(len(df), start_idx + duration)
    df.loc[start_idx:end_idx - 1, "solar_power"] -= magnitude_w
    df.loc[start_idx:end_idx - 1, "solar_power"] = df.loc[start_idx:end_idx - 1, "solar_power"].clip(lower=0)
    df.loc[start_idx:end_idx - 1, "is_anomaly"] = True
    df.loc[start_idx:end_idx - 1, "anomaly_type"] = "abnormal_power"
    return df


def inject_sensor_drift(df: pd.DataFrame, start_idx: int, duration: int,
                         column: str = "reaction_wheel_speed", drift_per_step: float = 5.0) -> pd.DataFrame:
    end_idx = min(len(df), start_idx + duration)
    ramp = np.arange(end_idx - start_idx) * drift_per_step
    df.loc[start_idx:end_idx - 1, column] += ramp
    df.loc[start_idx:end_idx - 1, "is_anomaly"] = True
    df.loc[start_idx:end_idx - 1, "anomaly_type"] = "sensor_drift"
    return df


INJECTORS = {
    "overheating": inject_overheating,
    "battery_degradation": inject_battery_degradation,
    "communication_failure": inject_communication_failure,
    "abnormal_power": inject_abnormal_power,
    "sensor_drift": inject_sensor_drift,
}


def inject_random_anomalies(df: pd.DataFrame, n_anomalies: int = 5, min_duration: int = 10,
                             max_duration: int = 40, seed: int = 3, margin: int = 50) -> pd.DataFrame:
    """
    Injects `n_anomalies` non-overlapping anomaly windows of random type,
    randomly placed (away from the very start/end, via `margin`).
    Returns the modified DataFrame (also mutated in place).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = False
        df["anomaly_type"] = ""

    n = len(df)
    used_ranges = []
    placed = 0
    attempts = 0
    while placed < n_anomalies and attempts < n_anomalies * 20:
        attempts += 1
        duration = int(rng.integers(min_duration, max_duration + 1))
        start = int(rng.integers(margin, max(margin + 1, n - margin - duration)))
        end = start + duration
        if any(not (end < r0 or start > r1) for r0, r1 in used_ranges):
            continue  # overlaps an existing anomaly window, try again
        used_ranges.append((start, end))
        kind = rng.choice(ANOMALY_TYPES)
        df = INJECTORS[kind](df, start, duration)
        placed += 1

    return df
