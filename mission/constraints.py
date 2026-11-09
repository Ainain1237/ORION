"""
mission/constraints.py

Constraint checks the planner must satisfy before scheduling an
activity. Kept separate from planner.py so every planner (greedy, GA,
IP, RL) shares exactly the same rules — comparisons between planners
are only meaningful if they're all playing by the same physics.
"""

from __future__ import annotations
from dataclasses import dataclass

from mission.activities import ActivityType, ACTIVITY_PROFILES
from spacecraft.satellite import Satellite


@dataclass
class ConstraintViolation:
    reason: str


def check_activity_feasible(sat: Satellite, activity: ActivityType, dt_s: float,
                             in_eclipse: bool) -> ConstraintViolation | None:
    """
    Returns None if `activity` can be legally performed for dt_s seconds
    given the satellite's current resource state, else a
    ConstraintViolation explaining why not.
    """
    profile = ACTIVITY_PROFILES[activity]

    # 1. Imaging requires thermal safety
    if activity == ActivityType.IMAGE and not sat.thermal.payload_safe():
        return ConstraintViolation(
            f"Thermal state {sat.thermal.state.value} ({sat.thermal.temperature_c:.1f}C) "
            f"unsafe for payload operation")

    # 2. Imaging requires free storage
    if activity == ActivityType.IMAGE and not sat.storage.can_store(profile.data_generated_mb):
        return ConstraintViolation(
            f"Storage at {sat.storage.used_pct:.1f}%, insufficient free space for imaging")

    # 3. Downlink requires data to send (no point downlinking an empty buffer,
    #    though it isn't unsafe — treated as a soft inefficiency, not hard block)

    # 4. Power/battery feasibility for any activity
    solar_w = sat.solar_generation_w(in_eclipse)
    total_draw_w = profile.baseline_power_w + profile.payload_power_w + profile.comm_power_w
    net_w = solar_w - total_draw_w
    if net_w < 0 and not sat.battery.can_discharge(-net_w, dt_s):
        return ConstraintViolation(
            f"Battery at {sat.battery.soc_pct:.1f}%, insufficient energy for "
            f"{activity.value} ({-net_w:.1f} W net draw)")

    # 5. Battery-critical -> only SAFE_MODE or CHARGE/IDLE permitted
    if sat.battery.is_critical() and activity in (ActivityType.IMAGE, ActivityType.DOWNLINK):
        return ConstraintViolation(
            f"Battery critical ({sat.battery.soc_pct:.1f}%), only charging/safe-mode allowed")

    # 6. Thermal-critical -> force safe mode
    from spacecraft.thermal import ThermalState
    if sat.thermal.state == ThermalState.CRITICAL and activity != ActivityType.SAFE_MODE:
        return ConstraintViolation(
            f"Thermal state CRITICAL ({sat.thermal.temperature_c:.1f}C), safe mode required")

    return None


def battery_floor_ok(sat: Satellite) -> bool:
    return not sat.battery.is_critical()


def storage_ceiling_ok(sat: Satellite, threshold_pct: float = 95.0) -> bool:
    return not sat.storage.is_full(threshold_pct)
