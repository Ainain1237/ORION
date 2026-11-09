"""
mission/activities.py

Defines the discrete activities the mission planner chooses between at
each timestep, their power/data cost (as an ActivityPowerProfile the
spacecraft twin can consume directly), and the Opportunity windows
(imaging targets, comm passes) that make some activities available only
at certain times.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from spacecraft.satellite import ActivityPowerProfile


class ActivityType(Enum):
    IDLE = "IDLE"              # minimal power, no payload/comm
    CHARGE = "CHARGE"           # alias of IDLE but explicit intent to conserve/charge
    IMAGE = "IMAGE"              # payload on, generates data
    DOWNLINK = "DOWNLINK"        # comms on, drains stored data
    SAFE_MODE = "SAFE_MODE"      # everything non-essential off; recovery posture


# Nominal power/data costs per activity type. Real missions would tune
# these per-payload; these are representative 3U CubeSat numbers.
ACTIVITY_PROFILES = {
    ActivityType.IDLE: ActivityPowerProfile(baseline_power_w=2.0),
    ActivityType.CHARGE: ActivityPowerProfile(baseline_power_w=1.5),
    ActivityType.IMAGE: ActivityPowerProfile(baseline_power_w=2.0, payload_power_w=10.0,
                                              data_generated_mb=60.0),
    ActivityType.DOWNLINK: ActivityPowerProfile(baseline_power_w=2.0, comm_power_w=8.0,
                                                 downlink_rate_mbps=4.0),
    ActivityType.SAFE_MODE: ActivityPowerProfile(baseline_power_w=1.0),
}


@dataclass
class Opportunity:
    """
    A window during which a particular activity is available — an
    imaging target in view, or a ground-station comm pass. Built from
    orbital.ground_track.compute_access_windows() for comms, or from a
    target-visibility calculation (same geometry, different target) for
    imaging.
    """
    kind: ActivityType         # IMAGE or DOWNLINK
    label: str                  # e.g. "Bengaluru-GS pass" or "Target: Wildfire-CA-04"
    start: datetime
    end: datetime
    priority: float = 1.0        # mission-defined value/reward for taking this opportunity

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()

    def is_active(self, t: datetime) -> bool:
        return self.start <= t <= self.end


@dataclass
class ScheduledActivity:
    """One entry in a planner's output schedule."""
    activity: ActivityType
    start: datetime
    end: datetime
    opportunity_label: str = ""

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()
