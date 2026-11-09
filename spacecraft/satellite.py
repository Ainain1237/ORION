"""
spacecraft/satellite.py

The spacecraft digital twin: combines Battery, ThermalModel, and Storage
into one stepping simulation of "what happens to the satellite" when it
performs a given activity for a timestep.

This module deliberately knows nothing about *which* activity to choose
(that's mission/planner.py's job) or about orbital position (that's
orbital/'s job) — it just answers "given this activity, this eclipse
state, and this much time, what does my resource state look like now?"
That separation is what lets the planner be swapped (greedy/GA/RL)
without touching the physical model.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from spacecraft.battery import Battery, BatteryFault
from spacecraft.storage import Storage, StorageFault
from spacecraft.thermal import ThermalModel, ThermalState


@dataclass
class ActivityPowerProfile:
    """
    Power/data footprint of one timestep of an activity. Built by
    mission/activities.py from an Activity type; kept here as a plain
    data container so spacecraft/ has no dependency on mission/.
    """
    payload_power_w: float = 0.0       # imaging/payload draw
    comm_power_w: float = 0.0           # transmitter draw
    baseline_power_w: float = 2.0       # OBC, ADCS, sensors — always-on load
    data_generated_mb: float = 0.0      # e.g. from imaging this step
    downlink_rate_mbps: float = 0.0     # 0 unless actively downlinking


@dataclass
class SatelliteState:
    """A snapshot of spacecraft resource state at one instant — this is
    exactly the structure telemetry/generator.py and the dashboard consume."""
    timestamp: object                    # datetime
    battery_pct: float
    battery_wh: float
    solar_gen_w: float
    payload_power_w: float
    comm_active: bool
    thermal_state: str
    temperature_c: float
    memory_pct: float
    memory_used_mb: float
    in_eclipse: bool
    activity: str
    fault: str = ""                       # non-empty if a constraint was violated this step


@dataclass
class Satellite:
    battery: Battery = field(default_factory=Battery)
    storage: Storage = field(default_factory=Storage)
    thermal: ThermalModel = field(default_factory=ThermalModel)

    # Solar array performance
    solar_array_w_max: float = 20.0       # peak generation when sunlit & pointed well
    solar_pointing_efficiency: float = 0.85  # imperfect pointing / array degradation

    def solar_generation_w(self, in_eclipse: bool) -> float:
        if in_eclipse:
            return 0.0
        return self.solar_array_w_max * self.solar_pointing_efficiency

    def step(self, timestamp, dt_s: float, in_eclipse: bool,
              profile: ActivityPowerProfile, activity_name: str,
              strict: bool = False) -> SatelliteState:
        """
        Advance the digital twin by dt_s seconds while performing the
        activity described by `profile`.

        strict=True raises BatteryFault/StorageFault on constraint
        violation (useful in unit tests to catch planner bugs). In a
        running simulation we default to strict=False so the twin
        degrades gracefully (load-shed / truncate) and the fault shows
        up in SatelliteState.fault instead of crashing the sim — a
        constraint violation the planner should have avoided, but the
        *simulator* shouldn't die because of it.
        """
        fault = ""

        solar_w = self.solar_generation_w(in_eclipse)
        total_draw_w = profile.baseline_power_w + profile.payload_power_w + profile.comm_power_w
        net_w = solar_w - total_draw_w

        try:
            if net_w >= 0:
                self.battery.charge(net_w, dt_s)
            else:
                self.battery.discharge(-net_w, dt_s, strict=strict)
        except BatteryFault as e:
            fault = str(e)

        try:
            if profile.data_generated_mb > 0:
                self.storage.add_data(profile.data_generated_mb, strict=strict)
        except StorageFault as e:
            fault = (fault + "; " if fault else "") + str(e)

        if profile.downlink_rate_mbps > 0:
            self.storage.downlink(profile.downlink_rate_mbps, dt_s)

        internal_dissipation = profile.payload_power_w + profile.comm_power_w
        self.thermal.step(dt_s, in_eclipse, internal_dissipation_w=internal_dissipation)

        return SatelliteState(
            timestamp=timestamp,
            battery_pct=self.battery.soc_pct,
            battery_wh=self.battery.soc_wh,
            solar_gen_w=solar_w,
            payload_power_w=profile.payload_power_w,
            comm_active=profile.comm_power_w > 0,
            thermal_state=self.thermal.state.value,
            temperature_c=self.thermal.temperature_c,
            memory_pct=self.storage.used_pct,
            memory_used_mb=self.storage.used_mb,
            in_eclipse=in_eclipse,
            activity=activity_name,
            fault=fault,
        )
