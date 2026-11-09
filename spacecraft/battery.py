"""
spacecraft/battery.py

A simple energy-balance battery model: tracks state of charge in
watt-hours, applies charge/discharge efficiency losses, and enforces
depth-of-discharge and capacity limits — the things that actually
constrain what a CubeSat can do in a given orbit.
"""

from __future__ import annotations
from dataclasses import dataclass


class BatteryFault(Exception):
    """Raised when an operation would violate battery safety limits."""


@dataclass
class Battery:
    capacity_wh: float = 40.0          # typical 3U CubeSat battery pack, ~30-50 Wh
    soc_wh: float = 32.0                # current state of charge [Wh]
    charge_efficiency: float = 0.95     # energy lost to heat while charging
    discharge_efficiency: float = 0.98  # energy lost while discharging
    min_soc_frac: float = 0.20          # don't discharge below 20% (protects cell life)
    max_soc_frac: float = 1.00

    def __post_init__(self):
        self.soc_wh = min(self.soc_wh, self.capacity_wh)

    @property
    def soc_pct(self) -> float:
        return 100.0 * self.soc_wh / self.capacity_wh

    @property
    def usable_wh(self) -> float:
        """Energy available before hitting the minimum SoC floor."""
        return max(0.0, self.soc_wh - self.min_soc_frac * self.capacity_wh)

    @property
    def headroom_wh(self) -> float:
        """Energy that can still be absorbed before hitting max SoC."""
        return max(0.0, self.max_soc_frac * self.capacity_wh - self.soc_wh)

    def can_discharge(self, power_w: float, dt_s: float) -> bool:
        energy_needed = power_w * dt_s / 3600.0
        return energy_needed <= self.usable_wh

    def can_charge(self, power_w: float, dt_s: float) -> bool:
        energy_in = power_w * dt_s / 3600.0 * self.charge_efficiency
        return energy_in <= self.headroom_wh + 1e-9

    def discharge(self, power_w: float, dt_s: float, strict: bool = True) -> float:
        """
        Draw `power_w` watts for `dt_s` seconds.
        Returns the actual energy drawn [Wh]. If strict and insufficient
        energy is available, raises BatteryFault (the planner should
        never let this happen — it's a bug/constraint-violation signal).
        """
        energy_needed = power_w * dt_s / 3600.0
        if energy_needed > self.usable_wh:
            if strict:
                raise BatteryFault(
                    f"Discharge of {energy_needed:.2f} Wh exceeds usable "
                    f"capacity {self.usable_wh:.2f} Wh (SoC={self.soc_pct:.1f}%)")
            energy_needed = self.usable_wh  # clamp — brownout / load shed
        self.soc_wh -= energy_needed / self.discharge_efficiency
        self.soc_wh = max(0.0, self.soc_wh)
        return energy_needed

    def charge(self, power_w: float, dt_s: float) -> float:
        """
        Deliver `power_w` watts of generation for `dt_s` seconds.
        Returns actual energy stored [Wh] (clamped at max SoC — excess
        solar power is simply not captured, as on a real spacecraft
        without a dump load).
        """
        energy_in = power_w * dt_s / 3600.0 * self.charge_efficiency
        energy_in = min(energy_in, self.headroom_wh)
        self.soc_wh += energy_in
        self.soc_wh = min(self.soc_wh, self.capacity_wh)
        return energy_in

    def is_critical(self) -> bool:
        """True if the battery is at or below its minimum safe SoC."""
        return self.soc_pct <= self.min_soc_frac * 100.0 + 1e-6
