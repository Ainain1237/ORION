"""
spacecraft/storage.py

Onboard mass-memory model. Imaging fills it, downlink drains it — the
simplest resource loop that couples the payload to the comms subsystem,
which is exactly the kind of cross-subsystem constraint the mission
planner has to reason about.
"""

from __future__ import annotations
from dataclasses import dataclass


class StorageFault(Exception):
    """Raised when an operation would overflow or underflow storage."""


@dataclass
class Storage:
    capacity_mb: float = 8192.0   # e.g. 8 GB onboard flash, typical for a small CubeSat
    used_mb: float = 0.0

    @property
    def free_mb(self) -> float:
        return max(0.0, self.capacity_mb - self.used_mb)

    @property
    def used_pct(self) -> float:
        return 100.0 * self.used_mb / self.capacity_mb

    def can_store(self, data_mb: float) -> bool:
        return data_mb <= self.free_mb

    def add_data(self, data_mb: float, strict: bool = True) -> float:
        """Record newly captured data (e.g. from an imaging pass)."""
        if data_mb > self.free_mb:
            if strict:
                raise StorageFault(
                    f"Cannot store {data_mb:.1f} MB, only {self.free_mb:.1f} MB free "
                    f"(used={self.used_pct:.1f}%)")
            data_mb = self.free_mb  # image gets truncated/dropped past capacity
        self.used_mb += data_mb
        return data_mb

    def downlink(self, rate_mbps: float, dt_s: float) -> float:
        """
        Transmit at `rate_mbps` megabits/sec for `dt_s` seconds.
        Returns actual data removed [MB].
        """
        data_mb = rate_mbps * dt_s / 8.0   # Mb/s -> MB/s
        data_mb = min(data_mb, self.used_mb)
        self.used_mb -= data_mb
        return data_mb

    def is_full(self, threshold_pct: float = 95.0) -> bool:
        return self.used_pct >= threshold_pct
