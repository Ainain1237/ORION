"""
tests/test_spacecraft.py
"""

import pytest
from datetime import datetime, timezone

from spacecraft.battery import Battery, BatteryFault
from spacecraft.storage import Storage, StorageFault
from spacecraft.thermal import ThermalModel, ThermalState
from spacecraft.satellite import Satellite, ActivityPowerProfile


class TestBattery:
    def test_charge_increases_soc(self):
        b = Battery(capacity_wh=40, soc_wh=20)
        b.charge(power_w=10, dt_s=3600)  # 10W for 1hr = 10Wh in, minus efficiency
        assert b.soc_wh > 20
        assert b.soc_wh < 30  # efficiency loss applied

    def test_discharge_decreases_soc(self):
        b = Battery(capacity_wh=40, soc_wh=30)
        b.discharge(power_w=5, dt_s=3600)
        assert b.soc_wh < 30

    def test_strict_discharge_raises_when_insufficient(self):
        b = Battery(capacity_wh=40, soc_wh=8.5, min_soc_frac=0.2)  # near floor (8 Wh)
        with pytest.raises(BatteryFault):
            b.discharge(power_w=100, dt_s=3600, strict=True)

    def test_nonstrict_discharge_clamps(self):
        b = Battery(capacity_wh=40, soc_wh=8.5, min_soc_frac=0.2)
        usable_before = b.usable_wh
        drawn = b.discharge(power_w=100, dt_s=3600, strict=False)
        assert drawn <= usable_before + 1e-6
        # clamped near the floor (small overshoot is expected: the clamp
        # limits *delivered* energy, but discharge efficiency losses pull
        # slightly more than that out of the cell itself)
        assert b.soc_pct >= 20 - 1.0

    def test_charge_does_not_exceed_capacity(self):
        b = Battery(capacity_wh=40, soc_wh=39)
        b.charge(power_w=100, dt_s=3600)
        assert b.soc_wh <= 40


class TestStorage:
    def test_add_data_fills_storage(self):
        s = Storage(capacity_mb=1000, used_mb=0)
        s.add_data(200)
        assert s.used_mb == 200

    def test_overflow_raises_when_strict(self):
        s = Storage(capacity_mb=1000, used_mb=900)
        with pytest.raises(StorageFault):
            s.add_data(200, strict=True)

    def test_downlink_frees_storage(self):
        s = Storage(capacity_mb=1000, used_mb=500)
        removed = s.downlink(rate_mbps=8.0, dt_s=60)  # 8 Mbps * 60s /8 = 60 MB
        assert removed == pytest.approx(60.0)
        assert s.used_mb == pytest.approx(440.0)

    def test_downlink_cannot_go_negative(self):
        s = Storage(capacity_mb=1000, used_mb=10)
        s.downlink(rate_mbps=100, dt_s=60)
        assert s.used_mb == 0


class TestThermal:
    def test_heats_up_when_sunlit_with_dissipation(self):
        t = ThermalModel(temperature_c=20.0)
        for _ in range(20):
            t.step(dt_s=60, in_eclipse=False, internal_dissipation_w=15)
        assert t.temperature_c > 20.0

    def test_cools_down_in_eclipse(self):
        t = ThermalModel(temperature_c=20.0)
        for _ in range(50):
            t.step(dt_s=60, in_eclipse=True, internal_dissipation_w=0)
        assert t.temperature_c < 20.0

    def test_state_classification(self):
        t = ThermalModel(temperature_c=55.0, warm_limit_c=50, critical_high_c=60)
        assert t.state == ThermalState.WARM
        t.temperature_c = 65.0
        assert t.state == ThermalState.CRITICAL


class TestSatelliteTwin:
    def test_step_produces_valid_state(self):
        sat = Satellite()
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        profile = ActivityPowerProfile(payload_power_w=5, baseline_power_w=2,
                                        data_generated_mb=100)
        state = sat.step(ts, dt_s=60, in_eclipse=False, profile=profile, activity_name="IMAGE")
        assert 0 <= state.battery_pct <= 100
        assert state.memory_used_mb == pytest.approx(100)
        assert state.fault == ""

    def test_eclipse_zero_solar_generation(self):
        sat = Satellite()
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        state = sat.step(ts, dt_s=60, in_eclipse=True, profile=ActivityPowerProfile(),
                          activity_name="IDLE")
        assert state.solar_gen_w == 0.0

    def test_long_imaging_drains_battery_and_fills_storage(self):
        sat = Satellite(battery=Battery(capacity_wh=40, soc_wh=40),
                         storage=Storage(capacity_mb=2000, used_mb=0))
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        profile = ActivityPowerProfile(payload_power_w=12, baseline_power_w=2,
                                        data_generated_mb=50)
        for _ in range(30):  # 30 steps of imaging while in eclipse (no solar)
            state = sat.step(ts, dt_s=60, in_eclipse=True, profile=profile, activity_name="IMAGE")
        assert state.battery_pct < 100
        assert state.memory_used_mb == pytest.approx(1500)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
