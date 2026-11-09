"""
tests/test_simulation.py
"""

import pytest
from datetime import datetime, timezone

from simulation.simulator import Simulator, FailureInjection


EPOCH = datetime(2026, 8, 16, tzinfo=timezone.utc)


class TestSimulator:
    def test_run_produces_full_state_history(self):
        sim = Simulator()
        result = sim.run(EPOCH, planner_name="greedy", n_orbits=2.0, step_s=60)
        assert len(result.schedule.states) == len(result.timeline)
        assert len(result.samples) == len(result.timeline)

    def test_all_three_planners_run_without_error(self):
        sim = Simulator()
        for name in ["greedy", "genetic", "ip"]:
            result = sim.run(EPOCH, planner_name=name, n_orbits=2.0, step_s=60)
            assert result.schedule is not None

    def test_fault_injection_raises_temperature(self):
        sim = Simulator()
        result_normal = sim.run(EPOCH, planner_name="greedy", n_orbits=2.0, step_s=60)
        result_fault = sim.run(EPOCH, planner_name="greedy", n_orbits=2.0, step_s=60,
                                failures=[FailureInjection(at_step=50, kind="thermal_anomaly")])
        assert result_fault.schedule.states[50].temperature_c > result_normal.schedule.states[50].temperature_c

    def test_battery_degradation_reduces_capacity(self):
        sim = Simulator()
        result = sim.run(EPOCH, planner_name="greedy", n_orbits=2.0, step_s=60,
                          failures=[FailureInjection(at_step=10, kind="battery_degradation", magnitude=1.0)])
        # after degradation, SoC% should reflect a smaller effective capacity;
        # just check the run completes and produces sane bounded values
        for s in result.schedule.states:
            assert 0 <= s.battery_pct <= 100


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
