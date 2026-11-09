"""
tests/test_mission.py
"""

import pytest
from datetime import datetime, timedelta, timezone

from spacecraft.satellite import Satellite
from spacecraft.battery import Battery
from spacecraft.storage import Storage
from mission.activities import ActivityType, Opportunity, ACTIVITY_PROFILES
from mission.constraints import check_activity_feasible
from mission.planner import (
    TimeStep, GreedyPlanner, GeneticPlanner, IntegerProgramPlanner, evaluate_schedule,
)


def make_timeline(n=20, dt_s=60, eclipse_every=5, start=None):
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [TimeStep(t=start + timedelta(seconds=i * dt_s), dt_s=dt_s,
                      in_eclipse=(i % eclipse_every == 0))
            for i in range(n)]


class TestConstraints:
    def test_imaging_blocked_when_thermally_unsafe(self):
        sat = Satellite()
        sat.thermal.temperature_c = 55.0  # above nominal_range default (-10,40)
        v = check_activity_feasible(sat, ActivityType.IMAGE, 60, in_eclipse=False)
        assert v is not None

    def test_imaging_blocked_when_storage_full(self):
        sat = Satellite(storage=Storage(capacity_mb=100, used_mb=99))
        v = check_activity_feasible(sat, ActivityType.IMAGE, 60, in_eclipse=False)
        assert v is not None

    def test_idle_always_feasible_with_healthy_battery(self):
        sat = Satellite()
        v = check_activity_feasible(sat, ActivityType.IDLE, 60, in_eclipse=False)
        assert v is None

    def test_battery_critical_blocks_imaging(self):
        sat = Satellite(battery=Battery(capacity_wh=40, soc_wh=8, min_soc_frac=0.2))
        v = check_activity_feasible(sat, ActivityType.IMAGE, 60, in_eclipse=True)
        assert v is not None


class TestEvaluateSchedule:
    def test_all_idle_schedule_has_zero_reward_zero_violations(self):
        sat = Satellite()
        timeline = make_timeline()
        choices = [ActivityType.IDLE] * len(timeline)
        result = evaluate_schedule(sat, timeline, [], choices)
        assert result.reward == 0
        assert result.violations == 0
        assert result.images_captured == 0

    def test_imaging_without_opportunity_still_simulates_but_counts_image(self):
        """evaluate_schedule doesn't itself check opportunity windows —
        that's the planner's job — it just simulates physics + feasibility."""
        sat = Satellite()
        timeline = make_timeline(n=5)
        choices = [ActivityType.IMAGE] * 5
        result = evaluate_schedule(sat, timeline, [], choices)
        assert result.images_captured == 5
        assert len(result.states) == 5


class TestPlanners:
    def _scenario(self):
        timeline = make_timeline(n=60, dt_s=30, eclipse_every=1000)  # no eclipse, plenty of sun
        start = timeline[0].t
        opportunities = [
            Opportunity(kind=ActivityType.IMAGE, label="tgt1",
                        start=start + timedelta(seconds=5*30), end=start + timedelta(seconds=15*30)),
            Opportunity(kind=ActivityType.DOWNLINK, label="gs1",
                        start=start + timedelta(seconds=25*30), end=start + timedelta(seconds=40*30)),
        ]
        sat = Satellite(battery=Battery(capacity_wh=40, soc_wh=40),
                         storage=Storage(capacity_mb=2000, used_mb=0))
        return sat, timeline, opportunities

    def test_greedy_captures_available_image(self):
        sat, timeline, opportunities = self._scenario()
        planner = GreedyPlanner()
        choices = planner.plan(sat, timeline, opportunities)
        assert ActivityType.IMAGE in choices

    def test_greedy_produces_one_choice_per_step(self):
        sat, timeline, opportunities = self._scenario()
        planner = GreedyPlanner()
        choices = planner.plan(sat, timeline, opportunities)
        assert len(choices) == len(timeline)

    def test_genetic_planner_beats_or_matches_all_idle(self):
        sat, timeline, opportunities = self._scenario()
        planner = GeneticPlanner(population_size=15, generations=8)
        choices = planner.plan(sat, timeline, opportunities)
        result = evaluate_schedule(sat, timeline, opportunities, choices)
        idle_result = evaluate_schedule(sat, timeline, opportunities,
                                         [ActivityType.IDLE] * len(timeline))
        assert result.reward >= idle_result.reward

    def test_ip_planner_never_violates_constraints(self):
        sat, timeline, opportunities = self._scenario()
        planner = IntegerProgramPlanner()
        choices = planner.plan(sat, timeline, opportunities)
        result = evaluate_schedule(sat, timeline, opportunities, choices)
        assert result.violations == 0

    def test_all_planners_return_valid_activity_types(self):
        sat, timeline, opportunities = self._scenario()
        for planner in [GreedyPlanner(), GeneticPlanner(population_size=10, generations=5),
                        IntegerProgramPlanner()]:
            choices = planner.plan(sat, timeline, opportunities)
            assert all(isinstance(c, ActivityType) for c in choices)
            assert len(choices) == len(timeline)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
