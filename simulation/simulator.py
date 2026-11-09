"""
simulation/simulator.py

Ties Layers 1-3 into one run: propagate the orbit, build opportunities,
run a chosen planner, and step the spacecraft digital twin through the
resulting schedule — producing the full SatelliteState history that
Layer 4 (telemetry) and the dashboard consume.

This is deliberately a thin orchestration layer — all the actual physics
and decision logic live in orbital/, spacecraft/, and mission/. Keeping
it thin is what makes it easy to support "accelerate time" (just don't
even generate finer samples) and "inject failures" (perturb the twin
mid-run) without touching the other layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from mission.activities import ActivityType, ACTIVITY_PROFILES, Opportunity
from mission.constraints import check_activity_feasible
from mission.planner import TimeStep, ScheduleResult, PLANNERS
from mission.scenario import build_default_scenario
from spacecraft.satellite import Satellite, SatelliteState


@dataclass
class FailureInjection:
    """A scripted fault injected at a specific point in the simulation,
    for Experiment 3 (autonomous response to anomalies)."""
    at_step: int
    kind: str    # "battery_degradation" | "comm_failure" | "thermal_anomaly"
    magnitude: float = 1.0


@dataclass
class SimulationResult:
    planner_name: str
    timeline: list
    opportunities: list
    schedule: ScheduleResult
    samples: list          # raw orbital (dt, r_eci, v_eci) samples, for ground track / 3D viz


class Simulator:
    def __init__(self, satellite: Satellite | None = None):
        self.satellite = satellite or Satellite()

    def run(self, epoch: datetime, planner_name: str = "greedy", n_orbits: float = 8.0,
             step_s: float = 30.0, failures: list[FailureInjection] | None = None) -> SimulationResult:
        timeline, opportunities, samples = build_default_scenario(epoch, n_orbits=n_orbits,
                                                                    step_s=step_s)

        planner_cls = PLANNERS[planner_name]
        planner = planner_cls()
        choices = planner.plan(self.satellite, timeline, opportunities)

        if failures:
            choices = self._apply_failures(choices, failures)

        schedule = self._simulate_with_faults(timeline, opportunities, choices, failures or [])

        return SimulationResult(planner_name=planner.name, timeline=timeline,
                                 opportunities=opportunities, schedule=schedule, samples=samples)

    def _apply_failures(self, choices, failures):
        # Failures are applied physically during _simulate_with_faults;
        # choices themselves are unaffected (the planner didn't know
        # about the fault in advance — that's the point of the experiment).
        return choices

    def _simulate_with_faults(self, timeline: list[TimeStep], opportunities: list[Opportunity],
                               activity_choices: list[ActivityType],
                               failures: list[FailureInjection]) -> ScheduleResult:
        import copy
        sat = copy.deepcopy(self.satellite)
        states: list[SatelliteState] = []
        images_captured = 0
        data_downlinked_mb = 0.0
        energy_used_wh = 0.0
        violations = 0
        actual_activities = []

        failures_by_step = {f.at_step: f for f in failures}

        for i, (step, requested) in enumerate(zip(timeline, activity_choices)):
            if i in failures_by_step:
                self._inject_fault(sat, failures_by_step[i])

            violation = check_activity_feasible(sat, requested, step.dt_s, step.in_eclipse)
            activity = requested
            if violation is not None:
                violations += 1
                activity = ActivityType.SAFE_MODE if sat.battery.is_critical() else ActivityType.IDLE

            profile = ACTIVITY_PROFILES[activity]
            soc_before = sat.battery.soc_wh
            state = sat.step(step.t, step.dt_s, step.in_eclipse, profile, activity.value,
                              strict=False)
            states.append(state)
            actual_activities.append(activity)

            energy_used_wh += max(0.0, soc_before - sat.battery.soc_wh)
            if activity == ActivityType.IMAGE and state.fault == "":
                images_captured += 1
            if activity == ActivityType.DOWNLINK:
                data_downlinked_mb += profile.downlink_rate_mbps * step.dt_s / 8.0

        reward = images_captured * 10.0 + data_downlinked_mb * 0.2 - violations * 15.0

        return ScheduleResult(
            activities=actual_activities, states=states, reward=reward,
            images_captured=images_captured, data_downlinked_mb=data_downlinked_mb,
            energy_used_wh=energy_used_wh, violations=violations,
        )

    def _inject_fault(self, sat: Satellite, failure: FailureInjection):
        if failure.kind == "battery_degradation":
            # sudden capacity loss, e.g. a cell fault
            sat.battery.capacity_wh *= (1 - 0.3 * failure.magnitude)
            sat.battery.soc_wh = min(sat.battery.soc_wh, sat.battery.capacity_wh)
        elif failure.kind == "comm_failure":
            # simulated by zeroing the DOWNLINK profile's effectiveness for
            # remaining steps would require deeper plumbing; here we model
            # it as an instantaneous battery/thermal shock from a fault event
            sat.battery.discharge(sat.battery.soc_wh * 0.1 * failure.magnitude, 1, strict=False)
        elif failure.kind == "thermal_anomaly":
            sat.thermal.temperature_c += 25.0 * failure.magnitude
