"""
mission/planner.py — the heart of ORION.

Three planners, in increasing sophistication, all answering the same
question: "Should the satellite image, downlink, charge, or stand by
right now?" — and all sharing one evaluation function so their outputs
are directly comparable (Experiment 1 in the design doc).

    1. GreedyPlanner            — myopic rule-based baseline
    2. GeneticPlanner            — population-based search over whole schedules
    3. IntegerProgramPlanner     — linear-relaxation optimal schedule (PuLP/CBC)

A common `TimeStep` timeline (from the orbital layer: eclipse flag per
step) and a list of `Opportunity` windows (imaging targets, comm passes)
drive all three. Each returns a `list[ScheduledActivity]` and the
resulting `SatelliteState` history from actually simulating it against
the digital twin.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from mission.activities import ActivityType, ACTIVITY_PROFILES, Opportunity, ScheduledActivity
from mission.constraints import check_activity_feasible
from spacecraft.satellite import Satellite, SatelliteState


@dataclass
class TimeStep:
    t: datetime
    dt_s: float
    in_eclipse: bool


@dataclass
class ScheduleResult:
    activities: list          # ActivityType per timestep
    states: list               # SatelliteState per timestep
    reward: float
    images_captured: int
    data_downlinked_mb: float
    energy_used_wh: float
    violations: int


def opportunities_active_at(opportunities: list[Opportunity], t: datetime,
                             kind: ActivityType) -> list[Opportunity]:
    return [o for o in opportunities if o.kind == kind and o.is_active(t)]


def evaluate_schedule(satellite: Satellite, timeline: list[TimeStep],
                       opportunities: list[Opportunity],
                       activity_choices: list[ActivityType],
                       image_reward: float = 10.0, downlink_reward_per_mb: float = 0.2,
                       violation_penalty: float = 15.0) -> ScheduleResult:
    """
    Simulate `activity_choices` (one ActivityType per timeline step)
    against a *copy* of `satellite`, enforcing constraints (infeasible
    choices are downgraded to IDLE and counted as a violation), and
    compute the mission reward.

    This is the single source of truth for "how good is this schedule",
    used by every planner and by experiments/ for comparison.
    """
    sat = copy.deepcopy(satellite)
    states: list[SatelliteState] = []
    images_captured = 0
    data_downlinked_mb = 0.0
    energy_used_wh = 0.0
    violations = 0
    actual_activities = []

    for step, requested in zip(timeline, activity_choices):
        violation = check_activity_feasible(sat, requested, step.dt_s, step.in_eclipse)
        activity = requested
        if violation is not None:
            violations += 1
            activity = ActivityType.IDLE  # safe fallback

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

    reward = (images_captured * image_reward
              + data_downlinked_mb * downlink_reward_per_mb
              - violations * violation_penalty)

    return ScheduleResult(
        activities=actual_activities, states=states, reward=reward,
        images_captured=images_captured, data_downlinked_mb=data_downlinked_mb,
        energy_used_wh=energy_used_wh, violations=violations,
    )


# ---------------------------------------------------------------------------
# 1. Greedy planner
# ---------------------------------------------------------------------------

class GreedyPlanner:
    """
    Myopic rule-based baseline: at each timestep, prefer (in order)
    imaging an available target, downlinking if data is queued and a
    pass is open, else charging/idling. Never looks ahead.
    """
    name = "Greedy"

    def plan(self, satellite: Satellite, timeline: list[TimeStep],
             opportunities: list[Opportunity]) -> list[ActivityType]:
        sat = copy.deepcopy(satellite)
        choices = []
        for step in timeline:
            img_opps = opportunities_active_at(opportunities, step.t, ActivityType.IMAGE)
            dl_opps = opportunities_active_at(opportunities, step.t, ActivityType.DOWNLINK)

            choice = ActivityType.IDLE
            if img_opps and sat.storage.used_pct < 90 and sat.battery.soc_pct > 30:
                choice = ActivityType.IMAGE
            elif dl_opps and sat.storage.used_mb > 0 and sat.battery.soc_pct > 25:
                choice = ActivityType.DOWNLINK
            elif sat.battery.soc_pct < 60:
                choice = ActivityType.CHARGE

            violation = check_activity_feasible(sat, choice, step.dt_s, step.in_eclipse)
            if violation is not None:
                choice = ActivityType.IDLE

            profile = ACTIVITY_PROFILES[choice]
            sat.step(step.t, step.dt_s, step.in_eclipse, profile, choice.value, strict=False)
            choices.append(choice)
        return choices


# ---------------------------------------------------------------------------
# 2. Genetic algorithm planner
# ---------------------------------------------------------------------------

class GeneticPlanner:
    """
    Whole-schedule search: each chromosome is a full sequence of
    activity choices over the timeline. Fitness = evaluate_schedule's
    reward. Unlike GreedyPlanner, this can trade off "skip an easy
    image now to save battery for a better one later" because it
    evaluates entire candidate schedules, not one step at a time.
    """
    name = "Genetic Algorithm"

    def __init__(self, population_size: int = 40, generations: int = 25,
                 mutation_rate: float = 0.08, elite_frac: float = 0.15, seed: int = 42):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.rng = random.Random(seed)

    def _allowed_actions(self, t: datetime, opportunities: list[Opportunity]) -> list[ActivityType]:
        actions = [ActivityType.IDLE, ActivityType.CHARGE]
        if opportunities_active_at(opportunities, t, ActivityType.IMAGE):
            actions.append(ActivityType.IMAGE)
        if opportunities_active_at(opportunities, t, ActivityType.DOWNLINK):
            actions.append(ActivityType.DOWNLINK)
        return actions

    def _random_chromosome(self, timeline, allowed_per_step):
        return [self.rng.choice(allowed_per_step[i]) for i in range(len(timeline))]

    def plan(self, satellite: Satellite, timeline: list[TimeStep],
             opportunities: list[Opportunity]) -> list[ActivityType]:
        n = len(timeline)
        allowed_per_step = [self._allowed_actions(step.t, opportunities) for step in timeline]

        population = [self._random_chromosome(timeline, allowed_per_step)
                      for _ in range(self.population_size)]

        n_elite = max(1, int(self.population_size * self.elite_frac))

        for gen in range(self.generations):
            scored = [(evaluate_schedule(satellite, timeline, opportunities, chrom).reward, chrom)
                      for chrom in population]
            scored.sort(key=lambda x: x[0], reverse=True)

            next_gen = [chrom for _, chrom in scored[:n_elite]]  # elitism

            while len(next_gen) < self.population_size:
                parent_a = self._tournament_select(scored)
                parent_b = self._tournament_select(scored)
                child = self._crossover(parent_a, parent_b, n)
                child = self._mutate(child, allowed_per_step)
                next_gen.append(child)

            population = next_gen

        final_scored = [(evaluate_schedule(satellite, timeline, opportunities, chrom).reward, chrom)
                         for chrom in population]
        final_scored.sort(key=lambda x: x[0], reverse=True)
        return final_scored[0][1]

    def _tournament_select(self, scored, k: int = 3):
        contenders = self.rng.sample(scored, min(k, len(scored)))
        return max(contenders, key=lambda x: x[0])[1]

    def _crossover(self, a, b, n):
        point = self.rng.randint(1, n - 1) if n > 1 else 0
        return a[:point] + b[point:]

    def _mutate(self, chrom, allowed_per_step):
        return [self.rng.choice(allowed_per_step[i]) if self.rng.random() < self.mutation_rate
                else gene for i, gene in enumerate(chrom)]


# ---------------------------------------------------------------------------
# 3. Integer programming planner
# ---------------------------------------------------------------------------

class IntegerProgramPlanner:
    """
    Formulates scheduling as a 0/1 integer program (solved with PuLP's
    bundled CBC solver): choose exactly one activity per timestep to
    maximize (images * image_reward + downlinked_mb * dl_reward) subject
    to a *linearized* cumulative energy budget and storage budget.

    This is an approximation of the true (nonlinear, stateful) physics —
    the thermal model and efficiency losses aren't captured in the LP —
    but the linear energy/storage bookkeeping is accurate enough to find
    genuinely good schedules, and any LP infeasibility w.r.t. the *real*
    twin is caught the same way as for the other planners: downgraded to
    IDLE and counted as a violation in evaluate_schedule.
    """
    name = "Integer Programming"

    def __init__(self, image_reward: float = 10.0, downlink_reward_per_mb: float = 0.2):
        self.image_reward = image_reward
        self.downlink_reward_per_mb = downlink_reward_per_mb

    def plan(self, satellite: Satellite, timeline: list[TimeStep],
             opportunities: list[Opportunity]) -> list[ActivityType]:
        import pulp

        n = len(timeline)
        actions = [ActivityType.IDLE, ActivityType.CHARGE, ActivityType.IMAGE, ActivityType.DOWNLINK]

        allowed = []
        for step in timeline:
            step_allowed = [ActivityType.IDLE, ActivityType.CHARGE]
            if opportunities_active_at(opportunities, step.t, ActivityType.IMAGE):
                step_allowed.append(ActivityType.IMAGE)
            if opportunities_active_at(opportunities, step.t, ActivityType.DOWNLINK):
                step_allowed.append(ActivityType.DOWNLINK)
            allowed.append(step_allowed)

        prob = pulp.LpProblem("ORION_mission_schedule", pulp.LpMaximize)

        x = {}  # x[i, a] binary: activity a chosen at step i
        for i, step in enumerate(timeline):
            for a in allowed[i]:
                x[i, a] = pulp.LpVariable(f"x_{i}_{a.value}", cat="Binary")

        # exactly one activity per step
        for i in range(n):
            prob += pulp.lpSum(x[i, a] for a in allowed[i]) == 1

        # objective: reward images + downlinked data
        obj_terms = []
        for i, step in enumerate(timeline):
            if ActivityType.IMAGE in allowed[i]:
                obj_terms.append(self.image_reward * x[i, ActivityType.IMAGE])
            if ActivityType.DOWNLINK in allowed[i]:
                dl_mb = ACTIVITY_PROFILES[ActivityType.DOWNLINK].downlink_rate_mbps * step.dt_s / 8.0
                obj_terms.append(self.downlink_reward_per_mb * dl_mb * x[i, ActivityType.DOWNLINK])
        prob += pulp.lpSum(obj_terms)

        # linearized cumulative energy budget: net energy over the whole
        # horizon must not be more negative than the battery's usable capacity
        energy_terms = []
        for i, step in enumerate(timeline):
            solar_w = satellite.solar_generation_w(step.in_eclipse)
            for a in allowed[i]:
                profile = ACTIVITY_PROFILES[a]
                draw_w = profile.baseline_power_w + profile.payload_power_w + profile.comm_power_w
                net_wh = (solar_w - draw_w) * step.dt_s / 3600.0
                energy_terms.append(net_wh * x[i, a])
        prob += pulp.lpSum(energy_terms) >= -satellite.battery.usable_wh * 0.9  # small safety margin

        # linearized storage budget: cumulative data generated - downlinked <= free capacity
        storage_terms = []
        for i, step in enumerate(timeline):
            for a in allowed[i]:
                profile = ACTIVITY_PROFILES[a]
                net_mb = profile.data_generated_mb - profile.downlink_rate_mbps * step.dt_s / 8.0
                storage_terms.append(net_mb * x[i, a])
        prob += pulp.lpSum(storage_terms) <= satellite.storage.free_mb

        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        choices = []
        for i in range(n):
            chosen = ActivityType.IDLE
            for a in allowed[i]:
                if pulp.value(x[i, a]) and pulp.value(x[i, a]) > 0.5:
                    chosen = a
                    break
            choices.append(chosen)
        return choices


PLANNERS = {
    "greedy": GreedyPlanner,
    "genetic": GeneticPlanner,
    "ip": IntegerProgramPlanner,
}
