"""
experiments/experiment3_autonomous_response.py

Experiment 3 — Autonomous response to injected failures (per the ORION
design doc): inject battery degradation / thermal anomaly mid-mission
and compare how a simple rule-based response and the RL-trained planner
each hold up, versus an oblivious baseline that doesn't adapt at all.

"Rule-based response" here means: once a fault is detected (thermal
critical or battery critical), force SAFE_MODE until the satellite
recovers — the standard onboard FDIR (Fault Detection, Isolation and
Recovery) approach real small satellites use. This is compared against
the RL planner, which was never told about the fault in advance (same
as every other planner) but can be evaluated on how much of its
pre-fault reward it retains afterward.

Run:
    python -m experiments.experiment3_autonomous_response
Outputs:
    data/experiment3_results.csv
    data/experiment3_comparison.png
"""

import os
import copy
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt

from mission.scenario import build_default_scenario
from mission.activities import ActivityType, ACTIVITY_PROFILES
from mission.constraints import check_activity_feasible
from mission.planner import GreedyPlanner, evaluate_schedule
from ai.reinforcement_learning import RLPlanner
from spacecraft.satellite import Satellite
from spacecraft.thermal import ThermalState

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def _simulate_with_fault_and_response(satellite, timeline, opportunities, base_choices,
                                       fault_step: int, fault_kind: str, use_fdir: bool):
    """
    Re-simulates `base_choices` against a fresh twin, injecting a fault
    at `fault_step`. If use_fdir, a simple rule-based FDIR overrides the
    planned activity with SAFE_MODE whenever the twin is in a critical
    state, until it recovers.
    """
    sat = copy.deepcopy(satellite)
    images, dl_mb, violations, reward = 0, 0.0, 0, 0.0

    for i, (step, planned) in enumerate(zip(timeline, base_choices)):
        if i == fault_step:
            if fault_kind == "battery_degradation":
                sat.battery.capacity_wh *= 0.6
                # a real degradation event (e.g. a cell going open-circuit)
                # also knocks down available charge, not just capacity —
                # otherwise the fault never actually threatens the battery floor
                sat.battery.soc_wh = 0.15 * sat.battery.capacity_wh
            elif fault_kind == "thermal_anomaly":
                sat.thermal.temperature_c += 30.0

        activity = planned
        if use_fdir and (sat.battery.is_critical() or sat.thermal.state == ThermalState.CRITICAL):
            activity = ActivityType.SAFE_MODE

        violation = check_activity_feasible(sat, activity, step.dt_s, step.in_eclipse)
        if violation is not None:
            violations += 1
            activity = ActivityType.IDLE

        profile = ACTIVITY_PROFILES[activity]
        state = sat.step(step.t, step.dt_s, step.in_eclipse, profile, activity.value, strict=False)

        if activity == ActivityType.IMAGE and state.fault == "":
            images += 1
        if activity == ActivityType.DOWNLINK:
            dl_mb += profile.downlink_rate_mbps * step.dt_s / 8.0

    reward = images * 10.0 + dl_mb * 0.2 - violations * 15.0
    return dict(images_captured=images, data_downlinked_mb=dl_mb, violations=violations, reward=reward)


def run_experiment(n_orbits: float = 8.0, step_s: float = 30.0):
    epoch = datetime(2026, 8, 16, tzinfo=timezone.utc)
    timeline, opportunities, samples = build_default_scenario(epoch, n_orbits=n_orbits, step_s=step_s)
    sat = Satellite()
    fault_step = len(timeline) // 3

    greedy_choices = GreedyPlanner().plan(sat, timeline, opportunities)
    rl_planner = RLPlanner(episodes=150, seed=2)
    rl_choices = rl_planner.plan(sat, timeline, opportunities)

    scenarios = [
        ("battery_degradation", "No FDIR (oblivious)", greedy_choices, False),
        ("battery_degradation", "Rule-based FDIR", greedy_choices, True),
        ("battery_degradation", "RL planner alone", rl_choices, False),
        ("battery_degradation", "RL + rule-based safety layer", rl_choices, True),
        ("thermal_anomaly", "No FDIR (oblivious)", greedy_choices, False),
        ("thermal_anomaly", "Rule-based FDIR", greedy_choices, True),
        ("thermal_anomaly", "RL planner alone", rl_choices, False),
        ("thermal_anomaly", "RL + rule-based safety layer", rl_choices, True),
    ]

    rows = []
    for fault_kind, label, choices, use_fdir in scenarios:
        outcome = _simulate_with_fault_and_response(sat, timeline, opportunities, choices,
                                                      fault_step, fault_kind, use_fdir)
        rows.append(dict(fault=fault_kind, response=label, **outcome))
        print(f"[{fault_kind:20s}] {label:32s} reward={outcome['reward']:7.1f}  "
              f"images={outcome['images_captured']:3d}  violations={outcome['violations']}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "experiment3_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    _plot_comparison(df)
    return df


def _plot_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, fault_kind in zip(axes, df["fault"].unique()):
        sub = df[df["fault"] == fault_kind]
        colors = ["tab:red", "tab:green", "tab:orange", "tab:blue"][:len(sub)]
        ax.bar(sub["response"], sub["reward"], color=colors)
        ax.set_title(f"Fault: {fault_kind}")
        ax.set_ylabel("Mission reward (post-fault)")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Experiment 3 — Autonomous Response to Injected Failures", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "experiment3_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_experiment()
