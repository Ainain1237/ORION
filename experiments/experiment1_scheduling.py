"""
experiments/experiment1_scheduling.py

Experiment 1 — Mission scheduling comparison (per the ORION design doc):
    Greedy vs Genetic Algorithm vs Integer Programming vs Reinforcement Learning

Measures: mission reward, images captured, data transmitted, energy
consumed, constraint violations, and planning wall-clock time.

Run:
    python -m experiments.experiment1_scheduling
Outputs:
    data/experiment1_results.csv
    data/experiment1_comparison.png
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt

from mission.scenario import build_default_scenario
from mission.planner import GreedyPlanner, GeneticPlanner, IntegerProgramPlanner, evaluate_schedule
from ai.reinforcement_learning import RLPlanner
from spacecraft.satellite import Satellite

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def run_experiment(n_orbits: float = 8.0, step_s: float = 30.0):
    epoch = datetime(2026, 8, 16, tzinfo=timezone.utc)
    timeline, opportunities, samples = build_default_scenario(epoch, n_orbits=n_orbits, step_s=step_s)
    sat = Satellite()

    planners = [
        GreedyPlanner(),
        GeneticPlanner(population_size=40, generations=25, seed=42),
        IntegerProgramPlanner(),
        RLPlanner(episodes=200, seed=1),
    ]

    rows = []
    for planner in planners:
        t0 = time.time()
        choices = planner.plan(sat, timeline, opportunities)
        plan_time_s = time.time() - t0

        result = evaluate_schedule(sat, timeline, opportunities, choices)
        rows.append(dict(
            planner=planner.name,
            reward=result.reward,
            images_captured=result.images_captured,
            data_downlinked_mb=result.data_downlinked_mb,
            energy_used_wh=result.energy_used_wh,
            violations=result.violations,
            plan_time_s=plan_time_s,
        ))
        print(f"{planner.name:35s} reward={result.reward:7.1f}  images={result.images_captured:3d}  "
              f"dl_mb={result.data_downlinked_mb:7.1f}  violations={result.violations}  "
              f"plan_time={plan_time_s:.2f}s")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "experiment1_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    _plot_comparison(df)
    return df


def _plot_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].bar(df["planner"], df["reward"], color="tab:blue")
    axes[0, 0].set_title("Mission Reward")
    axes[0, 0].tick_params(axis="x", rotation=20)

    axes[0, 1].bar(df["planner"], df["images_captured"], color="tab:green")
    axes[0, 1].set_title("Images Captured")
    axes[0, 1].tick_params(axis="x", rotation=20)

    axes[1, 0].bar(df["planner"], df["data_downlinked_mb"], color="tab:orange")
    axes[1, 0].set_title("Data Downlinked (MB)")
    axes[1, 0].tick_params(axis="x", rotation=20)

    axes[1, 1].bar(df["planner"], df["violations"], color="tab:red")
    axes[1, 1].set_title("Constraint Violations")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.suptitle("Experiment 1 — Mission Scheduling: Greedy vs GA vs IP vs RL", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "experiment1_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_experiment()
