"""
experiments/experiment2_fault_detection.py

Experiment 2 — Fault detection comparison (per the ORION design doc):
    Isolation Forest vs PCA Reconstruction (linear-autoencoder equivalent)

Measures: precision, recall, F1, mean detection latency (steps).

Run:
    python -m experiments.experiment2_fault_detection
Outputs:
    data/experiment2_results.csv
    data/experiment2_comparison.png
"""

import os
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt

from simulation.simulator import Simulator
from telemetry.generator import generate_telemetry
from telemetry.anomalies import inject_random_anomalies
from ai.anomaly_detection import IsolationForestDetector, PCAReconstructionDetector, evaluate_detector

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def run_experiment(n_orbits: float = 10.0, step_s: float = 30.0, n_anomalies: int = 10,
                    seed: int = 11):
    epoch = datetime(2026, 8, 16, tzinfo=timezone.utc)
    sim = Simulator()
    result = sim.run(epoch, planner_name="greedy", n_orbits=n_orbits, step_s=step_s)
    df = generate_telemetry(result.schedule.states)
    df = inject_random_anomalies(df, n_anomalies=n_anomalies, min_duration=15, max_duration=35,
                                  seed=seed)

    split = int(len(df) * 0.5)
    train_df, test_df = df.iloc[:split], df.iloc[split:].reset_index(drop=True)
    print(f"Telemetry: {len(df)} steps | train anomalies={train_df.is_anomaly.sum()} "
          f"test anomalies={test_df.is_anomaly.sum()}")

    detectors = [IsolationForestDetector(contamination=0.08), PCAReconstructionDetector(threshold_std=2.5)]

    rows = []
    for det in detectors:
        ev = evaluate_detector(det, train_df, test_df)
        rows.append(dict(
            detector=ev.detector_name, precision=ev.precision, recall=ev.recall, f1=ev.f1,
            detection_latency_steps=ev.mean_detection_latency_steps,
            n_true_anomalies=ev.n_true_anomalies, n_flagged=ev.n_flagged,
            fit_time_s=ev.fit_time_s, predict_time_s=ev.predict_time_s,
        ))
        print(f"{ev.detector_name:40s} P={ev.precision:.2f} R={ev.recall:.2f} F1={ev.f1:.2f} "
              f"latency={ev.mean_detection_latency_steps:.1f} steps  flagged={ev.n_flagged}")

    result_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "experiment2_results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    _plot_comparison(result_df)
    return result_df


def _plot_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(df))
    width = 0.25
    axes[0].bar([i - width for i in x], df["precision"], width, label="Precision")
    axes[0].bar(x, df["recall"], width, label="Recall")
    axes[0].bar([i + width for i in x], df["f1"], width, label="F1")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(df["detector"], rotation=15, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Detection Quality")
    axes[0].legend()

    axes[1].bar(df["detector"], df["detection_latency_steps"], color="tab:purple")
    axes[1].set_title("Mean Detection Latency (steps into anomaly window)")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("Experiment 2 — Fault Detection: Isolation Forest vs PCA Reconstruction", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "experiment2_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_experiment()
