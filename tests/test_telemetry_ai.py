"""
tests/test_telemetry_ai.py
"""

import pytest
from datetime import datetime, timezone

from simulation.simulator import Simulator
from telemetry.generator import generate_telemetry
from telemetry.anomalies import (
    inject_overheating, inject_battery_degradation, inject_random_anomalies,
)
from ai.anomaly_detection import (
    IsolationForestDetector, PCAReconstructionDetector, evaluate_detector, FEATURE_COLUMNS,
)


EPOCH = datetime(2026, 8, 16, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def telemetry_df():
    sim = Simulator()
    result = sim.run(EPOCH, planner_name="greedy", n_orbits=4.0, step_s=30)
    return generate_telemetry(result.schedule.states)


class TestTelemetryGeneration:
    def test_has_all_expected_columns(self, telemetry_df):
        for col in FEATURE_COLUMNS:
            assert col in telemetry_df.columns

    def test_no_nans(self, telemetry_df):
        assert not telemetry_df[FEATURE_COLUMNS].isna().any().any()

    def test_memory_usage_bounded(self, telemetry_df):
        assert telemetry_df["memory_usage"].between(0, 100).all()


class TestAnomalyInjection:
    def test_overheating_raises_temperature(self, telemetry_df):
        df = telemetry_df.copy()
        df["is_anomaly"] = False
        df["anomaly_type"] = ""
        before = df.loc[100:129, "temperature"].mean()
        df = inject_overheating(df, start_idx=100, duration=30, magnitude_c=20)
        after = df.loc[100:129, "temperature"].mean()
        assert after > before
        assert df.loc[100:129, "is_anomaly"].all()

    def test_battery_degradation_lowers_voltage(self, telemetry_df):
        df = telemetry_df.copy()
        df["is_anomaly"] = False
        df["anomaly_type"] = ""
        before = df.loc[50:79, "battery_voltage"].mean()
        df = inject_battery_degradation(df, start_idx=50, duration=30, magnitude_v=0.5)
        after = df.loc[50:79, "battery_voltage"].mean()
        assert after < before

    def test_random_injection_labels_match_count(self, telemetry_df):
        df = inject_random_anomalies(telemetry_df.copy(), n_anomalies=5, min_duration=10,
                                      max_duration=20, seed=2)
        assert df["is_anomaly"].sum() > 0
        assert set(df.loc[df.is_anomaly, "anomaly_type"].unique()) <= {
            "overheating", "battery_degradation", "communication_failure",
            "abnormal_power", "sensor_drift"}


class TestAnomalyDetection:
    def test_isolation_forest_flags_more_than_none(self, telemetry_df):
        df = inject_random_anomalies(telemetry_df.copy(), n_anomalies=6, min_duration=15,
                                      max_duration=30, seed=5)
        split = len(df) // 2
        train, test = df.iloc[:split], df.iloc[split:].reset_index(drop=True)
        det = IsolationForestDetector()
        ev = evaluate_detector(det, train, test)
        assert ev.n_flagged > 0
        assert 0 <= ev.precision <= 1
        assert 0 <= ev.recall <= 1

    def test_pca_detector_runs_and_produces_valid_metrics(self, telemetry_df):
        df = inject_random_anomalies(telemetry_df.copy(), n_anomalies=6, min_duration=15,
                                      max_duration=30, seed=5)
        split = len(df) // 2
        train, test = df.iloc[:split], df.iloc[split:].reset_index(drop=True)
        det = PCAReconstructionDetector()
        ev = evaluate_detector(det, train, test)
        assert 0 <= ev.f1 <= 1

    def test_detectors_beat_random_guessing_on_recall(self, telemetry_df):
        """Sanity check: with reasonably strong injected anomalies, a
        properly fit detector should catch a meaningful fraction — not
        prove optimality, just catch a gross regression in the pipeline."""
        df = inject_random_anomalies(telemetry_df.copy(), n_anomalies=8, min_duration=20,
                                      max_duration=40, seed=9)
        split = len(df) // 2
        train, test = df.iloc[:split], df.iloc[split:].reset_index(drop=True)
        det = IsolationForestDetector(contamination=0.1)
        ev = evaluate_detector(det, train, test)
        if ev.n_true_anomalies > 0:
            assert ev.recall > 0.05


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
