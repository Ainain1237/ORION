"""
ai/anomaly_detection.py

Two unsupervised anomaly detectors trained on telemetry, evaluated
against the ground-truth labels telemetry.anomalies injects
(Experiment 2 in the design doc: precision / recall / F1 / detection
latency).

    1. IsolationForestDetector — scikit-learn's Isolation Forest,
       an ensemble of random partitioning trees; anomalies are points
       that get isolated in fewer splits than normal points.

    2. PCAReconstructionDetector — a linear-autoencoder equivalent:
       fit PCA on (assumed-mostly-normal) training telemetry, then flag
       points whose reconstruction error (distance from the PCA
       subspace) exceeds a threshold. This is deliberately the *linear*
       special case of an autoencoder — a genuine deep autoencoder
       (nonlinear encoder/decoder, e.g. a small PyTorch MLP) is a
       natural extension once this baseline is established, and the
       same evaluation harness below would apply unchanged.

Both share the same fit/score/predict interface so they can be swapped
in `evaluate_detector` without touching the evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "battery_voltage", "battery_current", "temperature",
    "solar_power", "memory_usage", "reaction_wheel_speed",
]


@dataclass
class DetectorEvaluation:
    detector_name: str
    precision: float
    recall: float
    f1: float
    mean_detection_latency_steps: float
    n_true_anomalies: int
    n_flagged: int
    fit_time_s: float
    predict_time_s: float


class IsolationForestDetector:
    name = "Isolation Forest"

    def __init__(self, contamination: float = 0.05, n_estimators: int = 150, seed: int = 0):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                                      random_state=seed)

    def fit(self, df: pd.DataFrame):
        X = self.scaler.fit_transform(df[FEATURE_COLUMNS].values)
        self.model.fit(X)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns boolean array: True where flagged as anomalous."""
        X = self.scaler.transform(df[FEATURE_COLUMNS].values)
        preds = self.model.predict(X)   # -1 = anomaly, 1 = normal
        return preds == -1

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[FEATURE_COLUMNS].values)
        return -self.model.score_samples(X)  # higher = more anomalous


class PCAReconstructionDetector:
    name = "PCA Reconstruction (linear autoencoder)"

    def __init__(self, n_components: int = 3, threshold_std: float = 3.0):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.threshold_std = threshold_std
        self.threshold_ = None

    def fit(self, df: pd.DataFrame):
        X = self.scaler.fit_transform(df[FEATURE_COLUMNS].values)
        self.pca.fit(X)
        recon = self.pca.inverse_transform(self.pca.transform(X))
        errors = np.mean((X - recon) ** 2, axis=1)
        self.threshold_ = errors.mean() + self.threshold_std * errors.std()
        return self

    def _reconstruction_error(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[FEATURE_COLUMNS].values)
        recon = self.pca.inverse_transform(self.pca.transform(X))
        return np.mean((X - recon) ** 2, axis=1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        errors = self._reconstruction_error(df)
        return errors > self.threshold_

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        return self._reconstruction_error(df)


DETECTORS = {
    "isolation_forest": IsolationForestDetector,
    "pca": PCAReconstructionDetector,
}


def evaluate_detector(detector, train_df: pd.DataFrame, test_df: pd.DataFrame) -> DetectorEvaluation:
    """
    Fits `detector` on train_df (assumed mostly/entirely normal),
    predicts on test_df (which has ground-truth is_anomaly labels from
    telemetry.anomalies), and computes precision/recall/F1 plus mean
    detection latency: for each true anomaly window, how many steps
    after its start did the detector first flag it (NaN-excluded if
    never detected).
    """
    t0 = time.time()
    detector.fit(train_df)
    fit_time = time.time() - t0

    t0 = time.time()
    flagged = detector.predict(test_df)
    predict_time = time.time() - t0

    truth = test_df["is_anomaly"].values.astype(bool)

    tp = int(np.sum(flagged & truth))
    fp = int(np.sum(flagged & ~truth))
    fn = int(np.sum(~flagged & truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    latency = _mean_detection_latency(truth, flagged)

    return DetectorEvaluation(
        detector_name=detector.name, precision=precision, recall=recall, f1=f1,
        mean_detection_latency_steps=latency, n_true_anomalies=int(truth.sum()),
        n_flagged=int(flagged.sum()), fit_time_s=fit_time, predict_time_s=predict_time,
    )


def _mean_detection_latency(truth: np.ndarray, flagged: np.ndarray) -> float:
    """
    Finds contiguous True runs in `truth` (anomaly windows) and measures
    how many steps into each window the detector first raises a flag.
    Windows never detected are excluded (reported separately via recall).
    """
    latencies = []
    n = len(truth)
    i = 0
    while i < n:
        if truth[i]:
            start = i
            while i < n and truth[i]:
                i += 1
            end = i  # exclusive
            window_flags = flagged[start:end]
            hit_indices = np.nonzero(window_flags)[0]
            if len(hit_indices) > 0:
                latencies.append(int(hit_indices[0]))
        else:
            i += 1
    return float(np.mean(latencies)) if latencies else float("nan")
