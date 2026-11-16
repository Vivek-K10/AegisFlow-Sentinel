"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: models/ensemble_detector.py
Description: Weighted soft-voting and consensus ensemble detector fusing Isolation Forest,
Deep Autoencoder, and One-Class SVM with severity tiering (NORMAL, LOW, MEDIUM, HIGH, CRITICAL).
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, List, Optional, Tuple
from dataset_generator import FEATURE_NAMES
from .isolation_forest_model import IsolationForestDetector
from .autoencoder_model import AutoencoderDetector
from .one_class_svm_model import OneClassSVMDetector


class EnsembleDetector:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        consensus_threshold: float = 0.45,
        contamination: float = 0.05
    ):
        self.weights = weights or {
            "isolation_forest": 0.35,
            "autoencoder": 0.40,
            "one_class_svm": 0.25
        }
        self.consensus_threshold = consensus_threshold
        self.contamination = contamination

        self.iso_forest = IsolationForestDetector(contamination=self.contamination)
        self.autoencoder = AutoencoderDetector(input_dim=len(FEATURE_NAMES), bottleneck_dim=6)
        self.oc_svm = OneClassSVMDetector(nu=self.contamination)

        self.is_fitted = False
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.baseline_sample_coords: List[Dict[str, float]] = []

    def _extract_features(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """Extracts and orders the 14 NSL-KDD/CICIDS features."""
        if isinstance(X, dict):
            return np.array([[float(X.get(f, 0.0)) for f in FEATURE_NAMES]], dtype=np.float64)
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            rows = [[float(row.get(f, 0.0)) for f in FEATURE_NAMES] for row in X]
            return np.array(rows, dtype=np.float64)
        elif isinstance(X, pd.DataFrame):
            # Select only existing feature names or all if matching length
            available_cols = [c for c in FEATURE_NAMES if c in X.columns]
            if len(available_cols) == len(FEATURE_NAMES):
                return X[FEATURE_NAMES].values.astype(np.float64)
            return X.iloc[:, :len(FEATURE_NAMES)].values.astype(np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        epochs: int = 25,
        sample_size: Optional[int] = None
    ) -> "EnsembleDetector":
        """Trains all 3 unsupervised detectors on baseline normal traffic."""
        X_arr = self._extract_features(X)
        if sample_size and sample_size < len(X_arr):
            indices = np.random.choice(len(X_arr), size=sample_size, replace=False)
            X_arr = X_arr[indices]

        # Calculate and store baseline statistics for XAI attribution
        self.baseline_stats = {}
        for idx, feat in enumerate(FEATURE_NAMES):
            vals = X_arr[:, idx]
            self.baseline_stats[feat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if np.std(vals) > 1e-6 else 1.0,
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals))
            }

        # Train individual models
        self.iso_forest.train(X_arr)
        self.autoencoder.train(X_arr, epochs=epochs)
        
        # Subsample for SVM if dataset is large (SVM O(N^2) complexity)
        if len(X_arr) > 3000:
            svm_indices = np.random.choice(len(X_arr), size=3000, replace=False)
            self.oc_svm.train(X_arr[svm_indices])
        else:
            self.oc_svm.train(X_arr)

        # Cache a small sample of normal baseline 2D coordinates for background scatter plot
        try:
            sub_normal = X_arr[:150]
            coords = self.autoencoder.extract_latent_2d(sub_normal)
            self.baseline_sample_coords = [
                {"x": round(float(c[0]), 3), "y": round(float(c[1]), 3), "label": "Normal Baseline"}
                for c in coords
            ]
        except Exception:
            self.baseline_sample_coords = []

        self.is_fitted = True
        return self

    def score_all_models(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Runs inference across all 3 models and returns scores + predictions."""
        if not self.is_fitted:
            raise RuntimeError("EnsembleDetector is not fitted yet.")

        X_arr = self._extract_features(X)
        iso_scores = self.iso_forest.score_anomaly(X_arr)
        iso_preds = self.iso_forest.predict(X_arr)

        ae_scores = self.autoencoder.score_anomaly(X_arr)
        ae_preds = self.autoencoder.predict(X_arr)

        svm_scores = self.oc_svm.score_anomaly(X_arr)
        svm_preds = self.oc_svm.predict(X_arr)

        # Weighted fusion score
        w_iso = self.weights.get("isolation_forest", 0.35)
        w_ae = self.weights.get("autoencoder", 0.40)
        w_svm = self.weights.get("one_class_svm", 0.25)
        w_total = w_iso + w_ae + w_svm

        ensemble_scores = (w_iso * iso_scores + w_ae * ae_scores + w_svm * svm_scores) / w_total

        # Consensus decision: flagged if weighted score >= threshold OR 2+ models agree
        votes = iso_preds + ae_preds + svm_preds
        ensemble_preds = ((ensemble_scores >= self.consensus_threshold) | (votes >= 2)).astype(int)

        return {
            "iso_scores": iso_scores,
            "iso_preds": iso_preds,
            "ae_scores": ae_scores,
            "ae_preds": ae_preds,
            "svm_scores": svm_scores,
            "svm_preds": svm_preds,
            "ensemble_scores": ensemble_scores,
            "ensemble_preds": ensemble_preds,
            "votes": votes
        }

    def determine_severity(self, score: float, is_anomaly: bool) -> str:
        """Determines cybersecurity severity tier based on anomaly score and consensus."""
        if not is_anomaly and score < 0.40:
            return "NORMAL"
        if score < 0.55:
            return "LOW"
        elif score < 0.72:
            return "MEDIUM"
        elif score < 0.88:
            return "HIGH"
        else:
            return "CRITICAL"

    def evaluate_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates a single network flow dictionary for real-time telemetry."""
        results = self.score_all_models(flow)
        score = float(results["ensemble_scores"][0])
        is_anomaly = bool(results["ensemble_preds"][0])
        votes = int(results["votes"][0])
        severity = self.determine_severity(score, is_anomaly)

        # 2D latent projection
        X_arr = self._extract_features(flow)
        try:
            coords = self.autoencoder.extract_latent_2d(X_arr)
            latent_x = round(float(coords[0][0]), 3)
            latent_y = round(float(coords[0][1]), 3)
        except Exception:
            latent_x, latent_y = 0.0, 0.0

        return {
            "ensemble_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "severity": severity,
            "model_agreement_votes": votes,
            "isolation_forest_score": round(float(results["iso_scores"][0]), 4),
            "autoencoder_score": round(float(results["ae_scores"][0]), 4),
            "one_class_svm_score": round(float(results["svm_scores"][0]), 4),
            "latent_coords": {"x": latent_x, "y": latent_y}
        }

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts binary anomaly labels (0 for normal, 1 for anomaly) for a batch."""
        res = self.score_all_models(X)
        return res["ensemble_preds"]

    def score_ensemble(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns continuous fused anomaly score between 0.0 and 1.0."""
        res = self.score_all_models(X)
        return res["ensemble_scores"]

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_fitted": self.is_fitted,
            "consensus_threshold": self.consensus_threshold,
            "weights": self.weights,
            "models": {
                "isolation_forest": self.iso_forest.get_info(),
                "autoencoder": self.autoencoder.get_info(),
                "one_class_svm": self.oc_svm.get_info()
            },
            "loss_history": self.autoencoder.loss_history
        }
