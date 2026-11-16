"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: models/one_class_svm_model.py
Description: One-Class SVM with RBF kernel to learn non-linear boundaries around normal traffic,
computing signed hyperplane distance scores and normalized anomaly probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Union, Dict, Any


class OneClassSVMDetector:
    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.is_fitted = False
        self.baseline_dist_mean = 0.0
        self.baseline_dist_std = 1.0

    def _ensure_numpy(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        return np.asarray(X, dtype=np.float64)

    def train(self, X: Union[np.ndarray, pd.DataFrame]) -> "OneClassSVMDetector":
        """Fits standard scaler and One-Class SVM on normal baseline flows."""
        X_arr = self._ensure_numpy(X)
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled)
        
        # Decision function: positive for normal inliers, negative for outliers
        decisions = self.model.decision_function(X_scaled)
        self.baseline_dist_mean = float(np.mean(decisions))
        self.baseline_dist_std = float(np.std(decisions)) if np.std(decisions) > 1e-4 else 1.0
        self.is_fitted = True
        return self

    def get_hyperplane_distances(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns raw signed hyperplane distances from decision_function."""
        if not self.is_fitted:
            raise RuntimeError("OneClassSVMDetector must be trained before inference.")
        X_arr = self._ensure_numpy(X)
        X_scaled = self.scaler.transform(X_arr)
        return self.model.decision_function(X_scaled)

    def score_anomaly(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Calculates normalized anomaly score between 0.0 and 1.0.
        Negative distance (outside support boundary) yields higher anomaly scores.
        """
        distances = self.get_hyperplane_distances(X)
        # Invert: negative distance -> higher anomaly
        anomaly_metric = -distances
        # Sigmoid calibrated such that distance=0 (the decision boundary) maps to ~0.50
        scale_factor = max(self.baseline_dist_std, 0.5)
        scores = 1.0 / (1.0 + np.exp(-1.8 * (anomaly_metric / scale_factor)))
        return np.clip(scores, 0.0, 1.0)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns 1 for anomaly (hyperplane distance < 0), 0 for normal inlier."""
        distances = self.get_hyperplane_distances(X)
        return (distances < 0.0).astype(int)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "One-Class SVM",
            "type": "Kernel Hyperplane Boundary (RBF)",
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "is_fitted": self.is_fitted
        }
