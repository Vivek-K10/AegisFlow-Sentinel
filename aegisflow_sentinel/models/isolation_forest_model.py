"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: models/isolation_forest_model.py
Description: Scikit-Learn IsolationForest with contamination control and
calibrated normalized anomaly scoring (0.0 to 1.0).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from typing import Union, Tuple, Dict, Any


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 150, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.baseline_score_min = 0.0
        self.baseline_score_max = 1.0

    def _ensure_numpy(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        return np.asarray(X, dtype=np.float64)

    def train(self, X: Union[np.ndarray, pd.DataFrame]) -> "IsolationForestDetector":
        """Fits scaler and Isolation Forest on normal baseline traffic."""
        X_arr = self._ensure_numpy(X)
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled)
        
        # Raw scores: lower score_samples means more anomalous
        # We invert so higher means more anomalous
        raw_scores = -self.model.score_samples(X_scaled)
        self.baseline_score_min = float(np.percentile(raw_scores, 1))
        self.baseline_score_max = float(np.percentile(raw_scores, 99))
        self.is_fitted = True
        return self

    def score_anomaly(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes calibrated anomaly score between 0.0 and 1.0.
        0.0 = completely normal, 1.0 = highly anomalous zero-day/intrusion.
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector must be trained before scoring.")
        
        X_arr = self._ensure_numpy(X)
        X_scaled = self.scaler.transform(X_arr)
        raw_scores = -self.model.score_samples(X_scaled)
        
        # Calibrate using baseline range and logistic soft clip
        spread = max(self.baseline_score_max - self.baseline_score_min, 1e-6)
        normalized = (raw_scores - self.baseline_score_min) / spread
        
        # Soft sigmoid squashing for extreme outliers
        calibrated = 1.0 / (1.0 + np.exp(-3.5 * (normalized - 0.75)))
        return np.clip(calibrated, 0.0, 1.0)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.50) -> np.ndarray:
        """Returns binary predictions: 1 for anomaly, 0 for normal."""
        scores = self.score_anomaly(X)
        return (scores >= threshold).astype(int)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Isolation Forest",
            "type": "Tree-based Partitioning",
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "is_fitted": self.is_fitted
        }
