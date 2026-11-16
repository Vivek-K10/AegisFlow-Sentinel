"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: models/autoencoder_model.py
Description: Deep Neural Network Autoencoder built with TensorFlow/Keras.
Architecture: Input (14) -> Dense 12 -> Bottleneck 6 -> Dense 12 -> Output (14).
Classification based on dynamic 93rd-percentile MSE threshold, with PCA 2D latent projection.
"""

import os
import sys

# Ensure Keras config & cache stay inside sandbox workspace
_workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_keras_cache_dir = os.path.join(_workspace_dir, ".keras_cache")
os.makedirs(_keras_cache_dir, exist_ok=True)
os.environ["KERAS_HOME"] = _keras_cache_dir
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Union, Dict, Any, List, Tuple


class AutoencoderDetector:
    def __init__(self, input_dim: int = 14, bottleneck_dim: int = 6, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.pca_2d = PCA(n_components=2, random_state=42)
        
        self.autoencoder: Optional[Model] = None
        self.encoder_model: Optional[Model] = None
        self.reconstruction_threshold: float = 0.0
        self.loss_history: List[float] = []
        self.is_fitted: bool = False
        self._build_architecture()

    def _build_architecture(self):
        """Constructs Autoencoder and Sub-Encoder models."""
        tf.keras.backend.clear_session()
        
        inputs = layers.Input(shape=(self.input_dim,), name="flow_input")
        enc1 = layers.Dense(12, activation="relu", name="enc_layer_1")(inputs)
        bottleneck = layers.Dense(self.bottleneck_dim, activation="relu", name="bottleneck_latent")(enc1)
        dec1 = layers.Dense(12, activation="relu", name="dec_layer_1")(bottleneck)
        outputs = layers.Dense(self.input_dim, activation="linear", name="reconstructed_flow")(dec1)

        self.autoencoder = Model(inputs=inputs, outputs=outputs, name="AegisAutoencoder")
        self.encoder_model = Model(inputs=inputs, outputs=bottleneck, name="AegisLatentExtractor")

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.autoencoder.compile(optimizer=optimizer, loss="mse")

    def _ensure_numpy(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        return np.asarray(X, dtype=np.float64)

    def train(self, X: Union[np.ndarray, pd.DataFrame], epochs: int = 25, batch_size: int = 64) -> "AutoencoderDetector":
        """Trains the Autoencoder on normal baseline network flows."""
        X_arr = self._ensure_numpy(X)
        X_scaled = self.scaler.fit_transform(X_arr)

        # Train Autoencoder
        history = self.autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        self.loss_history = [float(loss) for loss in history.history.get("loss", [])]

        # Calculate dynamic 93rd-percentile reconstruction error threshold
        reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
        mse_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.reconstruction_threshold = float(np.percentile(mse_errors, 93))

        # Fit PCA on bottleneck latent representations for 2D scatter visualization
        latents = self.encoder_model.predict(X_scaled, verbose=0)
        self.pca_2d.fit(latents)

        self.is_fitted = True
        return self

    def compute_mse(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes per-sample Mean Squared Reconstruction Error."""
        if not self.is_fitted:
            raise RuntimeError("AutoencoderDetector must be trained before inference.")
        X_arr = self._ensure_numpy(X)
        X_scaled = self.scaler.transform(X_arr)
        reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
        mse_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        return mse_errors

    def score_anomaly(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Calculates normalized anomaly score (0.0 to 1.0) based on reconstruction MSE.
        Values above threshold map smoothly towards 1.0.
        """
        mse_errors = self.compute_mse(X)
        # Scaled relative to dynamic 93rd-percentile threshold
        ratio = mse_errors / max(self.reconstruction_threshold, 1e-6)
        # Sigmoid calibration centered around threshold (ratio = 1.0 -> score ~ 0.50)
        scores = 1.0 / (1.0 + np.exp(-2.8 * (ratio - 1.0)))
        return np.clip(scores, 0.0, 1.0)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns 1 if sample reconstruction MSE exceeds 93rd-percentile threshold, 0 otherwise."""
        mse_errors = self.compute_mse(X)
        return (mse_errors >= self.reconstruction_threshold).astype(int)

    def extract_latent_2d(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Passes network flows through bottleneck layer and projects onto 2D latent PCA
        coordinates for real-time SOC scatter visualizer.
        """
        if not self.is_fitted:
            raise RuntimeError("AutoencoderDetector must be trained before latent extraction.")
        X_arr = self._ensure_numpy(X)
        X_scaled = self.scaler.transform(X_arr)
        latents = self.encoder_model.predict(X_scaled, verbose=0)
        coords_2d = self.pca_2d.transform(latents)
        return coords_2d

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Deep Autoencoder",
            "type": "Neural Network Reconstruction",
            "threshold_93rd_percentile_mse": round(self.reconstruction_threshold, 5),
            "loss_history_last": round(self.loss_history[-1], 5) if self.loss_history else None,
            "bottleneck_dim": self.bottleneck_dim,
            "is_fitted": self.is_fitted
        }
