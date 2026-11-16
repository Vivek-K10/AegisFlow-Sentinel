"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: app.py
Description: Production-ready Flask REST API backend providing real-time telemetry streaming,
unsupervised inference, model retraining, CSV upload processing, and forensic XAI diagnostics.
"""

import sys
import os
import io
import time
import logging

# Configure in-workspace sandbox cache paths for Keras/TensorFlow/Matplotlib
_base_dir = os.path.dirname(os.path.abspath(__file__))
_cache_dir = os.path.join(_base_dir, ".keras_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["KERAS_HOME"] = _cache_dir
os.environ["MPLCONFIGDIR"] = _cache_dir
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set UTF-8 encoding for Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataset_generator import (
    generate_baseline_dataset,
    generate_test_dataset,
    generate_stream_packet,
    FEATURE_NAMES,
    ATTACK_TYPES
)
from models.ensemble_detector import EnsembleDetector
from explainability import XAIAttributionEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] AegisFlow Sentinel: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AegisFlow")

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
CORS(app)

# Global Sentinel State
sentinel_state = {
    "is_initialized": False,
    "is_training": False,
    "ensemble": None,
    "xai": None,
    "last_trained_timestamp": None,
    "training_samples": 2500,
    "epochs": 20,
    "contamination": 0.05
}


def initialize_sentinel(n_samples: int = 2500, epochs: int = 20, contamination: float = 0.05):
    """Initializes and trains the unsupervised baseline models."""
    global sentinel_state
    sentinel_state["is_training"] = True
    logger.info(f"Initializing baseline with {n_samples} normal flows, {epochs} epochs, nu={contamination}...")

    df_train = generate_baseline_dataset(n_samples=n_samples, seed=42)
    ensemble = EnsembleDetector(contamination=contamination)
    ensemble.train(df_train, epochs=epochs)

    xai = XAIAttributionEngine(ensemble.baseline_stats)

    sentinel_state["ensemble"] = ensemble
    sentinel_state["xai"] = xai
    sentinel_state["is_initialized"] = True
    sentinel_state["is_training"] = False
    sentinel_state["last_trained_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    sentinel_state["training_samples"] = n_samples
    sentinel_state["epochs"] = epochs
    sentinel_state["contamination"] = contamination
    logger.info("AegisFlow Sentinel successfully initialized and armed.")


# Warm up models upon server launch
with app.app_context():
    try:
        initialize_sentinel(n_samples=2000, epochs=15, contamination=0.05)
    except Exception as e:
        logger.error(f"Error initializing sentinel: {e}", exc_info=True)


@app.route("/")
def index():
    """Serves the Security Operations Center (SOC) dashboard."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """System health, model readiness, and autoencoder loss history."""
    ensemble = sentinel_state.get("ensemble")
    if not ensemble or not sentinel_state["is_initialized"]:
        return jsonify({
            "status": "INITIALIZING",
            "is_ready": False,
            "message": "Models are currently training..."
        }), 503

    status_data = ensemble.get_status()
    return jsonify({
        "status": "OPERATIONAL",
        "is_ready": True,
        "is_training": sentinel_state["is_training"],
        "last_trained": sentinel_state["last_trained_timestamp"],
        "training_samples": sentinel_state["training_samples"],
        "epochs": sentinel_state["epochs"],
        "contamination": sentinel_state["contamination"],
        "models": status_data["models"],
        "loss_history": status_data["loss_history"],
        "baseline_cluster_samples": ensemble.baseline_sample_coords[:60]
    })


@app.route("/api/traffic/stream", methods=["GET"])
def stream_traffic():
    """
    Emits a single real-time telemetry flow with optional attack injection.
    Query param `attack`: none, ddos, port_scan, exfiltration, c2_beacon.
    """
    ensemble = sentinel_state.get("ensemble")
    xai = sentinel_state.get("xai")
    if not ensemble or not sentinel_state["is_initialized"]:
        return jsonify({"error": "Sentinel not ready"}), 503

    attack_param = request.args.get("attack", "none")
    flow_packet = generate_stream_packet(attack=attack_param)

    # Inference & Evaluation
    eval_result = ensemble.evaluate_flow(flow_packet)
    explanation = xai.explain(flow_packet, top_k=5)

    return jsonify({
        "timestamp": time.strftime("%H:%M:%S"),
        "flow": flow_packet,
        "detection": eval_result,
        "xai": explanation
    })


@app.route("/api/detect", methods=["POST"])
def detect_flow():
    """Runs inference on a custom packet JSON payload."""
    ensemble = sentinel_state.get("ensemble")
    xai = sentinel_state.get("xai")
    if not ensemble or not sentinel_state["is_initialized"]:
        return jsonify({"error": "Sentinel not ready"}), 503

    data = request.get_json(force=True, silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON flow payload provided"}), 400

    eval_result = ensemble.evaluate_flow(data)
    explanation = xai.explain(data, top_k=5)

    return jsonify({
        "status": "SUCCESS",
        "detection": eval_result,
        "xai": explanation
    })


@app.route("/api/train", methods=["POST"])
def retrain_models():
    """Retrains the baseline models with user parameters."""
    if sentinel_state["is_training"]:
        return jsonify({"error": "A training session is already in progress"}), 409

    body = request.get_json(force=True, silent=True) or {}
    sample_size = int(body.get("sample_size", 2500))
    sample_size = max(500, min(sample_size, 10000))

    contamination = float(body.get("contamination", 0.05))
    contamination = max(0.01, min(contamination, 0.20))

    epochs = int(body.get("epochs", 20))
    epochs = max(5, min(epochs, 60))

    try:
        initialize_sentinel(n_samples=sample_size, epochs=epochs, contamination=contamination)
        return jsonify({
            "status": "SUCCESS",
            "message": f"Retrained AegisFlow baseline with {sample_size} samples across {epochs} epochs.",
            "last_trained": sentinel_state["last_trained_timestamp"],
            "loss_history": sentinel_state["ensemble"].autoencoder.loss_history
        })
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """Parses and analyzes uploaded network capture CSV files."""
    ensemble = sentinel_state.get("ensemble")
    xai = sentinel_state.get("xai")
    if not ensemble or not sentinel_state["is_initialized"]:
        return jsonify({"error": "Sentinel not ready"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No CSV file provided in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        # Check required features
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            return jsonify({
                "error": f"Uploaded CSV missing required features: {', '.join(missing_features)}"
            }), 400

        # Cap processing at 3000 rows for real-time responsiveness
        df_eval = df.head(3000).copy()
        X_eval = df_eval[FEATURE_NAMES].values

        scores_dict = ensemble.score_all_models(X_eval)
        df_eval["anomaly_score"] = scores_dict["ensemble_scores"].round(4)
        df_eval["is_anomaly"] = scores_dict["ensemble_preds"]

        total_rows = len(df_eval)
        anomaly_count = int(df_eval["is_anomaly"].sum())
        normal_count = total_rows - anomaly_count
        anomaly_rate = round((anomaly_count / max(total_rows, 1)) * 100.0, 2)

        # Classify severities
        severities = [
            ensemble.determine_severity(s, bool(p))
            for s, p in zip(scores_dict["ensemble_scores"], scores_dict["ensemble_preds"])
        ]
        df_eval["severity"] = severities

        # Sample preview (first 50 records)
        sample_results = []
        for i in range(min(50, total_rows)):
            row = df_eval.iloc[i].to_dict()
            sample_results.append({
                "row_index": i + 1,
                "protocol": row.get("protocol", "TCP"),
                "duration": row.get("duration", 0),
                "src_bytes": row.get("src_bytes", 0),
                "dst_bytes": row.get("dst_bytes", 0),
                "packet_count": row.get("packet_count", 0),
                "anomaly_score": row["anomaly_score"],
                "is_anomaly": bool(row["is_anomaly"]),
                "severity": row["severity"]
            })

        return jsonify({
            "status": "SUCCESS",
            "filename": file.filename,
            "total_records": total_rows,
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_rate_pct": anomaly_rate,
            "records_preview": sample_results
        })

    except Exception as e:
        logger.error(f"Error processing CSV: {e}", exc_info=True)
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 500


@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    """Returns comparative accuracy, precision, recall, and confusion matrix across models."""
    ensemble = sentinel_state.get("ensemble")
    if not ensemble or not sentinel_state["is_initialized"]:
        return jsonify({"error": "Sentinel not ready"}), 503

    # Generate benchmark test dataset
    df_test = generate_test_dataset(n_normal=800, n_per_attack=75, seed=123)
    y_true = df_test["is_anomaly"].values
    X_test = df_test[FEATURE_NAMES].values

    # Run predictions
    scores_dict = ensemble.score_all_models(X_test)

    benchmarks = {}
    models_mapping = {
        "isolation_forest": ("Isolation Forest", scores_dict["iso_preds"]),
        "autoencoder": ("Deep Autoencoder", scores_dict["ae_preds"]),
        "one_class_svm": ("One-Class SVM", scores_dict["svm_preds"]),
        "ensemble": ("AegisFlow Sentinel (Ensemble)", scores_dict["ensemble_preds"])
    }

    for key, (label, y_pred) in models_mapping.items():
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        benchmarks[key] = {
            "name": label,
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": {
                "tn": cm[0][0],
                "fp": cm[0][1],
                "fn": cm[1][0],
                "tp": cm[1][1]
            }
        }

    return jsonify({
        "status": "SUCCESS",
        "benchmark_samples": len(df_test),
        "normal_samples": int(np.sum(y_true == 0)),
        "attack_samples": int(np.sum(y_true == 1)),
        "benchmarks": benchmarks
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[+] AegisFlow Sentinel Web SOC running on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
