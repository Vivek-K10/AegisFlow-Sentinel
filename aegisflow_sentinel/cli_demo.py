"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: cli_demo.py
Description: Standalone Terminal CLI utility demonstrating unsupervised training,
real-time attack detection, Explainable AI attribution, and multi-model benchmark evaluation.
"""

import sys
import os
import time

# Configure in-workspace sandbox cache paths for Keras/TensorFlow/Matplotlib
_base_dir = os.path.dirname(os.path.abspath(__file__))
_cache_dir = os.path.join(_base_dir, ".keras_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["KERAS_HOME"] = _cache_dir
os.environ["MPLCONFIGDIR"] = _cache_dir
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure clean UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataset_generator import (
    generate_baseline_dataset,
    generate_test_dataset,
    generate_stream_packet,
    ATTACK_TYPES,
    FEATURE_NAMES
)
from models.ensemble_detector import EnsembleDetector
from explainability import XAIAttributionEngine


def print_banner():
    banner = r"""
================================================================================
   ___              _      ____ _                 ____             _   _            _ 
  / _ \             (_)    |  __| |               / ___|           | | (_)          | |
 / /_\ \ ___   __ _  _ ___ | |_ | | _____      __ \___ \  ___ _ __ | |_ _ _ __   ___| |
 |  _  |/ _ \ / _` || / __||  _|| |/ _ \ \ /\ / /  ___) |/ _ \ '_ \| __| | '_ \ / _ \ |
 | | | |  __/| (_| || \__ \| |  | | (_) \ V  V /  |____/|  __/ | | | |_| | | | |  __/ |
 \_| |_/\___| \__, ||_|___/\_|  |_|\___/ \_/\_/         \___|_| |_|\__|_|_| |_|\___|_|
               __/ |                                                                  
              |___/      Autonomous Zero-Day Network Intrusion Sentinel               
================================================================================
    [Unsupervised ML Suite: Isolation Forest + Autoencoder + One-Class SVM]
"""
    print(banner)


def run_cli_demonstration():
    print_banner()
    print("[+] Step 1: Synthesizing Normal Network Baseline Traffic...")
    start_time = time.time()
    df_train = generate_baseline_dataset(n_samples=2500, seed=42)
    print(f"    -> Generated {len(df_train)} normal baseline flows across HTTP, HTTPS, DNS, SSH, and API.")
    
    print("\n[+] Step 2: Training Unsupervised Machine Learning Suite...")
    print("    [1/3] Fitting Robust Isolation Forest (contamination=0.05)...")
    print("    [2/3] Training Deep Keras Autoencoder (14 -> 12 -> 6 -> 12 -> 14, 20 epochs)...")
    print("    [3/3] Constructing One-Class SVM RBF Hyperplane Boundary...")
    
    ensemble = EnsembleDetector(contamination=0.05)
    ensemble.train(df_train, epochs=20)
    train_time = time.time() - start_time
    print(f"    [✔] Training Complete in {train_time:.2f} seconds.")
    print(f"    [✔] Dynamic 93rd-percentile Autoencoder Reconstruction MSE Threshold: {ensemble.autoencoder.reconstruction_threshold:.5f}")

    xai = XAIAttributionEngine(ensemble.baseline_stats)

    print("\n" + "=" * 80)
    print("         LIVE TELEMETRY & ATTACK INJECTION SIMULATION")
    print("=" * 80)

    test_scenarios = [
        ("Normal Flow (HTTPS API)", None),
        ("Cyber Threat Scenario 1: DDoS SYN Flood", "ddos_syn_flood"),
        ("Cyber Threat Scenario 2: Port Scan Recon Sweep", "port_scan_sweep"),
        ("Cyber Threat Scenario 3: High-Volume Data Exfiltration", "data_exfiltration"),
        ("Cyber Threat Scenario 4: Stealthy Zero-Day C2 Beacon", "zero_day_c2_beacon"),
    ]

    for title, attack_code in test_scenarios:
        packet = generate_stream_packet(attack_code)
        eval_result = ensemble.evaluate_flow(packet)
        explanation = xai.explain(packet, top_k=4)

        score = eval_result["ensemble_score"]
        status = "ANOMALY FLAGGED" if eval_result["is_anomaly"] else "NORMAL (BENIGN)"
        severity = eval_result["severity"]

        print(f"\n[Test Vector] {title}")
        print(f"  Protocol: {packet['protocol']} | True Label: {packet['attack_type']}")
        print(f"  Ensemble Anomaly Score : {score:.4f} [Threshold: {ensemble.consensus_threshold}]")
        print(f"  Verdict & Severity     : [{status}] - Severity: {severity}")
        print(f"  Multi-Model Consensus  : {eval_result['model_agreement_votes']}/3 Models Agreed")
        print(f"    - Isolation Forest   : {eval_result['isolation_forest_score']:.4f}")
        print(f"    - Autoencoder MSE    : {eval_result['autoencoder_score']:.4f}")
        print(f"    - One-Class SVM      : {eval_result['one_class_svm_score']:.4f}")
        print(f"    - Latent 2D Coords   : ({eval_result['latent_coords']['x']}, {eval_result['latent_coords']['y']})")
        
        print("  XAI Forensic Attribution:")
        print(f"    Summary: {explanation['explanation_summary']}")
        if explanation['threat_indicators']:
            for ind in explanation['threat_indicators']:
                print(f"    Indicator: [!] {ind}")
        print("    Top Deviating Features:")
        for item in explanation["top_contributors"][:3]:
            print(f"      * {item['display_name']:<30}: Val={item['observed_value']:<10.2f} (Base={item['baseline_mean']:<8.2f}, Z={item['z_score']:+6.2f}σ, Impact={item['contribution_pct']}%)")

    print("\n" + "=" * 80)
    print("         MULTI-ALGORITHM BENCHMARK PERFORMANCE EVALUATION")
    print("=" * 80)
    print("[+] Synthesizing Mixed Ground-Truth Benchmark Test Set (1000 Normal + 400 Attacks)...")
    df_test = generate_test_dataset(n_normal=1000, n_per_attack=100, seed=999)
    y_true = df_test["is_anomaly"].values
    X_test = df_test[FEATURE_NAMES].values

    # Run predictions
    scores_dict = ensemble.score_all_models(X_test)

    models_eval = [
        ("Isolation Forest", scores_dict["iso_preds"]),
        ("Deep Autoencoder", scores_dict["ae_preds"]),
        ("One-Class SVM", scores_dict["svm_preds"]),
        ("AegisFlow Ensemble", scores_dict["ensemble_preds"]),
    ]

    print(f"\n{'Algorithm':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 72)

    for name, y_pred in models_eval:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{name:<22} | {acc*100:8.2f}% | {prec*100:8.2f}% | {rec*100:8.2f}% | {f1*100:8.2f}%")

    cm = confusion_matrix(y_true, scores_dict["ensemble_preds"])
    print("\nEnsemble Confusion Matrix:")
    print(f"  True Negatives (Normal detected as Normal) : {cm[0][0]}")
    print(f"  False Positives (Normal false alarms)      : {cm[0][1]}")
    print(f"  False Negatives (Attacks missed)          : {cm[1][0]}")
    print(f"  True Positives (Attacks caught)           : {cm[1][1]}")

    print("\n[✔] AegisFlow Sentinel demonstration completed successfully.")


if __name__ == "__main__":
    run_cli_demonstration()
