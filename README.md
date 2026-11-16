# AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
> **Production-Ready, Signature-Free Cybersecurity Threat Intelligence Platform Powered by Unsupervised Machine Learning & Explainable AI (XAI)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![ML Suite](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20TensorFlow-orange.svg)](https://scikit-learn.org/)
[![UI Style](https://img.shields.io/badge/SOC%20UI-Glassmorphism%20Neon-00f3ff.svg)](https://chartjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🛡️ 1. Overview & Threat Defense Paradigm

Traditional Network Intrusion Detection Systems (NIDS) like Snort and Suricata rely heavily on **predefined signature rules**. This creates a critical vulnerability against **zero-day exploits**, stealthy Command & Control (C2) channels, and novel attack payloads that do not match existing CVE signatures.

**AegisFlow Sentinel** is an autonomous, **signature-free** behavioral threat sentinel. Rather than searching for known signatures, it establishes a high-fidelity statistical baseline of normal network behavior (HTTP, HTTPS, DNS, SSH, API) using an unsupervised multi-algorithm machine learning suite. When traffic flows deviate from this baseline manifold, AegisFlow flags the intrusion, computes consensus severity, projects the anomaly into a 2D latent bottleneck space, and provides **Explainable AI (XAI) forensic attribution** explaining *why* the traffic was flagged in plain English.

---
🏗️ 2. System Architecture

aegisflow_sentinel/ ├── app.py # Production Flask REST API server & SOC controller ├── cli_demo.py # Standalone terminal demonstration & benchmark utility ├── dataset_generator.py # 14-feature flow simulator & 4 cyber attack injectors ├── explainability.py # Explainable AI (XAI) attribution & Z-score engine ├── test_endpoints.py # Automated REST API endpoint verification suite ├── sample_traffic.csv # Sample test network capture for CSV ingestion ├── requirements.txt # Production dependencies ├── models/ │ ├── init.py # Package exports │ ├── isolation_forest_model.py # Tree-based partitioning with contamination control │ ├── autoencoder_model.py # Deep Keras Autoencoder (14 -> 12 -> 6 -> 12 -> 14) + 2D PCA │ ├── one_class_svm_model.py # Non-linear RBF kernel hyperplane boundary detector │ └── ensemble_detector.py # Soft-voting + consensus fusion & severity tiering ├── templates/ │ └── index.html # Dark glassmorphism SOC dashboard UI ├── static/ │ ├── css/ │ │ └── styles.css # Neon-accented obsidian theme (cyan, violet, emerald, crimson) │ └── js/ │ └── dashboard.js # Chart.js live telemetry, 2D latent scatter & XAI modals └── .vscode/ ├── launch.json # F5 Debugger configurations for Web SOC and CLI └── settings.json # Python path and workspace configuration

---
## 📊 3. The 14 Flow-Level Network Features
Modeled after industry-standard network security benchmarks (**NSL-KDD** and **CICIDS**):
| # | Feature Name | Unit / Range | Security Relevance |
|---|--------------|--------------|-------------------|
| 1 | `duration` | Seconds (float) | Length of the active network connection |
| 2 | `src_bytes` | Bytes (int) | Outbound payload volume transmitted from source |
| 3 | `dst_bytes` | Bytes (int) | Inbound payload volume received from destination |
| 4 | `packet_count`| Count (int) | Total number of packets exchanged in flow |
| 5 | `srv_count` | Count (int) | Number of connections to the same service in time window |
| 6 | `serror_rate` | 0.0 - 1.0 (float) | % of connections that produced SYN errors (handshake failures) |
| 7 | `rerror_rate` | 0.0 - 1.0 (float) | % of connections that produced REJ errors (rejected connections) |
| 8 | `same_srv_rate`| 0.0 - 1.0 (float) | Proportion of connections targeting the same destination service |
| 9 | `diff_srv_rate`| 0.0 - 1.0 (float) | Proportion of connections dispersed across different services |
| 10| `dst_host_count` | 1 - 255 (int) | Number of connections targeting the destination host |
| 11| `dst_host_srv_count` | 1 - 255 (int) | Number of connections targeting destination host on same service |
| 12| `syn_ack_ratio` | Ratio (float) | Imbalance of SYN to ACK packets (extreme during SYN floods) |
| 13| `payload_entropy` | 0.0 - 8.0 (float) | Shannon byte entropy (spikes during encrypted exfiltration) |
| 14| `jitter_ms` | Milliseconds (float) | Packet arrival interval variance (collapses in robotic C2 beacons) |
---
## ⚔️ 4. Simulated Attack Scenarios
AegisFlow features synthetic injection functions for 4 primary cyber threat vectors:
1. **DDoS SYN Flood**: Extreme packet count burst, sub-second duration, massive `syn_ack_ratio` (> 35.0), and elevated `serror_rate` (handshakes left unanswered).
2. **Port Scan Sweep**: Rapid horizontal reconnaissance sweeps with low byte transfer, high `dst_host_count`, high `diff_srv_rate` (> 0.80), and elevated `rerror_rate` from closed ports.
3. **Data Exfiltration Spike**: Massive outbound `src_bytes` burst (2.5MB to 35MB), high packet count, and high Shannon `payload_entropy` (> 7.85) indicative of encrypted/compressed archives.
4. **Zero-Day C2 Heartbeat Beacon**: Stealthy persistent Command & Control beacon characterized by mechanical near-zero `jitter_ms` (< 0.06 ms), uniform byte cadence, and single destination service access.
---
## 🧠 5. Multi-Algorithm Unsupervised Machine Learning Suite
The detection engine combines three distinct geometric and neural approaches:
1. **Isolation Forest** (`models/isolation_forest_model.py`):
   - Fast recursive tree partitioning isolating anomalies in shallow depths.
   - Normalized continuous anomaly scoring ($0.0 - 1.0$) with contamination rate $\nu$.
2. **Deep Neural Network Autoencoder** (`models/autoencoder_model.py`):
   - Architecture: `Input(14) -> Dense(12, relu) -> Bottleneck(6, relu) -> Dense(12, relu) -> Output(14, linear)`.
   - Learns normal flow reconstruction. Classification is dynamically anchored to the **93rd-percentile Mean Squared Error (MSE)** reconstruction threshold.
   - Bottleneck activations are projected into **2D Latent PCA coordinates** for real-time SOC scatter visualizers.
3. **One-Class Support Vector Machine** (`models/one_class_svm_model.py`):
   - Non-linear RBF kernel boundary fitting a tight hypersphere around baseline flows.
   - Computes signed hyperplane distance scores where negative distances mark intrusions.
4. **Weighted Soft-Voting Consensus Ensemble** (`models/ensemble_detector.py`):
   - Fuses all 3 models: $\text{Score} = 0.35 \times \text{IsoForest} + 0.40 \times \text{Autoencoder} + 0.25 \times \text{OneClassSVM}$.
   - **Consensus Rule**: Anomaly is flagged if $\text{Score} \ge 0.45$ OR if 2 or more models independently flag an intrusion.
   - **Severity Tiers**: `NORMAL` (< 0.40), `LOW` (0.40 - 0.55), `MEDIUM` (0.55 - 0.72), `HIGH` (0.72 - 0.88), `CRITICAL` (>= 0.88).
---
## 🔍 6. Explainable AI (XAI) Attribution Module
Security analysts cannot act on black-box alerts. The XAI engine (`explainability.py`):
- Measures the exact statistical **Z-Score deviation** of each feature against the learned normal baseline:
  $$Z = \frac{x_i - \mu_i}{\sigma_i}$$
- Computes each feature's **relative percentage contribution** to the anomaly score.
- Outputs **plain-English natural language forensic summaries**, e.g.:
  > *"SYN / ACK Ratio is abnormally high: 95.53 vs expected baseline 1.00 (+3016.3σ); SYN Error Rate is abnormally high: 0.92 vs expected baseline 0.00 (+1007.2σ); Source Outbound Bytes is abnormally high: 606,249.00 vs expected baseline 1,982.90 (+331.3σ)."*
- Matches behavioral patterns with specific tactical threat signatures.
---
## 🖥️ 7. Security Operations Center (SOC) Dashboard
- **Dark Glassmorphism Theme**: Cyber obsidian backdrop (`#060911`) with neon cyan (`#00f3ff`), violet (`#a855f7`), emerald (`#10b981`), and crimson (`#ef4444`).
- **Live Telemetry Stream**: Real-time Chart.js line visualizer tracking Ensemble, Isolation Forest, and Autoencoder scores simultaneously.
- **2D Latent Space Scatter**: Autoencoder bottleneck PCA scatter plotting normal baseline cluster points alongside flagged intrusions.
- **Algorithm Comparison Radar**: Benchmarking Accuracy, Precision, Recall, F1, and Zero-Day resilience across all 3 algorithms.
- **Top KPI Cards**: Real-time packet throughput, intercepted intrusions, alert ratio %, and multi-model consensus agreement.
- **Interactive Cyber Simulator**: One-click threat injection buttons (`Benign`, `DDoS`, `Port Scan`, `Data Exfiltration`, `Zero-Day C2`).
- **Live Flow Feed Table**: Color-coded severity badges with an **"Inspect XAI"** button opening an in-depth forensic diagnosis modal with horizontal contribution bars and a 14-feature deviation matrix.
- **Retraining Modal**: Adjust sample size, contamination, and epochs on the fly.
- **CSV Ingestion Modal**: Drag-and-drop network traffic CSV files for batch classification.
---
## 📡 8. REST API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the interactive Security Operations Center (SOC) dashboard. |
| `GET` | `/api/status` | Returns system health, model readiness, loss curve, and 2D baseline clusters. |
| `GET` | `/api/traffic/stream?attack=<type>` | Streams single flow telemetry with optional attack injection (`none`, `ddos`, `port_scan`, `exfiltration`, `c2_beacon`). |
| `POST` | `/api/detect` | Evaluates custom packet JSON payload and returns scores + XAI attribution. |
| `POST` | `/api/train` | Retrains models with JSON payload: `{"sample_size": 2500, "contamination": 0.05, "epochs": 20}`. |
| `POST` | `/api/upload` | Multipart file upload of `.csv` network logs; returns summary metrics and preview table. |
| `GET` | `/api/benchmark` | Generates a benchmark set and returns comparative accuracy, precision, recall, and confusion matrices. |
---
## 📈 9. Benchmark Performance Summary
Evaluated on 1,400 mixed test flows (1,000 normal baseline + 400 zero-day intrusions):
```text
Algorithm              | Accuracy   | Precision  | Recall     | F1-Score  
------------------------------------------------------------------------
Isolation Forest       |    89.79% |    87.46% |    75.00% |    80.75%
Deep Autoencoder       |    87.50% |    80.00% |    75.00% |    77.42%
One-Class SVM          |    94.79% |    84.57% |   100.00% |    91.64%
AegisFlow Ensemble     |    95.07% |    85.29% |   100.00% |    92.06%
Ensemble Confusion Matrix:
  True Negatives (Normal detected as Normal) : 931
  False Positives (Normal false alarms)      : 69
  False Negatives (Attacks missed)          : 0
  True Positives (Attacks caught)           : 400 (100% Recall)
🚀 10. Quick Start Guide
Prerequisites
Python 3.10+
Pip package manager
Installation
bash
git clone https://github.com/Vivek-K10/AegisFlow-Sentinel.git
cd AegisFlow-Sentinel
pip install -r requirements.txt
Running the Web SOC Dashboard
bash
python app.py

Open http://127.0.0.1:5000 in your browser.

Running the Standalone CLI Demonstration
bash
python cli_demo.py
Running Automated Test Suite
bash
python test_endpoints.py
