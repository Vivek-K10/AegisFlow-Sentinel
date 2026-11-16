"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: dataset_generator.py
Description: Generates realistic synthetic network traffic baseline datasets
and injects advanced cybersecurity attack vectors modeled after NSL-KDD and CICIDS.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any

FEATURE_NAMES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "packet_count",
    "srv_count",
    "serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "syn_ack_ratio",
    "payload_entropy",
    "jitter_ms"
]

PROTOCOL_TYPES = ["HTTP", "HTTPS", "DNS", "SSH", "API"]
ATTACK_TYPES = ["ddos_syn_flood", "port_scan_sweep", "data_exfiltration", "zero_day_c2_beacon"]


def generate_single_normal_flow(protocol: Optional[str] = None) -> Dict[str, Any]:
    """Generates a single realistic normal background network flow."""
    if protocol is None:
        protocol = np.random.choice(PROTOCOL_TYPES, p=[0.35, 0.40, 0.10, 0.05, 0.10])
    
    if protocol == "HTTP":
        duration = float(np.random.exponential(scale=0.8) + 0.05)
        src_bytes = int(np.random.gamma(shape=3.0, scale=400) + 150)
        dst_bytes = int(np.random.gamma(shape=4.0, scale=3500) + 500)
        packet_count = int(max(4, int(src_bytes / 300) + int(dst_bytes / 1200)))
        srv_count = int(np.random.randint(5, 45))
        serror_rate = float(np.random.beta(a=0.5, b=25.0) * 0.05)
        rerror_rate = float(np.random.beta(a=0.5, b=30.0) * 0.04)
        same_srv_rate = float(np.clip(np.random.normal(0.92, 0.05), 0.70, 1.0))
        diff_srv_rate = float(np.clip(1.0 - same_srv_rate + np.random.normal(0, 0.02), 0.0, 0.30))
        dst_host_count = int(np.random.randint(15, 120))
        dst_host_srv_count = int(np.clip(dst_host_count * same_srv_rate, 5, 255))
        syn_ack_ratio = float(np.clip(np.random.normal(1.0, 0.04), 0.85, 1.15))
        payload_entropy = float(np.clip(np.random.normal(4.5, 0.6), 2.8, 6.2))
        jitter_ms = float(np.random.gamma(shape=2.5, scale=6.0) + 2.0)

    elif protocol == "HTTPS":
        duration = float(np.random.exponential(scale=1.8) + 0.1)
        src_bytes = int(np.random.gamma(shape=3.5, scale=650) + 300)
        dst_bytes = int(np.random.gamma(shape=5.0, scale=8000) + 1200)
        packet_count = int(max(6, int(src_bytes / 250) + int(dst_bytes / 1100)))
        srv_count = int(np.random.randint(10, 60))
        serror_rate = float(np.random.beta(a=0.4, b=30.0) * 0.03)
        rerror_rate = float(np.random.beta(a=0.4, b=35.0) * 0.03)
        same_srv_rate = float(np.clip(np.random.normal(0.95, 0.03), 0.80, 1.0))
        diff_srv_rate = float(np.clip(1.0 - same_srv_rate + np.random.normal(0, 0.015), 0.0, 0.20))
        dst_host_count = int(np.random.randint(20, 150))
        dst_host_srv_count = int(np.clip(dst_host_count * same_srv_rate, 10, 255))
        syn_ack_ratio = float(np.clip(np.random.normal(1.0, 0.03), 0.90, 1.10))
        payload_entropy = float(np.clip(np.random.normal(7.4, 0.25), 6.5, 7.95))
        jitter_ms = float(np.random.gamma(shape=3.0, scale=5.0) + 3.0)

    elif protocol == "DNS":
        duration = float(np.clip(np.random.exponential(scale=0.03) + 0.005, 0.001, 0.20))
        src_bytes = int(np.random.randint(48, 140))
        dst_bytes = int(np.random.randint(80, 520))
        packet_count = int(np.random.choice([2, 4, 6], p=[0.75, 0.20, 0.05]))
        srv_count = int(np.random.randint(2, 25))
        serror_rate = 0.0
        rerror_rate = float(np.random.choice([0.0, 0.02, 0.05], p=[0.90, 0.08, 0.02]))
        same_srv_rate = float(np.clip(np.random.normal(0.96, 0.03), 0.85, 1.0))
        diff_srv_rate = float(np.clip(1.0 - same_srv_rate, 0.0, 0.15))
        dst_host_count = int(np.random.randint(5, 50))
        dst_host_srv_count = int(np.clip(dst_host_count * same_srv_rate, 2, 255))
        syn_ack_ratio = 1.0
        payload_entropy = float(np.clip(np.random.normal(3.2, 0.4), 1.8, 4.4))
        jitter_ms = float(np.random.exponential(scale=3.0) + 0.5)

    elif protocol == "SSH":
        duration = float(np.random.gamma(shape=3.0, scale=25.0) + 5.0)
        src_bytes = int(np.random.gamma(shape=3.0, scale=2000) + 1200)
        dst_bytes = int(np.random.gamma(shape=3.5, scale=4000) + 2000)
        packet_count = int(max(20, int(src_bytes / 120) + int(dst_bytes / 250)))
        srv_count = int(np.random.randint(1, 10))
        serror_rate = float(np.random.beta(a=0.3, b=40.0) * 0.02)
        rerror_rate = float(np.random.beta(a=0.3, b=40.0) * 0.02)
        same_srv_rate = float(np.clip(np.random.normal(0.98, 0.02), 0.90, 1.0))
        diff_srv_rate = float(np.clip(1.0 - same_srv_rate, 0.0, 0.10))
        dst_host_count = int(np.random.randint(1, 25))
        dst_host_srv_count = int(np.clip(dst_host_count * same_srv_rate, 1, 255))
        syn_ack_ratio = 1.0
        payload_entropy = float(np.clip(np.random.normal(7.6, 0.18), 7.0, 7.98))
        jitter_ms = float(np.random.gamma(shape=2.0, scale=8.0) + 5.0)

    else:  # API
        duration = float(np.random.exponential(scale=0.3) + 0.02)
        src_bytes = int(np.random.gamma(shape=2.5, scale=400) + 250)
        dst_bytes = int(np.random.gamma(shape=3.5, scale=1200) + 400)
        packet_count = int(max(4, int(src_bytes / 250) + int(dst_bytes / 600)))
        srv_count = int(np.random.randint(8, 40))
        serror_rate = float(np.random.beta(a=0.4, b=30.0) * 0.02)
        rerror_rate = float(np.random.beta(a=0.4, b=30.0) * 0.02)
        same_srv_rate = float(np.clip(np.random.normal(0.93, 0.04), 0.80, 1.0))
        diff_srv_rate = float(np.clip(1.0 - same_srv_rate, 0.0, 0.20))
        dst_host_count = int(np.random.randint(10, 80))
        dst_host_srv_count = int(np.clip(dst_host_count * same_srv_rate, 5, 255))
        syn_ack_ratio = float(np.clip(np.random.normal(1.0, 0.02), 0.95, 1.05))
        payload_entropy = float(np.clip(np.random.normal(5.1, 0.5), 3.8, 6.4))
        jitter_ms = float(np.random.gamma(shape=2.0, scale=4.0) + 1.5)

    return {
        "duration": round(float(duration), 4),
        "src_bytes": int(src_bytes),
        "dst_bytes": int(dst_bytes),
        "packet_count": int(packet_count),
        "srv_count": int(srv_count),
        "serror_rate": round(float(serror_rate), 4),
        "rerror_rate": round(float(rerror_rate), 4),
        "same_srv_rate": round(float(same_srv_rate), 4),
        "diff_srv_rate": round(float(diff_srv_rate), 4),
        "dst_host_count": int(dst_host_count),
        "dst_host_srv_count": int(dst_host_srv_count),
        "syn_ack_ratio": round(float(syn_ack_ratio), 4),
        "payload_entropy": round(float(payload_entropy), 4),
        "jitter_ms": round(float(jitter_ms), 3),
        "protocol": protocol,
        "is_anomaly": 0,
        "attack_type": "None"
    }


def generate_ddos_syn_flood() -> Dict[str, Any]:
    """Generates synthetic DDoS SYN Flood attack telemetry."""
    duration = float(np.random.uniform(0.001, 0.06))
    packet_count = int(np.random.randint(3000, 18000))
    src_bytes = int(packet_count * np.random.randint(40, 64))
    dst_bytes = int(np.random.randint(0, 90))
    srv_count = int(np.random.randint(200, 500))
    serror_rate = float(np.random.uniform(0.88, 1.00))
    rerror_rate = float(np.random.uniform(0.0, 0.05))
    same_srv_rate = float(np.random.uniform(0.92, 1.00))
    diff_srv_rate = float(np.random.uniform(0.0, 0.08))
    dst_host_count = int(np.random.randint(220, 255))
    dst_host_srv_count = int(np.random.randint(210, 255))
    syn_ack_ratio = float(np.random.uniform(35.0, 150.0))
    payload_entropy = float(np.random.uniform(1.1, 2.8))
    jitter_ms = float(np.random.uniform(0.02, 1.2))

    return {
        "duration": round(duration, 4),
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "packet_count": packet_count,
        "srv_count": srv_count,
        "serror_rate": round(serror_rate, 4),
        "rerror_rate": round(rerror_rate, 4),
        "same_srv_rate": round(same_srv_rate, 4),
        "diff_srv_rate": round(diff_srv_rate, 4),
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "syn_ack_ratio": round(syn_ack_ratio, 4),
        "payload_entropy": round(payload_entropy, 4),
        "jitter_ms": round(jitter_ms, 3),
        "protocol": "TCP/SYN",
        "is_anomaly": 1,
        "attack_type": "DDoS SYN Flood"
    }


def generate_port_scan_sweep() -> Dict[str, Any]:
    """Generates synthetic Port Scan Sweep / Reconnaissance attack flow."""
    duration = float(np.random.uniform(0.05, 0.60))
    src_bytes = int(np.random.randint(32, 96))
    dst_bytes = int(np.random.choice([0, 20, 40], p=[0.75, 0.15, 0.10]))
    packet_count = int(np.random.randint(1, 4))
    srv_count = int(np.random.randint(1, 5))
    serror_rate = float(np.random.uniform(0.05, 0.35))
    rerror_rate = float(np.random.uniform(0.60, 0.98))
    same_srv_rate = float(np.random.uniform(0.01, 0.15))
    diff_srv_rate = float(np.random.uniform(0.80, 0.99))
    dst_host_count = int(np.random.randint(180, 255))
    dst_host_srv_count = int(np.random.randint(1, 8))
    syn_ack_ratio = float(np.random.uniform(4.0, 18.0))
    payload_entropy = float(np.random.uniform(1.0, 2.5))
    jitter_ms = float(np.random.uniform(1.0, 12.0))

    return {
        "duration": round(duration, 4),
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "packet_count": packet_count,
        "srv_count": srv_count,
        "serror_rate": round(serror_rate, 4),
        "rerror_rate": round(rerror_rate, 4),
        "same_srv_rate": round(same_srv_rate, 4),
        "diff_srv_rate": round(diff_srv_rate, 4),
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "syn_ack_ratio": round(syn_ack_ratio, 4),
        "payload_entropy": round(payload_entropy, 4),
        "jitter_ms": round(jitter_ms, 3),
        "protocol": "TCP/SCAN",
        "is_anomaly": 1,
        "attack_type": "Port Scan Sweep"
    }


def generate_data_exfiltration() -> Dict[str, Any]:
    """Generates synthetic high-volume encrypted Data Exfiltration Spike."""
    duration = float(np.random.uniform(20.0, 180.0))
    packet_count = int(np.random.randint(4000, 45000))
    src_bytes = int(np.random.randint(2_500_000, 35_000_000))
    dst_bytes = int(np.random.randint(200, 2500))
    srv_count = int(np.random.randint(1, 8))
    serror_rate = float(np.random.uniform(0.0, 0.02))
    rerror_rate = float(np.random.uniform(0.0, 0.02))
    same_srv_rate = float(np.random.uniform(0.90, 1.00))
    diff_srv_rate = float(np.random.uniform(0.0, 0.10))
    dst_host_count = int(np.random.randint(1, 15))
    dst_host_srv_count = int(np.random.randint(1, 12))
    syn_ack_ratio = float(np.random.uniform(0.96, 1.04))
    payload_entropy = float(np.random.uniform(7.82, 7.995))
    jitter_ms = float(np.random.uniform(0.5, 4.0))

    return {
        "duration": round(duration, 4),
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "packet_count": packet_count,
        "srv_count": srv_count,
        "serror_rate": round(serror_rate, 4),
        "rerror_rate": round(rerror_rate, 4),
        "same_srv_rate": round(same_srv_rate, 4),
        "diff_srv_rate": round(diff_srv_rate, 4),
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "syn_ack_ratio": round(syn_ack_ratio, 4),
        "payload_entropy": round(payload_entropy, 4),
        "jitter_ms": round(jitter_ms, 3),
        "protocol": "HTTPS/EXFIL",
        "is_anomaly": 1,
        "attack_type": "Data Exfiltration"
    }


def generate_zero_day_c2_beacon() -> Dict[str, Any]:
    """Generates synthetic stealthy Zero-Day C2 Heartbeat Beacon."""
    duration = float(np.random.uniform(0.015, 0.045))
    src_bytes = int(np.random.choice([384, 512, 768]))
    dst_bytes = int(np.random.choice([64, 128, 256]))
    packet_count = int(np.random.choice([4, 6]))
    srv_count = 1
    serror_rate = 0.0
    rerror_rate = 0.0
    same_srv_rate = 1.0
    diff_srv_rate = 0.0
    dst_host_count = 1
    dst_host_srv_count = 1
    syn_ack_ratio = 1.0
    payload_entropy = float(np.random.uniform(6.9, 7.55))
    jitter_ms = float(np.random.uniform(0.001, 0.060))  # Machine-like near-zero jitter

    return {
        "duration": round(duration, 4),
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "packet_count": packet_count,
        "srv_count": srv_count,
        "serror_rate": round(serror_rate, 4),
        "rerror_rate": round(rerror_rate, 4),
        "same_srv_rate": round(same_srv_rate, 4),
        "diff_srv_rate": round(diff_srv_rate, 4),
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "syn_ack_ratio": round(syn_ack_ratio, 4),
        "payload_entropy": round(payload_entropy, 4),
        "jitter_ms": round(jitter_ms, 3),
        "protocol": "C2/BEACON",
        "is_anomaly": 1,
        "attack_type": "Zero-Day C2 Beacon"
    }


def generate_stream_packet(attack: Optional[str] = None) -> Dict[str, Any]:
    """Returns a single live flow sample, either normal or attack injected."""
    if not attack or attack.lower() in ["none", "normal", ""]:
        return generate_single_normal_flow()
    
    clean_attack = attack.lower().strip()
    if "ddos" in clean_attack or "syn" in clean_attack:
        return generate_ddos_syn_flood()
    elif "port" in clean_attack or "scan" in clean_attack:
        return generate_port_scan_sweep()
    elif "exfil" in clean_attack or "data" in clean_attack:
        return generate_data_exfiltration()
    elif "c2" in clean_attack or "zero" in clean_attack or "beacon" in clean_attack:
        return generate_zero_day_c2_beacon()
    else:
        return generate_single_normal_flow()


def generate_baseline_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generates a pure normal baseline dataset for unsupervised model training."""
    np.random.seed(seed)
    records = [generate_single_normal_flow() for _ in range(n_samples)]
    return pd.DataFrame(records)


def generate_test_dataset(n_normal: int = 2000, n_per_attack: int = 150, seed: int = 1337) -> pd.DataFrame:
    """Generates a mixed benchmark evaluation dataset."""
    np.random.seed(seed)
    records = [generate_single_normal_flow() for _ in range(n_normal)]
    for _ in range(n_per_attack):
        records.append(generate_ddos_syn_flood())
        records.append(generate_port_scan_sweep())
        records.append(generate_data_exfiltration())
        records.append(generate_zero_day_c2_beacon())
    
    df = pd.DataFrame(records)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    print("Generating baseline and benchmark sets...")
    df_train = generate_baseline_dataset(1000)
    print(f"Generated train baseline shape: {df_train.shape}")
    df_test = generate_test_dataset(500, 50)
    print(f"Generated test set shape: {df_test.shape}")
    print(df_test["attack_type"].value_counts())
