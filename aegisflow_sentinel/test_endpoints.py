"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: test_endpoints.py
Description: Automated test suite validating all REST API endpoints.
"""

import os
import sys
import json
import io

_base_dir = os.path.dirname(os.path.abspath(__file__))
_cache_dir = os.path.join(_base_dir, ".keras_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["KERAS_HOME"] = _cache_dir
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from app import app, initialize_sentinel

def run_tests():
    print("[*] Creating Flask Test Client...")
    client = app.test_client()

    print("[1] Testing GET / (SOC Dashboard UI)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"AEGIS" in res.data, "Dashboard title not found in HTML"
    print("    [✔] Passed.")

    print("[2] Testing GET /api/status...")
    res = client.get("/api/status")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "OPERATIONAL"
    assert "models" in data
    print("    [✔] Passed. Sentinel is OPERATIONAL.")

    print("[3] Testing GET /api/traffic/stream (Normal)...")
    res = client.get("/api/traffic/stream?attack=none")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "detection" in data
    assert "xai" in data
    print(f"    [✔] Passed. Normal flow score: {data['detection']['ensemble_score']}")

    print("[4] Testing GET /api/traffic/stream (DDoS Injection)...")
    res = client.get("/api/traffic/stream?attack=ddos")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["detection"]["is_anomaly"] == True
    print(f"    [✔] Passed. DDoS flow score: {data['detection']['ensemble_score']} ({data['detection']['severity']})")

    print("[5] Testing POST /api/detect (Custom JSON Packet)...")
    sample_flow = {
        "duration": 0.02,
        "src_bytes": 500000,
        "dst_bytes": 0,
        "packet_count": 8000,
        "srv_count": 300,
        "serror_rate": 0.95,
        "rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "dst_host_count": 255,
        "dst_host_srv_count": 255,
        "syn_ack_ratio": 80.0,
        "payload_entropy": 2.1,
        "jitter_ms": 0.5
    }
    res = client.post("/api/detect", json=sample_flow)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["detection"]["is_anomaly"] == True
    print(f"    [✔] Passed. Custom packet detected: {data['detection']['severity']}")

    print("[6] Testing GET /api/benchmark...")
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "benchmarks" in data
    ens_acc = data["benchmarks"]["ensemble"]["accuracy"]
    print(f"    [✔] Passed. Benchmark ensemble accuracy: {ens_acc}%")

    print("[7] Testing POST /api/upload (sample_traffic.csv)...")
    with open("sample_traffic.csv", "rb") as f:
        file_bytes = f.read()
    data_dict = {
        "file": (io.BytesIO(file_bytes), "sample_traffic.csv")
    }
    res = client.post("/api/upload", data=data_dict, content_type="multipart/form-data")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "SUCCESS"
    assert data["total_records"] > 0
    print(f"    [✔] Passed. Evaluated {data['total_records']} rows, Anomaly Rate: {data['anomaly_rate_pct']}%")

    print("\n[ALL 7 ENDPOINT TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    run_tests()
