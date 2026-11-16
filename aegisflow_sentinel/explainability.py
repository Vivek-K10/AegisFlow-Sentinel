"""
AegisFlow: Autonomous Zero-Day Network Intrusion Sentinel
Module: explainability.py
Description: Explainable AI (XAI) feature attribution engine. Computes per-feature Z-scores,
relative percentage contributions, and plain-English natural language forensic summaries.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataset_generator import FEATURE_NAMES

HUMAN_FEATURE_NAMES = {
    "duration": "Flow Duration (s)",
    "src_bytes": "Source Outbound Bytes",
    "dst_bytes": "Destination Inbound Bytes",
    "packet_count": "Total Packet Count",
    "srv_count": "Service Connection Count",
    "serror_rate": "SYN Error Rate",
    "rerror_rate": "REJ Error Rate",
    "same_srv_rate": "Same Service Rate",
    "diff_srv_rate": "Different Service Rate",
    "dst_host_count": "Destination Host Count",
    "dst_host_srv_count": "Destination Host Service Count",
    "syn_ack_ratio": "SYN / ACK Ratio",
    "payload_entropy": "Payload Byte Entropy",
    "jitter_ms": "Packet Arrival Jitter (ms)"
}


class XAIAttributionEngine:
    def __init__(self, baseline_stats: Dict[str, Dict[str, float]]):
        self.baseline_stats = baseline_stats

    def update_baseline(self, baseline_stats: Dict[str, Dict[str, float]]):
        """Updates baseline statistics after retraining."""
        self.baseline_stats = baseline_stats

    def explain(self, flow: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Analyzes a network flow and produces full XAI attribution diagnostics:
        - Z-score deviation per feature
        - Percentage contribution towards the anomaly
        - Natural language forensic explanations
        """
        if not self.baseline_stats:
            return {
                "top_contributors": [],
                "feature_deviations": [],
                "explanation_summary": "Baseline statistics not yet established.",
                "threat_indicators": []
            }

        deviations = []
        raw_deviations = []

        for feat in FEATURE_NAMES:
            val = float(flow.get(feat, 0.0))
            stats = self.baseline_stats.get(feat, {"mean": 0.0, "std": 1.0})
            mean = stats["mean"]
            std = max(stats["std"], 1e-4)

            # Signed and absolute Z-scores
            z_score = (val - mean) / std
            abs_z = abs(z_score)
            raw_deviations.append(abs_z)

            direction = "ELEVATED" if z_score > 0 else "DEPRESSED"
            if abs_z < 0.8:
                direction = "NORMAL"

            deviations.append({
                "feature": feat,
                "display_name": HUMAN_FEATURE_NAMES.get(feat, feat),
                "observed_value": val,
                "baseline_mean": round(mean, 2),
                "baseline_std": round(std, 2),
                "z_score": round(z_score, 2),
                "abs_z_score": round(abs_z, 2),
                "direction": direction
            })

        # Calculate relative contribution percentage
        total_abs_z = sum(raw_deviations)
        if total_abs_z < 1e-6:
            for item in deviations:
                item["contribution_pct"] = round(100.0 / len(deviations), 1)
        else:
            for item in deviations:
                pct = (item["abs_z_score"] / total_abs_z) * 100.0
                item["contribution_pct"] = round(pct, 1)

        # Sort features by absolute Z-score / contribution descending
        sorted_deviations = sorted(deviations, key=lambda x: x["abs_z_score"], reverse=True)
        top_contributors = sorted_deviations[:top_k]

        # Generate human-readable plain English forensic summary
        explanations = []
        threat_indicators = []

        for item in top_contributors:
            feat = item["feature"]
            display = item["display_name"]
            val = item["observed_value"]
            mean = item["baseline_mean"]
            z = item["z_score"]

            if abs(z) >= 1.8:
                if z > 0:
                    explanations.append(
                        f"{display} is abnormally high: {val:,.2f} vs expected baseline {mean:,.2f} (+{z:.1f}σ)"
                    )
                else:
                    explanations.append(
                        f"{display} is abnormally low: {val:,.2f} vs expected baseline {mean:,.2f} ({z:.1f}σ)"
                    )

        # Threat scenario pattern correlation
        syn_ratio = float(flow.get("syn_ack_ratio", 1.0))
        serror = float(flow.get("serror_rate", 0.0))
        rerror = float(flow.get("rerror_rate", 0.0))
        src_bytes = float(flow.get("src_bytes", 0.0))
        dst_host_count = float(flow.get("dst_host_count", 0.0))
        diff_srv = float(flow.get("diff_srv_rate", 0.0))
        entropy = float(flow.get("payload_entropy", 0.0))
        jitter = float(flow.get("jitter_ms", 10.0))

        if syn_ratio > 10.0 or serror > 0.60:
            threat_indicators.append("DDoS / SYN Flood Signature (Extreme SYN/ACK imbalance, high TCP handshake failure)")
        if dst_host_count > 160 and diff_srv > 0.60 and rerror > 0.40:
            threat_indicators.append("Reconnaissance Sweep / Port Scanning (High target dispersion, elevated connection rejects)")
        if src_bytes > 1_000_000 and entropy > 7.5:
            threat_indicators.append("Data Exfiltration Spike (Massive outbound transfer volume with encrypted payload entropy)")
        if jitter < 0.10 and flow.get("srv_count", 1) <= 2 and flow.get("duration", 0) < 0.06:
            threat_indicators.append("Zero-Day C2 Heartbeat Beacon (Near-zero mechanical jitter with fixed-packet cadence)")

        if not explanations:
            summary = "Flow metrics are within normal statistical tolerances of the learned baseline."
        else:
            summary = "Key Anomalous Factors: " + "; ".join(explanations[:3]) + "."

        return {
            "top_contributors": top_contributors,
            "feature_deviations": sorted_deviations,
            "explanation_summary": summary,
            "threat_indicators": threat_indicators
        }
