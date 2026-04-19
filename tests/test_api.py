"""
AuraNet API test suite.

Uses FastAPI's built-in TestClient — no running server required.
The TestClient is used as a context manager so the lifespan hook fires
and loads the model artifacts before any requests are sent.

Run with:  python tests/test_api.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path and is the CWD so model paths resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.serve import app

# ── payloads ──────────────────────────────────────────────────────────────────

# Normal: modest bytes, multi-second duration, logged-in HTTP session
NORMAL_PAYLOAD = {
    "duration":          5.0,
    "protocol_type":     "tcp",
    "service":           "http",
    "flag":              "SF",
    "src_bytes":         4800,
    "dst_bytes":         7200,
    "logged_in":         1,
    "hot":               0,
    "num_failed_logins": 0,
    "same_srv_rate":     1.0,
    "dst_host_count":    20,
}

# Malicious: neptune SYN-flood (most common DoS in NSL-KDD).
# Characteristics: flag=S0 (SYN sent, no ACK), zero payload bytes,
# 511 connections in 2-second window, 100% SYN-error rate across all
# traffic-window features — the textbook fingerprint in the dataset.
MALICIOUS_PAYLOAD = {
    "duration":                     0,
    "protocol_type":                "tcp",
    "service":                      "http",
    "flag":                         "S0",
    "src_bytes":                    0,
    "dst_bytes":                    0,
    "logged_in":                    0,
    "hot":                          0,
    "num_failed_logins":            0,
    "count":                        511,
    "srv_count":                    511,
    "serror_rate":                  1.0,
    "srv_serror_rate":              1.0,
    "rerror_rate":                  0.0,
    "srv_rerror_rate":              0.0,
    "same_srv_rate":                1.0,
    "diff_srv_rate":                0.0,
    "dst_host_count":               255,
    "dst_host_srv_count":           255,
    "dst_host_same_srv_rate":       1.0,
    "dst_host_diff_srv_rate":       0.0,
    "dst_host_same_src_port_rate":  0.0,
    "dst_host_srv_diff_host_rate":  0.0,
    "dst_host_serror_rate":         1.0,
    "dst_host_srv_serror_rate":     1.0,
    "dst_host_rerror_rate":         0.0,
    "dst_host_srv_rerror_rate":     0.0,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'='*52}")
    print(f"  {title}")
    print(f"{'='*52}")


def _print_result(label: str, response) -> None:
    data   = response.json()
    status = response.status_code
    _banner(label)
    print(f"  HTTP status        : {status}")
    if status == 200:
        print(f"  Prediction         : {data['prediction']}")
        print(f"  Confidence (P(Atk)): {data['confidence']:.4f}")
        print(f"  Risk Level         : {data['risk_level']}")
        print(f"  Network Intensity  : {data['network_intensity']:,.1f} bytes/sec")
    else:
        print(f"  Error: {data}")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_health(client: TestClient) -> None:
    _banner("Health Check")
    r = client.get("/health")
    print(f"  Response: {r.json()}")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True
    print("  [PASS] /health")


def test_normal_traffic(client: TestClient) -> None:
    r = client.post("/analyze", json=NORMAL_PAYLOAD)
    _print_result("Test: Normal Traffic", r)
    assert r.status_code == 200
    result = r.json()
    assert result["prediction"] == "Normal", (
        f"Expected Normal, got {result['prediction']} (confidence={result['confidence']:.4f})"
    )
    print("  [PASS] Correctly classified as Normal")


def test_malicious_traffic(client: TestClient) -> None:
    r = client.post("/analyze", json=MALICIOUS_PAYLOAD)
    _print_result("Test: Malicious Traffic (DoS-style)", r)
    assert r.status_code == 200
    result = r.json()
    assert result["prediction"] == "Attack", (
        f"Expected Attack, got {result['prediction']} (confidence={result['confidence']:.4f})"
    )
    assert result["risk_level"] == "HIGH", (
        f"Expected HIGH risk, got {result['risk_level']}"
    )
    print("  [PASS] Correctly classified as Attack with HIGH risk")


def test_unseen_service_label(client: TestClient) -> None:
    """API must not crash on a service value not seen during training."""
    payload = {**NORMAL_PAYLOAD, "service": "unknown_protocol_xyz"}
    r = client.post("/analyze", json=payload)
    _banner("Test: Unseen service label (robustness)")
    print(f"  HTTP status : {r.status_code}")
    print(f"  Response    : {r.json()}")
    assert r.status_code == 200, "Server must handle unseen labels gracefully"
    print("  [PASS] Graceful fallback for unknown service label")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nAuraNet API Test Suite")
    print("=" * 52)

    with TestClient(app) as client:
        test_health(client)
        test_normal_traffic(client)
        test_malicious_traffic(client)
        test_unseen_service_label(client)

    print(f"\n{'='*52}")
    print("  ALL TESTS PASSED")
    print(f"{'='*52}\n")
