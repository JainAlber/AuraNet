"""
AuraNet — FastAPI inference server.

Endpoints
---------
  POST /analyze   Accept raw NSL-KDD-style network features, return prediction + confidence.
  GET  /health    Liveness check.

Run
---
  uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── paths (absolute, so the server works from any CWD) ───────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# ── artifact loading ──────────────────────────────────────────────────────────

_artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    required = ["xgb_tuned.joblib", "scaler.joblib", "label_encoders.joblib", "feature_meta.joblib"]
    missing  = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing model artifacts: {missing}. "
            "Run `python src/features.py` then `python src/tune.py` first."
        )
    _artifacts["model"]    = joblib.load(MODELS_DIR / "xgb_tuned.joblib")
    _artifacts["scaler"]   = joblib.load(MODELS_DIR / "scaler.joblib")
    _artifacts["encoders"] = joblib.load(MODELS_DIR / "label_encoders.joblib")
    _artifacts["meta"]     = joblib.load(MODELS_DIR / "feature_meta.joblib")
    print(f"[AuraNet] Loaded model + artifacts from {MODELS_DIR}")
    yield
    _artifacts.clear()


app = FastAPI(
    title="AuraNet",
    description="Real-time anomaly detection for network traffic (NSL-KDD / XGBoost)",
    version="1.0.0",
    lifespan=lifespan,
)


# ── request schema ─────────────────────────────────────────────────────────────
# All 41 raw NSL-KDD features. Categoricals are strings; the API encodes them.
# Defaults represent a benign, idle connection so callers only need to pass the
# fields that differ from normal.

class TrafficFeatures(BaseModel):
    duration:                    float = Field(0.0,  description="Connection duration (seconds)")
    protocol_type:               str   = Field("tcp", description="tcp | udp | icmp")
    service:                     str   = Field("http")
    flag:                        str   = Field("SF")
    src_bytes:                   int   = Field(0,    description="Bytes from source to dest")
    dst_bytes:                   int   = Field(0,    description="Bytes from dest to source")
    land:                        int   = Field(0)
    wrong_fragment:              int   = Field(0)
    urgent:                      int   = Field(0)
    hot:                         int   = Field(0)
    num_failed_logins:           int   = Field(0)
    logged_in:                   int   = Field(1)
    num_compromised:             int   = Field(0)
    root_shell:                  int   = Field(0)
    su_attempted:                int   = Field(0)
    num_root:                    int   = Field(0)
    num_file_creations:          int   = Field(0)
    num_shells:                  int   = Field(0)
    num_access_files:            int   = Field(0)
    num_outbound_cmds:           int   = Field(0)
    is_host_login:               int   = Field(0)
    is_guest_login:              int   = Field(0)
    count:                       int   = Field(1)
    srv_count:                   int   = Field(1)
    serror_rate:                 float = Field(0.0)
    srv_serror_rate:             float = Field(0.0)
    rerror_rate:                 float = Field(0.0)
    srv_rerror_rate:             float = Field(0.0)
    same_srv_rate:               float = Field(1.0)
    diff_srv_rate:               float = Field(0.0)
    srv_diff_host_rate:          float = Field(0.0)
    dst_host_count:              int   = Field(1)
    dst_host_srv_count:          int   = Field(1)
    dst_host_same_srv_rate:      float = Field(1.0)
    dst_host_diff_srv_rate:      float = Field(0.0)
    dst_host_same_src_port_rate: float = Field(0.0)
    dst_host_srv_diff_host_rate: float = Field(0.0)
    dst_host_serror_rate:        float = Field(0.0)
    dst_host_srv_serror_rate:    float = Field(0.0)
    dst_host_rerror_rate:        float = Field(0.0)
    dst_host_srv_rerror_rate:    float = Field(0.0)


# ── response schema ───────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    prediction:        str
    confidence:        float
    risk_level:        str
    network_intensity: float


# ── inference helpers ─────────────────────────────────────────────────────────

_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def _safe_encode(le, value: str) -> int:
    """LabelEncode with a fallback to 0 for unseen categories."""
    return int(le.transform([value])[0]) if value in le.classes_ else 0


def _preprocess(features: TrafficFeatures) -> pd.DataFrame:
    row = features.model_dump()

    # 1. Derive engineered feature
    safe_dur = max(row["duration"], 1e-6)
    row["network_intensity"] = (row["src_bytes"] + row["dst_bytes"]) / safe_dur

    # 2. Label-encode categoricals
    encoders = _artifacts["encoders"]
    for col in _CATEGORICAL_COLS:
        if col in encoders:
            row[col] = _safe_encode(encoders[col], str(row[col]))

    # 3. Build DataFrame in the exact column order the model was trained on
    feature_order = _artifacts["meta"]["feature_order"]
    df = pd.DataFrame([row])[feature_order]

    # 4. Scale numerical columns
    num_cols = _artifacts["meta"]["num_cols"]
    df[num_cols] = _artifacts["scaler"].transform(df[num_cols])

    return df


def _risk_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.50:
        return "MEDIUM"
    return "LOW"


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalysisResult)
def analyze(features: TrafficFeatures) -> AnalysisResult:
    """
    Classify a single network connection as Normal or Attack.

    Returns the predicted class, the model's confidence (probability of Attack),
    a risk level tier, and the derived network_intensity for transparency.
    """
    try:
        df = _preprocess(features)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {exc}")

    model  = _artifacts["model"]
    proba  = float(model.predict_proba(df)[0][1])       # P(Attack)
    label  = "Attack" if proba >= 0.5 else "Normal"
    safe_dur = max(features.duration, 1e-6)
    raw_ni = (features.src_bytes + features.dst_bytes) / safe_dur

    return AnalysisResult(
        prediction=label,
        confidence=round(proba, 6),
        risk_level=_risk_level(proba),
        network_intensity=round(raw_ni, 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _artifacts}
