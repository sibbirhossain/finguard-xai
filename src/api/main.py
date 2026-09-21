"""FinGuard-XAI real-time scoring API.

Run:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Env:  FINGUARD_ARTIFACTS (default "artifacts"), FINGUARD_API_KEY (optional),
      FINGUARD_CORS_ORIGINS (comma-separated, default http://localhost:5173)
"""
from __future__ import annotations

import os
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ..models.pipeline import FinGuardEngine
from ..monitoring.drift import DriftMonitor

ARTIFACT_DIR = Path(os.getenv("FINGUARD_ARTIFACTS", "artifacts"))
API_KEY = os.getenv("FINGUARD_API_KEY")
CORS_ORIGINS = [o.strip() for o in os.getenv("FINGUARD_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
HISTORY = 5000


class TransactionIn(BaseModel):
    """Incoming payment event. Identifiers must be pseudonymous (no raw PII)."""

    model_config = ConfigDict(extra="ignore")
    txn_id: str = Field(..., min_length=1, max_length=128)
    timestamp: float = Field(..., ge=0, description="event time in seconds")
    card_id: str = Field(..., min_length=1, max_length=128)
    device_id: str = Field(..., min_length=1, max_length=128)
    merchant_id: str = Field(..., min_length=1, max_length=128)
    ip_id: str = Field(..., min_length=1, max_length=128)
    amount: float = Field(..., gt=0, le=10_000_000)


class State:
    """Process-wide engine and rolling history (single-worker deployment)."""

    engine: Optional[FinGuardEngine] = None
    drift: Optional[DriftMonitor] = None
    lock = threading.Lock()  # guards graph updates + scoring
    explain_lock = threading.Lock()  # guards the explainer's model copy
    recent: deque[dict[str, Any]] = deque(maxlen=HISTORY)
    alerts: deque[dict[str, Any]] = deque(maxlen=500)
    latencies: deque[float] = deque(maxlen=HISTORY)
    processed: int = 0
    started: float = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once at startup."""
    torch.set_num_threads(int(os.getenv("FINGUARD_TORCH_THREADS", "1")))
    if (ARTIFACT_DIR / "model.pt").exists():
        State.engine = FinGuardEngine.load(ARTIFACT_DIR)
        State.drift = DriftMonitor(State.engine.scorer._legit_sorted)
    yield


app = FastAPI(title="FinGuard-XAI", version="1.2.0", lifespan=lifespan,
              description="Real-time graph intelligence for financial fraud detection (research prototype).")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Enforce an API key when FINGUARD_API_KEY is set."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def get_engine() -> FinGuardEngine:
    """Return the loaded engine or fail with a clear message."""
    if State.engine is None:
        raise HTTPException(status_code=503, detail="model not loaded: run `python -m src.models.train` first")
    return State.engine


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe."""
    return {"status": "ok", "model_loaded": State.engine is not None, "uptime_s": round(time.time() - State.started, 1)}


@app.post("/api/v1/transactions", dependencies=[Depends(require_key)])
def score_transaction(txn: TransactionIn, explain: bool = Query(True, description="explain alerts")) -> dict[str, Any]:
    """Update the graph with a transaction and return its risk score, latency and explanation."""
    engine = get_engine()
    with State.lock:  # the temporal graph is shared mutable state
        result, data = engine.score_core(txn.model_dump())
    if result["is_alert"] and explain:
        with State.explain_lock:  # explanation runs outside the scoring lock
            engine.attach_explanation(result, data)
    with State.lock:
        State.processed += 1
        State.latencies.append(result["latency_ms"]["scoring_total"])
        if State.drift is not None:
            State.drift.update(result["components"]["anomaly_score"])
        feed_item = {k: result[k] for k in ("txn_id", "card_id", "merchant_id", "amount", "timestamp",
                                            "risk_score", "is_alert")}
        State.recent.append(feed_item)
        if result["is_alert"]:
            State.alerts.append(result)
    return result


@app.get("/api/v1/transactions/recent", dependencies=[Depends(require_key)])
def recent(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    """Most recent scored transactions (newest first)."""
    return list(State.recent)[-limit:][::-1]


@app.get("/api/v1/alerts", dependencies=[Depends(require_key)])
def alerts(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    """Most recent alerts with explanations (newest first)."""
    return list(State.alerts)[-limit:][::-1]


@app.get("/api/v1/stats", dependencies=[Depends(require_key)])
def stats() -> dict[str, Any]:
    """Aggregate counters, risk distribution and measured latency percentiles."""
    scores = np.array([r["risk_score"] for r in State.recent]) if State.recent else np.array([])
    counts = Counter(np.minimum((scores * 10).astype(int), 9).tolist()) if scores.size else Counter()
    lat = np.array(State.latencies) if State.latencies else None
    return {
        "processed": State.processed,
        "alerts": len(State.alerts),
        "alert_rate": round(sum(r["is_alert"] for r in State.recent) / len(State.recent), 5) if State.recent else 0.0,
        "risk_histogram": [{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "count": counts.get(i, 0)} for i in range(10)],
        "latency_ms": ({"p50": round(float(np.percentile(lat, 50)), 3), "p95": round(float(np.percentile(lat, 95)), 3),
                        "p99": round(float(np.percentile(lat, 99)), 3)} if lat is not None else None),
        "graph_nodes": get_engine().graph.num_nodes if State.engine else 0,
        "threshold": round(State.engine.scorer.threshold, 4) if State.engine else None,
    }


@app.get("/api/v1/drift", dependencies=[Depends(require_key)])
def drift() -> dict[str, Any]:
    """Population Stability Index of live traffic vs the calibration reference."""
    get_engine()
    return State.drift.status() if State.drift else {"status": "unavailable"}
