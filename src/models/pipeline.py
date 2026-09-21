"""Single-transaction scoring engine shared by the API and the latency benchmark.

Flow per event: update temporal graph -> extract k-hop subgraph -> GNN
representation + supervised logit -> Deep SVDD anomaly score -> calibrated
hybrid probability -> (alerts only) GNNExplainer rationale.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

from torch_geometric.data import Data

import joblib
import numpy as np
import torch

from .anomaly_detector import AnomalyDetector, HybridRiskScorer
from .explainer import SubgraphExplainer
from .gnn_model import FinGuardGNN
from .graph_engine import TemporalGraph, Transaction


@dataclass
class EngineConfig:
    """Graph sampling and explanation settings (must match training)."""

    num_hops: int = 2
    fanout: int = 8
    max_neighbors: int = 64
    explain_alerts: bool = True
    explainer_epochs: int = 50


def save_artifacts(out_dir: Union[str, Path], model: FinGuardGNN, detector: AnomalyDetector,
                   scorer: HybridRiskScorer, config: EngineConfig, metadata: dict[str, Any]) -> None:
    """Persist all components needed for inference."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, out / "model.pt")
    joblib.dump(detector, out / "detector.joblib")
    joblib.dump(scorer, out / "scorer.joblib")
    (out / "metadata.json").write_text(json.dumps({"engine_config": asdict(config), **metadata}, indent=2))


class FinGuardEngine:
    """Stateful streaming scorer. Not thread-safe: callers must serialise ``score``."""

    def __init__(self, model: FinGuardGNN, detector: AnomalyDetector, scorer: HybridRiskScorer,
                 config: Optional[EngineConfig] = None) -> None:
        self.model = model.eval()
        self.detector = detector
        self.scorer = scorer
        self.config = config or EngineConfig()
        self.graph = TemporalGraph(max_neighbors=self.config.max_neighbors)
        # The explainer gets its own model copy: GNNExplainer attaches masks to the
        # message-passing layers, so sharing weights would corrupt concurrent scoring.
        self.explainer = SubgraphExplainer(copy.deepcopy(self.model), epochs=self.config.explainer_epochs)

    @classmethod
    def load(cls, artifact_dir: Union[str, Path]) -> "FinGuardEngine":
        """Load trusted artifacts produced by ``python -m src.models.train``.

        joblib/torch loading can execute code: only load artifacts you created.
        """
        path = Path(artifact_dir)
        ckpt = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model = FinGuardGNN(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        meta = json.loads((path / "metadata.json").read_text())
        return cls(model, joblib.load(path / "detector.joblib"), joblib.load(path / "scorer.joblib"),
                   EngineConfig(**meta["engine_config"]))

    def score_core(self, record: Union[dict[str, Any], Transaction]) -> tuple[dict[str, Any], Data]:
        """Update the graph and score one transaction (no explanation). Returns (result, subgraph)."""
        txn = record if isinstance(record, Transaction) else Transaction.from_dict(record)
        t0 = time.perf_counter()
        seeds = self.graph.add_transaction(txn)
        data = self.graph.build_txn_graph(txn, seeds, self.config.num_hops, self.config.fanout)
        t1 = time.perf_counter()
        with torch.inference_mode():
            rep = self.model.represent(data.x, data.edge_index, data.edge_attr, data.seed_index, data.txn_feat)
            logit = float(self.model.head(rep).squeeze(-1)[0])
        anomaly = float(self.detector.score(rep.numpy())[0])
        lg, an = np.array([logit]), np.array([anomaly])
        prob = float(self.scorer.risk(lg, an)[0])
        flags, reasons = self.scorer.alert(lg, an)
        t2 = time.perf_counter()
        result = {
            "txn_id": txn.txn_id,
            "card_id": txn.card_id,
            "merchant_id": txn.merchant_id,
            "amount": txn.amount,
            "timestamp": txn.timestamp,
            "risk_score": round(prob, 6),
            "is_alert": bool(flags[0]),
            "alert_reason": ["none", "known-pattern", "novel-pattern", "known+novel"][int(reasons[0])],
            "threshold": round(self.scorer.threshold, 6),
            "components": {"supervised_prob": round(float(1 / (1 + np.exp(-logit))), 6),
                           "anomaly_score": round(anomaly, 6),
                           "anomaly_percentile": round(float(self.scorer.novelty_percentile(an)[0]), 6)},
            "subgraph": {"nodes": int(data.num_nodes), "edges": int(data.num_edges)},
            "latency_ms": {"graph_update": round((t1 - t0) * 1000, 3),
                           "model_inference": round((t2 - t1) * 1000, 3),
                           "scoring_total": round((t2 - t0) * 1000, 3),
                           "explanation": None},
            "explanation": None,
        }
        return result, data

    def attach_explanation(self, result: dict[str, Any], data: Data) -> dict[str, Any]:
        """Run GNNExplainer on an already-scored subgraph (safe to call outside the scoring lock)."""
        explanation = self.explainer.explain(data, self.graph)
        result["explanation"] = explanation
        result["latency_ms"]["explanation"] = explanation["explanation_ms"]
        return result

    def score(self, record: Union[dict[str, Any], Transaction], explain: Optional[bool] = None) -> dict[str, Any]:
        """Score one transaction; explain it if it is an alert and explanations are enabled."""
        result, data = self.score_core(record)
        do_explain = self.config.explain_alerts if explain is None else explain
        if result["is_alert"] and do_explain:
            self.attach_explanation(result, data)
        return result
