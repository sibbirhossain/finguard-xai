"""Local subgraph explanations with GNNExplainer (Ying et al., NeurIPS 2019).

Returns the entities and relations that most influenced a transaction's fraud
logit. Explanations describe model behaviour, not causality, and should be
reviewed by an analyst rather than treated as proof of fraud.
"""
from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.explain import Explainer, GNNExplainer

from .gnn_model import FinGuardGNN
from .graph_engine import RELATIONS, TemporalGraph, TxnGraph


class _SingleTxnWrapper(nn.Module):
    """Adapts FinGuardGNN to the (x, edge_index, **kwargs) signature Explainer expects."""

    def __init__(self, model: FinGuardGNN) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, seed_index: Tensor,
                txn_feat: Tensor) -> Tensor:
        return self.model(x, edge_index, edge_attr, seed_index, txn_feat)


class SubgraphExplainer:
    """Produce analyst-facing node/edge importances for one transaction subgraph."""

    def __init__(self, model: FinGuardGNN, epochs: int = 50, lr: float = 0.01) -> None:
        self.explainer = Explainer(
            model=_SingleTxnWrapper(model),
            algorithm=GNNExplainer(epochs=epochs, lr=lr),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type="object",
            model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"),
        )

    def explain(self, data: TxnGraph, graph: TemporalGraph, top_k_edges: int = 12) -> dict[str, Any]:
        """Explain one transaction; returns nodes, top edges and explanation latency."""
        start = time.perf_counter()
        exp = self.explainer(
            data.x, data.edge_index, edge_attr=data.edge_attr,
            seed_index=data.seed_index, txn_feat=data.txn_feat,
        )
        node_mask = exp.node_mask.detach().view(-1) if exp.node_mask is not None else torch.zeros(data.num_nodes)
        edge_mask = exp.edge_mask.detach() if exp.edge_mask is not None else torch.zeros(data.num_edges)

        # Edges are stored in both directions; merge each undirected pair.
        best: dict[tuple[int, int, int], float] = {}
        for i in range(data.num_edges):
            u, v = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            rel = int(data.edge_attr[i, 2:].argmax())
            key = (min(u, v), max(u, v), rel)
            best[key] = max(best.get(key, 0.0), float(edge_mask[i]))
        top = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k_edges]
        used = {n for (u, v, _), _ in top for n in (u, v)} | set(data.seed_index.view(-1).tolist())
        seeds = set(data.seed_index.view(-1).tolist())

        nodes = []
        for local in sorted(used):
            node_type, raw_id = graph.node_meta(int(data.n_id[local]))
            nodes.append({"id": f"{node_type}:{raw_id}", "type": node_type, "label": raw_id,
                          "importance": round(float(node_mask[local]), 4), "is_transaction_entity": local in seeds})
        edges = []
        for (u, v, rel), w in top:
            tu, ru = graph.node_meta(int(data.n_id[u]))
            tv, rv = graph.node_meta(int(data.n_id[v]))
            edges.append({"source": f"{tu}:{ru}", "target": f"{tv}:{rv}",
                          "relation": f"{RELATIONS[rel][0]}-{RELATIONS[rel][1]}", "weight": round(w, 4)})
        return {"method": "GNNExplainer", "nodes": nodes, "edges": edges,
                "explanation_ms": round((time.perf_counter() - start) * 1000, 2),
                "note": "Importances reflect model behaviour, not proof of fraud or causality."}
