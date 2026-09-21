"""Multi-relational temporal transaction graph.

Each transaction links four entity nodes (card, device, merchant, IP) with
timestamped, typed edges. The graph is updated incrementally (streaming) and
exposes point-in-time k-hop subgraphs around a transaction for GNN inference.
Neighbour lists keep only the most recent ``max_neighbors`` interactions per
node, which bounds memory and inference latency (TGN-style recent sampling).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Any, Optional

import torch
from torch_geometric.data import Data

NODE_TYPES: tuple[str, ...] = ("card", "device", "merchant", "ip")
RELATIONS: tuple[tuple[str, str], ...] = (
    ("card", "device"),
    ("card", "merchant"),
    ("card", "ip"),
    ("device", "ip"),
)
NODE_FEATURE_DIM: int = len(NODE_TYPES) + 5
EDGE_FEATURE_DIM: int = 2 + len(RELATIONS)  # [log_amount, dt_hours, relation one-hot]
TXN_FEATURE_DIM: int = 3  # [log_amount, sin(hour), cos(hour)]


@dataclass(frozen=True)
class Transaction:
    """A single payment event. ``timestamp`` is in seconds."""

    txn_id: str
    timestamp: float
    card_id: str
    device_id: str
    merchant_id: str
    ip_id: str
    amount: float

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Transaction":
        """Build a transaction from a dict, ignoring extra keys (e.g. labels)."""
        return cls(**{name: record[name] for name in cls.__dataclass_fields__})


class TxnGraph(Data):
    """PyG ``Data`` for one transaction subgraph.

    ``seed_index`` ([1, 4]) holds the local indices of the transaction's card,
    device, merchant and IP nodes. It is concatenated along dim 0 and offset by
    ``num_nodes`` when batched, so ``x[seed_index]`` works on batches too.
    """

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "seed_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class TemporalGraph:
    """Incrementally updated heterogeneous graph of payment entities."""

    def __init__(self, max_neighbors: int = 64, time_scale: float = 3600.0) -> None:
        self.max_neighbors = max_neighbors
        self.time_scale = time_scale
        self._index: dict[tuple[str, str], int] = {}
        self._meta: list[tuple[str, str]] = []
        self._type_id: list[int] = []
        self._count: list[int] = []
        self._sum_log_amount: list[float] = []
        self._first_seen: list[float] = []
        self._last_seen: list[float] = []
        self._distinct: list[set[int]] = []
        self._adj: list[deque[tuple[int, float, float, int]]] = []

    @property
    def num_nodes(self) -> int:
        """Number of entity nodes seen so far."""
        return len(self._meta)

    def node_meta(self, node: int) -> tuple[str, str]:
        """Return ``(node_type, raw_id)`` for a global node index."""
        return self._meta[node]

    def _node(self, node_type: str, raw_id: str, ts: float) -> int:
        key = (node_type, raw_id)
        idx = self._index.get(key)
        if idx is None:
            idx = len(self._meta)
            self._index[key] = idx
            self._meta.append(key)
            self._type_id.append(NODE_TYPES.index(node_type))
            self._count.append(0)
            self._sum_log_amount.append(0.0)
            self._first_seen.append(ts)
            self._last_seen.append(ts)
            self._distinct.append(set())
            self._adj.append(deque(maxlen=self.max_neighbors))
        return idx

    def add_transaction(self, txn: Transaction) -> list[int]:
        """Insert a transaction; return global ids of [card, device, merchant, ip]."""
        ts = float(txn.timestamp)
        ids = [
            self._node("card", txn.card_id, ts),
            self._node("device", txn.device_id, ts),
            self._node("merchant", txn.merchant_id, ts),
            self._node("ip", txn.ip_id, ts),
        ]
        by_type = dict(zip(NODE_TYPES, ids))
        log_amount = math.log1p(max(txn.amount, 0.0))
        for rel, (src_t, dst_t) in enumerate(RELATIONS):
            u, v = by_type[src_t], by_type[dst_t]
            self._adj[u].append((v, ts, log_amount, rel))
            self._adj[v].append((u, ts, log_amount, rel))
            self._distinct[u].add(v)
            self._distinct[v].add(u)
        for node in ids:
            self._count[node] += 1
            self._sum_log_amount[node] += log_amount
            self._last_seen[node] = ts
        return ids

    def node_features(self, nodes: list[int], now: float) -> torch.Tensor:
        """Per-node features: type one-hot, activity, amount, fan-out, recency, age."""
        feats = torch.zeros(len(nodes), NODE_FEATURE_DIM)
        k = len(NODE_TYPES)
        for row, n in enumerate(nodes):
            cnt = self._count[n]
            feats[row, self._type_id[n]] = 1.0
            feats[row, k] = math.log1p(cnt)
            feats[row, k + 1] = self._sum_log_amount[n] / cnt if cnt else 0.0
            feats[row, k + 2] = math.log1p(len(self._distinct[n]))
            feats[row, k + 3] = math.log1p(max(now - self._last_seen[n], 0.0) / self.time_scale)
            feats[row, k + 4] = math.log1p(max(now - self._first_seen[n], 0.0) / self.time_scale)
        return feats

    @staticmethod
    def txn_features(txn: Transaction) -> torch.Tensor:
        """Transaction-level features, shape [1, TXN_FEATURE_DIM]."""
        hour = (txn.timestamp / 3600.0) % 24.0
        angle = 2.0 * math.pi * hour / 24.0
        return torch.tensor([[math.log1p(max(txn.amount, 0.0)), math.sin(angle), math.cos(angle)]])

    def build_txn_graph(
        self,
        txn: Transaction,
        seeds: list[int],
        num_hops: int = 2,
        fanout: int = 8,
        label: Optional[int] = None,
    ) -> TxnGraph:
        """Extract the point-in-time k-hop subgraph around a transaction's entities.

        Only interactions already inserted into the graph are visible, so no
        future information leaks into the features.
        """
        now = float(txn.timestamp)
        local: dict[int, int] = {}
        order: list[int] = []
        for s in seeds:
            if s not in local:
                local[s] = len(order)
                order.append(s)
        seen_edges: set[tuple[int, int, float, int]] = set()
        edges: list[tuple[int, int, float, float, int]] = []
        frontier = list(dict.fromkeys(seeds))
        for _ in range(num_hops):
            nxt: list[int] = []
            for u in frontier:
                for v, ts, log_amount, rel in islice(reversed(self._adj[u]), fanout):
                    key = (min(u, v), max(u, v), ts, rel)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    if v not in local:
                        local[v] = len(order)
                        order.append(v)
                        nxt.append(v)
                    edges.append((local[u], local[v], ts, log_amount, rel))
            frontier = nxt
        if edges:
            src = [e[0] for e in edges] + [e[1] for e in edges]
            dst = [e[1] for e in edges] + [e[0] for e in edges]
            attr = torch.zeros(len(edges), EDGE_FEATURE_DIM)
            for i, (_, _, ts, log_amount, rel) in enumerate(edges):
                attr[i, 0] = log_amount
                attr[i, 1] = math.log1p(max(now - ts, 0.0) / self.time_scale)
                attr[i, 2 + rel] = 1.0
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.cat([attr, attr], dim=0)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, EDGE_FEATURE_DIM)
        data = TxnGraph(
            x=self.node_features(order, now),
            edge_index=edge_index,
            edge_attr=edge_attr,
            seed_index=torch.tensor([[local[s] for s in seeds]], dtype=torch.long),
            txn_feat=self.txn_features(txn),
            n_id=torch.tensor(order, dtype=torch.long),
        )
        if label is not None:
            data.y = torch.tensor([float(label)])
        return data
