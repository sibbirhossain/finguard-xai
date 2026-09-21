"""Temporal GraphSAGE / GAT encoder with a transaction-level fraud head.

Edge timestamps are encoded with a learnable functional time encoding
(cos(w * dt + b), as in TGN). GraphSAGE aggregates time-encoded edge features
onto nodes before message passing; GAT consumes them directly via ``edge_dim``.
A transaction is represented by the embeddings of its card, device, merchant
and IP nodes plus transaction features.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import scatter

from .graph_engine import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, TXN_FEATURE_DIM


class TimeEncoder(nn.Module):
    """Functional time encoding: phi(dt) = cos(w * dt + b)."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(1, dim)
        with torch.no_grad():
            self.lin.weight.copy_((1.0 / 10 ** torch.linspace(0, 3, dim)).view(dim, 1))
            self.lin.bias.zero_()

    def forward(self, dt: Tensor) -> Tensor:
        return torch.cos(self.lin(dt))


class FinGuardGNN(nn.Module):
    """Two-layer temporal GNN producing node embeddings and a fraud logit."""

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        txn_dim: int = TXN_FEATURE_DIM,
        hidden: int = 64,
        time_dim: int = 8,
        arch: str = "sage",
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if arch not in {"sage", "gat"}:
            raise ValueError("arch must be 'sage' or 'gat'")
        self.config: dict[str, Any] = dict(
            node_dim=node_dim, edge_dim=edge_dim, txn_dim=txn_dim, hidden=hidden,
            time_dim=time_dim, arch=arch, heads=heads, dropout=dropout,
        )
        self.arch = arch
        self.dropout = dropout
        self.time_enc = TimeEncoder(time_dim)
        enc_edge_dim = edge_dim - 1 + time_dim
        if arch == "sage":
            self.conv1 = SAGEConv(node_dim + enc_edge_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        else:
            self.conv1 = GATConv(node_dim, hidden // heads, heads=heads, edge_dim=enc_edge_dim)
            self.conv2 = GATConv(hidden, hidden // heads, heads=heads, edge_dim=enc_edge_dim)
        self.rep_dim = 4 * hidden + txn_dim
        self.head = nn.Sequential(
            nn.Linear(self.rep_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def encode_edges(self, edge_attr: Tensor) -> Tensor:
        """Replace raw dt (column 1) with its time encoding."""
        return torch.cat([edge_attr[:, :1], self.time_enc(edge_attr[:, 1:2]), edge_attr[:, 2:]], dim=-1)

    def node_embeddings(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Compute node embeddings of shape [num_nodes, hidden]."""
        e = self.encode_edges(edge_attr)
        if self.arch == "sage":
            agg = scatter(e, edge_index[1], dim=0, dim_size=x.size(0), reduce="mean")
            h = torch.relu(self.conv1(torch.cat([x, agg], dim=-1), edge_index))
            h = nn.functional.dropout(h, self.dropout, self.training)
            return self.conv2(h, edge_index)
        h = nn.functional.elu(self.conv1(x, edge_index, e))
        h = nn.functional.dropout(h, self.dropout, self.training)
        return self.conv2(h, edge_index, e)

    def represent(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, seed_index: Tensor, txn_feat: Tensor
    ) -> Tensor:
        """Transaction representation [B, 4*hidden + txn_dim] used by the head and anomaly detector."""
        h = self.node_embeddings(x, edge_index, edge_attr)
        seeds = h[seed_index].reshape(seed_index.size(0), -1)
        return torch.cat([seeds, txn_feat], dim=-1)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, seed_index: Tensor, txn_feat: Tensor
    ) -> Tensor:
        """Return fraud logits of shape [B]."""
        return self.head(self.represent(x, edge_index, edge_attr, seed_index, txn_feat)).squeeze(-1)
