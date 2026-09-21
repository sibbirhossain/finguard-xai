"""Fast end-to-end tests (tiny synthetic run)."""
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.api.stream_simulator import generate_transactions
from src.models.graph_engine import TemporalGraph, Transaction
from src.models.train import main as train_main


def test_generator_is_imbalanced_and_ordered() -> None:
    rows = generate_transactions(n_normal=2000, n_rings=3, n_ato=10, seed=1)
    y = np.array([r["label"] for r in rows])
    assert 0 < y.mean() < 0.2
    assert all(a["timestamp"] <= b["timestamp"] for a, b in zip(rows, rows[1:]))


def test_subgraph_is_point_in_time() -> None:
    rows = generate_transactions(n_normal=500, n_rings=1, n_ato=2, seed=2)
    g = TemporalGraph()
    for r in rows[:300]:
        g.add_transaction(Transaction.from_dict(r))
    txn = Transaction.from_dict(rows[300])
    data = g.build_txn_graph(txn, g.add_transaction(txn))
    assert data.seed_index.shape == (1, 4)
    assert (data.edge_attr[:, 1] >= 0).all()  # no edge from the future


def test_train_and_api(tmp_path: Path, monkeypatch) -> None:
    art = tmp_path / "art"
    train_main(["--n-normal", "1500", "--epochs", "1", "--bench-n", "20", "--out", str(art),
                "--report", str(tmp_path / "R.md")])
    monkeypatch.setenv("FINGUARD_ARTIFACTS", str(art))
    import importlib
    import src.api.main as api
    importlib.reload(api)
    with TestClient(api.app) as client:
        rec = generate_transactions(n_normal=200, n_rings=1, n_ato=1, seed=9)[0]
        body = client.post("/api/v1/transactions", json=rec).json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["latency_ms"]["scoring_total"] > 0
        assert client.get("/api/v1/stats").json()["processed"] == 1
