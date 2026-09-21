"""Adversarial robustness: how detection degrades when fraud rings actively evade.

The model is trained on clean history; attacks are applied ONLY to fraud-ring
activity in the future test window (the attacker adapts after deployment):

  * device_dispersion - every ring transaction uses a fresh device and IP,
                        erasing shared-infrastructure links (the key graph signal)
  * camouflage        - before each ring transaction, the ring card makes two
                        ordinary-looking purchases at popular merchants from its
                        usual device, diluting its neighborhood with benign edges
  * combined          - both at once

Usage:  python -m src.experiments.robustness --seed 0     (one seed per call)
        python -m src.experiments.robustness --aggregate
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..api.stream_simulator import generate_transactions
from ..models.anomaly_detector import AnomalyDetector, HybridRiskScorer
from ..models.gnn_model import FinGuardGNN
from ..models.graph_engine import TemporalGraph, Transaction, TxnGraph
from ..models.pipeline import EngineConfig
from ..models.train import build_dataset, f1_threshold, predict, train_gnn

ATTACKS = ("none", "device_dispersion", "camouflage", "combined")
OUT = Path("reports/robustness")


def apply_attack(records: list[dict[str, Any]], start: int, attack: str, seed: int) -> list[dict[str, Any]]:
    """Return a new stream where ring transactions at index >= start are perturbed."""
    if attack == "none":
        return copy.deepcopy(records)
    rng = np.random.default_rng(seed + 99)
    history = records[:start]
    last_device: dict[str, str] = {}
    last_ip: dict[str, str] = {}
    popular = [r["merchant_id"] for r in history if r["scenario"] == "normal"][-5000:]
    for r in history:
        last_device[r["card_id"]], last_ip[r["card_id"]] = r["device_id"], r["ip_id"]
    out = [dict(r) for r in history]
    for i, r in enumerate(records[start:]):
        r = dict(r)
        if r["scenario"] == "ring":
            if attack in ("camouflage", "combined"):
                for k in range(2):
                    out.append(dict(txn_id=f"{r['txn_id']}_cf{k}", timestamp=r["timestamp"] - 60.0 * (k + 1),
                                    card_id=r["card_id"], device_id=last_device.get(r["card_id"], r["device_id"]),
                                    ip_id=last_ip.get(r["card_id"], r["ip_id"]),
                                    merchant_id=str(rng.choice(popular)),
                                    amount=round(float(rng.lognormal(3.6, 0.8)), 2), label=0, scenario="camouflage"))
            if attack in ("device_dispersion", "combined"):
                r["device_id"], r["ip_id"] = f"dev_adv{i}", f"ip_adv{i}"
        out.append(r)
    out[start:] = sorted(out[start:], key=lambda x: x["timestamp"])
    return out


def build_test_graphs(stream: list[dict[str, Any]], start: int, cfg: EngineConfig) -> list[TxnGraph]:
    """Replay history into the graph (no samples), then build subgraphs for the test window only."""
    graph = TemporalGraph(max_neighbors=cfg.max_neighbors)
    for rec in stream[:start]:
        graph.add_transaction(Transaction.from_dict(rec))
    out = []
    for rec in stream[start:]:
        txn = Transaction.from_dict(rec)
        out.append(graph.build_txn_graph(txn, graph.add_transaction(txn), cfg.num_hops, cfg.fanout))
    return out


def run_seed(seed: int, n_normal: int = 20_000, epochs: int = 6) -> dict[str, Any]:
    """Train once on clean data, then evaluate the frozen system under each attack."""
    torch.set_num_threads(1)
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = EngineConfig()
    records = generate_transactions(n_normal=n_normal, n_rings=max(3, n_normal // 1000),
                                    n_ato=max(10, n_normal // 200), seed=seed)
    graphs = build_dataset(records, cfg)
    n = len(graphs); a, b = int(0.70 * n), int(0.85 * n)
    y = np.array([r["label"] for r in records])
    model = FinGuardGNN()
    train_gnn(model, graphs[:a], epochs, 3e-3, 256, seed)
    lg_tr, rep_tr = predict(model, graphs[:a]); lg_va, rep_va = predict(model, graphs[a:b])
    detector = AnomalyDetector("deep_svdd", seed=seed).fit(rep_tr[y[:a] == 0])
    an_va = detector.score(rep_va)
    scorer = HybridRiskScorer().fit(lg_va, an_va, y[a:b])
    gnn_thr = f1_threshold(y[a:b], lg_va)

    result: dict[str, Any] = {"seed": seed}
    for attack in ATTACKS:
        stream = apply_attack(records, b, attack, seed)
        test_graphs = build_test_graphs(stream, b, cfg)
        scen = np.array([r["scenario"] for r in stream[b:]])
        lg, rep = predict(model, test_graphs)
        an = detector.score(rep)
        dual, _ = scorer.alert(lg, an)
        gnn = lg >= gnn_thr
        ring, legit = scen == "ring", scen == "normal"
        result[attack] = {
            "gnn_ring_recall": float(gnn[ring].mean()), "dual_ring_recall": float(dual[ring].mean()),
            "gnn_fpr": float(gnn[legit].mean()), "dual_fpr": float(dual[legit].mean()),
            "n_ring": int(ring.sum()),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seed{seed}.json").write_text(json.dumps(result, indent=2))
    return result


def aggregate() -> str:
    """Summarise all finished seeds into reports/robustness/ROBUSTNESS.md."""
    runs = [json.loads(p.read_text()) for p in sorted(OUT.glob("seed*.json"))]
    lines = ["# Adversarial robustness (OBSERVED — SYNTHETIC DATA)", "",
             f"Seeds: {[r['seed'] for r in runs]}. The model is trained on clean history; the attack is applied only to "
             "fraud-ring activity in the future test window. Values: mean ± std across seeds.", "",
             "| Attack on fraud rings | Ring recall — GNN only | Ring recall — GNN + dual-channel | FPR — GNN | FPR — dual |",
             "|---|---|---|---|---|"]
    summary: dict[str, Any] = {}
    for att in ATTACKS:
        stats = {k: (float(np.mean([r[att][k] for r in runs])), float(np.std([r[att][k] for r in runs], ddof=1)))
                 for k in ("gnn_ring_recall", "dual_ring_recall", "gnn_fpr", "dual_fpr")}
        summary[att] = stats
        f = lambda k, d=3: f"{stats[k][0]:.{d}f} ± {stats[k][1]:.{d}f}"  # noqa: E731
        lines.append(f"| {att} | {f('gnn_ring_recall')} | {f('dual_ring_recall')} | {f('gnn_fpr', 4)} | {f('dual_fpr', 4)} |")
    (OUT / "ROBUSTNESS.json").write_text(json.dumps(summary, indent=2))
    md = "\n".join(lines) + "\n"
    (OUT / "ROBUSTNESS.md").write_text(md)
    return md


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, nargs="*", default=[])
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()
    for s in args.seed:
        print(json.dumps(run_seed(s), indent=1))
    if args.aggregate:
        print(aggregate())


if __name__ == "__main__":
    main()
