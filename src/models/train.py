"""Train and evaluate FinGuard-XAI on synthetic data with a chronological split.

Produces artifacts/ (model, detector, scorer, metadata) and reports/RESULTS.md.
All reported numbers are OBSERVED on SYNTHETIC data - not real-world performance.

Usage:
  python -m src.models.train                  # default run
  python -m src.models.train --arch gat --detector isolation_forest
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch_geometric.loader import DataLoader

from ..api.stream_simulator import generate_transactions
from ..data.ieee_cis import load_ieee_cis
from .anomaly_detector import AnomalyDetector, HybridRiskScorer
from .gnn_model import FinGuardGNN
from .graph_engine import TemporalGraph, Transaction, TxnGraph
from .pipeline import EngineConfig, FinGuardEngine, save_artifacts


def build_dataset(records: list[dict[str, Any]], cfg: EngineConfig) -> list[TxnGraph]:
    """Replay the stream through the temporal graph, building one subgraph per event."""
    graph = TemporalGraph(max_neighbors=cfg.max_neighbors)
    out: list[TxnGraph] = []
    for rec in records:
        txn = Transaction.from_dict(rec)
        seeds = graph.add_transaction(txn)
        out.append(graph.build_txn_graph(txn, seeds, cfg.num_hops, cfg.fanout, label=rec["label"]))
    return out


@torch.no_grad()
def predict(model: FinGuardGNN, graphs: list[TxnGraph], batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, representations) for a list of subgraphs."""
    model.eval()
    logits, reps = [], []
    for b in DataLoader(graphs, batch_size=batch_size):
        rep = model.represent(b.x, b.edge_index, b.edge_attr, b.seed_index, b.txn_feat)
        reps.append(rep)
        logits.append(model.head(rep).squeeze(-1))
    return torch.cat(logits).numpy(), torch.cat(reps).numpy()


def f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 (selected on validation data only)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[int(np.argmax(f1[:-1]))]) if len(t) else 0.5


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float | None = None,
             pred: np.ndarray | None = None) -> dict[str, float]:
    """Imbalance-aware metrics at a validation-selected threshold (or explicit decisions)."""
    pred = score >= threshold if pred is None else pred
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"pr_auc": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score)),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "alerts": tp + fp}


def tabular_features(graphs: list[TxnGraph]) -> np.ndarray:
    """Non-graph baseline: raw features of the 4 transaction entities + txn features (no message passing)."""
    return np.stack([torch.cat([g.x[g.seed_index[0]].reshape(-1), g.txn_feat[0]]).numpy() for g in graphs])


def train_gnn(model: FinGuardGNN, train: list[TxnGraph], epochs: int, lr: float, batch_size: int,
              seed: int) -> list[float]:
    """Class-weighted BCE training; returns per-epoch loss."""
    torch.manual_seed(seed)
    y = torch.tensor([float(g.y) for g in train])
    pos_weight = ((len(y) - y.sum()) / y.sum().clamp(min=1.0)).detach()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for b in DataLoader(train, batch_size=batch_size, shuffle=True):
            loss = loss_fn(model(b.x, b.edge_index, b.edge_attr, b.seed_index, b.txn_feat), b.y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * b.num_graphs
        history.append(total / len(train))
    return history


def latency_benchmark(engine: FinGuardEngine, records: list[dict[str, Any]], split: int,
                      n: int, n_explain: int) -> dict[str, Any]:
    """Warm the graph with history, then score events one at a time (single CPU thread)."""
    for rec in records[:split]:
        engine.graph.add_transaction(Transaction.from_dict(rec))
    scoring, explain = [], []
    for rec in records[split : split + n]:
        out = engine.score(rec, explain=len(explain) < n_explain)
        scoring.append(out["latency_ms"]["scoring_total"])
        if out["latency_ms"]["explanation"] is not None:
            explain.append(out["latency_ms"]["explanation"])
    q = lambda v, p: round(float(np.percentile(v, p)), 3) if v else None  # noqa: E731
    return {"n_scored": len(scoring), "scoring_ms": {"p50": q(scoring, 50), "p95": q(scoring, 95), "p99": q(scoring, 99)},
            "n_explained": len(explain), "explanation_ms": {"p50": q(explain, 50), "p95": q(explain, 95)}}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the full training/evaluation pipeline."""
    p = argparse.ArgumentParser()
    p.add_argument("--n-normal", type=int, default=30_000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--arch", choices=["sage", "gat"], default="sage")
    p.add_argument("--detector", choices=["deep_svdd", "isolation_forest"], default="deep_svdd")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--report", default="reports/RESULTS.md")
    p.add_argument("--bench-n", type=int, default=500)
    p.add_argument("--data", choices=["synthetic", "ieee-cis"], default="synthetic")
    p.add_argument("--data-dir", default="data/ieee-cis", help="folder with train_transaction.csv")
    p.add_argument("--max-rows", type=int, default=None, help="limit rows (ieee-cis)")
    p.add_argument("--holdout-scenario", default=None,
                   help="exclude this fraud scenario from train/validation to test detection of UNSEEN fraud")
    args = p.parse_args(argv)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(1)  # reflects single-core serving; set higher for faster training
    cfg = EngineConfig()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    if args.data == "ieee-cis":
        records = load_ieee_cis(args.data_dir, args.max_rows)
    else:
        records = generate_transactions(n_normal=args.n_normal, n_rings=max(3, args.n_normal // 1000),
                                        n_ato=max(10, args.n_normal // 200), seed=args.seed)
    graphs = build_dataset(records, cfg)
    timings["graph_build_s"] = time.perf_counter() - t
    n = len(graphs); a, b = int(0.70 * n), int(0.85 * n)
    y = np.array([r["label"] for r in records])
    scen = np.array([r["scenario"] for r in records])
    keep = np.ones(n, bool)
    if args.holdout_scenario:  # the held-out fraud type never appears in train/validation
        keep[:b] = scen[:b] != args.holdout_scenario
    tr_idx = np.where(keep[:a])[0]; va_idx = a + np.where(keep[a:b])[0]
    train = [graphs[i] for i in tr_idx]; val = [graphs[i] for i in va_idx]; test = graphs[b:]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[b:]
    for name, part in (("train", y_tr), ("validation", y_va), ("test", y_te)):
        if len(set(part.tolist())) < 2:
            raise ValueError(f"{name} split has only one class; use more data (e.g. --n-normal 20000)")

    model = FinGuardGNN(hidden=args.hidden, arch=args.arch)
    t = time.perf_counter()
    loss_history = train_gnn(model, train, args.epochs, args.lr, args.batch_size, args.seed)
    timings["gnn_train_s"] = time.perf_counter() - t

    lg_tr, rep_tr = predict(model, train); lg_va, rep_va = predict(model, val); lg_te, rep_te = predict(model, test)
    t = time.perf_counter()
    detector = AnomalyDetector(args.detector, seed=args.seed).fit(rep_tr[y_tr == 0])
    timings["anomaly_fit_s"] = time.perf_counter() - t
    an_va, an_te = detector.score(rep_va), detector.score(rep_te)
    scorer = HybridRiskScorer().fit(lg_va, an_va, y_va)

    tab = HistGradientBoostingClassifier(class_weight="balanced", random_state=args.seed)
    tab.fit(tabular_features(train), y_tr)
    tab_va, tab_te = tab.predict_proba(tabular_features(val))[:, 1], tab.predict_proba(tabular_features(test))[:, 1]

    results = {
        "tabular_gbdt_no_graph": evaluate(y_te, tab_te, f1_threshold(y_va, tab_va)),
        "anomaly_only": evaluate(y_te, an_te, f1_threshold(y_va, an_va)),
        "gnn_only": evaluate(y_te, lg_te, f1_threshold(y_va, lg_va)),
        "hybrid_gnn_plus_anomaly": evaluate(y_te, scorer.risk(lg_te, an_te), pred=scorer.alert(lg_te, an_te)[0]),
    }
    scen_te = scen[b:]
    hyb_pred, reasons = scorer.alert(lg_te, an_te)
    recall_by_scenario = {s: float(hyb_pred[scen_te == s].mean()) for s in sorted(set(scen_te) - {"normal"})}
    recall_by_scenario_gnn = {s: float((lg_te >= f1_threshold(y_va, lg_va))[scen_te == s].mean())
                              for s in sorted(set(scen_te) - {"normal"})}

    metadata = {"holdout_scenario": args.holdout_scenario, "dataset": args.data, "arch": args.arch, "detector": args.detector, "seed": args.seed, "epochs": args.epochs,
                "fusion_weights": scorer.weights, "threshold": scorer.threshold,
                "novelty_threshold": scorer.novelty_threshold,
                "data": "SYNTHETIC (src/api/stream_simulator.py)" if args.data == "synthetic" else "IEEE-CIS (public)", "n_transactions": n,
                "fraud_rate": {"train": float(y_tr.mean()), "val": float(y_va.mean()), "test": float(y_te.mean())}}
    save_artifacts(args.out, model, detector, scorer, cfg, metadata)

    engine = FinGuardEngine.load(args.out)
    bench = latency_benchmark(engine, records, b, args.bench_n, n_explain=min(20, args.bench_n))
    env = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
           "processor": platform.processor() or platform.machine(), "torch_threads": torch.get_num_threads()}
    label = ("OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA" if args.data == "synthetic"
             else "OBSERVED IN THIS IMPLEMENTATION - IEEE-CIS PUBLIC DATA")
    report = {"label": label, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "environment": env, "metadata": metadata, "results_test": results,
              "hybrid_recall_by_scenario": recall_by_scenario,
              "gnn_recall_by_scenario": recall_by_scenario_gnn, "latency": bench,
              "timings_s": {k: round(v, 2) for k, v in timings.items()}, "train_loss": loss_history}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).with_suffix(".json").write_text(json.dumps(report, indent=2))
    Path(args.report).write_text(render_markdown(report))
    print(render_markdown(report))
    return report


def render_markdown(r: dict[str, Any]) -> str:
    """Human-readable results report."""
    lines = ["# FinGuard-XAI results", "",
             f"**{r['label']}.** Single seed; see README limitations.", "",
             f"- Run: {r['timestamp']} · arch={r['metadata']['arch']} · detector={r['metadata']['detector']} "
             f"· seed={r['metadata']['seed']} · transactions={r['metadata']['n_transactions']}",
             f"- Test fraud rate: {r['metadata']['fraud_rate']['test']:.2%} · chronological split 70/15/15 "
             "· thresholds chosen on validation only",
             f"- Environment: `{json.dumps(r['environment'])}`", "",
             "| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |", "|---|---|---|---|---|---|---|---|"]
    for name, m in r["results_test"].items():
        lines.append(f"| {name} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} "
                     f"| {m['f1']:.3f} | {m['fpr']:.4f} | {m['alerts']} |")
    lat = r["latency"]
    lines += ["", f"Hybrid recall by scenario: `{json.dumps(r['hybrid_recall_by_scenario'])}`", "",
              f"Learned fusion weights: `{json.dumps(r['metadata']['fusion_weights'])}`", "",
              f"**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, {lat['n_scored']} events, "
              f"warm graph, 1 CPU thread): p50 {lat['scoring_ms']['p50']} ms · p95 {lat['scoring_ms']['p95']} ms "
              f"· p99 {lat['scoring_ms']['p99']} ms", "",
              f"**GNNExplainer latency** (alerts only, {lat['n_explained']} explanations): "
              f"p50 {lat['explanation_ms']['p50']} ms · p95 {lat['explanation_ms']['p95']} ms", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
