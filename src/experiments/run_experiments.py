"""Reproducible experiment suite: multi-seed benchmark + unseen-fraud generalization.

Usage:  python -m src.experiments.run_experiments --seeds 0 1 2 3 4
Writes reports/EXPERIMENTS.md and reports/EXPERIMENTS.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..models.train import main as train_main

MODELS = ["tabular_gbdt_no_graph", "anomaly_only", "gnn_only", "hybrid_gnn_plus_anomaly"]


def _run(seed: int, extra: list[str], tag: str, n_normal: int, epochs: int) -> dict[str, Any]:
    out = f"reports/runs/{tag}_seed{seed}"
    done = Path(out) / "RESULTS.json"
    if done.exists():  # resumable: reuse finished runs
        return json.loads(done.read_text())
    return train_main(["--seed", str(seed), "--n-normal", str(n_normal), "--epochs", str(epochs),
                       "--out", f"artifacts/runs/{tag}_seed{seed}", "--report", f"{out}/RESULTS.md",
                       "--bench-n", "200", *extra])


def _ci(values: list[float]) -> tuple[float, float, float]:
    """Mean, std and 95% bootstrap CI of the mean across seeds."""
    v = np.array(values)
    rng = np.random.default_rng(0)
    boots = rng.choice(v, (5000, len(v))).mean(axis=1)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, *np.percentile(boots, [2.5, 97.5])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-normal", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=6)
    args = p.parse_args()
    start = time.time()
    suites = {"standard": [], "unseen_ato": []}
    for s in args.seeds:
        suites["standard"].append(_run(s, [], "standard", args.n_normal, args.epochs))
        suites["unseen_ato"].append(_run(s, ["--holdout-scenario", "ato"], "unseen_ato", args.n_normal, args.epochs))

    summary: dict[str, Any] = {"seeds": args.seeds, "n_normal": args.n_normal, "epochs": args.epochs}
    md = ["# Experiment suite (OBSERVED — SYNTHETIC DATA)", "",
          f"Seeds: {args.seeds} · {args.n_normal:,} legitimate + injected fraud per seed · chronological 70/15/15 split · "
          "mean ± std and 95% bootstrap CI across seeds.", ""]
    for suite, runs in suites.items():
        summary[suite] = {}
        title = ("## 1. Standard benchmark (all fraud types seen in training)" if suite == "standard" else
                 "## 2. Unseen-fraud generalization (account-takeover removed from training and validation)")
        md += [title, "", "| Model | PR-AUC | Recall | Precision | FPR |", "|---|---|---|---|---|"]
        for m in MODELS:
            row = {k: _ci([r["results_test"][m][k] for r in runs]) for k in ("pr_auc", "recall", "precision", "fpr")}
            summary[suite][m] = row
            f = lambda k: f"{row[k][0]:.3f} ± {row[k][1]:.3f} [{row[k][2]:.3f}, {row[k][3]:.3f}]"  # noqa: E731
            md.append(f"| {m} | {f('pr_auc')} | {f('recall')} | {f('precision')} | {row['fpr'][0]:.4f} |")
        for who in ("gnn", "hybrid"):
            key = f"{who}_recall_by_scenario"
            scen = sorted(runs[0][key])
            vals = {s: _ci([r[key][s] for r in runs]) for s in scen}
            summary[suite][key] = vals
            md.append("")
            md.append(f"{who.upper()} recall by fraud type: " + " · ".join(
                f"{s} {v[0]:.3f} ± {v[1]:.3f}" for s, v in vals.items()))
        md.append("")
    lat = [r["latency"]["scoring_ms"]["p95"] for r in suites["standard"]]
    md += [f"Scoring latency p95 across seeds: {np.mean(lat):.2f} ± {np.std(lat, ddof=1):.2f} ms (1 CPU thread).", "",
           f"Total runtime: {(time.time() - start) / 60:.1f} min."]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/EXPERIMENTS.json").write_text(json.dumps(summary, indent=2, default=float))
    Path("reports/EXPERIMENTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
