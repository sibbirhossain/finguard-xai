# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:12:06 · arch=sage · detector=deep_svdd · seed=0 · transactions=10389
- Test fraud rate: 3.14% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.550 | 0.935 | 0.618 | 0.429 | 0.506 | 0.0086 | 34 |
| anomaly_only | 0.302 | 0.920 | 0.290 | 0.633 | 0.397 | 0.0503 | 107 |
| gnn_only | 0.856 | 0.993 | 0.702 | 0.816 | 0.755 | 0.0113 | 57 |
| hybrid_gnn_plus_anomaly | 0.567 | 0.987 | 0.606 | 0.816 | 0.696 | 0.0172 | 66 |

Hybrid recall by scenario: `{"ato": 0.9, "ring": 0.7586206896551724}`

Learned fusion weights: `{"supervised_logit": 0.7333258103949206, "log_anomaly": -0.008587344583516034, "intercept": -1.550011254785116, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 100 events, warm graph, 1 CPU thread): p50 3.701 ms · p95 4.781 ms · p99 5.0 ms

**GNNExplainer latency** (alerts only, 10 explanations): p50 107.545 ms · p95 119.701 ms
