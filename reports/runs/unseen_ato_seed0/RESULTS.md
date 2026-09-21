# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:03:47 · arch=sage · detector=deep_svdd · seed=0 · transactions=20777
- Test fraud rate: 2.34% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.367 | 0.901 | 0.553 | 0.288 | 0.378 | 0.0056 | 38 |
| anomaly_only | 0.618 | 0.966 | 0.692 | 0.493 | 0.576 | 0.0053 | 52 |
| gnn_only | 0.735 | 0.982 | 0.914 | 0.438 | 0.593 | 0.0010 | 35 |
| hybrid_gnn_plus_anomaly | 0.688 | 0.989 | 0.691 | 0.644 | 0.667 | 0.0069 | 68 |

Hybrid recall by scenario: `{"ato": 0.3902439024390244, "ring": 0.96875}`

Learned fusion weights: `{"supervised_logit": 1.0347746014686567, "log_anomaly": 0.009857036402021523, "intercept": -3.0378179495177755, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 3.807 ms · p95 4.509 ms · p99 4.964 ms

**GNNExplainer latency** (alerts only, 8 explanations): p50 95.85 ms · p95 110.606 ms
