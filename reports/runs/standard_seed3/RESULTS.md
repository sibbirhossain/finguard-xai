# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:07:26 · arch=sage · detector=deep_svdd · seed=3 · transactions=20749
- Test fraud rate: 3.12% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.724 | 0.935 | 0.481 | 0.773 | 0.593 | 0.0269 | 156 |
| anomaly_only | 0.562 | 0.953 | 0.379 | 0.763 | 0.507 | 0.0401 | 195 |
| gnn_only | 0.910 | 0.996 | 0.800 | 0.866 | 0.832 | 0.0070 | 105 |
| hybrid_gnn_plus_anomaly | 0.762 | 0.994 | 0.720 | 0.876 | 0.791 | 0.0109 | 118 |

Hybrid recall by scenario: `{"ato": 0.8292682926829268, "ring": 0.9107142857142857}`

Learned fusion weights: `{"supervised_logit": 0.906907530372226, "log_anomaly": 0.0006104710827437969, "intercept": -2.2704252474278577, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 3.774 ms · p95 4.394 ms · p99 4.694 ms

**GNNExplainer latency** (alerts only, 9 explanations): p50 100.56 ms · p95 104.31 ms
