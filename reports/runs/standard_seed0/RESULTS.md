# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 05:57:33 · arch=sage · detector=deep_svdd · seed=0 · transactions=20777
- Test fraud rate: 2.34% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.570 | 0.941 | 0.397 | 0.712 | 0.510 | 0.0260 | 131 |
| anomaly_only | 0.516 | 0.958 | 0.322 | 0.671 | 0.436 | 0.0338 | 152 |
| gnn_only | 0.895 | 0.997 | 0.765 | 0.849 | 0.805 | 0.0062 | 81 |
| hybrid_gnn_plus_anomaly | 0.728 | 0.993 | 0.640 | 0.877 | 0.740 | 0.0118 | 100 |

Hybrid recall by scenario: `{"ato": 0.7804878048780488, "ring": 1.0}`

Learned fusion weights: `{"supervised_logit": 0.9556480216865529, "log_anomaly": 0.009812375556727168, "intercept": -2.9317913050551803, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 3.812 ms · p95 4.81 ms · p99 6.851 ms

**GNNExplainer latency** (alerts only, 9 explanations): p50 100.88 ms · p95 131.788 ms
