# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:00:51 · arch=sage · detector=deep_svdd · seed=1 · transactions=20762
- Test fraud rate: 2.34% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.528 | 0.957 | 0.821 | 0.315 | 0.455 | 0.0016 | 28 |
| anomaly_only | 0.437 | 0.918 | 0.293 | 0.562 | 0.385 | 0.0325 | 140 |
| gnn_only | 0.629 | 0.950 | 0.971 | 0.466 | 0.630 | 0.0003 | 35 |
| hybrid_gnn_plus_anomaly | 0.598 | 0.961 | 0.705 | 0.589 | 0.642 | 0.0059 | 61 |

Hybrid recall by scenario: `{"ato": 0.23684210526315788, "ring": 0.9714285714285714}`

Learned fusion weights: `{"supervised_logit": 1.0406636682320516, "log_anomaly": -0.004016532093044084, "intercept": -3.786341747552044, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 3.74 ms · p95 4.65 ms · p99 5.132 ms

**GNNExplainer latency** (alerts only, 7 explanations): p50 94.78 ms · p95 98.677 ms
