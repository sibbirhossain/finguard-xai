# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:06:13 · arch=sage · detector=deep_svdd · seed=2 · transactions=20711
- Test fraud rate: 5.02% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.517 | 0.760 | 0.807 | 0.429 | 0.561 | 0.0054 | 83 |
| anomaly_only | 0.831 | 0.969 | 0.775 | 0.750 | 0.762 | 0.0115 | 151 |
| gnn_only | 0.847 | 0.955 | 0.966 | 0.737 | 0.836 | 0.0014 | 119 |
| hybrid_gnn_plus_anomaly | 0.815 | 0.964 | 0.716 | 0.808 | 0.759 | 0.0169 | 176 |

Hybrid recall by scenario: `{"ato": 0.2903225806451613, "ring": 0.936}`

Learned fusion weights: `{"supervised_logit": 1.3789909423776947, "log_anomaly": 0.0006050608233984876, "intercept": -2.7990577312484666, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 4.064 ms · p95 5.072 ms · p99 5.647 ms

**GNNExplainer latency** (alerts only, 12 explanations): p50 114.215 ms · p95 126.799 ms
