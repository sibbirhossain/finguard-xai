# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:05:02 · arch=sage · detector=deep_svdd · seed=2 · transactions=20711
- Test fraud rate: 5.02% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.898 | 0.974 | 0.931 | 0.782 | 0.850 | 0.0030 | 131 |
| anomaly_only | 0.878 | 0.979 | 0.891 | 0.731 | 0.803 | 0.0047 | 128 |
| gnn_only | 0.971 | 0.998 | 0.901 | 0.929 | 0.915 | 0.0054 | 161 |
| hybrid_gnn_plus_anomaly | 0.941 | 0.997 | 0.808 | 0.942 | 0.870 | 0.0119 | 182 |

Hybrid recall by scenario: `{"ato": 0.8387096774193549, "ring": 0.968}`

Learned fusion weights: `{"supervised_logit": 1.3170671915121024, "log_anomaly": 0.002857278726158489, "intercept": -2.9030613675669596, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 3.793 ms · p95 4.57 ms · p99 4.799 ms

**GNNExplainer latency** (alerts only, 7 explanations): p50 95.75 ms · p95 107.226 ms
