# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:09:47 · arch=sage · detector=deep_svdd · seed=4 · transactions=20763
- Test fraud rate: 3.08% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.606 | 0.958 | 0.279 | 0.781 | 0.411 | 0.0643 | 269 |
| anomaly_only | 0.812 | 0.992 | 0.700 | 0.802 | 0.748 | 0.0109 | 110 |
| gnn_only | 0.956 | 0.998 | 0.809 | 0.927 | 0.864 | 0.0070 | 110 |
| hybrid_gnn_plus_anomaly | 0.864 | 0.996 | 0.692 | 0.958 | 0.803 | 0.0136 | 133 |

Hybrid recall by scenario: `{"ato": 0.9, "ring": 0.9848484848484849}`

Learned fusion weights: `{"supervised_logit": 0.8469826744304164, "log_anomaly": -0.004023409883144284, "intercept": -2.3688966428292026, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 3.816 ms · p95 4.662 ms · p99 4.791 ms

**GNNExplainer latency** (alerts only, 11 explanations): p50 101.38 ms · p95 113.97 ms
