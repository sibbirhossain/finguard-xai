# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:10:59 · arch=sage · detector=deep_svdd · seed=4 · transactions=20763
- Test fraud rate: 3.08% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.153 | 0.667 | 0.205 | 0.188 | 0.196 | 0.0232 | 88 |
| anomaly_only | 0.819 | 0.979 | 0.537 | 0.823 | 0.650 | 0.0225 | 147 |
| gnn_only | 0.772 | 0.953 | 0.887 | 0.656 | 0.754 | 0.0026 | 71 |
| hybrid_gnn_plus_anomaly | 0.828 | 0.970 | 0.785 | 0.760 | 0.772 | 0.0066 | 93 |

Hybrid recall by scenario: `{"ato": 0.36666666666666664, "ring": 0.9393939393939394}`

Learned fusion weights: `{"supervised_logit": 0.8914959061842909, "log_anomaly": -0.006008537052905441, "intercept": -3.873340252312652, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 3.812 ms · p95 4.701 ms · p99 5.027 ms

**GNNExplainer latency** (alerts only, 7 explanations): p50 105.9 ms · p95 116.044 ms
