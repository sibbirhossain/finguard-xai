# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:08:38 · arch=sage · detector=deep_svdd · seed=3 · transactions=20749
- Test fraud rate: 3.12% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.629 | 0.929 | 0.561 | 0.567 | 0.564 | 0.0143 | 98 |
| anomaly_only | 0.494 | 0.952 | 0.409 | 0.629 | 0.496 | 0.0292 | 149 |
| gnn_only | 0.801 | 0.984 | 0.824 | 0.629 | 0.713 | 0.0043 | 74 |
| hybrid_gnn_plus_anomaly | 0.736 | 0.988 | 0.722 | 0.722 | 0.722 | 0.0090 | 97 |

Hybrid recall by scenario: `{"ato": 0.3902439024390244, "ring": 0.9642857142857143}`

Learned fusion weights: `{"supervised_logit": 1.3287129744251316, "log_anomaly": 0.010975985457421025, "intercept": -4.887711032344505, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 4.245 ms · p95 5.097 ms · p99 5.617 ms

**GNNExplainer latency** (alerts only, 5 explanations): p50 123.19 ms · p95 125.698 ms
