# FinGuard-XAI results

**OBSERVED IN THIS IMPLEMENTATION - SYNTHETIC DATA.** Single seed; see README limitations.

- Run: 2026-09-21 06:02:38 · arch=sage · detector=deep_svdd · seed=1 · transactions=20762
- Test fraud rate: 2.34% · chronological split 70/15/15 · thresholds chosen on validation only
- Environment: `{"python": "3.12.3", "torch": "2.14.0+cu130", "platform": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.39", "processor": "x86_64", "torch_threads": 1}`

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.708 | 0.983 | 0.727 | 0.548 | 0.625 | 0.0049 | 55 |
| anomaly_only | 0.617 | 0.968 | 0.754 | 0.589 | 0.662 | 0.0046 | 57 |
| gnn_only | 0.931 | 0.998 | 0.835 | 0.904 | 0.868 | 0.0043 | 79 |
| hybrid_gnn_plus_anomaly | 0.786 | 0.996 | 0.710 | 0.904 | 0.795 | 0.0089 | 93 |

Hybrid recall by scenario: `{"ato": 0.8157894736842105, "ring": 1.0}`

Learned fusion weights: `{"supervised_logit": 0.9341552306513374, "log_anomaly": 0.00044116077961943727, "intercept": -2.5403890360757155, "novelty_fpr_budget": 0.005}`

**Single-event scoring latency** (graph update + GNN + Deep SVDD + fusion, 200 events, warm graph, 1 CPU thread): p50 3.943 ms · p95 5.275 ms · p99 6.4 ms

**GNNExplainer latency** (alerts only, 8 explanations): p50 102.74 ms · p95 125.472 ms
