# Experiment suite (OBSERVED — SYNTHETIC DATA)

Seeds: [0, 1, 2, 3, 4] · 20,000 legitimate + injected fraud per seed · chronological 70/15/15 split · mean ± std and 95% bootstrap CI across seeds.

## 1. Standard benchmark (all fraud types seen in training)

| Model | PR-AUC | Recall | Precision | FPR |
|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.701 ± 0.128 [0.608, 0.805] | 0.719 ± 0.100 [0.628, 0.780] | 0.563 ± 0.263 [0.366, 0.768] | 0.0250 |
| anomaly_only | 0.677 ± 0.159 [0.555, 0.801] | 0.711 ± 0.083 [0.640, 0.772] | 0.609 ± 0.247 [0.420, 0.798] | 0.0188 |
| gnn_only | 0.933 ± 0.032 [0.908, 0.957] | 0.895 ± 0.036 [0.867, 0.923] | 0.822 ± 0.051 [0.786, 0.862] | 0.0060 |
| hybrid_gnn_plus_anomaly | 0.816 ± 0.086 [0.753, 0.883] | 0.912 ± 0.038 [0.882, 0.941] | 0.714 ± 0.061 [0.668, 0.767] | 0.0114 |

GNN recall by fraud type: ato 0.785 ± 0.034 · ring 0.973 ± 0.037

HYBRID recall by fraud type: ato 0.833 ± 0.044 · ring 0.973 ± 0.037

## 2. Unseen-fraud generalization (account-takeover removed from training and validation)

| Model | PR-AUC | Recall | Precision | FPR |
|---|---|---|---|---|
| tabular_gbdt_no_graph | 0.439 ± 0.185 [0.281, 0.566] | 0.357 ± 0.145 [0.248, 0.467] | 0.589 ± 0.251 [0.396, 0.765] | 0.0100 |
| anomaly_only | 0.640 ± 0.182 [0.507, 0.784] | 0.651 ± 0.135 [0.548, 0.756] | 0.541 ± 0.198 [0.389, 0.694] | 0.0202 |
| gnn_only | 0.757 ± 0.083 [0.692, 0.816] | 0.585 ± 0.128 [0.487, 0.683] | 0.913 ± 0.061 [0.865, 0.958] | 0.0019 |
| hybrid_gnn_plus_anomaly | 0.733 ± 0.095 [0.659, 0.805] | 0.705 ± 0.088 [0.637, 0.772] | 0.724 ± 0.036 [0.701, 0.755] | 0.0091 |

GNN recall by fraud type: ato 0.059 ± 0.067 · ring 0.948 ± 0.028

HYBRID recall by fraud type: ato 0.335 ± 0.068 · ring 0.956 ± 0.017

Scoring latency p95 across seeds: 4.74 ± 0.33 ms (1 CPU thread).

Total runtime: 0.0 min.
