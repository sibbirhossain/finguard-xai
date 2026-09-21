# Adversarial robustness (OBSERVED — SYNTHETIC DATA)

Seeds: [0, 1, 2, 3, 4]. The model is trained on clean history; the attack is applied only to fraud-ring activity in the future test window. Values: mean ± std across seeds.

| Attack on fraud rings | Ring recall — GNN only | Ring recall — GNN + dual-channel | FPR — GNN | FPR — dual |
|---|---|---|---|---|
| none | 0.973 ± 0.037 | 0.973 ± 0.037 | 0.0060 ± 0.0011 | 0.0128 ± 0.0016 |
| device_dispersion | 0.981 ± 0.026 | 0.988 ± 0.018 | 0.0059 ± 0.0012 | 0.0136 ± 0.0017 |
| camouflage | 0.918 ± 0.043 | 0.968 ± 0.037 | 0.0066 ± 0.0013 | 0.0143 ± 0.0018 |
| combined | 0.839 ± 0.201 | 0.976 ± 0.030 | 0.0066 ± 0.0016 | 0.0144 ± 0.0017 |
