# Model Card — FinGuard-XAI v1.1

**Model.** A 2-layer temporal GraphSAGE (or GAT) with a fraud head, a Deep SVDD novelty detector on its embeddings, a logistic-regression fusion stacker, and a novelty threshold.

**Intended use.** Prioritizing payment transactions for human fraud review, and researching graph-based fraud detection. **Not intended** for automatic account closure, credit decisions, or any decision without human review.

**Inputs.** Pseudonymous card, device, merchant, and network identifiers; amount; timestamp. No names, addresses, or other direct personal data.

**Training data.** Synthetic transactions with documented scenarios: normal behavior, fraud rings using compromised devices and IPs, and account takeover (`src/api/stream_simulator.py`). The public IEEE-CIS loader is provided (`src/data/ieee_cis.py`).

**Evaluation.** Chronological 70/15/15 split; 5 seeds; PR-AUC, recall, precision, FPR; per-scenario recall; unseen-fraud holdout. See `reports/EXPERIMENTS.md`.

**Known limitations.**
- The synthetic data encodes the generator's assumptions.
- The novelty channel raises FPR (default budget 0.5% of legitimate traffic).
- Explanations are model rationales, not causal evidence.
- Cold start: new entities have little graph context.
- Not yet evaluated against adaptive adversaries.

**Fairness considerations.** No protected attributes are used. However, device, network, and merchant signals can correlate with geography or income, so deployments should monitor alert rates across customer segments.

**Security.** Load only self-trained artifacts, since deserialization can execute code. Use API-key auth, input validation, and restricted CORS.
