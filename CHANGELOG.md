# Changelog

## v1.2.0
- Adversarial robustness study (`src/experiments/robustness.py`, notebook 08): device dispersion, camouflage, and combined attacks over 5 seeds.
- PSI drift monitor (`src/monitoring/drift.py`) with a `/api/v1/drift` endpoint.
- Preprint (`paper/finguard_xai.pdf`) generated directly from experiment logs.
- Fixes: added pandas, matplotlib, and networkx to requirements; root `conftest.py` so `pytest` works in CI; corrected `CITATION.cff`.

## v1.1.0
- Dual-channel alerting (known-pattern + budgeted novelty channel).
- 5-seed experiment suite with bootstrap CIs; unseen-fraud (ATO holdout) generalization study.
- IEEE-CIS public-dataset loader; 7 Colab notebooks with committed outputs; impact analysis and model card.

## v1.0.0
- Streaming temporal graph, temporal GraphSAGE/GAT, Deep SVDD, GNNExplainer, FastAPI service, React dashboard, CI.
