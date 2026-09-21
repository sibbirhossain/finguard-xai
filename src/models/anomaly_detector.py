"""Unsupervised novelty detection on GNN transaction representations, plus a
calibrated hybrid risk scorer.

* ``DeepSVDD`` (Ruff et al., ICML 2018) learns a hypersphere around legitimate
  transactions; distance from the centre is the anomaly score.
* ``AnomalyDetector`` wraps Deep SVDD or Isolation Forest with standardisation.
* ``HybridRiskScorer`` fuses the supervised logit and the anomaly score with a
  logistic-regression stacker fitted on a held-out validation window, so the
  fusion weights are learned and the output is a calibrated probability.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn


class DeepSVDD(nn.Module):
    """One-class Deep SVDD with a bias-free encoder (avoids hypersphere collapse)."""

    def __init__(self, in_dim: int, rep_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False), nn.LeakyReLU(), nn.Linear(hidden, rep_dim, bias=False)
        )
        self.register_buffer("center", torch.zeros(rep_dim))

    @torch.no_grad()
    def init_center(self, x: Tensor, eps: float = 0.1) -> None:
        """Set the centre to the mean initial representation, pushed away from 0."""
        c = self.net(x).mean(dim=0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.center.copy_(c)

    def forward(self, x: Tensor) -> Tensor:
        """Squared distance to the centre, shape [N]."""
        return ((self.net(x) - self.center) ** 2).sum(dim=-1)

    def fit(
        self, x: Tensor, epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-6,
        batch_size: int = 512, seed: int = 0,
    ) -> list[float]:
        """Train on (assumed) normal samples; return per-epoch mean loss."""
        torch.manual_seed(seed)
        self.init_center(x)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        history: list[float] = []
        self.train()
        for _ in range(epochs):
            perm = torch.randperm(x.size(0))
            total = 0.0
            for i in range(0, x.size(0), batch_size):
                batch = x[perm[i : i + batch_size]]
                loss = self(batch).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * batch.size(0)
            history.append(total / x.size(0))
        self.eval()
        return history


class AnomalyDetector:
    """Standardise representations, then score them with Deep SVDD or Isolation Forest."""

    def __init__(self, method: str = "deep_svdd", seed: int = 0) -> None:
        if method not in {"deep_svdd", "isolation_forest"}:
            raise ValueError("method must be 'deep_svdd' or 'isolation_forest'")
        self.method = method
        self.seed = seed
        self.scaler = StandardScaler()
        self.model: DeepSVDD | IsolationForest | None = None

    def fit(self, normal_reps: np.ndarray, epochs: int = 30) -> "AnomalyDetector":
        """Fit on representations of legitimate training transactions only."""
        z = self.scaler.fit_transform(normal_reps).astype(np.float32)
        if self.method == "deep_svdd":
            self.model = DeepSVDD(z.shape[1])
            self.model.fit(torch.from_numpy(z), epochs=epochs, seed=self.seed)
        else:
            self.model = IsolationForest(n_estimators=200, random_state=self.seed).fit(z)
        return self

    def score(self, reps: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        if self.model is None:
            raise RuntimeError("AnomalyDetector is not fitted")
        z = self.scaler.transform(reps).astype(np.float32)
        if isinstance(self.model, DeepSVDD):
            with torch.no_grad():
                return self.model(torch.from_numpy(z)).numpy()
        return -self.model.score_samples(z)


class HybridRiskScorer:
    """Dual-channel risk scoring.

    Channel 1 (known fraud): a logistic-regression stacker fuses the supervised
    GNN logit and the anomaly score into a calibrated probability, with an
    F1-optimal threshold chosen on validation data.

    Channel 2 (novel fraud): because a supervised stacker learns to down-weight
    the anomaly signal when validation data contains no novel fraud, a separate
    novelty channel alerts whenever the anomaly score exceeds the (1 - novelty_fpr)
    quantile of *legitimate* validation traffic - a fixed, auditable false-positive
    budget for fraud patterns never seen in training.
    """

    def __init__(self, novelty_fpr: float = 0.005) -> None:
        self.stacker = LogisticRegression(max_iter=1000)
        self.threshold: float = 0.5
        self.novelty_fpr = novelty_fpr
        self.novelty_threshold: float = float("inf")
        self._legit_sorted: np.ndarray = np.array([0.0])

    @staticmethod
    def _features(sup_logit: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        return np.column_stack([np.asarray(sup_logit), np.log1p(np.maximum(np.asarray(anomaly), 0.0))])

    def fit(self, sup_logit: np.ndarray, anomaly: np.ndarray, y: np.ndarray) -> "HybridRiskScorer":
        """Fit the stacker, the F1-optimal threshold and the novelty threshold on validation data."""
        self.stacker.fit(self._features(sup_logit, anomaly), y)
        prob = self.predict_proba(sup_logit, anomaly)
        precision, recall, thresholds = precision_recall_curve(y, prob)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        if len(thresholds):
            self.threshold = float(thresholds[int(np.argmax(f1[:-1]))])
        legit = np.sort(np.asarray(anomaly)[np.asarray(y) == 0])
        self._legit_sorted = legit
        self.novelty_threshold = float(np.quantile(legit, 1.0 - self.novelty_fpr))
        return self

    def predict_proba(self, sup_logit: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        """Calibrated known-fraud probability in [0, 1]."""
        return self.stacker.predict_proba(self._features(sup_logit, anomaly))[:, 1]

    def novelty_percentile(self, anomaly: np.ndarray) -> np.ndarray:
        """Fraction of legitimate validation traffic less anomalous than each input."""
        return np.searchsorted(self._legit_sorted, np.asarray(anomaly), side="right") / len(self._legit_sorted)

    def risk(self, sup_logit: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        """Final risk score: known-fraud probability, raised to the novelty percentile when novel."""
        prob = self.predict_proba(sup_logit, anomaly)
        novel = np.asarray(anomaly) >= self.novelty_threshold
        return np.where(novel, np.maximum(prob, self.novelty_percentile(anomaly)), prob)

    def alert(self, sup_logit: np.ndarray, anomaly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (alert flags, reason codes: 0 none, 1 supervised, 2 novelty, 3 both)."""
        sup = self.predict_proba(sup_logit, anomaly) >= self.threshold
        nov = np.asarray(anomaly) >= self.novelty_threshold
        return sup | nov, sup.astype(int) + 2 * nov.astype(int)

    @property
    def weights(self) -> dict[str, float]:
        """Learned fusion coefficients and thresholds (for transparency in reports)."""
        coef = self.stacker.coef_[0]
        return {"supervised_logit": float(coef[0]), "log_anomaly": float(coef[1]),
                "intercept": float(self.stacker.intercept_[0]), "novelty_fpr_budget": self.novelty_fpr}
