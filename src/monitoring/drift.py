"""Population Stability Index (PSI) drift monitor.

Compares the live distribution of a model signal (default: the Deep SVDD
anomaly score of the GNN embedding) against its distribution on legitimate
validation traffic. Rising PSI means incoming traffic no longer resembles what
the model was calibrated on, a signal to investigate and possibly retrain.
Conventional reading: PSI < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class DriftMonitor:
    """Rolling-window PSI against a fixed reference sample."""

    def __init__(self, reference: np.ndarray, bins: int = 10, window: int = 2000, min_samples: int = 200) -> None:
        ref = np.asarray(reference, dtype=float)
        # quantile bin edges from the reference -> each reference bin holds ~1/bins of mass
        self.edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
        self.ref_frac = self._fractions(ref)
        self.window: deque[float] = deque(maxlen=window)
        self.min_samples = min_samples

    def _fractions(self, values: np.ndarray) -> np.ndarray:
        counts = np.bincount(np.searchsorted(self.edges, values, side="right"), minlength=len(self.edges) + 1)
        return np.clip(counts / max(len(values), 1), 1e-6, None)

    def update(self, value: float) -> None:
        """Add one live observation."""
        self.window.append(float(value))

    def psi(self) -> float | None:
        """PSI of the current window vs the reference (None until enough samples)."""
        if len(self.window) < self.min_samples:
            return None
        live = self._fractions(np.array(self.window))
        return float(np.sum((live - self.ref_frac) * np.log(live / self.ref_frac)))

    def status(self) -> dict[str, Any]:
        """JSON-ready drift summary."""
        value = self.psi()
        level = ("insufficient data" if value is None else "stable" if value < 0.10
                 else "moderate shift" if value < 0.25 else "significant shift")
        return {"metric": "PSI of anomaly score vs legitimate validation traffic",
                "psi": None if value is None else round(value, 4), "status": level,
                "window_size": len(self.window), "thresholds": {"moderate": 0.10, "significant": 0.25}}
