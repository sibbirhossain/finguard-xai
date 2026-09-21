import numpy as np

from src.monitoring.drift import DriftMonitor


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(0)
    mon = DriftMonitor(rng.normal(0, 1, 5000))
    for v in rng.normal(0, 1, 2000):
        mon.update(v)
    assert mon.status()["status"] == "stable"
    for v in rng.normal(1.5, 1, 2000):
        mon.update(v)
    assert mon.psi() > 0.25
