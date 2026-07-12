"""baselines.py — non-anytime-valid triggers the paper compares against.

CusumMonitor      one-sided CUSUM per channel on the standardized
                  residuals (k = 0.5 reference drift), alarm when any
                  channel's statistic crosses h. The classical foil: no
                  anytime type-I guarantee; h must be tuned per horizon.
PeriodicMonitor   fires every `period` events regardless of data (the
                  rolling-horizon industry default).
BonferroniFixed   fixed-sample z-tests per channel at pre-committed
                  checkpoints with Bonferroni correction — valid ONLY
                  at the committed peeks, illustrating what anytime
                  validity buys.

All expose .update(event) -> bool like MasterEProcess, so run_day works
unchanged. Accident counts and breakdown indicators are standardized to
z-scores under their null laws so every monitor sees the same stream.
"""

from __future__ import annotations

import numpy as np


def _event_z(event: dict) -> float:
    ch = event["channel"]
    if ch in ("travel", "demand", "dwell"):
        return float(event["z"])
    if ch == "accident":
        lam = max(event["lam0dt"], 1e-12)
        return float((event["n"] - lam) / np.sqrt(lam))
    p0 = min(max(event["p0"], 1e-12), 1 - 1e-12)
    return float((event["x"] - p0) / np.sqrt(p0 * (1 - p0)))


class CusumMonitor:
    def __init__(self, h: float = 8.0, k: float = 0.5):
        self.h, self.k = h, k
        self.reset()

    def reset(self):
        self.stat = {}

    def update(self, event: dict) -> bool:
        z = _event_z(event)
        ch = event["channel"]
        s = max(0.0, self.stat.get(ch, 0.0) + z - self.k)
        self.stat[ch] = s
        return s >= self.h


class PeriodicMonitor:
    def __init__(self, period: int = 20):
        self.period = period
        self.reset()

    def reset(self):
        self.t = 0

    def update(self, event: dict) -> bool:
        self.t += 1
        return self.t % self.period == 0


class BonferroniFixed:
    """z-tests on per-channel running means at fixed checkpoints."""

    def __init__(self, alpha: float = 0.05, checkpoints=(20, 40, 60, 80)):
        self.alpha = alpha
        self.checkpoints = set(checkpoints)
        self.n_tests = max(1, len(checkpoints)) * 5   # channels
        self.reset()

    def reset(self):
        self.t = 0
        self.sums = {}
        self.cnts = {}

    def update(self, event: dict) -> bool:
        from scipy import stats as sps
        z = _event_z(event)
        ch = event["channel"]
        self.sums[ch] = self.sums.get(ch, 0.0) + z
        self.cnts[ch] = self.cnts.get(ch, 0) + 1
        self.t += 1
        if self.t not in self.checkpoints:
            return False
        crit = sps.norm.ppf(1.0 - self.alpha / self.n_tests)
        for c, n in self.cnts.items():
            if n and self.sums[c] / np.sqrt(n) >= crit:
                return True
        return False
