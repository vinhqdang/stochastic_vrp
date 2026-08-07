"""Calibration corpus (PROJECT.md §5.2): estimate each probe tier's
alarm rates and costs on HELD-OUT problem families (CALIB_SPECS),
using known-faithful models and screened mutants, then produce
conservative betting parameters:

  p0_bar  = Clopper–Pearson 97.5% UPPER bound on the null alarm rate
            (validity needs only p0_true <= p0_bar — an upper
            confidence bound, not a point estimate)
  p1_bet  = shrunk detection-rate estimate (affects power only)
  delta   = Tier-B mutant alarm rate (routing only; validity of Tier B
            is by proof, and calibration ASSERTS zero faithful alarms)

The evaluation families (EVAL_SPECS) are disjoint, so the experiment
also tests whether conservative calibration transfers across families
— the honest version of the paper's held-out-corpus claim.
"""

from __future__ import annotations

import math
import statistics

from .certify import TIERS
from .mutations import make_faithful, make_mutant
from .probes import tier_a, tier_b, tier_c
from .problems import CALIB_SPECS


def _binom_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * p**i * (1 - p) ** (n - i)
    return min(total, 1.0)


def cp_upper(k, n, conf=0.975):
    """Clopper–Pearson upper confidence bound on a binomial rate."""
    if k >= n:
        return 1.0
    lo, hi = k / n if n else 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def calibrate(rng, n_per_family=120, n_mutants=8, pool_size=6):
    """Run the calibration corpus; returns the bets dict for Bets()."""
    counts = {t: {"h0_alarm": 0, "h0_n": 0, "h1_alarm": 0, "h1_n": 0,
                  "costs": []} for t in TIERS}

    for spec in CALIB_SPECS:
        faithful = make_faithful(spec)
        mutants = []
        while len(mutants) < n_mutants:
            m = make_mutant(spec, rng)
            if m is not None:
                mutants.append(m)

        for i in range(n_per_family):
            mut = mutants[i % len(mutants)]
            for tier, fn in (("A", tier_a), ("B", tier_b)):
                for cand, h in ((faithful, "h0"), (mut, "h1")):
                    alarm, cost = fn(cand, rng)
                    counts[tier][f"{h}_alarm"] += alarm
                    counts[tier][f"{h}_n"] += 1
                    counts[tier]["costs"].append(cost)
            # Tier C: representative mixed pool around the probed candidate
            others = [make_faithful(spec) if rng.random() < 0.5
                      else mutants[rng.randrange(len(mutants))]
                      for _ in range(pool_size - 1)]
            for cand, h in ((faithful, "h0"), (mut, "h1")):
                alarm, cost = tier_c(cand, [cand] + others, rng)
                counts["C"][f"{h}_alarm"] += alarm
                counts["C"][f"{h}_n"] += 1
                counts["C"]["costs"].append(cost)

    assert counts["B"]["h0_alarm"] == 0, (
        "Tier B alarmed on a faithful model — an MR is NOT valid; "
        "fix problems.py before trusting anything downstream.")

    calib = {}
    for t in TIERS:
        c = counts[t]
        p0_bar = cp_upper(c["h0_alarm"], c["h0_n"])
        p1_hat = c["h1_alarm"] / max(c["h1_n"], 1)
        p1_bet = max(0.15, 0.8 * p1_hat)
        p1_bet = max(p1_bet, min(3 * p0_bar, 0.5))
        calib[t] = {
            "h0_alarms": c["h0_alarm"], "h0_n": c["h0_n"],
            "p0_hat": c["h0_alarm"] / max(c["h0_n"], 1),
            "p0_bar": round(p0_bar, 6),
            "h1_alarms": c["h1_alarm"], "h1_n": c["h1_n"],
            "p1_hat": round(p1_hat, 6), "p1_bet": round(p1_bet, 6),
            "delta": round(p1_hat, 6),
            "mean_cost": round(statistics.fmean(c["costs"]), 6),
        }
    return calib
