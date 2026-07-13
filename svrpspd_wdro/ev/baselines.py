"""baselines.py — the comparison set TEMPO is benchmarked against.

Two tiers, and they should not be confused:

INVALID WITHOUT ORACLE CALIBRATION (classical, no anytime guarantee):
  CusumMonitor      one-sided CUSUM per channel on the standardized
                    residuals (k = 0.5 reference drift); h must be
                    tuned per horizon or its false-alarm rate exceeds
                    the nominal level (see results_ev_grid.csv).
  PageHinkleyMonitor the data-stream analogue of CUSUM (Page, 1954;
                    Gama et al. 2014 tradition); same caveat.
  BonferroniFixed   fixed-sample z-tests at pre-committed checkpoints
                    with Bonferroni correction — valid ONLY at the
                    committed peeks, illustrating what anytime
                    validity buys over periodic peeking.
  PeriodicMonitor   fires every `period` events regardless of data —
                    the rolling-horizon industry default; its trigger
                    mechanism is the same fixed-epoch re-solve used by
                    the chance-constrained SMPC strategy of He, Li, Li,
                    Huang, Huang & Duan (2026), "A Stochastic Model
                    Predictive Control Strategy for Vehicle Routing
                    with Correlated Stochastic Service Times,"
                    Mathematics 14(6):1032, DOI 10.3390/math14061032 —
                    that paper re-solves a chance-constrained MILP at
                    every planning epoch rather than on evidence; we
                    reproduce its TRIGGER cadence (period = one full
                    stop cycle by default) without its time-window MILP,
                    which TEMPO's harness does not model.
  RollingOptMonitor period = one stop cycle (5 events): the "Rolling-
                    opt" rolling-horizon heuristic of Gmira, Glize,
                    Hoenig & Rousseau (2021a), cited and benchmarked as
                    a baseline in Chen, Imdahl, Lai & Van Woensel
                    (2025), "The Dynamic Traveling Salesman Problem
                    with Time-Dependent and Stochastic travel times: A
                    deep reinforcement learning approach," Transport-
                    ation Research Part C 172:105022, DOI
                    10.1016/j.trc.2025.105022. That paper's OWN
                    contribution is a Dynamic Graph Temporal Attention
                    RL policy (DGTA-RL) that reacts to time-dependent
                    stochastic travel times end-to-end with no
                    explicit change-detection layer and no anytime
                    guarantee; retraining its attention architecture is
                    out of scope here, so we reproduce the rolling-
                    horizon heuristic it was benchmarked against
                    instead, and note the RL comparison as related
                    work rather than a reimplemented baseline.

ANYTIME-VALID, NO CALIBRATION NEEDED (the fair statistical comparison):
  EShiryaevRoberts, ECusum   the e-detector framework of Shin, Ramdas &
                    Rinaldo (2024), "E-detectors: A Nonparametric
                    Framework for Sequential Change Detection," The New
                    England Journal of Statistics in Data Science
                    2(2):229-260, DOI 10.51387/23-NEJSDS51. An
                    e-detector sums (SR) or maxes (CUSUM) e-processes
                    RESTARTED AT EVERY CANDIDATE CHANGEPOINT. We build
                    their baseline e_j-processes from EXACTLY the same
                    per-channel bets TEMPO uses (gaussian_step /
                    poisson_step / the rare-event Bernoulli bet) with
                    s = 1 (no cost-sensitivity tilt) and no adaptive
                    EWMA theta, isolating what TEMPO's coupling adds
                    over the general nonparametric machinery it is a
                    special case of. Per their Remark 2.13, e-SR always
                    detects at least as fast as e-CUSUM for the same
                    threshold, since M^SR_n = sum_j >= max_j = M^CU_n
                    pointwise; both are implemented for completeness.

                    IMPORTANT — their guarantee is NOT the one TEMPO's
                    Ville bound gives. An e-detector satisfies Def. 2.2,
                    E[M_tau] <= E[tau] (NOT <= 1 like a genuine
                    e-process), so Theorem 2.4 is an AVERAGE RUN LENGTH
                    bound, E_infty[N*] >= 1/alpha, over an INDEFINITELY
                    run stream — not a Ville-style P(ever alarm in n
                    steps) <= alpha bound over a fixed n-event window.
                    Our evaluation protocol resets every ~100-event day,
                    so the raw threshold 1/alpha does NOT by itself
                    control our per-day false-alarm rate (a day can be
                    short relative to the target ARL, so alarms that
                    would average out over an infinite run still land
                    inside single finite days). For the apples-to-apples
                    per-day comparison against TEMPO and the classical
                    foils, ev_grid_eval.py / ev_detect_eval.py oracle-
                    calibrate the M-statistic threshold exactly as for
                    CusumMonitor/PageHinkleyMonitor. test_ev.py verifies
                    the NATIVE ARL guarantee separately, on a single
                    continuous (non-reset) stream, which is the object
                    Theorem 2.4 actually bounds.

All expose .update(event) -> bool like MasterEProcess, so run_day works
unchanged. Accident counts and breakdown indicators are standardized to
z-scores under their null laws so every monitor sees the same stream.
"""

from __future__ import annotations

import numpy as np

from .eprocess import gaussian_step, poisson_step, ADAPT_GRID


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


class PageHinkleyMonitor:
    """Page-Hinkley drift detector on the pooled standardized residuals
    (data-stream tradition, Gama et al.): m_t = sum(z_i - delta),
    alarm when m_t - min_s m_s > h. One-sided (harmful drift raises z)."""

    def __init__(self, h: float = 15.0, delta: float = 0.25):
        self.h, self.delta = h, delta
        self.reset()

    def reset(self):
        self.m = 0.0
        self.m_min = 0.0

    def update(self, event: dict) -> bool:
        self.m += _event_z(event) - self.delta
        self.m_min = min(self.m_min, self.m)
        return (self.m - self.m_min) >= self.h


def calibrate_threshold(make_monitor, stat_fn, null_event_streams,
                        alpha: float):
    """Oracle calibration of a non-anytime detector: run the monitor
    over null-day event streams recording its running max statistic,
    return the (1-alpha) quantile of per-day maxima — the LOWEST
    threshold whose empirical null false-alarm rate is ~alpha. This is
    strictly generous to the baseline (it needs the null distribution
    of its own statistic, which no dispatcher has)."""
    import numpy as np
    maxima = []
    for events in null_event_streams:
        mon = make_monitor()
        peak = 0.0
        for ev in events:
            mon.update(ev)
            peak = max(peak, stat_fn(mon))
        maxima.append(peak)
    return float(np.quantile(maxima, 1.0 - alpha))


class RollingOptMonitor:
    """Rolling-opt heuristic (Gmira, Glize, Hoenig & Rousseau, 2021a),
    benchmarked in Chen, Imdahl, Lai & Van Woensel (2025), DTSP-TDS
    paper: replan at every decision epoch (one stop cycle). Distinct
    from PeriodicMonitor only in its citation and default cadence —
    kept as a separate class so the two rolling-horizon literatures
    (chance-constrained MPC vs. dynamic-routing RL benchmarks) are not
    conflated in results tables."""

    def __init__(self, events_per_stop: int = 5):
        self.period = events_per_stop
        self.reset()

    def reset(self):
        self.t = 0

    def update(self, event: dict) -> bool:
        self.t += 1
        return self.t % self.period == 0


# ═══════════════════════════════════════════════════════════════════════════
# E-detectors (Shin, Ramdas & Rinaldo, 2024) — anytime-valid, no calibration
# ═══════════════════════════════════════════════════════════════════════════

def _baseline_increment(event: dict, alpha: float, grid=ADAPT_GRID) -> float:
    """The one-step 'baseline increment' L_n of Definition 2.8: any
    process with sup_P E[L_n | F_{n-1}] <= 1 qualifies. We reuse
    TEMPO's own per-channel bets at s=1 (flat, no cost-relevance tilt)
    so the e-detector is built from identical statistical primitives —
    the only difference from TEMPO is the COMBINATION rule (restart-
    and-sum/max vs. a single running product) and the absence of
    adaptive/decision-relevant tilting."""
    ch = event["channel"]
    if ch in ("travel", "demand", "dwell"):
        return gaussian_step(event["z"], 1.0, grid)
    if ch == "accident":
        return poisson_step(event["n"], event["lam0dt"], 1.0)
    # breakdown: same rare-event stake as TEMPO's, calibrated to alpha
    p0 = min(max(event["p0"], 1e-12), 1 - 1e-12)
    w = min(0.5, 1.1 * (1.0 / alpha - 1.0) / (1.0 / p0 - 1.0))
    return 1.0 - w + w * (event["x"] / p0)


class EShiryaevRoberts:
    """e-SR e-detector (Shin-Ramdas-Rinaldo 2024, Def. 2.6): M_n^SR =
    sum_{j<=n} Lambda_n^(j), computed by the O(1) recursion
    M_n = L_n * (M_{n-1} + 1). Alarm at M_n >= 1/alpha; Theorem 2.4
    guarantees ARL >= 1/alpha with NO calibration, for ANY pre-change
    process (non-stationary nulls included) — the paper's Remark 2.13
    recommends exactly this threshold when the null may drift, which is
    our setting (diurnal travel, forecast rain)."""

    def __init__(self, alpha: float = 0.05, grid=ADAPT_GRID):
        self.alpha = alpha
        self.grid = grid
        self.reset()

    def reset(self):
        self.M = 0.0

    def update(self, event: dict) -> bool:
        L = _baseline_increment(event, self.alpha, self.grid)
        self.M = L * (self.M + 1.0)
        return self.M >= 1.0 / self.alpha


class ECusum:
    """e-CUSUM e-detector (Def. 2.6): M_n^CU = max_{j<=n} Lambda_n^(j),
    recursion M_n = L_n * max(M_{n-1}, 1). Alarm at M_n >= 1/alpha
    (c_alpha = 1/alpha is the simplest valid threshold per the paper;
    e-SR is provably at least as fast for the same threshold, since
    M^SR >= M^CU pointwise — kept here for completeness/symmetry with
    the classical CUSUM/e-SR pairing)."""

    def __init__(self, alpha: float = 0.05, grid=ADAPT_GRID):
        self.alpha = alpha
        self.grid = grid
        self.reset()

    def reset(self):
        self.M = 0.0

    def update(self, event: dict) -> bool:
        L = _baseline_increment(event, self.alpha, self.grid)
        self.M = L * max(self.M, 1.0)
        return self.M >= 1.0 / self.alpha
