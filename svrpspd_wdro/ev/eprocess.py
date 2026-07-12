"""eprocess.py — the master e-process with predictable sensitivity tilts.

Construction (PROJECT.md §3–4). At each event the firing channel
contributes one conditionally-mean-one factor

    e_step = sum_j pi_j * e(theta_j * s(ctx))            (mixture bet)

where e(theta) is the channel's likelihood-ratio bet, {theta_j} a fixed
grid, and s(ctx) in [0, 1] a PREDICTABLE sensitivity computed from the
event's ctx dict (plan state strictly before the realization). Each
component has conditional mean 1 under H0 for any predictable theta, so
the mixture does too, so the running product E_t is a nonnegative
martingale with E[E_t] = 1; Ville gives P(sup E_t >= 1/alpha) <= alpha.

Bets:
  gaussian   e = exp(theta*z - theta^2/2)                z ~ N(0,1) under H0
  poisson    e = exp(theta*n - lam0dt*(e^theta - 1))     n ~ Poi(lam0dt)
  bernoulli  e = 1 - w + w*x/p0 (rare-event stake w from alpha)

Sensitivities (all read only ctx, hence predictable):
  demand     boundary closeness 1 - clip(|B - W_prev| / B, 0, 1) —
             large when the running load is near the recourse boundary,
             i.e. when demand drift is decision-relevant (paper-1 logic;
             a Chat_k-based version can be plugged in via ctx later)
  travel     remaining-stop fraction (drift early in the day threatens
             more of the plan)
  dwell      remaining-stop fraction
  accident   remaining-stop fraction
  breakdown  1 (a breakdown is always decision-relevant)

Floors keep every channel awake: s -> S_MIN + (1 - S_MIN) * s.
"""

from __future__ import annotations

import numpy as np

THETA_GRID = (0.0, 0.25, 0.5, 1.0)      # 0-component floors the null bleed
POISSON_GRID = (0.0, 0.5, 1.0, 2.0)     # rare-event Poisson wants larger tilts
S_MIN = 0.25


# ── per-channel step bets (mixture over the theta grid) ──────────────────────

def gaussian_step(z: float, s: float, grid=THETA_GRID) -> float:
    th = np.asarray(grid) * s
    return float(np.mean(np.exp(th * z - 0.5 * th ** 2)))


def poisson_step(n: int, lam0dt: float, s: float, grid=POISSON_GRID) -> float:
    th = np.asarray(grid) * s
    return float(np.mean(np.exp(th * n - lam0dt * (np.exp(th) - 1.0))))


def bernoulli_step(x: int, p0: float, alpha: float = 0.05) -> float:
    """Rare-event bet e = 1 - w + w*x/p0 (conditional mean 1 for any
    predictable w in [0,1]). w is the smallest stake for which a SINGLE
    event crosses the Ville threshold 1/alpha, times a 10% margin, so a
    breakdown fires the master immediately while the per-leg bleed on
    quiet legs is only log(1-w) ~ -w."""
    p0 = min(max(p0, 1e-12), 1 - 1e-12)
    w = min(0.5, 1.1 * (1.0 / alpha - 1.0) / (1.0 / p0 - 1.0))
    return 1.0 - w + w * (x / p0)


# ── predictable sensitivities ────────────────────────────────────────────────

def _sensitivity(channel: str, ctx: dict) -> float:
    if channel == "demand":
        B = max(ctx["B"], 1e-9)
        s = 1.0 - min(abs(B - ctx["W_prev"]) / B, 1.0)
    elif channel in ("travel", "dwell", "accident"):
        s = ctx["rem_frac"]
    else:                                   # breakdown
        return 1.0
    return S_MIN + (1.0 - S_MIN) * float(s)


# ── the master process ───────────────────────────────────────────────────────

class MasterEProcess:
    """Running product of mixture bets across all channels, with
    per-channel log attribution and Ville-threshold triggering."""

    CHANNELS = ("travel", "demand", "dwell", "accident", "breakdown")

    def __init__(self, alpha: float = 0.05, use_sensitivity: bool = True,
                 grid=THETA_GRID):
        self.alpha = alpha
        self.use_sensitivity = use_sensitivity
        self.grid = grid
        self.reset()

    def reset(self):
        self.logE = 0.0
        self.log_by_channel = {c: 0.0 for c in self.CHANNELS}
        self.n_steps = 0

    @property
    def E(self) -> float:
        return float(np.exp(self.logE))

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    def update(self, event: dict) -> bool:
        """Consume one world event; return True if the alarm fires."""
        ch = event["channel"]
        s = _sensitivity(ch, event["ctx"]) if self.use_sensitivity else 1.0
        if ch in ("travel", "demand", "dwell"):
            e = gaussian_step(event["z"], s, self.grid)
        elif ch == "accident":
            e = poisson_step(event["n"], event["lam0dt"], s)
        else:
            e = bernoulli_step(event["x"], event["p0"], self.alpha)
        le = float(np.log(max(e, 1e-300)))
        self.logE += le
        self.log_by_channel[ch] += le
        self.n_steps += 1
        return self.logE >= np.log(self.threshold)

    def attribution(self) -> dict:
        """Per-channel share of the current log e-value (diagnosis:
        'why did it fire')."""
        return dict(self.log_by_channel)


def run_day(monitor, events) -> dict:
    """Feed a day's events to a monitor with .update(event) -> bool.
    Returns first-alarm info: fired, alarm_idx, alarm_clock, and the
    ground-truth drifted flag of the alarm event (for scoring delays).
    The monitor is NOT reset here."""
    for i, ev in enumerate(events):
        if monitor.update(ev):
            return dict(fired=True, alarm_idx=i, alarm_clock=ev["clock"],
                        alarm_on_drifted=bool(ev["ctx"].get("drifted", False)))
    return dict(fired=False, alarm_idx=None, alarm_clock=None,
                alarm_on_drifted=False)
