"""env.py — DTSP-TDS instance generator and realized-time environment,
following Chen, Imdahl, Lai & Van Woensel (2025) Section 3 exactly:

  time-dependent expected travel time c_hat_ij(t): base euclidean
  distance x a diurnal multiplier (we reuse TEMPO's own diurnal curve,
  ev.world.diurnal_mult, so both baselines share one time-of-day model)

  stochastic realization (paper's Eq. 3): once decision step d+1 falls
  in a NEW time interval (relative to the interval the LAST refresh
  happened in), every edge's travel time for that interval is redrawn,
  c_ij ~ Gamma(c_hat_ij(t)/beta, beta) — mean c_hat, variance beta*c_hat;
  edges are held fixed while the clock stays inside the same interval
  (paper's Eq. 3 case split). This is naturally a WHOLE-MATRIX refresh
  event (the interval is a property of the clock, not of an edge), so
  the realized matrix (B, N+1, N+1) is resampled in full whenever the
  discretized time index changes, not cached per traversed edge.

The reward (paper's Eq. 5) is the NEGATIVE realized tour duration
(single vehicle, visits all N customers once, returns to the depot).
"""

from __future__ import annotations

import torch

import sys
from pathlib import Path
_WDRO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_WDRO))
from ev.world import diurnal_mult, DayParams   # noqa: E402


def random_instance(batch, n_customers, device="cpu", seed=None):
    """coords in [0,1]^2, node 0 = depot. Returns (B, N+1, 2)."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    coords = torch.rand(batch, n_customers + 1, 2, generator=g).to(device)
    return coords


def travel_hours(dist, speed=0.35):
    """Base euclidean distance -> base travel hours at free-flow speed,
    scaled so a full tour of ~n_customers legs plausibly spans the
    working day (kept simple/synthetic, not physically calibrated)."""
    return dist / speed


class DTSPEnv:
    """Batched DTSP-TDS environment. All tensors (B, ...); vehicle
    starts at node 0 (depot), must visit all customers once, returns
    to depot. Time-dependent + stochastic travel times per module doc.

    Travel times live in one shared (B, N+1, N+1) `self.realized`
    matrix that is entirely resampled whenever the discretized time
    interval advances (`_maybe_refresh`), matching the paper's Eq. 3
    "same interval -> reuse, new interval -> redraw" semantics at the
    matrix level rather than per traversed edge.
    """

    def __init__(self, coords, n_time=8, day_hours=8.0, beta=0.15,
                t0_hours=8.0, speed=0.35, device="cpu"):
        self.coords = coords
        self.B, self.Np1, _ = coords.shape
        self.N = self.Np1 - 1
        self.n_time = n_time
        self.day_hours = day_hours
        self.beta = beta
        self.t0 = t0_hours
        self.speed = speed
        self.device = device
        d = coords.unsqueeze(2) - coords.unsqueeze(1)
        self.dist = torch.linalg.norm(d, dim=-1)          # (B, N+1, N+1)
        self.base_hours = travel_hours(self.dist, speed)
        self.diurnal = DayParams(tau=__import__("numpy").zeros(1),
                                 mu_g=__import__("numpy").zeros(1),
                                 sig_g=__import__("numpy").ones(1),
                                 B=1.0).diurnal
        self.reset()

    def _t_idx(self, clock_hours):
        frac = (clock_hours - self.t0) / self.day_hours
        idx = (frac.clamp(0, 0.999) * self.n_time).long()
        return idx

    def _mult(self, clock_hours):
        import numpy as np
        t = clock_hours.detach().cpu().numpy()
        mult = torch.tensor([diurnal_mult(float(ti), self.diurnal)
                             for ti in t], device=self.device,
                            dtype=torch.float32)
        return mult

    def _refresh_matrix(self):
        """Resample the FULL (B, N+1, N+1) realized-time matrix at the
        current clock's diurnal multiplier — one refresh event per
        time-interval change, shared by every edge (paper's Eq. 3)."""
        mult = self._mult(self.clock).view(self.B, 1, 1)
        chat = self.base_hours * mult
        shape = (chat / self.beta).clamp(min=1e-3)
        gamma = torch.distributions.Gamma(shape, 1.0 / self.beta)
        self.realized = gamma.sample()          # (B, N+1, N+1)
        self.matrix_tidx = self._t_idx(self.clock)

    def _maybe_refresh(self):
        idx = self._t_idx(self.clock)
        if bool((idx != self.matrix_tidx).any()):
            self._refresh_matrix()

    def reset(self):
        self.cur = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.clock = torch.full((self.B,), self.t0, device=self.device)
        self.visited = torch.zeros(self.B, self.Np1, dtype=torch.bool,
                                   device=self.device)
        self.visited[:, 0] = True
        self.arrival = torch.zeros(self.B, self.Np1, device=self.device)
        self.n_visited = torch.ones(self.B, dtype=torch.long,
                                    device=self.device)
        self.matrix_tidx = torch.full((self.B,), -1, dtype=torch.long,
                                      device=self.device)
        self._refresh_matrix()
        return self

    def expected_travel(self, clock_hours):
        """c_hat_ij(t) at the given clock time, for the CURRENT node
        row: (B, N+1)."""
        mult = self._mult(clock_hours)
        row = self.base_hours[torch.arange(self.B), self.cur]  # (B,N+1)
        return row * mult.unsqueeze(-1)

    def step(self, nxt):
        """Move to node `nxt` (B,). Realizes travel time per Eq. 3 and
        returns (realized_hours, done)."""
        self._maybe_refresh()
        realized = self.realized[torch.arange(self.B), self.cur, nxt]
        self.clock = self.clock + realized
        self.arrival[torch.arange(self.B), nxt] = self.clock
        self.visited[torch.arange(self.B), nxt] = True
        self.cur = nxt
        self.n_visited += 1
        done = bool((self.n_visited >= self.Np1).all())
        return realized, done

    def close_tour(self):
        """Final leg back to the depot (does not touch visited/count
        state — call once, after all customers are visited)."""
        self._maybe_refresh()
        realized = self.realized[torch.arange(self.B), self.cur, 0]
        self.clock = self.clock + realized
        return realized
