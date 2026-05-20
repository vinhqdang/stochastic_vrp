"""Phase 1 cache for fast W-DRO evaluation under route perturbation.

Implements the three-table structure of Proposition M10:
    L_r[s, k]      = load at stage k under scenario s
    Omega_r[s, k]  = prefix peak: max_{k' <= k} L_r[s, k']
    Psi_r[s, k]    = suffix peak: max_{k' >= k} L_r[s, k']

Memory: O(N * (m+1)) per route. Construction: O(N * m) time.

Enables O(N log N) per-candidate insertion evaluation (Theorem M11),
independent of route length m.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.instance import Instance
from core.route import Route


@dataclass
class RouteCache:
    """Phase 1 cache: load + prefix-peak + suffix-peak tables for one route.

    Built once from (route, scenarios). To handle a route mutation (insert,
    remove, 2-opt swap), rebuild via .rebuild() — for Day 4 we accept the
    O(Nm) rebuild cost; Day 5 will add incremental updates.

    Attributes
    ----------
    route : Route
    scenarios : np.ndarray, shape (N, 2n)
        Reference, not copy.
    n_customers : int
        From the instance, used for mask matrix shape.
    L : np.ndarray, shape (N, m+1)
        Load table.
    Omega : np.ndarray, shape (N, m+1)
        Prefix-peak table.
    Psi : np.ndarray, shape (N, m+1)
        Suffix-peak table.
    """

    route: Route
    scenarios: np.ndarray
    n_customers: int
    L: np.ndarray = field(init=False)
    Omega: np.ndarray = field(init=False)
    Psi: np.ndarray = field(init=False)

    def __post_init__(self):
        self._build()

    def _build(self):
        """Build all three tables from route and scenarios."""
        N = self.scenarios.shape[0]
        m = len(self.route)

        if m == 0:
            # Empty route: f_r === 0 by convention
            self.L = np.zeros((N, 1), dtype=np.float64)
            self.Omega = np.zeros((N, 1), dtype=np.float64)
            self.Psi = np.zeros((N, 1), dtype=np.float64)
            return

        # ---- Load table via mask product ----
        # L[s, k] = beta_{r,k}^T xi^(s)
        # Use Route.loads_at_stages_batch which returns shape (N, m+1)
        self.L = self.route.loads_at_stages_batch(self.scenarios, self.n_customers)
        assert self.L.shape == (N, m + 1)

        # ---- Prefix-peak table: forward cumulative max ----
        # Omega[s, k] = max_{k' <= k} L[s, k']
        self.Omega = np.maximum.accumulate(self.L, axis=1)

        # ---- Suffix-peak table: backward cumulative max ----
        # Psi[s, k] = max_{k' >= k} L[s, k']
        self.Psi = np.maximum.accumulate(self.L[:, ::-1], axis=1)[:, ::-1]

    def rebuild(self):
        """Rebuild cache after route mutation (O(Nm))."""
        self._build()

    @property
    def peak_loads(self) -> np.ndarray:
        """f_r(xi^(s)) for s = 1..N. Shape (N,)."""
        if len(self.route) == 0:
            return np.zeros(self.scenarios.shape[0])
        return self.Omega[:, -1]  # = Psi[:, 0] = full peak

    @property
    def N(self) -> int:
        return self.scenarios.shape[0]

    @property
    def m(self) -> int:
        return len(self.route)

    def memory_bytes(self) -> int:
        """Total memory footprint of cache tables."""
        return self.L.nbytes + self.Omega.nbytes + self.Psi.nbytes