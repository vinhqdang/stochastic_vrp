"""Route data structure for SVRPSPD.

A Route is an ordered sequence of customer indices. The depot is implicit
(every route starts and ends at the depot).

Key construct: the Mask Matrix B_r in {0,1}^{2n x (m+1)} encoding the truck's
cargo state at each stage k in {0, 1, ..., m}. Specifically, for stage k:
    - position 2j is set if customer j has its DELIVERY still on the truck
    - position 2j+1 is set if customer j has had its PICKUP loaded
This matches the demand vector convention xi[2j] = d_j, xi[2j+1] = p_j.

The peak load f_r(xi) = max_k beta_{r,k}^T xi is the operator computed via:
    f_r(xi) = max_k L[k]
where L[k] = beta_{r,k}^T xi is the load at stage k. We provide three ways to
compute L:
    1) build_mask_matrix(n) @ xi -- O(n*m) memory, slow
    2) load_at_stages(xi, n)     -- O(m) via recurrence, single scenario
    3) loads_at_stages_batch(X, n) -- O(N*m) via recurrence, N scenarios
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class Route:
    """Ordered sequence of customer indices.

    Customer indices are 0-based; the depot is implicit and not stored.

    Parameters
    ----------
    customers : Iterable[int], optional
        Initial customer sequence. Defaults to empty route.

    Examples
    --------
    >>> r = Route([1, 3, 4])
    >>> len(r)
    3
    >>> r.insert(2, pos=2)        # insert customer 2 at position 2
    >>> r.customers
    [1, 2, 3, 4]
    >>> r.remove_at(pos=3)         # remove the customer at position 3
    3
    >>> r.customers
    [1, 2, 4]
    """

    __slots__ = ("customers",)

    def __init__(self, customers: Iterable[int] | None = None):
        if customers is None:
            self.customers: list[int] = []
        else:
            self.customers = [int(c) for c in customers]
        # Optional invariant check (cheap): distinct customers
        if len(set(self.customers)) != len(self.customers):
            raise ValueError(
                f"Route has duplicate customers: {self.customers}"
            )

    # ---------- container protocol ----------

    def __len__(self) -> int:
        return len(self.customers)

    def __iter__(self):
        return iter(self.customers)

    def __getitem__(self, idx: int) -> int:
        return self.customers[idx]

    def __repr__(self) -> str:
        return f"Route({self.customers})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Route):
            return NotImplemented
        return self.customers == other.customers

    def copy(self) -> Route:
        """Return a deep copy (Route with same customer sequence)."""
        return Route(self.customers)

    # ---------- structural operations ----------

    def insert(self, j: int, pos: int) -> None:
        """Insert customer j at position pos (1-indexed).

        After insertion, the new route has j at position pos. pos in {1, ..., m+1}
        where m = len(self) BEFORE insertion. Use pos=1 to insert at start,
        pos=m+1 to insert at end.

        Parameters
        ----------
        j : int
            Customer index to insert.
        pos : int
            1-indexed position in the NEW route (after insertion). Must satisfy
            1 <= pos <= len(self) + 1.
        """
        if not (1 <= pos <= len(self) + 1):
            raise IndexError(
                f"Insertion position {pos} out of range "
                f"[1, {len(self) + 1}]"
            )
        if j in self.customers:
            raise ValueError(f"Customer {j} already in route")
        self.customers.insert(pos - 1, int(j))

    def remove_at(self, pos: int) -> int:
        """Remove customer at position pos (1-indexed). Returns removed id."""
        if not (1 <= pos <= len(self)):
            raise IndexError(
                f"Removal position {pos} out of range [1, {len(self)}]"
            )
        return self.customers.pop(pos - 1)

    # ---------- mask matrix ----------

    def build_mask_matrix(self, n: int) -> np.ndarray:
        """Construct B_r in {0,1}^{2n x (m+1)}.

        Stage 0: all customers' deliveries loaded.
            B[2c, 0] = 1 for each c in self.customers.
        Stage k (k >= 1): drop d_{c_k}, add p_{c_k}.
            B[:, k] = B[:, k-1] with B[2*c_k, k] = 0 and B[2*c_k+1, k] = 1.

        Parameters
        ----------
        n : int
            Total number of customers (for the dimension 2n).

        Returns
        -------
        B : np.ndarray of dtype int8, shape (2*n, m+1)
        """
        m = len(self.customers)
        B = np.zeros((2 * n, m + 1), dtype=np.int8)

        # Stage 0: all deliveries loaded
        for c in self.customers:
            B[2 * c, 0] = 1

        # Stages 1..m: incremental update
        for k in range(1, m + 1):
            B[:, k] = B[:, k - 1]
            c_k = self.customers[k - 1]
            B[2 * c_k, k] = 0      # drop delivery for c_k
            B[2 * c_k + 1, k] = 1  # add pickup for c_k

        return B

    # ---------- load computations ----------

    def load_at_stages(self, xi: np.ndarray, n: int) -> np.ndarray:
        """Compute L[k] = beta_{r,k}^T xi for k = 0, ..., m.

        Uses the load recurrence:
            L[0] = sum_{c in route} xi[2c]    (all deliveries on truck)
            L[k] = L[k-1] - xi[2*c_k] + xi[2*c_k+1]

        This is O(m), avoiding O(n*m) matrix multiplication.

        Parameters
        ----------
        xi : np.ndarray, shape (2n,)
            Demand realization vector.
        n : int
            Total number of customers.

        Returns
        -------
        L : np.ndarray, shape (m+1,)
        """
        if xi.shape != (2 * n,):
            raise ValueError(
                f"xi has shape {xi.shape}, expected ({2 * n},)"
            )
        m = len(self.customers)
        L = np.empty(m + 1, dtype=np.float64)

        # Stage 0: sum of deliveries on truck
        if m == 0:
            L[0] = 0.0
            return L

        delivery_idx = [2 * c for c in self.customers]
        L[0] = xi[delivery_idx].sum()

        for k in range(1, m + 1):
            c_k = self.customers[k - 1]
            L[k] = L[k - 1] - xi[2 * c_k] + xi[2 * c_k + 1]

        return L

    def loads_at_stages_batch(self, X: np.ndarray, n: int) -> np.ndarray:
        """Batched load computation over N scenarios.

        Vectorized form of load_at_stages over a scenario matrix X of shape
        (N, 2n). Uses the same recurrence, fully vectorized.

        Parameters
        ----------
        X : np.ndarray, shape (N, 2n)
            Each row is one scenario xi^(s).
        n : int
            Total number of customers.

        Returns
        -------
        L : np.ndarray, shape (N, m+1)
            L[s, k] = beta_{r,k}^T X[s].
        """
        if X.ndim != 2 or X.shape[1] != 2 * n:
            raise ValueError(
                f"X has shape {X.shape}, expected (N, {2 * n})"
            )
        N = X.shape[0]
        m = len(self.customers)
        L = np.empty((N, m + 1), dtype=np.float64)

        if m == 0:
            L[:, 0] = 0.0
            return L

        delivery_idx = [2 * c for c in self.customers]
        L[:, 0] = X[:, delivery_idx].sum(axis=1)

        for k in range(1, m + 1):
            c_k = self.customers[k - 1]
            L[:, k] = L[:, k - 1] - X[:, 2 * c_k] + X[:, 2 * c_k + 1]

        return L

    def peak_load(self, xi: np.ndarray, n: int) -> float:
        """Peak load f_r(xi) = max_k L[k] for a single scenario."""
        if len(self.customers) == 0:
            return 0.0
        return float(self.load_at_stages(xi, n).max())

    def peak_loads_batch(self, X: np.ndarray, n: int) -> np.ndarray:
        """Peak load for each of N scenarios. Returns shape (N,)."""
        if len(self.customers) == 0:
            return np.zeros(X.shape[0], dtype=np.float64)
        return self.loads_at_stages_batch(X, n).max(axis=1)

    # ---------- routing cost (deterministic, distance-based) ----------

    def travel_cost(self, distances: np.ndarray) -> float:
        """Total travel distance for this route (depot -> c1 -> ... -> cm -> depot).

        distances : (n+1) x (n+1) matrix with index 0 = depot, index j+1 = customer j.

        Note: customer j in the Route corresponds to index j+1 in the
        distance matrix (which includes depot at index 0).
        """
        if len(self.customers) == 0:
            return 0.0
        # Build path: depot -> c_1 -> ... -> c_m -> depot
        # In distance matrix indexing: depot is 0, customer j is j+1.
        path = [0] + [c + 1 for c in self.customers] + [0]
        cost = 0.0
        for i in range(len(path) - 1):
            cost += distances[path[i], path[i + 1]]
        return float(cost)