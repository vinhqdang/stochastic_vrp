"""SVRPSPD Instance: loader and data container.

Conventions:
    - Customer indices: 0 to n-1 (n customers, excluding depot).
    - Depot has implicit index "depot" (not in customer set).
    - Demand vector convention: xi in R^{2n}, xi[2j] = d_j, xi[2j+1] = p_j.
    - Coords array: shape (n+1, 2), coords[0] = depot, coords[j+1] = customer j.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class Instance:
    """Container for a single SVRPSPD instance.

    Attributes
    ----------
    name : str
        Instance identifier.
    n_customers : int
        Number of customers (depot excluded).
    capacity : float
        Single-vehicle capacity Q.
    coords : np.ndarray, shape (n_customers + 1, 2)
        Coordinates. Row 0 = depot. Rows 1..n_customers = customers 0..n_customers-1.
    nominal_d : np.ndarray, shape (n_customers,)
        Nominal delivery demands.
    nominal_p : np.ndarray, shape (n_customers,)
        Nominal pickup demands.
    """

    name: str
    n_customers: int
    capacity: float
    coords: np.ndarray
    nominal_d: np.ndarray
    nominal_p: np.ndarray

    # ---------- convenience properties ----------

    @property
    def n(self) -> int:
        """Alias for n_customers."""
        return self.n_customers

    @property
    def Q(self) -> float:
        """Alias for capacity."""
        return self.capacity

    # ---------- derived quantities ----------

    def nominal_xi(self) -> np.ndarray:
        """Return nominal demand vector xi in R^{2n}.

        xi[2j]   = nominal_d[j]
        xi[2j+1] = nominal_p[j]
        """
        xi = np.empty(2 * self.n, dtype=np.float64)
        xi[0::2] = self.nominal_d
        xi[1::2] = self.nominal_p
        return xi

    def distances(self) -> np.ndarray:
        """Pairwise Euclidean distance matrix. Shape (n+1, n+1).

        D[i, j] = ||coords[i] - coords[j]||_2.
        """
        return cdist(self.coords, self.coords)

    # ---------- I/O ----------

    @classmethod
    def from_tsplib_mvrpb(cls, path) -> "Instance":
        """Parse TSPLIB MVRPB / SVRPSPD format with PICKUP_AND_DELIVERY_SECTION
        in 7-column layout: node_id ... ... ... ... delivery pickup.
        """
        from pathlib import Path
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        dimension = 0
        capacity = 0.0
        vehicles = None

        for line in lines:
            if ':' in line:
                key, val = (x.strip() for x in line.split(':', 1))
                if   key == 'DIMENSION': dimension = int(val)
                elif key == 'CAPACITY':  capacity = float(val)
                elif key == 'VEHICLES':  vehicles = int(val)

        n_customers = dimension - 1
        d_arr = np.zeros(n_customers)
        p_arr = np.zeros(n_customers)

        mode = None
        edge_vals = []
        for line in lines:
            if line.startswith('EDGE_WEIGHT_SECTION'):
                mode = 'EDGE'; continue
            elif line.startswith('PICKUP_AND_DELIVERY_SECTION'):
                mode = 'PD'; continue
            elif 'SECTION' in line or line == 'EOF':
                mode = None; continue

            if mode == 'EDGE':
                edge_vals.extend(float(x) for x in line.split())
            elif mode == 'PD':
                parts = line.split()
                if len(parts) >= 7:
                    node = int(parts[0])
                    if node > 1:  # skip depot (node 1)
                        d_arr[node - 2] = float(parts[5])
                        p_arr[node - 2] = float(parts[6])

        D = np.array(edge_vals).reshape(dimension, dimension)

        inst = cls(
            name=path.stem,
            n_customers=n_customers,
            capacity=capacity,
            coords=np.zeros((dimension, 2)),  # placeholder; distances overridden
            nominal_d=d_arr,
            nominal_p=p_arr,
        )
        inst.D = D
        inst.distances = lambda: inst.D
        inst.n_vehicles = vehicles
        inst.nominal_xi = lambda: np.column_stack((inst.nominal_d, inst.nominal_p)).flatten()
        return inst

    # ---------- pretty print ----------
    @classmethod
    def from_file(cls, path) -> "Instance":
        """Hàm đọc file toy.txt phục vụ riêng cho Unit Test."""
        import numpy as np
        from scipy.spatial import distance
        from pathlib import Path
        
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        n_cust = int(lines[0])
        cap = float(lines[1])
        data = np.loadtxt(lines[2:])
        coords = data[:, :2]
        d_arr = data[1:, 2]
        p_arr = data[1:, 3]
        
        inst = cls(name=path.stem, n_customers=n_cust, capacity=cap, 
                   coords=coords, nominal_d=d_arr, nominal_p=p_arr)
                   
        # Bơm thêm mấy hàm động để Test không bị crash
        inst.D = distance.cdist(coords, coords)
        inst.distances = lambda: inst.D
        inst.nominal_xi = lambda: np.column_stack((inst.nominal_d, inst.nominal_p)).flatten()
        return inst
    def __repr__(self) -> str:
        return (
            f"Instance(name={self.name!r}, n={self.n}, Q={self.Q}, "
            f"d_range=[{self.nominal_d.min():.1f}, {self.nominal_d.max():.1f}], "
            f"p_range=[{self.nominal_p.min():.1f}, {self.nominal_p.max():.1f}])"
        )

    def summary(self) -> str:
        """Multiline diagnostic summary."""
        total_d = self.nominal_d.sum()
        total_p = self.nominal_p.sum()
        n_vehicles_lb_d = int(np.ceil(total_d / self.Q))
        n_vehicles_lb_p = int(np.ceil(total_p / self.Q))
        return (
            f"Instance: {self.name}\n"
            f"  Customers: {self.n}\n"
            f"  Capacity:  {self.Q}\n"
            f"  Sum d:     {total_d:.1f}   (>= {n_vehicles_lb_d} vehicles)\n"
            f"  Sum p:     {total_p:.1f}   (>= {n_vehicles_lb_p} vehicles)\n"
            f"  d range:   [{self.nominal_d.min():.1f}, "
            f"{self.nominal_d.max():.1f}]\n"
            f"  p range:   [{self.nominal_p.min():.1f}, "
            f"{self.nominal_p.max():.1f}]"
        )