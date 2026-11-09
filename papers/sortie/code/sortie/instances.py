"""Module A: CVRPLIB parsing and instance construction."""

import math
import os
from dataclasses import dataclass

import numpy as np

DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "X-n641-k35.vrp")
)


def load_x641(path=DATA_PATH):
    """Parse NODE_COORD_SECTION and DEMAND_SECTION of a CVRPLIB file.

    Returns (coords (N,2) float64, demand (N,) int64) in file order; node 1
    of the file is index 0. Whitespace (tabs or spaces) is arbitrary.
    """
    section = None
    coords_rows = []
    demand_rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            head = parts[0]
            if head in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"):
                section = head
                if head == "EOF":
                    break
                continue
            if section == "NODE_COORD_SECTION":
                if len(parts) != 3:
                    raise ValueError(f"bad coordinate line: {raw!r}")
                coords_rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
            elif section == "DEMAND_SECTION":
                if len(parts) != 2:
                    raise ValueError(f"bad demand line: {raw!r}")
                demand_rows.append((int(parts[0]), int(parts[1])))
    for pos, (node_id, _, _) in enumerate(coords_rows):
        if node_id != pos + 1:
            raise ValueError("node ids are not consecutive from 1 in file order")
    for pos, (node_id, _) in enumerate(demand_rows):
        if node_id != pos + 1:
            raise ValueError("demand ids are not consecutive from 1 in file order")
    if len(coords_rows) != len(demand_rows):
        raise ValueError("coordinate and demand sections differ in length")
    coords = np.array([[x, y] for _, x, y in coords_rows], dtype=np.float64)
    demand = np.array([q for _, q in demand_rows], dtype=np.int64)
    return coords, demand


@dataclass
class Instance:
    name: str
    coords: np.ndarray
    demand: np.ndarray
    m: int
    Q: int
    C: np.ndarray


def make_instance(coords, demand, n, b, m=5, cap_div=4.0):
    """Block b of the file: depot plus n customers, Q = floor(sum q / cap_div)."""
    coords = np.asarray(coords, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.int64)
    block = n + 1
    n_blocks = len(coords) // block
    if not 0 <= b < n_blocks:
        raise ValueError(f"block {b} out of range; {n_blocks} blocks of size {block} available")
    idx = np.arange(b * block, b * block + block)
    xy = coords[idx].copy()
    q = demand[idx].copy()
    q[0] = 0
    Q = int(math.floor(q[1:].sum() / cap_div))
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    C = np.ceil(dist)
    np.fill_diagonal(C, 0.0)
    return Instance(name=f"X641-n{n}-b{b}", coords=xy, demand=q, m=int(m), Q=Q, C=C)
