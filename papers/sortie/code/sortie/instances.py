"""Module A: CVRPLIB parsing and instance construction."""

import math
import os
import re
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

_DEFAULT_DATA = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "X-n641-k35.vrp")
)
DATA_PATH = os.path.normpath(os.environ.get("SORTIE_DATA") or _DEFAULT_DATA)
CONG_SUFFIX = "-cong"
_BASE_RE = re.compile(r"^X641-n(\d+)-b(\d+)$")


_SECTIONS = ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF")


def load_vrp(path=DATA_PATH):
    """TSPLIB CVRP reader for EUC_2D files.

    Returns (name, coords (N,2) float64, demand (N,) int64, capacity int). The
    depot is the node listed in DEPOT_SECTION (node 1 when the section is
    absent) and is returned first; the other nodes keep their file order.
    Line endings (LF or CRLF), tabs and repeated or trailing blanks are arbitrary.
    """
    section = None
    seen = set()
    header = {}
    coords_rows = []
    demand_rows = []
    depot_ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            head = parts[0]
            if head in _SECTIONS:
                section = head
                seen.add(head)
                if head == "EOF":
                    break
                continue
            if section is None:
                if ":" in raw:
                    key, _, val = raw.partition(":")
                    header[key.strip().upper()] = val.strip()
                continue
            if section == "NODE_COORD_SECTION":
                if len(parts) != 3:
                    raise ValueError(f"bad coordinate line: {raw!r}")
                coords_rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
            elif section == "DEMAND_SECTION":
                if len(parts) != 2:
                    raise ValueError(f"bad demand line: {raw!r}")
                demand_rows.append((int(parts[0]), int(parts[1])))
            elif section == "DEPOT_SECTION":
                if len(parts) != 1:
                    raise ValueError(f"bad depot line: {raw!r}")
                if int(parts[0]) != -1:
                    depot_ids.append(int(parts[0]))
    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError(f"only EUC_2D files are supported, got {header.get('EDGE_WEIGHT_TYPE')!r}")
    if "CAPACITY" not in header:
        raise ValueError("missing CAPACITY header")
    capacity = int(header["CAPACITY"])
    for pos, (node_id, _, _) in enumerate(coords_rows):
        if node_id != pos + 1:
            raise ValueError("node ids are not consecutive from 1 in file order")
    for pos, (node_id, _) in enumerate(demand_rows):
        if node_id != pos + 1:
            raise ValueError("demand ids are not consecutive from 1 in file order")
    if len(coords_rows) != len(demand_rows):
        raise ValueError("coordinate and demand sections differ in length")
    n_nodes = len(coords_rows)
    if n_nodes == 0:
        raise ValueError("no nodes in the file")
    if "DIMENSION" in header and int(header["DIMENSION"]) != n_nodes:
        raise ValueError("DIMENSION disagrees with the number of nodes")
    if "DEPOT_SECTION" in seen:
        if len(depot_ids) != 1:
            raise ValueError("DEPOT_SECTION must list exactly one depot")
        depot = depot_ids[0]
    else:
        depot = 1
    if not 1 <= depot <= n_nodes:
        raise ValueError(f"depot {depot} is not a node of the file")
    coords = np.array([[x, y] for _, x, y in coords_rows], dtype=np.float64)
    demand = np.array([q for _, q in demand_rows], dtype=np.int64)
    d = depot - 1
    order = [d] + [i for i in range(n_nodes) if i != d]
    return header.get("NAME", ""), coords[order], demand[order], capacity


def load_x641(path=DATA_PATH):
    """Parse NODE_COORD_SECTION and DEMAND_SECTION of a CVRPLIB file.

    Returns (coords (N,2) float64, demand (N,) int64) in file order; node 1
    of the file is index 0 (the depot of the X set, see load_vrp).
    Whitespace (tabs or spaces) is arbitrary.
    """
    _, coords, demand, _ = load_vrp(path)
    return coords, demand


@dataclass
class Instance:
    name: str
    coords: np.ndarray
    demand: np.ndarray
    m: int
    Q: int
    C: np.ndarray
    H: Optional[np.ndarray] = None


def make_instance(coords, demand, n, b, m=5, cap_div=4.0, prefix="X641"):
    """Block b of the file: depot plus n customers, Q = floor(sum q / cap_div).

    The instance name is f"{prefix}-n{n}-b{b}".
    """
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
    return Instance(name=f"{prefix}-n{n}-b{b}", coords=xy, demand=q, m=int(m), Q=Q, C=C)


def make_head_instance(coords, demand, prefix, n, m=5, cap_div=4.0):
    """The depot of the file plus its first n customers in file order.

    `coords` and `demand` must be depot-first (as returned by load_vrp), so the
    depot is the file's own depot. Q = floor(sum of the n demands / cap_div),
    C = ceiled Euclidean distances, name f"{prefix}-h{n}".
    """
    coords = np.asarray(coords, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.int64)
    if not 1 <= int(n) <= len(coords) - 1:
        raise ValueError(f"n must be in [1, {len(coords) - 1}], got {n}")
    n = int(n)
    xy = coords[: n + 1].copy()
    q = demand[: n + 1].copy()
    q[0] = 0
    Q = int(math.floor(q[1:].sum() / cap_div))
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    C = np.ceil(dist)
    np.fill_diagonal(C, 0.0)
    return Instance(name=f"{prefix}-h{n}", coords=xy, demand=q, m=int(m), Q=Q, C=C)


def base_name(name):
    """Instance name without the congestion suffix (the seed plan is shared with the base)."""
    return name[:-len(CONG_SUFFIX)] if name.endswith(CONG_SUFFIX) else name


def with_congestion(inst, v_lo_mult=1.0 / 3.0, v_hi_mult=2.0, radius_frac=0.3, seed=0):
    """Copy of inst with a heterogeneous arc-noise multiplier H (arc cv = v * H_ij).

    The centre is the coordinate of a customer drawn from default_rng([seed, n, b]);
    r = radius_frac * (largest pairwise coordinate distance); H_ij = lo + (hi - lo) *
    exp(-d^2 / (2 r^2)) with d the distance from the midpoint of arc (i, j) to the
    centre. H is symmetric, lies in [lo, hi] and equals 1 on the diagonal.
    """
    base = base_name(inst.name)
    match = _BASE_RE.match(base)
    if match is None:
        raise ValueError(f"cannot parse n, b from instance name {inst.name!r}")
    n, b = int(match.group(1)), int(match.group(2))
    xy = np.asarray(inst.coords, dtype=np.float64)
    N = xy.shape[0]
    if N - 1 != n:
        raise ValueError("instance name and node count disagree")
    rng = np.random.default_rng([int(seed), n, b])
    centre = xy[int(rng.integers(1, N))]
    diff = xy[:, None, :] - xy[None, :, :]
    span = float(np.sqrt((diff ** 2).sum(axis=2)).max())
    r = float(radius_frac) * span
    mid = 0.5 * (xy[:, None, :] + xy[None, :, :])
    d2 = ((mid - centre) ** 2).sum(axis=2)
    lo = float(v_lo_mult)
    hi = float(v_hi_mult)
    H = lo + (hi - lo) * np.exp(-d2 / (2.0 * r * r))
    np.fill_diagonal(H, 1.0)
    return replace(inst, name=base + CONG_SUFFIX, H=H)


def effective_H(inst):
    """The multiplier matrix, or None when it is absent or identically one.

    Returning None for an all-ones H keeps every downstream computation on the
    homogeneous code path, so H = 1 reproduces the homogeneous results exactly.
    """
    H = getattr(inst, "H", None)
    if H is None or np.all(H == 1.0):
        return None
    return np.ascontiguousarray(H, dtype=np.float64)


def noise_matrix(inst):
    """D = C o H (elementwise), the arc standard-deviation weights; C itself when H is None."""
    H = effective_H(inst)
    if H is None:
        return inst.C
    return np.ascontiguousarray(inst.C * H, dtype=np.float64)
