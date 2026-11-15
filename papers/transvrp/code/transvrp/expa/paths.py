"""Route-level scenario pools for EXPERIMENT A (module M1).

Purposes and seeds follow EXPERIMENT_A.md section 5. ``base`` is
``data.stable_seed(inst)``.

* train, replicate r, chunk c: ``gen_scenarios`` with seed ``[base, 1, r, c]``,
  100,000 paths per chunk. N <= 10^5 uses the first N rows of chunk 0; larger N
  uses chunks 0.. in order (N <= 10^6).
* dev: seed ``[base, 3]``, 5,000 paths. ``rep`` is ignored.
* test: seed ``base + 99991``, 20,000 paths. ``rep`` is ignored.
* test_std: seed ``base + 99991``, 2,000 paths (parity with ``route_problem``).

Only route columns are returned. Chunks are generated one at a time; the
generator is cached with ``lru_cache(maxsize=1)``, so one chunk's instance
arrays stay alive at most until the next chunk is requested.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from ..baton.costs import LastMileCosts, route_cost_schedules
from ..data import DATA_DIR, gen_scenarios, load_plans, parse_dethloff, stable_seed

CHUNK = 100_000
N_CHUNKS = 10
PURPOSES = ("train", "dev", "test", "test_std")
DEV_N = 5_000
TEST_N = 20_000
TEST_STD_N = 2_000
TRAIN_REPS = (0, 1, 2)


@dataclass(frozen=True)
class RouteSpec:
    inst: str
    route_id: int
    route: tuple[int, ...]
    m: int
    B: float
    H: np.ndarray   # (m,)
    E: np.ndarray   # (m,)
    E0: float


@dataclass
class Paths:
    p: np.ndarray   # (N, m) float64, route order
    d: np.ndarray

    @property
    def g(self) -> np.ndarray:
        return self.p - self.d


def route_specs(inst: str, plan_key: str = "Det") -> list[RouteSpec]:
    """Route data in the same order as ``data.route_problem``; empty routes skipped."""
    D, Q, _, scale = parse_dethloff(DATA_DIR / f"{inst}.vrpspd")
    sol = load_plans(inst)
    dbar = sol["dbar"]
    costs = LastMileCosts()
    out: list[RouteSpec] = []
    for ri, route in enumerate(sol["res"][plan_key]["plan"]):
        if not route:
            continue
        r = np.array(route)
        B = float(Q - dbar[r].sum())
        H, E = route_cost_schedules(route, D, scale, costs)
        out.append(RouteSpec(inst=inst, route_id=ri,
                             route=tuple(int(x) for x in route),
                             m=len(route), B=B,
                             H=np.asarray(H, dtype=float),
                             E=np.asarray(E, dtype=float),
                             E0=float(E[0])))
    return out


def _stream(inst: str, purpose: str, rep: int, c: int) -> tuple[object, int]:
    """Seed and size of one generated chunk."""
    base = stable_seed(inst)
    if purpose == "train":
        return [base, 1, rep, c], CHUNK
    if purpose == "dev":
        return [base, 3], DEV_N
    if purpose == "test":
        return base + 99_991, TEST_N
    if purpose == "test_std":
        return base + 99_991, TEST_STD_N
    raise ValueError(f"unknown purpose {purpose!r}")


@functools.lru_cache(maxsize=1)
def _generate_chunk(inst: str, purpose: str, rep: int, c: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(deliveries, pickups) for every customer of one chunk, shape (n, cust)."""
    seed, n = _stream(inst, purpose, rep, c)
    sol = load_plans(inst)
    return gen_scenarios(sol["dbar"], sol["pbar"], n, seed)


def route_pool(inst: str, purpose: str, rep: int = 0,
               N: int | None = None) -> dict[int, Paths]:
    """route_id -> Paths for every non-empty route of the 'Det' plan."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    specs = route_specs(inst)
    if purpose == "train":
        if rep not in TRAIN_REPS:
            raise ValueError("train replicate must be 0, 1 or 2")
        n_total = CHUNK if N is None else int(N)
        if not 1 <= n_total <= N_CHUNKS * CHUNK:
            raise ValueError(f"N must be in [1, {N_CHUNKS * CHUNK}]")
        chunks = [(c, min(CHUNK, n_total - c * CHUNK))
                  for c in range(-(-n_total // CHUNK))]
    else:
        size = {"dev": DEV_N, "test": TEST_N, "test_std": TEST_STD_N}[purpose]
        if N is not None and int(N) != size:
            raise ValueError(f"{purpose} has a fixed size of {size}")
        chunks = [(0, size)]

    parts_p: dict[int, list[np.ndarray]] = {s.route_id: [] for s in specs}
    parts_d: dict[int, list[np.ndarray]] = {s.route_id: [] for s in specs}
    for c, take in chunks:
        dsc, psc = _generate_chunk(inst, purpose, rep, c)
        for s in specs:
            r = np.array(s.route)
            parts_p[s.route_id].append(psc[:take, r].copy())
            parts_d[s.route_id].append(dsc[:take, r].copy())

    out: dict[int, Paths] = {}
    for s in specs:
        out[s.route_id] = Paths(p=np.concatenate(parts_p[s.route_id], axis=0),
                                d=np.concatenate(parts_d[s.route_id], axis=0))
    return out
