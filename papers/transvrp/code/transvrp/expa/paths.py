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
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..baton.costs import LastMileCosts, route_cost_schedules
from ..data import DATA_DIR, gen_scenarios, load_plans, parse_dethloff, stable_seed

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parents[1]
RESEARCH_LOG = CODE_DIR.parent / "RESEARCH_LOG.md"
FROZEN_HASH_RE = re.compile(r"[0-9a-f]{40}")

CHUNK = 100_000
N_CHUNKS = 10
PURPOSES = ("train", "dev", "test", "test_std")
DEV_N = 5_000
TEST_N = 20_000
TEST_STD_N = 2_000
TRAIN_REPS = (0, 1, 2)


class TestStreamGateError(RuntimeError):
    """Raised when test-stream data are requested without a verified frozen commit."""


def current_commit(start: Path = HERE) -> str:
    """Commit hash of the checkout that holds this file, read from .git files
    (no git process). Returns 'unknown' when it cannot be determined."""
    gitpath = None
    for d in [start, *start.parents]:
        if (d / ".git").exists():
            gitpath = d / ".git"
            break
    if gitpath is None:
        return "unknown"
    if gitpath.is_file():
        target = gitpath.read_text().split("gitdir:", 1)[-1].strip()
        gitdir = (gitpath.parent / target).resolve()
    else:
        gitdir = gitpath
    try:
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head[4:].strip()
        loose = gitdir / ref
        if loose.exists():
            return loose.read_text().strip()
        packed = gitdir / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        pass
    return "unknown"


def tree_is_clean(path: Path = CODE_DIR) -> bool:
    """True when no tracked file under ``path`` has uncommitted changes
    (``git status --porcelain --untracked-files=no``). Any failure counts as dirty."""
    try:
        out = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                              "--untracked-files=no", "--", "."],
                             capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.strip() == ""


def verify_test_gate(frozen: str | None) -> str:
    """Data-layer gate for the test stream (EXPERIMENT_A.md section 5).

    Passes only when ``frozen`` is a full 40-character lowercase hash that is
    recorded in RESEARCH_LOG.md, equals the commit of the running checkout, and
    the tracked files under papers/transvrp/code have no uncommitted changes.
    Returns the verified hash. The log path, commit reader and tree check are
    read from module globals at call time (tests replace them; no caller can)."""
    if not frozen:
        raise TestStreamGateError("test-stream data need a frozen commit hash")
    if not FROZEN_HASH_RE.fullmatch(frozen):
        raise TestStreamGateError("the frozen commit must be a full 40-character lowercase hash")
    if not Path(RESEARCH_LOG).exists():
        raise TestStreamGateError(f"research log not found: {RESEARCH_LOG}")
    if frozen not in Path(RESEARCH_LOG).read_text():
        raise TestStreamGateError("the frozen hash is not recorded in RESEARCH_LOG.md")
    head = current_commit()
    if head != frozen:
        raise TestStreamGateError(f"running checkout is at {head}, not at the frozen commit")
    if not tree_is_clean():
        raise TestStreamGateError("uncommitted changes under papers/transvrp/code; refusing test stream")
    return frozen


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
               N: int | None = None, *, frozen: str | None = None) -> dict[int, Paths]:
    """route_id -> Paths for every non-empty route of the 'Det' plan.

    purpose 'test' is refused (TestStreamGateError) unless ``frozen`` passes
    ``verify_test_gate``; nothing is generated in that case."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if purpose == "test":
        verify_test_gate(frozen)
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
