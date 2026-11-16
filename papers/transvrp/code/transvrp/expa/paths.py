"""Route-level scenario pools for EXPERIMENT A (module M1).

Purposes and seeds follow EXPERIMENT_A.md section 5. ``base`` is
``data.stable_seed(inst)``.

* train, replicate r, chunk c: ``gen_scenarios`` with seed ``[base, 1, r, c]``,
  100,000 paths per chunk. N <= 10^5 uses the first N rows of chunk 0; larger N
  uses chunks 0.. in order (N <= 10^6).
* dev: seed ``[base, 3]``, 5,000 paths. ``rep`` is ignored.
* test: seed ``base + 99991``, 20,000 paths. ``rep`` is ignored.
* test_std: seed ``base + 99991``, 2,000 paths (parity with ``route_problem``).
  It shares its seed with test (its rows are the first 2,000 test rows): use it
  for parity tests only, never for tuning.

Only route columns are returned (``routes=`` restricts them further). Chunks
are processed one at a time and released after their route columns are copied.
With ``cache_dir`` each chunk is also kept on disk, so the routes of one
instance (and repeated jobs) reuse it instead of regenerating it. Cache
directories belong to a run (sweep uses ``<results-dir>/chunks``), never to the
repository.
"""
from __future__ import annotations

import os
import re
import subprocess
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
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


_PORCELAIN_LINE = re.compile(r"^(.{2}) (.+)$")
_GATE_ISSUER = object()    # only verify_test_gate holds this object


def tree_is_clean(path: Path = CODE_DIR) -> bool:
    """True only when the checkout under ``path`` is clean for the gate:

    * no tracked file has uncommitted changes (any porcelain line whose status
      is not ``??`` makes the tree dirty), and
    * no untracked ``.py`` file exists (untracked data files such as results
      are allowed).

    Runs ``git status --porcelain --untracked-files=all -- .`` inside ``path``.
    Fails closed: a non-zero exit, a launch error, or any stdout line that is not
    in porcelain form counts as dirty."""
    try:
        out = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                              "--untracked-files=all", "--", "."],
                             capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    for line in out.stdout.splitlines():
        m = _PORCELAIN_LINE.match(line)
        if m is None:
            return False                      # not parseable: fail closed
        status, name = m.group(1), m.group(2).strip().strip('"')
        if status != "??":
            return False                      # tracked change (staged or not)
        if name.endswith(".py"):
            return False                      # untracked source file
    return True


@dataclass(frozen=True)
class GateToken:
    """Proof that ``verify_test_gate`` passed for ``commit``. Only that function
    creates one; private loaders accept a test-purpose request only with a token."""
    commit: str
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _GATE_ISSUER:
            raise TypeError("GateToken is issued by paths.verify_test_gate only")


def verify_test_gate(frozen: str | None) -> GateToken:
    """Data-layer gate for the test stream (EXPERIMENT_A.md section 5).

    Passes only when ``frozen`` is a full 40-character lowercase hash that is
    recorded in RESEARCH_LOG.md, equals the commit of the running checkout, and
    the code directory has a clean tree (``tree_is_clean``). Returns a
    ``GateToken`` for the verified hash. The log path, commit reader and tree
    check are read from module globals at call time (tests replace them; no
    caller can)."""
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
        raise TestStreamGateError("uncommitted or untracked .py changes under papers/transvrp/code; "
                                  "refusing test stream")
    return GateToken(commit=frozen, _issuer=_GATE_ISSUER)


def _require_gate(purpose: str, gate: GateToken | None) -> None:
    """Refuses test-purpose data unless ``gate`` is a token from verify_test_gate."""
    if purpose == "test" and not (isinstance(gate, GateToken)
                                  and gate._issuer is _GATE_ISSUER):
        raise TestStreamGateError("test-stream chunks need a GateToken from verify_test_gate")


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


def _stream(inst: str, purpose: str, rep: int, c: int,
            gate: GateToken | None = None) -> tuple[object, int]:
    """Seed and size of one generated chunk. Test-purpose streams need ``gate``."""
    _require_gate(purpose, gate)
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


def chunk_file(cache_dir: Path, inst: str, purpose: str, rep: int, c: int,
               seed: object, n: int) -> Path:
    """Path of one cached chunk. The key is (instance, purpose, rep, chunk, seed);
    the size n is included because it is fixed by the purpose."""
    seed_tag = format(zlib.crc32(repr(seed).encode()), "08x")
    return Path(cache_dir) / (f"{inst}__{purpose}__r{rep}__c{c}__n{n}__s{seed_tag}.npy")


def _generate_chunk(inst: str, purpose: str, rep: int, c: int,
                    gate: GateToken | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(deliveries, pickups) for every customer of one chunk, shape (n, cust)."""
    seed, n = _stream(inst, purpose, rep, c, gate)
    sol = load_plans(inst)
    return gen_scenarios(sol["dbar"], sol["pbar"], n, seed)


def _chunk_arrays(inst: str, purpose: str, rep: int, c: int,
                  cache_dir: Path | None, gate: GateToken | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """One chunk's (deliveries, pickups). With ``cache_dir`` the chunk is read from
    ``chunk_file`` when present and otherwise generated and saved there (one
    ``np.save`` of the stacked pair, written atomically). A cached chunk is the
    exact float64 array that generation returned, so both paths agree bit for bit.
    Test-purpose chunks are refused unless ``gate`` is a GateToken."""
    if cache_dir is None:
        return _generate_chunk(inst, purpose, rep, c, gate)
    seed, n = _stream(inst, purpose, rep, c, gate)
    path = chunk_file(cache_dir, inst, purpose, rep, c, seed, n)
    if path.exists():
        stacked = np.load(path)
        return stacked[0], stacked[1]
    dsc, psc = _generate_chunk(inst, purpose, rep, c, gate)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npy")
    np.save(tmp, np.stack([dsc, psc]))
    os.replace(tmp, path)
    return dsc, psc


def route_pool(inst: str, purpose: str, rep: int = 0,
               N: int | None = None, *, frozen: str | None = None,
               cache_dir: Path | None = None,
               routes: Iterable[int] | None = None) -> dict[int, Paths]:
    """route_id -> Paths for the non-empty routes of the 'Det' plan (or only
    ``routes`` when given).

    purpose 'test' is refused (TestStreamGateError) unless ``frozen`` passes
    ``verify_test_gate``; nothing is generated in that case. ``rep`` is ignored
    (fixed to 0) for every purpose except 'train', so dev and test pools do not
    depend on it. ``cache_dir`` enables the on-disk chunk cache (see
    ``_chunk_arrays``); it must be a run directory, never the repository."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    gate = verify_test_gate(frozen) if purpose == "test" else None
    if purpose != "train":
        rep = 0
    specs = route_specs(inst)
    if routes is not None:
        wanted = {int(r) for r in routes}
        specs = [s for s in specs if s.route_id in wanted]
        if len(specs) != len(wanted):
            raise ValueError(f"unknown or empty route ids for {inst}: {sorted(wanted)}")
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
        dsc, psc = _chunk_arrays(inst, purpose, rep, c, cache_dir, gate)
        for s in specs:
            r = np.array(s.route)
            # fancy indexing copies, so only the route columns outlive the chunk
            parts_p[s.route_id].append(psc[:take][:, r])
            parts_d[s.route_id].append(dsc[:take][:, r])
        del dsc, psc

    out: dict[int, Paths] = {}
    for s in specs:
        out[s.route_id] = Paths(p=np.concatenate(parts_p[s.route_id], axis=0),
                                d=np.concatenate(parts_d[s.route_id], axis=0))
    return out
