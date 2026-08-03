"""
Failure-first probe: does a normal-day VRPSPD plan affect recoverability from a
single pre-departure vehicle outage under a limited same-day repair class?

The script is deliberately self-contained.  It copies only the deterministic
parser / ILS ingredients needed from the existing Dethloff runner and does not
import BATON, CVaR, stochastic-demand, or any other project file.

Repair class A(delta)
---------------------
* One planned vehicle/route is unavailable before departure.
* Surviving route customer orders are preserved.
* The lost route order is preserved.
* The lost route may be split into contiguous blocks.
* Each surviving route may accept at most one such block, inserted at its best
  position, provided nominal VRPSPD peak load remains <= Q and its own distance
  increase is <= delta * original route distance.
* If all lost customers can be absorbed, no extra vehicle is required.
  Otherwise one extra vehicle can run the residual lost-route sequence, so the
  single-outage emergency requirement is 1.

Thus the primary metric is not a rich severity distribution. It is the fraction
of possible single-route outages that can be absorbed internally (R_A^*=0),
plus how much of the lost workload can be absorbed when full recovery fails.

Typical command (Windows CMD / PowerShell):
  python probe_freight_recoverability.py instance="HANOI-100-1(1).vrpspd" ^
      seeds=0:20 t=8 noimp=3 splits=0,1,2 deltas=0,0.05,0.10,0.20 pair_tol=0.02

Quick smoke:
  python probe_freight_recoverability.py instance="HANOI-100-1(1).vrpspd" ^
      seeds=0:3 t=1 noimp=0.5 deltas=0.05,0.10
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9


# ============================================================================
# Parsing
# ============================================================================

def _lines_after(text: str, tag: str) -> List[str]:
    out: List[str] = []
    capture = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not capture:
            if s.upper().startswith(tag.upper()):
                capture = True
            continue
        # Sections in this format contain only numeric rows. Stop at next tag.
        if any(ch.isalpha() for ch in s):
            break
        out.append(s)
    return out


def _header_int(text: str, key: str) -> Optional[int]:
    for line in text.splitlines():
        if line.strip().upper().startswith(key.upper()):
            m = re.findall(r"-?\d+", line)
            if m:
                return int(m[-1])
    return None


def _header_float(text: str, key: str) -> Optional[float]:
    for line in text.splitlines():
        if line.strip().upper().startswith(key.upper()):
            m = re.findall(r"-?\d+(?:\.\d+)?", line)
            if m:
                return float(m[-1])
    return None


def parse_vrpspd(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    text = path.read_text(errors="ignore")
    n = _header_int(text, "DIMENSION")
    cap = _header_float(text, "CAPACITY")
    if n is None or cap is None:
        raise ValueError("Missing DIMENSION or CAPACITY header")

    tokens: List[str] = []
    for row in _lines_after(text, "EDGE_WEIGHT_SECTION"):
        tokens.extend(row.split())
    vals = [float(x) for x in tokens]
    if len(vals) != n * n:
        raise ValueError(
            f"EDGE_WEIGHT_SECTION has {len(vals)} values; expected {n*n}"
        )
    D = np.asarray(vals, dtype=float).reshape(n, n)

    delivery = np.zeros(n, dtype=float)
    pickup = np.zeros(n, dtype=float)
    rows = _lines_after(text, "PICKUP_AND_DELIVERY_SECTION")
    for row in rows:
        t = row.split()
        if len(t) < 3:
            continue
        idx = int(float(t[0])) - 1
        if 0 <= idx < n:
            delivery[idx] = float(t[-2])
            pickup[idx] = float(t[-1])

    return D, delivery, pickup, float(cap), int(n)


# ============================================================================
# Deterministic VRPSPD primitives
# ============================================================================

def route_cost(route: Sequence[int], D: np.ndarray) -> float:
    if not route:
        return 0.0
    total = D[0, route[0]] + D[route[-1], 0]
    for a, b in zip(route, route[1:]):
        total += D[a, b]
    return float(total)


def nominal_peak(route: Sequence[int], delivery: np.ndarray, pickup: np.ndarray) -> float:
    if not route:
        return 0.0
    idx = np.asarray(route, dtype=int)
    d = delivery[idx]
    p = pickup[idx]
    total_d = float(d.sum())
    middle = total_d - np.cumsum(d) + np.cumsum(p)
    return float(max(total_d, float(middle.max(initial=0.0))))


class DetGate:
    def __init__(self, cap: float, delivery: np.ndarray, pickup: np.ndarray):
        self.cap = float(cap)
        self.delivery = delivery
        self.pickup = pickup

    def feasible(self, route: Sequence[int]) -> bool:
        return nominal_peak(route, self.delivery, self.pickup) <= self.cap + EPS


def solution_distance(solution: Sequence[Sequence[int]], D: np.ndarray) -> float:
    return float(sum(route_cost(r, D) for r in solution if r))


def validate_solution(
    solution: Sequence[Sequence[int]],
    n: int,
    gate: DetGate,
) -> None:
    flat = [c for r in solution for c in r]
    expected = list(range(1, n))
    if sorted(flat) != expected:
        missing = sorted(set(expected) - set(flat))
        dup = sorted({c for c in flat if flat.count(c) > 1})
        raise AssertionError(f"Invalid coverage: missing={missing[:10]}, duplicates={dup[:10]}")
    bad = [i for i, r in enumerate(solution) if not gate.feasible(r)]
    if bad:
        raise AssertionError(f"Infeasible routes at indices {bad}")


# ============================================================================
# Existing deterministic ILS core, copied into this self-contained probe
# ============================================================================

def cw_init(D: np.ndarray, gate: DetGate, n: int) -> List[List[int]]:
    routes = [[c] for c in range(1, n) if gate.feasible([c])]
    placed = {c for r in routes for c in r}
    leftovers = [c for c in range(1, n) if c not in placed]

    savings: List[Tuple[float, int, int]] = []
    for a in range(1, n):
        for b in range(a + 1, n):
            savings.append((float(D[0, a] + D[0, b] - D[a, b]), a, b))
    savings.sort(reverse=True)

    active: Dict[int, List[int]] = {i: r for i, r in enumerate(routes)}
    where: Dict[int, int] = {r[0]: i for i, r in active.items()}

    for saving, a, b in savings:
        if saving <= 0:
            break
        ra, rb = where.get(a), where.get(b)
        if ra is None or rb is None or ra == rb or ra not in active or rb not in active:
            continue
        A, B = active[ra], active[rb]
        if A[-1] == a and B[0] == b:
            merged = A + B
        elif A[0] == a and B[-1] == b:
            merged = B + A
        elif A[-1] == a and B[-1] == b:
            merged = A + B[::-1]
        elif A[0] == a and B[0] == b:
            merged = A[::-1] + B
        else:
            continue
        if not gate.feasible(merged):
            continue
        active[ra] = merged
        for c in B:
            where[c] = ra
        del active[rb]

    solution = [r for r in active.values()]
    solution.extend([[c] for c in leftovers])
    return solution


def two_opt_gate(route: List[int], D: np.ndarray, gate: DetGate) -> List[int]:
    if len(route) < 4:
        return route[:]
    r = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(r) - 1):
            for k in range(i + 1, len(r)):
                a = r[i - 1] if i > 0 else 0
                b = r[i]
                c = r[k]
                d = r[k + 1] if k + 1 < len(r) else 0
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -EPS:
                    cand = r[:i] + r[i : k + 1][::-1] + r[k + 1 :]
                    if gate.feasible(cand):
                        r = cand
                        improved = True
                        break
            if improved:
                break
    return r


def relocate_gate(solution: List[List[int]], D: np.ndarray, gate: DetGate) -> List[List[int]]:
    sol = [r[:] for r in solution]
    improved = True
    while improved:
        improved = False
        for ri in range(len(sol)):
            R = sol[ri]
            for pi in range(len(R)):
                c = R[pi]
                a = R[pi - 1] if pi > 0 else 0
                b = R[pi + 1] if pi + 1 < len(R) else 0
                removal_gain = D[a, c] + D[c, b] - D[a, b]
                for rj in range(len(sol)):
                    if rj == ri:
                        continue
                    S = sol[rj]
                    for q in range(len(S) + 1):
                        u = S[q - 1] if q > 0 else 0
                        v = S[q] if q < len(S) else 0
                        delta = D[u, c] + D[c, v] - D[u, v] - removal_gain
                        if delta >= -EPS:
                            continue
                        new_R = R[:pi] + R[pi + 1 :]
                        new_S = S[:q] + [c] + S[q:]
                        if gate.feasible(new_R) and gate.feasible(new_S):
                            sol[ri] = new_R
                            sol[rj] = new_S
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        sol = [r for r in sol if r]
    return sol


def greedy_insert(
    solution: Sequence[Sequence[int]],
    customer: int,
    D: np.ndarray,
    gate: DetGate,
) -> Tuple[Optional[int], Optional[int], float]:
    best: Tuple[Optional[int], Optional[int], float] = (None, None, math.inf)
    for ri, route in enumerate(solution):
        R = list(route)
        for pos in range(len(R) + 1):
            cand = R[:pos] + [customer] + R[pos:]
            if not gate.feasible(cand):
                continue
            u = R[pos - 1] if pos > 0 else 0
            v = R[pos] if pos < len(R) else 0
            delta = float(D[u, customer] + D[customer, v] - D[u, v])
            if delta < best[2]:
                best = (ri, pos, delta)
    return best


def ruin_recreate(
    solution: Sequence[Sequence[int]],
    D: np.ndarray,
    gate: DetGate,
    rng: random.Random,
    q_frac: float,
) -> List[List[int]]:
    sol = [list(r) for r in solution]
    customers = [c for r in sol for c in r]
    q = max(1, int(q_frac * len(customers)))
    removed = rng.sample(customers, min(q, len(customers)))
    removed_set = set(removed)
    sol = [[c for c in r if c not in removed_set] for r in sol]
    sol = [r for r in sol if r]
    rng.shuffle(removed)
    for c in removed:
        ri, pos, _ = greedy_insert(sol, c, D, gate)
        if ri is None or pos is None:
            sol.append([c])
        else:
            sol[ri].insert(pos, c)
    return sol


def local_search(solution: Sequence[Sequence[int]], D: np.ndarray, gate: DetGate) -> List[List[int]]:
    sol = [two_opt_gate(list(r), D, gate) for r in solution if r]
    return relocate_gate(sol, D, gate)


def relocate_fixed_k(solution: List[List[int]], D: np.ndarray, gate: DetGate) -> List[List[int]]:
    """Distance descent that never empties a route, so fleet size stays fixed."""
    sol = [r[:] for r in solution]
    improved = True
    while improved:
        improved = False
        for ri in range(len(sol)):
            R = sol[ri]
            if len(R) <= 1:
                continue
            for pi in range(len(R)):
                c = R[pi]
                a = R[pi - 1] if pi > 0 else 0
                b = R[pi + 1] if pi + 1 < len(R) else 0
                removal_gain = D[a, c] + D[c, b] - D[a, b]
                for rj in range(len(sol)):
                    if rj == ri:
                        continue
                    S = sol[rj]
                    for q in range(len(S) + 1):
                        u = S[q - 1] if q > 0 else 0
                        v = S[q] if q < len(S) else 0
                        delta = D[u, c] + D[c, v] - D[u, v] - removal_gain
                        if delta >= -EPS:
                            continue
                        new_R = R[:pi] + R[pi + 1 :]
                        new_S = S[:q] + [c] + S[q:]
                        if gate.feasible(new_R) and gate.feasible(new_S):
                            sol[ri] = new_R
                            sol[rj] = new_S
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return sol


def fixed_k_local_search(solution: Sequence[Sequence[int]], D: np.ndarray, gate: DetGate) -> List[List[int]]:
    sol = [two_opt_gate(list(r), D, gate) for r in solution if r]
    sol = relocate_fixed_k(sol, D, gate)
    return [two_opt_gate(list(r), D, gate) for r in sol]


def split_variant(
    solution: Sequence[Sequence[int]],
    split_count: int,
    D: np.ndarray,
    gate: DetGate,
    seed: int,
) -> List[List[int]]:
    """
    Add planned slack by splitting routes, then improve distance while preserving K.
    This is only a plan-diversification device; all plans are evaluated under the
    same normal-day distance metric and compared only against equal-K plans.
    """
    rng = random.Random(seed)
    sol = [list(r) for r in solution]
    for _ in range(split_count):
        candidates = [i for i, r in enumerate(sol) if len(r) >= 2]
        if not candidates:
            break
        # Prefer long/high-cost routes, but randomize among the top few to avoid
        # producing the same K+1/K+2 partition from every seed.
        candidates.sort(key=lambda i: (len(sol[i]), route_cost(sol[i], D)), reverse=True)
        chosen = rng.choice(candidates[: min(3, len(candidates))])
        route = sol[chosen]
        split_positions = list(range(1, len(route)))
        best_value = math.inf
        best_positions: List[int] = []
        for pos in split_positions:
            left, right = route[:pos], route[pos:]
            value = route_cost(left, D) + route_cost(right, D)
            if value < best_value - EPS:
                best_value = value
                best_positions = [pos]
            elif abs(value - best_value) <= EPS:
                best_positions.append(pos)
        pos = rng.choice(best_positions)
        sol[chosen : chosen + 1] = [route[:pos], route[pos:]]
        sol = fixed_k_local_search(sol, D, gate)
    return sol


def economic_cost(solution: Sequence[Sequence[int]], D: np.ndarray, omega_vehicle: float) -> float:
    return solution_distance(solution, D) + omega_vehicle * len([r for r in solution if r])


def solve_plan(
    D: np.ndarray,
    gate: DetGate,
    n: int,
    omega_vehicle: float,
    time_limit: float,
    no_improve: float,
    seed: int,
) -> List[List[int]]:
    rng = random.Random(seed)
    current = local_search(cw_init(D, gate, n), D, gate)
    best = [r[:] for r in current]
    best_cost = economic_cost(best, D, omega_vehicle)
    start = time.time()
    last_improve = start

    while time.time() - start < time_limit:
        candidate = local_search(
            ruin_recreate(best, D, gate, rng, rng.choice([0.10, 0.15, 0.20, 0.30])),
            D,
            gate,
        )
        value = economic_cost(candidate, D, omega_vehicle)
        if value < best_cost - EPS:
            best = [r[:] for r in candidate]
            best_cost = value
            last_improve = time.time()
        elif time.time() - last_improve > no_improve:
            break
    return [r for r in best if r]


# ============================================================================
# Limited-repair oracle
# ============================================================================

@dataclass(frozen=True)
class InsertionOption:
    survivor: int
    start: int
    end: int  # exclusive
    position: int
    delta: float


@dataclass
class OutageResult:
    fully_absorbable: bool
    extra_units: int
    max_absorbed_customers: int
    max_absorbed_workload: float
    lost_customers: int
    lost_workload: float
    min_full_absorption_delta: Optional[float]
    options_count: int


def best_block_insertion(
    base_route: Sequence[int],
    block: Sequence[int],
    D: np.ndarray,
    gate: DetGate,
    delta_frac: float,
) -> Optional[Tuple[int, float]]:
    """Best feasible insertion of one contiguous block into one survivor route."""
    base = list(base_route)
    base_cost = route_cost(base, D)
    allowed = delta_frac * base_cost
    best_pos: Optional[int] = None
    best_delta = math.inf

    for pos in range(len(base) + 1):
        cand = base[:pos] + list(block) + base[pos:]
        if not gate.feasible(cand):
            continue
        delta = route_cost(cand, D) - base_cost
        if delta <= allowed + EPS and delta < best_delta:
            best_delta = float(delta)
            best_pos = pos

    if best_pos is None:
        return None
    return best_pos, best_delta


def build_insertion_options(
    lost_route: Sequence[int],
    survivors: Sequence[Sequence[int]],
    D: np.ndarray,
    gate: DetGate,
    delta_frac: float,
) -> Dict[int, List[InsertionOption]]:
    """All feasible block insertions keyed by block start index."""
    lost = list(lost_route)
    by_start: Dict[int, List[InsertionOption]] = {i: [] for i in range(len(lost))}
    for start in range(len(lost)):
        for end in range(start + 1, len(lost) + 1):
            block = lost[start:end]
            for survivor_idx, base in enumerate(survivors):
                ans = best_block_insertion(base, block, D, gate, delta_frac)
                if ans is None:
                    continue
                pos, delta = ans
                by_start[start].append(
                    InsertionOption(
                        survivor=survivor_idx,
                        start=start,
                        end=end,
                        position=pos,
                        delta=delta,
                    )
                )
    return by_start


def full_absorption_dp(
    m: int,
    n_survivors: int,
    options: Dict[int, List[InsertionOption]],
) -> Optional[float]:
    """
    Exact DP for this repair class: partition the whole lost route into blocks;
    assign each block to a distinct survivor. Returns minimum total added
    distance, or None when full internal absorption is impossible.
    """
    states: List[Dict[int, float]] = [dict() for _ in range(m + 1)]
    states[0][0] = 0.0
    for pos in range(m):
        if not states[pos]:
            continue
        for mask, current_delta in list(states[pos].items()):
            for opt in options.get(pos, []):
                bit = 1 << opt.survivor
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_delta = current_delta + opt.delta
                old = states[opt.end].get(new_mask)
                if old is None or new_delta < old:
                    states[opt.end][new_mask] = new_delta
    if not states[m]:
        return None
    return float(min(states[m].values()))


def max_absorption_dp(
    lost_route: Sequence[int],
    n_survivors: int,
    options: Dict[int, List[InsertionOption]],
    customer_weights: np.ndarray,
) -> Tuple[int, float]:
    """
    Exact maximum internally absorbed customer count and workload when
    unabsorbed customers may stay on one emergency route in original order.

    State walks left-to-right. It may skip a customer (leave it to emergency)
    or assign a contiguous block to one previously unused survivor.
    """
    m = len(lost_route)
    # (pos, mask) -> Pareto-best (count, workload); count first, workload tie-break.
    states: List[Dict[int, Tuple[int, float]]] = [dict() for _ in range(m + 1)]
    states[0][0] = (0, 0.0)

    def update(pos: int, mask: int, value: Tuple[int, float]) -> None:
        old = states[pos].get(mask)
        if old is None or value[0] > old[0] or (value[0] == old[0] and value[1] > old[1] + EPS):
            states[pos][mask] = value

    for pos in range(m):
        if not states[pos]:
            continue
        for mask, (count, work) in list(states[pos].items()):
            # Leave this customer to the emergency route.
            update(pos + 1, mask, (count, work))

            for opt in options.get(pos, []):
                bit = 1 << opt.survivor
                if mask & bit:
                    continue
                block_nodes = lost_route[opt.start : opt.end]
                add_work = float(customer_weights[np.asarray(block_nodes, dtype=int)].sum())
                update(
                    opt.end,
                    mask | bit,
                    (count + (opt.end - opt.start), work + add_work),
                )

    best = max(states[m].values(), key=lambda x: (x[0], x[1]))
    return int(best[0]), float(best[1])


def evaluate_route_outage(
    solution: Sequence[Sequence[int]],
    lost_idx: int,
    D: np.ndarray,
    gate: DetGate,
    delivery: np.ndarray,
    pickup: np.ndarray,
    delta_frac: float,
) -> OutageResult:
    lost = list(solution[lost_idx])
    survivors = [list(r) for i, r in enumerate(solution) if i != lost_idx]
    weights = delivery + pickup

    options = build_insertion_options(lost, survivors, D, gate, delta_frac)
    option_count = sum(len(v) for v in options.values())
    min_delta = full_absorption_dp(len(lost), len(survivors), options)
    max_count, max_work = max_absorption_dp(lost, len(survivors), options, weights)

    lost_work = float(weights[np.asarray(lost, dtype=int)].sum())
    fully = min_delta is not None
    return OutageResult(
        fully_absorbable=fully,
        extra_units=0 if fully else 1,
        max_absorbed_customers=max_count,
        max_absorbed_workload=max_work,
        lost_customers=len(lost),
        lost_workload=lost_work,
        min_full_absorption_delta=min_delta,
        options_count=option_count,
    )


# ============================================================================
# Experiment orchestration
# ============================================================================

def plan_signature(solution: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    # Routes are an unordered fleet, but route orientation/order is meaningful.
    return tuple(sorted(tuple(r) for r in solution))


def parse_seed_spec(spec: str) -> List[int]:
    spec = spec.strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError("seeds must be comma-list or start:stop[:step]")
        return list(range(start, stop, step))
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_args(argv: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            out[key.strip().lower()] = value.strip()
        else:
            out[arg.strip().lower()] = "1"
    return out


def choose_instance(arg: Optional[str]) -> Path:
    if arg:
        path = Path(arg)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    preferred = Path("HANOI-100-1(1).vrpspd")
    if preferred.exists():
        return preferred
    files = sorted(Path(".").glob("*.vrpspd"))
    if not files:
        raise FileNotFoundError("No .vrpspd file found; pass instance=<path>")
    return files[0]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    instance = choose_instance(args.get("instance"))
    seeds = parse_seed_spec(args.get("seeds", "0:12"))
    time_limit = float(args.get("t", "5"))
    no_improve = float(args.get("noimp", "2"))
    deltas = [float(x) for x in args.get("deltas", "0,0.05,0.10,0.20").split(",")]
    pair_tol = float(args.get("pair_tol", "0.02"))
    omega_mult = float(args.get("omega_mult", "1.0"))
    out_prefix = args.get("out", "recoverability_probe")
    max_unique = int(args.get("max_plans", "999999"))
    split_levels = [int(x) for x in args.get("splits", "0,1,2").split(",") if x.strip()]

    D, delivery, pickup, Q, n = parse_vrpspd(instance)
    gate = DetGate(Q, delivery, pickup)
    positive_edges = D[D > 0]
    omega_vehicle = omega_mult * float(positive_edges.mean())

    print("=" * 104)
    print(" LIMITED-REPAIR FREIGHT RECOVERABILITY PROBE")
    print("=" * 104)
    print(f" instance={instance}  n={n-1} customers  Q={Q:g}")
    print(f" seeds={seeds}  t={time_limit:g}s/seed  noimp={no_improve:g}s  split_levels={split_levels}")
    print(f" delta frontier={deltas}  pair_tol={100*pair_tol:.2f}%  omega_vehicle={omega_vehicle:.2f}")
    print(" repair: one pre-departure route loss; survivor orders fixed; lost route split into contiguous")
    print("         blocks; <=1 inserted block per survivor; per-survivor detour budget delta; exact DP")
    print("-" * 104)

    plans: List[Dict[str, object]] = []
    signatures = set()
    for ordinal, seed in enumerate(seeds, 1):
        start = time.time()
        base_solution = solve_plan(D, gate, n, omega_vehicle, time_limit, no_improve, seed)
        validate_solution(base_solution, n, gate)
        base_elapsed = time.time() - start

        for split_level in split_levels:
            variant_start = time.time()
            if split_level == 0:
                solution = [r[:] for r in base_solution]
                variant_elapsed = base_elapsed
            else:
                solution = split_variant(
                    base_solution, split_level, D, gate, seed=100003 * seed + split_level
                )
                validate_solution(solution, n, gate)
                variant_elapsed = base_elapsed + (time.time() - variant_start)

            sig = plan_signature(solution)
            dist = solution_distance(solution, D)
            K = len(solution)
            duplicate = sig in signatures
            print(
                f"[base {ordinal:>2}/{len(seeds)}] seed={seed:<4} split={split_level:<2} K={K:<3} "
                f"dist={dist:>12.1f} time={variant_elapsed:>6.2f}s "
                f"{'DUPLICATE' if duplicate else 'NEW'}"
            )
            if duplicate:
                continue
            signatures.add(sig)
            plans.append(
                {
                    "plan_id": len(plans),
                    "seed": seed,
                    "split_level": split_level,
                    "solution": solution,
                    "K": K,
                    "distance": dist,
                    "solve_time": variant_elapsed,
                    "signature": sig,
                }
            )
            if len(plans) >= max_unique:
                break
        if len(plans) >= max_unique:
            break

    if len(plans) < 2:
        print("ERROR: fewer than two unique plans; increase seeds/time or vary omega_mult.")
        return 2

    print("-" * 104)
    print(f" unique plans={len(plans)}; evaluating all route outages across {len(deltas)} delta values")

    outage_rows: List[Dict[str, object]] = []
    plan_rows: List[Dict[str, object]] = []
    plan_metric: Dict[Tuple[int, float], Dict[str, float]] = {}

    for plan in plans:
        pid = int(plan["plan_id"])
        solution = plan["solution"]  # type: ignore[assignment]
        assert isinstance(solution, list)

        base_row: Dict[str, object] = {
            "plan_id": pid,
            "seed": int(plan["seed"]),
            "split_level": int(plan["split_level"]),
            "K": int(plan["K"]),
            "distance": float(plan["distance"]),
            "solve_time": float(plan["solve_time"]),
        }

        for delta in deltas:
            full = 0
            total_abs_frac = 0.0
            total_work_frac = 0.0
            full_delta_values: List[float] = []

            print(f"  plan={pid:>2} seed={plan['seed']:<4} delta={delta:>6.3f}", end="", flush=True)
            tick = time.time()
            for lost_idx in range(len(solution)):
                result = evaluate_route_outage(
                    solution,
                    lost_idx,
                    D,
                    gate,
                    delivery,
                    pickup,
                    delta,
                )
                full += int(result.fully_absorbable)
                abs_frac = result.max_absorbed_customers / max(1, result.lost_customers)
                work_frac = result.max_absorbed_workload / max(EPS, result.lost_workload)
                total_abs_frac += abs_frac
                total_work_frac += work_frac
                if result.min_full_absorption_delta is not None:
                    full_delta_values.append(result.min_full_absorption_delta)

                outage_rows.append(
                    {
                        "plan_id": pid,
                        "seed": int(plan["seed"]),
                        "split_level": int(plan["split_level"]),
                        "K": int(plan["K"]),
                        "distance": float(plan["distance"]),
                        "delta_frac": delta,
                        "lost_route_idx": lost_idx,
                        "lost_route_len": result.lost_customers,
                        "lost_route_cost": route_cost(solution[lost_idx], D),
                        "lost_workload": result.lost_workload,
                        "fully_absorbable": int(result.fully_absorbable),
                        "extra_units": result.extra_units,
                        "max_absorbed_customers": result.max_absorbed_customers,
                        "absorbed_customer_frac": abs_frac,
                        "max_absorbed_workload": result.max_absorbed_workload,
                        "absorbed_workload_frac": work_frac,
                        "min_full_absorption_delta": (
                            "" if result.min_full_absorption_delta is None else result.min_full_absorption_delta
                        ),
                        "feasible_block_insertions": result.options_count,
                        "lost_route": " ".join(map(str, solution[lost_idx])),
                    }
                )

            count = len(solution)
            recovery_rate = full / max(1, count)
            mean_abs = total_abs_frac / max(1, count)
            mean_work = total_work_frac / max(1, count)
            mean_delta = float(np.mean(full_delta_values)) if full_delta_values else math.nan
            elapsed = time.time() - tick
            print(
                f"  P(extra=0)={recovery_rate:>6.3f}  "
                f"absCust={mean_abs:>6.3f} absWork={mean_work:>6.3f}  t={elapsed:>6.2f}s"
            )

            base_row[f"recovery_rate_d{delta:g}"] = recovery_rate
            base_row[f"mean_absorbed_customer_frac_d{delta:g}"] = mean_abs
            base_row[f"mean_absorbed_workload_frac_d{delta:g}"] = mean_work
            base_row[f"mean_full_absorption_delta_d{delta:g}"] = mean_delta
            plan_metric[(pid, delta)] = {
                "recovery_rate": recovery_rate,
                "mean_absorbed_customer_frac": mean_abs,
                "mean_absorbed_workload_frac": mean_work,
            }

        plan_rows.append(base_row)

    # Near-equal-cost, equal-K pair audit.
    pair_rows: List[Dict[str, object]] = []
    print("-" * 104)
    print(" NEAR-EQUAL NORMAL-DAY PLAN PAIRS")
    pair_count = 0
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            A, B = plans[i], plans[j]
            if int(A["K"]) != int(B["K"]):
                continue
            ca, cb = float(A["distance"]), float(B["distance"])
            rel_gap = abs(ca - cb) / max(EPS, min(ca, cb))
            if rel_gap > pair_tol + EPS:
                continue
            pair_count += 1
            for delta in deltas:
                ma = plan_metric[(int(A["plan_id"]), delta)]
                mb = plan_metric[(int(B["plan_id"]), delta)]
                pair_rows.append(
                    {
                        "plan_a": int(A["plan_id"]),
                        "seed_a": int(A["seed"]),
                        "split_a": int(A["split_level"]),
                        "plan_b": int(B["plan_id"]),
                        "seed_b": int(B["seed"]),
                        "split_b": int(B["split_level"]),
                        "K": int(A["K"]),
                        "distance_a": ca,
                        "distance_b": cb,
                        "relative_cost_gap": rel_gap,
                        "delta_frac": delta,
                        "recovery_rate_a": ma["recovery_rate"],
                        "recovery_rate_b": mb["recovery_rate"],
                        "abs_recovery_gap": abs(ma["recovery_rate"] - mb["recovery_rate"]),
                        "absorbed_customer_frac_a": ma["mean_absorbed_customer_frac"],
                        "absorbed_customer_frac_b": mb["mean_absorbed_customer_frac"],
                        "abs_absorbed_customer_gap": abs(
                            ma["mean_absorbed_customer_frac"] - mb["mean_absorbed_customer_frac"]
                        ),
                        "absorbed_workload_frac_a": ma["mean_absorbed_workload_frac"],
                        "absorbed_workload_frac_b": mb["mean_absorbed_workload_frac"],
                        "abs_absorbed_workload_gap": abs(
                            ma["mean_absorbed_workload_frac"] - mb["mean_absorbed_workload_frac"]
                        ),
                    }
                )

    if pair_rows:
        for delta in deltas:
            candidates = [r for r in pair_rows if float(r["delta_frac"]) == delta]
            best = max(
                candidates,
                key=lambda r: (
                    float(r["abs_recovery_gap"]),
                    float(r["abs_absorbed_workload_gap"]),
                ),
            )
            print(
                f" delta={delta:>6.3f}: best pair {best['plan_a']} vs {best['plan_b']}  "
                f"costGap={100*float(best['relative_cost_gap']):>5.2f}%  "
                f"recoveryGap={float(best['abs_recovery_gap']):>6.3f}  "
                f"workGap={float(best['abs_absorbed_workload_gap']):>6.3f}"
            )
    else:
        print(f" no equal-K pairs within {100*pair_tol:.2f}% distance; probe is INCONCLUSIVE")

    # Persist routes and tables.
    prefix = Path(out_prefix)
    plans_csv = prefix.with_name(prefix.name + "_plans.csv")
    outages_csv = prefix.with_name(prefix.name + "_outages.csv")
    pairs_csv = prefix.with_name(prefix.name + "_pairs.csv")
    routes_json = prefix.with_name(prefix.name + "_routes.json")

    write_csv(plans_csv, plan_rows)
    write_csv(outages_csv, outage_rows)
    write_csv(pairs_csv, pair_rows)
    routes_json.write_text(
        json.dumps(
            {
                "instance": str(instance),
                "capacity": Q,
                "repair_class": {
                    "outage_time": "pre-departure",
                    "survivor_order_fixed": True,
                    "lost_route_order_fixed": True,
                    "lost_route_contiguous_blocks": True,
                    "max_blocks_per_survivor": 1,
                    "per_survivor_detour_fractions": deltas,
                },
                "plans": [
                    {
                        "plan_id": int(p["plan_id"]),
                        "seed": int(p["seed"]),
                        "split_level": int(p["split_level"]),
                        "K": int(p["K"]),
                        "distance": float(p["distance"]),
                        "routes": p["solution"],
                    }
                    for p in plans
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("-" * 104)
    print(f" wrote {plans_csv}")
    print(f" wrote {outages_csv}")
    print(f" wrote {pairs_csv}")
    print(f" wrote {routes_json}")
    print("\nInterpretation gate:")
    print("  * A stable recovery-rate gap between equal-K, near-equal-distance plans means plan structure matters.")
    print("  * Near-zero gaps across a delta interval means this limited-repair semantics is likely too weak/trivial.")
    print("  * This probe makes no outage-probability or empirical-Hanoi reliability claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
