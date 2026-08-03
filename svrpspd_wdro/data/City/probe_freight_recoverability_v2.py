"""
Failure-first probe v2: does a balanced, directly generated fixed-K VRPSPD plan
change recoverability from a single pre-departure vehicle outage?

This script is self-contained. It deliberately removes the split-variant plan
generator that produced singleton pseudo-spare routes in v1. For each requested
fleet size K, it now:

1. creates a randomized giant tour from a deterministic feasible seed plan;
2. partitions that tour jointly into exactly K feasible routes by dynamic
   programming, subject to minimum route-size and minimum delivery-load floors;
3. improves the result with fixed-K perturbation/local search while preserving
   those floors.

The limited repair oracle is unchanged in spirit:
* one route is lost before departure;
* survivor orders and lost-route order remain fixed;
* the lost route may be partitioned into contiguous blocks;
* each survivor accepts at most one block;
* each survivor has a detour budget delta times its original route distance.

V2 also fixes the workload metric: maximum absorbed customer count and maximum
absorbed workload are solved by separate exact DPs. It reports recoverability
both over all route outages and over substantive outages (long or high-workload
routes), so a result cannot be driven only by tiny routes.

Typical command (Windows CMD / PowerShell):
  python probe_freight_recoverability_v2.py instance="HANOI-100-1(1).vrpspd" ^
      seeds=0:20 ks=7,8 base_t=3 t=8 noimp=3 ^
      min_route_len=5 min_route_delivery_frac=0.20 ^
      sub_min_len=10 sub_min_work_frac=0.50 ^
      deltas=0.05,0.10,0.20,0.40 pair_tol=0.02 out=hanoi_recoverability_v2

Quick smoke:
  python probe_freight_recoverability_v2.py instance="HANOI-100-1(1).vrpspd" ^
      seeds=0:3 ks=7,8 base_t=0.5 t=1 noimp=0.5 deltas=0.10
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



def route_delivery_sum(route: Sequence[int], delivery: np.ndarray) -> float:
    if not route:
        return 0.0
    return float(delivery[np.asarray(route, dtype=int)].sum())


def route_meets_floor(
    route: Sequence[int],
    delivery: np.ndarray,
    min_route_len: int,
    min_route_delivery: float,
) -> bool:
    return (
        len(route) >= min_route_len
        and route_delivery_sum(route, delivery) >= min_route_delivery - EPS
    )


def validate_fixed_k_solution(
    solution: Sequence[Sequence[int]],
    target_k: int,
    n: int,
    gate: DetGate,
    delivery: np.ndarray,
    min_route_len: int,
    min_route_delivery: float,
) -> None:
    validate_solution(solution, n, gate)
    if len(solution) != target_k:
        raise AssertionError(f"Expected K={target_k}, got K={len(solution)}")
    bad = [
        i
        for i, r in enumerate(solution)
        if not route_meets_floor(r, delivery, min_route_len, min_route_delivery)
    ]
    if bad:
        raise AssertionError(f"Route-floor violation at indices {bad}")


def randomized_giant_tour(
    base_solution: Sequence[Sequence[int]],
    rng: random.Random,
) -> List[int]:
    """Randomize route order/orientation, then concatenate into one giant tour."""
    routes = [list(r) for r in base_solution if r]
    rng.shuffle(routes)
    for i in range(len(routes)):
        if rng.random() < 0.5:
            routes[i].reverse()
        # A route is a depot-to-depot cycle. A random cyclic shift creates a
        # different giant-tour cut structure without changing customer coverage.
        if len(routes[i]) >= 4 and rng.random() < 0.5:
            shift = rng.randrange(len(routes[i]))
            routes[i] = routes[i][shift:] + routes[i][:shift]
    return [c for r in routes for c in r]


def partition_giant_tour_fixed_k(
    tour: Sequence[int],
    target_k: int,
    D: np.ndarray,
    gate: DetGate,
    delivery: np.ndarray,
    min_route_len: int,
    min_route_delivery: float,
    rng: random.Random,
    jitter_frac: float,
) -> Optional[List[List[int]]]:
    """Exact K-segment partition of a giant tour under feasibility/floor constraints."""
    nodes = list(tour)
    m = len(nodes)
    if target_k * min_route_len > m:
        return None

    # Segment scores are route distance plus a tiny seed-specific jitter used
    # only to diversify equal/near-equal partitions. Reported plan cost remains
    # the true, unjittered distance.
    seg_score: Dict[Tuple[int, int], float] = {}
    for i in range(m):
        for j in range(i + min_route_len, m + 1):
            route = nodes[i:j]
            if route_delivery_sum(route, delivery) < min_route_delivery - EPS:
                continue
            if not gate.feasible(route):
                continue
            base = route_cost(route, D)
            jitter = 1.0 + jitter_frac * rng.uniform(-1.0, 1.0)
            seg_score[(i, j)] = base * jitter

    inf = math.inf
    dp = [[inf] * (m + 1) for _ in range(target_k + 1)]
    parent = [[-1] * (m + 1) for _ in range(target_k + 1)]
    dp[0][0] = 0.0

    for k in range(1, target_k + 1):
        j_min = k * min_route_len
        j_max = m - (target_k - k) * min_route_len
        for j in range(j_min, j_max + 1):
            i_min = (k - 1) * min_route_len
            i_max = j - min_route_len
            for i in range(i_min, i_max + 1):
                score = seg_score.get((i, j))
                if score is None or not math.isfinite(dp[k - 1][i]):
                    continue
                value = dp[k - 1][i] + score
                if value < dp[k][j] - EPS:
                    dp[k][j] = value
                    parent[k][j] = i

    if parent[target_k][m] < 0:
        return None

    routes: List[List[int]] = []
    j = m
    for k in range(target_k, 0, -1):
        i = parent[k][j]
        if i < 0:
            return None
        routes.append(nodes[i:j])
        j = i
    routes.reverse()
    return routes


def relocate_fixed_k_floors(
    solution: Sequence[Sequence[int]],
    D: np.ndarray,
    gate: DetGate,
    delivery: np.ndarray,
    min_route_len: int,
    min_route_delivery: float,
) -> List[List[int]]:
    """Improving relocate descent that preserves K and all route floors."""
    sol = [list(r) for r in solution]
    improved = True
    while improved:
        improved = False
        for ri, R in enumerate(sol):
            if len(R) <= min_route_len:
                continue
            for pi, c in enumerate(R):
                new_R = R[:pi] + R[pi + 1 :]
                if not route_meets_floor(new_R, delivery, min_route_len, min_route_delivery):
                    continue
                a = R[pi - 1] if pi > 0 else 0
                b = R[pi + 1] if pi + 1 < len(R) else 0
                removal_gain = D[a, c] + D[c, b] - D[a, b]
                for rj, S in enumerate(sol):
                    if rj == ri:
                        continue
                    for q in range(len(S) + 1):
                        u = S[q - 1] if q > 0 else 0
                        v = S[q] if q < len(S) else 0
                        delta = D[u, c] + D[c, v] - D[u, v] - removal_gain
                        if delta >= -EPS:
                            continue
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


def swap_fixed_k_floors(
    solution: Sequence[Sequence[int]],
    D: np.ndarray,
    gate: DetGate,
    delivery: np.ndarray,
    min_route_len: int,
    min_route_delivery: float,
) -> List[List[int]]:
    """Improving one-customer swaps; useful when minimum route size blocks relocation."""
    sol = [list(r) for r in solution]
    improved = True
    while improved:
        improved = False
        for ri in range(len(sol)):
            for rj in range(ri + 1, len(sol)):
                A, B = sol[ri], sol[rj]
                old_cost = route_cost(A, D) + route_cost(B, D)
                for ia, ca in enumerate(A):
                    for ib, cb in enumerate(B):
                        new_A = A[:]
                        new_B = B[:]
                        new_A[ia], new_B[ib] = cb, ca
                        if not route_meets_floor(new_A, delivery, min_route_len, min_route_delivery):
                            continue
                        if not route_meets_floor(new_B, delivery, min_route_len, min_route_delivery):
                            continue
                        if not gate.feasible(new_A) or not gate.feasible(new_B):
                            continue
                        new_cost = route_cost(new_A, D) + route_cost(new_B, D)
                        if new_cost < old_cost - EPS:
                            sol[ri], sol[rj] = new_A, new_B
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return sol


def fixed_k_local_search_floors(
    solution: Sequence[Sequence[int]],
    D: np.ndarray,
    gate: DetGate,
    delivery: np.ndarray,
    min_route_len: int,
    min_route_delivery: float,
) -> List[List[int]]:
    sol = [two_opt_gate(list(r), D, gate) for r in solution]
    sol = relocate_fixed_k_floors(
        sol, D, gate, delivery, min_route_len, min_route_delivery
    )
    sol = swap_fixed_k_floors(
        sol, D, gate, delivery, min_route_len, min_route_delivery
    )
    sol = relocate_fixed_k_floors(
        sol, D, gate, delivery, min_route_len, min_route_delivery
    )
    return [two_opt_gate(list(r), D, gate) for r in sol]


def perturb_fixed_k_floors(
    solution: Sequence[Sequence[int]],
    D: np.ndarray,
    gate: DetGate,
    delivery: np.ndarray,
    min_route_len: int,
    min_route_delivery: float,
    rng: random.Random,
    moves: int,
) -> List[List[int]]:
    """Random feasible relocations, always preserving exact K and route floors."""
    sol = [list(r) for r in solution]
    for _ in range(moves):
        sources = [i for i, r in enumerate(sol) if len(r) > min_route_len]
        rng.shuffle(sources)
        moved = False
        for ri in sources:
            R = sol[ri]
            customer_positions = list(range(len(R)))
            rng.shuffle(customer_positions)
            for pi in customer_positions:
                c = R[pi]
                new_R = R[:pi] + R[pi + 1 :]
                if not route_meets_floor(new_R, delivery, min_route_len, min_route_delivery):
                    continue
                targets = [j for j in range(len(sol)) if j != ri]
                rng.shuffle(targets)
                candidates: List[Tuple[float, int, int, List[int]]] = []
                for rj in targets:
                    S = sol[rj]
                    for pos in range(len(S) + 1):
                        new_S = S[:pos] + [c] + S[pos:]
                        if not gate.feasible(new_S):
                            continue
                        delta = (
                            route_cost(new_R, D)
                            + route_cost(new_S, D)
                            - route_cost(R, D)
                            - route_cost(S, D)
                        )
                        candidates.append((float(delta), rj, pos, new_S))
                if not candidates:
                    continue
                candidates.sort(key=lambda x: x[0])
                # Pick from several best feasible moves rather than always the best,
                # which provides a controlled ILS kick.
                chosen = rng.choice(candidates[: min(6, len(candidates))])
                _, rj, _, new_S = chosen
                sol[ri] = new_R
                sol[rj] = new_S
                moved = True
                break
            if moved:
                break
    return sol


def solve_direct_fixed_k_plan(
    base_solution: Sequence[Sequence[int]],
    target_k: int,
    D: np.ndarray,
    gate: DetGate,
    delivery: np.ndarray,
    n: int,
    min_route_len: int,
    min_route_delivery: float,
    time_limit: float,
    no_improve: float,
    seed: int,
    init_tries: int,
    partition_jitter: float,
) -> List[List[int]]:
    """Construct and improve a balanced fixed-K plan without split variants."""
    rng = random.Random(seed)
    best: Optional[List[List[int]]] = None
    best_cost = math.inf

    for _ in range(init_tries):
        tour = randomized_giant_tour(base_solution, rng)
        initial = partition_giant_tour_fixed_k(
            tour,
            target_k,
            D,
            gate,
            delivery,
            min_route_len,
            min_route_delivery,
            rng,
            partition_jitter,
        )
        if initial is None:
            continue
        initial = fixed_k_local_search_floors(
            initial, D, gate, delivery, min_route_len, min_route_delivery
        )
        value = solution_distance(initial, D)
        if value < best_cost - EPS:
            best = [r[:] for r in initial]
            best_cost = value

    if best is None:
        raise RuntimeError(
            f"Could not construct K={target_k} plan with min_len={min_route_len}, "
            f"min_delivery={min_route_delivery:g}; relax floors or increase init_tries"
        )

    start = time.time()
    last_improve = start
    customer_count = sum(len(r) for r in best)
    while time.time() - start < time_limit:
        q_frac = rng.choice([0.05, 0.08, 0.10, 0.15])
        candidate = perturb_fixed_k_floors(
            best,
            D,
            gate,
            delivery,
            min_route_len,
            min_route_delivery,
            rng,
            moves=max(2, int(q_frac * customer_count)),
        )
        candidate = fixed_k_local_search_floors(
            candidate, D, gate, delivery, min_route_len, min_route_delivery
        )
        value = solution_distance(candidate, D)
        if value < best_cost - EPS:
            best = [r[:] for r in candidate]
            best_cost = value
            last_improve = time.time()
        elif time.time() - last_improve > no_improve:
            break

    validate_fixed_k_solution(
        best,
        target_k,
        n,
        gate,
        delivery,
        min_route_len,
        min_route_delivery,
    )
    return best

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



def max_absorbed_workload_dp(
    lost_route: Sequence[int],
    options: Dict[int, List[InsertionOption]],
    customer_weights: np.ndarray,
) -> float:
    """Exact maximum absorbed workload, independent of absorbed customer count."""
    m = len(lost_route)
    states: List[Dict[int, float]] = [dict() for _ in range(m + 1)]
    states[0][0] = 0.0

    def update(pos: int, mask: int, value: float) -> None:
        old = states[pos].get(mask)
        if old is None or value > old + EPS:
            states[pos][mask] = value

    for pos in range(m):
        if not states[pos]:
            continue
        for mask, work in list(states[pos].items()):
            update(pos + 1, mask, work)
            for opt in options.get(pos, []):
                bit = 1 << opt.survivor
                if mask & bit:
                    continue
                block_nodes = lost_route[opt.start : opt.end]
                add_work = float(customer_weights[np.asarray(block_nodes, dtype=int)].sum())
                update(opt.end, mask | bit, work + add_work)

    return float(max(states[m].values()))

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
    max_count, _count_tiebreak_work = max_absorption_dp(
        lost, len(survivors), options, weights
    )
    max_work = max_absorbed_workload_dp(lost, options, weights)

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


def _finite_gap(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan
    return abs(a - b)


def _route_profile(solution: Sequence[Sequence[int]], delivery: np.ndarray) -> Dict[str, object]:
    lengths = sorted(len(r) for r in solution)
    loads = sorted(route_delivery_sum(r, delivery) for r in solution)
    return {
        "min_route_len": min(lengths),
        "max_route_len": max(lengths),
        "mean_route_len": float(np.mean(lengths)),
        "std_route_len": float(np.std(lengths)),
        "route_len_signature": " ".join(map(str, lengths)),
        "min_route_delivery": min(loads),
        "max_route_delivery": max(loads),
        "mean_route_delivery": float(np.mean(loads)),
        "std_route_delivery": float(np.std(loads)),
        "route_delivery_signature": " ".join(f"{x:.3f}" for x in loads),
    }


def _profile_gap(
    A: Sequence[Sequence[int]],
    B: Sequence[Sequence[int]],
    delivery: np.ndarray,
) -> Tuple[float, float, int]:
    la = np.asarray(sorted(len(r) for r in A), dtype=float)
    lb = np.asarray(sorted(len(r) for r in B), dtype=float)
    da = np.asarray(sorted(route_delivery_sum(r, delivery) for r in A), dtype=float)
    db = np.asarray(sorted(route_delivery_sum(r, delivery) for r in B), dtype=float)
    len_gap = float(np.abs(la - lb).sum() / max(1.0, la.sum()))
    delivery_gap = float(np.abs(da - db).sum() / max(EPS, da.sum()))
    exact_len_match = int(np.array_equal(la, lb))
    return len_gap, delivery_gap, exact_len_match


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    instance = choose_instance(args.get("instance"))
    seeds = parse_seed_spec(args.get("seeds", "0:12"))
    target_ks = [int(x) for x in args.get("ks", "7,8").split(",") if x.strip()]
    base_t = float(args.get("base_t", "3"))
    time_limit = float(args.get("t", "8"))
    no_improve = float(args.get("noimp", "3"))
    deltas = [float(x) for x in args.get("deltas", "0.05,0.10,0.20,0.40").split(",")]
    pair_tol = float(args.get("pair_tol", "0.02"))
    omega_mult = float(args.get("omega_mult", "1.0"))
    out_prefix = args.get("out", "recoverability_probe_v2")
    max_unique = int(args.get("max_plans", "999999"))
    min_route_len = int(args.get("min_route_len", "5"))
    min_route_delivery_frac = float(args.get("min_route_delivery_frac", "0.20"))
    sub_min_len = int(args.get("sub_min_len", "10"))
    sub_min_work_frac = float(args.get("sub_min_work_frac", "0.50"))
    init_tries = int(args.get("init_tries", "30"))
    partition_jitter = float(args.get("partition_jitter", "0.02"))

    D, delivery, pickup, Q, n = parse_vrpspd(instance)
    gate = DetGate(Q, delivery, pickup)
    positive_edges = D[D > 0]
    omega_vehicle = omega_mult * float(positive_edges.mean())
    min_route_delivery = min_route_delivery_frac * Q
    sub_min_work = sub_min_work_frac * Q

    print("=" * 112)
    print(" BALANCED DIRECT-FIXED-K FREIGHT RECOVERABILITY PROBE v2")
    print("=" * 112)
    print(f" instance={instance}  n={n-1} customers  Q={Q:g}")
    print(f" seeds={seeds}  target_K={target_ks}  base_t={base_t:g}s  fixedK_t={time_limit:g}s  noimp={no_improve:g}s")
    print(
        f" route floors: len>={min_route_len}, delivery>={min_route_delivery:g} "
        f"({100*min_route_delivery_frac:.1f}%Q); init_tries={init_tries}, jitter={partition_jitter:g}"
    )
    print(
        f" substantive outages: len>={sub_min_len} OR workload>={sub_min_work:g} "
        f"({100*sub_min_work_frac:.1f}%Q)"
    )
    print(f" delta frontier={deltas}  pair_tol={100*pair_tol:.2f}%")
    print(" generation: randomized giant tour + exact K-partition DP + floor-preserving fixed-K ILS; NO split variants")
    print(" repair: pre-departure route loss; fixed orders; contiguous blocks; <=1 block/survivor; exact DP")
    print("-" * 112)

    plans: List[Dict[str, object]] = []
    signatures = set()
    for ordinal, seed in enumerate(seeds, 1):
        base_start = time.time()
        base_solution = solve_plan(D, gate, n, omega_vehicle, base_t, min(no_improve, base_t), seed)
        validate_solution(base_solution, n, gate)
        base_elapsed = time.time() - base_start

        for target_k in target_ks:
            start = time.time()
            try:
                solution = solve_direct_fixed_k_plan(
                    base_solution=base_solution,
                    target_k=target_k,
                    D=D,
                    gate=gate,
                    delivery=delivery,
                    n=n,
                    min_route_len=min_route_len,
                    min_route_delivery=min_route_delivery,
                    time_limit=time_limit,
                    no_improve=no_improve,
                    seed=1000003 * seed + 1009 * target_k,
                    init_tries=init_tries,
                    partition_jitter=partition_jitter,
                )
            except RuntimeError as exc:
                print(f"[seed {ordinal:>2}/{len(seeds)}] seed={seed:<4} K={target_k:<3} CONSTRUCTION FAIL: {exc}")
                continue
            elapsed = base_elapsed + (time.time() - start)
            sig = plan_signature(solution)
            dist = solution_distance(solution, D)
            duplicate = sig in signatures
            profile = _route_profile(solution, delivery)
            print(
                f"[seed {ordinal:>2}/{len(seeds)}] seed={seed:<4} K={target_k:<3} "
                f"dist={dist:>12.1f} minLen={profile['min_route_len']:<2} "
                f"minDel={float(profile['min_route_delivery']):>6.1f} time={elapsed:>6.2f}s "
                f"{'DUPLICATE' if duplicate else 'NEW'}"
            )
            if duplicate:
                continue
            signatures.add(sig)
            plans.append(
                {
                    "plan_id": len(plans),
                    "seed": seed,
                    "generation": "direct_fixed_k",
                    "solution": solution,
                    "K": target_k,
                    "distance": dist,
                    "solve_time": elapsed,
                    "signature": sig,
                    **profile,
                }
            )
            if len(plans) >= max_unique:
                break
        if len(plans) >= max_unique:
            break

    if len(plans) < 2:
        print("ERROR: fewer than two unique balanced plans; increase seeds/init_tries/time or relax floors.")
        return 2

    print("-" * 112)
    print(f" unique plans={len(plans)}; evaluating all route outages across {len(deltas)} delta values")

    outage_rows: List[Dict[str, object]] = []
    plan_rows: List[Dict[str, object]] = []
    plan_metric: Dict[Tuple[int, float], Dict[str, float]] = {}

    for plan in plans:
        pid = int(plan["plan_id"])
        solution = plan["solution"]  # type: ignore[assignment]
        assert isinstance(solution, list)

        base_row: Dict[str, object] = {
            key: value
            for key, value in plan.items()
            if key not in {"solution", "signature"}
        }

        for delta in deltas:
            full = 0
            total_abs_frac = 0.0
            total_work_frac = 0.0

            sub_len_n = sub_len_full = 0
            sub_len_abs = sub_len_work = 0.0
            sub_work_n = sub_work_full = 0
            sub_work_abs = sub_work_work = 0.0
            sub_either_n = sub_either_full = 0
            sub_either_abs = sub_either_work = 0.0

            print(f"  plan={pid:>2} seed={plan['seed']:<4} K={plan['K']} delta={delta:>6.3f}", end="", flush=True)
            tick = time.time()
            for lost_idx in range(len(solution)):
                result = evaluate_route_outage(
                    solution, lost_idx, D, gate, delivery, pickup, delta
                )
                full += int(result.fully_absorbable)
                abs_frac = result.max_absorbed_customers / max(1, result.lost_customers)
                work_frac = result.max_absorbed_workload / max(EPS, result.lost_workload)
                total_abs_frac += abs_frac
                total_work_frac += work_frac

                is_sub_len = result.lost_customers >= sub_min_len
                is_sub_work = result.lost_workload >= sub_min_work - EPS
                is_sub_either = is_sub_len or is_sub_work
                if is_sub_len:
                    sub_len_n += 1
                    sub_len_full += int(result.fully_absorbable)
                    sub_len_abs += abs_frac
                    sub_len_work += work_frac
                if is_sub_work:
                    sub_work_n += 1
                    sub_work_full += int(result.fully_absorbable)
                    sub_work_abs += abs_frac
                    sub_work_work += work_frac
                if is_sub_either:
                    sub_either_n += 1
                    sub_either_full += int(result.fully_absorbable)
                    sub_either_abs += abs_frac
                    sub_either_work += work_frac

                outage_rows.append(
                    {
                        "plan_id": pid,
                        "seed": int(plan["seed"]),
                        "generation": "direct_fixed_k",
                        "K": int(plan["K"]),
                        "distance": float(plan["distance"]),
                        "delta_frac": delta,
                        "lost_route_idx": lost_idx,
                        "lost_route_len": result.lost_customers,
                        "lost_route_delivery": route_delivery_sum(solution[lost_idx], delivery),
                        "lost_route_cost": route_cost(solution[lost_idx], D),
                        "lost_workload": result.lost_workload,
                        "substantive_by_len": int(is_sub_len),
                        "substantive_by_work": int(is_sub_work),
                        "substantive_either": int(is_sub_either),
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

            def rate(num: float, den: int) -> float:
                return num / den if den else math.nan

            metrics = {
                "recovery_rate": recovery_rate,
                "mean_absorbed_customer_frac": mean_abs,
                "mean_absorbed_workload_frac": mean_work,
                "sub_len_n": float(sub_len_n),
                "recovery_rate_sub_len": rate(sub_len_full, sub_len_n),
                "mean_absorbed_customer_frac_sub_len": rate(sub_len_abs, sub_len_n),
                "mean_absorbed_workload_frac_sub_len": rate(sub_len_work, sub_len_n),
                "sub_work_n": float(sub_work_n),
                "recovery_rate_sub_work": rate(sub_work_full, sub_work_n),
                "mean_absorbed_customer_frac_sub_work": rate(sub_work_abs, sub_work_n),
                "mean_absorbed_workload_frac_sub_work": rate(sub_work_work, sub_work_n),
                "sub_either_n": float(sub_either_n),
                "recovery_rate_sub_either": rate(sub_either_full, sub_either_n),
                "mean_absorbed_customer_frac_sub_either": rate(sub_either_abs, sub_either_n),
                "mean_absorbed_workload_frac_sub_either": rate(sub_either_work, sub_either_n),
            }
            elapsed = time.time() - tick
            print(
                f"  P0={recovery_rate:>6.3f}  P0[sub]={metrics['recovery_rate_sub_either']:>6.3f} "
                f"absWork[sub]={metrics['mean_absorbed_workload_frac_sub_either']:>6.3f}  t={elapsed:>6.2f}s"
            )

            for key, value in metrics.items():
                base_row[f"{key}_d{delta:g}"] = value
            plan_metric[(pid, delta)] = metrics

        plan_rows.append(base_row)

    pair_rows: List[Dict[str, object]] = []
    print("-" * 112)
    print(" NEAR-EQUAL NORMAL-DAY PLAN PAIRS")
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            A, B = plans[i], plans[j]
            if int(A["K"]) != int(B["K"]):
                continue
            ca, cb = float(A["distance"]), float(B["distance"])
            rel_gap = abs(ca - cb) / max(EPS, min(ca, cb))
            if rel_gap > pair_tol + EPS:
                continue
            len_profile_gap, delivery_profile_gap, exact_len_match = _profile_gap(
                A["solution"], B["solution"], delivery  # type: ignore[arg-type]
            )
            for delta in deltas:
                ma = plan_metric[(int(A["plan_id"]), delta)]
                mb = plan_metric[(int(B["plan_id"]), delta)]
                row: Dict[str, object] = {
                    "plan_a": int(A["plan_id"]),
                    "seed_a": int(A["seed"]),
                    "plan_b": int(B["plan_id"]),
                    "seed_b": int(B["seed"]),
                    "K": int(A["K"]),
                    "distance_a": ca,
                    "distance_b": cb,
                    "relative_cost_gap": rel_gap,
                    "route_len_profile_gap": len_profile_gap,
                    "route_delivery_profile_gap": delivery_profile_gap,
                    "exact_route_len_multiset_match": exact_len_match,
                    "delta_frac": delta,
                }
                for metric_name in [
                    "recovery_rate",
                    "mean_absorbed_customer_frac",
                    "mean_absorbed_workload_frac",
                    "recovery_rate_sub_len",
                    "mean_absorbed_customer_frac_sub_len",
                    "mean_absorbed_workload_frac_sub_len",
                    "recovery_rate_sub_work",
                    "mean_absorbed_customer_frac_sub_work",
                    "mean_absorbed_workload_frac_sub_work",
                    "recovery_rate_sub_either",
                    "mean_absorbed_customer_frac_sub_either",
                    "mean_absorbed_workload_frac_sub_either",
                ]:
                    a_val = float(ma[metric_name])
                    b_val = float(mb[metric_name])
                    row[f"{metric_name}_a"] = a_val
                    row[f"{metric_name}_b"] = b_val
                    row[f"abs_{metric_name}_gap"] = _finite_gap(a_val, b_val)
                pair_rows.append(row)

    if pair_rows:
        for delta in deltas:
            candidates = [r for r in pair_rows if float(r["delta_frac"]) == delta]
            best_all = max(
                candidates,
                key=lambda r: (
                    float(r["abs_recovery_rate_gap"]),
                    float(r["abs_mean_absorbed_workload_frac_gap"]),
                ),
            )
            sub_candidates = [
                r for r in candidates
                if math.isfinite(float(r["abs_recovery_rate_sub_either_gap"]))
            ]
            best_sub = max(
                sub_candidates,
                key=lambda r: (
                    float(r["abs_recovery_rate_sub_either_gap"]),
                    float(r["abs_mean_absorbed_workload_frac_sub_either_gap"]),
                ),
            ) if sub_candidates else None
            msg = (
                f" delta={delta:>6.3f}: ALL {best_all['plan_a']} vs {best_all['plan_b']} "
                f"costGap={100*float(best_all['relative_cost_gap']):>5.2f}% "
                f"recGap={float(best_all['abs_recovery_rate_gap']):>6.3f}"
            )
            if best_sub is not None:
                msg += (
                    f" | SUB {best_sub['plan_a']} vs {best_sub['plan_b']} "
                    f"costGap={100*float(best_sub['relative_cost_gap']):>5.2f}% "
                    f"recGap={float(best_sub['abs_recovery_rate_sub_either_gap']):>6.3f} "
                    f"workGap={float(best_sub['abs_mean_absorbed_workload_frac_sub_either_gap']):>6.3f}"
                )
            print(msg)
    else:
        print(f" no equal-K pairs within {100*pair_tol:.2f}% distance; probe is INCONCLUSIVE")

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
                "plan_generation": {
                    "mode": "direct_fixed_k_giant_tour_partition_ils",
                    "target_K": target_ks,
                    "min_route_len": min_route_len,
                    "min_route_delivery": min_route_delivery,
                    "init_tries": init_tries,
                    "partition_jitter": partition_jitter,
                },
                "repair_class": {
                    "outage_time": "pre-departure",
                    "survivor_order_fixed": True,
                    "lost_route_order_fixed": True,
                    "lost_route_contiguous_blocks": True,
                    "max_blocks_per_survivor": 1,
                    "per_survivor_detour_fractions": deltas,
                    "substantive_min_route_len": sub_min_len,
                    "substantive_min_workload": sub_min_work,
                },
                "plans": [
                    {
                        "plan_id": int(p["plan_id"]),
                        "seed": int(p["seed"]),
                        "K": int(p["K"]),
                        "distance": float(p["distance"]),
                        "route_len_signature": p["route_len_signature"],
                        "route_delivery_signature": p["route_delivery_signature"],
                        "routes": p["solution"],
                    }
                    for p in plans
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("-" * 112)
    print(f" wrote {plans_csv}")
    print(f" wrote {outages_csv}")
    print(f" wrote {pairs_csv}")
    print(f" wrote {routes_json}")
    print("\nInterpretation gate:")
    print("  * Overall recovery gaps are secondary; substantive-route gaps are the deciding result.")
    print("  * If substantive recovery is always zero, limited repair only rescues pseudo-spare/tiny routes: NO-GO.")
    print("  * If near-equal-cost balanced plans retain stable substantive gaps across delta, plan structure matters.")
    print("  * Maximum absorbed workload is now optimized directly, not used only as a count tie-break.")
    print("  * No outage probability or empirical-Hanoi reliability claim is made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
