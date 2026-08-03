"""
Failure-first probe v3: do near-equal-cost, directly generated fixed-K VRPSPD
plans require different emergency fleet sizes after TWO simultaneous
pre-departure vehicle outages?

This script is self-contained and reuses only the plan JSON produced by the v2
probe. It does not regenerate plans, so the expensive fixed-K search is not
repeated.

Repair semantics (exact within this class):
* two planned routes are lost before departure;
* survivor orders and both lost-route orders remain fixed;
* each lost route may donate contiguous blocks to surviving routes;
* each survivor accepts at most one donated block total, from either lost route;
* each survivor has a detour cap delta times its original route distance;
* customers not donated are assigned to emergency vehicles;
* 0 emergency vehicles are needed if every customer is donated;
* 1 emergency vehicle is enough iff the residual aggregate delivery <= Q and
  residual aggregate pickup <= Q; otherwise 2 are needed.

For standard simultaneous pickup-delivery without route-duration/time-window
constraints, the last test is exact: any residual set with total delivery and
total pickup each <= Q admits a capacity-feasible order (serve negative net-load
changes before positive net-load changes).

Typical command (Windows CMD / PowerShell):
  python probe_freight_recoverability_v3_two_outages.py ^
      instance=HANOI-100-1.vrpspd ^
      routes=hanoi_recoverability_v2_routes.json ^
      deltas=0.10,0.20 pair_tol=0.02 ^
      out=hanoi_recoverability_v3_two_outages

Quick smoke:
  python probe_freight_recoverability_v3_two_outages.py ^
      instance=HANOI-100-1.vrpspd routes=hanoi_recoverability_v2_routes.json ^
      deltas=0.10 max_plans=4 out=hanoi_v3_smoke
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9


# ============================================================================
# Parsing and deterministic VRPSPD primitives
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
        raise ValueError(f"EDGE_WEIGHT_SECTION has {len(vals)} values; expected {n*n}")
    D = np.asarray(vals, dtype=float).reshape(n, n)

    delivery = np.zeros(n, dtype=float)
    pickup = np.zeros(n, dtype=float)
    for row in _lines_after(text, "PICKUP_AND_DELIVERY_SECTION"):
        t = row.split()
        if len(t) < 3:
            continue
        idx = int(float(t[0])) - 1
        if 0 <= idx < n:
            delivery[idx] = float(t[-2])
            pickup[idx] = float(t[-1])
    return D, delivery, pickup, float(cap), int(n)


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


def validate_solution(solution: Sequence[Sequence[int]], n: int, gate: DetGate) -> None:
    flat = [c for r in solution for c in r]
    expected = list(range(1, n))
    if sorted(flat) != expected:
        missing = sorted(set(expected) - set(flat))
        counts: Dict[int, int] = {}
        for c in flat:
            counts[c] = counts.get(c, 0) + 1
        dup = sorted(c for c, v in counts.items() if v > 1)
        raise AssertionError(f"Invalid coverage: missing={missing[:10]}, duplicates={dup[:10]}")
    bad = [i for i, r in enumerate(solution) if not gate.feasible(r)]
    if bad:
        raise AssertionError(f"Infeasible routes at indices {bad}")


# ============================================================================
# Arguments and I/O
# ============================================================================

def parse_args(argv: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            out[k.strip().lower()] = v.strip().strip('"').strip("'")
        else:
            out[arg.strip().lower()] = "1"
    return out


def parse_float_list(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one delta")
    return sorted(set(vals))


def choose_existing(path_text: str, label: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    raise FileNotFoundError(f"{label} not found: {path} (cwd={Path.cwd()})")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# Block insertion precomputation
# ============================================================================

@dataclass(frozen=True)
class InsertionOption:
    target_route: int          # original route index in the plan
    start: int
    end: int                   # exclusive
    position: int
    delta: float
    add_delivery_i: int
    add_pickup_i: int
    add_work: float


def best_block_insertion(
    base_route: Sequence[int],
    block: Sequence[int],
    D: np.ndarray,
    gate: DetGate,
    delta_frac: float,
) -> Optional[Tuple[int, float]]:
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
            best_pos = pos
            best_delta = float(delta)
    if best_pos is None:
        return None
    return best_pos, best_delta


def precompute_options_for_lost_route(
    solution: Sequence[Sequence[int]],
    lost_idx: int,
    D: np.ndarray,
    gate: DetGate,
    delivery_i: np.ndarray,
    pickup_i: np.ndarray,
    delivery: np.ndarray,
    pickup: np.ndarray,
    delta_frac: float,
) -> Dict[int, List[InsertionOption]]:
    """Build once, then filter the second lost route out for every outage pair."""
    lost = list(solution[lost_idx])
    out: Dict[int, List[InsertionOption]] = {i: [] for i in range(len(lost))}

    # Prefix sums make block demands O(1).
    idx = np.asarray(lost, dtype=int)
    d_pref = np.concatenate(([0], np.cumsum(delivery_i[idx], dtype=np.int64)))
    p_pref = np.concatenate(([0], np.cumsum(pickup_i[idx], dtype=np.int64)))
    w_pref = np.concatenate(([0.0], np.cumsum((delivery + pickup)[idx], dtype=float)))

    for start in range(len(lost)):
        for end in range(start + 1, len(lost) + 1):
            block = lost[start:end]
            add_d = int(d_pref[end] - d_pref[start])
            add_p = int(p_pref[end] - p_pref[start])
            add_w = float(w_pref[end] - w_pref[start])
            for target_idx, base in enumerate(solution):
                if target_idx == lost_idx:
                    continue
                ans = best_block_insertion(base, block, D, gate, delta_frac)
                if ans is None:
                    continue
                pos, add_dist = ans
                out[start].append(
                    InsertionOption(
                        target_route=target_idx,
                        start=start,
                        end=end,
                        position=pos,
                        delta=add_dist,
                        add_delivery_i=add_d,
                        add_pickup_i=add_p,
                        add_work=add_w,
                    )
                )
    return out


def filter_pair_options(
    options: Dict[int, List[InsertionOption]],
    lost_a: int,
    lost_b: int,
) -> Dict[int, List[InsertionOption]]:
    return {
        start: [o for o in vals if o.target_route not in (lost_a, lost_b)]
        for start, vals in options.items()
    }


# ============================================================================
# Exact two-route repair DPs
# ============================================================================


def _filter_options_for_pair(
    options: Dict[int, List[InsertionOption]],
    lost_a: int,
    lost_b: int,
) -> Dict[int, List[InsertionOption]]:
    return {
        start: [o for o in vals if o.target_route not in (lost_a, lost_b)]
        for start, vals in options.items()
    }


def _full_absorption_process(
    nodes: Sequence[int],
    options: Dict[int, List[InsertionOption]],
    start_masks: Iterable[int],
) -> set[int]:
    """Reachable survivor masks when every customer in this sequence is donated."""
    stages: List[set[int]] = [set() for _ in range(len(nodes) + 1)]
    stages[0].update(start_masks)
    for pos in range(len(nodes)):
        for mask in stages[pos]:
            for opt in options.get(pos, []):
                bit = 1 << opt.target_route
                if mask & bit:
                    continue
                stages[opt.end].add(mask | bit)
    return stages[len(nodes)]


def full_absorption_possible(
    route_a: Sequence[int],
    options_a: Dict[int, List[InsertionOption]],
    route_b: Sequence[int],
    options_b: Dict[int, List[InsertionOption]],
) -> bool:
    masks_after_a = _full_absorption_process(route_a, options_a, [0])
    if not masks_after_a:
        return False
    return bool(_full_absorption_process(route_b, options_b, masks_after_a))


def _pareto_pairs(pairs: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Keep only nondominated (absorbed delivery, absorbed pickup) pairs."""
    uniq = sorted(set(pairs), key=lambda x: (x[0], x[1]), reverse=True)
    out: List[Tuple[int, int]] = []
    best_p = -1
    for d, p in uniq:
        if p <= best_p:
            continue
        out.append((d, p))
        best_p = p
    return out


def _threshold_process(
    nodes: Sequence[int],
    options: Dict[int, List[InsertionOption]],
    start_states: Dict[int, List[Tuple[int, int]]],
    req_d: int,
    req_p: int,
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Exact sparse DP for whether enough delivery and pickup can be donated so
    that the residual fits on one emergency vehicle. Work/count are deliberately
    excluded; keeping them in the state causes unnecessary frontier explosion.
    """
    stages: List[Dict[int, set[Tuple[int, int]]]] = [dict() for _ in range(len(nodes) + 1)]
    for mask, pairs in start_states.items():
        stages[0][mask] = set(pairs)

    def add(stage: Dict[int, set[Tuple[int, int]]], mask: int, pair: Tuple[int, int]) -> None:
        stage.setdefault(mask, set()).add(pair)

    for pos in range(len(nodes)):
        current = stages[pos]
        if not current:
            continue
        for mask, pairs in current.items():
            for d, p in pairs:
                # Skip this customer: it remains for emergency service.
                add(stages[pos + 1], mask, (d, p))
                for opt in options.get(pos, []):
                    bit = 1 << opt.target_route
                    if mask & bit:
                        continue
                    add(
                        stages[opt.end],
                        mask | bit,
                        (
                            min(req_d, d + opt.add_delivery_i),
                            min(req_p, p + opt.add_pickup_i),
                        ),
                    )

        # Pareto-prune the next ordinary step. Other future stages are pruned
        # when they become current, so block jumps remain exact.
        nxt = stages[pos + 1]
        for mask, pairs in list(nxt.items()):
            nxt[mask] = set(_pareto_pairs(pairs))

    final: Dict[int, List[Tuple[int, int]]] = {}
    for mask, pairs in stages[len(nodes)].items():
        final[mask] = _pareto_pairs(pairs)
    return final


def one_emergency_unit_possible(
    route_a: Sequence[int],
    options_a: Dict[int, List[InsertionOption]],
    route_b: Sequence[int],
    options_b: Dict[int, List[InsertionOption]],
    req_d: int,
    req_p: int,
) -> Tuple[bool, int]:
    if req_d <= 0 and req_p <= 0:
        return True, 1
    states = _threshold_process(route_a, options_a, {0: [(0, 0)]}, req_d, req_p)
    states = _threshold_process(route_b, options_b, states, req_d, req_p)
    n_states = sum(len(v) for v in states.values())
    feasible = any(d >= req_d and p >= req_p for vals in states.values() for d, p in vals)
    return feasible, n_states


# mask -> (absorbed workload, absorbed customer count,
#          absorbed delivery integer, absorbed pickup integer)
WorkVal = Tuple[float, int, int, int]


def _work_better(a: WorkVal, b: Optional[WorkVal]) -> bool:
    if b is None:
        return True
    if a[0] > b[0] + EPS:
        return True
    if abs(a[0] - b[0]) <= EPS and a[1] > b[1]:
        return True
    return False


def _max_work_process(
    nodes: Sequence[int],
    options: Dict[int, List[InsertionOption]],
    start_states: Dict[int, WorkVal],
) -> Dict[int, WorkVal]:
    """Exact maximum donated workload for each survivor mask."""
    stages: List[Dict[int, WorkVal]] = [dict() for _ in range(len(nodes) + 1)]
    stages[0] = dict(start_states)

    def update(stage: Dict[int, WorkVal], mask: int, val: WorkVal) -> None:
        if _work_better(val, stage.get(mask)):
            stage[mask] = val

    for pos in range(len(nodes)):
        for mask, val in stages[pos].items():
            update(stages[pos + 1], mask, val)  # leave customer to emergency service
            work, count, act_d, act_p = val
            for opt in options.get(pos, []):
                bit = 1 << opt.target_route
                if mask & bit:
                    continue
                update(
                    stages[opt.end],
                    mask | bit,
                    (
                        work + opt.add_work,
                        count + (opt.end - opt.start),
                        act_d + opt.add_delivery_i,
                        act_p + opt.add_pickup_i,
                    ),
                )
    return stages[len(nodes)]


@dataclass
class TwoOutageResult:
    extra_units: int
    absorbed_customers: int
    absorbed_workload: float
    residual_delivery_at_max_work: float
    residual_pickup_at_max_work: float
    residual_workload: float
    total_customers: int
    total_delivery: float
    total_pickup: float
    total_workload: float
    one_unit_capacity_feasible: bool
    fully_absorbed: bool
    terminal_states: int


def evaluate_two_route_outage(
    solution: Sequence[Sequence[int]],
    lost_a: int,
    lost_b: int,
    options_a_all: Dict[int, List[InsertionOption]],
    options_b_all: Dict[int, List[InsertionOption]],
    delivery: np.ndarray,
    pickup: np.ndarray,
    delivery_i: np.ndarray,
    pickup_i: np.ndarray,
    capacity_i: int,
) -> TwoOutageResult:
    route_a = list(solution[lost_a])
    route_b = list(solution[lost_b])
    nodes = route_a + route_b
    idx = np.asarray(nodes, dtype=int)

    total_d_i = int(delivery_i[idx].sum())
    total_p_i = int(pickup_i[idx].sum())
    total_work = float((delivery + pickup)[idx].sum())
    total_n = len(nodes)
    req_d = max(0, total_d_i - capacity_i)
    req_p = max(0, total_p_i - capacity_i)

    options_a = _filter_options_for_pair(options_a_all, lost_a, lost_b)
    options_b = _filter_options_for_pair(options_b_all, lost_a, lost_b)

    fully = full_absorption_possible(route_a, options_a, route_b, options_b)
    if fully:
        one_possible, terminal_states = True, 0
        extra = 0
    else:
        one_possible, terminal_states = one_emergency_unit_possible(
            route_a, options_a, route_b, options_b, req_d, req_p
        )
        extra = 1 if one_possible else 2

    work_states = _max_work_process(route_a, options_a, {0: (0.0, 0, 0, 0)})
    work_states = _max_work_process(route_b, options_b, work_states)
    best = max(work_states.values(), key=lambda v: (v[0], v[1]))
    absorbed_work, absorbed_count, absorbed_d_i, absorbed_p_i = best

    return TwoOutageResult(
        extra_units=extra,
        absorbed_customers=int(absorbed_count),
        absorbed_workload=float(absorbed_work),
        residual_delivery_at_max_work=float(total_d_i - absorbed_d_i) / 10.0,
        residual_pickup_at_max_work=float(total_p_i - absorbed_p_i) / 10.0,
        residual_workload=max(0.0, total_work - absorbed_work),
        total_customers=total_n,
        total_delivery=float(total_d_i) / 10.0,
        total_pickup=float(total_p_i) / 10.0,
        total_workload=total_work,
        one_unit_capacity_feasible=one_possible,
        fully_absorbed=fully,
        terminal_states=terminal_states,
    )


# ============================================================================
# Experiment orchestration
# ============================================================================

def route_is_substantive(
    route: Sequence[int],
    delivery: np.ndarray,
    pickup: np.ndarray,
    min_len: int,
    min_work: float,
) -> bool:
    idx = np.asarray(route, dtype=int)
    work = float((delivery + pickup)[idx].sum())
    return len(route) >= min_len or work >= min_work - EPS


def profile_gap(
    a: Sequence[Sequence[int]],
    b: Sequence[Sequence[int]],
    delivery: np.ndarray,
) -> Tuple[float, float, int]:
    la = np.asarray(sorted(len(r) for r in a), dtype=float)
    lb = np.asarray(sorted(len(r) for r in b), dtype=float)
    da = np.asarray(sorted(float(delivery[np.asarray(r, dtype=int)].sum()) for r in a))
    db = np.asarray(sorted(float(delivery[np.asarray(r, dtype=int)].sum()) for r in b))
    len_gap = float(np.mean(np.abs(la - lb))) if len(la) == len(lb) else math.nan
    del_gap = float(np.mean(np.abs(da - db))) if len(da) == len(db) else math.nan
    exact = int(tuple(la.tolist()) == tuple(lb.tolist()))
    return len_gap, del_gap, exact


def finite_gap(a: float, b: float) -> float:
    return abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.nan


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    instance = choose_existing(args.get("instance", "HANOI-100-1.vrpspd"), "Instance")
    routes_path = choose_existing(
        args.get("routes", "hanoi_recoverability_v2_routes.json"), "V2 routes JSON"
    )
    deltas = parse_float_list(args.get("deltas", "0.10,0.20"))
    pair_tol = float(args.get("pair_tol", "0.02"))
    out_prefix = args.get("out", "hanoi_recoverability_v3_two_outages")
    max_plans = int(args.get("max_plans", "0"))
    sub_min_len = int(args.get("sub_min_len", "10"))
    sub_min_work_frac = float(args.get("sub_min_work_frac", "0.50"))

    D, delivery, pickup, Q, n = parse_vrpspd(instance)
    gate = DetGate(Q, delivery, pickup)

    # Current benchmark uses one decimal. Keep the integer DP exact at 0.1 units.
    scale = 10
    delivery_i = np.rint(delivery * scale).astype(np.int64)
    pickup_i = np.rint(pickup * scale).astype(np.int64)
    capacity_i = int(round(Q * scale))
    if np.max(np.abs(delivery_i / scale - delivery)) > 1e-7 or np.max(np.abs(pickup_i / scale - pickup)) > 1e-7:
        raise ValueError("Demands have more than one decimal; increase the DP scaling factor")

    payload = json.loads(routes_path.read_text(encoding="utf-8"))
    raw_plans = payload.get("plans", [])
    if not raw_plans:
        raise ValueError("No plans found in routes JSON")
    if max_plans > 0:
        raw_plans = raw_plans[:max_plans]

    plans: List[Dict[str, object]] = []
    for p in raw_plans:
        solution = [[int(c) for c in r] for r in p["routes"]]
        validate_solution(solution, n, gate)
        distance = float(p.get("distance", sum(route_cost(r, D) for r in solution)))
        plans.append({
            "plan_id": int(p["plan_id"]),
            "seed": int(p.get("seed", -1)),
            "K": int(p.get("K", len(solution))),
            "distance": distance,
            "solution": solution,
        })

    sub_min_work = sub_min_work_frac * Q
    print("=" * 120)
    print(" TWO-VEHICLE PRE-DEPARTURE OUTAGE PROBE (V3)")
    print(f" instance={instance}  routes={routes_path}")
    print(f" plans={len(plans)}  deltas={deltas}  pair_tol={100*pair_tol:.2f}%")
    print(f" substantive route: len>={sub_min_len} OR workload>={sub_min_work:g}")
    print(" repair: fixed orders; contiguous donated blocks; <=1 block/survivor across both lost routes")
    print(" severity: exact emergency units in {0,1,2}; one unit iff residual D<=Q and residual P<=Q")
    print("=" * 120)

    scenario_rows: List[Dict[str, object]] = []
    plan_rows: List[Dict[str, object]] = []
    plan_metric: Dict[Tuple[int, float], Dict[str, float]] = {}

    for plan_pos, plan in enumerate(plans, start=1):
        pid = int(plan["plan_id"])
        solution = plan["solution"]  # type: ignore[assignment]
        K = int(plan["K"])
        print(f"[{plan_pos:>2}/{len(plans)}] plan={pid:>3} K={K} dist={float(plan['distance']):.0f}")
        base_row: Dict[str, object] = {
            "plan_id": pid,
            "seed": int(plan["seed"]),
            "K": K,
            "distance": float(plan["distance"]),
            "n_outage_pairs": K * (K - 1) // 2,
        }

        substantive_flags = [
            route_is_substantive(r, delivery, pickup, sub_min_len, sub_min_work)
            for r in solution
        ]

        for delta in deltas:
            tick = time.time()
            # Expensive insertion checks are done once per lost route, not once per pair.
            precomputed: Dict[int, Dict[int, List[InsertionOption]]] = {}
            for lost_idx in range(K):
                precomputed[lost_idx] = precompute_options_for_lost_route(
                    solution, lost_idx, D, gate,
                    delivery_i, pickup_i, delivery, pickup, delta,
                )

            pair_results: List[Tuple[bool, TwoOutageResult]] = []
            for lost_a, lost_b in combinations(range(K), 2):
                result = evaluate_two_route_outage(
                    solution, lost_a, lost_b,
                    precomputed[lost_a], precomputed[lost_b],
                    delivery, pickup, delivery_i, pickup_i, capacity_i,
                )
                both_sub = substantive_flags[lost_a] and substantive_flags[lost_b]
                pair_results.append((both_sub, result))

                ra = solution[lost_a]
                rb = solution[lost_b]
                idx_a = np.asarray(ra, dtype=int)
                idx_b = np.asarray(rb, dtype=int)
                work_a = float((delivery + pickup)[idx_a].sum())
                work_b = float((delivery + pickup)[idx_b].sum())
                scenario_rows.append({
                    "plan_id": pid,
                    "seed": int(plan["seed"]),
                    "K": K,
                    "distance": float(plan["distance"]),
                    "delta_frac": delta,
                    "lost_a": lost_a,
                    "lost_b": lost_b,
                    "lost_a_len": len(ra),
                    "lost_b_len": len(rb),
                    "lost_a_workload": work_a,
                    "lost_b_workload": work_b,
                    "lost_a_substantive": int(substantive_flags[lost_a]),
                    "lost_b_substantive": int(substantive_flags[lost_b]),
                    "both_substantive": int(both_sub),
                    "extra_units": result.extra_units,
                    "need_two_units": int(result.extra_units == 2),
                    "need_at_most_one": int(result.extra_units <= 1),
                    "fully_absorbed": int(result.fully_absorbed),
                    "total_customers": result.total_customers,
                    "absorbed_customers": result.absorbed_customers,
                    "absorbed_customer_frac": result.absorbed_customers / max(1, result.total_customers),
                    "total_delivery": result.total_delivery,
                    "total_pickup": result.total_pickup,
                    "total_workload": result.total_workload,
                    "absorbed_workload": result.absorbed_workload,
                    "absorbed_workload_frac": result.absorbed_workload / max(EPS, result.total_workload),
                    "residual_delivery_at_max_work": result.residual_delivery_at_max_work,
                    "residual_pickup_at_max_work": result.residual_pickup_at_max_work,
                    "residual_workload": result.residual_workload,
                    "residual_workload_frac": result.residual_workload / max(EPS, result.total_workload),
                    "one_unit_capacity_feasible": int(result.one_unit_capacity_feasible),
                    "terminal_dp_states": result.terminal_states,
                    "lost_route_a": " ".join(map(str, ra)),
                    "lost_route_b": " ".join(map(str, rb)),
                })

            def summarize(rows: Iterable[TwoOutageResult]) -> Dict[str, float]:
                rr = list(rows)
                if not rr:
                    return {
                        "n": 0.0,
                        "mean_extra_units": math.nan,
                        "p_need_two": math.nan,
                        "p_need_at_most_one": math.nan,
                        "p_zero": math.nan,
                        "mean_absorbed_workload_frac": math.nan,
                        "mean_residual_workload_frac": math.nan,
                    }
                return {
                    "n": float(len(rr)),
                    "mean_extra_units": float(np.mean([x.extra_units for x in rr])),
                    "p_need_two": float(np.mean([x.extra_units == 2 for x in rr])),
                    "p_need_at_most_one": float(np.mean([x.extra_units <= 1 for x in rr])),
                    "p_zero": float(np.mean([x.extra_units == 0 for x in rr])),
                    "mean_absorbed_workload_frac": float(np.mean([
                        x.absorbed_workload / max(EPS, x.total_workload) for x in rr
                    ])),
                    "mean_residual_workload_frac": float(np.mean([
                        x.residual_workload / max(EPS, x.total_workload) for x in rr
                    ])),
                }

            all_m = summarize(r for _, r in pair_results)
            sub_m = summarize(r for both, r in pair_results if both)
            metrics: Dict[str, float] = {}
            for k, v in all_m.items():
                metrics[k] = v
            for k, v in sub_m.items():
                metrics[f"both_sub_{k}"] = v
            plan_metric[(pid, delta)] = metrics
            for k, v in metrics.items():
                base_row[f"{k}_d{delta:g}"] = v

            elapsed = time.time() - tick
            print(
                f"  d={delta:.2f} meanR={all_m['mean_extra_units']:.3f} "
                f"P(R=2)={all_m['p_need_two']:.3f} "
                f"SUB meanR={sub_m['mean_extra_units']:.3f} "
                f"SUB P(R=2)={sub_m['p_need_two']:.3f} "
                f"resWork={sub_m['mean_residual_workload_frac']:.3f} t={elapsed:.1f}s"
            )
        plan_rows.append(base_row)

    pair_rows: List[Dict[str, object]] = []
    print("-" * 120)
    print(" NEAR-EQUAL NORMAL-DAY PLAN PAIRS")
    for a_pos in range(len(plans)):
        for b_pos in range(a_pos + 1, len(plans)):
            A, B = plans[a_pos], plans[b_pos]
            if int(A["K"]) != int(B["K"]):
                continue
            ca, cb = float(A["distance"]), float(B["distance"])
            rel_gap = abs(ca - cb) / max(EPS, min(ca, cb))
            if rel_gap > pair_tol + EPS:
                continue
            len_gap, del_gap, exact_len = profile_gap(
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
                    "route_len_profile_gap": len_gap,
                    "route_delivery_profile_gap": del_gap,
                    "exact_route_len_multiset_match": exact_len,
                    "delta_frac": delta,
                }
                for metric in [
                    "mean_extra_units", "p_need_two", "p_need_at_most_one", "p_zero",
                    "mean_absorbed_workload_frac", "mean_residual_workload_frac",
                    "both_sub_mean_extra_units", "both_sub_p_need_two",
                    "both_sub_p_need_at_most_one", "both_sub_p_zero",
                    "both_sub_mean_absorbed_workload_frac",
                    "both_sub_mean_residual_workload_frac",
                ]:
                    av = float(ma[metric])
                    bv = float(mb[metric])
                    row[f"{metric}_a"] = av
                    row[f"{metric}_b"] = bv
                    row[f"abs_{metric}_gap"] = finite_gap(av, bv)
                pair_rows.append(row)

    if pair_rows:
        for delta in deltas:
            cand = [r for r in pair_rows if float(r["delta_frac"]) == delta]
            best = max(cand, key=lambda r: (
                float(r["abs_both_sub_p_need_two_gap"]),
                float(r["abs_both_sub_mean_extra_units_gap"]),
                float(r["abs_both_sub_mean_residual_workload_frac_gap"]),
            ))
            print(
                f" d={delta:.2f}: {best['plan_a']} vs {best['plan_b']} K={best['K']} "
                f"costGap={100*float(best['relative_cost_gap']):.2f}% "
                f"SUB P2gap={float(best['abs_both_sub_p_need_two_gap']):.3f} "
                f"SUB meanRgap={float(best['abs_both_sub_mean_extra_units_gap']):.3f} "
                f"SUB residualGap={float(best['abs_both_sub_mean_residual_workload_frac_gap']):.3f}"
            )
    else:
        print(f" no equal-K plan pairs within {100*pair_tol:.2f}% cost")

    prefix = Path(out_prefix)
    plans_csv = prefix.with_name(prefix.name + "_plans.csv")
    scenarios_csv = prefix.with_name(prefix.name + "_two_outages.csv")
    pairs_csv = prefix.with_name(prefix.name + "_pairs.csv")
    meta_json = prefix.with_name(prefix.name + "_meta.json")
    write_csv(plans_csv, plan_rows)
    write_csv(scenarios_csv, scenario_rows)
    write_csv(pairs_csv, pair_rows)
    meta_json.write_text(json.dumps({
        "instance": str(instance),
        "source_routes": str(routes_path),
        "capacity": Q,
        "deltas": deltas,
        "repair_class": {
            "outage_time": "pre-departure",
            "simultaneous_lost_routes": 2,
            "survivor_order_fixed": True,
            "lost_route_orders_fixed": True,
            "lost_route_contiguous_blocks": True,
            "max_blocks_per_survivor_total": 1,
            "emergency_route_capacity_test": "residual total delivery <= Q and residual total pickup <= Q",
        },
        "substantive_route": {
            "min_len": sub_min_len,
            "min_workload": sub_min_work,
            "pair_filter": "both lost routes substantive",
        },
        "plans": [{
            "plan_id": int(p["plan_id"]),
            "seed": int(p["seed"]),
            "K": int(p["K"]),
            "distance": float(p["distance"]),
        } for p in plans],
    }, indent=2), encoding="utf-8")

    print("-" * 120)
    print(f" wrote {plans_csv}")
    print(f" wrote {scenarios_csv}")
    print(f" wrote {pairs_csv}")
    print(f" wrote {meta_json}")
    print("\nGO / NO-GO gate:")
    print("  * Primary metric: among pairs where BOTH lost routes are substantive, P(extra units = 2).")
    print("  * GO only if near-equal-cost plans retain a stable P(R=2) or mean-R gap at d=0.10 and d=0.20.")
    print("  * If only residual workload differs while every plan has the same R distribution, routing-risk framing is NO-GO.")
    print("  * This is a structural probe, not an empirical outage-probability claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
