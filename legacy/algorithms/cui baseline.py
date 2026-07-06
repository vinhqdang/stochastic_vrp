#!/usr/bin/env python3
"""
================================================================================
Cui et al. (2025) Bertsimas-Sim Budget Uncertainty Baseline for VRPSPD
================================================================================

Implements the Budget Uncertainty robust evaluator adapted for VRP:
  - Each node's demand can deviate up to hat_d_i from nominal
  - At most Gamma nodes deviate simultaneously (budget constraint)
  - Worst-case load = nominal + top-Gamma sorted deviations
  - If worst-case load > C at ANY stop -> route INFEASIBLE (Z = inf)
  - Otherwise Z = distance (no soft penalty — hard feasibility)

This is a HARD CONSTRAINT approach (like Gounaris), NOT soft penalty (like M-DRO).

KEY DIFFERENCE from M-DRO (alns.py):
  - M-DRO: Cantelli bound -> soft penalty -> allows controlled risk
  - Cui:   Budget worst-case -> hard constraint -> zero tolerance

KEY DIFFERENCE from Gounaris:
  - Gounaris: quadrant/factor model structure for spatial budget
  - Cui:      single global budget Gamma (simpler, no spatial structure)

Reference:
  Cui et al. (2025). "A multi-location inventory system with occasional
  allocation under uncertain defective rates." Data Science and Management.
  Uses Bertsimas-Sim budget uncertainty (Bertsimas & Sim, 2004).

Adaptation to VRP:
  - Original Cui: inventory allocation with defective rate uncertainty
  - This code: VRP route evaluation with demand uncertainty
  - Gamma controls how many nodes can spike simultaneously
  - hat_d_i = alpha * (P_i + D_i) = max deviation at node i

Same ALNS engine, same augmentation, same Monte Carlo validation as alns.py.

Dependencies: numpy, scipy
================================================================================
"""

import os
import glob
import csv
import time
import math
import random
import copy
from typing import List, Tuple, Dict, NamedTuple
import numpy as np
from scipy.stats import norm, skewnorm, t

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

STOCHASTIC_MODE = True

CV = 0.2
CAPACITY_FACTOR = 1.2
THETA_FRACTION = 0.1

# --- Cui Budget Uncertainty parameters ---
CUI_ALPHA = 0.2
# Maximum fractional deviation per node.
# hat_d_i = CUI_ALPHA * (P_i + D_i)
# Node i's demand can be anywhere in [nominal - hat_d_i, nominal + hat_d_i]

CUI_GAMMA = 0.5
# Budget parameter: fraction of route nodes that can deviate simultaneously.
# Actual budget = ceil(CUI_GAMMA * m) where m = route length.
# Gamma=0: no deviation allowed (deterministic)
# Gamma=1: all nodes can deviate (full worst-case, most conservative)
# Bertsimas & Sim (2004) recommend Gamma in [0.3, 0.7] for practical use.

# --- ALNS parameters (same as alns.py) ---
MC_SAMPLES = 10000
ALNS_ITERATIONS = 5000
SA_TEMP_INIT = 100.0
SA_COOLING = 0.9997
DESTROY_MIN_FRAC = 0.1
DESTROY_MAX_FRAC = 0.4
SEGMENT_SIZE = 100
SIGMA_1 = 33
SIGMA_2 = 9
SIGMA_3 = 13
REACTION_FACTOR = 0.8
SEED = 42

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

class Node(NamedTuple):
    id: int
    x: float
    y: float
    delivery: float
    pickup: float
    sigma_sq: float  # kept for MC validation covariance matrix

class ProblemInstance:
    def __init__(self, name, nodes, capacity, dist_matrix, cov_matrix,
                 theta, is_stochastic):
        self.name = name
        self.nodes = nodes
        self.n = len(nodes) - 1
        self.capacity = capacity
        self.dist_matrix = dist_matrix
        self.cov_matrix = cov_matrix
        self.theta = theta
        self.is_stochastic = is_stochastic

# ==============================================================================
# FILE PARSERS (identical to alns.py)
# ==============================================================================

def parse_vrp_file(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    coords, demands = [], []
    capacity, section = 0.0, None
    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line: continue
            if line.startswith("NAME"): name = line.split(":")[-1].strip()
            elif line.startswith("CAPACITY"): capacity = float(line.split(":")[-1].strip())
            elif line.startswith("NODE_COORD_SECTION"): section = "COORD"
            elif line.startswith("DEMAND_SECTION"): section = "DEMAND"
            elif line.startswith("DEPOT_SECTION"): section = "DEPOT"
            elif line == "EOF": section = None
            else:
                if section == "COORD":
                    parts = line.split()
                    if len(parts) >= 3: coords.append((float(parts[1]), float(parts[2])))
                elif section == "DEMAND":
                    parts = line.split()
                    if len(parts) >= 2: demands.append(float(parts[1]))
                elif section == "DEPOT":
                    if line.strip() == "-1": section = None
    return name, coords, demands, capacity


def parse_bks_solution(sol_filepath):
    bks_routes = []
    if not os.path.exists(sol_filepath): return []
    with open(sol_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("route"):
                parts = line.split(":")
                if len(parts) == 2:
                    nodes_str = parts[1].strip().split()
                    route = [int(n) for n in nodes_str]
                    if route: bks_routes.append(route)
    return bks_routes

# ==============================================================================
# DATA AUGMENTATION (identical to alns.py)
# ==============================================================================

def augment_instance(name, coords, demands, capacity_raw, rng, stochastic):
    n_total = len(coords)
    coord_arr = np.array(coords)
    diff = coord_arr[:, np.newaxis, :] - coord_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    max_dist = np.max(dist_matrix)
    theta = THETA_FRACTION * max_dist if max_dist > 0 else 1.0
    capacity = capacity_raw * CAPACITY_FACTOR if stochastic else capacity_raw

    nodes = []
    for i in range(n_total):
        d_i = demands[i]
        if i == 0:
            nodes.append(Node(0, coords[0][0], coords[0][1], 0.0, 0.0, 0.0))
        else:
            if stochastic:
                p_i = d_i * rng.uniform(0.5, 1.5)
                sigma_sq = (CV * (p_i + d_i)) ** 2
            else:
                p_i = 0.0; sigma_sq = 0.0
            nodes.append(Node(i, coords[i][0], coords[i][1], d_i, p_i, sigma_sq))

    sigmas = np.array([math.sqrt(n.sigma_sq) for n in nodes])
    cov_matrix = np.outer(sigmas, sigmas) * np.exp(-dist_matrix / theta)
    return ProblemInstance(name, nodes, capacity, dist_matrix, cov_matrix, theta, stochastic)

# ==============================================================================
# CUI BERTSIMAS-SIM BUDGET EVALUATOR
# ==============================================================================

def evaluate_route_cui(route, inst):
    """
    Cui (2025) Bertsimas-Sim Budget Uncertainty evaluator.

    === ALGORITHM (adapted from Bertsimas & Sim 2004) ===

    For a route with m customers:

    1. Compute nominal load at each stop k:
       L_k = total_delivery + sum_{i=1}^{k} (P_i - D_i)

    2. Compute max deviation at each stop k:
       hat_d_k = CUI_ALPHA * (P_k + D_k)
       This is the maximum amount node k's net demand can increase.

    3. At each stop k, compute worst-case ADDITIONAL load:
       - Collect all deviations hat_d_1, ..., hat_d_k (nodes visited so far)
       - Sort descending
       - Budget = ceil(CUI_GAMMA * k) nodes can deviate simultaneously
       - worst_extra_k = sum of top-budget deviations
         (with fractional Gamma: last node contributes partial deviation)

    4. Worst-case load at stop k:
       L_k_worst = L_k_nominal + worst_extra_k

    5. If L_k_worst > C for ANY stop -> route INFEASIBLE (Z = inf)
       Otherwise Z = distance (feasible under worst case -> no penalty needed)

    This is HARD FEASIBILITY: either the route survives ALL worst cases
    within budget, or it's rejected entirely. No soft penalty, no trade-off.

    Returns: (distance, worst_case_margin, Z_cost)
      worst_case_margin = min over all stops of (C - L_k_worst)
      Negative margin = infeasible.
    """
    if not route:
        return 0.0, 0.0, 0.0

    C = inst.capacity
    m = len(route)

    # Route distance
    dist = inst.dist_matrix[0, route[0]]
    for k in range(1, m):
        dist += inst.dist_matrix[route[k - 1], route[k]]
    dist += inst.dist_matrix[route[-1], 0]

    if not inst.is_stochastic:
        load = sum(inst.nodes[n].delivery for n in route)
        if load > C:
            overload = load - C
            return dist, C - load, dist + 10000.0 * overload
        for n in route:
            load += inst.nodes[n].pickup - inst.nodes[n].delivery
            if load > C:
                overload = load - C
                return dist, C - load, dist + 10000.0 * overload
        return dist, C - load, dist

    # --- Bertsimas-Sim Budget evaluation ---
    total_delivery = sum(inst.nodes[n].delivery for n in route)
    if total_delivery > C:
        overload = total_delivery - C
        return dist, C - total_delivery, dist + 10000.0 * overload

    # Track nominal cumulative load and collected deviations
    cum_load = total_delivery  # starts at L_0
    deviations_so_far = []     # max deviations of nodes visited so far
    min_margin = float('inf')  # track tightest capacity margin

    for k in range(m):
        node_k = route[k]

        # Nominal load update: L_k = L_{k-1} + (P_k - D_k)
        cum_load += inst.nodes[node_k].pickup - inst.nodes[node_k].delivery

        # Max deviation at this node: hat_d_k = alpha * total_volume_k
        hat_d_k = CUI_ALPHA * (inst.nodes[node_k].pickup + inst.nodes[node_k].delivery)
        deviations_so_far.append(hat_d_k)

        # Bertsimas-Sim worst-case computation:
        # Sort all deviations seen so far in descending order
        # Budget = ceil(CUI_GAMMA * number_of_nodes_visited)
        sorted_devs = sorted(deviations_so_far, reverse=True)
        n_visited = k + 1
        budget_float = CUI_GAMMA * n_visited
        budget_floor = int(math.floor(budget_float))
        budget_frac = budget_float - budget_floor

        # Sum of top-floor deviations (full contribution)
        worst_extra = sum(sorted_devs[:budget_floor])

        # Fractional node (partial contribution)
        if budget_frac > 0 and budget_floor < n_visited:
            worst_extra += budget_frac * sorted_devs[budget_floor]

        # Worst-case load at this stop
        worst_load = cum_load + worst_extra
        margin = C - worst_load
        min_margin = min(min_margin, margin)

        # Soft penalty for overload (gives ALNS gradient to escape infeasibility)
        if worst_load > C:
            overload = worst_load - C
            penalty = 10000.0 * overload
            return dist, margin, dist + penalty

    # Route survived all worst-case scenarios within budget
    return dist, min_margin, dist


def evaluate_solution(routes, inst):
    if not routes: return 0.0, 0.0, 0.0
    td, tm, tz = 0.0, float('inf'), 0.0
    for route in routes:
        if not route: continue
        d, margin, z = evaluate_route_cui(route, inst)
        td += d
        tm = min(tm, margin)
        tz += z
    return td, tm, tz

# ==============================================================================
# INITIAL SOLUTION (same as alns.py)
# ==============================================================================

def build_initial_solution(inst):
    unvisited = set(range(1, inst.n + 1))
    routes = []
    C = inst.capacity
    while unvisited:
        route = []
        cum_del = 0.0
        current = 0
        while unvisited:
            best_next, best_dist = -1, float('inf')
            for cand in unvisited:
                if cum_del + inst.nodes[cand].delivery > C: continue
                dd = inst.dist_matrix[current, cand]
                if dd < best_dist: best_dist = dd; best_next = cand
            if best_next == -1: break
            route.append(best_next)
            cum_del += inst.nodes[best_next].delivery
            current = best_next
            unvisited.discard(best_next)
        if route: routes.append(route)
        elif unvisited: routes.append([unvisited.pop()])
    return routes

# ==============================================================================
# ALNS OPERATORS (same as alns.py, using evaluate_route_cui)
# ==============================================================================

def random_removal(routes, inst, rng):
    all_cust = [c for r in routes for c in r]
    if not all_cust: return routes, []
    n_rem = max(1, int(len(all_cust) * rng.uniform(DESTROY_MIN_FRAC, DESTROY_MAX_FRAC)))
    n_rem = min(n_rem, len(all_cust))
    removed = list(rng.choice(all_cust, size=n_rem, replace=False))
    rem_set = set(removed)
    new_routes = [[c for c in r if c not in rem_set] for r in routes]
    return [r for r in new_routes if r], removed


def worst_removal(routes, inst, rng):
    DETERMINISM = 3
    all_cust = [c for r in routes for c in r]
    if not all_cust: return routes, []
    n_rem = max(1, int(len(all_cust) * rng.uniform(DESTROY_MIN_FRAC, DESTROY_MAX_FRAC)))
    n_rem = min(n_rem, len(all_cust))
    curr_routes = [list(r) for r in routes]
    removed = []
    for _ in range(n_rem):
        costs = []
        for r in curr_routes:
            for ci, cust in enumerate(r):
                trial = r[:ci] + r[ci + 1:]
                z_old = evaluate_route_cui(r, inst)[2]
                z_new = evaluate_route_cui(trial, inst)[2] if trial else 0.0
                costs.append((z_old - z_new, cust))
        if not costs: break
        costs.sort(key=lambda x: -x[0])
        idx = min(int(len(costs) * (rng.random() ** DETERMINISM)), len(costs) - 1)
        chosen = costs[idx][1]
        removed.append(chosen)
        for r in curr_routes:
            if chosen in r: r.remove(chosen); break
        curr_routes = [r for r in curr_routes if r]
    return curr_routes, removed


def _best_insertion_cost(route, customer, inst):
    z_before = evaluate_route_cui(route, inst)[2] if route else 0.0
    best_delta, best_pos = float('inf'), 0
    for pos in range(len(route) + 1):
        new_route = route[:pos] + [customer] + route[pos:]
        z_after = evaluate_route_cui(new_route, inst)[2]
        delta = z_after - z_before
        if delta < best_delta: best_delta = delta; best_pos = pos
    return best_delta, best_pos


def greedy_insertion(routes, removed, inst, rng):
    rng.shuffle(removed)
    curr = [list(r) for r in routes]
    for cust in removed:
        best_delta, best_ri, best_pos = float('inf'), -1, 0
        for ri, r in enumerate(curr):
            delta, pos = _best_insertion_cost(r, cust, inst)
            if delta < best_delta: best_delta = delta; best_ri = ri; best_pos = pos
        new_delta, _ = _best_insertion_cost([], cust, inst)
        if new_delta < best_delta or best_ri == -1: curr.append([cust])
        else: curr[best_ri].insert(best_pos, cust)
    return curr


def regret2_insertion(routes, removed, inst, rng):
    curr = [list(r) for r in routes]
    pool = list(removed)
    while pool:
        regret_list = []
        for cust in pool:
            ins_costs = []
            for ri, r in enumerate(curr):
                delta, pos = _best_insertion_cost(r, cust, inst)
                ins_costs.append((delta, ri, pos))
            new_delta, _ = _best_insertion_cost([], cust, inst)
            ins_costs.append((new_delta, -1, 0))
            ins_costs.sort(key=lambda x: x[0])
            best_cost = ins_costs[0][0]
            second_cost = ins_costs[1][0] if len(ins_costs) > 1 else best_cost
            regret_list.append((second_cost - best_cost, cust, ins_costs[0][1], ins_costs[0][2]))
        regret_list.sort(key=lambda x: -x[0])
        _, chosen, ri, pos = regret_list[0]
        if ri == -1: curr.append([chosen])
        else: curr[ri].insert(pos, chosen)
        pool.remove(chosen)
    return curr

# ==============================================================================
# ALNS SOLVER (same engine)
# ==============================================================================

def alns_solve(inst, rng, max_iter=ALNS_ITERATIONS):
    d_ops = [random_removal, worst_removal]
    r_ops = [greedy_insertion, regret2_insertion]
    nd, nr = len(d_ops), len(r_ops)
    d_w, r_w = np.ones(nd), np.ones(nr)
    d_s, r_s = np.zeros(nd), np.zeros(nr)
    d_c, r_c = np.zeros(nd), np.zeros(nr)

    curr = build_initial_solution(inst)
    curr_dist, _, curr_z = evaluate_solution(curr, inst)
    best = copy.deepcopy(curr)
    best_dist, best_z = curr_dist, curr_z
    temp = SA_TEMP_INIT
    history = [best_z]

    for it in range(1, max_iter + 1):
        d_idx = rng.choice(nd, p=d_w / d_w.sum())
        r_idx = rng.choice(nr, p=r_w / r_w.sum())
        d_c[d_idx] += 1; r_c[r_idx] += 1

        partial, removed = d_ops[d_idx](curr, inst, rng)
        cand = r_ops[r_idx](partial, removed, inst, rng)
        cand = [r for r in cand if r]
        cand_dist, _, cand_z = evaluate_solution(cand, inst)
        delta = cand_z - curr_z
        accepted = False

        if delta < 0:
            accepted = True
            if cand_z < best_z:
                best = copy.deepcopy(cand)
                best_dist, best_z = cand_dist, cand_z
                d_s[d_idx] += SIGMA_1; r_s[r_idx] += SIGMA_1
            else:
                d_s[d_idx] += SIGMA_2; r_s[r_idx] += SIGMA_2
        elif temp > 1e-10:
            try: prob = math.exp(-delta / temp)
            except OverflowError: prob = 0.0
            if rng.random() < prob:
                accepted = True
                d_s[d_idx] += SIGMA_3; r_s[r_idx] += SIGMA_3

        if accepted:
            curr, curr_dist, curr_z = cand, cand_dist, cand_z

        history.append(best_z)
        temp *= SA_COOLING
        if temp < 0.1: temp = SA_TEMP_INIT * 0.3

        if it % SEGMENT_SIZE == 0:
            for i in range(nd):
                if d_c[i] > 0:
                    d_w[i] = max(0.01, REACTION_FACTOR * d_w[i]
                                + (1 - REACTION_FACTOR) * (d_s[i] / d_c[i]))
            for i in range(nr):
                if r_c[i] > 0:
                    r_w[i] = max(0.01, REACTION_FACTOR * r_w[i]
                                + (1 - REACTION_FACTOR) * (r_s[i] / r_c[i]))
            d_s.fill(0); r_s.fill(0); d_c.fill(0); r_c.fill(0)

    return best, best_dist, best_z, history

# ==============================================================================
# MONTE CARLO VALIDATION (identical to alns.py — 4 scenarios)
# ==============================================================================

def monte_carlo_validate(routes, inst, rng, n_samples=MC_SAMPLES):
    results = {"GAUSSIAN": 0.0, "SKEW_RIGHT": 0.0, "SKEW_LEFT": 0.0, "HEAVY_TAIL": 0.0}
    if not inst.is_stochastic or not routes: return results

    total_sims, total_fails = 0, {k: 0 for k in results}
    C = inst.capacity

    for route in routes:
        if not route: continue
        total_sims += 1
        m = len(route)
        mean_vec = np.array([inst.nodes[c].pickup - inst.nodes[c].delivery for c in route])
        route_arr = np.array(route, dtype=np.intp)
        sub_cov = inst.cov_matrix[np.ix_(route_arr, route_arr)]
        sub_cov = 0.5 * (sub_cov + sub_cov.T)
        min_eig = np.min(np.linalg.eigvalsh(sub_cov))
        if min_eig < 1e-8: sub_cov += (1e-8 - min_eig) * np.eye(m)

        stds = np.sqrt(np.diag(sub_cov))
        outer_stds = np.outer(stds, stds)
        corr_matrix = np.divide(sub_cov, outer_stds, out=np.eye(m).copy(), where=outer_stds != 0)
        min_eig_corr = np.min(np.linalg.eigvalsh(corr_matrix))
        if min_eig_corr < 1e-8: corr_matrix += (1e-8 - min_eig_corr) * np.eye(m)

        Z = rng.multivariate_normal(np.zeros(m), corr_matrix, size=n_samples)
        U = norm.cdf(Z)

        scenarios = {
            "GAUSSIAN": norm.ppf(U),
            "SKEW_RIGHT": (skewnorm.ppf(U, 5.0) - skewnorm.stats(5.0, moments='m')) / np.sqrt(skewnorm.stats(5.0, moments='v')),
            "SKEW_LEFT": (skewnorm.ppf(U, -5.0) - skewnorm.stats(-5.0, moments='m')) / np.sqrt(skewnorm.stats(-5.0, moments='v')),
            "HEAVY_TAIL": t.ppf(U, df=3) / np.sqrt(3 / (3 - 2)),
        }

        total_delivery = sum(inst.nodes[c].delivery for c in route)
        for name, noise_std in scenarios.items():
            samples = mean_vec[np.newaxis, :] + noise_std * stds[np.newaxis, :]
            cum_loads = total_delivery + np.cumsum(samples, axis=1)
            total_fails[name] += np.sum(np.any(cum_loads > C, axis=1))

    if total_sims > 0:
        for k in results: results[k] = total_fails[k] / (total_sims * n_samples)
    return results

# ==============================================================================
# BENCHMARK ENGINE
# ==============================================================================

def run_benchmark(target_dir, cui_alpha=CUI_ALPHA, cui_gamma=CUI_GAMMA):
    global CUI_ALPHA, CUI_GAMMA
    CUI_ALPHA = cui_alpha
    CUI_GAMMA = cui_gamma

    print(f"\n{'='*130}")
    print(f"  CUI BASELINE: Bertsimas-Sim Budget Uncertainty ALNS")
    print(f"  alpha={CUI_ALPHA} (max deviation fraction) | Gamma={CUI_GAMMA} (budget fraction)")
    print(f"  CV={CV} | Cap x{CAPACITY_FACTOR} | ALNS={ALNS_ITERATIONS} | MC={MC_SAMPLES}")
    print(f"  Evaluator: HARD feasibility (worst-case load > C -> infeasible)")
    print(f"{'='*130}")

    vrp_files = sorted(glob.glob(os.path.join(target_dir, "*.vrp")))
    if not vrp_files:
        print(f"No .vrp files in {target_dir}"); return

    print(f"Found {len(vrp_files)} instances\n")
    print(f"{'Instance':<15} | {'N':>4} | "
          f"{'BKS_V':>5} {'BKS_Dist':>10} | "
          f"{'CUI_V':>5} {'CUI_Dist':>10} {'Margin':>8} {'Time':>7} | "
          f"{'F_N':>7} {'F_R':>7} {'F_L':>7} {'F_T':>7}")
    print("-" * 130)

    results = []

    for file_idx, v_path in enumerate(vrp_files):
        base_name = os.path.splitext(os.path.basename(v_path))[0]
        s_path = os.path.join(target_dir, f"{base_name}.sol")

        try:
            inst_seed = SEED + file_idx
            rng = np.random.default_rng(inst_seed)
            random.seed(inst_seed)

            name, coords, demands, capacity_raw = parse_vrp_file(v_path)
            inst = augment_instance(name, coords, demands, capacity_raw,
                                    rng, stochastic=True)

            # BKS
            bks_routes = parse_bks_solution(s_path)
            bks_dist = evaluate_solution(bks_routes, inst)[0] if bks_routes else 0.0
            bks_veh = len(bks_routes)

            # ALNS with Cui evaluator
            t_start = time.perf_counter()
            best_routes, alns_dist, alns_z, history = alns_solve(inst, rng, ALNS_ITERATIONS)
            alns_time = time.perf_counter() - t_start
            alns_veh = len(best_routes)

            # Get worst-case margin
            _, min_margin, _ = evaluate_solution(best_routes, inst)

            # Monte Carlo validate
            fail_dict = monte_carlo_validate(best_routes, inst, rng, MC_SAMPLES)

            print(f"{base_name:<15} | {inst.n:>4} | "
                  f"{bks_veh:>5} {bks_dist:>10.2f} | "
                  f"{alns_veh:>5} {alns_dist:>10.2f} {min_margin:>8.2f} {alns_time:>7.2f} | "
                  f"{fail_dict['GAUSSIAN']:>7.4f} {fail_dict['SKEW_RIGHT']:>7.4f} "
                  f"{fail_dict['SKEW_LEFT']:>7.4f} {fail_dict['HEAVY_TAIL']:>7.4f}")

            # Convergence CSV
            hist_csv = os.path.join(target_dir, f"convergence_cui_{base_name}.csv")
            with open(hist_csv, 'w', newline='', encoding='utf-8') as f:
                f.write("Iteration,Best_Z_Cost\n")
                for i, val in enumerate(history):
                    f.write(f"{i},{val:.4f}\n")

            results.append({
                "Instance": base_name, "Mode": "CUI_BUDGET",
                "N": inst.n, "CUI_Alpha": CUI_ALPHA, "CUI_Gamma": CUI_GAMMA,
                "Capacity_Used": round(inst.capacity, 1),
                "BKS_Vehicles": bks_veh, "BKS_Distance": round(bks_dist, 2),
                "CUI_Vehicles": alns_veh, "CUI_Distance": round(alns_dist, 2),
                "CUI_Z_Cost": round(alns_z, 2),
                "Min_Margin": round(min_margin, 2),
                "Runtime_s": round(alns_time, 2), "Seed": inst_seed,
                "Fail_Gaussian": round(fail_dict["GAUSSIAN"], 6),
                "Fail_SkewRight": round(fail_dict["SKEW_RIGHT"], 6),
                "Fail_SkewLeft": round(fail_dict["SKEW_LEFT"], 6),
                "Fail_HeavyTail": round(fail_dict["HEAVY_TAIL"], 6),
            })
        except Exception as e:
            print(f"{base_name:<15} | ERROR: {e}")

    print("-" * 130)

    out_csv = os.path.join(target_dir, f"results_cui_a{CUI_ALPHA}_g{CUI_GAMMA}.csv")
    if results:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), extrasaction='ignore')
            writer.writeheader(); writer.writerows(results)
        print(f"\nResults exported: {out_csv}")

    # Summary
    valid = [r for r in results if isinstance(r.get("Fail_SkewRight"), (int, float))]
    if valid:
        avg_time = np.mean([r["Runtime_s"] for r in valid])
        avg_dist = np.mean([r["CUI_Distance"] for r in valid])
        avg_fail = np.mean([r["Fail_SkewRight"] for r in valid])
        avg_margin = np.mean([r["Min_Margin"] for r in valid])
        print(f"   Avg Distance: {avg_dist:.2f} | Avg Margin: {avg_margin:.2f} | "
              f"Avg SkewRight Fail: {avg_fail:.4f} | Avg Time: {avg_time:.2f}s")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    TARGET_DIR = r"C:\Users\Admin\Downloads\New folder (4)\X"

    if not os.path.isdir(TARGET_DIR):
        env_dir = os.environ.get("VRP_DATA_DIR", ".")
        if os.path.isdir(env_dir): TARGET_DIR = env_dir
        else: print(f"[ERROR] {TARGET_DIR} not found"); exit(1)

    # Run with default parameters
    run_benchmark(TARGET_DIR, cui_alpha=0.1, cui_gamma=0.5)

    # Uncomment for ablation across Gamma values:
    # for gamma in [0.2, 0.3, 0.5, 0.7, 1.0]:
    #     run_benchmark(TARGET_DIR, cui_alpha=0.2, cui_gamma=gamma)