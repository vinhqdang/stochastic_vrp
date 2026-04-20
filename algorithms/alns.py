#!/usr/bin/env python3
"""
================================================================================
Advanced Benchmark Suite: M-DRO ALNS vs BKS vs Exact Methods
Purpose: Automate the evaluation of Stochastic VRPSPD on classical datasets
         (Set A, B, P) to compare runtime, optimality gap, and failure rates.
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
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky, LinAlgError

# ==============================================================================
# 0. GLOBAL CONSTANTS & HYPERPARAMETERS
# ==============================================================================

CV = 0.1                   # Coefficient of Variation for demand noise
THETA_FRACTION = 0.1       # Spatial correlation range
ALPHA_BASE = 0.05          # Base risk threshold 
GAMMA = 0.5                # Risk threshold exponent (Square root law)
LAMBDA_0 = 10000.0         # Base penalty multiplier
MC_SAMPLES = 500           # Monte Carlo samples
ALNS_ITERATIONS = 50       # ALNS budget
SA_TEMP_INIT = 100.0       # Simulated Annealing initial temp
SA_COOLING = 0.9997        # SA geometric cooling rate
DESTROY_MIN_FRAC = 0.1     # Min customers removed
DESTROY_MAX_FRAC = 0.15    # Max customers removed
SEED = 42                  # Reproducibility
SEGMENT_SIZE = 100         # ALNS weight update interval

# Operator scoring
SIGMA_1, SIGMA_2, SIGMA_3 = 33, 9, 13    
REACTION_FACTOR = 0.8  

# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================

class Node(NamedTuple):
    id: int
    x: float
    y: float
    delivery: float    
    pickup: float      
    sigma_sq: float    

class ProblemInstance:
    def __init__(self, name: str, nodes: List[Node], capacity: float,
                 dist_matrix: np.ndarray, cov_matrix: np.ndarray, theta: float):
        self.name = name
        self.nodes = nodes              
        self.n = len(nodes) - 1         
        self.capacity = capacity
        self.dist_matrix = dist_matrix  
        self.cov_matrix = cov_matrix    
        self.theta = theta

# ==============================================================================
# 2. FILE PARSERS
# ==============================================================================

def parse_vrp_file(filepath: str) -> Tuple[str, List[Tuple[float, float]], List[float], float]:
    name = os.path.splitext(os.path.basename(filepath))[0]
    coords, demands = [], []
    capacity, section = 0.0, None

    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line: continue
            if line.startswith("NAME"): name = line.split(":")[-1].strip()
            elif line.startswith("CAPACITY"): capacity = float(line.split(":")[-1].strip()) * 1.2
            elif line.startswith("DEPOT_SECTION"): section = "DEPOT"
            elif line.startswith("NODE_COORD_SECTION"): section = "COORD"
            elif line.startswith("DEMAND_SECTION"): section = "DEMAND"
            elif line in ("EOF", "DEPOT_SECTION"): section = "DEPOT" if line == "DEPOT_SECTION" else None
            else:
                if section == "COORD":
                    parts = line.split()
                    if len(parts) >= 3: coords.append((float(parts[1]), float(parts[2])))
                elif section == "DEMAND":
                    parts = line.split()
                    if len(parts) >= 2: demands.append(float(parts[1]))
                elif section == "DEPOT" and line.strip() == "-1":
                    section = None
    return name, coords, demands, capacity

def parse_bks_solution(sol_filepath: str) -> List[List[int]]:
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
# 3. DATA AUGMENTATION (Deterministic -> Stochastic)
# ==============================================================================

def augment_instance(name: str, coords: List[Tuple[float, float]],
                     demands: List[float], capacity: float, rng: np.random.Generator) -> ProblemInstance:
    n_total = len(coords)
    coord_arr = np.array(coords)
    diff = coord_arr[:, np.newaxis, :] - coord_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    theta = THETA_FRACTION * np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1.0

    nodes = []
    for i in range(n_total):
        d_i = demands[i]
        if i == 0:
            nodes.append(Node(id=0, x=coords[0][0], y=coords[0][1], delivery=0.0, pickup=0.0, sigma_sq=0.0))
        else:
            p_i = d_i * rng.uniform(0.5, 1.5)
            sigma_sq = (CV * (p_i + d_i)) ** 2
            nodes.append(Node(id=i, x=coords[i][0], y=coords[i][1], delivery=d_i, pickup=p_i, sigma_sq=sigma_sq))

    sigmas = np.array([math.sqrt(n.sigma_sq) for n in nodes])
    cov_matrix = np.outer(sigmas, sigmas) * np.exp(-dist_matrix / theta)
    return ProblemInstance(name, nodes, capacity, dist_matrix, cov_matrix, theta)

# ==============================================================================
# 4. M-DRO ROUTE EVALUATOR (O(m) Prefix-Sum)
# ==============================================================================

def evaluate_route_dro(route: List[int], inst: ProblemInstance) -> Tuple[float, float, float]:
    if not route: return 0.0, 0.0, 0.0
    C, m = inst.capacity, len(route)

    dist = inst.dist_matrix[0, route[0]]
    for k in range(1, m): dist += inst.dist_matrix[route[k - 1], route[k]]
    dist += inst.dist_matrix[route[-1], 0]

    cum_mean = sum(inst.nodes[n].delivery for n in route)
    cum_var, rri = 0.0, 0.0
    cov_prefix_vector = np.zeros(inst.n + 1)

    for k in range(m):
        node_k = route[k]
        cum_mean += inst.nodes[node_k].pickup - inst.nodes[node_k].delivery
        if k == 0: cum_var = inst.nodes[node_k].sigma_sq
        else: cum_var += inst.nodes[node_k].sigma_sq + 2.0 * cov_prefix_vector[node_k]
        
        cov_prefix_vector += inst.cov_matrix[node_k]

        slack = C - cum_mean
        p_fail = 1.0 if slack <= 0 else (0.0 if cum_var <= 0 else cum_var / (cum_var + slack * slack))
        rri += p_fail

    alpha_m = ALPHA_BASE * (m ** GAMMA)
    z_penalty = (LAMBDA_0 / math.log(m + 1)) * max(0.0, rri - alpha_m)
    return dist, rri, dist + z_penalty

def evaluate_solution(routes: List[List[int]], inst: ProblemInstance) -> Tuple[float, float, float]:
    return tuple(sum(x) for x in zip(*(evaluate_route_dro(r, inst) for r in routes))) if routes else (0,0,0)

# ==============================================================================
# 5. INITIALIZATION & ALNS OPERATORS
# ==============================================================================

def build_initial_solution(inst: ProblemInstance) -> List[List[int]]:
    unvisited, routes, C = set(range(1, inst.n + 1)), [], inst.capacity
    while unvisited:
        route, cum_delivery, current = [], 0.0, 0
        while unvisited:
            best_next, best_dist = -1, float('inf')
            for cand in unvisited:
                if cum_delivery + inst.nodes[cand].delivery > C: continue
                dd = inst.dist_matrix[current, cand]
                if dd < best_dist: best_dist, best_next = dd, cand
            if best_next == -1: break
            route.append(best_next)
            cum_delivery += inst.nodes[best_next].delivery
            current = best_next
            unvisited.discard(best_next)
        routes.append(route) if route else routes.append([unvisited.pop()])
    return routes

def random_removal(routes, inst, rng):
    all_cust = [c for r in routes for c in r]
    n_rem = min(len(all_cust), max(1, int(len(all_cust) * rng.uniform(DESTROY_MIN_FRAC, DESTROY_MAX_FRAC))))
    rem = set(rng.choice(all_cust, size=n_rem, replace=False))
    return [[c for c in r if c not in rem] for r in routes if [c for c in r if c not in rem]], list(rem)

def worst_removal(routes, inst, rng):
    all_cust = [c for r in routes for c in r]
    n_rem = min(len(all_cust), max(1, int(len(all_cust) * rng.uniform(DESTROY_MIN_FRAC, DESTROY_MAX_FRAC))))
    curr_routes = [list(r) for r in routes]
    removed = []
    
    for _ in range(n_rem):
        costs = []
        for r in curr_routes:
            for ci, cust in enumerate(r):
                trial = r[:ci] + r[ci+1:]
                savings = evaluate_route_dro(r, inst)[2] - (evaluate_route_dro(trial, inst)[2] if trial else 0)
                costs.append((savings, cust))
        if not costs: break
        costs.sort(key=lambda x: -x[0])
        chosen = costs[min(int(len(costs) * (rng.random() ** 3)), len(costs)-1)][1]
        removed.append(chosen)
        for r in curr_routes:
            if chosen in r: r.remove(chosen); break
        curr_routes = [r for r in curr_routes if r]
    return curr_routes, removed

def _best_ins(route, cust, inst):
    zb = evaluate_route_dro(route, inst)[2] if route else 0
    best_d, best_p = float('inf'), 0
    for p in range(len(route) + 1):
        d = evaluate_route_dro(route[:p] + [cust] + route[p:], inst)[2] - zb
        if d < best_d: best_d, best_p = d, p
    return best_d, best_p

def greedy_insertion(routes, removed, inst, rng):
    rng.shuffle(removed)
    curr = [list(r) for r in routes]
    for cust in removed:
        best_d, best_r, best_p = float('inf'), -1, 0
        for ri, r in enumerate(curr):
            d, p = _best_ins(r, cust, inst)
            if d < best_d: best_d, best_r, best_p = d, ri, p
        nd, _ = _best_ins([], cust, inst)
        if nd < best_d or best_r == -1: curr.append([cust])
        else: curr[best_r].insert(best_p, cust)
    return curr

def regret2_insertion(routes, removed, inst, rng):
    curr = [list(r) for r in routes]
    pool = list(removed)
    while pool:
        reg_list = []
        for cust in pool:
            costs = [(_best_ins(r, cust, inst)[0], ri, _best_ins(r, cust, inst)[1]) for ri, r in enumerate(curr)]
            costs.append((_best_ins([], cust, inst)[0], -1, 0))
            costs.sort(key=lambda x: x[0])
            reg_list.append((costs[1][0] - costs[0][0] if len(costs)>1 else 0, cust, costs[0][1], costs[0][2]))
        reg_list.sort(key=lambda x: -x[0])
        _, chosen, ri, pos = reg_list[0]
        if ri == -1: curr.append([chosen])
        else: curr[ri].insert(pos, chosen)
        pool.remove(chosen)
    return curr

# ==============================================================================
# 6. ALNS SOLVER
# ==============================================================================

def alns_solve(inst: ProblemInstance, rng: np.random.Generator, max_iter: int = ALNS_ITERATIONS):
    d_ops, r_ops = [random_removal, worst_removal], [greedy_insertion, regret2_insertion]
    d_w, r_w = np.ones(len(d_ops)), np.ones(len(r_ops))
    d_s, r_s, d_c, r_c = np.zeros(len(d_ops)), np.zeros(len(r_ops)), np.zeros(len(d_ops)), np.zeros(len(r_ops))
    
    curr = build_initial_solution(inst)
    curr_dist, _, curr_z = evaluate_solution(curr, inst)
    best, best_dist, best_z = copy.deepcopy(curr), curr_dist, curr_z
    temp = SA_TEMP_INIT

    for it in range(1, max_iter + 1):
        d_idx = rng.choice(len(d_ops), p=d_w/d_w.sum())
        r_idx = rng.choice(len(r_ops), p=r_w/r_w.sum())
        d_c[d_idx] += 1; r_c[r_idx] += 1

        cand = r_ops[r_idx](*d_ops[d_idx](curr, inst, rng), inst, rng)
        cand_dist, _, cand_z = evaluate_solution(cand, inst)
        
        delta = cand_z - curr_z
        acc = False
        if delta < 0:
            acc = True
            if cand_z < best_z:
                best, best_dist, best_z = copy.deepcopy(cand), cand_dist, cand_z
                d_s[d_idx] += SIGMA_1; r_s[r_idx] += SIGMA_1
            else:
                d_s[d_idx] += SIGMA_2; r_s[r_idx] += SIGMA_2
        elif temp > 1e-10 and rng.random() < math.exp(-delta / temp):
            acc = True
            d_s[d_idx] += SIGMA_3; r_s[r_idx] += SIGMA_3

        if acc: curr, curr_dist, curr_z = cand, cand_dist, cand_z
        
        temp *= SA_COOLING
        if temp < 0.1: temp = SA_TEMP_INIT * 0.3

        if it % SEGMENT_SIZE == 0:
            for i in range(len(d_ops)):
                if d_c[i] > 0: d_w[i] = max(0.01, REACTION_FACTOR * d_w[i] + (1 - REACTION_FACTOR) * (d_s[i]/d_c[i]))
            for i in range(len(r_ops)):
                if r_c[i] > 0: r_w[i] = max(0.01, REACTION_FACTOR * r_w[i] + (1 - REACTION_FACTOR) * (r_s[i]/r_c[i]))
            d_s.fill(0); r_s.fill(0); d_c.fill(0); r_c.fill(0)

    return best, best_dist, best_z

# ==============================================================================
# 7. MONTE CARLO VALIDATION
# ==============================================================================

def monte_carlo_validate(routes: List[List[int]], inst: ProblemInstance, rng: np.random.Generator, n_samples: int = MC_SAMPLES) -> float:
    if not routes: return 0.0
    total_sims, total_fails, C = 0, 0, inst.capacity
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
        
        try: samples = rng.multivariate_normal(mean_vec, sub_cov, size=n_samples)
        except: samples = mean_vec[np.newaxis, :] + rng.standard_normal((n_samples, m)) * np.sqrt(np.diag(sub_cov))[np.newaxis, :]
        
        cum_loads = sum(inst.nodes[c].delivery for c in route) + np.cumsum(samples, axis=1)
        total_fails += np.sum(np.any(cum_loads > C, axis=1))
    return total_fails / (total_sims * n_samples) if total_sims > 0 else 0.0

# ==============================================================================
# 8. THE BENCHMARK ENGINE
# ==============================================================================

# ==============================================================================
# 8. THE BENCHMARK ENGINE (BẢN CẬP NHẬT CÓ SỐ XE)
# ==============================================================================

def run_paper_benchmark(target_dir: str):
    rng = np.random.default_rng(SEED) 
    
    vrp_files = sorted(glob.glob(os.path.join(target_dir, "*.vrp")))
    if not vrp_files:
        print(f"❌ Không tìm thấy file .vrp trong {target_dir}")
        return

    results = []
    print("="*125)
    print(f"{'Instance':<15} | {'N':<4} | {'Veh(B)':<6} {'BKS_Cost':<10} {'Fail(B)':<8} | {'Veh(A)':<6} {'ALNS_Cost':<10} {'Fail(A)':<8} {'Time(s)':<8} | {'Gap(%)':<8}")
    print("="*125)

    for v_path in vrp_files:
        base_name = os.path.splitext(os.path.basename(v_path))[0]
        s_path = os.path.join(target_dir, f"{base_name}.sol")
        
        try:
            # 1. Parse & Augment
            name, coords, demands, capacity = parse_vrp_file(v_path)
            inst = augment_instance(name, coords, demands, capacity, rng)
            
            # 2. Đánh giá BKS
            bks_routes = parse_bks_solution(s_path)
            bks_fail, bks_dist, bks_veh = 0.0, 0.0, len(bks_routes)
            if bks_routes:
                bks_fail = monte_carlo_validate(bks_routes, inst, rng, MC_SAMPLES)
                bks_dist = evaluate_solution(bks_routes, inst)[0]

            # 3. Chạy ALNS 
            start_time = time.perf_counter()
            best_routes, alns_dist, _ = alns_solve(inst, rng, max_iter=ALNS_ITERATIONS)
            alns_time = time.perf_counter() - start_time
            alns_veh = len(best_routes)
            
            alns_fail = monte_carlo_validate(best_routes, inst, rng, MC_SAMPLES)
            
            # 4. Tính Gap
            gap = ((alns_dist - bks_dist) / bks_dist * 100) if bks_dist > 0 else 0.0

            print(f"{base_name:<15} | {inst.n:<4} | {bks_veh:<6} {bks_dist:<10.2f} {bks_fail:<8.4f} | {alns_veh:<6} {alns_dist:<10.2f} {alns_fail:<8.4f} {alns_time:<8.2f} | {gap:<8.2f}%")
            
            # 5. Lưu data full không che
            results.append({
                "Instance": base_name,
                "Nodes (N)": inst.n,
                "BKS_Vehicles": bks_veh,
                "BKS_Cost (Det)": round(bks_dist, 2),
                "BKS_FailRate": round(bks_fail, 4),
                "ALNS_Vehicles": alns_veh,
                "ALNS_Cost (Stoch)": round(alns_dist, 2),
                "ALNS_FailRate": round(alns_fail, 4),
                "ALNS_Runtime (s)": round(alns_time, 2),
                "ALNS_vs_BKS_Gap (%)": round(gap, 2),
                "Exact_Cost_Paper": "", 
                "Exact_Time_Paper": ""
            })
            
        except Exception as e:
            print(f"{base_name:<15} ❌ LỖI: {e}")

    print("="*125)
    
    # 6. Xuất CSV
    out_csv = os.path.join(target_dir, "chưa tắt stochastic(20% theo naris).csv")
    if results:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Xong! Đã xuất bảng số liệu tại: {out_csv}")

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    
    TARGET_DIR = r"C:\Users\Admin\Downloads\New folder (4)\X" 
    run_paper_benchmark(TARGET_DIR)