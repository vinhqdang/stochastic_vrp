"""
Unified VRPSPD Benchmark: Deterministic <-> Stochastic 

Solves the Vehicle Routing Problem with Simultaneous Pickup & Delivery (VRPSPD)
using Adaptive Large Neighborhood Search (ALNS) metaheuristic, with an optional
Moment-Based Distributionally Robust Optimization (M-DRO) risk layer.

TWO MODES controlled by STOCHASTIC_MODE flag:

  DETERMINISTIC (False):
    - Pure CVRP: delivery only, pickup = 0, no randomness
    - Capacity = original value from .vrp file
    - Objective = minimize total Euclidean distance
    - Gap vs BKS (Best Known Solution) is meaningful

  STOCHASTIC (True):
    - VRPSPD: random pickup P_i = D_i * Uniform(0.5, 1.5) added
    - Capacity = original * 1.2 (compensates for added pickup load)
    - Demand uncertainty modeled with spatially correlated covariance
    - Objective = distance + DRO risk penalty (Cantelli-Chebyshev bound)
    - Post-optimization Monte Carlo validation under Gaussian scenario
    - Gap vs BKS is N/A (different optimization problem)

Each instance gets its own RNG seed (SEED + file_index) so results are
reproducible regardless of which other .vrp files are in the directory.

References:
  [1] Cantelli, F.P. (1928). "Sui confini della probabilita."
  [2] Schoenberg, I.J. (1938). "Metric spaces and positive definite functions."
      Transactions of the AMS.
  [3] Ropke, S. & Pisinger, D. (2006). "An Adaptive Large Neighborhood Search
      Heuristic for the Pickup and Delivery Problem with Time Windows."
      Transportation Science 40(4).
  [4] Hajek, B. (1988). "Cooling schedules for optimal annealing."
      Mathematics of Operations Research 13(2).
  [5] Delage, E. & Ye, Y. (2010). "Distributionally robust optimization under
      moment uncertainty sets." Operations Research 58(3).
  [6] Schur, I. (1911). Hadamard product of PSD matrices is PSD.

Dependencies: numpy, scipy
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
from scipy.linalg import LinAlgError
from scipy.stats import norm, skewnorm, t  # THÊM THƯ VIỆN CHO COPULA 4 VŨ TRỤ

# 0. MASTER TOGGLE

STOCHASTIC_MODE = True
# True  = Stochastic VRPSPD with DRO risk penalty + Monte Carlo validation
# False = Deterministic CVRP with hard capacity check, no randomness

# 1. HYPERPARAMETERS — each one annotated with its source

#Stochastic demand model (active only when STOCHASTIC_MODE=True)

CV = 0.2
# Coefficient of Variation for demand noise.
# Defines how much individual demand fluctuates around its mean:
#   standard_deviation = CV * mean_demand
# SELF-SET heuristic. Typical range in supply chain literature: 0.1-0.3.
# Must be estimated from historical data in real applications.

CAPACITY_FACTOR = 1.2
# Multiply the original CVRP capacity by this factor in stochastic mode.
# Purpose: the original capacity was designed for delivery-only (CVRP).
# Adding pickup demand increases total load, so we relax capacity by 20%.
# SELF-SET heuristic following Gnounaris adapt convention.
# Set to 1.0 if you want no relaxation.

THETA_FRACTION = 0.1
# Spatial correlation range parameter.
# theta = THETA_FRACTION * max_pairwise_distance
# At distance = theta, correlation drops to exp(-1) ~ 0.37.
# This controls how far a demand "shock" propagates geographically.
# SELF-SET heuristic. Should be tuned via variogram fitting on real data.

# DRO risk penalty calibration --
# These control the penalty term in the objective function:
#   Z = distance + [LAMBDA_0 / ln(m+1)] * max(0, RRI - ALPHA_BASE * m^GAMMA)
# ALL THREE are SELF-SET calibration heuristics with NO theoretical derivation.
# They should be tuned via ablation study on each problem class.

ALPHA_BASE = 0.05
# Base risk threshold. A single-customer route is allowed RRI up to 0.05
# before penalty kicks in. Larger value = more risk-tolerant.

GAMMA = 0.5
# Exponent for scaling threshold with route length m.
# threshold(m) = ALPHA_BASE * m^GAMMA
# Purpose: longer routes naturally accumulate more RRI terms (union bound),
# so the threshold must grow to avoid unfairly penalizing long routes.
# gamma=0.5 means threshold grows as sqrt(m). This is NOT derived from CLT
# (CLT concerns normalized sums, not tail probability sums). Pure heuristic.

LAMBDA_0 = 10000.0
# Base penalty multiplier. Controls how strongly the solver avoids risky routes.
# Divided by ln(m+1) to soften the penalty for longer routes.
# SELF-SET. Too small -> ignores risk. Too large -> overly conservative routing.

# Monte Carlo validation 

MC_SAMPLES = 10000
# Number of random demand realizations to simulate per route.
# Used ONLY for post-optimization validation (not during ALNS search).
# Estimation error scales as 1/sqrt(MC_SAMPLES): 10000 -> ~1% precision.
# Skipped entirely in deterministic mode (variance = 0 -> nothing to simulate).

#  ALNS search engine 

ALNS_ITERATIONS = 5000
# Total destroy-repair cycles. More iterations = better solutions but slower.
# SELF-SET. For production: 10000-25000. For quick tests: 50-500.

SA_TEMP_INIT = 100.0
# Simulated Annealing initial temperature.
# Controls initial willingness to accept worse solutions.
# At T=100, a solution 100 units worse is accepted with prob exp(-1) ~ 37%.
# SELF-SET heuristic.

SA_COOLING = 0.9997
# Geometric cooling rate: T(k+1) = T(k) * SA_COOLING.
# After 1000 iterations: T ~ 100 * 0.9997^1000 ~ 74 (still warm).
# After 10000 iterations: T ~ 100 * 0.9997^10000 ~ 5 (cooling down).
# WARNING: Geometric cooling does NOT satisfy Hajek (1988) [ref 4] conditions
# for guaranteed convergence to global optimum (requires logarithmic cooling:
# T_k = Gamma/ln(k), which is impractically slow). This is standard practice
# in applied OR -- justification is empirical, not theoretical.

DESTROY_MIN_FRAC = 0.1
DESTROY_MAX_FRAC = 0.4
# Each destroy operator removes between 10% and 40% of all customers.
# Small destruction = safe but slow exploration.
# Large destruction = aggressive jumps, may find better solutions or waste time.
# SELF-SET. Ropke & Pisinger [ref 3] used similar range.

SEGMENT_SIZE = 100
# Update adaptive operator weights every this many iterations.
# SELF-SET. From Ropke & Pisinger [ref 3].

# Operator scoring (from Ropke & Pisinger 2006 [ref 3]) 

SIGMA_1 = 33   # Score bonus when operator produces a new GLOBAL BEST solution
SIGMA_2 = 9    # Score bonus when operator improves the CURRENT solution
SIGMA_3 = 13   # Score bonus when operator's result is ACCEPTED by SA (but worse)
# These values are taken directly from [ref 3] Table 1.

REACTION_FACTOR = 0.8
# Smoothing factor for adaptive weight update:
#   w_new = REACTION_FACTOR * w_old + (1 - REACTION_FACTOR) * (score/count)
# DEVIATION FROM LITERATURE: Ropke & Pisinger (2006) use r=0.1 (equivalent 
# to REACTION_FACTOR=0.9 here). We deliberately use 0.8 (equivalent to r=0.2) 
# to make the ALNS react twice as fast to the volatile DRO landscape, escaping
# local optima more efficiently without losing historical stability. 
# However, still a heurestic calibration



SEED = 42


# 2. DATA STRUCTURES

class Node(NamedTuple):
    """
    Represents one location (depot or customer) in the problem.

    Fields:
      id        - Node index (0 = depot, 1..n = customers)
      x, y      - 2D Euclidean coordinates
      delivery  - D_i: quantity the vehicle must DELIVER to this customer
      pickup    - P_i: quantity the vehicle must PICK UP from this customer
                  (= 0 in deterministic mode, since original CVRP has no pickup)
      sigma_sq  - Var(X_i) where X_i = P_i - D_i is the "net demand" at node i.
                  Represents how much this customer's demand fluctuates.
                  (= 0 in deterministic mode)
    """
    id: int
    x: float
    y: float
    delivery: float
    pickup: float
    sigma_sq: float


class ProblemInstance:
    """
    Complete problem data for one VRPSPD instance.

    Fields:
      name          - Instance identifier (e.g., "X-n101-k25")
      nodes         - List of Node objects. nodes[0] = depot, nodes[1..n] = customers.
      n             - Number of customers (excluding depot)
      capacity      - Maximum vehicle load C (possibly inflated in stochastic mode)
      dist_matrix   - (n+1) x (n+1) matrix of Euclidean distances between all nodes
      cov_matrix    - (n+1) x (n+1) spatial covariance matrix Sigma_ij
                      (all zeros in deterministic mode)
      theta         - Correlation length parameter for the exponential decay kernel
      is_stochastic - Whether this instance uses stochastic demand
    """
    def __init__(self, name: str, nodes: List[Node], capacity: float,
                 dist_matrix: np.ndarray, cov_matrix: np.ndarray,
                 theta: float, is_stochastic: bool):
        self.name = name
        self.nodes = nodes
        self.n = len(nodes) - 1
        self.capacity = capacity
        self.dist_matrix = dist_matrix
        self.cov_matrix = cov_matrix
        self.theta = theta
        self.is_stochastic = is_stochastic


# 3. FILE PARSERS

def parse_vrp_file(filepath: str) -> Tuple[str, List[Tuple[float, float]],
                                            List[float], float]:
    """
    Parse a CVRPLIB .vrp file (works with Set X, A, B, P formats).

    Returns:
      name      - Instance name from the file header
      coords    - List of (x, y) coordinates; coords[0] = depot
      demands   - Delivery demand for each node; demands[0] = 0 (depot)
      capacity  - ORIGINAL vehicle capacity from file (no scaling applied here;
                  scaling for stochastic mode is done in augment_instance())
    """
    name = os.path.splitext(os.path.basename(filepath))[0]
    coords: List[Tuple[float, float]] = []
    demands: List[float] = []
    capacity = 0.0
    section = None

    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("NAME"):
                name = line.split(":")[-1].strip()
            elif line.startswith("CAPACITY"):
                capacity = float(line.split(":")[-1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                section = "COORD"
            elif line.startswith("DEMAND_SECTION"):
                section = "DEMAND"
            elif line.startswith("DEPOT_SECTION"):
                section = "DEPOT"
            elif line == "EOF":
                section = None
            else:
                if section == "COORD":
                    parts = line.split()
                    if len(parts) >= 3:
                        coords.append((float(parts[1]), float(parts[2])))
                elif section == "DEMAND":
                    parts = line.split()
                    if len(parts) >= 2:
                        demands.append(float(parts[1]))
                elif section == "DEPOT":
                    if line.strip() == "-1":
                        section = None

    if not coords or not demands:
        raise ValueError(f"Failed to parse {filepath}: empty coords or demands")

    return name, coords, demands, capacity


def parse_bks_solution(sol_filepath: str) -> List[List[int]]:
    """
    Parse a .sol file containing Best Known Solution (BKS) routes.
    Expected format: "Route #k: 3 7 1 5 ..." (one route per line).
    Returns empty list if file doesn't exist.
    """
    bks_routes: List[List[int]] = []
    if not os.path.exists(sol_filepath):
        return []
    with open(sol_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("route"):
                parts = line.split(":")
                if len(parts) == 2:
                    nodes_str = parts[1].strip().split()
                    route = [int(n) for n in nodes_str]
                    if route:
                        bks_routes.append(route)
    return bks_routes


# 4. DATA AUGMENTATION: CVRP -> Stochastic VRPSPD

def augment_instance(name: str, coords: List[Tuple[float, float]],
                     demands: List[float], capacity_raw: float,
                     rng: np.random.Generator,
                     stochastic: bool) -> ProblemInstance:
    """
    Build a ProblemInstance from raw CVRP data.

    DETERMINISTIC mode:
      P_i = 0, sigma_sq = 0, capacity = original. Pure delivery problem.

    STOCHASTIC mode:
      1. Pickup demand:  P_i = D_i * Uniform(0.5, 1.5)
         SELF-SET augmentation rule to create VRPSPD from delivery-only data.

      2. Independent variance:  sigma_sq_i = (CV * (P_i + D_i))^2
         Models individual demand fluctuation proportional to total volume.

      3. Spatial covariance:  Sigma_ij = sigma_i * sigma_j * exp(-d_ij / theta)
         Exponential decay kernel from geostatistics (Kriging / Gaussian Processes).
         Nearby customers have correlated demand (e.g., regional weather effects).

         This kernel produces a valid (PSD) covariance matrix because:
           - exp(-||x-y||) is a positive definite kernel on Euclidean space
             [Schoenberg 1938, ref 2]
           - Sigma = diag(sigma) * K * diag(sigma) where K_ij = exp(-d_ij/theta)
             is the Hadamard product of two PSD matrices -> PSD
             [Schur product theorem, ref 6]
           - IMPORTANT: This proof ONLY holds when d_ij is Euclidean distance.
             If d_ij were road-network shortest-path distance, the kernel may NOT
             be PSD (general graphs don't embed isometrically into Hilbert space).

      4. Capacity:  C = capacity_raw * CAPACITY_FACTOR
         Relaxed to compensate for the pickup load that doesn't exist in CVRP.
    """
    n_total = len(coords)

    # --- Euclidean distance matrix from 2D coordinates ---
    # d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    coord_arr = np.array(coords)
    diff = coord_arr[:, np.newaxis, :] - coord_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    # --- Correlation length: theta = fraction of the map diameter ---
    # At distance = theta, correlation = exp(-1) ~ 0.37
    max_dist = np.max(dist_matrix)
    theta = THETA_FRACTION * max_dist if max_dist > 0 else 1.0

    # --- Capacity: relax for stochastic, keep original for deterministic ---
    capacity = capacity_raw * CAPACITY_FACTOR if stochastic else capacity_raw

    # --- Build node list ---
    nodes: List[Node] = []
    for i in range(n_total):
        d_i = demands[i]
        if i == 0:
            # Depot: no demand by definition
            nodes.append(Node(id=0, x=coords[0][0], y=coords[0][1],
                              delivery=0.0, pickup=0.0, sigma_sq=0.0))
        else:
            if stochastic:
                # Random pickup demand (SELF-SET augmentation)
                p_i = d_i * rng.uniform(0.5, 1.5)
                # Variance of net demand X_i = P_i - D_i
                # sigma_sq = (CV * total_volume)^2 where total_volume = P_i + D_i
                sigma_sq = (CV * (p_i + d_i)) ** 2
            else:
                p_i = 0.0      # No pickup in deterministic CVRP
                sigma_sq = 0.0  # No uncertainty
            nodes.append(Node(id=i, x=coords[i][0], y=coords[i][1],
                              delivery=d_i, pickup=p_i, sigma_sq=sigma_sq))

    # --- Spatial covariance matrix ---
    # Sigma_ij = sigma_i * sigma_j * exp(-d_ij / theta)
    # In deterministic mode: all sigma_i = 0, so cov_matrix is all zeros.
    sigmas = np.array([math.sqrt(n.sigma_sq) for n in nodes])
    cov_matrix = np.outer(sigmas, sigmas) * np.exp(-dist_matrix / theta)

    return ProblemInstance(name, nodes, capacity, dist_matrix, cov_matrix,
                          theta, stochastic)


# 5. ROUTE EVALUATOR: Distance + DRO Risk Penalty


def evaluate_route_dro(route: List[int], inst: ProblemInstance) -> Tuple[float, float, float]:
    """
    Evaluate a single route. Returns (distance, RRI, Z_cost).

    DETERMINISTIC: Hard capacity check. If any stop exceeds C -> Z = infinity.
                   Otherwise Z = distance.

    STOCHASTIC: Uses Moment-Based Distributionally Robust Optimization (M-DRO).

     LOAD MODEL (both modes) 
    The vehicle leaves the depot carrying ALL delivery goods for this route:
      L_0 = sum(D_i for i in route)

    At each customer k, the vehicle:
      - Drops off D_k (load decreases)
      - Picks up P_k (load increases)
      Net change: X_k = P_k - D_k

    Load after visiting customer k:
      L_k = L_0 + sum_{i=1}^{k} X_i = L_0 + S_k

    Constraint: L_k <= C at every stop k = 1, ..., m.

     M-DRO EVALUATION (stochastic only) 
    Since X_i is random, S_k is random, so we can't check L_k <= C exactly.
    Instead we bound the worst-case probability of violation:

    1. Cumulative mean:  mu_k = L_0 + sum_{i=1}^{k} E[X_i]
       (expected load at stop k)

    2. Cumulative variance:  var_k = Var(S_k) = sum of individual variances
       + 2 * sum of pairwise covariances for all visited customers up to k.
       Computed incrementally using the recurrence:
         var_k = var_{k-1} + sigma^2_{node_k} + 2 * sum_{j<k} Sigma_{route[j], node_k}
       This follows from Var(A+B) = Var(A) + Var(B) + 2*Cov(A,B).

    3. Cantelli-Chebyshev bound [Cantelli 1928, ref 1]:
       For ANY random variable S with known mean mu and variance sigma^2:
         max_{all distributions} P(S >= mu + a) = sigma^2 / (sigma^2 + a^2)
       where a = C - mu_k is the "slack" (remaining capacity margin).
       This bound is TIGHT: no better bound exists using only 2 moments [ref 5].

    4. Route Risk Index (RRI) = sum of P_fail_k over all stops.
       This uses the UNION BOUND (Boole's inequality):
         P(any stop overloaded) <= sum P(stop k overloaded)
       WARNING: Union bound OVERESTIMATES true failure probability, especially
       when demands are positively correlated (which they are here).
       RRI is used as a RANKING SURROGATE, not a probability estimate.

    5. Objective:  Z = distance + penalty * max(0, RRI - threshold)
       where penalty and threshold are calibration heuristics (see constants).
    """
    if not route:
        return 0.0, 0.0, 0.0

    C = inst.capacity
    m = len(route)

    # --- Route distance: depot -> route[0] -> route[1] -> ... -> depot ---
    dist = inst.dist_matrix[0, route[0]]
    for k in range(1, m):
        dist += inst.dist_matrix[route[k - 1], route[k]]
    dist += inst.dist_matrix[route[-1], 0]

    # DETERMINISTIC MODE: hard capacity check, no probabilistic evaluation
    if not inst.is_stochastic:
        load = sum(inst.nodes[n].delivery for n in route)  # L_0
        if load > C:
            return dist, 0.0, float('inf')  # Overloaded before leaving depot
        for n in route:
            load += inst.nodes[n].pickup - inst.nodes[n].delivery  # L_k update
            if load > C:
                return dist, 0.0, float('inf')  # Overloaded at customer n
        return dist, 0.0, dist  # Feasible: Z = distance, no penalty

    # STOCHASTIC MODE: M-DRO with prefix-sum covariance tracking

    # L_0 = sum of all deliveries (truck leaves depot carrying all goods to deliver)
    total_delivery = sum(inst.nodes[n].delivery for n in route)

    # HARD CONSTRAINT: if delivery alone exceeds capacity, the vehicle is
    # overloaded before it even leaves the depot. No risk tolerance changes this.
    if total_delivery > C:
        return dist, 1.0, float('inf')

    # mu_k tracks E[L_k] = expected cumulative load at stop k
    cum_mean = total_delivery  # starts at L_0

    # var_k tracks Var(S_k) = variance of cumulative net demand
    cum_var = 0.0

    # RRI = sum of worst-case failure probabilities (union bound)
    rri = 0.0

    #  Prefix-sum cross-covariance vector (the key computational trick) 
    #
    # To compute var_k we need: sum_{j<k} Sigma_{route[j], route[k]}
    # (covariance of the new customer with ALL previously visited customers).
    #
    # Naive approach: loop over all j < k -> O(k) per step -> O(m^2) total.
    #
    # Trick: maintain a vector v of size (n+1) where:
    #   v[u] = sum_{j visited so far} Sigma_{route[j], u}
    # Then sum_{j<k} Sigma_{route[j], route[k]} = v[route[k]]  -> O(1) lookup.
    # After processing node k, update: v += Sigma[route[k], :]  -> O(n) vector add.
    #
    # Total complexity: O(m*n) theoretically, but the O(n) vector add runs in
    # near-constant time due to NumPy SIMD vectorization.
    # The information-theoretic lower bound is Omega(m^2) since we must access
    # m*(m-1)/2 covariance pairs.
    cov_prefix_vector = np.zeros(inst.n + 1)

    for k in range(m):
        node_k = route[k]

        # Update expected load 
        # L_k = L_{k-1} + (P_k - D_k)
        # P_k - D_k > 0 means the truck gets heavier (picked up more than delivered)
        cum_mean += inst.nodes[node_k].pickup - inst.nodes[node_k].delivery

        # Update cumulative variance (recurrence) 
        # Mathematical derivation (from Var(A+B) = Var(A) + Var(B) + 2*Cov(A,B)):
        #   Var(S_k) = Var(S_{k-1} + X_k)
        #            = Var(S_{k-1}) + Var(X_k) + 2*Cov(S_{k-1}, X_k)
        #   where Cov(S_{k-1}, X_k) = sum_{j=1}^{k-1} Cov(X_j, X_k)
        #                           = sum_{j=1}^{k-1} Sigma_{route[j], node_k}
        #                           = cov_prefix_vector[node_k]  (pre-accumulated)
        if k == 0:
            cum_var = inst.nodes[node_k].sigma_sq  # Var(X_1) = sigma^2_1
        else:
            cross_cov_sum = cov_prefix_vector[node_k]  # O(1) lookup
            cum_var += inst.nodes[node_k].sigma_sq + 2.0 * cross_cov_sum

        # Accumulate this node's covariance contributions for future steps
        cov_prefix_vector += inst.cov_matrix[node_k]  # v += row of Sigma matrix

        #  Cantelli-Chebyshev bound [Cantelli 1928, ref 1] 
        # P_fail = max over ALL distributions with mean=cum_mean, var=cum_var
        #          of P(load > C)
        #        = cum_var / (cum_var + slack^2)    when slack > 0
        #        = 1.0                              when slack <= 0
        # where slack = C - cum_mean = remaining capacity margin.
        # This bound is TIGHT in the 2-moment ambiguity set [Delage & Ye, ref 5].
        slack = C - cum_mean
        if slack <= 0:
            # Expected load already >= capacity -> worst case = certain failure
            p_fail = 1.0
        elif cum_var <= 0:
            # Zero variance -> deterministic demand -> no overload possible
            p_fail = 0.0
        else:
            p_fail = cum_var / (cum_var + slack * slack)

        # Accumulate into RRI (union bound [Boole's inequality] over all stops)
        rri += p_fail

    #  Objective function 
    # Z = distance + [lambda_0 / ln(m+1)] * max(0, RRI - alpha_base * m^gamma)
    #
    # alpha(m) = ALPHA_BASE * m^GAMMA:
    #   Dynamic risk threshold that grows with route length.
    #   Longer routes accumulate more RRI terms, so threshold must grow
    #   to avoid the solver fragmenting routes into tiny single-customer trips.
    #   SELF-SET heuristic. Not derived from any theorem (despite superficial
    #   resemblance to CLT scaling -- CLT does not apply here).
    #
    # lambda(m) = LAMBDA_0 / ln(m+1):
    #   Penalty strength that decreases with route length for the same reason.
    #   SELF-SET heuristic.
    alpha_m = ALPHA_BASE * (m ** GAMMA)
    penalty_term = max(0.0, rri - alpha_m)
    z_penalty = (LAMBDA_0 / math.log(m + 1)) * penalty_term
    z_cost = dist + z_penalty

    return dist, rri, z_cost


def evaluate_solution(routes: List[List[int]],
                      inst: ProblemInstance) -> Tuple[float, float, float]:
    """Sum (distance, RRI, Z) over all routes in the solution."""
    if not routes:
        return 0.0, 0.0, 0.0
    total_dist = 0.0
    total_rri = 0.0
    total_z = 0.0
    for route in routes:
        if not route:
            continue
        d, r, z = evaluate_route_dro(route, inst)
        total_dist += d
        total_rri += r
        total_z += z
    return total_dist, total_rri, total_z


# 6. INITIAL SOLUTION: Greedy Nearest-Neighbor

def build_initial_solution(inst: ProblemInstance) -> List[List[int]]:
    """
    Build a feasible starting solution using greedy nearest-neighbor heuristic.

    At each step, add the nearest unvisited customer that doesn't violate
    the delivery capacity constraint. When no more customers fit, start a
    new route (new vehicle).

    Uses only delivery demand for capacity check (conservative in stochastic
    mode, since pickup amounts are uncertain and may add or reduce load).
    """
    unvisited = set(range(1, inst.n + 1))
    routes: List[List[int]] = []
    C = inst.capacity

    while unvisited:
        route: List[int] = []
        cum_delivery = 0.0   # running total of delivery demand loaded
        current = 0           # current position (start at depot, node 0)

        while unvisited:
            best_next = -1
            best_dist = float('inf')

            for cand in unvisited:
                # Conservative capacity check: only delivery demand counted
                if cum_delivery + inst.nodes[cand].delivery > C:
                    continue
                dd = inst.dist_matrix[current, cand]
                if dd < best_dist:
                    best_dist = dd
                    best_next = cand

            if best_next == -1:
                break  # No feasible candidate -> close this route

            route.append(best_next)
            cum_delivery += inst.nodes[best_next].delivery
            current = best_next
            unvisited.discard(best_next)

        if route:
            routes.append(route)
        elif unvisited:
            # Edge case: a single customer's delivery exceeds C
            # Force it into its own route (will be flagged as infeasible by evaluator)
            forced = unvisited.pop()
            routes.append([forced])

    return routes


# 7. ALNS DESTROY OPERATORS

def random_removal(routes: List[List[int]], inst: ProblemInstance,
                   rng: np.random.Generator) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a random subset of customers from the solution.
    The number removed is uniform between DESTROY_MIN_FRAC and DESTROY_MAX_FRAC
    of all customers currently in routes.
    """
    all_cust = [c for r in routes for c in r]
    if not all_cust:
        return routes, []
    n_rem = max(1, int(len(all_cust) * rng.uniform(DESTROY_MIN_FRAC, DESTROY_MAX_FRAC)))
    n_rem = min(n_rem, len(all_cust))

    removed = list(rng.choice(all_cust, size=n_rem, replace=False))
    rem_set = set(removed)

    new_routes = []
    for r in routes:
        new_r = [c for c in r if c not in rem_set]
        if new_r:
            new_routes.append(new_r)

    return new_routes, removed


def worst_removal(routes: List[List[int]], inst: ProblemInstance,
                  rng: np.random.Generator) -> Tuple[List[List[int]], List[int]]:
    """
    Remove customers whose removal saves the most cost (they are "worst positioned").

    Uses randomized selection with determinism factor p=3 (from Shaw 1997):
      index = len(candidates) * random()^3
    This biases toward the top of the sorted list but adds diversification.
    """
    DETERMINISM = 3  # Higher = more deterministic (always pick worst)
    all_cust = [c for r in routes for c in r]
    if not all_cust:
        return routes, []
    n_rem = max(1, int(len(all_cust) * rng.uniform(DESTROY_MIN_FRAC, DESTROY_MAX_FRAC)))
    n_rem = min(n_rem, len(all_cust))

    curr_routes = [list(r) for r in routes]
    removed: List[int] = []

    for _ in range(n_rem):
        costs: List[Tuple[float, int]] = []
        for r in curr_routes:
            for ci, cust in enumerate(r):
                trial = r[:ci] + r[ci + 1:]
                z_old = evaluate_route_dro(r, inst)[2]
                z_new = evaluate_route_dro(trial, inst)[2] if trial else 0.0
                savings = z_old - z_new  # Positive = removing this customer is beneficial
                costs.append((savings, cust))
        if not costs:
            break
        costs.sort(key=lambda x: -x[0])  # Sort by savings, descending
        # Randomized pick: biased toward top (worst customers)
        idx = min(int(len(costs) * (rng.random() ** DETERMINISM)), len(costs) - 1)
        chosen = costs[idx][1]

        removed.append(chosen)
        for r in curr_routes:
            if chosen in r:
                r.remove(chosen)
                break
        curr_routes = [r for r in curr_routes if r]  # Remove now-empty routes

    return curr_routes, removed


# 8. ALNS REPAIR OPERATORS

def _best_insertion_cost(route: List[int], customer: int,
                         inst: ProblemInstance) -> Tuple[float, int]:
    """
    Find the best position to insert 'customer' into 'route'.
    Tries every possible position and returns (cost_increase, best_position).
    """
    z_before = evaluate_route_dro(route, inst)[2] if route else 0.0
    best_delta = float('inf')
    best_pos = 0

    for pos in range(len(route) + 1):
        new_route = route[:pos] + [customer] + route[pos:]
        z_after = evaluate_route_dro(new_route, inst)[2]
        delta = z_after - z_before
        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    return best_delta, best_pos


def greedy_insertion(routes: List[List[int]], removed: List[int],
                     inst: ProblemInstance,
                     rng: np.random.Generator) -> List[List[int]]:
    """
    Greedy insertion: for each removed customer (in random order), insert it
    at the position with the smallest cost increase across all existing routes.
    If opening a new route is cheaper, do that instead.
    """
    rng.shuffle(removed)
    curr = [list(r) for r in routes]

    for cust in removed:
        best_delta = float('inf')
        best_ri = -1
        best_pos = 0

        for ri, r in enumerate(curr):
            delta, pos = _best_insertion_cost(r, cust, inst)
            if delta < best_delta:
                best_delta = delta
                best_ri = ri
                best_pos = pos

        # Compare with starting a brand-new route for this customer
        new_delta, _ = _best_insertion_cost([], cust, inst)
        if new_delta < best_delta or best_ri == -1:
            curr.append([cust])
        else:
            curr[best_ri].insert(best_pos, cust)

    return curr


def regret2_insertion(routes: List[List[int]], removed: List[int],
                      inst: ProblemInstance,
                      rng: np.random.Generator) -> List[List[int]]:
    """
    Regret-2 insertion (Ropke & Pisinger [ref 3]):
    Prioritize customers where the difference between the BEST and SECOND-BEST
    insertion cost is largest. These customers "regret" not being inserted now
    the most -- if we wait, their good positions may be taken.
    Generally produces better solutions than greedy insertion.
    """
    curr = [list(r) for r in routes]
    pool = list(removed)

    while pool:
        regret_list: List[Tuple[float, int, int, int]] = []

        for cust in pool:
            ins_costs: List[Tuple[float, int, int]] = []
            for ri, r in enumerate(curr):
                delta, pos = _best_insertion_cost(r, cust, inst)
                ins_costs.append((delta, ri, pos))
            # Also consider a brand-new route
            new_delta, _ = _best_insertion_cost([], cust, inst)
            ins_costs.append((new_delta, -1, 0))

            ins_costs.sort(key=lambda x: x[0])
            best_cost = ins_costs[0][0]
            second_cost = ins_costs[1][0] if len(ins_costs) > 1 else best_cost
            regret = second_cost - best_cost  # How much worse the 2nd-best is
            _, best_ri, best_pos = ins_costs[0]
            regret_list.append((regret, cust, best_ri, best_pos))

        # Insert the customer with highest regret first
        regret_list.sort(key=lambda x: -x[0])
        _, chosen, ri, pos = regret_list[0]

        if ri == -1:
            curr.append([chosen])
        else:
            curr[ri].insert(pos, chosen)

        pool.remove(chosen)

    return curr


# 9. ALNS SOLVER with Simulated Annealing
def alns_solve(inst: ProblemInstance, rng: np.random.Generator,
               max_iter: int = ALNS_ITERATIONS) -> Tuple[List[List[int]], float, float]:
    """
    Adaptive Large Neighborhood Search (ALNS) with Simulated Annealing (SA).

    Algorithm (each iteration):
      1. SELECT destroy + repair operators via roulette wheel (adaptive weights)
      2. DESTROY: remove a fraction of customers from the current solution
      3. REPAIR: reinsert removed customers at good positions
      4. ACCEPT/REJECT using SA criterion:
         - Always accept improvements
         - Accept worse solutions with probability exp(-delta/T)
           (allows escaping local optima early on when T is high)
      5. UPDATE operator weights based on performance scores [ref 3]

    Cooling schedule: geometric T(k+1) = T(k) * SA_COOLING, with reheat when
    T drops below 0.1 (jump to 30% of initial temperature).

    THEORETICAL NOTE: Geometric cooling does NOT satisfy the conditions in
    Hajek (1988) [ref 4] for guaranteed convergence to global optimum.
    Hajek requires logarithmic cooling T_k = Gamma/ln(k+k0), which needs
    exponentially many iterations. Geometric cooling is standard practice in
    applied OR -- its justification is empirical, not theoretical.

    Returns: (best_routes, best_distance, best_Z_cost)
    """
    # Two destroy operators, two repair operators
    d_ops = [random_removal, worst_removal]
    r_ops = [greedy_insertion, regret2_insertion]
    nd, nr = len(d_ops), len(r_ops)

    # Adaptive weights (start uniform at 1.0 each)
    d_w = np.ones(nd)   # destroy operator weights
    r_w = np.ones(nr)   # repair operator weights
    d_s = np.zeros(nd)   # accumulated scores this segment
    r_s = np.zeros(nr)
    d_c = np.zeros(nd)   # usage counts this segment
    r_c = np.zeros(nr)

    # Initialize with greedy nearest-neighbor solution
    curr = build_initial_solution(inst)
    curr_dist, _, curr_z = evaluate_solution(curr, inst)
    best = copy.deepcopy(curr)
    best_dist = curr_dist
    best_z = curr_z
    temp = SA_TEMP_INIT

    for it in range(1, max_iter + 1):
        # --- Operator selection: roulette wheel proportional to weights ---
        d_idx = rng.choice(nd, p=d_w / d_w.sum())
        r_idx = rng.choice(nr, p=r_w / r_w.sum())
        d_c[d_idx] += 1
        r_c[r_idx] += 1

        # --- Destroy then repair ---
        partial, removed = d_ops[d_idx](curr, inst, rng)
        cand = r_ops[r_idx](partial, removed, inst, rng)
        cand = [r for r in cand if r]  # Remove any empty routes

        cand_dist, _, cand_z = evaluate_solution(cand, inst)

        #  SA acceptance criterion 
        # Boltzmann acceptance: P(accept) = exp(-delta/T)
        # When T is high, almost anything is accepted (exploration).
        # When T is low, only improvements pass (exploitation).
        delta = cand_z - curr_z
        accepted = False

        if delta < 0:
            # Improvement: always accept
            accepted = True
            if cand_z < best_z:
                # New global best
                best = copy.deepcopy(cand)
                best_dist = cand_dist
                best_z = cand_z
                d_s[d_idx] += SIGMA_1  # Highest reward [ref 3]
                r_s[r_idx] += SIGMA_1
            else:
                d_s[d_idx] += SIGMA_2  # Improved current but not global best
                r_s[r_idx] += SIGMA_2
        elif temp > 1e-10:
            try:
                prob = math.exp(-delta / temp)
            except OverflowError:
                prob = 0.0
            if rng.random() < prob:
                # Accepted a worse solution (diversification)
                accepted = True
                d_s[d_idx] += SIGMA_3
                r_s[r_idx] += SIGMA_3

        if accepted:
            curr = cand
            curr_dist = cand_dist
            curr_z = cand_z

        # Geometric cooling + reheat 
        temp *= SA_COOLING
        if temp < 0.1:
            # Reheat: temperature too low -> solver "frozen" at local optimum.
            # Jump to 30% of initial temperature. SELF-SET heuristic.
            # Not a standard SA technique; breaks homogeneous Markov chain
            # properties but helps escape in practice.
            temp = SA_TEMP_INIT * 0.3

        #  Adaptive weight update (every SEGMENT_SIZE iterations)
        # Formula from Ropke & Pisinger [ref 3]:
        #   w_new = rho * w_old + (1 - rho) * (score / count)
        # rho = REACTION_FACTOR controls how much history matters.
        if it % SEGMENT_SIZE == 0:
            for i in range(nd):
                if d_c[i] > 0:
                    d_w[i] = max(0.01, REACTION_FACTOR * d_w[i]
                                + (1 - REACTION_FACTOR) * (d_s[i] / d_c[i]))
            for i in range(nr):
                if r_c[i] > 0:
                    r_w[i] = max(0.01, REACTION_FACTOR * r_w[i]
                                + (1 - REACTION_FACTOR) * (r_s[i] / r_c[i]))
            # Reset scores and counts for next segment
            d_s.fill(0)
            r_s.fill(0)
            d_c.fill(0)
            r_c.fill(0)

    return best, best_dist, best_z


# 10. MONTE CARLO VALIDATION (MULTI-UNIVERSE COPULA)

def monte_carlo_validate(routes: List[List[int]], inst: ProblemInstance,
                         rng: np.random.Generator, n_samples: int = MC_SAMPLES) -> Dict[str, float]:
 
    results = {
        "GAUSSIAN": 0.0,
        "SKEW_RIGHT": 0.0,
        "SKEW_LEFT": 0.0,
        "HEAVY_TAIL": 0.0
    }
    
    if not inst.is_stochastic or not routes:
        return results

    total_sims = 0
    total_fails = {k: 0 for k in results.keys()}
    C = inst.capacity

    for route in routes:
        if not route: continue
        total_sims += 1
        m = len(route)

        mean_vec = np.array([inst.nodes[c].pickup - inst.nodes[c].delivery for c in route])
        route_arr = np.array(route, dtype=np.intp)
        sub_cov = inst.cov_matrix[np.ix_(route_arr, route_arr)]
        sub_cov = 0.5 * (sub_cov + sub_cov.T) # Bắt buộc đối xứng
        
        min_eig = np.min(np.linalg.eigvalsh(sub_cov))
        if min_eig < 1e-8:
            sub_cov += (1e-8 - min_eig) * np.eye(m)

        stds = np.sqrt(np.diag(sub_cov))

        outer_stds = np.outer(stds, stds)
        corr_matrix = np.divide(sub_cov, outer_stds, out=np.eye(m), where=outer_stds!=0)
        min_eig_corr = np.min(np.linalg.eigvalsh(corr_matrix))
        if min_eig_corr < 1e-8:
            corr_matrix += (1e-8 - min_eig_corr) * np.eye(m)

        Z = rng.multivariate_normal(np.zeros(m), corr_matrix, size=n_samples)
        U = norm.cdf(Z) # Uniform mang sẵn độ lây nhiễm không gian

        scenarios = {
            "GAUSSIAN": norm.ppf(U),
            "SKEW_RIGHT": (skewnorm.ppf(U, 5.0) - skewnorm.stats(5.0, moments='m')) / np.sqrt(skewnorm.stats(5.0, moments='v')),
            "SKEW_LEFT": (skewnorm.ppf(U, -5.0) - skewnorm.stats(-5.0, moments='m')) / np.sqrt(skewnorm.stats(-5.0, moments='v')),
            "HEAVY_TAIL": t.ppf(U, df=3) / np.sqrt(3 / (3 - 2)) 
        }

        total_delivery = sum(inst.nodes[c].delivery for c in route)
        
        for name, noise_std in scenarios.items():
            samples = mean_vec[np.newaxis, :] + noise_std * stds[np.newaxis, :]
            cum_loads = total_delivery + np.cumsum(samples, axis=1)
            total_fails[name] += np.sum(np.any(cum_loads > C, axis=1))

    if total_sims > 0:
        for k in results.keys():
            results[k] = total_fails[k] / (total_sims * n_samples)
            
    return results


# 11. BENCHMARK ENGINE

def run_benchmark(target_dir: str):
    """
    Main benchmark loop:
      1. Find all .vrp files in target_dir
      2. For each: parse, augment, evaluate BKS, run ALNS, MC validate
      3. Print results table and export CSV

    In DETERMINISTIC mode: reports distance gap vs BKS (meaningful comparison).
    In STOCHASTIC mode: reports failure rates (gap vs BKS is N/A because
    ALNS solves a different problem -- VRPSPD with risk penalty -- than the
    original CVRP that BKS was optimized for).
    """
    mode_str = "STOCHASTIC" if STOCHASTIC_MODE else "DETERMINISTIC"
    print(f"\n{'='*130}")
    print(f"  Mode: {mode_str}  |  CV={CV if STOCHASTIC_MODE else 0}  "
          f"|  Cap x{CAPACITY_FACTOR if STOCHASTIC_MODE else 1.0}  "
          f"|  ALNS={ALNS_ITERATIONS} iters  "
          f"|  MC={MC_SAMPLES if STOCHASTIC_MODE else 'SKIP'}")
    print(f"{'='*130}")

    vrp_files = sorted(glob.glob(os.path.join(target_dir, "*.vrp")))
    if not vrp_files:
        print(f"No .vrp files found in {target_dir}")
        return

    print(f"Found {len(vrp_files)} instances\n")

    # Print table header (different columns depending on mode)
    if STOCHASTIC_MODE:
        print(f"{'Instance':<15} | {'N':>4} | "
              f"{'BKS_V':>5} {'BKS_Dist':>10} {'BKS_Fail':>9} | "
              f"{'ALNS_V':>6} {'ALNS_Dist':>10} {'Time':>7} | "
              f"{'Fail_Rates (Norm/Right/Left/Tail)'}")
    else:
        print(f"{'Instance':<15} | {'N':>4} | "
              f"{'BKS_V':>5} {'BKS_Dist':>10} | "
              f"{'ALNS_V':>6} {'ALNS_Dist':>10} {'Time':>7} | "
              f"{'Gap%':>7}")
    print("-" * 130)

    results: List[Dict] = []

    for file_idx, v_path in enumerate(vrp_files):
        base_name = os.path.splitext(os.path.basename(v_path))[0]
        s_path = os.path.join(target_dir, f"{base_name}.sol")

        try:
            # --- Per-instance seed for independent reproducibility ---
            # Instance k always gets seed SEED+k, regardless of what other
            # files exist in the directory or what order they're processed.
            inst_seed = SEED + file_idx
            rng = np.random.default_rng(inst_seed)
            random.seed(inst_seed)

            # 1. Parse raw data and build problem instance
            name, coords, demands, capacity_raw = parse_vrp_file(v_path)
            inst = augment_instance(name, coords, demands, capacity_raw,
                                    rng, stochastic=STOCHASTIC_MODE)

            # 2. Evaluate Best Known Solution (if .sol file exists)
            # NOTE: In stochastic mode, BKS routes are evaluated on the
            # augmented instance (with pickup demands they weren't designed for).
            # BKS_Fail shows "what would happen if we used deterministic routes
            # in a stochastic world" -- a motivation for stochastic optimization.
            bks_routes = parse_bks_solution(s_path)
            bks_dist = 0.0
            bks_veh = len(bks_routes)
            bks_fail = 0.0

            if bks_routes:
                bks_dist = evaluate_solution(bks_routes, inst)[0]
                if STOCHASTIC_MODE:
                    bks_fail_dict = monte_carlo_validate(bks_routes, inst, rng, MC_SAMPLES)
                    bks_fail = bks_fail_dict['SKEW_RIGHT'] # Show worst-case skew for BKS

            # 3. Run ALNS solver
            t_start = time.perf_counter()
            best_routes, alns_dist, alns_z = alns_solve(inst, rng, ALNS_ITERATIONS)
            alns_time = time.perf_counter() - t_start
            alns_veh = len(best_routes)

            # 4. Monte Carlo validation
            alns_fail_dict = monte_carlo_validate(best_routes, inst, rng, MC_SAMPLES)

            # 5. Gap vs BKS
            # ONLY meaningful in deterministic mode: same problem, same capacity.
            # In stochastic mode: ALNS solves VRPSPD (with pickup + risk penalty),
            # BKS solves CVRP (delivery only) -> incomparable.
            if STOCHASTIC_MODE:
                gap_val = "N/A"
                gap_str = "  N/A  "
            else:
                gap_val = ((alns_dist - bks_dist) / bks_dist * 100) if bks_dist > 0 else 0.0
                gap_str = f"{gap_val:>6.2f}%"

            # 6. Print row
            if STOCHASTIC_MODE:
                fail_str = (f"N:{alns_fail_dict.get('GAUSSIAN', 0.0):.4f} "
                            f"R:{alns_fail_dict.get('SKEW_RIGHT', 0.0):.4f} "
                            f"L:{alns_fail_dict.get('SKEW_LEFT', 0.0):.4f} "
                            f"T:{alns_fail_dict.get('HEAVY_TAIL', 0.0):.4f}")
                print(f"{base_name:<15} | {inst.n:>4} | "
                      f"{bks_veh:>5} {bks_dist:>10.2f} {bks_fail:>9.4f} | "
                      f"{alns_veh:>6} {alns_dist:>10.2f} {alns_time:>7.2f} | {fail_str}")
            else:
                print(f"{base_name:<15} | {inst.n:>4} | "
                      f"{bks_veh:>5} {bks_dist:>10.2f} | "
                      f"{alns_veh:>6} {alns_dist:>10.2f} {alns_time:>7.2f} | "
                      f"{gap_str}")

            # 7. Collect result for CSV export
            row = {
                "Instance": base_name,
                "Mode": mode_str,
                "N": inst.n,
                "Capacity_Used": round(inst.capacity, 1),
                "CV": CV if STOCHASTIC_MODE else 0,
                "BKS_Vehicles": bks_veh,
                "BKS_Distance": round(bks_dist, 2),
                "ALNS_Vehicles": alns_veh,
                "ALNS_Distance": round(alns_dist, 2),
                "ALNS_Z_Cost": round(alns_z, 2),
                "Gap_vs_BKS_%": round(gap_val, 2) if not STOCHASTIC_MODE else gap_val,
                "Runtime_s": round(alns_time, 2),
                "Seed": inst_seed,
                "Fail_Gaussian": round(alns_fail_dict.get('GAUSSIAN', 0.0), 6) if STOCHASTIC_MODE else "N/A",
                "Fail_SkewRight": round(alns_fail_dict.get('SKEW_RIGHT', 0.0), 6) if STOCHASTIC_MODE else "N/A",
                "Fail_SkewLeft": round(alns_fail_dict.get('SKEW_LEFT', 0.0), 6) if STOCHASTIC_MODE else "N/A",
                "Fail_HeavyTail": round(alns_fail_dict.get('HEAVY_TAIL', 0.0), 6) if STOCHASTIC_MODE else "N/A",
            }
            results.append(row)

        except Exception as e:
            print(f"{base_name:<15} | ERROR: {e}")
            results.append({"Instance": base_name, "Error": str(e)})

    print("-" * 130)

    # --- Export CSV ---
    suffix = "stochastic" if STOCHASTIC_MODE else "deterministic"
    out_csv = os.path.join(target_dir, f"results_{suffix}.csv")

    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults exported: {out_csv}")

    # --- Summary statistics ---
    valid = [r for r in results if "Error" not in r]
    if valid:
        avg_time = np.mean([r["Runtime_s"] for r in valid])
        if STOCHASTIC_MODE:
            fail_rates = [r["Fail_SkewRight"] for r in valid
                          if isinstance(r.get("Fail_SkewRight"), (int, float))]
            avg_fail = np.mean(fail_rates) if fail_rates else 0.0
            print(f"   Avg SkewRight FailRate: {avg_fail:.4f}  |  Avg Time: {avg_time:.2f}s  "
                  f"|  Instances: {len(valid)}/{len(vrp_files)}")
        else:
            gaps = [r["Gap_vs_BKS_%"] for r in valid
                    if isinstance(r.get("Gap_vs_BKS_%"), (int, float))]
            avg_gap = np.mean(gaps) if gaps else 0.0
            print(f"   Avg Gap: {avg_gap:.2f}%  |  Avg Time: {avg_time:.2f}s  "
                  f"|  Instances: {len(valid)}/{len(vrp_files)}")


# 12. ENTRY POINT

if __name__ == "__main__":
    TARGET_DIR = r""

    # Fallback: check environment variable or current directory
    if not os.path.isdir(TARGET_DIR):
        env_dir = os.environ.get("VRP_DATA_DIR", ".")
        if os.path.isdir(env_dir):
            TARGET_DIR = env_dir
            print(f"[WARN] Using VRP_DATA_DIR={TARGET_DIR}")
        else:
            print(f"[ERROR] Directory not found: {TARGET_DIR}")
            print("  Set VRP_DATA_DIR environment variable or update TARGET_DIR.")
            exit(1)

    run_benchmark(TARGET_DIR)
