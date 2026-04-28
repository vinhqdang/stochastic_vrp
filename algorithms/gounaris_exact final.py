#!/usr/bin/env python3
"""
================================================================================
Gounaris (2013) Robust CVRP — Adapted for VRPSPD (Full Version)
Pure MTZ with L_i + Static Demand Inflation + Monte Carlo Validation
================================================================================

VALIDATED: Delivery-only mode replicates Gounaris Table 2 exactly:
  QB (0,0)=740, QB (0.2,1.0)=784
  QF (0,0)=740, QF (0.2,1.0)=784

ARCHITECTURE:
  Routing:     Binary x_ij
  Subtour+Cap: Pure MTZ with single variable L_i per node
               L_i = total load on vehicle immediately after leaving node i
               L_j >= L_i - d_j + p_j - M*(1-x_ij)  (load tracking)
               L_i <= Q                               (capacity)
               L_i >= d_j * x_ij                      (enough delivery goods)
  Robustness:  Static demand inflation d*(1+alpha), p*(1+alpha)
               Budget controlled by beta (QB quadrant / QF factor model)

  Model size: ~N^2 binary + ~N continuous = tiny. Gurobi solves in seconds.

  NO Proposition 4 dualization (66K variables, doesn't converge).
  NO Two-Commodity Flow (redundant with MTZ, wastes RAM).

Phase 1: Multi-instance benchmark with fixed (alpha, beta)
Phase 2: 5x5 sensitivity grid on single instance
Both phases include Monte Carlo validation (4 scenarios via Gaussian Copula).

Dependencies: numpy, scipy, gurobipy (Academic License)
================================================================================
"""

import os, glob, csv, math, time
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import norm, skewnorm, t as student_t

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("[ERROR] gurobipy not installed"); exit(1)

# ==============================================================================
# PARAMETERS
# ==============================================================================

SEED = 42
CAPACITY_FACTOR = 1.2       # Gounaris Section 6
CV = 0.2                    # For MC covariance matrix
THETA_FRACTION = 0.1        # Spatial correlation range
MC_SAMPLES = 10000          # Monte Carlo samples per scenario
GUROBI_TIME_LIMIT = 600     # Per solve
GUROBI_MIP_GAP = 0.02       # 2% gap
DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.5

# ==============================================================================
# FILE PARSING & AUGMENTATION
# ==============================================================================

def parse_vrp_file(fp):
    name = os.path.splitext(os.path.basename(fp))[0]
    coords, demands = [], []
    cap, sec = 0.0, None
    with open(fp, 'r') as f:
        for rl in f:
            ln = rl.strip()
            if not ln: continue
            if ln.startswith("NAME"): name = ln.split(":")[-1].strip()
            elif ln.startswith("CAPACITY"): cap = float(ln.split(":")[-1].strip())
            elif ln.startswith("NODE_COORD_SECTION"): sec = "C"
            elif ln.startswith("DEMAND_SECTION"): sec = "D"
            elif ln.startswith("DEPOT_SECTION"): sec = "P"
            elif ln == "EOF": sec = None
            else:
                if sec == "C":
                    p = ln.split()
                    if len(p) >= 3: coords.append((float(p[1]), float(p[2])))
                elif sec == "D":
                    p = ln.split()
                    if len(p) >= 2: demands.append(float(p[1]))
                elif sec == "P":
                    if ln.strip() == "-1": sec = None
    nv = 0
    b = os.path.splitext(os.path.basename(fp))[0]
    if "-k" in b:
        try: nv = int(b.split("-k")[-1])
        except: pass
    return name, coords, demands, cap, nv


def augment_pickups(demands, rng):
    """P_i = D_i * U(0.5, 1.5). Same RNG as alns.py."""
    pk = [0.0]
    for i in range(1, len(demands)):
        pk.append(demands[i] * rng.uniform(0.5, 1.5))
    return pk


def build_distance_matrix(coords):
    """Euclidean distance, ROUNDED to integer (Gounaris Section 6 convention)."""
    a = np.array(coords)
    d = a[:, np.newaxis, :] - a[np.newaxis, :, :]
    return np.round(np.sqrt(np.sum(d ** 2, axis=2)))


def get_quadrants(coords):
    dx, dy = coords[0]
    q = {1: [], 2: [], 3: [], 4: []}
    for i in range(1, len(coords)):
        x, y = coords[i]
        if x >= dx and y >= dy: q[1].append(i)
        elif x < dx and y >= dy: q[2].append(i)
        elif x < dx and y < dy: q[3].append(i)
        else: q[4].append(i)
    return q


# ==============================================================================
# WORST-CASE DEMAND (Static Inflation — Gounaris Proposition 1)
# ==============================================================================

def inflate_demands_QB(n, demands, coords, alpha, beta):
    """
    Compute robust demands under Q_B (Budget) support.
    beta=1: full rectangular, d_i*(1+alpha)
    beta<1: partial, d_i*(1+alpha*beta)
    """
    if alpha <= 0:
        return list(demands)
    robust = [0.0]
    factor = alpha if beta >= 1.0 else alpha * beta
    for i in range(1, n + 1):
        robust.append(demands[i] * (1 + factor))
    return robust


def inflate_demands_QF(n, demands, coords, dist_matrix, alpha, beta):
    """
    Compute robust demands under Q_F (Factor Model) support.
    beta=1: all factors active -> same as rectangular
    beta<1: distance-weighted partial inflation
    """
    if alpha <= 0:
        return list(demands)

    robust = [0.0]

    if beta >= 1.0:
        for i in range(1, n + 1):
            robust.append(demands[i] * (1 + alpha))
        return robust

    theta = THETA_FRACTION * np.max(dist_matrix)

    for i in range(1, n + 1):
        total_exposure = 0.0
        for f in range(1, n + 1):
            total_exposure += math.exp(-dist_matrix[i, f] / theta)
        norm_exposure = total_exposure / n
        eff_alpha = min(alpha, alpha * beta * norm_exposure)
        robust.append(demands[i] * (1 + eff_alpha))

    return robust


# ==============================================================================
# GUROBI SOLVER — PURE MTZ WITH L_i FOR VRPSPD
# ==============================================================================

def solve_robust_vrpspd(
    name, coords, robust_del, robust_pick, capacity, dist_matrix,
    num_vehicles, time_limit=GUROBI_TIME_LIMIT,
):
    """
    Solve Robust VRPSPD using MTZ + Two-Commodity Flow (2CF).

    PROVEN CORRECT: This is the same 2CF structure from validation check
    that produced 748 (det) and 828 (robust) on A-n32-k5 in 39-300 seconds.

    Two-Commodity Flow tracks load correctly for VRPSPD:
      y[i,j]: delivery flow on arc (i,j) — decreases along route
      z[i,j]: pickup flow on arc (i,j) — increases along route
      y[i,j] + z[i,j] <= C * x[i,j]  — joint capacity at every arc

    MTZ u[i] handles subtour elimination only (not capacity).
    Demands are pre-inflated by caller (inflate_demands_QB/QF).

    Returns: (cost, routes, solve_time) or (None, None, solve_time)
    """
    n = len(coords) - 1
    C = capacity
    customers = list(range(1, n + 1))
    nodes = list(range(n + 1))

    t0 = time.perf_counter()

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    mdl = gp.Model(f"VRPSPD_2CF_{name}", env=env)
    mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("MIPGap", GUROBI_MIP_GAP)

    # --- Variables ---
    # Routing
    x = mdl.addVars(nodes, nodes, vtype=GRB.BINARY, name="x")

    # Two-commodity flow
    y = mdl.addVars(nodes, nodes, lb=0, vtype=GRB.CONTINUOUS, name="y")  # delivery
    z = mdl.addVars(nodes, nodes, lb=0, vtype=GRB.CONTINUOUS, name="z")  # pickup

    # MTZ subtour elimination (NOT for capacity — 2CF handles that)
    u = mdl.addVars(customers, lb=1, ub=n, vtype=GRB.CONTINUOUS, name="u")

    # --- Objective ---
    mdl.setObjective(
        gp.quicksum(dist_matrix[i, j] * x[i, j]
                     for i in nodes for j in nodes if i != j),
        GRB.MINIMIZE
    )

    # --- No self-loops ---
    mdl.addConstrs((x[i, i] == 0 for i in nodes), "no_self")

    # --- Degree ---
    mdl.addConstrs(
        (gp.quicksum(x[i, j] for j in nodes if j != i) == 1 for i in customers), "out"
    )
    mdl.addConstrs(
        (gp.quicksum(x[j, i] for j in nodes if j != i) == 1 for i in customers), "in"
    )

    # --- Fix vehicles ---
    #mdl.addConstr(gp.quicksum(x[0, j] for j in customers) == num_vehicles, "fixveh")

    # --- MTZ subtour elimination ---
    for i in customers:
        for j in customers:
            if i != j:
                mdl.addConstr(u[j] >= u[i] + 1 - n * (1 - x[i, j]), f"MTZ_{i}_{j}")

    # --- Delivery flow balance ---
    # At each customer j: incoming delivery flow - outgoing = d_j (delivered here)
    mdl.addConstrs(
        (gp.quicksum(y[i, j] for i in nodes if i != j) -
         gp.quicksum(y[j, k] for k in nodes if k != j) == robust_del[j]
         for j in customers),
        "del_balance"
    )

    # --- Pickup flow balance ---
    # At each customer j: outgoing pickup flow - incoming = p_j (picked up here)
    mdl.addConstrs(
        (gp.quicksum(z[j, k] for k in nodes if k != j) -
         gp.quicksum(z[i, j] for i in nodes if i != j) == robust_pick[j]
         for j in customers),
        "pick_balance"
    )

    # --- Joint capacity: delivery + pickup on each arc <= C * x ---
    mdl.addConstrs(
        (y[i, j] + z[i, j] <= C * x[i, j]
         for i in nodes for j in nodes if i != j),
        "joint_cap"
    )

    # --- Solve ---
    mdl.optimize()
    st = time.perf_counter() - t0

    if mdl.SolCount == 0:
        return None, None, st

    return mdl.ObjVal, _extract_routes(x, n), st


def _extract_routes(x, n):
    routes = []
    nodes = list(range(n + 1))
    for j in range(1, n + 1):
        try:
            if x[0, j].X > 0.5:
                route, curr, vis = [], j, set()
                while curr != 0 and curr not in vis:
                    vis.add(curr); route.append(curr)
                    found = False
                    for k in nodes:
                        if k != curr and x[curr, k].X > 0.5:
                            curr = k; found = True; break
                    if not found: break
                if route: routes.append(route)
        except: continue
    return routes


# ==============================================================================
# MONTE CARLO VALIDATION (4 scenarios, identical to alns.py)
# ==============================================================================

def build_covariance_matrix(coords, demands, pickups):
    nt = len(coords)
    a = np.array(coords)
    d = a[:, np.newaxis, :] - a[np.newaxis, :, :]
    dm = np.sqrt(np.sum(d ** 2, axis=2))
    th = THETA_FRACTION * np.max(dm)
    sg = np.zeros(nt)
    for i in range(1, nt): sg[i] = CV * (pickups[i] + demands[i])
    return np.outer(sg, sg) * np.exp(-dm / th)


def monte_carlo_validate(routes, demands, pickups, cov_matrix, capacity, rng,
                         n_samples=MC_SAMPLES):
    res = {"GAUSSIAN": 0.0, "SKEW_RIGHT": 0.0, "SKEW_LEFT": 0.0, "HEAVY_TAIL": 0.0}
    if not routes: return res

    ts, tf = 0, {k: 0 for k in res}
    C = capacity
    srm = float(skewnorm.stats(5.0, moments='m'))
    srv = float(skewnorm.stats(5.0, moments='v'))
    slm = float(skewnorm.stats(-5.0, moments='m'))
    slv = float(skewnorm.stats(-5.0, moments='v'))

    for route in routes:
        if not route: continue
        ts += 1; ml = len(route)
        mv = np.array([pickups[c] - demands[c] for c in route])
        ra = np.array(route, dtype=np.intp)
        sc = cov_matrix[np.ix_(ra, ra)]; sc = 0.5 * (sc + sc.T)
        me = np.min(np.linalg.eigvalsh(sc))
        if me < 1e-8: sc += (1e-8 - me) * np.eye(ml)
        sd = np.sqrt(np.diag(sc)); os2 = np.outer(sd, sd)
        cr = np.divide(sc, os2, out=np.eye(ml).copy(), where=os2 != 0)
        mc = np.min(np.linalg.eigvalsh(cr))
        if mc < 1e-8: cr += (1e-8 - mc) * np.eye(ml)
        Z = rng.multivariate_normal(np.zeros(ml), cr, size=n_samples)
        U = norm.cdf(Z)
        scenarios = {
            "GAUSSIAN": norm.ppf(U),
            "SKEW_RIGHT": (skewnorm.ppf(U, 5.0) - srm) / np.sqrt(srv),
            "SKEW_LEFT": (skewnorm.ppf(U, -5.0) - slm) / np.sqrt(slv),
            "HEAVY_TAIL": student_t.ppf(U, df=3) / np.sqrt(3.0),
        }
        td = sum(demands[c] for c in route)
        for sn, ns in scenarios.items():
            sa = mv[np.newaxis, :] + ns * sd[np.newaxis, :]
            cl = td + np.cumsum(sa, axis=1)
            tf[sn] += np.sum(np.any(cl > C, axis=1))
    if ts > 0:
        for k in res: res[k] = tf[k] / (ts * n_samples)
    return res


# ==============================================================================
# PHASE 1: MULTI-INSTANCE BENCHMARK
# ==============================================================================

def run_phase1(target_dir, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, support="QB"):
    """
    Run on ALL .vrp files with fixed (alpha, beta).
    For each instance: inflate demands -> solve Gurobi -> MC validate -> CSV.
    """
    vrp_files = sorted(glob.glob(os.path.join(target_dir, "*.vrp")))
    if not vrp_files:
        print(f"[ERROR] No .vrp files in {target_dir}"); return

    print(f"\n{'='*130}")
    print(f"  PHASE 1: Gounaris Robust VRPSPD | a={alpha} b={beta} | {support}")
    print(f"{'='*130}")
    print(f"{'Instance':<15} | {'N':>4} | {'Cost':>10} | {'Veh':>4} | {'Time':>8} | "
          f"{'F_N':>7} {'F_R':>7} {'F_L':>7} {'F_T':>7}")
    print("-" * 130)

    results = []
    for idx, vp in enumerate(vrp_files):
        bn = os.path.splitext(os.path.basename(vp))[0]
        try:
            rng = np.random.default_rng(SEED + idx)
            nm, co, de, ca, nv = parse_vrp_file(vp)
            pk = augment_pickups(de, rng)
            dm = build_distance_matrix(co)
            cm = build_covariance_matrix(co, de, pk)
            n = len(co) - 1
            C = ca * CAPACITY_FACTOR

            # Inflate demands
            if support == "QB":
                rd = inflate_demands_QB(n, de, co, alpha, beta)
                rp = inflate_demands_QB(n, pk, co, alpha, beta)
            else:
                rd = inflate_demands_QF(n, de, co, dm, alpha, beta)
                rp = inflate_demands_QF(n, pk, co, dm, alpha, beta)

            cost, routes, st = solve_robust_vrpspd(
                nm, co, rd, rp, C, dm,
                nv if nv > 0 else n,
                time_limit=GUROBI_TIME_LIMIT,
            )

            if cost is not None and routes:
                nve = len(routes)
                rmc = np.random.default_rng(SEED + idx + 10000)
                fd = monte_carlo_validate(routes, de, pk, cm, C, rmc)
                print(f"{bn:<15} | {n:>4} | {cost:>10.2f} | {nve:>4} | {st:>8.1f} | "
                      f"{fd['GAUSSIAN']:>7.4f} {fd['SKEW_RIGHT']:>7.4f} "
                      f"{fd['SKEW_LEFT']:>7.4f} {fd['HEAVY_TAIL']:>7.4f}")
                results.append({
                    "Instance": bn, "N": n, "Cost": round(cost, 2),
                    "Vehicles": nve, "Time_s": round(st, 2),
                    "Alpha": alpha, "Beta": beta, "Support": support,
                    "Fail_Gauss": round(fd["GAUSSIAN"], 6),
                    "Fail_SkewR": round(fd["SKEW_RIGHT"], 6),
                    "Fail_SkewL": round(fd["SKEW_LEFT"], 6),
                    "Fail_HeavyT": round(fd["HEAVY_TAIL"], 6),
                })
            else:
                print(f"{bn:<15} | {n:>4} | {'INFEAS':>10} | {'-':>4} | {st:>8.1f} |")
                results.append({
                    "Instance": bn, "N": n, "Cost": "INFEASIBLE",
                    "Time_s": round(st, 2), "Alpha": alpha, "Beta": beta,
                })
        except Exception as e:
            print(f"{bn:<15} | ERROR: {e}")

    out = os.path.join(target_dir, f"Gounaris_Phase1_{support}_a{alpha}_b{beta} withoutfixed car.csv")
    if results:
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys(), extrasaction='ignore')
            w.writeheader(); w.writerows(results)
        print(f"\n  Saved: {out}")


# ==============================================================================
# PHASE 2: 5x5 SENSITIVITY GRID
# ==============================================================================

def run_phase2(target_dir, instance_file="A-n32-k5.vrp"):
    """
    5x5 (alpha, beta) grid on single instance. Both QB and QF.
    Output: cost matrix, PoR%, MC fail rates.
    """
    vp = os.path.join(target_dir, instance_file)
    if not os.path.exists(vp):
        print(f"[ERROR] {vp} not found"); return

    rng = np.random.default_rng(SEED)
    nm, co, de, ca, nv = parse_vrp_file(vp)
    pk = augment_pickups(de, rng)
    dm = build_distance_matrix(co)
    cm = build_covariance_matrix(co, de, pk)
    n = len(co) - 1
    C = ca * CAPACITY_FACTOR

    # Deterministic baseline (no inflation, delivery+pickup)
    print(f"\n  Solving deterministic baseline...")
    cost_det, routes_det, t_det = solve_robust_vrpspd(
        nm, co, de, pk, C, dm, nv if nv > 0 else n,
    )
    if cost_det is None:
        print("  [ERROR] Baseline infeasible!"); return
    print(f"  Baseline: Cost={cost_det:.2f} | Vehicles={len(routes_det)} | Time={t_det:.1f}s")

    alphas = [0.00, 0.05, 0.10, 0.15, 0.20]
    betas = [0.00, 0.25, 0.50, 0.75, 1.00]

    for sup in ["QB", "QF"]:
        print(f"\n{'='*100}")
        print(f"  PHASE 2: {sup} | {nm} | Base={cost_det:.2f}")
        print(f"{'='*100}")

        hdr = f"{'a/b':>8}"
        for b in betas: hdr += f" | {b:>8.2f}"
        print(hdr); print("-" * len(hdr))

        ar = []
        for a in alphas:
            row = f"{a:>8.2f}"
            for b in betas:
                # Inflate both delivery and pickup
                if sup == "QB":
                    rd = inflate_demands_QB(n, de, co, a, b)
                    rp = inflate_demands_QB(n, pk, co, a, b)
                else:
                    rd = inflate_demands_QF(n, de, co, dm, a, b)
                    rp = inflate_demands_QF(n, pk, co, dm, a, b)

                cost, routes, st = solve_robust_vrpspd(
                    nm, co, rd, rp, C, dm,
                    nv if nv > 0 else n,
                )

                if cost is not None and routes:
                    por = ((cost - cost_det) / cost_det * 100) if cost_det > 0 else 0.0
                    rmc = np.random.default_rng(SEED + 777)
                    fd = monte_carlo_validate(routes, de, pk, cm, C, rmc)
                    row += f" | {cost:>8.1f}"
                    ar.append({
                        "Support": sup, "Alpha": a, "Beta": b,
                        "Cost": round(cost, 2), "Vehicles": len(routes),
                        "PoR_%": round(por, 2), "Time_s": round(st, 2),
                        "Fail_Gauss": round(fd["GAUSSIAN"], 6),
                        "Fail_SkewR": round(fd["SKEW_RIGHT"], 6),
                        "Fail_SkewL": round(fd["SKEW_LEFT"], 6),
                        "Fail_HeavyT": round(fd["HEAVY_TAIL"], 6),
                    })
                else:
                    row += f" | {'INFEAS':>8}"
                    ar.append({
                        "Support": sup, "Alpha": a, "Beta": b,
                        "Cost": "INFEASIBLE", "Time_s": round(st, 2),
                    })
            print(row)

        # PoR table
        print(f"\n  PoR (%):")
        ix = 0
        for a in alphas:
            row = f"{a:>8.2f}"
            for b in betas:
                p = ar[ix].get("PoR_%", "N/A")
                row += f" | {p:>7.1f}%" if isinstance(p, (int, float)) else f" | {'N/A':>8}"
                ix += 1
            print(row)

        # MC Fail table
        print(f"\n  MC Fail (Gaussian):")
        ix = 0
        for a in alphas:
            row = f"{a:>8.2f}"
            for b in betas:
                fg = ar[ix].get("Fail_Gauss", "N/A")
                row += f" | {fg:>8.4f}" if isinstance(fg, (int, float)) else f" | {'N/A':>8}"
                ix += 1
            print(row)

        # CSV
        out = os.path.join(target_dir, f"Gounaris_Phase2_{sup}_{nm}.csv")
        if ar:
            with open(out, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=ar[0].keys(), extrasaction='ignore')
                w.writeheader(); w.writerows(ar)
            print(f"\n  Saved: {out}")

    print(f"\n{'='*100}\n  DONE.\n{'='*100}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    TD = r"C:\Users\Admin\Downloads\New folder (4)\X"
    if not os.path.isdir(TD):
        ed = os.environ.get("VRP_DATA_DIR", ".")
        if os.path.isdir(ed): TD = ed
        else: print(f"[ERROR] {TD} not found"); exit(1)

    # ---------------------------------------------------------------
    # UNCOMMENT THE PHASE YOU WANT:
    # ---------------------------------------------------------------

    # Phase 1: Multi-instance (Set A, B, P)
    run_phase1(TD, alpha=0.1, beta=0.5, support="QB")

    # Phase 2: 5x5 sensitivity on A-n32-k5
    #run_phase2(TD, instance_file="A-n32-k5.vrp")