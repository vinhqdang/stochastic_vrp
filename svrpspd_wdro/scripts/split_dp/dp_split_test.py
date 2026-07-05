"""
C4 BENCHMARK v2  --  FAIR comparison: differentiable-split routing vs DRO baselines.

Three fixes over v1, in order:
  (a) MULTI-START TOURS: the split DP is optimal *given a giant tour*, but a single
      nearest-neighbour tour handicaps it. We now generate many candidate tours
      (NN from depot, NN from random starts, random perms, Clarke-Wright via the
      runner's cw_init), 2-opt each, split all, and keep the best frontier point.
  (b) CVaR ADMISSIBILITY: v1's split used the Cantelli gate (looser than the exact
      CVaR gate SAA/WDRO use, so it over-provisions). We can now drive the split with
      YOUR exact TwoPhaseGate (Phase-1 Cantelli prune + Phase-2 empirical CVaR), so
      split-vs-WDRO is SAME GATE, different solver. --gate cvar | cantelli | both.
  (c) WHOLE-SET SWEEP: one instance is not a verdict. --sweep <folder> runs the full
      head-to-head over every *.vrpspd and aggregates (mean gap, win rate, speed).

Risk is a CONSTRAINT, distance the OBJECTIVE; methods are compared at MATCHED OOS
reliability (Monte Carlo feasibility), never raw distance. Demand model / risk measure /
OOS evaluation match dethloff_runner exactly.

USAGE:
  python benchmark_c4.py <instance.vrpspd>                 # single instance, full report
  python benchmark_c4.py <instance.vrpspd> --gate cvar     # split uses exact CVaR gate
  python benchmark_c4.py --sweep <folder> --gate cvar      # whole Dethloff set
  optional: --tours 16  --tlim 10  --no-improve 3
"""

import sys, os, math, time, glob, random, argparse
from pathlib import Path
import numpy as np

# NOTE: we deliberately do NOT import vectorized_split here. That module runs a self-test
# at import time (and exits), which would hijack this script before main() ever runs.
# The benchmark uses its own exact split DP (split_min_distance); the differentiable layer
# in vectorized_split.py is only needed for end-to-end training, not for this evaluation.
HAVE_TORCH = False


# ============================================================================
# Runner import + instance loading
# ============================================================================

def _runner_path():
    """Edit if dethloff_runner lives elsewhere."""
    for p in ["dethloff_runner", "dethloff_runner.py",
              "./dethloff_runner", "/mnt/project/dethloff_runner"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("dethloff_runner not found; set _runner_path()")


def _load_runner():
    import importlib.machinery
    return importlib.machinery.SourceFileLoader("drmod", _runner_path()).load_module()


def load_instance(path=None, n_syn=40, seed=0, dr=None):
    if path is not None:
        if dr is None:
            dr = _load_runner()
        D, dem, Q, n, _ = dr.parse_dethloff(path)
        return (np.asarray(D, float), dem[:, 0].astype(float), dem[:, 1].astype(float),
                float(Q), int(n))
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n_syn, 2)); coords[0] = [50, 50]
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    dbar = np.concatenate([[0.0], rng.uniform(5, 25, size=n_syn - 1)])
    pbar = np.concatenate([[0.0], rng.uniform(5, 25, size=n_syn - 1)])
    Q = float(np.percentile(dbar[1:] + pbar[1:], 90) * 4)
    return D, dbar, pbar, Q, n_syn


# ============================================================================
# Demand model + risk measures (match dethloff_runner)
# ============================================================================

def sample_demands_corr(mean, N, cv, rho, rng):
    from scipy import stats as st
    n = len(mean); out = np.zeros((N, n))
    active = [i for i in range(n) if mean[i] > 0]
    if not active:
        return out
    m = len(active)
    if rho and rho > 0:
        Sigma = np.full((m, m), float(rho)); np.fill_diagonal(Sigma, 1.0)
        L = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal((N, m)) @ L.T
        U = np.clip(st.norm.cdf(Z), 1e-12, 1 - 1e-12)
        for j, i in enumerate(active):
            k = 1.0 / (cv * cv)
            out[:, i] = st.gamma.ppf(U[:, j], k, scale=mean[i] / k)
    else:
        for i in active:
            k = 1.0 / (cv * cv)
            out[:, i] = rng.gamma(k, mean[i] / k, N)
    return out


def cantelli_risk(route, dbar, pbar, sig_d, sig_p, z, rho):
    if not route:
        return 0.0
    d = dbar[route]; p = pbar[route]; sd = sig_d[route]; sp = sig_p[route]
    total_d = d.sum()
    M = np.concatenate(([total_d], total_d - np.cumsum(d) + np.cumsum(p)))
    v2 = (sd ** 2).sum()
    Vind = np.concatenate(([v2], v2 - np.cumsum(sd ** 2) + np.cumsum(sp ** 2)))
    s1 = sd.sum()
    S = np.concatenate(([s1], s1 - np.cumsum(sd) + np.cumsum(sp)))
    Vcorr = (1.0 - rho) * Vind + rho * S ** 2
    return float(np.max(M + z * np.sqrt(np.clip(Vcorr, 0.0, None))))


def route_peaks_oos(route, dsc, psc):
    if not route:
        return np.zeros(dsc.shape[0])
    d = dsc[:, route]; p = psc[:, route]
    total_d = d.sum(1)
    Lmid = total_d[:, None] - np.cumsum(d, 1) + np.cumsum(p, 1)
    return np.maximum(total_d, Lmid.max(1))


# ============================================================================
# Distance, tours (D-matrix only -- Dethloff gives no coordinates), split DP
# ============================================================================

def route_distance(route, D):
    if not route:
        return 0.0
    dd = D[0, route[0]]
    for a, b in zip(route[:-1], route[1:]):
        dd += D[a, b]
    return float(dd + D[route[-1], 0])


def nn_tour(D, n, start=None):
    unv = set(range(1, n)); tour = []; cur = 0
    if start is not None and start in unv:
        tour.append(start); unv.discard(start); cur = start
    while unv:
        nxt = min(unv, key=lambda j: D[cur, j]); tour.append(nxt); unv.discard(nxt); cur = nxt
    return tour


def two_opt(tour, D, max_pass=4):
    """Standard 2-opt on the depot-to-depot cycle, O(1) delta per move."""
    c = [0] + list(tour) + [0]; L = len(c)
    improved = True; p = 0
    while improved and p < max_pass:
        improved = False; p += 1
        for i in range(1, L - 2):
            for j in range(i + 1, L - 1):
                a, b, cc, d = c[i - 1], c[i], c[j], c[j + 1]
                if (D[a, cc] + D[b, d]) - (D[a, b] + D[cc, d]) < -1e-9:
                    c[i:j + 1] = c[i:j + 1][::-1]; improved = True
    return c[1:-1]


def candidate_tours(D, dbar, pbar, Q, n, dr=None, k_starts=8, k_random=4, seed=0, do_2opt=True):
    rng = random.Random(seed); tours = {}
    tours["nn_depot"] = nn_tour(D, n)
    custs = list(range(1, n))
    for s in rng.sample(custs, min(k_starts, len(custs))):
        tours[f"nn_{s}"] = nn_tour(D, n, start=s)
    for r in range(k_random):
        perm = custs[:]; rng.shuffle(perm); tours[f"rand_{r}"] = perm
    if dr is not None:                                   # Clarke-Wright giant tour (strong)
        try:
            sol = dr.cw_init(D, dr.DetGate(Q, dbar, pbar), n)
            flat = [c for route in sol for c in route if c != 0]
            if len(flat) == n - 1:
                tours["cw"] = flat
        except Exception:
            pass
    if do_2opt:
        tours = {k: two_opt(v, D) for k, v in tours.items()}
    uniq = {}; seen = set()
    for k, v in tours.items():
        key = tuple(v)
        if key not in seen:
            seen.add(key); uniq[k] = v
    return uniq


def split_min_distance(tour, D, admissible, max_len):
    """Exact min-distance split into admissible routes (eta->0 limit of the diff. split)."""
    n = len(tour); INF = float("inf")
    dp = [INF] * (n + 1); dp[0] = 0.0; back = [-1] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_len), j):
            seg = tour[i:j]
            if not admissible(seg):
                continue
            c = dp[i] + route_distance(seg, D)
            if c < dp[j]:
                dp[j] = c; back[j] = i
    if dp[n] == INF:
        return INF, None
    routes = []; j = n
    while j > 0:
        i = back[j]; routes.append(tour[i:j]); j = i
    return dp[n], routes[::-1]


# ============================================================================
# Out-of-sample evaluation (the reliability axis)
# ============================================================================

def evaluate_oos(routes, dbar, pbar, Q, cv, rho, N=20000, seed=12345, dsc=None, psc=None):
    if dsc is None or psc is None:
        rng = np.random.default_rng(seed)
        dsc = sample_demands_corr(dbar, N, cv, rho, rng)
        psc = sample_demands_corr(pbar, N, cv, rho, rng)
    ok_all = np.ones(dsc.shape[0], dtype=bool); viol = 0; tot = 0
    for r in routes:
        peaks = route_peaks_oos(r, dsc, psc)
        bad = peaks > Q + 1e-9
        ok_all &= ~bad; viol += int(bad.sum()); tot += dsc.shape[0]
    return dict(scenario_feas=float(ok_all.mean()),
                route_violation=float(viol / max(tot, 1)))


def make_oos_scenarios(dbar, pbar, cv, rho, N=20000, seed=12345):
    """Generate the fixed OOS scenario set ONCE per instance (reused across all evaluations)."""
    rng = np.random.default_rng(seed)
    return (sample_demands_corr(dbar, N, cv, rho, rng),
            sample_demands_corr(pbar, N, cv, rho, rng))


# ============================================================================
# Multi-start split frontier (fix a + optional fix b)
# ============================================================================

def split_frontier(tours, D, dbar, pbar, Q, n, dr, gate_mode, risk_grid,
                   cv=0.30, rho=0.6, max_len=12, oos_N=20000):
    """Collect (distance, OOS-feasibility) points across all tours x risk levels.
    gate_mode='cantelli': risk_grid = Cantelli multipliers z.
    gate_mode='cvar'    : risk_grid = epsilon (cap = Q*(1-eps)), exact TwoPhaseGate.

    Timing is split into:
      solve_time : time inside split_min_distance ONLY (the actual algorithm).
      eval_time  : time inside evaluate_oos ONLY (Monte Carlo, NOT part of the method).
    Each point carries solve_t; the totals are returned alongside so the speed claim
    reflects the solver, not the benchmark harness."""
    sig_d = cv * dbar; sig_p = cv * pbar
    points = []
    tot_solve = 0.0; tot_eval = 0.0
    oos_dsc, oos_psc = make_oos_scenarios(dbar, pbar, cv, rho, N=oos_N)   # once per instance

    if gate_mode == "cantelli":
        # FAST PATH: precompute each tour's per-segment risk tensors ONCE (z-independent),
        # vectorized over all segments; reuse across the whole reliability sweep.
        from fast_split import TourPrecomp, split_from_matrices
        for tid, tour in tours.items():
            t0 = time.perf_counter()
            pc = TourPrecomp(tour, D, dbar, pbar, sig_d, sig_p, rho, max_len)
            precomp_t = time.perf_counter() - t0
            tot_solve += precomp_t
            for z in risk_grid:
                t1 = time.perf_counter()
                mask = pc.admissible_mask(z, Q)
                cost, routes_pos = split_from_matrices(pc.dist, mask, max_len)
                split_dt = time.perf_counter() - t1
                tot_solve += split_dt
                if routes_pos is None:
                    continue
                routes = [tour[i:j] for (i, j) in routes_pos]
                t2 = time.perf_counter()
                ev = evaluate_oos(routes, dbar, pbar, Q, cv, rho, dsc=oos_dsc, psc=oos_psc)
                tot_eval += time.perf_counter() - t2
                points.append(dict(level=z, tour=tid, dist=cost, n_veh=len(routes),
                                   scen_feas=ev["scenario_feas"], route_viol=ev["route_violation"],
                                   solve_t=precomp_t / max(len(risk_grid), 1) + split_dt))
        return points, dict(solve_time=tot_solve, eval_time=tot_eval)

    # CVaR path: exact TwoPhaseGate (CVaR over scenarios) -- kept on the per-segment path.
    for level in risk_grid:
        gdsc, gpsc = dr.make_scenarios(dbar, pbar, dr.N_DATA, dr.CV, dr.DIST, dr.SEED)
        gate = dr.TwoPhaseGate(Q * (1 - level), dr.ALPHA, dbar, pbar, sig_d, sig_p,
                               dr.Z_CVAR, rho, gdsc, gpsc, True)
        cache = {}
        def admissible(seg, gate=gate, cache=cache):
            key = tuple(seg)
            if key not in cache:
                cache[key] = bool(gate.feasible(list(seg)))
            return cache[key]
        for tid, tour in tours.items():
            t0 = time.perf_counter()
            dist, routes = split_min_distance(tour, D, admissible, max_len)
            split_dt = time.perf_counter() - t0
            tot_solve += split_dt
            if routes is None:
                continue
            t1 = time.perf_counter()
            ev = evaluate_oos(routes, dbar, pbar, Q, cv, rho, dsc=oos_dsc, psc=oos_psc)
            tot_eval += time.perf_counter() - t1
            points.append(dict(level=level, tour=tid, dist=dist, n_veh=len(routes),
                               scen_feas=ev["scenario_feas"], route_viol=ev["route_violation"],
                               solve_t=split_dt))
    return points, dict(solve_time=tot_solve, eval_time=tot_eval)


def cost_at_reliability(rows, target_feas, dist_key="dist", feas_key="scen_feas"):
    feas = [r for r in rows if r[feas_key] >= target_feas and math.isfinite(r[dist_key])]
    return min(feas, key=lambda r: r[dist_key]) if feas else None


# ============================================================================
# ALNS + DRO baselines (your dethloff_runner.solve_instance, all 6 policies)
# ============================================================================

def run_alns_baseline(path, D, dbar, pbar, Q, n, dr, cv=0.30, rho=0.6,
                      tlim=10.0, no_improve=3.0, oos_N=20000):
    sol = dr.solve_instance(path, tlim, no_improve, True)
    oos_dsc, oos_psc = make_oos_scenarios(dbar, pbar, cv, rho, N=oos_N)
    rows = []
    for policy, r in sol["res"].items():
        plan = [rt for rt in r["plan"] if rt]
        ev = evaluate_oos(plan, dbar, pbar, Q, cv, rho, dsc=oos_dsc, psc=oos_psc)
        dist = sum(route_distance(rt, D) for rt in plan)
        rows.append(dict(policy=policy, dist=dist, n_veh=r["K"],
                         scen_feas=ev["scenario_feas"], route_viol=ev["route_violation"],
                         time=r["time"]))
    return rows


# ============================================================================
# TOTAL BUSINESS COST (TBC) -- the framework's headline metric
#   TBC = distance + omega_V*K + omega_F * E_w[V_extra]   (stress-mixture violation)
# Prices reliability directly, so methods are compared by ONE number (no reliability matching).
# ============================================================================

def _load_otr_core():
    """Load the OTR primitives from otr.py (= core.otr). Edit paths if needed."""
    import importlib.machinery, os
    for p in ["core/otr.py", "otr.py", "./core/otr.py", "/mnt/project/otr.py"]:
        if os.path.exists(p):
            return importlib.machinery.SourceFileLoader("otrcore", p).load_module()
    raise FileNotFoundError("otr.py (core.otr) not found; set path in _load_otr_core()")


def _otr_gen_reference(dr, dbar, pbar, N, seed):
    n = len(dbar); rng = np.random.default_rng(seed)
    return (dr.sample_demands(dbar, n, N, dr.CV, dr.DIST, rng),
            dr.sample_demands(pbar, n, N, dr.CV, dr.DIST, rng))


def _otr_gen_test_mixture(dr, dbar, pbar, N, seed):
    n = len(dbar); d_parts, p_parts = [], []
    for si, (shape, w) in enumerate(dr.SHAPE_W.items()):
        ns = int(round(N * w))
        if ns <= 0:
            continue
        dist = "lognormal" if shape == "heavy" else shape
        rng = np.random.default_rng(seed + 7919 * (si + 1))
        d_parts.append(dr.sample_demands(dbar, n, ns, dr.CV, dist, rng))
        p_parts.append(dr.sample_demands(pbar, n, ns, dr.CV, dist, rng))
    return np.vstack(d_parts), np.vstack(p_parts)


def _otr_eval_plan(otr, dr, plan, dbar, pbar, Q, omegaF, Cfail, n_train, n_test, seed):
    """Exact replica of run_otr._eval_plan: train OTR on gamma reference, test on stress mixture.
    Returns per-threshold dict with 'total_exec_cost'."""
    dsc_tr, psc_tr = _otr_gen_reference(dr, dbar, pbar, n_train, seed)
    dsc_te, psc_te = _otr_gen_test_mixture(dr, dbar, pbar, n_test, seed + 99_991)
    routes = [r for r in plan if r]
    agg = {lbl: [] for lbl in ("tuned", "myopic", "none")}
    for route in routes:
        rr = np.asarray(route)
        models = otr.fit_otr(dsc_tr[:, rr], psc_tr[:, rr], Q)
        tau_myo = otr.tau_myopic(omegaF, Cfail)
        tau_tun = otr.tune_tau_fast(dsc_tr[:, rr], psc_tr[:, rr], Q, models, omegaF, Cfail) if models else tau_myo
        for lbl, tau in [("tuned", tau_tun), ("myopic", tau_myo), ("none", 1.0)]:
            s = otr.simulate_fast(dsc_te[:, rr], psc_te[:, rr], Q, tau, omegaF, Cfail, models)
            agg[lbl].append(s["mean_cost"])
    return {lbl: {"total_exec_cost": float(sum(v)) if v else 0.0} for lbl, v in agg.items()}


def otr_exec_cost(plan, dbar, pbar, n, Q, omega_F, cfail_ratio, otrmod, name,
                  n_train=1000, n_test=2000, dr=None):
    """Apply OTR to ONE plan, exactly as run_otr does. Returns (tuned_exec, none_exec).
    otrmod = (otr_core_module, dr). name seeds the same per-instance RNG as run_otr."""
    import hashlib
    otr, drmod = otrmod
    Cfail = cfail_ratio * omega_F
    seed = drmod.SEED + int(hashlib.md5(name.encode()).hexdigest(), 16) % 10_000
    res = _otr_eval_plan(otr, drmod, [r for r in plan if r], dbar, pbar, Q,
                         omega_F, Cfail, n_train, n_test, seed)
    return res["tuned"]["total_exec_cost"], res["none"]["total_exec_cost"]


def baseline_tbcs(sol, dr, D):
    dbar, pbar, n, Q = sol["dbar"], sol["pbar"], sol["n"], sol["Q"]
    omega_V = sol["omega_V"]; omega_F = dr.OMEGA_RATIO * omega_V
    out = {}
    for k, r in sol["res"].items():
        plan = [rt for rt in r["plan"] if rt]
        _, evx = dr.eval_evextra(plan, dbar, pbar, n, Q)
        dist = r["dist"]                              # already in REAL units (runner divides by scale)
        out[k] = dict(K=r["K"], dist=dist, evx=evx, time=r["time"],
                      tbc=dist + omega_V * r["K"] + omega_F * evx)
    return out, omega_V, omega_F


def split_best_tbc(tours, D, dbar, pbar, n, Q, dr, omega_V, omega_F, scale,
                   z_grid, cv=0.30, rho=0.6, max_len=12, otr_select=None):
    """For each reliability level z, take the min-distance split across tours, price its TBC.
    Distances divided by `scale` to match the runner's real units.

    Selection of the returned plan:
      otr_select=None  -> pick z minimizing STATIC TBC (dist + omega_V*K + omega_F*evx).
      otr_select=(otr_core, dr, cfail, name) -> pick z minimizing OTR_TBC (dist + omega_V*K +
        OTR tuned exec). This is symmetric to choosing the best baseline by OTR_TBC: both sides
        pick their best point on the conservativeness axis under the same final metric.
    Returns the chosen split (with both static tbc and, if otr_select, otr_tbc) + solve time.
    All candidate (z) plans are kept in 'candidates' for inspection."""
    from fast_split import TourPrecomp, split_from_matrices
    sig_d = cv * dbar; sig_p = cv * pbar
    t0 = time.perf_counter()
    pcs = {tid: TourPrecomp(tour, D, dbar, pbar, sig_d, sig_p, rho, max_len)
           for tid, tour in tours.items()}
    precomp_t = time.perf_counter() - t0
    solve_t = precomp_t
    cands = []
    for z in z_grid:
        zbest = None
        t1 = time.perf_counter()
        for tid, tour in tours.items():
            mask = pcs[tid].admissible_mask(z, Q)
            cost, routes_pos = split_from_matrices(pcs[tid].dist, mask, max_len)
            if routes_pos is None:
                continue
            if zbest is None or cost < zbest[0]:
                zbest = (cost, [tour[i:j] for (i, j) in routes_pos], tid)
        solve_t += time.perf_counter() - t1
        if zbest is None:
            continue
        cost, plan, tid = zbest
        dist = sum(dr.route_cost(rt, D) for rt in plan) / scale
        _, evx = dr.eval_evextra(plan, dbar, pbar, n, Q)
        static_tbc = dist + omega_V * len(plan) + omega_F * evx
        cand = dict(static_tbc=static_tbc, dist=dist, K=len(plan), evx=evx, z=z, tour=tid, plan=plan)
        cands.append(cand)

    if not cands:
        return None
    if otr_select is not None:
        otr_core, drmod, cfail, name = otr_select
        for c in cands:
            tuned, _ = otr_exec_cost(c["plan"], dbar, pbar, n, Q, omega_F, cfail,
                                     (otr_core, drmod), name)
            c["otr_tbc"] = c["dist"] + omega_V * c["K"] + tuned
        best = min(cands, key=lambda c: c["otr_tbc"])
    else:
        best = min(cands, key=lambda c: c["static_tbc"])
    best["tbc"] = best["static_tbc"]
    best["solve_t"] = solve_t
    best["candidates"] = cands
    return best


def report_tbc(path, args, dr):
    D, dbar, pbar, Q, n = load_instance(path, dr=dr)
    scale = dr.parse_dethloff(path)[4]
    name = Path(path).stem
    print(f"instance: n={n}, Q={Q:.1f}, {Path(path).name}")
    sol = dr.solve_instance(path, args.tlim, args.no_improve, True)
    base, omega_V, omega_F = baseline_tbcs(sol, dr, D)
    tours = candidate_tours(D, dbar, pbar, Q, n, dr=dr, k_starts=args.tours, seed=0)
    z_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    # If OTR is requested, load it and let split pick z by OTR_TBC (symmetric to best-baseline).
    otr_core = None; otr_select = None
    if args.otr:
        try:
            otr_core = _load_otr_core()
            otr_select = (otr_core, dr, args.cfail, name)
        except Exception as e:
            print(f"[OTR skipped: {type(e).__name__}: {e}]")
    sp = split_best_tbc(tours, D, dbar, pbar, n, Q, dr, omega_V, omega_F, scale, z_grid,
                        cv=dr.CV, rho=dr.RHO, otr_select=otr_select)
    print(f"omega_V={omega_V:.1f}  omega_F={omega_F:.1f} (=50*omega_V)  "
          f"TBC = dist + omega_V*K + (exec cost)\n")

    otr = {}
    if otr_core is not None and sp is not None:
        for k, v in base.items():
            tuned, _ = otr_exec_cost(sol["res"][k]["plan"], dbar, pbar, n, Q,
                                     omega_F, args.cfail, (otr_core, dr), name)
            otr[k] = v["dist"] + omega_V * v["K"] + tuned
        otr["SPLIT"] = sp["otr_tbc"]      # split's chosen plan, OTR-priced

    hdr = f"  {'method':>9} {'staticTBC':>12} {'dist':>10} {'K':>3} {'E[Vextra]':>10}"
    hdr += f" {'OTR_TBC':>12}" if otr else ""
    hdr += f" {'time(s)':>8}"
    print(hdr)
    rows = [(k, v) for k, v in base.items()]
    if sp:
        rows.append(("SPLIT", dict(tbc=sp["tbc"], dist=sp["dist"], K=sp["K"],
                                   evx=sp["evx"], time=sp["solve_t"])))
    sortkey = (lambda kv: otr.get(kv[0], kv[1]["tbc"])) if otr else (lambda kv: kv[1]["tbc"])
    for k, v in sorted(rows, key=sortkey):
        star = "  <-- SPLIT" if k == "SPLIT" else ""
        line = f"  {k:>9} {v['tbc']:>12.1f} {v['dist']:>10.1f} {v['K']:>3d} {v['evx']:>10.4f}"
        line += f" {otr[k]:>12.1f}" if otr else ""
        line += f" {v['time']:>8.2f}{star}"
        print(line)

    if sp:
        if otr:
            metric_base = {k: otr[k] for k in base}
            split_val = otr["SPLIT"]; label = "OTR_TBC"
        else:
            metric_base = {k: base[k]["tbc"] for k in base}
            split_val = sp["tbc"]; label = "static TBC"
        # show what split selected over (per-z candidates)
        if sp.get("candidates"):
            print("\n  split candidates per z (selection metric = "
                  f"{'OTR_TBC' if otr_select else 'static TBC'}):")
            print(f"    {'z':>5} {'K':>3} {'dist':>9} {'E[Vextra]':>10}"
                  + (f" {'OTR_TBC':>11}" if otr_select else f" {'staticTBC':>11}"))
            for c in sp["candidates"]:
                val = c.get("otr_tbc", c["static_tbc"])
                pick = "  <--" if (abs(c["z"] - sp["z"]) < 1e-9) else ""
                print(f"    {c['z']:>5.2f} {c['K']:>3d} {c['dist']:>9.1f} {c['evx']:>10.4f} {val:>11.1f}{pick}")
        best_base = min(metric_base.items(), key=lambda kv: kv[1])
        gap = 100 * (split_val - best_base[1]) / best_base[1]
        win = "SPLIT" if split_val < best_base[1] else best_base[0]
        sp_t = sp["solve_t"]; base_t = sum(v["time"] for v in base.values())
        print(f"\n  [{label}] best baseline = {best_base[0]} ({best_base[1]:.1f}); "
              f"SPLIT = {split_val:.1f} at z={sp['z']} (K={sp['K']})")
        print(f"  gap (SPLIT vs best baseline): {gap:+.1f}%   winner: {win}")
        print(f"  SOLVE time: SPLIT {1000*sp_t:.0f} ms vs all-baselines {base_t:.1f} s "
              f"(~{base_t/max(sp_t,1e-9):.0f}x)")
    return base, sp


def sweep_tbc(folder, args, dr):
    files = sorted(glob.glob(str(Path(folder) / "*.vrpspd")))
    if not files:
        print(f"no *.vrpspd in {folder}"); return
    use_otr = args.otr
    otrmod = None
    if use_otr:
        try:
            otrmod = _load_otr_core()
        except Exception as e:
            print(f"[OTR unavailable, falling back to static TBC: {e}]"); use_otr = False
    metric_name = "OTR_TBC" if use_otr else "static TBC"
    print(f"TBC sweep over {len(files)} instances (tlim={args.tlim}s/policy, metric={metric_name})\n")
    print(f"{'instance':>14} {'SPLIT':>12} {'bestBASE':>12} {'base':>5} "
          f"{'gap%':>7} {'win':>6} {'spd':>6}")
    z_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    wins = {"SPLIT": 0, "BASE": 0}; gaps = []; spds = []
    per_base_gap = {}
    for path in files:
        try:
            D, dbar, pbar, Q, n = load_instance(path, dr=dr)
            scale = dr.parse_dethloff(path)[4]
            sol = dr.solve_instance(path, args.tlim, args.no_improve, True)
            base, omega_V, omega_F = baseline_tbcs(sol, dr, D)
            tours = candidate_tours(D, dbar, pbar, Q, n, dr=dr, k_starts=args.tours, seed=0)
            name = Path(path).stem
            otr_select = (otrmod, dr, args.cfail, name) if use_otr else None
            sp = split_best_tbc(tours, D, dbar, pbar, n, Q, dr, omega_V, omega_F, scale, z_grid,
                                cv=dr.CV, rho=dr.RHO, otr_select=otr_select)
            # metric per method: OTR_TBC if requested, else static TBC
            if use_otr and sp:
                mbase = {}
                for k, v in base.items():
                    tuned, _ = otr_exec_cost(sol["res"][k]["plan"], dbar, pbar, n, Q,
                                             omega_F, args.cfail, (otrmod, dr), name)
                    mbase[k] = v["dist"] + omega_V * v["K"] + tuned
                msplit = sp["otr_tbc"]
            else:
                mbase = {k: v["tbc"] for k, v in base.items()}
                msplit = sp["tbc"] if sp else float("inf")
            bb = min(mbase.items(), key=lambda kv: kv[1])
            gap = 100 * (msplit - bb[1]) / bb[1] if sp else float("nan")
            win = "SPLIT" if (sp and msplit < bb[1]) else "BASE"
            wins[win] += 1
            if sp:
                gaps.append(gap)
                spd = sum(v["time"] for v in base.values()) / max(sp["solve_t"], 1e-9)
                spds.append(spd)
                for k in base:
                    per_base_gap.setdefault(k, []).append(100 * (msplit - mbase[k]) / mbase[k])
            print(f"{name:>14} {msplit:>12.1f} {bb[1]:>12.1f} {bb[0]:>5} "
                  f"{gap:>+7.1f} {win:>6} {spd if sp else 0:>5.0f}x")
        except Exception as e:
            print(f"{Path(path).stem:>14}  ERROR {type(e).__name__}: {e}")
    print(f"\n=== TBC SWEEP SUMMARY (metric: {metric_name}) ===")
    print(f"  win rate: SPLIT {wins['SPLIT']} | baselines {wins['BASE']}")
    if gaps:
        g = np.array(gaps)
        print(f"  SPLIT vs BEST baseline gap: mean {g.mean():+.1f}%  median {np.median(g):+.1f}%")
        print(f"  SOLVE speedup vs all baselines: mean ~{np.mean(spds):.0f}x")
        print("  SPLIT vs each baseline (mean gap%, negative = SPLIT cheaper):")
        for k in sorted(per_base_gap):
            gg = np.array(per_base_gap[k])
            print(f"      vs {k:>9}: {gg.mean():+.1f}%")


# ============================================================================
# Single-instance report
# ============================================================================

def report_instance(path, args, dr):
    D, dbar, pbar, Q, n = load_instance(path, dr=dr)
    print(f"instance: n={n}, Q={Q:.1f}, {Path(path).name}\n")
    tours = candidate_tours(D, dbar, pbar, Q, n, dr=dr, k_starts=args.tours, seed=0)
    print(f"candidate giant tours: {len(tours)}  (NN x starts, random, Clarke-Wright; 2-opt'd)\n")

    modes = ["cantelli", "cvar"] if args.gate == "both" else [args.gate]
    split_points_by_mode = {}
    for mode in modes:
        grid = ([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0] if mode == "cantelli"
                else [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        t0 = time.time()
        pts, timing = split_frontier(tours, D, dbar, pbar, Q, n, dr, mode, grid,
                                     cv=dr.CV, rho=dr.RHO)
        split_points_by_mode[mode] = pts
        solve_ms = 1000 * timing["solve_time"]
        per_solve_ms = 1000 * np.median([p["solve_t"] for p in pts]) if pts else float("nan")
        print(f"SPLIT frontier [{mode} gate, multi-start] -- best distance at each OOS target:")
        print(f"   timing: SOLVE {solve_ms:.0f} ms total ({per_solve_ms:.1f} ms/split, "
              f"{len(pts)} splits) | OOS-eval {timing['eval_time']:.1f} s (NOT part of method)")
        for tgt in [0.90, 0.95, 0.99]:
            b = cost_at_reliability(pts, tgt)
            if b:
                print(f"   feas>={tgt:.2f}: dist={b['dist']:.1f}  veh={b['n_veh']}  "
                      f"(tour={b['tour']}, level={b['level']})")
            else:
                print(f"   feas>={tgt:.2f}: (not reached on grid)")
        print()

    print(f"ALNS + DRO baselines (tlim={args.tlim}s/policy):")
    alns = run_alns_baseline(path, D, dbar, pbar, Q, n, dr, cv=dr.CV, rho=dr.RHO,
                             tlim=args.tlim, no_improve=args.no_improve)
    print(f"   {'policy':>9} {'distance':>10} {'veh':>4} {'OOS feas':>9} {'time(s)':>8}")
    for r in sorted(alns, key=lambda x: x["dist"]):
        print(f"   {r['policy']:>9} {r['dist']:>10.1f} {r['n_veh']:>4d} "
              f"{r['scen_feas']:>9.4f} {r['time']:>8.2f}")
    print()

    print("=== HEAD-TO-HEAD: min distance at matched OOS reliability (lower = better) ===")
    primary = "cvar" if "cvar" in split_points_by_mode else modes[0]
    sp = split_points_by_mode[primary]
    print(f"(SPLIT shown for [{primary}] gate -- same gate as WDRO/SAA when cvar)")
    print(f"   {'target':>7} {'SPLIT':>10} {'best DRO':>10} {'policy':>9} {'winner':>7} {'gap%':>7}")
    for tgt in [0.90, 0.95, 0.99]:
        bs = cost_at_reliability(sp, tgt)
        ba = cost_at_reliability(alns, tgt)
        sd = bs["dist"] if bs else float("inf")
        ad = ba["dist"] if ba else float("inf")
        pol = ba["policy"] if ba else "--"
        win = "SPLIT" if sd < ad - 1e-9 else ("DRO" if ad < sd - 1e-9 else "tie")
        gap = (100 * (sd - ad) / ad) if (math.isfinite(sd) and math.isfinite(ad) and ad > 0) else float("nan")
        sd_s = f"{sd:.1f}" if math.isfinite(sd) else "--"
        ad_s = f"{ad:.1f}" if math.isfinite(ad) else "--"
        gap_s = f"{gap:+.1f}" if math.isfinite(gap) else "--"
        print(f"   {tgt:>7.2f} {sd_s:>10} {ad_s:>10} {pol:>9} {win:>7} {gap_s:>7}")

    mdro = [r for r in alns if r["policy"] == "MDRO"]
    if mdro and "cantelli" in split_points_by_mode:
        print("\nSame-gate check (both Cantelli): SPLIT[cantelli] vs MDRO+ALNS")
        m = mdro[0]
        bs = cost_at_reliability(split_points_by_mode["cantelli"], m["scen_feas"])
        if bs:
            who = "SPLIT" if bs["dist"] < m["dist"] else "MDRO"
            print(f"   at MDRO's reliability {m['scen_feas']:.4f}: "
                  f"SPLIT dist={bs['dist']:.1f} vs MDRO dist={m['dist']:.1f} ({who} wins)")
    return split_points_by_mode, alns


# ============================================================================
# Whole-directory sweep (fix c)
# ============================================================================

def sweep_directory(folder, args, dr):
    files = sorted(glob.glob(str(Path(folder) / "*.vrpspd")))
    if not files:
        print(f"no *.vrpspd files in {folder}"); return
    print(f"sweeping {len(files)} instances in {folder} "
          f"(gate={args.gate}, tlim={args.tlim}s/policy)\n")
    tgt = 0.95
    agg = {"split_win": 0, "dro_win": 0, "tie": 0, "gaps_vs_best": [],
           "gaps_vs_mdro": [], "split_t": [], "dro_t": []}
    print(f"{'instance':>16} {'SPLITdist':>11} {'bestDRO':>11} {'pol':>5} "
          f"{'gap%':>7} {'MDROgap%':>8} {'win':>6}")
    mode = "cvar" if args.gate in ("cvar", "both") else "cantelli"
    grid = ([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] if mode == "cvar"
            else [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    for path in files:
        try:
            D, dbar, pbar, Q, n = load_instance(path, dr=dr)
            tours = candidate_tours(D, dbar, pbar, Q, n, dr=dr, k_starts=args.tours, seed=0)
            t0 = time.time()
            sp, timing = split_frontier(tours, D, dbar, pbar, Q, n, dr, mode, grid, cv=dr.CV, rho=dr.RHO)
            split_solve_t = timing["solve_time"]      # algorithm only (no Monte Carlo)
            alns = run_alns_baseline(path, D, dbar, pbar, Q, n, dr, cv=dr.CV, rho=dr.RHO,
                                     tlim=args.tlim, no_improve=args.no_improve)
            bs = cost_at_reliability(sp, tgt)
            ba = cost_at_reliability(alns, tgt)
            mdro = next((r for r in alns if r["policy"] == "MDRO"), None)
            sd = bs["dist"] if bs else float("inf")
            ad = ba["dist"] if ba else float("inf")
            pol = ba["policy"] if ba else "--"
            gap = (100 * (sd - ad) / ad) if (math.isfinite(sd) and math.isfinite(ad)) else float("nan")
            mgap = (100 * (sd - mdro["dist"]) / mdro["dist"]) if (mdro and math.isfinite(sd)) else float("nan")
            win = "SPLIT" if sd < ad - 1e-9 else ("DRO" if ad < sd - 1e-9 else "tie")
            agg["split_win" if win == "SPLIT" else ("dro_win" if win == "DRO" else "tie")] += 1
            if math.isfinite(gap): agg["gaps_vs_best"].append(gap)
            if math.isfinite(mgap): agg["gaps_vs_mdro"].append(mgap)
            agg["split_t"].append(split_solve_t)
            agg["dro_t"].append(sum(r["time"] for r in alns))
            print(f"{Path(path).stem:>16} {sd:>11.1f} {ad:>11.1f} {pol:>5} "
                  f"{gap:>+7.1f} {mgap:>+8.1f} {win:>6}")
        except Exception as e:
            print(f"{Path(path).stem:>16}  ERROR {type(e).__name__}: {e}")
    print("\n=== SWEEP SUMMARY (target OOS feasibility = 0.95) ===")
    print(f"  win rate: SPLIT {agg['split_win']} | DRO {agg['dro_win']} | tie {agg['tie']}")
    if agg["gaps_vs_best"]:
        g = np.array(agg["gaps_vs_best"])
        print(f"  SPLIT vs best-DRO distance gap: mean {g.mean():+.1f}%  median {np.median(g):+.1f}%")
    if agg["gaps_vs_mdro"]:
        g = np.array(agg["gaps_vs_mdro"])
        print(f"  SPLIT vs MDRO (same gate) gap:  mean {g.mean():+.1f}%  median {np.median(g):+.1f}%")
    if agg["split_t"] and agg["dro_t"]:
        st_, dt_ = np.array(agg["split_t"]), np.array(agg["dro_t"])
        print(f"  SOLVE time (algorithm only, no Monte-Carlo eval):")
        print(f"    SPLIT mean {1000*st_.mean():.0f} ms | DRO+ALNS mean {dt_.mean():.1f} s "
              f"| speedup ~{dt_.mean()/max(st_.mean(),1e-9):.0f}x")
        print(f"  (note: SPLIT solve time = all {len(st_)} instances' multi-start splits; "
              f"per-instance it is the time to split K tours across the reliability grid.)")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None, help="instance .vrpspd, or folder with --sweep")
    ap.add_argument("--sweep", action="store_true", help="treat path as a folder of *.vrpspd")
    ap.add_argument("--gate", choices=["cantelli", "cvar", "both"], default="both")
    ap.add_argument("--tours", type=int, default=8, help="# NN random-start tours (plus NN/CW/rand)")
    ap.add_argument("--tlim", type=float, default=10.0, help="ALNS time limit per policy (s)")
    ap.add_argument("--no-improve", type=float, default=3.0, dest="no_improve")
    ap.add_argument("--tbc", action="store_true",
                    help="compare on Total Business Cost (dist + omega_V*K + omega_F*violation) -- the headline metric")
    ap.add_argument("--otr", action="store_true",
                    help="also apply the OTR execution layer (online threshold reassignment) to every plan")
    ap.add_argument("--cfail", type=float, default=5.0, help="Cfail / omega_F ratio for OTR (default 5.0)")
    args = ap.parse_args()

    if args.path is None:
        print("synthetic smoke test (no instance given):")
        D, dbar, pbar, Q, n = load_instance(None)
        tours = candidate_tours(D, dbar, pbar, Q, n, dr=None, k_starts=args.tours)
        pts, _ = split_frontier(tours, D, dbar, pbar, Q, n, None, "cantelli",
                                [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], cv=0.30, rho=0.6)
        for tgt in [0.90, 0.95, 0.99]:
            b = cost_at_reliability(pts, tgt)
            print(f"  feas>={tgt:.2f}: " + (f"dist={b['dist']:.1f} veh={b['n_veh']} tour={b['tour']}"
                                            if b else "(not reached)"))
        return

    dr = _load_runner()
    if args.sweep and args.tbc:
        sweep_tbc(args.path, args, dr)
    elif args.tbc:
        report_tbc(args.path, args, dr)
    elif args.sweep:
        sweep_directory(args.path, args, dr)
    else:
        report_instance(args.path, args, dr)


if __name__ == "__main__":
    main()