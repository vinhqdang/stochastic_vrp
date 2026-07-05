r"""
OUT-OF-SAMPLE robustness on REAL Dethloff instances -> the aggregate C&OR table.

Loads each Dethloff .vrpspd via the existing parser (distances /scale=10000), takes the benchmark
delivery/pickup values as MEANS, imposes a low-rank factor uncertainty model with a HEAVY-TAILED
true distribution, and runs the robust / SAA / deterministic comparison out-of-sample over the
whole instance set. Reports per-family and aggregate violation rates + the robustness-price frontier.

MODELLING NOTE (state this in the paper): the Dethloff demands are deterministic; we impose a
factor demand model d_i(F)=dbar_i(1+ sum_k lam_ik F_k), F heavy-tailed, as the uncertainty source.
This is a modelling choice a referee will scrutinise -- justify loadings and tail explicitly.

Run on YOUR machine (needs data\Dethloff\*.vrpspd + dethloff_runner + matheuristic_core.py):
    python dethloff_oos.py data\Dethloff
"""
import sys, glob, math, time
from pathlib import Path
import numpy as np
import importlib.machinery

from matheuristic_core import Instance, alns, route_cvar_peak


def _load_runner():
    import os
    for p in ["dethloff_runner", "dethloff_runner.py", "/mnt/project/dethloff_runner"]:
        if os.path.exists(p):
            return importlib.machinery.SourceFileLoader("dr", p).load_module()
    raise FileNotFoundError("dethloff_runner not found")


def load_dethloff(path, dr):
    D, dem, Q, n, scale = dr.parse_dethloff(path)
    C = D.astype(float) / scale                    # REAL units (the /10000 fix)
    dbar = dem[1:n, 0].astype(float)               # customer deliveries (node 0 = depot)
    pbar = dem[1:n, 1].astype(float)               # customer pickups
    ncust = n - 1
    return C, dbar, pbar, Q, ncust                  # C is (n x n) with node 0 = depot


def factor_samples(dbar, pbar, r, N, seed, cv=0.30, heavy=True):
    """Low-rank factor demand: d_i = dbar_i*(1 + sum_k lam_ik F_k), F heavy-tailed if heavy."""
    rng = np.random.default_rng(seed)
    n = len(dbar)
    lam_d = rng.normal(0, 1, (n, r)) * cv / math.sqrt(r)
    lam_p = rng.normal(0, 1, (n, r)) * cv / math.sqrt(r)
    if heavy:
        mask = rng.random((N, r)) < 0.15
        F = np.where(mask, rng.standard_t(df=3, size=(N, r)) * 2.2, rng.normal(0, 1, (N, r)))
    else:
        F = rng.normal(0, 1, (N, r))
    d = np.clip(dbar[None, :] * (1 + F @ lam_d.T), 0, None)
    p = np.clip(pbar[None, :] * (1 + F @ lam_p.T), 0, None)
    return d, p, (lam_d, lam_p)


def make_variant(C, dbar, pbar, Q, mode, eps, alpha, d_tr, p_tr):
    if mode == "det":
        d_scen, p_scen, Qeff = dbar[None, :].copy(), pbar[None, :].copy(), Q
    elif mode == "saa":
        d_scen, p_scen, Qeff = d_tr, p_tr, Q
    else:  # robust
        d_scen, p_scen, Qeff = d_tr, p_tr, Q - eps / (1 - alpha)
    n = len(dbar)
    return Instance(n=n, r=0, N=d_scen.shape[0], alpha=alpha, coords=None, C=C,
                    dbar=dbar, pbar=pbar, d_scen=d_scen, p_scen=p_scen, Q=Q, Qeff=Qeff)


def eval_oos(routes, C, dbar, pbar, Q, alpha, d_te, p_te):
    te = Instance(n=len(dbar), r=0, N=d_te.shape[0], alpha=alpha, coords=None, C=C,
                  dbar=dbar, pbar=pbar, d_scen=d_te, p_scen=p_te, Q=Q, Qeff=Q)
    viol = nr = 0; worst = 0.0
    for rt in routes:
        if not rt: continue
        nr += 1; cp = route_cvar_peak(rt, te); worst = max(worst, cp / Q)
        if cp > Q + 1e-6: viol += 1
    dist = 0.0
    for rt in routes:
        if not rt: continue
        nodes = [0] + [c + 1 for c in rt] + [0]
        dist += sum(C[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1))
    return viol, nr, dist, worst


def run(folder, r=3, N_train=25, N_test=5000, alpha=0.9, frac_list=(0.0, 0.05, 0.10, 0.15), iters=1500):
    """frac = shadow buffer as a FRACTION of capacity: Qeff = Q*(1-frac).
    (Wasserstein radius eps = frac*Q*(1-alpha); the /(1-alpha) amplification makes small eps
    correspond to a large capacity haircut, so we parameterise by the haircut directly.)"""
    dr = _load_runner()
    files = sorted(glob.glob(str(Path(folder) / "*.vrpspd")))
    if not files:
        print(f"no *.vrpspd in {folder}"); return
    print(f"Out-of-sample over {len(files)} Dethloff instances "
          f"(factor r={r}, N_train={N_train}, N_test={N_test}, alpha={alpha}, heavy-tailed truth)")
    print(f"robust shadow buffer as fraction of Q: {[f for f in frac_list if f>0]}\n")
    agg = {}
    def add(tag, v, nr, dist, worst, base_dist):
        a = agg.setdefault(tag, dict(v=0, nr=0, dist=0.0, base=0.0, worst=[]))
        a["v"] += v; a["nr"] += nr; a["dist"] += dist; a["base"] += base_dist; a["worst"].append(worst)

    def solve_eval(mode, frac, C, dbar, pbar, Q, omega_V, d_tr, p_tr, d_te, p_te):
        Qeff = Q * (1 - frac) if mode == "robust" else Q
        try:
            inst = make_variant2(C, dbar, pbar, Q, Qeff, mode, alpha, d_tr, p_tr)
            best, _ = alns(inst, iters=iters, seed=1, omega_V=omega_V, verbose=False)
            return eval_oos(best.routes, C, dbar, pbar, Q, alpha, d_te, p_te)
        except RuntimeError:
            return None                                  # eps/haircut too large -> infeasible

    for path in files:
        try:
            C, dbar, pbar, Q, ncust = load_dethloff(path, dr)
            omega_V = C[0, 1:].mean()
            d_tr, p_tr, _ = factor_samples(dbar, pbar, r, N_train, seed=1, heavy=True)
            d_te, p_te, _ = factor_samples(dbar, pbar, r, N_test, seed=999, heavy=True)
            res = solve_eval("saa", 0.0, C, dbar, pbar, Q, omega_V, d_tr, p_tr, d_te, p_te)
            if res is None: 
                print(f"  {Path(path).stem:>10}: SAA infeasible, skip"); continue
            v, nr, dist_saa, w = res; add("SAA", v, nr, dist_saa, w, dist_saa)
            res = solve_eval("det", 0.0, C, dbar, pbar, Q, omega_V, d_tr, p_tr, d_te, p_te)
            if res: add("det", *res[:3], res[3], dist_saa)
            for frac in frac_list:
                if frac == 0.0: continue
                res = solve_eval("robust", frac, C, dbar, pbar, Q, omega_V, d_tr, p_tr, d_te, p_te)
                if res: add(f"robust {int(frac*100)}%Q", *res[:3], res[3], dist_saa)
            print(f"  {Path(path).stem:>10}: done (n={ncust}, Q={Q:.0f})")
        except Exception as e:
            print(f"  {Path(path).stem:>10}: ERROR {type(e).__name__}: {e}")

    print("\n=== AGGREGATE out-of-sample (over all instances) ===")
    print(f"{'variant':>16} {'viol routes%':>12} {'mean worstCVaR/Q':>17} {'dist vs SAA%':>13}")
    order = ["det", "SAA"] + [f"robust {int(f*100)}%Q" for f in frac_list if f != 0.0]
    for tag in order:
        if tag not in agg: continue
        a = agg[tag]
        vr = 100 * a["v"] / max(a["nr"], 1)
        dv = 100 * (a["dist"] - a["base"]) / max(a["base"], 1e-9)
        print(f"{tag:>16} {vr:>11.1f}% {np.mean(a['worst']):>17.3f} {dv:>+12.1f}%")
    print("\nFRONTIER: robustness (violation ->0) bought with distance. Pick the haircut knee.")
    print("Also run large N_train / heavy=False to show DR is unnecessary when data is abundant")
    print("(honest scoping of when robustness pays).")


def make_variant2(C, dbar, pbar, Q, Qeff, mode, alpha, d_tr, p_tr):
    if mode == "det":
        d_scen, p_scen = dbar[None, :].copy(), pbar[None, :].copy()
    else:
        d_scen, p_scen = d_tr, p_tr
    return Instance(n=len(dbar), r=0, N=d_scen.shape[0], alpha=alpha, coords=None, C=C,
                    dbar=dbar, pbar=pbar, d_scen=d_scen, p_scen=p_scen, Q=Q, Qeff=Qeff)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else r"data\Dethloff"
    run(folder)
