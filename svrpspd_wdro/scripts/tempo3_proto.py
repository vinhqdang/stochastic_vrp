"""tempo3_proto.py: go/no-go for fleet-pooled Bayesian recourse.

Each day carries a common pickup factor a_d (prior: lognormal with sd s,
discretised on the policy-library grid). Every vehicle's recourse
decision at stop k compares the handoff and restock prices with the
continuation value U_a(k+1, W) of the library policy for scale a,
weighted by:
  fixed   : a = 1 (BATON under the planning law)
  plug_mle: a = quantised ratio of observed to planned pickups (fleet)
  plug_pm : a = quantised posterior mean
  bayes_route: posterior from the vehicle's OWN observations only
           (per-route learning, as in Florio et al. 2023)
  bayes   : the full posterior over a (fleet-pooled observations)
  oracle  : the true a_d
Usage (from svrpspd_wdro/): python scripts/tempo3_proto.py [n_inst=10 days=200 workers=4]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
_W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_W))
from ev.fleet import FleetPlan, PolicyLibrary, A_GRID, CV, quantize_a  # noqa

A = np.array(A_GRID, float)
SHAPE = 1.0 / CV ** 2
POL = ["fixed", "plug_mle", "plug_pm", "bayes_route", "bayes", "oracle"]


def prior(s):
    w = np.exp(-0.5 * (np.log(A) / s) ** 2) / A
    return w / w.sum()


def run_day(fl, lib, a_true, pri, rng):
    """Common random numbers: one draw of base gamma variates per day."""
    out = {}
    base = []
    for rt in fl.routes:
        up = rng.gamma(SHAPE, 1.0, rt.m) / SHAPE      # mean 1, cv
        ud = rng.gamma(SHAPE, 1.0, rt.m) / SHAPE
        base.append((up, ud))
    for pol in POL:
        cost = 0.0
        W = np.zeros(len(fl.routes))
        alive = [True] * len(fl.routes)
        loglik = np.zeros(len(A))
        ll_r = np.zeros((len(fl.routes), len(A)))
        sp = spb = 0.0
        for k in range(1, fl.m_max + 1):
            # observe stop k on every alive route, then decide
            obs = []
            for r, rt in enumerate(fl.routes):
                if not alive[r] or k > rt.m:
                    continue
                p = a_true * rt.pbar[k - 1] * base[r][0][k - 1]
                d = rt.dbar[k - 1] * base[r][1][k - 1]
                if rt.pbar[k - 1] > 0:
                    inc = -SHAPE * np.log(A) - p / (A * rt.pbar[k - 1] * CV ** 2)
                    loglik += inc
                    ll_r[r] += inc
                    sp += p
                    spb += rt.pbar[k - 1]
                W[r] += p - d
                obs.append(r)
            post = pri * np.exp(loglik - loglik.max())
            post /= post.sum()
            for r in obs:
                rt = fl.routes[r]
                if W[r] > rt.B:
                    cost += rt.E[k - 1]
                    alive[r] = False
                    continue
                if k == rt.m:
                    alive[r] = False
                    continue
                if pol == "fixed":
                    wts = {1.0: 1.0}
                elif pol == "plug_mle":
                    wts = {quantize_a(sp / spb if spb > 0 else 1.0): 1.0}
                elif pol == "plug_pm":
                    wts = {quantize_a(float(post @ A)): 1.0}
                elif pol == "bayes_route":
                    pr = pri * np.exp(ll_r[r] - ll_r[r].max())
                    pr /= pr.sum()
                    wts = {a: w for a, w in zip(A, pr) if w > 1e-6}
                elif pol == "oracle":
                    wts = {quantize_a(a_true): 1.0}
                else:
                    wts = {a: w for a, w in zip(A, post) if w > 1e-6}
                cont = sum(w * lib.get(r, a).rv.U(k + 1, W[r]) for a, w in wts.items())
                rest = rt.R[k - 1] + sum(w * lib.get(r, a).rv.U(k + 1, 0.0)
                                         for a, w in wts.items())
                ho = rt.H[k - 1]
                if ho < cont and ho <= rest:
                    cost += ho
                    alive[r] = False
                elif rest < cont:
                    cost += rt.R[k - 1]
                    W[r] = 0.0
        out[pol] = cost
    return out


def run(path, days, s_true, s_model):
    fl = FleetPlan.from_instance(path)
    lib = PolicyLibrary(fl, seed=1)
    pt, pm = prior(s_true), prior(s_model)
    rng = np.random.default_rng(7)
    rows = []
    for d in range(days):
        a = float(rng.choice(A, p=pt))
        res = run_day(fl, lib, a, pm, rng)
        rows.append(dict(instance=fl.name, s_true=s_true, s_model=s_model,
                         day=d, a=a, **res))
    return rows


if __name__ == "__main__":
    kw = dict(x.split("=") for x in sys.argv[1:])
    n, days, wk = int(kw.get("n_inst", 10)), int(kw.get("days", 200)), int(kw.get("workers", 4))
    paths = [p for p in sorted((_W / "data" / "Dethloff").glob("*.vrpspd"))
             if (_W / "results" / "plans" / f"{p.stem}.json").exists()][:n]
    combos = [(0.15, 0.15), (0.3, 0.3), (0.3, 0.15)]
    jobs = [(p, days, st, sm) for st, sm in combos for p in paths]
    with ProcessPoolExecutor(max_workers=wk) as ex:
        res = list(ex.map(run, *zip(*jobs)))
    df = pd.DataFrame([r for rs in res for r in rs])
    (_W / "results" / "tempo3").mkdir(parents=True, exist_ok=True)
    df.to_csv(_W / "results" / "tempo3" / "proto.csv", index=False)
    g = df.groupby(["s_true", "s_model"])[POL].mean().round(2)
    print(g.to_string())
