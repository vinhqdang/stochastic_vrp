"""tempo3_pool.py: go/no-go for learning-driven standby booking.

Standby vehicles must be booked: before the day at c0 per vehicle, or at
a top-up point (after stop index k_top) at c1 > c0. A handoff consumes a
booked vehicle; with none left the vehicle cannot hand off (it restocks
or continues). Handoff prices exclude the standby day rate (paid via the
booking). Day factor: city-level, log a ~ N(0, s^2). Every vehicle's
recourse uses the library policy at the fleet-pooled posterior mean of a
(the same for every booking rule); only the booking differs:
  ppu        unlimited pay-per-use at the standby day rate (BATON default)
  static     book quantile_q of the prior predictive handoff count
  twostage_k book quantile_q0 of the prior predictive before the day,
             top up at stop k to u + quantile_q of the remaining-handoff
             predictive under the pooled posterior (u = used so far)
  clairv     book quantile_q of the handoff count given the true a
q, q0 tuned per instance on separate days.
Usage (from svrpspd_wdro/): python scripts/tempo3_pool.py [n_inst=10 days=100 tune=40 workers=4 s=0.3]
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
from ev.fleet import FleetPlan, PolicyLibrary, CV, quantize_a, A_GRID  # noqa
from core.costs import LastMileCosts  # noqa

SHAPE = 1.0 / CV ** 2
A = np.array(A_GRID)
BG = np.log(A)
C0, C1 = 20.0, 30.0
FSB = LastMileCosts().F_standby
KTOPS = (2, 3, 4, 6)
QS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
Q0S = (0.0, 0.3, 0.5, 0.7)
M_OFF = 200


def prior_w(s):
    w = np.exp(-0.5 * (BG / s) ** 2)
    return w / w.sum()


def sim_day(fl, lib, a_true, base, prior, mode, pool=None, top=None,
            record=None, fixed_a=None):
    """mode: 'ppu' (unlimited, price H), 'book' (pool, price H - FSB).
    pool: initial booked count; top: (k_top, fn(u, post) -> new pool size).
    fixed_a: decide with this a instead of the posterior mean (offline).
    record: dict to fill with handoff counts by stop."""
    K = len(fl.routes)
    W = np.zeros(K)
    alive = [True] * K
    ll = np.zeros(len(A))
    cost = 0.0
    used = 0
    n_pool = np.inf if mode == "ppu" else pool
    if mode == "book":
        cost += C0 * pool
    for k in range(1, fl.m_max + 1):
        obs = []
        for r, rt in enumerate(fl.routes):
            if not alive[r] or k > rt.m:
                continue
            p = a_true * rt.pbar[k - 1] * base[r][0][k - 1]
            d = rt.dbar[k - 1] * base[r][1][k - 1]
            if rt.pbar[k - 1] > 0:
                ll += -SHAPE * BG - p / (A * rt.pbar[k - 1] * CV ** 2)
            W[r] += p - d
            obs.append(r)
        post = prior * np.exp(ll - ll.max())
        post /= post.sum()
        a_hat = fixed_a if fixed_a is not None else float(post @ A)
        for r in obs:
            rt = fl.routes[r]
            if W[r] > rt.B:
                cost += rt.E[k - 1]
                alive[r] = False
                continue
            if k == rt.m:
                alive[r] = False
                continue
            rv = lib.get(r, quantize_a(a_hat)).rv
            cont = rv.U(k + 1, W[r])
            rest = rt.R[k - 1] + rv.U(k + 1, 0.0)
            ho = rt.H[k - 1] - (0.0 if mode == "ppu" else FSB)
            if ho < cont and ho <= rest and used < n_pool:
                cost += ho
                used += 1
                alive[r] = False
            elif rest < cont:
                cost += rt.R[k - 1]
                W[r] = 0.0
        if record is not None:
            record[k] = used
        if top is not None and k == top[0]:
            new = top[1](used, post)
            if new > n_pool:
                cost += C1 * (new - n_pool)
                n_pool = new
    return cost, used


def draw_base(fl, rng):
    return [(rng.gamma(SHAPE, 1.0, rt.m) / SHAPE, rng.gamma(SHAPE, 1.0, rt.m) / SHAPE)
            for rt in fl.routes]


def offline_table(fl, lib, prior):
    """Per grid a: handoff totals N and handoffs by each k_top, unlimited
    pool, decisions with the true a (fixed_a)."""
    rng = np.random.default_rng(123)
    tab = {}
    for a in A:
        rows = []
        for _ in range(M_OFF):
            rec = {}
            _, n = sim_day(fl, lib, a, draw_base(fl, rng), prior, "ppu",
                           record=rec, fixed_a=a)
            rows.append([n] + [rec.get(k, n) for k in KTOPS])
        tab[a] = np.array(rows)
    return tab


def quantile_mix(samples, weights, q):
    """q-quantile of a mixture of empirical samples."""
    vals = np.concatenate(samples)
    w = np.concatenate([np.full(len(s), wt / len(s)) for s, wt in zip(samples, weights)])
    o = np.argsort(vals)
    c = np.cumsum(w[o])
    return float(vals[o][min(np.searchsorted(c, q * c[-1]), len(vals) - 1)])


def policies(tab, prior):
    """Booking rules as functions of (q, q0, k_top)."""
    Ntot = [tab[a][:, 0] for a in A]
    pw = list(prior)

    def static(q):
        return int(np.ceil(quantile_mix(Ntot, pw, q)))

    def topup(ki, q):
        def fn(u, post):
            rem = [tab[a][:, 0] - tab[a][:, 1 + ki] for a in A]
            return int(np.ceil(u + quantile_mix(rem, list(post), q)))
        return fn
    return static, topup


def run(path, days, tune, s):
    fl = FleetPlan.from_instance(path)
    lib = PolicyLibrary(fl, seed=1)
    prior = prior_w(s)
    tab = offline_table(fl, lib, prior)
    static, topup = policies(tab, prior)
    seed0 = abs(hash(fl.name)) % 10_000

    def evaluate(n_days, seed, rule):
        rng = np.random.default_rng(seed)
        tot = []
        for _ in range(n_days):
            a = float(rng.choice(A, p=prior))
            base = draw_base(fl, rng)
            tot.append(rule(a, base))
        return np.array(tot)

    rules = {}
    rules["ppu"] = lambda a, b: sim_day(fl, lib, a, b, prior, "ppu")[0]
    for q in QS:
        rules[("static", q)] = (lambda q: lambda a, b: sim_day(
            fl, lib, a, b, prior, "book", pool=static(q))[0])(q)
        rules[("clairv", q)] = (lambda q: lambda a, b: sim_day(
            fl, lib, a, b, prior, "book",
            pool=int(np.ceil(np.quantile(tab[quantize_a(a)][:, 0], q))))[0])(q)
        for ki, kt in enumerate(KTOPS):
            for q0 in Q0S:
                rules[("two", kt, q, q0)] = (lambda q, q0, ki, kt: lambda a, b: sim_day(
                    fl, lib, a, b, prior, "book",
                    pool=static(q0) if q0 > 0 else 0,
                    top=(kt, topup(ki, q)))[0])(q, q0, ki, kt)
    # tune on separate days
    tmean = {k: evaluate(tune, seed0 + 1, f).mean() for k, f in rules.items()}

    def best(pred):
        ks = [k for k in tmean if pred(k)]
        return min(ks, key=lambda k: tmean[k])
    chosen = {"ppu": "ppu",
              "static": best(lambda k: isinstance(k, tuple) and k[0] == "static"),
              "clairv": best(lambda k: isinstance(k, tuple) and k[0] == "clairv")}
    for kt in KTOPS:
        chosen[f"two_k{kt}"] = best(lambda k, kt=kt: isinstance(k, tuple) and k[0] == "two" and k[1] == kt)
    rows = []
    for name, key in chosen.items():
        c = evaluate(days, seed0 + 2, rules[key])
        rows.append(dict(instance=fl.name, K=len(fl.routes), s=s, policy=name,
                         param=str(key), mean=c.mean(), se=c.std() / np.sqrt(len(c))))
    return rows


if __name__ == "__main__":
    kw = dict(x.split("=") for x in sys.argv[1:])
    n, days, tune = int(kw.get("n_inst", 10)), int(kw.get("days", 100)), int(kw.get("tune", 40))
    wk, s = int(kw.get("workers", 4)), float(kw.get("s", 0.3))
    fam = kw.get("fam", "Dethloff")
    paths = [p for p in sorted((_W / "data" / fam).glob("*.vrpspd"))
             if (_W / "results" / "plans" / f"{p.stem}.json").exists()][:n]
    with ProcessPoolExecutor(max_workers=wk) as ex:
        res = list(ex.map(run, paths, [days] * len(paths), [tune] * len(paths), [s] * len(paths)))
    df = pd.DataFrame([r for rs in res for r in rs])
    (_W / "results" / "tempo3").mkdir(parents=True, exist_ok=True)
    df.to_csv(_W / "results" / "tempo3" / f"pool_{fam}_s{s}.csv", index=False)
    print(df.pivot_table(index="instance", columns="policy", values="mean").round(1).to_string())
    print(df.groupby("policy")["mean"].mean().round(2).to_string())
