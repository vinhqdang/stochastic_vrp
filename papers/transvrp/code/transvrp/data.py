"""Instance, plan and scenario access, copied in spirit from
svrpspd_wdro/scripts/dethloff_runner.py and run_realistic_eval.py (parser and
sampler are verbatim logic; the plan loader reads the copied plans/ folder)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.stats as st

HERE = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data" / "Dethloff"
PLANS_DIR = HERE / "plans"

SEED = 7
CV = 0.30        # coefficient of variation of each demand component
RHO = 0.6        # Gaussian-copula equicorrelation
DIST = "gamma"


def _lines_after(txt, tag):
    out, cap = [], False
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if not cap:
            if s.upper().startswith(tag):
                cap = True
            continue
        if any(ch.isalpha() for ch in s):
            break
        out.append(s)
    return out


def _header(txt, key, cast):
    pat = r"-?\d+" if cast is int else r"-?\d+\.?\d*"
    for line in txt.splitlines():
        if line.strip().upper().startswith(key):
            m = re.findall(pat, line)
            if m:
                return cast(m[-1])
    return None


def parse_dethloff(path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    n = _header(txt, "DIMENSION", int)
    Q = _header(txt, "CAPACITY", float)
    toks = []
    for row in _lines_after(txt, "EDGE_WEIGHT_SECTION"):
        toks.extend(row.split())
    vals = [int(float(t)) for t in toks]
    if n is None or len(vals) != n * n:
        raise ValueError("need the full n*n distance matrix")
    D = np.array(vals, dtype=np.int64).reshape(n, n)
    return D, Q, n, 10000


def _marginal_ppf(u, mu, cv, dist):
    if dist == "lognormal":
        s2 = math.log(1.0 + cv * cv)
        return np.exp((math.log(mu) - 0.5 * s2) + math.sqrt(s2) * st.norm.ppf(u))
    k = 1.0 / (cv * cv)                      # gamma (default)
    return st.gamma.ppf(u, k, scale=mu / k)


def sample_demands(mean, n, N, cv, dist, rng, rho=RHO):
    """(N, n) non-negative demands; Gaussian-copula equicorrelation rho over
    the active customers, marginals gamma (or lognormal) with mean `mean`, cv."""
    out = np.zeros((N, n))
    active = [i for i in range(1, n) if mean[i] > 0]
    if not active:
        return out
    if rho and rho > 0.0:
        m = len(active)
        Sigma = np.full((m, m), float(rho))
        np.fill_diagonal(Sigma, 1.0)
        L = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal((N, m)) @ L.T
        U = np.clip(st.norm.cdf(Z), 1e-12, 1.0 - 1e-12)
        for j, i in enumerate(active):
            out[:, i] = _marginal_ppf(U[:, j], mean[i], cv, dist)
        return out
    for i in active:
        mu = mean[i]
        k = 1.0 / (cv * cv)
        out[:, i] = rng.gamma(k, mu / k, N)
    return out


def _active_customers(mean) -> list[int]:
    return [i for i in range(1, len(mean)) if mean[i] > 0]


@dataclass(frozen=True)
class World:
    """Copula of one generated world (EXPERIMENT_C.md 1.4). Marginals are A's
    (gamma, CV 0.3, same means) in every world. ``shared`` selects one equicorrelated
    copula over (active deliveries, then active pickups) instead of two separate ones."""
    code: str
    rho_d: float
    rho_p: float
    shared: bool


WORLDS: dict[str, World] = {
    "W0": World("W0", 0.0, 0.0, False),
    "W3": World("W3", 0.3, 0.3, False),
    "W6": World("W6", 0.6, 0.6, False),
    "W9": World("W9", 0.9, 0.9, False),
    "WS": World("WS", 0.6, 0.6, True),
}
WORLD_SEED_CODE = {"W0": 10, "W3": 13, "W6": 16, "W9": 19, "WS": 26}


def _shared_demands(dbar, pbar, N, rho, rng):
    """One equicorrelated Gaussian copula (correlation rho for every pair) over the
    active deliveries followed by the active pickups; one standard_normal draw."""
    act_d, act_p = _active_customers(dbar), _active_customers(pbar)
    assert len(act_d) == len(act_p), "WS draws 2m' scores: active delivery and pickup counts must match"
    dsc = np.zeros((N, len(dbar)))
    psc = np.zeros((N, len(pbar)))
    m_d = len(act_d)
    m = m_d + len(act_p)
    if m == 0:
        return dsc, psc
    Sigma = np.full((m, m), float(rho))
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((N, m)) @ L.T
    U = np.clip(st.norm.cdf(Z), 1e-12, 1.0 - 1e-12)
    for j, i in enumerate(act_d):
        dsc[:, i] = _marginal_ppf(U[:, j], dbar[i], CV, DIST)
    for j, i in enumerate(act_p):
        psc[:, i] = _marginal_ppf(U[:, m_d + j], pbar[i], CV, DIST)
    return dsc, psc


def stable_seed(name: str) -> int:
    return SEED + int(hashlib.md5(name.encode()).hexdigest(), 16) % 10_000


def load_plans(name: str) -> dict:
    d = json.loads((PLANS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    d["dbar"] = np.array(d["dbar"], dtype=float)
    d["pbar"] = np.array(d["pbar"], dtype=float)
    return d


def gen_scenarios(dbar, pbar, N, seed, rho=None, world=None):
    """(deliveries, pickups) scenarios. ``world=None`` (rho default RHO) is the original
    generator: separate copulas, rho for both, deliveries drawn before pickups. A
    ``World`` (or its code) takes its copula from the registry; an explicit ``rho`` that
    differs from the world's copula is a ValueError (EXPERIMENT_C 1.4)."""
    if isinstance(world, str):
        world = WORLDS[world]
    if world is not None and rho is not None and (rho != world.rho_d or rho != world.rho_p):
        raise ValueError(f"rho={rho} conflicts with world {world.code} "
                         f"(rho_d={world.rho_d}, rho_p={world.rho_p})")
    if world is not None and world.shared and world.rho_d > 0.0:
        return _shared_demands(dbar, pbar, N, world.rho_d, np.random.default_rng(seed))
    if world is None and rho is None:
        rho = RHO
    rho_d, rho_p = (rho, rho) if world is None else (world.rho_d, world.rho_p)
    rng = np.random.default_rng(seed)
    dsc = sample_demands(dbar, len(dbar), N, CV, DIST, rng, rho=rho_d)
    psc = sample_demands(pbar, len(pbar), N, CV, DIST, rng, rho=rho_p)
    return dsc, psc


def route_problem(name, plan_key="Det", n_train=500, n_test=2000):
    """Yield per-route problems (route, g_train, g_test, B, H, E) exactly as
    run_realistic_eval._eval_route_realistic builds them."""
    from .baton.costs import LastMileCosts, route_cost_schedules
    D, Q, n, scale = parse_dethloff(DATA_DIR / f"{name}.vrpspd")
    sol = load_plans(name)
    dbar, pbar = sol["dbar"], sol["pbar"]
    seed = stable_seed(name)
    dtr, ptr = gen_scenarios(dbar, pbar, n_train, seed)
    dte, pte = gen_scenarios(dbar, pbar, n_test, seed + 99_991)
    costs = LastMileCosts()
    for ri, route in enumerate(sol["res"][plan_key]["plan"]):
        if not route:
            continue
        r = np.array(route)
        B = float(Q - dbar[r].sum())
        H, E = route_cost_schedules(route, D, scale, costs)
        yield dict(route_id=ri, route=route, B=B, H=H, E=E,
                   g_train=ptr[:, r] - dtr[:, r], g_test=pte[:, r] - dte[:, r])
