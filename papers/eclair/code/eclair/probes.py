"""The three probe tiers (PROJECT.md §3 Step 2), as real solver calls.

Every probe returns (alarm: bool, cost: float) where cost is the
measured wall-clock seconds of ALL compute the probe consumed (brute
force + CP-SAT solves) — the "probe-seconds" currency of the paper.

Tier A  micro-oracle: brute-force the SPEC (independent checker) on an
        exhaustively enumerable instance; alarm iff the candidate's
        CP-SAT optimum differs. Expensive, near-exact.
Tier B  metamorphic relation: draw a mid instance and one of the
        family's provably valid MRs; solve the candidate on both
        sides; alarm iff the asserted direction is violated. Under
        faithfulness the alarm probability is EXACTLY zero (both
        solves exact, MR valid at spec level) — a hard certificate.
Tier C  cross-check: solve the candidate and up to 3 alive partners
        on a fresh mid instance; alarm iff a partner-majority value
        exists (>= 2 agreeing) and the candidate deviates from it.
        Cheap-ish, noisy, entangled with pool composition
        (PROJECT.md §5.3) — calibrated conservatively.
"""

from __future__ import annotations

import time


def solve_value(model):
    """Exact CP-SAT solve; optimal objective as int, or None if infeasible."""
    ok = model.solve(solver="ortools")
    if not ok:
        return None
    return int(round(float(model.objective_value())))


def brute_optimum(spec, params):
    """Exhaustive spec-level optimum on a micro instance (the oracle)."""
    best = None
    for a in spec.enum(params):
        if spec.checker(params, a):
            v = spec.objective(params, a)
            if best is None:
                best = v
            elif spec.sense == "max":
                best = max(best, v)
            else:
                best = min(best, v)
    return best


def _as_num(v, sense):
    if v is not None:
        return v
    return float("-inf") if sense == "max" else float("inf")


def _cmp_holds(v1, v2, cmp, sense):
    """Does opt(I2)=v2 CMP opt(I1)=v1 hold? (None = infeasible)"""
    a, b = _as_num(v1, sense), _as_num(v2, sense)
    if cmp == "ge":
        return b >= a
    if cmp == "le":
        return b <= a
    return b == a


def tier_a(cand, rng):
    t0 = time.perf_counter()
    params = cand.spec.gen(rng, "micro")
    truth = brute_optimum(cand.spec, params)
    got = solve_value(cand.build(params))
    return got != truth, time.perf_counter() - t0


def tier_b(cand, rng):
    t0 = time.perf_counter()
    spec = cand.spec
    params = spec.gen(rng, "mid")
    _, mr_fn = spec.mrs[rng.randrange(len(spec.mrs))]
    params2, cmp = mr_fn(params, rng)
    v1 = solve_value(cand.build(params))
    v2 = solve_value(cand.build(params2))
    return not _cmp_holds(v1, v2, cmp, spec.sense), time.perf_counter() - t0


def tier_c(cand, alive_pool, rng, max_partners=3):
    t0 = time.perf_counter()
    spec = cand.spec
    partners = [c for c in alive_pool if c.cid != cand.cid]
    rng.shuffle(partners)
    partners = partners[:max_partners]
    if len(partners) < 2:
        return False, time.perf_counter() - t0     # no majority possible
    params = spec.gen(rng, "mid")
    mine = solve_value(cand.build(params))
    votes = {}
    for p in partners:
        v = solve_value(p.build(params))
        votes[v] = votes.get(v, 0) + 1
    mode, count = max(votes.items(), key=lambda kv: kv[1])
    alarm = count >= 2 and mine != mode
    return alarm, time.perf_counter() - t0


TIER_FNS = {"A": tier_a, "B": tier_b, "C": tier_c}


def estimate_error_rate(cand, rng, n=400, z=1.96):
    """Independent Monte-Carlo estimate of

        err(m) = P_{I ~ D_mu}( v_m(I) != v*(I) )

    the quantity Theorem 3's null is stated in terms of, on FRESH
    micro instances drawn from the same distribution the tier-A oracle
    uses. Returns (err_hat, n, wilson_lo, wilson_hi).

    This is an evaluation instrument, not part of the protocol: it is
    what lets us report whether a CERTIFIED candidate actually
    satisfies err < eps, instead of inferring it from a construction
    label (review R4.1)."""
    bad = 0
    for _ in range(n):
        params = cand.spec.gen(rng, "micro")
        try:
            got = solve_value(cand.build(params))
        except Exception:
            bad += 1
            continue
        if got != brute_optimum(cand.spec, params):
            bad += 1
    p = bad / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, n, max(0.0, c - h), min(1.0, c + h)
