"""Does Sinkhorn DRO collapse to Wasserstein DRO at a shifted radius?

This is the question that decides whether a Sinkhorn-ambiguity paper is viable,
and it is the same failure mode that killed `papers/wrnf/`: there, a type-1
Wasserstein ball over unbounded support turned out to add a *decision-
independent constant* to the sample average, so the robust decision was
identical to the SAA decision and the robustness was inert. If a Sinkhorn ball
likewise differs from a Wasserstein ball only by a reparametrisation of the
radius, then "use Sinkhorn instead" is a change of notation, not of model, and
the referee objection is fatal.

Concretely we test three nested claims, in increasing order of what would kill
the idea:

  C1  Sinkhorn is NOT a radius reparametrisation of Wasserstein.
      For each Sinkhorn radius there may exist a Wasserstein radius matching
      its *objective value* at one decision — that is trivially true by
      continuity. The question is whether one Wasserstein radius matches it at
      EVERY decision. If the residual S(x) - W_rho(x) is constant in x for some
      rho, the two models induce the same preference ordering over decisions and
      Sinkhorn buys nothing.

  C2  Sinkhorn's worst case is genuinely smooth in the decision, where
      Wasserstein's is kinked. This is the claimed differentiator: the entropic
      inner problem is a soft-max rather than a hard sup, so the value function
      should be differentiable in x while the Wasserstein one is piecewise
      linear. Measured here as the discrete second difference of the value.

  C3  With a DECISION-DEPENDENT reference measure the two disagree about the
      optimal decision, not merely about the objective value. This is the only
      thing that would make a decision-dependent Sinkhorn model do work that
      the published decision-dependent Wasserstein models do not.

RESULT (2026-07-27): C1 and C2 pass, **C3 fails**, and the idea was abandoned.
Sinkhorn does not collapse (best-matching Wasserstein radius leaves residual
sd 2.02, and the argmins differ) and its value is ~10.8x smoother in the
decision. But it never changes which cell gets selected: 0/6 flips across a
dispersion sweep, and 0 across a location sweep.

**Methodological finding worth keeping.** A location sweep CANNOT discriminate
between ambiguity models here, and this is structural rather than a numerical
accident: the newsvendor loss is translation-equivariant, so translating a
cell's samples translates the optimal decision by the same amount and leaves
the optimal value exactly unchanged. Both models are then indifferent between
cells by symmetry — `W_rho` gaps came out identically 0.0000 at every shift, and
the tiny non-zero Sinkhorn gaps (~1e-3) were support-truncation artefacts, not
signal. Only differences in **dispersion/shape** can discriminate. That matters
because `data_sources/probe_nyc_traffic_weather.py` found real weather shifts
the travel-time **mean without inflating the spread** — i.e. real data supplies
exactly the kind of difference that provably cannot distinguish these models.

Model.  Newsvendor / simple-recourse loss, the same family Wang-Gao-Xie use, so
the comparison is against their setting rather than a contrivance:

    f(x, z) = b * max(0, z - x) + h * max(0, x - z)

with b the shortage penalty and h the holding cost, x the stocking decision and
z the demand. One-dimensional z keeps both worst-case computations accurate to
grid resolution, which matters: a sloppy answer here is worse than none.

Sinkhorn worst case, from the strong dual (Wang, Gao & Xie, *Operations
Research* 2026, DOI 10.1287/opre.2023.0294, Theorem 1):

    S(x) = inf_{lambda>0} [ lambda*rho + lambda*eps * (1/N) sum_k
                            log E_{z~Q_k} exp( f(x,z) / (lambda*eps) ) ]

where the kernel Q_k has density proportional to exp(-c(zhat_k, z)/eps) against
a base measure. As lambda -> infinity this tends to the sample average; as
lambda -> 0+ the soft-max hardens and it tends to the worst case over the
support. Computed with logsumexp throughout, because exp(f/(lambda*eps))
overflows for small lambda*eps.

Wasserstein worst case, by brute force over transport plans so nothing depends
on a reformulation being right:

    W_rho(x) = max_pi sum_{k,g} pi[k,g] f(x, z_g)
               s.t.  sum_g pi[k,g] = 1/N,
                     sum_{k,g} pi[k,g] c(zhat_k, z_g) <= rho,  pi >= 0

Run:  python3 probes/sinkhorn_collapse_check.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

os.environ.setdefault("GRB_LICENSE_FILE", "/root/gurobi.lic")
import gurobipy as gp
from gurobipy import GRB

ENV = None
RNG = np.random.default_rng(20260727)

B_SHORT, H_HOLD = 8.0, 2.0


def loss(x, z):
    """Newsvendor loss; broadcasts over either argument."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    return B_SHORT * np.maximum(0.0, z - x) + H_HOLD * np.maximum(0.0, x - z)


# ----------------------------------------------------------------------------
def sinkhorn_worstcase(x, samples, rho, eps, grid, cost, log_base=None):
    """S(x) via the strong dual, minimised over the multiplier lambda."""
    # kernel: log q_k(g) = -c(zhat_k, z_g)/eps + log nu(g), normalised over g
    lq = -cost / eps
    if log_base is not None:
        lq = lq + log_base[None, :]
    lq = lq - logsumexp(lq, axis=1, keepdims=True)
    fz = loss(x, grid)

    def dual(log_lam):
        lam = np.exp(log_lam)
        t = lam * eps
        # (1/N) sum_k log E_{Q_k} exp(f/t), stable form
        inner = logsumexp(lq + fz[None, :] / t, axis=1).mean()
        return lam * rho + t * inner

    r = minimize_scalar(dual, bounds=(-9.0, 9.0), method="bounded",
                        options={"xatol": 1e-10})
    return float(r.fun)


def wasserstein_worstcase(x, samples, rho, grid, cost):
    """W_rho(x) by LP over transport plans on the discretised support."""
    N, G = len(samples), len(grid)
    fz = loss(x, grid)
    mdl = gp.Model(env=ENV)
    mdl.Params.OutputFlag = 0
    pi = mdl.addVars(N, G, lb=0.0)
    for k in range(N):
        mdl.addConstr(gp.quicksum(pi[k, g] for g in range(G)) == 1.0 / N)
    mdl.addConstr(gp.quicksum(float(cost[k, g]) * pi[k, g]
                              for k in range(N) for g in range(G)) <= rho)
    mdl.setObjective(gp.quicksum(float(fz[g]) * pi[k, g]
                                 for k in range(N) for g in range(G)), GRB.MAXIMIZE)
    mdl.optimize()
    assert mdl.Status == GRB.OPTIMAL, mdl.Status
    v = mdl.ObjVal
    mdl.dispose()
    return v


def build_support(lo, hi, n_grid, samples):
    grid = np.linspace(lo, hi, n_grid)
    grid = np.unique(np.concatenate([grid, np.asarray(samples, dtype=float)]))
    cost = np.abs(np.asarray(samples)[:, None] - grid[None, :])   # type-1, |.|
    return grid, cost


# ----------------------------------------------------------------------------
def main():
    global ENV
    ENV = gp.Env(params={"OutputFlag": 0})

    samples = np.array([4.0, 6.5, 9.0, 11.5, 14.0])
    grid, cost = build_support(0.0, 30.0, 601, samples)
    xs = np.linspace(2.0, 18.0, 81)
    eps = 1.0
    rho_s = 1.0

    print("=" * 76)
    print("Does Sinkhorn DRO collapse to Wasserstein DRO at a shifted radius?")
    print("=" * 76)
    print(f"newsvendor b={B_SHORT} h={H_HOLD}; samples {samples.tolist()}")
    print(f"support [0,30] on {len(grid)} points; Sinkhorn eps={eps} rho={rho_s}")

    S = np.array([sinkhorn_worstcase(x, samples, rho_s, eps, grid, cost) for x in xs])
    saa = np.array([loss(x, samples).mean() for x in xs])

    # ---- C1: is some single Wasserstein radius equivalent to Sinkhorn? -----
    print("\n--- C1: search for a Wasserstein radius that reproduces Sinkhorn ---")
    print(f"{'rho_W':>7} {'mean|S-W|':>11} {'sd(S-W)':>10} {'argmin W':>9} "
          f"{'argmin S':>9} {'same?':>6}")
    xS = xs[int(np.argmin(S))]
    best = None
    for rho_w in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        W = np.array([wasserstein_worstcase(x, samples, rho_w, grid, cost) for x in xs])
        d = S - W
        xW = xs[int(np.argmin(W))]
        same = abs(xW - xS) < 1e-9
        print(f"{rho_w:>7.2f} {np.abs(d).mean():>11.4f} {d.std():>10.5f} "
              f"{xW:>9.2f} {xS:>9.2f} {str(same):>6}")
        if best is None or d.std() < best[1]:
            best = (rho_w, d.std())
    print(f"\nbest-matching radius rho_W={best[0]} leaves residual sd {best[1]:.5f}")
    print("  A residual that is CONSTANT in x (sd ~ 0) would mean the two models")
    print("  rank decisions identically => Sinkhorn is a reparametrisation.")
    collapses = best[1] < 1e-6
    print(f"  C1: {'COLLAPSES -- fatal' if collapses else 'does NOT collapse'}")

    # ---- C2: smoothness in the decision ----------------------------------
    print("\n--- C2: smoothness of the value function in x ---")
    W1 = np.array([wasserstein_worstcase(x, samples, best[0], grid, cost) for x in xs])
    d2S = np.abs(np.diff(S, 2))
    d2W = np.abs(np.diff(W1, 2))
    print(f"  max |2nd difference|  Sinkhorn {d2S.max():.3e}   "
          f"Wasserstein {d2W.max():.3e}")
    print(f"  ratio (W/S)                    {d2W.max()/max(d2S.max(),1e-15):.1f}x")
    smoother = d2S.max() < d2W.max()
    print(f"  C2: Sinkhorn is {'SMOOTHER (differentiator holds)' if smoother else 'NOT smoother'}")

    # ---- C3: decision-dependent reference measure -------------------------
    # Two candidate cells (e.g. dispatch early vs late); the decision selects
    # which empirical conditional it faces. Emulates the routing mechanism.
    print("\n--- C3: decision-dependent reference measure, do argmins differ? ---")
    cellA = np.array([4.0, 5.0, 6.0, 7.0, 8.0])       # low-demand cell
    cellB = np.array([9.0, 11.0, 13.0, 15.0, 17.0])   # high-demand cell
    print(f"  cell A samples {cellA.tolist()}")
    print(f"  cell B samples {cellB.tolist()}")
    rows = []
    for name, smp in [("A", cellA), ("B", cellB)]:
        g, c = build_support(0.0, 30.0, 601, smp)
        Sv = np.array([sinkhorn_worstcase(x, smp, rho_s, eps, g, c) for x in xs])
        Wv = np.array([wasserstein_worstcase(x, smp, best[0], g, c) for x in xs])
        rows.append((name, xs[int(np.argmin(Sv))], Sv.min(),
                     xs[int(np.argmin(Wv))], Wv.min()))
    print(f"  {'cell':>5} {'x*_Sinkhorn':>12} {'val':>9} {'x*_Wass':>9} {'val':>9}")
    for n, xsk, vsk, xw, vw in rows:
        print(f"  {n:>5} {xsk:>12.2f} {vsk:>9.4f} {xw:>9.2f} {vw:>9.4f}")
    # If the model must CHOOSE the cell, it compares the two minima.
    pick_S = rows[0][0] if rows[0][2] < rows[1][2] else rows[1][0]
    pick_W = rows[0][0] if rows[0][4] < rows[1][4] else rows[1][0]
    print(f"  cell chosen under Sinkhorn: {pick_S}    under Wasserstein: {pick_W}")
    print(f"  C3: selection {'AGREES -- no added value from Sinkhorn here' if pick_S == pick_W else 'DIFFERS'}")

    # ---- C3b: dispersion is the ONLY discriminating dimension -------------
    # A location sweep is uninformative by symmetry (see module docstring), so
    # vary spread at fixed mean instead.
    print("\n--- C3b: same mean, different spread (the discriminating axis) ---")
    base = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    xs2 = np.linspace(0.0, 30.0, 151)
    print(f"  {'sd_A':>5} {'sd_B':>5} {'pick_S':>7} {'pick_W':>7} "
          f"{'gap_S':>10} {'gap_B':>10} {'flip?':>6}")
    flips = 0
    for sB in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        vals = {}
        for nm, smp in [("A", 12.0 + base * 2.0), ("B", 12.0 + base * sB)]:
            g, c = build_support(-10.0, 40.0, 1001, smp)
            Sv = np.array([sinkhorn_worstcase(x, smp, rho_s, eps, g, c) for x in xs2])
            Wv = np.array([wasserstein_worstcase(x, smp, best[0], g, c) for x in xs2])
            vals[nm] = (Sv.min(), Wv.min())
        gS = vals["A"][0] - vals["B"][0]
        gW = vals["A"][1] - vals["B"][1]
        pS, pW = ("A" if gS < 0 else "B"), ("A" if gW < 0 else "B")
        f = (pS != pW) and abs(gS) > 1e-3 and abs(gW) > 1e-3
        flips += f
        print(f"  {2.0:>5.1f} {sB:>5.1f} {pS:>7} {pW:>7} {gS:>10.4f} {gW:>10.4f} "
              f"{str(f):>6}")
    print(f"  selection flips attributable to the ambiguity model: {flips}/6")

    print("\n" + "=" * 76)
    print("VERDICT INPUTS")
    print(f"  C1 not a reparametrisation      : {not collapses}")
    print(f"  C2 smoother in x                : {smoother}")
    print(f"  C3  changes the selection       : {pick_S != pick_W}")
    print(f"  C3b changes it on the only axis")
    print(f"      that can discriminate       : {flips > 0}")
    print("  C3/C3b are the 'so what'. Both failing means Sinkhorn buys a")
    print("  smoother objective but never a different decision here.")
    print("=" * 76)


if __name__ == "__main__":
    main()
