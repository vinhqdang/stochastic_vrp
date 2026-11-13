"""RARECERT (PEIS version): check the spectral conditions of the efficiency theorems
on every route of the region plans (results/plans_Region, gate SAA).

For each route and copula loading rho, and for every stage k = 0..L, the stage-k
delivery block (customers k+1..L) and pickup block (customers 1..k) have top
eigenvalues lam_d[k], lam_p[k] and second eigenvalues mu_d[k], mu_p[k] of
Theta^{1/2} (rho 11' + (1-rho) I) Theta^{1/2}. With lam_k = max(lam_d, lam_p),
varrho_k = min/max and nu_k = max(mu_d, mu_p)/lam_k, Theorem 3 of the manuscript
covers stage k if
  (a) lam_k <= lam*/2,                                   (defensive component)
  (b) varrho_k <= 1/3 and nu_k <= 1/3,                   (dominant-block component)
  (c) lam_k <= lam*/sqrt(2) and nu_k <= 1/2.             (both block components)
and in general if psi(varrho_k, nu_k) >= 4 lam_k / lam* (rarecert.covering.psi).
thm3_massw: the same with mixture weights proportional to exp(-c/lam_m).
Theorem 2: the single shift at the dominant end is efficient if varrho <= 1/4 and the
dominant end block has mu <= lam*/4, and not efficient if varrho > 1/4.
The conditions do not depend on cv (it scales every theta by the same factor).
Output: results/rarecert/peis/conditions.csv and conditions_summary.md.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_quasar_rare as rq  # noqa: E402
from rarecert.smis import block_scale  # noqa: E402
from rarecert.covering import psi  # noqa: E402

OUT = ROOT / "results" / "rarecert" / "peis"
RHOS = (0.6, 0.3, 0.8)


def route_conditions(dbar, pbar, rho):
    d = np.asarray(dbar, float)
    p = np.asarray(pbar, float)
    L = d.size
    lam_d = np.zeros(L + 1); mu_d = np.zeros(L + 1)
    lam_p = np.zeros(L + 1); mu_p = np.zeros(L + 1)
    for k in range(L + 1):
        th = d[k:][d[k:] > 0]
        if th.size:
            lam_d[k], mu_d[k], _ = block_scale(th, rho)
        th = p[:k][p[:k] > 0]
        if th.size:
            lam_p[k], mu_p[k], _ = block_scale(th, rho)
    lam_star = max(lam_d[0], lam_p[L])
    lam_k = np.maximum(lam_d, lam_p)
    lo = np.minimum(lam_d, lam_p)
    ok = lam_k > 0
    vr = np.where(ok, lo / np.where(ok, lam_k, 1), 0.0)
    nu = np.where(ok, np.maximum(mu_d, mu_p) / np.where(ok, lam_k, 1), 0.0)
    ca = lam_k <= lam_star / 2 + 1e-12
    cb = (vr <= 1 / 3) & (nu <= 1 / 3)
    cc = (lam_k <= lam_star / math.sqrt(2) + 1e-12) & (nu <= 0.5)
    covered = ca | cb | cc
    # exact covering constant for the stages the closed-form conditions miss
    cpsi = covered.copy()
    psi_min_margin = math.inf
    for k in np.nonzero(~covered & ok)[0]:
        val = psi(float(vr[k]), float(nu[k]), n=201)
        need = 4 * lam_k[k] / lam_star
        psi_min_margin = min(psi_min_margin, val - need)
        cpsi[k] = val >= need - 1e-9
    # weights proportional to exp(-c/lam_m) (mass-proportional, as in SMIS):
    # penalties 2(1 - r) on the dominant block and 2(1/varrho - r) on the other
    cmass = ca.copy()
    for k in np.nonzero(~ca & ok)[0]:
        r = lam_k[k] / lam_star
        pl = 2 * (1 / vr[k] - r) if vr[k] > 0 else 0.0
        val = psi(float(vr[k]), float(nu[k]), n=201, pen_h=2 * (1 - r), pen_lo=pl)
        cmass[k] = val >= 4 * r - 1e-9
    dom_d = lam_d[0] >= lam_p[L]
    other = lam_p[L] if dom_d else lam_d[0]
    mu_dom = mu_d[0] if dom_d else mu_p[L]
    varrho = other / lam_star if lam_star > 0 else math.nan
    return {
        "lam_star": lam_star, "varrho": varrho,
        "nu_end": mu_dom / lam_star if lam_star > 0 else math.nan,
        "thm2_eff": bool(varrho <= 0.25 and mu_dom <= lam_star / 4),
        "thm2_ineff": bool(varrho > 0.25),
        "thm3_closed": bool(covered.all()),
        "thm3": bool(cpsi.all()), "psi_margin": psi_min_margin, "thm3_massw": bool(cmass.all()),
        "n_stage_a": int(ca.sum()), "n_stage_b_only": int((cb & ~ca).sum()),
        "n_stage_c_only": int((cc & ~ca & ~cb).sum()), "n_uncovered": int((~cpsi).sum()),
        "end_blocks_gap3": int((mu_d[0] <= lam_d[0] / 3) + (mu_p[L] <= lam_p[L] / 3)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = rq.collect_routes(rq.PLANS, min_len=2)
    rows = []
    for reg, routes in pool.items():
        for r in routes:
            for rho in RHOS:
                c = route_conditions(r["dbar"], r["pbar"], rho)
                rows.append({"region": reg, "instance": r["instance"], "route": r["route_index"],
                             "L": len(r["nodes"]), "rho": rho, **c})
    with open(OUT / "conditions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    lines = ["# Theorem conditions on the region plans (routes with >= 2 stops)", "",
             "| rho | region | routes | Thm 2 eff. | Thm 2 ineff. | Thm 3 closed form | Thm 3 (psi) | Thm 3 mass weights | stages needing (b) | (c) | uncovered stages | median varrho |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rho in RHOS:
        for reg in ("VNN", "GD", "all"):
            s = [x for x in rows if x["rho"] == rho and (reg == "all" or x["region"] == reg)]
            n = len(s)
            lines.append(
                f"| {rho} | {reg} | {n} | {sum(x['thm2_eff'] for x in s)/n:.1%} | "
                f"{sum(x['thm2_ineff'] for x in s)/n:.1%} | {sum(x['thm3_closed'] for x in s)/n:.1%} | {sum(x['thm3'] for x in s)/n:.1%} | {sum(x['thm3_massw'] for x in s)/n:.1%} | "
                f"{sum(x['n_stage_b_only'] for x in s)} | {sum(x['n_stage_c_only'] for x in s)} | "
                f"{sum(x['n_uncovered'] for x in s)} | {np.median([x['varrho'] for x in s]):.3f} |")
    (OUT / "conditions_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
