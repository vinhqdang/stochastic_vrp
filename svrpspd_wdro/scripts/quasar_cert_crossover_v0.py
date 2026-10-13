import json, glob, math, sys, multiprocessing as mp
import numpy as np
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[1])
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import run_quasar_m3 as M3
from quantum.experiments import route_peak
import dethloff_runner as DR
N = 20000
def work(f):
    plan = json.load(open(f))
    dsc, psc = M3.draw_scenarios(plan, N, DR.RHO, DR.SEED)
    out = {}
    for g in ("Det", "SAA"):
        out[g] = [float((route_peak(r, dsc, psc) > plan["Q"]).mean()) for r in plan["res"][g]["plan"] if r]
    return plan["name"], out
if __name__ == "__main__":
    files = sorted(glob.glob("results/plans_Region/*.json"))
    with mp.Pool(4) as pool:
        res = dict(pool.map(work, files))
    json.dump(res, open("results/quasar/region_route_risks.json", "w"))
    delta = 0.05
    print("region gate d p_med p_max v_max | eps classical_H classical_B quantum ratio_H ratio_B")
    for reg in ("VNN", "GD"):
        for g in ("Det", "SAA"):
            ps = np.array([p for k, v in res.items() if k.startswith(reg + "-") for p in v[g]])
            d = len(ps); v = float(np.max(ps * (1 - ps)))
            for eps in (1e-2, 1e-3, 1e-4):
                L = math.log(2 * d / delta)
                cH = L / (2 * eps * eps); cB = (2 * v + 2 * eps / 3) * L / (eps * eps)
                q = 2 * math.sqrt(d) * math.log(d / delta) / eps
                print(f"{reg} {g} {d} {np.median(ps):.3f} {ps.max():.3f} {v:.3f} | {eps:g} {cH:.3g} {cB:.3g} {q:.3g} {q/cH:.3g} {q/cB:.3g}", flush=True)
