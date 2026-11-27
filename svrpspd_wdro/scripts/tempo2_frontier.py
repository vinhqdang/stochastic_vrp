"""tempo2_frontier.py: replans on harmless days against cost on harmful
days, sweeping the significance level of the statistical triggers.

Usage (from svrpspd_wdro/): python scripts/tempo2_frontier.py [n_inst=10 days=40 workers=4]
Output: results/tempo2/frontier.csv
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
from ev.fleet import (FleetPlan, PolicyLibrary, sample_day, simulate_fleet_day,
                      ValueInnovation, ModelFitLR, CostBased, Never)

SCEN = ["null", "light", "heavy_del", "surge20", "surge30", "surge50"]
ALPHAS = [0.05, 0.2, 0.5, 0.8]


def run(path, days):
    fl = FleetPlan.from_instance(path)
    lib = PolicyLibrary(fl, seed=1)
    rows = []
    for s in SCEN:
        rng = np.random.default_rng(1000 + SCEN.index(s))
        dlist = [sample_day(fl, s, rng) for _ in range(days)]
        trig = [("Never", None, Never())]
        trig += [("VI", a, ValueInnovation(alpha=a)) for a in ALPHAS]
        trig += [("LR", a, ModelFitLR(alpha=a)) for a in ALPHAS]
        for C in (0.0, 5.0, 20.0):
            trig.append(("CostBased", C, CostBased(C_rp=C)))
        for name, par, t in trig:
            for d, day in enumerate(dlist):
                C = par if name == "CostBased" else 0.0
                out = simulate_fleet_day(lib, day, t, C)
                rows.append(dict(instance=fl.name, scenario=s, trigger=name,
                                 param=par, day=d,
                                 recourse=out["recourse_cost"],
                                 replans=out["n_replans"]))
    return rows


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:])
    n, days, wk = int(kw.get("n_inst", 10)), int(kw.get("days", 40)), int(kw.get("workers", 4))
    paths = sorted((_W / "data" / "Dethloff").glob("*.vrpspd"))
    paths = [p for p in paths if (_W / "results" / "plans" / f"{p.stem}.json").exists()][:n]
    with ProcessPoolExecutor(max_workers=wk) as ex:
        res = list(ex.map(run, paths, [days] * len(paths)))
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(_W / "results" / "tempo2" / "frontier.csv", index=False)
    print("done", len(df))
