"""Reference computations for the bit-identity test (run with either code tree on sys.path)."""

import numpy as np

from sortie.instances import load_x641, make_instance
from sortie.search import construct_initial, solve_fair
from sortie.sim import evaluate_plans, simulate
from sortie.tsp import TourSolver, tour_stats

PLAN = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11], [12, 13, 14, 15]]


def compute(data_path):
    coords, demand = load_x641(data_path)
    inst = make_instance(coords, demand, 15, 0, m=5)
    inst20 = make_instance(coords, demand, 20, 1, m=5)
    rng = np.random.default_rng(5)
    out = {}

    tours = [[1, 2, 3], [4, 5, 6, 7, 8], list(range(9, 16))]
    tours += [rng.permutation(np.arange(1, 16)).tolist() for _ in range(3)]
    out["tour_stats"] = [[float(x) for x in tour_stats(t, inst.C)] for t in tours]

    sets15 = [list(range(1, 8)), list(range(1, 16)), [2, 5, 9, 11, 14]]
    sets20 = [list(range(1, 21)), sorted(rng.permutation(np.arange(1, 21))[:17].tolist())]
    tsp = {}
    for kappa in (0.0, 1.0, 2.0):
        for label, I, sets in (("n15", inst, sets15), ("n20", inst20, sets20)):
            solver = TourSolver(I.C, kappa, 0.3)
            for idx, S in enumerate(sets):
                tour, L, W = solver.solve(S)
                tsp[f"{label}|k{kappa:g}|{idx}"] = [[int(x) for x in tour], float(L), float(W)]
    out["tsp"] = tsp

    X, Y = simulate(inst, PLAN, v=0.3, sY=0.15, S=300, seed=5)
    out["sim_X"] = X.tolist()
    out["sim_Y"] = Y.tolist()
    ev = evaluate_plans(inst, [PLAN], 0.3, 0.15, S=300, seed=2)[0]
    out["eval"] = {k: ev[k] for k in ("E_range", "E_gmd", "E_max", "E_var", "CVaR90_range",
                                      "W_driver", "L_driver", "mean_total")}

    start = construct_initial(inst, np.random.default_rng(3))
    budget = 1.10 * sum(tour_stats(r, inst.C)[0] for r in start)
    runs = {
        "pboi_k1": dict(mode="pboi", measure="range", kappa=1.0, consistency="lower",
                        max_iters=30, screen="price"),
        "pboi_free": dict(mode="pboi", measure="gmd", kappa=0.0, consistency="free",
                          max_iters=15, screen="price"),
        "saa_k0": dict(mode="saa", measure="max", kappa=0.0, consistency="lower",
                       max_iters=10, screen="price"),
        "det_random": dict(mode="det", measure="range", kappa=0.0, consistency="lower",
                           max_iters=20, screen="random"),
    }
    for name, spec in runs.items():
        res = solve_fair(inst, start, spec["mode"], spec["measure"], 0.3, spec["kappa"], budget,
                         1e6, 11, screen=spec["screen"], consistency=spec["consistency"],
                         max_iters=spec["max_iters"])
        out[name] = {"value": float(res["value"]), "routes": res["routes"],
                     "L": [float(x) for x in res["L"]], "W": [float(x) for x in res["W"]],
                     "iters": int(res["iters"]), "n_exact": int(res["n_exact_evals"])}
    return out
