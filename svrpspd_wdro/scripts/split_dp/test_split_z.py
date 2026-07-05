"""Quick probe: for CON3-0, what K (vehicles) and distance does the split give at each z?
Run from your scripts/ dir:  python probe_z.py [path_to_instance]
"""
import sys
import benchmark_c4 as B
from fast_split import TourPrecomp, split_from_matrices

dr = B._load_runner()
path = sys.argv[1] if len(sys.argv) > 1 else r"data\Dethloff\CON3-0.vrpspd"
D, db, pb, Q, n = B.load_instance(path, dr=dr)
scale = dr.parse_dethloff(path)[4]
tours = B.candidate_tours(D, db, pb, Q, n, dr=dr, k_starts=8, seed=0)

print(f"{path}   (omega_V will multiply each vehicle in TBC)")
print(f"{'z':>5} {'K':>4} {'dist':>10}   best tour")
for z in [1.5, 2.0, 2.25, 2.5, 2.75, 3.0]:
    best = None
    for tid, tour in tours.items():
        pc = TourPrecomp(tour, D, db, pb, dr.CV*db, dr.CV*pb, dr.RHO, 12)
        mask = pc.admissible_mask(z, Q)
        c, rp = split_from_matrices(pc.dist, mask, 12)
        if rp and (best is None or c < best[0]):
            best = (c, [tour[i:j] for (i, j) in rp], tid)
    if best is None:
        print(f"{z:>5.2f}  (no feasible split)")
        continue
    plan = best[1]; K = len(plan)
    dist = sum(dr.route_cost(r, D) for r in plan) / scale
    print(f"{z:>5.2f} {K:>4d} {dist:>10.1f}   {best[2]}")