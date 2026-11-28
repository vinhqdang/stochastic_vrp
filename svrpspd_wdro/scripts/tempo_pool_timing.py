"""Wall-clock costs of the TEMPO pooling estimators, per instance.

Usage (from svrpspd_wdro/):
    python scripts/tempo_pool_timing.py fam=Dethloff [n_inst=5] [tag=]

Per instance (sequential, one thread per process, time.perf_counter):
  (a) lib_seconds: PolicyLibrary build, forcing every route and every grid
      value of ev.fleet.A_GRID through PolicyLibrary.get.
  (b) hier_ms: mean over 200 calls of hier_means(Lz, 0.2, 0.2), Lz built from
      10 random pickups per zone (customers with pbar > 0, drawn with
      replacement, gamma pickups at scale 1). One untimed warm-up call.
  (c) history_ms: mean over 200 calls of history_estimate with H = 60 (mixed
      scenario, fresh seeded generator per call; one untimed warm-up).

Output: results/tempo_pool/timing_<fam><tag>.csv with columns instance, K,
n_cust, Z, lib_seconds, n_lib_entries, hier_ms, history_ms.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import csv
import time
from pathlib import Path

import numpy as np

_W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_W))
from ev.fleet import A_GRID, CV, FleetPlan, PolicyLibrary, parse_instance  # noqa: E402
from ev.pooling import (BG, SCENARIOS, build_structure, history_estimate,  # noqa: E402
                        hier_means, loglik)

OUT = _W / "results" / "tempo_pool"
COLUMNS = ["instance", "K", "n_cust", "Z", "lib_seconds", "n_lib_entries",
           "hier_ms", "history_ms"]
N_CALLS = 200
H_TIMING = 60


def time_instance(path):
    path = Path(path)
    fl = FleetPlan.from_instance(path)
    D = parse_instance(path)[0]
    st = build_structure(fl, D)

    # (a) library build, every route x every grid value
    t0 = time.perf_counter()
    lib = PolicyLibrary(fl, seed=1)
    for r in range(st.K):
        for a in A_GRID:
            lib.get(r, a)
    lib_seconds = time.perf_counter() - t0
    n_lib = len(lib._pol)

    # (b) hier_means on Lz from 10 random pickups per zone
    rng = np.random.default_rng(12345)
    shape = 1.0 / CV ** 2
    Lz = np.zeros((st.Z, len(BG)))
    for z in range(st.Z):
        idx = np.where((st.zc == z) & (st.pb > 0))[0]
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=10, replace=True)
        pbar = st.pb[pick]
        p = pbar * rng.gamma(shape, 1.0, 10) / shape
        for pi, bi in zip(p, pbar):
            Lz[z] += loglik(pi, bi)
    hier_means(Lz, 0.2, 0.2)                       # warm-up
    t0 = time.perf_counter()
    for _ in range(N_CALLS):
        hier_means(Lz, 0.2, 0.2)
    hier_ms = (time.perf_counter() - t0) / N_CALLS * 1e3

    # (c) history_estimate with H = 60
    scn = SCENARIOS["mixed"]
    def _hist(seed):
        return history_estimate(st.zc, st.sk, st.pb, st.Z, scn, H_TIMING,
                                np.random.default_rng(seed), zct=st.zct,
                                Zt=st.Zt)
    _hist(0)                                       # warm-up
    t0 = time.perf_counter()
    for i in range(N_CALLS):
        _hist(i)
    history_ms = (time.perf_counter() - t0) / N_CALLS * 1e3

    return {"instance": path.stem, "K": st.K, "n_cust": len(st.zc),
            "Z": st.Z, "lib_seconds": round(lib_seconds, 4),
            "n_lib_entries": n_lib, "hier_ms": round(hier_ms, 4),
            "history_ms": round(history_ms, 4)}


def main(kw):
    fam = kw.get("fam", "Dethloff")
    tag = kw.get("tag", "")
    n_inst = int(kw.get("n_inst", 5))
    paths = [p for p in sorted((_W / "data" / fam).glob("*.vrpspd"))
             if (_W / "results" / "plans" / f"{p.stem}.json").exists()]
    paths = paths[:n_inst]
    if not paths:
        raise SystemExit(f"no instances with cached plans for {fam}")
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"timing_{fam}{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        for p in paths:
            row = time_instance(p)
            wr.writerow(row)
            fh.flush()
            print(row, flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    main(kw)
