"""TEMPO pooling diagnostics: squared decision-factor error by stop index.

Usage (from svrpspd_wdro/):
    python scripts/tempo_pool_trace.py fam=Dethloff [n_inst=all] [days=100]
        [scen=mixed] [sc=<float> sz=<float>] [workers=4] [tag=]

For every instance, builds the zone structure, the policy library and the
H_HISTORY history estimate exactly as scripts/tempo_pool_eval.py does, draws
`days` fleet days from the same seeds, and runs the estimator policies
route, full, zone, hier, hier_eb and mle with a trace dict per policy
(accumulated over the days). The trace gives, per stop index k, the sum of
squared decision-factor errors against the known decision factor (sse) and
the number of decisions (n).

Output: results/tempo_pool/trace_<fam><tag>.csv with columns
instance, scenario, policy, k, sse, n. Parallel over instances.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import csv
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_W))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ev.fleet import FleetPlan, PolicyLibrary, parse_instance  # noqa: E402
from ev.pooling import (SCENARIOS, H_HISTORY, build_structure, draw_day,  # noqa: E402
                        history_estimate, simulate_policy)
from tempo_pool_eval import adhoc_scenario  # noqa: E402

OUT = _W / "results" / "tempo_pool"
TRACE_POLS = ["route", "full", "zone", "hier", "hier_eb", "mle"]
COLUMNS = ["instance", "scenario", "policy", "k", "sse", "n"]


def run_instance(path, scen_list, days):
    """Trace rows for one instance over the given (name, Scenario) list."""
    path = Path(path)
    fl = FleetPlan.from_instance(path)
    D = parse_instance(path)[0]
    rows = []
    for name, scn in scen_list:
        alt = scn.zones_true == "alt"
        st = build_structure(fl, D, alt=alt)
        lib = PolicyLibrary(fl, seed=1)
        seed = zlib.crc32(f"{path.stem}|{name}".encode())
        rng_h = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        rng_d = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        sc_hat, sz_hat = history_estimate(st.zc, st.sk, st.pb, st.Z, scn,
                                          H_HISTORY, rng_h, zct=st.zct,
                                          Zt=st.Zt)
        traces = {pol: {} for pol in TRACE_POLS}
        for _ in range(days):
            day = draw_day(fl, st, scn, rng_d)
            for pol in TRACE_POLS:
                simulate_policy(pol, fl, lib, st, day, scn.s_c, scn.s_z,
                                sc_hat, sz_hat, trace=traces[pol])
        for pol in TRACE_POLS:
            for k in sorted(traces[pol]):
                sse, n = traces[pol][k]
                rows.append({"instance": path.stem, "scenario": name,
                             "policy": pol, "k": k, "sse": sse, "n": n})
    return rows


def main(kw):
    fam = kw.get("fam", "Dethloff")
    days = int(kw.get("days", 100))
    workers = int(kw.get("workers", 4))
    tag = kw.get("tag", "")
    adhoc = adhoc_scenario(kw)
    if adhoc is not None:
        scen_list = [adhoc]
    else:
        names = kw.get("scen", "mixed").split(",")
        for s in names:
            if s not in SCENARIOS:
                raise SystemExit(f"unknown scenario {s}")
        scen_list = [(s, SCENARIOS[s]) for s in names]
    paths = [p for p in sorted((_W / "data" / fam).glob("*.vrpspd"))
             if (_W / "results" / "plans" / f"{p.stem}.json").exists()]
    if "n_inst" in kw and kw["n_inst"] != "all":
        paths = paths[:int(kw["n_inst"])]
    if not paths:
        raise SystemExit(f"no instances with cached plans for {fam}")
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"trace_{fam}{tag}.csv"
    print(f"start fam={fam} instances={len(paths)} "
          f"scenarios={[n for n, _ in scen_list]} days={days} "
          f"workers={workers} -> {csv_path.name}", flush=True)
    n_rows = 0
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_instance, str(p), scen_list, days): p
                    for p in paths}
            for f in as_completed(futs):
                rows = f.result()
                for row in rows:
                    wr.writerow(row)
                fh.flush()
                n_rows += len(rows)
                print(f"done {futs[f].stem}: {len(rows)} rows", flush=True)
    print(f"finished fam={fam}: {n_rows} rows -> {csv_path}", flush=True)


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    main(kw)
