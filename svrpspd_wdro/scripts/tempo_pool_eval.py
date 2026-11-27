"""TEMPO pooling experiment: geography zones, factor estimators, fleet days.

Usage (from svrpspd_wdro/):
    python scripts/tempo_pool_eval.py fam=Dethloff [n_inst=all] [days=150]
        [workers=4] [scen=city,mixed,...] [tag=]

Parallel over (instance, scenario) jobs. Each job builds its instance's
PolicyLibrary once (seed 1), draws H = 60 history days and `days` fleet days
from its own seed zlib.crc32("<instance>|<scenario>"), and simulates the
policies in ev.pooling.POL on common random numbers within each day.

Outputs: results/tempo_pool/<fam><tag>.csv (one row per instance x scenario
x day) and results/tempo_pool/<fam><tag>.log.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import csv
import time
import zlib
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_W))
from ev.fleet import FleetPlan, PolicyLibrary, parse_instance  # noqa: E402
from ev.pooling import (POL, SCENARIOS, H_HISTORY, build_structure, draw_day,  # noqa: E402
                        history_estimate, simulate_day)

OUT = _W / "results" / "tempo_pool"
TZ7 = dt.timezone(dt.timedelta(hours=7))


def _now():
    return dt.datetime.now(TZ7).strftime("%Y-%m-%d %H:%M:%S GMT+7")


def run_job(path, scen_name, days):
    """All fleet days of one (instance, scenario). Returns (rows, seconds)."""
    t0 = time.time()
    path = Path(path)
    fl = FleetPlan.from_instance(path)
    D = parse_instance(path)[0]
    st = build_structure(fl, D)
    lib = PolicyLibrary(fl, seed=1)
    scn = SCENARIOS[scen_name]
    seed = zlib.crc32(f"{path.stem}|{scen_name}".encode())
    rng_h = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    rng_d = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    sc_hat, sz_hat = history_estimate(st.zc, st.sk, st.pb, st.Z, scn,
                                      H_HISTORY, rng_h)
    rows = []
    for d in range(days):
        day = draw_day(fl, st, scn, rng_d)
        res = simulate_day(fl, lib, st, day, scn, sc_hat, sz_hat)
        row = {"instance": path.stem, "K": st.K, "Z": st.Z,
               "scenario": scen_name, "day": d,
               "sc_hat": sc_hat, "sz_hat": sz_hat}
        for pol in POL:
            row[f"cost_{pol}"] = res[pol]["cost"]
            row[f"se_{pol}"] = res[pol]["se"]
            row[f"nd_{pol}"] = res[pol]["nd"]
        rows.append(row)
    return rows, time.time() - t0


def columns():
    cols = ["instance", "K", "Z", "scenario", "day", "sc_hat", "sz_hat"]
    for pol in POL:
        cols += [f"cost_{pol}", f"se_{pol}", f"nd_{pol}"]
    return cols


def main(kw):
    fam = kw.get("fam", "Dethloff")
    days = int(kw.get("days", 150))
    workers = int(kw.get("workers", 4))
    tag = kw.get("tag", "")
    scen_names = (kw["scen"].split(",") if "scen" in kw else list(SCENARIOS))
    for s in scen_names:
        if s not in SCENARIOS:
            raise SystemExit(f"unknown scenario {s}")
    paths = [p for p in sorted((_W / "data" / fam).glob("*.vrpspd"))
             if (_W / "results" / "plans" / f"{p.stem}.json").exists()]
    if "n_inst" in kw and kw["n_inst"] != "all":
        paths = paths[:int(kw["n_inst"])]
    if not paths:
        raise SystemExit(f"no instances with cached plans for {fam}")
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"{fam}{tag}.csv"
    log_path = OUT / f"{fam}{tag}.log"
    jobs = [(str(p), s) for p in paths for s in scen_names]
    logf = open(log_path, "a")

    def log(msg):
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"start fam={fam} instances={len(paths)} scenarios={scen_names} "
        f"days={days} workers={workers} jobs={len(jobs)} -> {csv_path.name}")
    t_all = time.time()
    n_rows = 0
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=columns())
        wr.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_job, p, s, days): (p, s) for p, s in jobs}
            for f in as_completed(futs):
                p, s = futs[f]
                rows, secs = f.result()
                for row in rows:
                    wr.writerow(row)
                fh.flush()
                n_rows += len(rows)
                log(f"done {Path(p).stem} {s}: {secs:.1f}s, {len(rows)} days")
    log(f"finished fam={fam}: {n_rows} rows in {time.time() - t_all:.1f}s")
    logf.close()


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    main(kw)
