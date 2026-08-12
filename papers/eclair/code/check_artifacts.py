#!/usr/bin/env python3
"""Artifact-consistency checker for the ECLAIR submission package.

    ../../../.venv/bin/python check_artifacts.py

Fails loudly (non-zero exit) when the shipped package is not
self-consistent -- the failure mode a reviewer meets first. Every check
below is *positive*: it must find the thing it looks for, and a check
that matches nothing FAILS rather than silently passing. (An earlier
version of this script matched quoted intervals with a regex that could
not span the line break the manuscript actually uses, so the interval
test executed zero times and still reported success. Vacuous checks are
worse than no checks; each one now asserts a minimum match count.)

Checks:
  1. main.pdf newer than main.tex, references.bib and the figures.
  2. results are a PUBLICATION run (n_reps >= 100) with a provenance
     block, a manifest, a clean-tree commit, and recorded solver params.
  3. Table 4 is parsed row by row: per-policy detection and certified
     counts must equal the JSON.
  4. audit totals quoted in the audit paragraph (certificates, below,
     above/inconclusive, abstentions) equal the JSON.
  5. the screening allowance quoted in the manuscript equals this run's,
     and no stale allowance figure survives elsewhere.
  6. exactly two certified-mutant intervals are quoted; each is matched
     to an audit-log entry, RECOMPUTED from err_hat * err_n at the
     recorded z, and its stated confidence label checked.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PAPER = HERE.parent
RES = HERE / "results" / "mutation_experiment.json"
PUBLICATION_REPS = 100
POLICY_ROW = {"kelly": "Kelly", "round_robin": "round-robin",
              "cost_blind": "cost-blind"}

fails: list[str] = []
notes: list[str] = []


def check(cond, msg):
    (notes if cond else fails).append(msg)


def wilson(k, n, z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    tex, pdf = PAPER / "main.tex", PAPER / "main.pdf"

    # 1. PDF freshness ----------------------------------------------------
    if not pdf.exists():
        fails.append("main.pdf missing")
        return report()
    newest_src = max(
        [tex.stat().st_mtime, (PAPER / "references.bib").stat().st_mtime]
        + [f.stat().st_mtime for f in (PAPER / "figures").glob("*.pdf")])
    check(pdf.stat().st_mtime >= newest_src,
          "main.pdf is at least as new as source + figures")

    # 2. provenance -------------------------------------------------------
    if not RES.exists():
        fails.append("results/mutation_experiment.json missing")
        return report()
    d = json.loads(RES.read_text())
    check(d.get("n_reps", 0) >= PUBLICATION_REPS,
          f"results are a publication run (n_reps={d.get('n_reps')})")
    p = d.get("provenance") or {}
    check(bool(p), "results JSON carries a provenance block")
    check(p.get("git_dirty") is False,
          f"run came from a CLEAN tree (git_dirty={p.get('git_dirty')})")
    check(len(str(p.get("git_commit", ""))) >= 40,
          "provenance records a full-length commit hash")
    check(bool(p.get("source_sha256_16")),
          "provenance records per-source content hashes")
    check(p.get("solver_params", {}).get("num_search_workers") == 1,
          "solver thread count recorded and pinned to 1")
    man = HERE / "results" / "MANIFEST.json"
    if man.exists():
        m = json.loads(man.read_text())["files"]
        ok = all(hashlib.sha256((HERE / "results" / f).read_bytes()).hexdigest()
                 == h for f, h in m.items()
                 if (HERE / "results" / f).exists())
        check(ok and "mutation_experiment.json" in m,
              "results MANIFEST matches the shipped files")
    else:
        fails.append("results/MANIFEST.json missing (run not promoted)")
    if p:
        notes.append(f"provenance: python {p.get('python')}, cpmpy "
                     f"{p.get('cpmpy')}, ortools {p.get('ortools')}, "
                     f"commit {str(p.get('git_commit'))[:12]}, "
                     f"audit z={p.get('audit_z')}")

    src = tex.read_text()
    R = d["results"]

    # 3. Table 4 rows -----------------------------------------------------
    for pol, label in POLICY_ROW.items():
        v = R[pol]
        row = re.search(re.escape(label) + r"\s*&(.+?)\\\\", src, re.S)
        if not row:
            fails.append(f"Table 4 row for {label} not found")
            continue
        cells = row.group(1)
        det = f"{v['detect']:.3f}".lstrip("0")
        check(det in cells or f"{v['detect']:.3f}" in cells,
              f"Table 4 [{label}] detection {v['detect']:.3f} matches JSON")
        check(f"${v['n_certified']}$" in cells,
              f"Table 4 [{label}] certificates {v['n_certified']} matches JSON")
        check(f"${v['n_cert_err_below_eps']}$" in cells,
              f"Table 4 [{label}] audited-below "
              f"{v['n_cert_err_below_eps']} matches JSON")

    # 4. audit totals -----------------------------------------------------
    runs = sum(v["n_runs"] for v in R.values())
    cert = sum(v["n_certified"] for v in R.values())
    below = sum(v["n_cert_err_below_eps"] for v in R.values())
    above = sum(v["n_cert_err_above_eps"] for v in R.values())
    incon = cert - below - above
    para = re.search(r"Auditing the certificates.*?(?=\\paragraph)", src, re.S)
    check(bool(para), "audit paragraph located")
    if para:
        t = para.group(0)
        for name, val in (("runs", runs), ("certificates", cert),
                          ("below", below), ("abstentions", runs - cert)):
            check(f"${val}$" in t,
                  f"audit paragraph quotes {name} = {val}")
        check(above == 0, f"no certificate audited above eps (got {above})")
    notes.append(f"audit: {cert} certificates / {runs} runs, {below} below, "
                 f"{above} above, {incon} inconclusive")

    # 5. screening allowance ----------------------------------------------
    budget_ms = round(d["budget_s"] * 1000)
    check(f"${budget_ms}$\\,ms" in src,
          f"manuscript quotes this run's screening allowance ({budget_ms} ms)")
    ranges = re.findall(r"\$(6\d\d)\$--\$(6\d\d)\$", src)
    in_range = {x for pair in ranges for x in pair}
    stale = ({b for b in re.findall(r"\$(6\d\d)\$\\,ms", src)}
             - {str(budget_ms)} - in_range)
    check(not stale, f"no stale allowance figures (found {stale or 'none'})")

    # 6. quoted certified-mutant intervals --------------------------------
    logged = [c for v in R.values() for c in v["certified_log"]
              if not c["label_faithful"]]
    quoted = re.findall(r"CI\s*\$\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\$",
                        src, re.S)
    check(len(quoted) == len(logged) and len(quoted) > 0,
          f"manuscript quotes all {len(logged)} certified-mutant intervals "
          f"(found {len(quoted)})")
    for lo, hi in quoted:
        lo, hi = float(lo), float(hi)
        hit = None
        for c in logged:
            if (abs(lo - c["err_ci"][0]) < 5e-4
                    and abs(hi - c["err_ci"][1]) < 5e-4):
                hit = c
                break
        if hit is None:
            fails.append(f"quoted interval [{lo}, {hi}] absent from audit log")
            continue
        k = round(hit["err_hat"] * hit["err_n"])
        rlo, rhi = wilson(k, hit["err_n"], hit["z"])
        check(abs(rlo - lo) < 5e-4 and abs(rhi - hi) < 5e-4,
              f"interval [{lo}, {hi}] recomputes from {k}/{hit['err_n']} "
              f"at z={hit['z']}")
        label = hit.get("conf", "")
        check("97.5" in label and "97.5" in src,
              f"confidence label '{label}' is stated in the manuscript")
    return report()


def report() -> int:
    for n in notes:
        print(f"  ok   {n}")
    for f in fails:
        print(f"  FAIL {f}")
    print(f"\n{len(notes)} checks passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
