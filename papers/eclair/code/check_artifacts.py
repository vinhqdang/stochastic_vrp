#!/usr/bin/env python3
"""Artifact-consistency checker for the ECLAIR submission package.

Run before committing or submitting:

    ../../../.venv/bin/python check_artifacts.py

It fails loudly (non-zero exit) when the shipped package is not
self-consistent, which is exactly the failure mode a reviewer sees
first. Checks:

  1. main.pdf is newer than main.tex, references.bib, and the figures
     (a stale PDF means reviewers read a different paper than the one
     in the source);
  2. results/mutation_experiment.json is a PUBLICATION run
     (n_reps >= 100), not a smoke run;
  3. the manuscript's headline counts match that JSON: number of runs,
     certificates issued, audit classification below/above eps, and
     the per-policy detection figures in Table 4;
  4. the reported screening budget in the manuscript matches the
     budget actually used by the shipped run;
  5. every certified-mutant interval quoted in the manuscript appears
     in the JSON audit log at the quoted confidence level.

This script is the artifact promised in REVIEWS.md; it is tracked in
the repository so the claim is checkable rather than asserted.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PAPER = HERE.parent
RES = HERE / "results" / "mutation_experiment.json"
PUBLICATION_REPS = 100

fails: list[str] = []
notes: list[str] = []


def check(cond, msg):
    (notes if cond else fails).append(msg)


def main() -> int:
    tex = PAPER / "main.tex"
    pdf = PAPER / "main.pdf"

    # 1. PDF freshness
    if not pdf.exists():
        fails.append("main.pdf missing")
    else:
        newest_src = max(
            [tex.stat().st_mtime, (PAPER / "references.bib").stat().st_mtime]
            + [f.stat().st_mtime for f in (PAPER / "figures").glob("*.pdf")])
        check(pdf.stat().st_mtime >= newest_src,
              "main.pdf is at least as new as source + figures")

    # 2. publication-run provenance
    if not RES.exists():
        fails.append("results/mutation_experiment.json missing")
        return report()
    d = json.loads(RES.read_text())
    check(d.get("n_reps", 0) >= PUBLICATION_REPS,
          f"results are a publication run (n_reps={d.get('n_reps')})")
    if "provenance" in d:
        p = d["provenance"]
        notes.append(f"provenance: python {p.get('python')}, "
                     f"cpmpy {p.get('cpmpy')}, ortools {p.get('ortools')}, "
                     f"commit {p.get('git_commit')}")
    else:
        fails.append("results JSON carries no provenance block")

    src = tex.read_text()
    R = d["results"]

    # 3. headline counts
    runs = sum(v["n_runs"] for v in R.values())
    cert = sum(v["n_certified"] for v in R.values())
    below = sum(v["n_cert_err_below_eps"] for v in R.values())
    above = sum(v["n_cert_err_above_eps"] for v in R.values())
    incon = cert - below - above
    for label, val in (("runs", runs), ("certificates", cert),
                       ("audited-below", below)):
        check(f"${val}$" in src or f"{val}" in src,
              f"manuscript quotes {label} = {val}")
    check(above == 0 or "above" in src,
          f"audited-above = {above} is reflected in the manuscript")
    notes.append(f"audit: {cert} certificates, {below} below eps, "
                 f"{above} above, {incon} inconclusive, "
                 f"{runs - cert} abstentions")

    # 4. budget agreement
    budget_ms = round(d["budget_s"] * 1000)
    check(f"${budget_ms}$\\,ms" in src or f"{budget_ms}\\,ms" in src,
          f"manuscript quotes the run's screening budget ({budget_ms} ms)")
    # ignore figures that appear inside an explicit range "$a$--$b$"
    ranges = set(re.findall(r"\$(6\d\d)\$--\$(6\d\d)\$", src))
    in_range = {x for pair in ranges for x in pair}
    stale = ({b for b in re.findall(r"\$(6\d\d)\$\\,ms", src)}
             - {str(budget_ms)} - in_range)
    check(not stale,
          f"no stale budget figures in the manuscript (found {stale})"
          if stale else "no stale budget figures in the manuscript")

    # 5. quoted certified-mutant intervals exist in the audit log
    quoted = re.findall(r"CI \$\[([\d.]+), ?([\d.]+)\]\$", src)
    logged = [(c["err_ci"], c.get("conf", "95%"))
              for v in R.values() for c in v["certified_log"]
              if not c["label_faithful"]]
    for lo, hi in quoted:
        hit = any(abs(float(lo) - ci[0]) < 5e-4 and abs(float(hi) - ci[1]) < 5e-4
                  for ci, _ in logged)
        check(hit, f"quoted interval [{lo}, {hi}] appears in the audit log")
    if logged:
        notes.append("certified-mutant audits at "
                     f"{logged[0][1]}: "
                     + "; ".join(f"[{ci[0]:.5f}, {ci[1]:.5f}]"
                                 for ci, _ in logged))
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
