"""Ambiguous-specification study (manuscript Sec. ambiguity): three
NL specs each admitting two defensible readings. The harness's
reference checker fixes ONE canonical reading; an LLM candidate that
implements the other reading is a SILENT unfaithful model — it builds,
solves, and returns plausible optima that are wrong under the
canonical reading. This is the regime where certification (and
abstention with an evidence ledger) earns its keep on real LLM
output, complementing the loud-failure regime of the main LLM study.

Ambiguities (canonical reading = the implemented checker):
  coloring   objective "minimize the number of colors used" WITHOUT
             the convention sentence: 1+max index (canonical) vs
             count of distinct colors used
  knapsack   "the selected items must be lighter than the capacity":
             <= cap (canonical checker) vs strict <
  ontime     "each selected job must be finished before its deadline":
             completion <= d (canonical) vs strict <

Usage: python3 run_ambiguous.py     (needs OpenRouter key or warm cache)
Writes results/ambiguous.{json,txt}.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import time

from eclair.certify import Bets, run_certification
from eclair.llm import MODELS, PROMPT_STYLES, make_llm_candidate, proxy_label
from eclair.problems import COLORING, KNAPSACK, SCHED
from run_experiment import CERT_EPS, CERT_PROBE_EQUIV

ALPHA = 0.05
SEED = 20260808
TEMPS = {"direct": 0.3, "stepwise": 0.7, "expert": 0.5}

AMBIG = {
    "coloring": (COLORING, """\
Color the n vertices of an undirected graph, using color indices from
0 to n-1, so that the two endpoints of every edge receive different
colors. Minimize the number of colors used.

params dict schema: {"n": int, "edges": list of (u, v) vertex pairs}."""),
    "knapsack_conflicts": (KNAPSACK, """\
You must select a subset of n items. Item i has weight w[i] and value
v[i]. The selected items together must be lighter than the capacity
cap. Some pairs of items conflict: for each pair (i, j) in the list
`conflicts`, items i and j must not both be selected. Maximize the
total value of the selected items.

params dict schema: {"n": int, "w": list[int] of length n,
"v": list[int] of length n, "cap": int,
"conflicts": list of (i, j) index pairs}."""),
    "ontime_selection": (SCHED, """\
A single machine processes jobs one at a time, with no idle time
between jobs. Job j has processing time p[j], deadline d[j] and weight
wt[j]. Select a subset of jobs, to be processed in non-decreasing
deadline order, such that every selected job is finished before its
deadline. Maximize the total weight of the selected jobs.

params dict schema: {"n": int, "p": list[int], "d": list[int],
"wt": list[int]}."""),
}


def main():
    out_dir = pathlib.Path(__file__).parent / "results"
    bets = Bets(json.loads((out_dir / "calibration.json").read_text()))
    budget = 80 * statistics.fmean(bets.cost.values())
    rng = random.Random(SEED)
    t0 = time.perf_counter()

    gen_log, pools = [], {}
    for name, (spec, text) in AMBIG.items():
        pool, labels = [], []
        for model in MODELS:
            for style in PROMPT_STYLES:
                cand, status = make_llm_candidate(
                    spec, model, style, TEMPS[style[0]], spec_text=text)
                entry = {"family": name, "model": model,
                         "style": style[0], "status": status}
                if cand is not None:
                    entry["faithful_canonical"] = proxy_label(cand, rng)
                    pool.append(cand)
                    labels.append(entry["faithful_canonical"])
                gen_log.append(entry)
                print(f"{name:<20} {model.split('/')[-1]:<38} "
                      f"{style[0]:<9} {status:<12} "
                      f"{'canonical' if entry.get('faithful_canonical') else ('OTHER READING' if cand else '')}",
                      flush=True)
        pools[name] = (pool, labels)

    n_valid = sum(1 for g in gen_log if g["status"] == "ok")
    n_canon = sum(1 for g in gen_log if g.get("faithful_canonical"))
    print(f"\ngeneration: {n_valid}/{len(gen_log)} valid, "
          f"{n_canon}/{n_valid} canonical reading, "
          f"{n_valid - n_canon} silent other-reading models", flush=True)

    results = {}
    for name, (pool, labels) in pools.items():
        if len(pool) < 2:
            continue
        res = run_certification(pool, bets, "kelly", budget, ALPHA,
                                random.Random(SEED + 1),
                                eps=CERT_EPS,
                                cert_budget=CERT_PROBE_EQUIV
                                * bets.cost["A"])
        fam = {"n": len(pool),
               "canonical": sum(labels),
               "rejected_canonical": 0, "rejected_other": 0,
               "survived_other": 0, "abstained": res["abstained"]}
        for st, canonical in zip(res["states"], labels):
            if st.rejected and canonical:
                fam["rejected_canonical"] += 1
            elif st.rejected:
                fam["rejected_other"] += 1
            elif not canonical:
                fam["survived_other"] += 1
        if res["screening_survivor"] is not None:
            fam["pick_canonical"] = labels[
                res["states"].index(res["screening_survivor"])]
        cp = res.get("certified_pick")
        fam["certified"] = cp is not None
        fam["certified_canonical"] = (
            labels[res["states"].index(cp)] if cp is not None else None)
        fam["certification_abstained"] = res["certification_abstained"]
        results[name] = fam
        print(f"{name:<20} pool={fam['n']} canonical={fam['canonical']} "
              f"rej-other={fam['rejected_other']} "
              f"rej-canonical={fam['rejected_canonical']} "
              f"surv-other={fam['survived_other']} "
              f"abstain={fam['abstained']} "
              f"certified={fam['certified']}"
              f"{'(canonical)' if fam['certified_canonical'] else ''}",
              flush=True)

    payload = {"alpha": ALPHA, "seed": SEED, "budget_s": budget,
               "generation": gen_log, "results": results}
    (out_dir / "ambiguous.json").write_text(json.dumps(payload, indent=2))
    lines = ["ECLAIR ambiguous-specification study",
             f"{n_valid}/{len(gen_log)} valid, {n_canon} canonical, "
             f"{n_valid - n_canon} silent other-reading models", ""]
    for name, fam in results.items():
        lines.append(f"{name}: {fam}")
    lines.append(f"\nwall time {time.perf_counter()-t0:.0f}s")
    (out_dir / "ambiguous.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
