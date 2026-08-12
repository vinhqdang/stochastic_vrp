"""End-to-end LLM experiment: generate candidate models from natural-
language specs via OpenRouter (diversified over model family x prompt
style x temperature), then certify each family's pool with the
calibrated probes under all three routing policies.

Usage:  python3 run_llm_experiment.py

Needs results/calibration.json (run run_experiment.py first) and an
OpenRouter key (see eclair/llm.py). LLM responses are disk-cached, so
reruns are free and deterministic.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import time

from eclair.certify import Bets, run_certification
from eclair.llm import MODELS, PROMPT_STYLES, make_llm_candidate, proxy_label
from eclair.problems import ALL_SPECS
from run_experiment import CERT_EPS, CERT_PROBE_EQUIV

ALPHA = 0.05
SEED = 20260807
BUDGET_PROBE_EQUIV = 80
TEMPS = {"direct": 0.3, "stepwise": 0.7, "expert": 0.5}


def main():
    out_dir = pathlib.Path(__file__).parent / "results"
    calib = json.loads((out_dir / "calibration.json").read_text())
    bets = Bets(calib)
    budget = BUDGET_PROBE_EQUIV * statistics.fmean(bets.cost.values())

    t0 = time.perf_counter()
    gen_log = []
    pools = {}
    rng = random.Random(SEED)
    for spec in ALL_SPECS:
        pool, labels = [], []
        for model in MODELS:
            for style in PROMPT_STYLES:
                cand, status = make_llm_candidate(
                    spec, model, style, TEMPS[style[0]])
                entry = {"family": spec.name, "model": model,
                         "style": style[0], "status": status}
                if cand is not None:
                    faithful = proxy_label(cand, rng)
                    entry["faithful"] = faithful
                    pool.append(cand)
                    labels.append(faithful)
                gen_log.append(entry)
                print(f"{spec.name:<20} {model.split('/')[-1]:<38} "
                      f"{style[0]:<9} {status:<12} "
                      f"{'faithful' if entry.get('faithful') else ''}",
                      flush=True)
        pools[spec.name] = (spec, pool, labels)

    n_req = len(gen_log)
    n_valid = sum(1 for g in gen_log if g["status"] == "ok")
    n_faithful = sum(1 for g in gen_log if g.get("faithful"))
    print(f"\ngeneration: {n_valid}/{n_req} valid, "
          f"{n_faithful}/{n_valid} proxy-faithful "
          f"({time.perf_counter()-t0:.0f}s)\n", flush=True)

    results = {}
    for policy in ("kelly", "round_robin", "cost_blind"):
        prng = random.Random(SEED + 2)
        agg = {"n_faith": 0, "faith_rej": 0, "n_unf": 0, "unf_rej": 0,
               "picks": 0, "pick_ok": 0, "abstain": 0,
               "ebh_false": 0, "ebh_total": 0, "families": {},
               "cert_runs": 0, "cert_issued": 0, "cert_ok": 0,
               "cert_abstain": 0}
        for name, (spec, pool, labels) in pools.items():
            if len(pool) < 2:
                continue
            res = run_certification(pool, bets, policy, budget, ALPHA,
                                    prng, eps=CERT_EPS,
                                    cert_budget=CERT_PROBE_EQUIV
                                    * bets.cost['A'])
            fam = {"rejected": [], "survived": []}
            cp = res.get("certified_pick")
            agg["cert_runs"] += 1
            if cp is None:
                agg["cert_abstain"] += 1
            else:
                agg["cert_issued"] += 1
                agg["cert_ok"] += labels[res["states"].index(cp)]
            for st, faithful in zip(res["states"], labels):
                tag = f"{st.cand.meta['model'].split('/')[-1]}/" \
                      f"{st.cand.meta['style']}" + \
                      ("*" if faithful else "")
                (fam["rejected"] if st.rejected else fam["survived"]).append(tag)
                if faithful:
                    agg["n_faith"] += 1
                    agg["faith_rej"] += st.rejected
                else:
                    agg["n_unf"] += 1
                    agg["unf_rej"] += st.rejected
            for i in res["ebh_rejected"]:
                agg["ebh_total"] += 1
                agg["ebh_false"] += labels[i]
            if res["screening_abstained"]:
                agg["abstain"] += 1
            elif any(labels):
                agg["picks"] += 1
                agg["pick_ok"] += labels[res["states"].index(res["pick"])]
            agg["families"][name] = fam
        results[policy] = agg
        print(f"{policy:<12} false-rej={agg['faith_rej']}/{agg['n_faith']}  "
              f"detect={agg['unf_rej']}/{agg['n_unf']}  "
              f"pick-ok={agg['pick_ok']}/{agg['picks']}  "
              f"abstain={agg['abstain']}  "
              f"eBH-FDR={agg['ebh_false']}/{agg['ebh_total']}  "
              f"certified={agg['cert_ok']}/{agg['cert_issued']} ok "
              f"({agg['cert_abstain']} abstain of {agg['cert_runs']})",
              flush=True)

    payload = {"alpha": ALPHA, "budget_s": budget, "seed": SEED,
               "models": MODELS, "generation": gen_log, "results": results}
    (out_dir / "llm_experiment.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "ECLAIR end-to-end LLM experiment (OpenRouter generation + CP-SAT probes)",
        f"models: {', '.join(MODELS)}",
        f"alpha={ALPHA}  budget={budget*1000:.0f}ms/pool  "
        f"pools: one per family, candidates = 2 models x 3 prompt styles",
        f"generation: {n_valid}/{n_req} valid code, "
        f"{n_faithful}/{n_valid} proxy-faithful",
        "",
    ]
    for pol, agg in results.items():
        lines.append(
            f"{pol:<12} false-rej={agg['faith_rej']}/{agg['n_faith']}  "
            f"detect={agg['unf_rej']}/{agg['n_unf']}  "
            f"pick-ok={agg['pick_ok']}/{agg['picks']}  "
            f"abstain={agg['abstain']}  "
            f"eBH-FDR={agg['ebh_false']}/{max(agg['ebh_total'],1)}")
    lines.append("")
    lines.append("per-family outcomes under kelly "
                 "(* = proxy-faithful candidate):")
    for name, fam in results["kelly"]["families"].items():
        lines.append(f"  {name}: rejected={fam['rejected']} "
                     f"survived={fam['survived']}")
    lines.append(f"\ntotal wall time: {time.perf_counter()-t0:.0f}s")
    (out_dir / "llm_experiment.txt").write_text("\n".join(lines) + "\n")
    print("\nwritten: results/llm_experiment.{json,txt}")


if __name__ == "__main__":
    main()
