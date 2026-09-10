"""Recompute REAL Gemini token counts for the matched-budget grid (n=120).

`tokens_of()` in musique_benchmark.py is `round(len(text.split()) * 1.33)`
-- a fixed word-count heuristic, not the model's own tokenizer. A review
correctly flagged this as a construct-validity problem for a paper about
token cost. Item SELECTION is deterministic (verified: `load_instances(60,
"branching", seed=0)` reproduces the exact 60-instance set logged in
results/musique_gemini-3.5-flash-lite_lexical_min.jsonl bit-for-bit), so
the exact prompt sent for every (instance, policy, agent) can be
rebuilt and passed to Gemini's countTokens endpoint -- no generation
call, no billing, no re-running the model.

Run: GEMINI_API_KEY=... python3 real_tokens.py
"""

import json
import os
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from musique_benchmark import POLICIES, build_prompt, load_instances, tokens_of


def build_context(inst, idx_list):
    """Just the concatenated paragraph text -- same scope as tokens_of(),
    excluding the fixed question/instruction scaffold that is identical
    across every policy and every arm."""
    by_idx = {p["idx"]: p for p in inst["paragraphs"]}
    paras = [by_idx[j] for j in idx_list if j in by_idx]
    if not paras:
        return ""
    return "\n\n".join(f"[{p['title']}] {p['text']}" for p in paras)

RESULTS = pathlib.Path(__file__).parent / "results"
MODEL = "gemini-3.5-flash-lite"

# The configs that appear, by name, in Table 2, Table 3, or Figure 2.
CONFIGS = (
    [("broadcast", {}), ("none", {}), ("oracle", {})]
    + [("random", {"k": 3})]
    + [("centrality", {"k": 3})]
    + [("topk", {"k": k}) for k in (1, 3, 5, 10)]
    + [("parcel", {"q": q}) for q in (0.95, 0.7, 0.3)]
)


def count_tokens(text, key, retries=5):
    if not text:
        return 0
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL}:countTokens?key={key}")
    payload = json.dumps({"contents": [{"parts": [{"text": text}]}]}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=30).read()
            return json.loads(resp)["totalTokens"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")
    rows = load_instances(60, "branching", seed=0)
    print(f"{len(rows)} instances loaded (must match the logged 60)")

    jobs = []   # (policy, cfg_json, text, est)
    scaffold_job = None
    for inst in rows:
        by_idx = {p["idx"]: p for p in inst["paragraphs"]}
        for name, kw in CONFIGS:
            rng = (random.Random(hash((inst["id"], "real_tokens")) % 997)
                   if name == "random" else None)
            alloc = POLICIES[name](inst, rng=rng, **kw)
            for i, a in enumerate(inst["agents"]):
                ids = alloc[i]
                text = build_context(inst, ids)
                est = sum(tokens_of(by_idx[j]) for j in ids if j in by_idx)
                jobs.append((name, json.dumps(kw), text, est))
                if scaffold_job is None:
                    # fixed per-call overhead (question + instructions),
                    # identical across every policy/arm -- measure it once
                    scaffold_job = build_prompt(inst, [], a, random.Random(i))

    print(f"{len(jobs)} prompts -> countTokens (est. word-heuristic total "
          f"{sum(j[3] for j in jobs)})")

    results = [None] * len(jobs)
    with ThreadPoolExecutor(12) as pool:
        futs = {pool.submit(count_tokens, j[2], key): idx
                for idx, j in enumerate(jobs)}
        done = 0
        for fut in futs:
            pass
        for fut, idx in futs.items():
            results[idx] = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}", file=sys.stderr)

    out = RESULTS / "real_token_counts.jsonl"
    with open(out, "w") as fh:
        for (name, cfg, text, est), real in zip(jobs, results):
            fh.write(json.dumps({"policy": name, "cfg": cfg,
                                  "est_tokens": est, "real_tokens": real}) + "\n")
    print(f"wrote {out}")

    scaffold_tokens = count_tokens(scaffold_job, key)
    print(f"fixed per-call scaffold (question+instructions, no context): "
          f"{scaffold_tokens} real tokens")

    import collections
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for (name, cfg, text, est), real in zip(jobs, results):
        k = (name, cfg)
        agg[k][0] += est
        agg[k][1] += real
        agg[k][2] += 1
    print(f"{'policy':11s} {'cfg':>14s} {'est/recv':>10s} {'real/recv':>10s} {'ratio':>6s}")
    for (name, cfg), (est, real, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        ratio = f"{real / est:.3f}" if est else "n/a"
        print(f"{name:11s} {cfg:>14s} {est / n:10.1f} {real / n:10.1f} {ratio:>6s}")


if __name__ == "__main__":
    main()
