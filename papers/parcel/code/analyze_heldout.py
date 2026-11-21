"""Analyse a held-out rebuttal grid (musique_benchmark.py --grid rebuttal*).

For every logged call the exact transmitted context is rebuilt (selection
is deterministic) and its REAL token count is taken from Gemini's
countTokens endpoint (cached by content hash in results/count_cache.json),
so token figures are on the same footing as the paper's Tables 2-3.
Accuracy comparisons use cluster_test.py's instance-level block
bootstrap and permutation test.

Run: python3 analyze_heldout.py <jsonl> --offset 130 --instances 120 [--grid rebuttal]
Writes results/summary_<jsonl stem>.json
"""

import argparse
import collections
import hashlib
import json
import os
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import musique_benchmark as mb  # noqa: E402
import cluster_test as ct  # noqa: E402
from real_tokens import build_context, count_tokens  # noqa: E402

RESULTS = mb.RESULTS
CACHE = RESULTS / "count_cache.json"

COMPARISONS = [
    ("topk{\"k\": 3}", "centrality{\"k\": 3}", "per-recv lex k3 vs seed lex k3"),
    ("oracle", "broadcast", "oracle vs broadcast"),
    ("parcel{\"q\": 0.3}", "broadcast", "density loose (frozen q=0.3) vs broadcast"),
    ("topk{\"k\": 5}", "broadcast", "uniform k5 (frozen) vs broadcast"),
    ("topk{\"k\": 1, \"scorer\": \"ce\", \"density\": false}", "broadcast", "per-recv CE k1 vs broadcast"),
    ("topk{\"k\": 2, \"scorer\": \"ce\", \"density\": false}", "broadcast", "per-recv CE k2 vs broadcast"),
    ("topk{\"k\": 3, \"scorer\": \"ce\", \"density\": false}", "broadcast", "per-recv CE k3 vs broadcast"),
    ("topk{\"k\": 2, \"scorer\": \"ce\", \"density\": false}", "centrality{\"k\": 3, \"scorer\": \"ce\"}", "per-recv CE k2 vs seed CE k3"),
    ("topk{\"k\": 2, \"scorer\": \"ce\", \"density\": false}", "topk{\"k\": 3}", "per-recv CE k2 vs per-recv lex k3"),
    ("topk{\"k\": 2, \"scorer\": \"ce\", \"density\": false}", "oracle", "per-recv CE k2 vs oracle"),
    ("topk{\"k\": 1, \"scorer\": \"ce\", \"density\": false}", "oracle", "per-recv CE k1 vs oracle"),
]


def label(policy, cfg):
    return policy + (cfg if cfg != "{}" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--offset", type=int, default=130)
    ap.add_argument("--instances", type=int, default=120)
    args = ap.parse_args()
    path = pathlib.Path(args.jsonl)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("correct") is not None]
    insts = {i["id"]: i for i in mb.load_instances(args.instances, "branching",
                                                   offset=args.offset)}

    # rebuild contexts, count real tokens
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    allocs = {}
    need = {}
    for r in rows:
        k = (r["id"], r["policy"], r["cfg"])
        if k not in allocs:
            inst = insts[r["id"]]
            allocs[k] = mb.allocate(r["policy"], inst, json.loads(r["cfg"]),
                                    random.Random(0))
        ids = allocs[k][r["agent"]]
        ctx = build_context(insts[r["id"]], ids)
        h = hashlib.sha1(ctx.encode()).hexdigest()
        r["_h"] = h
        if h not in cache:
            need[h] = ctx
    key = os.environ.get("GEMINI_API_KEY")
    if need:
        print(f"counting {len(need)} contexts")
        with ThreadPoolExecutor(8) as pool:
            for h, n in zip(need, pool.map(lambda c: count_tokens(c, key),
                                           need.values())):
                cache[h] = n
        CACHE.write_text(json.dumps(cache))

    per = collections.defaultdict(lambda: {"acc": [], "tok": []})
    for r in rows:
        lab = label(r["policy"], r["cfg"])
        per[lab]["acc"].append(bool(r["correct"]))
        per[lab]["tok"].append(cache[r["_h"]])
    b_tok = sum(per["broadcast"]["tok"]) / len(per["broadcast"]["tok"])
    b_acc = sum(per["broadcast"]["acc"]) / len(per["broadcast"]["acc"])
    out = {"source": path.name, "instances": len(insts), "policies": [],
           "comparisons": []}
    print(f"{'policy':58s} {'n':>4s} {'acc':>6s} {'dacc':>7s} {'tok':>6s} {'ratio':>6s}")
    for lab, d in sorted(per.items(), key=lambda x: -sum(x[1]["tok"]) / len(x[1]["tok"])):
        acc = sum(d["acc"]) / len(d["acc"])
        tok = sum(d["tok"]) / len(d["tok"])
        ratio = b_tok / tok if tok else None
        print(f"{lab:58s} {len(d['acc']):4d} {acc:6.3f} {acc - b_acc:+7.3f} "
              f"{tok:6.0f} {ratio if ratio is None else round(ratio, 1)!s:>6s}")
        out["policies"].append({"policy": lab, "n": len(d["acc"]),
                                "acc": round(acc, 4),
                                "d_acc_vs_broadcast": round(acc - b_acc, 4),
                                "real_tokens_per_receiver": round(tok, 1),
                                "ratio_vs_broadcast": None if ratio is None else round(ratio, 2)})
    print()
    import io
    import contextlib
    for a, b, desc in COMPARISONS:
        if a not in per or b not in per:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            res = ct.run(str(path), a, b)
        res["comparison"] = desc
        print(f"{desc:48s} d={res['delta']:+.3f} CI=[{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}] "
              f"p={res['p_cluster_perm']:.4f}  (n={res['n_receivers']}/{res['n_instances']} inst)")
        out["comparisons"].append(res)
    dest = RESULTS / f"summary_{path.stem}.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
