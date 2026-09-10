"""Numeric replacement for Table 5: three value-curve designs for
Algorithm 1's v_i(b), each compared against a matched-spend uniform
per-receiver rule (top-k), by DELIVERY RECALL (deterministic -- no
model call needed) at REAL token spend (via Gemini's countTokens, also
no generation call). Run on both the plain branching set (n instances,
2 receivers each) and the composite-receiver set (n instances, 3
receivers each, one needing ALL its supporting paragraphs).

A review found the paper's previous Table 5 unreproducible from any
code in this repo: no trace of three distinct curve implementations
existed anywhere. This script IS that implementation, committed so the
table is regenerable end to end. Because the exact code behind the
previously published numbers no longer exists, this is a fresh,
documented run, not a bit-for-bit reproduction of the old one -- the
STATUS.md entry for this fix says so explicitly.

Three curves, all consuming the SAME per-agent relevance scores:
  sum      -- v_i(S) = sum of raw relevance scores in S (linear, no
              saturation: what the code comment in musique_benchmark.py
              says was tried first and "poured tokens into receivers
              with many mediocre-but-scoring candidates").
  noisyor  -- v_i(S) = 1 - prod_{f in S}(1 - raw(f)), a proper noisy-OR
              over per-item scores (each already in [0,1] for the
              lexical scorer's cosine-like normalization).
  captured -- v_i(S) = sum_{f in S} raw(f) / sum_f raw(f), i.e. the
              fraction of the receiver's own relevance mass captured.
              This is musique_benchmark.py's existing p_dp curve.

Run: python3 value_curves_check.py [--real-tokens]
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from musique_benchmark import (POLICIES, add_composite_receivers,
                                load_instances, relevance, tokens_of)

N_INSTANCES = 130
BUDGETS = (120, 200, 350)   # per-receiver average token budget, matched to Table 2/3's range
BINS = 24
MODEL = "gemini-3.5-flash-lite"


def curve_rows(inst, agent, kind, levels):
    raw = {p["idx"]: max(0.0, relevance(p, agent))
           for p in inst["paragraphs"]}
    tot = sum(raw.values()) or 1.0
    if kind == "captured":
        val = {k: v / tot for k, v in raw.items()}
    else:
        val = raw
    ranked = sorted(inst["paragraphs"],
                     key=lambda p: val[p["idx"]] / tokens_of(p), reverse=True)
    row = []
    for lvl in levels:
        got, spend = [], 0
        for p in ranked:
            t = tokens_of(p)
            if spend + t > lvl:
                continue
            got.append(p["idx"])
            spend += t
        if kind == "noisyor":
            prod = 1.0
            for j in got:
                prod *= max(0.0, 1.0 - min(1.0, raw[j]))
            mass = 1.0 - prod
        elif kind == "sum":
            mass = sum(raw[j] for j in got)
        else:
            mass = sum(val[j] for j in got)
        row.append((mass, got))
    return row


def dp_alloc(inst, kind, per_recv):
    budget = per_recv * len(inst["agents"])
    step = max(1, budget // BINS)
    levels = [b * step for b in range(BINS + 1)]
    curves = [curve_rows(inst, a, kind, levels) for a in inst["agents"]]
    n = len(inst["agents"])
    NEG = float("-inf")
    dp = [[NEG] * (BINS + 1) for _ in range(n + 1)]
    pick = [[None] * (BINS + 1) for _ in range(n + 1)]
    for b in range(BINS + 1):
        dp[n][b] = 0.0
    for i in range(n - 1, -1, -1):
        for b in range(BINS + 1):
            for spend in range(b + 1):
                v, ids = curves[i][spend]
                nxt = dp[i + 1][b - spend]
                if nxt == NEG:
                    continue
                if v + nxt > dp[i][b]:
                    dp[i][b] = v + nxt
                    pick[i][b] = (spend, ids)
    out, b = {}, BINS
    for i in range(n):
        spend, ids = pick[i][b]
        out[i] = ids
        b -= spend
    return out


def recall(inst, alloc):
    tot = hit = 0
    for i, a in enumerate(inst["agents"]):
        need = a.get("needs_all", [a["needs"]])
        tot += 1
        hit += 1.0 if all(j in alloc[i] for j in need) else 0.0
    return hit, tot


def tokens_spent(inst, alloc):
    by_idx = {p["idx"]: p for p in inst["paragraphs"]}
    return sum(tokens_of(by_idx[j]) for ids in alloc.values()
               for j in ids if j in by_idx)


def eval_policy(rows, alloc_fn):
    hit = tot = tok = 0
    per_instance = []
    for inst in rows:
        alloc = alloc_fn(inst)
        h, t = recall(inst, alloc)
        hit += h
        tot += t
        s = tokens_spent(inst, alloc)
        tok += s
        per_instance.append((inst["id"], h / t if t else 0.0, s / t if t else 0.0))
    return hit / tot, tok / tot, per_instance


def cluster_bootstrap_delta(per_a, per_b, iters=5000, seed=0):
    rng = random.Random(seed)
    ids = [r[0] for r in per_a]
    a_by_id = {r[0]: r[1] for r in per_a}
    b_by_id = {r[0]: r[1] for r in per_b}
    diffs = []
    for _ in range(iters):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        da = sum(a_by_id[i] for i in s) / len(s)
        db = sum(b_by_id[i] for i in s) / len(s)
        diffs.append(da - db)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]


def best_matching_k(rows, target_budget):
    """Pick the top-k whose average per-receiver spend is closest to target."""
    best_k, best_gap = None, float("inf")
    for k in (1, 2, 3, 5, 10):
        _, tok, _ = eval_policy(
            rows, lambda inst, k=k: POLICIES["topk"](inst, k=k))
        gap = abs(tok - target_budget)
        if gap < best_gap:
            best_gap, best_k = gap, k
    return best_k


def count_tokens_real(text, key):
    if not text:
        return 0
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL}:countTokens?key={key}")
    payload = json.dumps({"contents": [{"parts": [{"text": text}]}]}).encode()
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["totalTokens"]


def real_tokens_for(rows, alloc_fn, key, workers=12):
    texts = []
    for inst in rows:
        by_idx = {p["idx"]: p for p in inst["paragraphs"]}
        alloc = alloc_fn(inst)
        for ids in alloc.values():
            paras = [by_idx[j] for j in ids if j in by_idx]
            texts.append("\n\n".join(f"[{p['title']}] {p['text']}" for p in paras)
                         if paras else "")
    with ThreadPoolExecutor(workers) as pool:
        counts = list(pool.map(lambda t: count_tokens_real(t, key), texts))
    return sum(counts) / len(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-tokens", action="store_true")
    args = ap.parse_args()
    key = os.environ.get("GEMINI_API_KEY") if args.real_tokens else None

    rows = load_instances(N_INSTANCES, "branching", seed=0)
    composite = add_composite_receivers(rows)
    print(f"plain: {len(rows)} instances, "
          f"{sum(len(r['agents']) for r in rows)} receivers")
    print(f"composite: {len(composite)} instances, "
          f"{sum(len(r['agents']) for r in composite)} receivers\n")

    for label, data in (("plain", rows), ("composite", composite)):
        target = 220 if label == "plain" else 350
        k = best_matching_k(data, target)
        uni_acc, uni_tok, uni_per = eval_policy(
            data, lambda inst, k=k: POLICIES["topk"](inst, k=k))
        print(f"=== {label} ({len(data)} instances) --- "
              f"uniform top-{k} baseline: recall={uni_acc:.3f} "
              f"tokens/recv={uni_tok:.0f} ===")
        if key:
            real_uni = real_tokens_for(
                data, lambda inst, k=k: POLICIES["topk"](inst, k=k), key)
            print(f"  (real tokens/recv: {real_uni:.0f})")
        for kind in ("sum", "noisyor", "captured"):
            acc, tok, per = eval_policy(
                data, lambda inst, kind=kind: dp_alloc(inst, kind, uni_tok))
            lo, hi = cluster_bootstrap_delta(per, uni_per)
            print(f"  {kind:10s} recall={acc:.3f} (delta {acc - uni_acc:+.3f}, "
                  f"95% CI [{lo:+.3f},{hi:+.3f}]) tokens/recv={tok:.0f}")
            if key:
                real = real_tokens_for(
                    data, lambda inst, kind=kind: dp_alloc(inst, kind, uni_tok), key)
                print(f"    (real tokens/recv: {real:.0f})")
        print()


if __name__ == "__main__":
    main()
