"""Separate relevance-ESTIMATION error from ALLOCATION-rule limitations.

Question (reviewer jkMC, Q2): when marginal allocation (Algorithm 1)
ties tuned uniform top-k, is that because the allocation rule is weak or
because its input scores are?

Design: hold the allocators fixed and vary only the quality of the
scores they receive.  For each receiver,

    s_alpha(p) = alpha * gold(p) + (1 - alpha) * lex(p) / max_p lex(p)

where gold(p) = 1 iff p is one of the receiver's supporting paragraphs
(the composite receiver needs all of them) and lex is the lexical score
used throughout the paper.  alpha = 0 is the paper's setting; alpha = 1
is perfect relevance estimation.  Optionally a cached scorer from
rebuttal_scorers.py (ce / bge / llm) replaces lex as the alpha = 0 end.

For every alpha we trace delivery recall vs tokens for
  uniform  : per-receiver top-k, k = 1..10      (one global k)
  dp       : Algorithm 1 (captured-mass curve), per-receiver budget swept
and report the DP-minus-uniform recall difference at matched tokens
(uniform's curve linearly interpolated at the DP's spend), averaged over
a grid of budgets, with a 95% cluster bootstrap over instances.

No API calls.  Tokens use the word-count proxy (tokens_of): this script
compares policies with each other at matched proxy spend, so a common
proxy is adequate; headline token counts in the paper use countTokens.

Run: python3 estimation_vs_allocation.py [--instances 250] [--base lexical|ce|bge|llm]
"""

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import musique_benchmark as mb  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"


def make_scorer(rows, alpha, base):
    gold = {}
    basef = mb._lexical if base == "lexical" else None
    if basef is None:
        from rebuttal_scorers import load_scorer
        basef = load_scorer(base)
    norm = {}
    for inst in rows:
        for a in inst["agents"]:
            need = set(a.get("needs_all", [a["needs"]]))
            vals = {p["idx"]: basef(p, a) for p in inst["paragraphs"]}
            lo, hi = min(vals.values()), max(vals.values())
            for p in inst["paragraphs"]:
                k = (inst["id"], a["q"], p["idx"])
                norm[k] = (vals[p["idx"]] - lo) / (hi - lo) if hi > lo else 0.0
                gold[k] = 1.0 if p["idx"] in need else 0.0
    return gold, norm


def run(rows, alpha, gold, norm):
    cur = {"id": None}

    def scorer(par, agent):
        k = (cur["id"], agent["q"], par["idx"])
        return alpha * gold[k] + (1 - alpha) * norm[k]

    mb.SCORER = scorer
    # per-instance (tokens, hits, receivers) for each config
    uni = {k: [] for k in range(1, 11)}
    dpb = {b: [] for b in (60, 90, 120, 160, 200, 260, 350, 450, 550)}
    for inst in rows:
        cur["id"] = inst["id"]
        by = {p["idx"]: p for p in inst["paragraphs"]}

        def score(alloc):
            t = h = 0
            for i, a in enumerate(inst["agents"]):
                ids = alloc[i]
                t += sum(mb.tokens_of(by[j]) for j in ids)
                h += all(j in ids for j in a.get("needs_all", [a["needs"]]))
            return t, h, len(inst["agents"])

        for k in uni:
            uni[k].append(score(mb.p_topk(inst, k=k)))
        for b in dpb:
            dpb[b].append(score(mb.p_dp(inst, per_recv=b)))
    mb.SCORER = None
    return uni, dpb


def curve(cfg, idx):
    pts = []
    for key, per in cfg.items():
        sub = [per[i] for i in idx]
        R = sum(x[2] for x in sub)
        pts.append((sum(x[0] for x in sub) / R, sum(x[1] for x in sub) / R))
    return sorted(pts)


def interp(pts, x):
    if x <= pts[0][0]:
        return pts[0][1] * x / pts[0][0] if pts[0][0] > 0 else pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return pts[-1][1]


def gap(uni, dpb, idx):
    u, d = curve(uni, idx), curve(dpb, idx)
    diffs = [y - interp(u, x) for x, y in d if u[0][0] <= x <= u[-1][0]]
    return sum(diffs) / len(diffs), d, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=250)
    ap.add_argument("--base", default="lexical")
    ap.add_argument("--plain", action="store_true",
                    help="sub-question receivers only (no composite)")
    args = ap.parse_args()
    rows = mb.load_instances(args.instances, "branching")
    if not args.plain:
        rows = mb.add_composite_receivers(rows)
    print(f"{len(rows)} instances, base={args.base}, "
          f"{'plain' if args.plain else 'composite'}")
    out = {"base": args.base, "instances": len(rows),
           "composite": not args.plain, "alphas": []}
    rng = random.Random(0)
    n = len(rows)
    boots = [[rng.randrange(n) for _ in range(n)] for _ in range(1000)]
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        gold, norm = make_scorer(rows, alpha, args.base)
        uni, dpb = run(rows, alpha, gold, norm)
        g, d, u = gap(uni, dpb, list(range(n)))
        bs = sorted(gap(uni, dpb, b)[0] for b in boots)
        lo, hi = bs[25], bs[974]
        # uniform recall at the oracle-ish budget ~ 1 paragraph/receiver
        print(f"alpha={alpha:4.2f}  DP-uniform recall gap at matched "
              f"tokens = {g:+.3f} [{lo:+.3f},{hi:+.3f}]   "
              f"uniform k=1/3/5 recall = "
              f"{u[0][1]:.2f}/{u[2][1]:.2f}/{u[4][1]:.2f}")
        out["alphas"].append({"alpha": alpha, "gap": round(g, 4),
                              "ci": [round(lo, 4), round(hi, 4)],
                              "uniform_curve": [[round(x, 1), round(y, 4)] for x, y in u],
                              "dp_curve": [[round(x, 1), round(y, 4)] for x, y in d]})
    tag = f"{args.base}_{'plain' if args.plain else 'composite'}"
    path = RESULTS / f"summary_est_vs_alloc_{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
