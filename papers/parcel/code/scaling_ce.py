"""Seed-set vs per-receiver delivery recall as receivers are fused (n = 2, 8, 16),
with the lexical scorer and with the ms-marco cross-encoder.

Extends the paper's fused-instance scaling check (Sec. 5, "The separation
appears in the data") to the strongest scorer, so the seed-set collapse
cannot be attributed to a weak lexical ranking. Cross-instance pairs are
scored on the fly and added to results/scores_ce.json. No API calls.
Run: python3 scaling_ce.py
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import musique_benchmark as mb  # noqa: E402
import rebuttal_scorers as rs  # noqa: E402


def ensure_ce(rows):
    rs.build_cross_encoder("ce", "cross-encoder/ms-marco-MiniLM-L-6-v2", rows)


def measure(rows, scorer, cfgs):
    mb.SCORER = scorer
    out = []
    for name, kw in cfgs:
        t = r = n = 0
        for inst in rows:
            al = mb.POLICIES[name](inst, rng=random.Random(0), **kw)
            by = {p["idx"]: p for p in inst["paragraphs"]}
            for i, a in enumerate(inst["agents"]):
                t += sum(mb.tokens_of(by[j]) for j in al[i])
                r += a["needs"] in al[i]
                n += 1
        out.append({"policy": name, "cfg": kw, "tokens": round(t / n),
                    "recall": round(r / n, 3), "receivers": n})
    mb.SCORER = None
    return out


def main():
    base = mb.load_instances(240, "branching")
    res = {}
    for m in (1, 4, 8):
        rows = mb.merge_instances(base, m) if m > 1 else base
        ensure_ce(rows)
        ce = rs.load_scorer("ce")
        lex = measure(rows, None, [("centrality", {"k": 3}),
                                   ("topk", {"k": 1}), ("topk", {"k": 3})])
        cex = measure(rows, ce, [("centrality", {"k": 3}),
                                 ("centrality", {"k": 2 * m}),
                                 ("topk", {"k": 1, "density": False}),
                                 ("topk", {"k": 2, "density": False})])
        res[f"n={2 * m}"] = {"lexical": lex, "ce": cex}
        print(f"n={2 * m}")
        for tag, rr in (("lex", lex), ("ce ", cex)):
            for x in rr:
                print(f"  {tag} {x['policy']:10s} {json.dumps(x['cfg']):32s} "
                      f"tok/recv={x['tokens']:5d} recall={x['recall']:.3f}")
    path = mb.RESULTS / "summary_scaling_ce.json"
    path.write_text(json.dumps(res, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
