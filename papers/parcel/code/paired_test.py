"""Paired comparisons between allocation policies.

Every policy answers the SAME (instance, agent) pairs, so comparing
marginal accuracies throws away the pairing and badly understates power.
On 36 observations the marginal standard error is ~0.07, which makes any
realistic difference look like noise; the paired test conditions on the
items both policies got right or wrong together and asks only about the
disagreements.

Two tests per comparison:
  McNemar (exact binomial on the discordant pairs) -- the standard test
    for paired binary outcomes.
  Paired bootstrap over items -- resamples (instance, agent) pairs and
    reports a CI on the accuracy difference, which also handles the
    token difference on the same resamples.

Reports accuracy difference AND token ratio together, because the claim
under test is about the trade, not either axis alone.

Run: python3 paired_test.py results/musique_<model>.jsonl
"""

import collections
import json
import math
import pathlib
import random
import sys


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_policy = collections.defaultdict(dict)
    for r in rows:
        if r["correct"] is None:
            continue
        key = (r["id"], r["agent"])
        by_policy[f"{r['policy']}{r['cfg'] if r['cfg'] != '{}' else ''}"][key] = (
            bool(r["correct"]), r["tokens"])
    return by_policy


def mcnemar(a, b, keys):
    """Exact two-sided McNemar on discordant pairs."""
    n01 = sum(1 for k in keys if not a[k][0] and b[k][0])
    n10 = sum(1 for k in keys if a[k][0] and not b[k][0])
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    lo = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    return n10, n01, min(1.0, p)


def boot(a, b, keys, iters=4000, seed=0):
    rng = random.Random(seed)
    keys = list(keys)
    diffs, ratios = [], []
    for _ in range(iters):
        s = [keys[rng.randrange(len(keys))] for _ in keys]
        ca = sum(a[k][0] for k in s) / len(s)
        cb = sum(b[k][0] for k in s) / len(s)
        ta = sum(a[k][1] for k in s) / len(s)
        tb = sum(b[k][1] for k in s) / len(s)
        diffs.append(ca - cb)
        if ta > 0:
            ratios.append(tb / ta)
    diffs.sort()
    ratios.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    rmid = ratios[len(ratios) // 2] if ratios else float("nan")
    return lo, hi, rmid


def compare(bp, x, y):
    if x not in bp or y not in bp:
        return None
    keys = sorted(set(bp[x]) & set(bp[y]))
    if not keys:
        return None
    ax = sum(bp[x][k][0] for k in keys) / len(keys)
    ay = sum(bp[y][k][0] for k in keys) / len(keys)
    tx = sum(bp[x][k][1] for k in keys) / len(keys)
    ty = sum(bp[y][k][1] for k in keys) / len(keys)
    w, l, p = mcnemar(bp[x], bp[y], keys)
    lo, hi, _ = boot(bp[x], bp[y], keys)
    return dict(x=x, y=y, n=len(keys), acc_x=ax, acc_y=ay, d=ax - ay,
                ci=(lo, hi), p=p, wins=w, losses=l,
                tok_x=tx, tok_y=ty, ratio=(ty / tx) if tx else float("inf"))


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path:
        sys.exit("usage: paired_test.py <results.jsonl>")
    bp = load(path)
    print(f"policies: {', '.join(sorted(bp))}\n")

    pairs = [
        # The claim that matters: can we cut tokens without losing accuracy?
        ("parcel{\"q\": 0.3}", "broadcast"),
        ("topk{\"k\": 5}", "broadcast"),
        ("parcel{\"q\": 0.7}", "broadcast"),
        ("oracle", "broadcast"),
        # Does the second price earn its keep against a single price?
        ("parcel{\"q\": 0.3}", "topk{\"k\": 10}"),
        ("parcel{\"q\": 0.7}", "topk{\"k\": 5}"),
        ("parcel{\"q\": 0.95}", "topk{\"k\": 3}"),
        # Does the classical-IM style seeding fail as predicted?
        ("topk{\"k\": 3}", "centrality{\"k\": 3}"),
    ]
    print(f"{'comparison':38s} {'n':>4s} {'dAcc':>7s} {'95% CI':>16s} "
          f"{'p':>7s} {'tok x':>7s}")
    for x, y in pairs:
        r = compare(bp, x, y)
        if not r:
            continue
        ci = f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
        star = "*" if r["p"] < 0.05 else " "
        print(f"{x + ' vs ' + y:38s} {r['n']:4d} {r['d']:+7.3f} {ci:>16s} "
              f"{r['p']:7.3f}{star} {r['ratio']:6.1f}x")
    print("\ndAcc = first minus second. tok x = second's tokens / first's "
          "(>1 means the first policy is cheaper). * = p < 0.05 (McNemar).")


if __name__ == "__main__":
    main()
