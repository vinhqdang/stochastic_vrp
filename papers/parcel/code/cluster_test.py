"""Cluster-robust paired comparison between two policies.

Each MuSiQue instance contributes 2 (branching) or 3 (composite)
receivers that share a paragraph pool and difficulty, so treating all
receivers as independent draws understates variance. This resamples and
permutes at the INSTANCE level (every receiver from a chosen instance
moves together), which is the primary statistic reported in Table 2 and
in Section 7's cross-model comparison.

Two numbers per comparison:
  Cluster block bootstrap  -- resample instances with replacement,
    recompute the accuracy difference each time, report the 95% CI.
  Cluster permutation test -- for each instance, independently swap
    which policy's outcomes are labelled "A" and which "B" with
    probability 0.5, recompute the difference, and report the two-sided
    fraction of permutations at least as extreme as observed. Valid
    under the sharp null of no policy effect on any receiver.

Also reports the flat (receiver-level) McNemar exact test for reference
-- this is what NON-cluster-robust inference would report, and Table 2's
caption quotes how much it differs from the cluster-robust numbers.

Run: python3 cluster_test.py <jsonl> <policy_a>[:cfg] <policy_b>[:cfg]
"""

import collections
import json
import math
import random
import sys


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_policy = collections.defaultdict(dict)
    for r in rows:
        if r["correct"] is None:
            continue
        key = (r["id"], r["agent"])
        name = r["policy"] + (r["cfg"] if r["cfg"] != "{}" else "")
        by_policy[name][key] = bool(r["correct"])
    return by_policy


def paired_keys(a, b):
    return sorted(set(a) & set(b))


def by_instance(keys):
    out = collections.defaultdict(list)
    for (iid, agent) in keys:
        out[iid].append((iid, agent))
    return out


def mcnemar(a, b, keys):
    n01 = sum(1 for k in keys if not a[k] and b[k])
    n10 = sum(1 for k in keys if a[k] and not b[k])
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    lo = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    return n10, n01, min(1.0, p)


def delta(a, b, keys):
    return (sum(a[k] for k in keys) - sum(b[k] for k in keys)) / len(keys)


def cluster_bootstrap(a, b, inst_keys, iters=10000, seed=0):
    rng = random.Random(seed)
    insts = list(inst_keys)
    diffs = []
    for _ in range(iters):
        sample = [insts[rng.randrange(len(insts))] for _ in insts]
        keys = [k for iid in sample for k in inst_keys[iid]]
        diffs.append(delta(a, b, keys))
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters) - 1]
    return lo, hi


def cluster_permutation(a, b, inst_keys, observed, iters=10000, seed=0):
    rng = random.Random(seed)
    insts = list(inst_keys)
    count = 0
    for _ in range(iters):
        flips = {iid: rng.random() < 0.5 for iid in insts}
        acc = 0.0
        n = 0
        for iid in insts:
            for k in inst_keys[iid]:
                x, y = (a[k], b[k]) if not flips[iid] else (b[k], a[k])
                acc += x - y
                n += 1
        if abs(acc / n) >= abs(observed) - 1e-12:
            count += 1
    return count / iters


def run(path, name_a, name_b):
    by_policy = load(path)
    a, b = by_policy[name_a], by_policy[name_b]
    keys = paired_keys(a, b)
    n_instances = len({iid for iid, _ in keys})
    print(f"{path}: {name_a} vs {name_b}")
    print(f"  paired receivers: {len(keys)}  source instances: {n_instances}")

    d = delta(a, b, keys)
    n10, n01, p_flat = mcnemar(a, b, keys)
    print(f"  delta acc: {d:+.4f}")
    print(f"  flat McNemar (receiver-level): n10={n10} n01={n01} p={p_flat:.4f}")

    inst_keys = by_instance(keys)
    lo, hi = cluster_bootstrap(a, b, inst_keys)
    p_perm = cluster_permutation(a, b, inst_keys, d)
    print(f"  cluster bootstrap 95% CI: [{lo:+.3f}, {hi:+.3f}]")
    print(f"  cluster permutation p: {p_perm:.4f}")
    return {"n_receivers": len(keys), "n_instances": n_instances,
            "delta": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
            "p_flat_mcnemar": round(p_flat, 4),
            "p_cluster_perm": round(p_perm, 4)}


if __name__ == "__main__":
    path, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    run(path, a, b)
