"""Read the MuSiQue run and report the token/accuracy frontier.

Reports policy DELTAS against broadcast, not absolute EM, because the
benchmark is contaminated (see musique_benchmark.py). Also splits out
the memorised instances -- those the model answers with NO context at
all -- and reports the frontier with and without them, since
memorisation is not arm-neutral.

Run: python3 analyze_musique.py [path]
"""

import collections
import json
import pathlib
import sys


def load(path):
    """Rows, de-duplicated by (instance, policy, config, agent).

    Two resume passes can briefly overlap and record the same call
    twice. A duplicate would inflate n and corrupt the denominator of
    every accuracy figure, so the first occurrence wins here rather
    than relying on the writer never racing.
    """
    seen, rows = set(), []
    for line in open(path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = (r.get("id"), r.get("policy"), r.get("cfg"), r.get("agent"))
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    return rows


def main():
    path = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
            else pathlib.Path(__file__).parent / "results" / "musique.jsonl")
    rows = load(path)
    usable = [r for r in rows if r["correct"] is not None]
    print(f"{len(rows)} records, {len(rows) - len(usable)} api errors\n")

    # Which (instance, agent) pairs are answered from parametric memory?
    memorised = {(r["id"], r["agent"]) for r in usable
                 if r["policy"] == "none" and r["correct"]}
    n_none = len({(r["id"], r["agent"]) for r in usable
                  if r["policy"] == "none"})
    if n_none:
        print(f"memorised (correct with NO context): {len(memorised)}/{n_none}"
              f" = {len(memorised) / n_none:.0%} of agent-instances\n")

    def table(rs, title):
        agg = collections.defaultdict(lambda: [0, 0, 0])
        for r in rs:
            a = agg[(r["policy"], r["cfg"])]
            a[0] += bool(r["correct"])
            a[1] += 1
            a[2] += r["tokens"]
        base = None
        for (pol, cfg), v in agg.items():
            if pol == "broadcast":
                base = (v[0] / v[1], v[2] / v[1])
        print(f"--- {title} ---")
        print(f"{'policy':11s} {'cfg':>13s} {'acc':>6s} {'tok/agent':>10s} "
              f"{'d_acc':>7s} {'tok_x':>7s} {'n':>5s}")
        order = sorted(agg.items(), key=lambda kv: kv[1][2] / max(kv[1][1], 1))
        rows_out = []
        for (pol, cfg), v in order:
            if not v[1]:
                continue
            acc, tok = v[0] / v[1], v[2] / v[1]
            if base:
                d = f"{acc - base[0]:+.3f}"
                x = f"{base[1] / tok:.1f}x" if tok > 0 else "inf"
            else:
                d = x = "-"
            print(f"{pol:11s} {cfg:>13s} {acc:6.3f} {tok:10.0f} "
                  f"{d:>7s} {x:>7s} {v[1]:5d}")
            rows_out.append({"policy": pol, "cfg": cfg, "acc": round(acc, 4),
                             "tokens_per_agent": round(tok, 1), "n": v[1],
                             "d_acc_vs_broadcast":
                                 round(acc - base[0], 4) if base else None,
                             "token_ratio_vs_broadcast":
                                 round(base[1] / tok, 2) if base and tok else None})
        print()
        return rows_out

    all_rows = table(usable, "ALL agent-instances")
    clean = [r for r in usable if (r["id"], r["agent"]) not in memorised]
    clean_rows = None
    if clean and len(clean) != len(usable):
        clean_rows = table(clean, "NON-MEMORISED only (contamination-controlled)")

    print("d_acc = accuracy change vs broadcast. tok_x = broadcast tokens / "
          "this policy's tokens (higher is cheaper).")

    # The per-call JSONL is gitignored as regenerable; the summary is what
    # the manuscript cites, so it is committed.
    # A summary read off a partial run must say so: the per-policy n is
    # the only thing separating a settled result from a snapshot, and a
    # bare accuracy number invites being quoted as final.
    per_policy_n = min((r["n"] for r in all_rows), default=0)
    summary = {
        "source": path.name,
        "status": ("interim snapshot -- run still in progress"
                   if per_policy_n < 20 else "complete"),
        "min_n_per_policy": per_policy_n,
        "caveat": ("With n per policy this small the standard error on a "
                   "proportion near 0.6 is ~0.13, so differences below "
                   "roughly 0.26 are not distinguishable from noise."),
        "records": len(rows),
        "api_errors": len(rows) - len(usable),
        "memorised_fraction": (len(memorised) / n_none) if n_none else None,
        "all": all_rows,
        "non_memorised": clean_rows,
    }
    out = path.parent / "summary_musique.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
