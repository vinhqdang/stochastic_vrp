"""Allocation policies on a PUBLIC benchmark: MuSiQue-Ans.

WHY A PUBLIC BENCHMARK
----------------------
`allocation_experiment.py` runs on a task we wrote, which invites the
obvious objection that we evaluated on our own benchmark. This runs the
same policies on MuSiQue-Ans (Trivedi et al., TACL 2022, CC BY 4.0),
where the structure PARCEL needs is ANNOTATED IN THE DATA rather than
constructed by us:

  fact pool  = the 20 `paragraphs`, each carrying `is_supporting`
  agents     = the `question_decomposition` sub-questions, each carrying
               `paragraph_support_idx` -- the one paragraph answering it
  cost       = tokens actually transmitted, summed over agents

Ground-truth relevance is labelled, so an oracle allocation is
computable and every policy can be scored against it.

WHAT IS OURS AND MUST BE DISCLOSED
----------------------------------
MuSiQue is published as a SINGLE-agent QA task. Splitting it across
per-sub-question agents is our framing, not a native property, and the
paper must say so.

Most MuSiQue decompositions are sequential -- sub-question 2 reads
"#1 >> spouse", referencing hop 1's answer -- which is a pipeline, not
parallel receivers. We therefore restrict to the BRANCHING families
(3hop2, 4hop2, 4hop3), where two sub-questions carry no back-reference
and are genuinely independent. Verified on the dev split: all 351 such
instances have exactly 2 independent sub-questions. The 2hop family is
retained as a single-receiver ablation.

CONTAMINATION
-------------
MuSiQue predates these models and its single-hop seeds overlap
SQuAD/NQ, so some answers are memorised. Two consequences, both handled
rather than hoped away:

  1. Memorisation COMPRESSES our effect -- a model answering from
     parametric memory does not need the gold paragraph, so starving an
     agent looks as good as feeding it. This biases AGAINST the method,
     making any positive result conservative.
  2. It is NOT arm-neutral. Memorised recall degrades more in long
     distractor-heavy contexts than in short clean ones, so a small
     -budget arm could win partly through a memory channel unrelated to
     our mechanism.

The `none` policy is therefore not merely a floor, it is the
contamination control: question only, zero paragraphs. Any sub-question
answered correctly with no context at all is memorised, and the analysis
reports results both with and without those instances.

Report POLICY DELTAS, never absolute EM.

USAGE
  python3 musique_benchmark.py --dry-run              # free, no API
  python3 musique_benchmark.py --instances 150
"""

import argparse
import json
import os
import pathlib
import random
import re
import string
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from degradation_sweep import call_gemini, TOKENS_PER_WORD  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
BRANCHING = {"3hop2", "4hop2", "4hop3"}
STOP = set("""a an the of in on at to for and or is was were be been by with
from as that this it its his her their who whom which what when where how
did does do done "" '' s""".split())


def load_instances(limit, family="branching", seed=0):
    from datasets import load_dataset
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    want = BRANCHING if family == "branching" else {"2hop"}
    rows = []
    for r in ds:
        if r["id"].split("__")[0] not in want:
            continue
        indep = [q for q in r["question_decomposition"]
                 if "#" not in q["question"]]
        if len(indep) < 2:
            continue
        rows.append({
            "id": r["id"],
            "paragraphs": [{"idx": p["idx"], "title": p["title"],
                            "text": p["paragraph_text"],
                            "gold": bool(p["is_supporting"])}
                           for p in r["paragraphs"]],
            "agents": [{"q": q["question"], "answer": q["answer"],
                        "needs": q["paragraph_support_idx"]}
                       for q in indep],
        })
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def norm(s):
    s = s.lower().strip()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    """Exact match after MuSiQue-style normalisation, plus containment.

    Containment is included because a model may answer "Steve Hillage,
    the performer" where the gold is "Steve Hillage". Being strict there
    would penalise every arm equally but add noise for no gain.
    """
    p, g = norm(pred), norm(gold)
    return bool(g) and (p == g or g in p)


def tokens_of(par):
    return max(1, round(len(par["text"].split()) * TOKENS_PER_WORD))


# Swapped at runtime by --scorer embed. Kept module-level so the policy
# functions stay pure lookups and the two scorers are exactly comparable.
SCORER = None


def relevance(par, agent):
    """Relevance signal. Lexical by default; embeddings via --scorer embed.

    Both are imperfect on purpose -- a real router has an approximate
    scorer, not gold labels.
    """
    if SCORER is not None:
        return SCORER(par, agent)
    return _lexical(par, agent)


def _lexical(par, agent):
    """Cheap bag-of-words overlap."""
    q = set(re.findall(r"[a-z0-9]+", agent["q"].lower())) - STOP
    t = set(re.findall(r"[a-z0-9]+",
                       (par["title"] + " " + par["text"]).lower())) - STOP
    if not q or not t:
        return 0.0
    return len(q & t) / (len(q) ** 0.5 * len(t) ** 0.5)


def rho_prime(load, c_star, scale):
    """Marginal saturation cost: rises to c*, then flattens (PROJECT.md 11.1)."""
    x = load / c_star
    return scale * (2.0 * x) / (1.0 + x * x)


def auto_scale(densities, quantile=0.5):
    """Set the saturation price in the units of THIS pool.

    Hand-picking `scale` is not portable: MuSiQue paragraphs are ~10x
    larger than the synthetic facts, so relevance densities shift by an
    order of magnitude and a constant tuned on one dataset cuts delivery
    off immediately on the other. That is a calibration artefact, not a
    property of the rule.

    Instead, anchor the price so that at load = c* it equals a quantile
    of the candidate pool's own density distribution. The rule then
    reads "stop once the marginal item is no better than a typical
    candidate", which is scale-free and needs no per-dataset tuning --
    an advantage over top-k, whose k must be tuned per dataset.

    Uses only the candidate densities, never the gold labels.
    """
    vals = sorted(d for d in densities if d > 0)
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, int(quantile * len(vals)))]


# ---------------------------------------------------------------- policies

def p_broadcast(inst, **_):
    ids = [p["idx"] for p in inst["paragraphs"]]
    return {i: list(ids) for i, _ in enumerate(inst["agents"])}


def p_none(inst, **_):
    return {i: [] for i, _ in enumerate(inst["agents"])}


def p_oracle(inst, **_):
    return {i: [a["needs"]] for i, a in enumerate(inst["agents"])}


def p_random(inst, k=3, rng=None, **_):
    ids = [p["idx"] for p in inst["paragraphs"]]
    return {i: rng.sample(ids, min(k, len(ids)))
            for i, _ in enumerate(inst["agents"])}


def p_centrality(inst, k=3, **_):
    """The classical-IM answer: one globally-ranked seed set for everyone."""
    score = {p["idx"]: sum(relevance(p, a) for a in inst["agents"])
             for p in inst["paragraphs"]}
    top = sorted(score, key=score.get, reverse=True)[:k]
    return {i: list(top) for i, _ in enumerate(inst["agents"])}


def p_topk(inst, k=3, **_):
    """Per-agent relevance ranking. The single-price baseline."""
    out = {}
    for i, a in enumerate(inst["agents"]):
        ranked = sorted(inst["paragraphs"],
                        key=lambda p: relevance(p, a) / tokens_of(p),
                        reverse=True)
        out[i] = [p["idx"] for p in ranked[:k]]
    return out


def p_parcel(inst, lam=0.0, c_star=1500.0, q=0.5, **_):
    """Two prices: a global budget price, and the receiver's own rising one.

    `lam` is the global budget price (swept to trace the frontier); the
    saturation price is self-calibrating via auto_scale, so the rule
    carries no dataset-specific constant.
    """
    out = {}
    for i, a in enumerate(inst["agents"]):
        dens = {p["idx"]: relevance(p, a) / tokens_of(p)
                for p in inst["paragraphs"]}
        scale = auto_scale(dens.values(), q)
        ranked = sorted(inst["paragraphs"], key=lambda p: dens[p["idx"]],
                        reverse=True)
        chosen, load = [], 0.0
        for p in ranked:
            if dens[p["idx"]] >= lam + rho_prime(load, c_star, scale):
                chosen.append(p["idx"])
                load += tokens_of(p)
        out[i] = chosen
    return out


POLICIES = {"broadcast": p_broadcast, "none": p_none, "oracle": p_oracle,
            "random": p_random, "centrality": p_centrality,
            "topk": p_topk, "parcel": p_parcel}

CONFIGS_FULL = (
    [("broadcast", {}), ("none", {}), ("oracle", {})]
    + [("random", {"k": k}) for k in (1, 3, 6)]
    + [("centrality", {"k": k}) for k in (1, 3, 6)]
    + [("topk", {"k": k}) for k in (1, 2, 3, 5, 10)]
    + [("parcel", {"q": q}) for q in (0.95, 0.85, 0.7, 0.5, 0.3)]
)

# The Gemini free tier allows 500 requests/day PER MODEL. The full grid
# is 19 configs x 2 agents = 38 calls per instance, which buys only ~13
# instances a day. This trimmed grid keeps one point per baseline family
# plus the frontier points that matter -- 12 configs, 24 calls per
# instance -- so ~20 instances fit inside one model's daily allowance.
CONFIGS_MIN = (
    [("broadcast", {}), ("none", {}), ("oracle", {})]
    + [("random", {"k": 3}), ("centrality", {"k": 3})]
    + [("topk", {"k": k}) for k in (1, 3, 5, 10)]
    + [("parcel", {"q": q}) for q in (0.95, 0.7, 0.3)]
)

# For the POWERED test, only the four arms that carry the claims. At 8
# calls per instance instead of 24, one model's 500/day allowance buys
# ~60 instances rather than ~20 -- and power on these four comparisons
# is what decides whether the mechanism separates from a tuned top-k.
#   broadcast  the cost baseline
#   oracle     the ceiling, and the contamination-free upper bound
#   topk k=5   the single-price baseline at its best observed setting
#   parcel     the two-price rule at its most permissive setting
CONFIGS_CORE = [("broadcast", {}), ("oracle", {}),
                ("topk", {"k": 5}), ("parcel", {"q": 0.3})]

CONFIGS = CONFIGS_MIN


def build_prompt(inst, idx_list, agent, rng):
    by_idx = {p["idx"]: p for p in inst["paragraphs"]}
    paras = [by_idx[j] for j in idx_list if j in by_idx]
    rng.shuffle(paras)
    if paras:
        ctx = "\n\n".join(f"[{p['title']}] {p['text']}" for p in paras)
    else:
        ctx = "(no documents provided)"
    return (f"{ctx}\n\nQuestion: {agent['q']}\n\n"
            f"Answer from the documents above. If the answer is not there, "
            f"reply UNKNOWN. Reply with only the answer, no explanation.\n"
            f"End your reply with the line: ANSWER: <answer>")


def extract(reply):
    if reply.startswith("__ERROR__"):
        return None
    m = re.search(r"ANSWER:\s*(.+)", reply, re.IGNORECASE)
    return (m.group(1) if m else reply.strip().splitlines()[-1]).strip()


def dry_run(rows):
    """Tokens and gold-recall per policy. No API calls."""
    print(f"{'policy':11s} {'cfg':>14s} {'tok/agent':>10s} {'recall':>8s}")
    for name, kw in CONFIGS:
        tok = rec = n = 0
        for inst in rows:
            alloc = POLICIES[name](inst, rng=random.Random(0), **kw)
            by_idx = {p["idx"]: p for p in inst["paragraphs"]}
            for i, a in enumerate(inst["agents"]):
                ids = alloc[i]
                tok += sum(tokens_of(by_idx[j]) for j in ids if j in by_idx)
                rec += 1.0 if a["needs"] in ids else 0.0
                n += 1
        print(f"{name:11s} {json.dumps(kw):>14s} {tok / n:10.0f} {rec / n:8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--instances", type=int, default=150)
    ap.add_argument("--family", default="branching",
                    choices=["branching", "2hop"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scorer", default="lexical",
                    choices=["lexical", "embed"],
                    help="relevance signal: bag-of-words, or cached "
                         "embedding cosine (recommended)")
    ap.add_argument("--grid", default="min", choices=["core", "min", "full"],
                    help="core=4 arms (8 calls/instance -- best power per "
                         "unit of the 500/day/model allowance); min=12; "
                         "full=19")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global CONFIGS
    CONFIGS = {"core": CONFIGS_CORE, "min": CONFIGS_MIN,
               "full": CONFIGS_FULL}[args.grid]
    rows = load_instances(args.instances, args.family)
    print(f"{len(rows)} instances, {sum(len(r['agents']) for r in rows)} agents")

    if args.scorer == "embed":
        global SCORER
        from embed_relevance import build_scorer
        print("embedding paragraphs and sub-questions (cached on disk)...")
        SCORER = build_scorer(rows)
        print("scorer ready")

    if args.dry_run:
        dry_run(rows)
        return

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")

    tag = args.model.replace("/", "-")
    out = (pathlib.Path(args.out) if args.out
           else RESULTS / f"musique_{tag}_{args.scorer}_{args.grid}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip work already present in the output file. Runs stop when
    # a daily allowance runs out, so re-running with fresh keys is the
    # normal path -- without this it redoes everything and appends
    # duplicate rows, which would silently double-count in the analysis.
    done = set()
    if out.exists():
        with open(out) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # tolerate a torn final line
                if r.get("correct") is not None:
                    done.add((r["id"], r["policy"], r["cfg"], r["agent"]))
    if done:
        print(f"resuming: {len(done)} calls already recorded")

    jobs, meta = [], []
    for inst in rows:
        by_idx = {p["idx"]: p for p in inst["paragraphs"]}
        for name, kw in CONFIGS:
            alloc = POLICIES[name](inst, rng=random.Random(hash(inst["id"]) % 9),
                                   **kw)
            for i, a in enumerate(inst["agents"]):
                if (inst["id"], name, json.dumps(kw), i) in done:
                    continue
                ids = alloc[i]
                jobs.append(build_prompt(inst, ids, a, random.Random(i)))
                meta.append({"id": inst["id"], "policy": name,
                             "cfg": json.dumps(kw), "agent": i,
                             "tokens": sum(tokens_of(by_idx[j])
                                           for j in ids if j in by_idx),
                             "n_sent": len(ids),
                             "had_gold": a["needs"] in ids,
                             "gold": a["answer"], "model": args.model})

    print(f"{len(jobs)} calls -> {out}")
    with open(out, "a") as fh, ThreadPoolExecutor(args.workers) as pool:
        for n, (m, reply) in enumerate(
                zip(meta, pool.map(lambda p: call_gemini(args.model, p, key),
                                   jobs)), 1):
            if reply.startswith("__QUOTA__"):
                # Daily free-tier quota gone. Continuing only burns
                # wall-clock and writes junk rows; the partial file up to
                # here is still usable.
                print(f"\nQUOTA EXHAUSTED for {args.model} after {n - 1} "
                      f"calls; partial results kept.", file=sys.stderr)
                break
            pred = extract(reply)
            rec = dict(m)
            rec["pred"] = (pred or "")[:120]
            rec["correct"] = None if pred is None else em(pred, m["gold"])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}", file=sys.stderr, flush=True)
    print("done")


if __name__ == "__main__":
    main()
