"""Does the two-price rule buy task quality per token? End-to-end test.

THE QUESTION THIS ANSWERS
-------------------------
The structure theorem (PROJECT.md §12) says what the objective looks
like. It does not show the policy is worth anything. The claim a
referee -- and a practitioner -- actually cares about is:

    at equal task quality, does PARCEL spend fewer tokens?
    at equal tokens, does it deliver higher quality?

And the sharper claim the theory predicts, which no purely-submodular
account can make: because utility is NON-MONOTONE in delivered context,
full broadcast is not merely wasteful, it is *actively worse*. If
saturation is real, sending everything should lose to sending less on
BOTH axes at once. That is the headline to test, and it is falsifiable:
if broadcast wins on accuracy, the paper's premise is wrong.

SETUP
-----
n agents, one shared pool of facts. Each agent holds a private sub-task
whose answer depends on a small subset of the pool. An allocation
decides which facts reach which agent. Every fact sent to every agent
costs its tokens -- information is copyable, but transmission is billed
per receiver, which is what keeps the budget meaningful.

Utility = fraction of sub-tasks answered correctly (measured by actually
asking a model, not simulated).
Cost    = total tokens transmitted across all agents.

POLICIES
--------
broadcast   send every fact to every agent. Current practice, and the
            cost baseline to beat.
none        send nothing. The utility floor.
random      random facts per agent at a matched budget.
centrality  the "what if you just used the classical IM answer" control:
            rank facts by how many agents they score well for, seed the
            top ones to everyone. Ignores saturation -- so the theory
            predicts it over-concentrates and saturates receivers.
topk        per-agent relevance ranking, no saturation term. This is the
            single-price baseline -- what prior work does.
parcel      the two-price rule (§12.3): send fact f to agent i iff
                score_i(f)/c(f)  >=  lambda + rho'_i(conf_i) * w_i(f)/c(f)
            lambda is swept to trace the budget/quality frontier.

`topk` vs `parcel` is the load-bearing comparison: identical relevance
information, the ONLY difference being the second (saturation) price.
If parcel does not beat topk, the second price earns nothing and the
paper's central mechanism is not doing work.

Relevance scores are deliberately IMPERFECT -- lexical overlap between a
fact and the agent's sub-task, which is what a cheap real system would
have. Using ground-truth relevance would assume away the problem.

USAGE
  python3 allocation_experiment.py --pilot
  python3 allocation_experiment.py --model gemini-3.5-flash-lite --trials 12
"""

import argparse
import json
import math
import os
import pathlib
import random
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from degradation_sweep import call_gemini, TOKENS_PER_WORD  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"

TOPICS = ["shipping", "payroll", "inventory", "routing", "billing",
          "staffing", "storage", "transit", "vendor", "customs"]
ATTRS = ["deadline", "capacity", "rate", "delay", "quota", "margin"]

# Function words carry no signal but appear in every fact, so leaving them
# in flattens the score to near-uniform (0.091 vs 0.061 across the whole
# pool) and the ranking stops discriminating. Real retrieval scores are
# imperfect, not uninformative -- this keeps them imperfect.
STOP = {"the", "for", "is", "of", "and", "what", "sum", "unit", "a", "in"}


def make_world(n_agents, n_facts, rng, hetero=True):
    """A pool of facts, and per-agent sub-tasks over a subset of them.

    Each sub-task sums several facts, so it is genuinely multi-hop: an
    agent missing any needed fact cannot answer, and extra facts are pure
    distraction. Both failure modes stay visible -- under-delivery and
    over-delivery.

    HETEROGENEITY IS LOAD-BEARING, and was added after a dry run showed
    why. With every agent needing exactly two facts, a single global
    top-k is already near-optimal (k=5 reached full recall on 55 tokens,
    beating the two-price rule's 221), so the per-receiver price earned
    nothing. That is a fair result for the homogeneous case and is
    reported as such.

    Real deployments are not homogeneous: a summariser needs a handful
    of facts, an auditor needs many. When agents differ, one global k
    must over-serve some and starve others, while a per-receiver price
    adapts. `hetero=False` recovers the homogeneous case so the contrast
    can be measured rather than asserted.
    """
    facts, values = [], {}
    for j in range(n_facts):
        topic = TOPICS[j % len(TOPICS)]
        attr = ATTRS[(j // len(TOPICS)) % len(ATTRS)]
        key = f"{topic}-{attr}-{j}"
        val = rng.randint(10, 99)
        values[key] = val
        facts.append({"id": j, "key": key,
                      "text": f"The {attr} for {topic} unit {j} is {val}."})

    agents = []
    for i in range(n_agents):
        n_need = rng.choice([1, 2, 2, 3, 5, 8]) if hetero else 2
        picked = rng.sample(range(n_facts), n_need)
        parts = [f"the {facts[j]['key'].split('-')[1]} for "
                 f"{facts[j]['key'].split('-')[0]} unit {facts[j]['id']}"
                 for j in picked]
        question = "What is the sum of " + ", ".join(parts) + "?"
        agents.append({"id": i, "question": question,
                       "needs": set(picked),
                       "answer": sum(values[facts[j]["key"]] for j in picked)})
    return facts, agents


def relevance(fact, agent):
    """Cheap, imperfect lexical score -- what a real system would have.

    Deliberately not ground truth: it rewards token overlap with the
    sub-task, so unrelated facts sharing a topic word score misleadingly
    high. That imperfection is the point.
    """
    q = set(re.findall(r"[a-z]+|\d+", agent["question"].lower())) - STOP
    f = set(re.findall(r"[a-z]+|\d+", fact["text"].lower())) - STOP
    if not f:
        return 0.0
    return len(q & f) / len(f)


def cost_tokens(fact):
    return max(1, round(len(fact["text"].split()) * TOKENS_PER_WORD))


def rho_prime(load, c_star=1500.0, scale=0.10):
    """Marginal saturation cost, convex below c* then flattening.

    Shape follows the S-curve measured in PROJECT.md §11.1: rising
    marginal damage up to the effective length, then saturating. Units
    are per-token so it compares directly against relevance density.

    `scale` is calibrated so the saturation price is negligible at light
    loads and becomes comparable to a typical relevance density only as
    the load approaches c*. Miscalibrating it upward makes the rule cut
    off almost immediately, which is a bug, not a finding.
    """
    x = load / c_star
    return scale * (2.0 * x) / (1.0 + x * x)


# ---------------------------------------------------------------- policies

def alloc_broadcast(facts, agents, **_):
    return {a["id"]: [f["id"] for f in facts] for a in agents}


def alloc_none(facts, agents, **_):
    return {a["id"]: [] for a in agents}


def alloc_random(facts, agents, rng=None, k=3, **_):
    return {a["id"]: [f["id"] for f in rng.sample(facts, min(k, len(facts)))]
            for a in agents}


def alloc_centrality(facts, agents, k=3, **_):
    """Classical-IM-style seeding: globally 'influential' facts to all.

    Ranks by summed relevance across agents and sends the same top-k to
    everyone -- structurally the seed-set answer. Predicted to fail by
    over-concentrating on hub facts and saturating every receiver.
    """
    score = {f["id"]: sum(relevance(f, a) for a in agents) for f in facts}
    top = sorted(score, key=score.get, reverse=True)[:k]
    return {a["id"]: list(top) for a in agents}


def alloc_topk(facts, agents, k=3, **_):
    """Per-agent relevance ranking. The single-price baseline."""
    out = {}
    for a in agents:
        ranked = sorted(facts, key=lambda f: relevance(f, a) / cost_tokens(f),
                        reverse=True)
        out[a["id"]] = [f["id"] for f in ranked[:k]]
    return out


def alloc_parcel(facts, agents, lam=0.002, c_star=1500.0, scale=0.10, **_):
    """The two-price rule.

    Greedy by relevance density; admit a fact only while it clears BOTH
    the global budget price lambda and the receiver's own rising
    saturation price. Sweeping lambda traces the frontier.
    """
    out = {}
    for a in agents:
        ranked = sorted(facts, key=lambda f: relevance(f, a) / cost_tokens(f),
                        reverse=True)
        chosen, load = [], 0.0
        for f in ranked:
            c = cost_tokens(f)
            w = c                      # uniform weighting: w_i(f) = tokens
            density = relevance(f, a) / c
            # Both sides are per-token, so the two prices are comparable.
            local_price = rho_prime(load, c_star, scale) * w / c
            if density >= lam + local_price:
                chosen.append(f["id"])
                load += w
        out[a["id"]] = chosen
    return out


POLICIES = {
    "broadcast": alloc_broadcast, "none": alloc_none, "random": alloc_random,
    "centrality": alloc_centrality, "topk": alloc_topk, "parcel": alloc_parcel,
}


# ---------------------------------------------------------------- evaluation

def build_agent_prompt(fact_ids, facts, agent, rng):
    sent = [facts[j]["text"] for j in fact_ids]
    rng.shuffle(sent)
    ctx = " ".join(sent) if sent else "(no information provided)"
    return (f"{ctx}\n\n{agent['question']}\n\n"
            f"If the information needed is not present, reply UNKNOWN.\n"
            f"End your reply with the line: ANSWER: <number or UNKNOWN>")


def score_reply(reply, answer):
    if reply.startswith("__ERROR__"):
        return None
    clean = reply.replace(",", "")
    m = re.search(r"ANSWER:\s*(-?\d+)", clean, re.IGNORECASE)
    if m:
        return int(m.group(1)) == answer
    nums = re.findall(r"-?\d+", clean)
    return bool(nums) and int(nums[-1]) == answer


def run_trial(model, key, seed, n_agents, n_facts, configs, workers,
              hetero=True):
    rng = random.Random(seed)
    facts, agents = make_world(n_agents, n_facts, rng, hetero)
    jobs, meta = [], []
    for name, kwargs in configs:
        alloc = POLICIES[name](facts, agents, rng=random.Random(seed), **kwargs)
        for a in agents:
            ids = alloc[a["id"]]
            tokens = sum(cost_tokens(facts[j]) for j in ids)
            jobs.append(build_agent_prompt(ids, facts, a,
                                           random.Random(seed + a["id"])))
            meta.append({"policy": name, "cfg": json.dumps(kwargs),
                         "agent": a["id"], "tokens": tokens,
                         "answer": a["answer"], "seed": seed,
                         "n_sent": len(ids),
                         "had_needed": a["needs"] <= set(ids)})

    with ThreadPoolExecutor(workers) as pool:
        replies = list(pool.map(lambda p: call_gemini(model, p, key), jobs))

    out = []
    for m, r in zip(meta, replies):
        m = dict(m)
        m["correct"] = score_reply(r, m["answer"])
        m["model"] = model
        out.append(m)
    return out


def dry_run(args):
    """Allocation statistics with no model calls.

    Reports, per policy: mean tokens sent per agent, and the RECALL of
    the two facts each agent actually needs. Recall is the ceiling on
    achievable accuracy -- a policy that never delivers the needed facts
    cannot be rescued by the model. Cheap sanity check before spending
    quota.
    """
    configs = (
        [("broadcast", {}), ("none", {})]
        + [("random", {"k": k}) for k in (5, 25, 100)]
        + [("centrality", {"k": k}) for k in (5, 25, 100)]
        + [("topk", {"k": k}) for k in (2, 5, 10, 25, 60, 150)]
        + [("parcel", {"lam": l}) for l in
           (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01)]
    )
    print(f"{'policy':11s} {'cfg':>16s} {'tok/agent':>10s} {'recall':>8s}")
    for name, kwargs in configs:
        tok = rec = cnt = 0
        for t in range(args.trials):
            rng = random.Random(1000 + t)
            facts, agents = make_world(args.agents, args.facts, rng,
                                       not args.homogeneous)
            alloc = POLICIES[name](facts, agents,
                                   rng=random.Random(1000 + t), **kwargs)
            for a in agents:
                ids = set(alloc[a["id"]])
                tok += sum(cost_tokens(facts[j]) for j in ids)
                rec += len(a["needs"] & ids) / len(a["needs"])
                cnt += 1
        print(f"{name:11s} {json.dumps(kwargs):>16s} "
              f"{tok / cnt:10.0f} {rec / cnt:8.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemini-3.5-flash-lite")
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--facts", type=int, default=300)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--homogeneous", action="store_true",
                   help="every agent needs exactly 2 facts (the case where "
                        "a tuned global k is already near-optimal)")
    p.add_argument("--dry-run", action="store_true",
                   help="report allocation statistics without calling a "
                        "model -- validates policy logic for free")
    args = p.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")

    if args.pilot:
        configs = [("broadcast", {}), ("topk", {"k": 3}),
                   ("parcel", {"lam": 0.02})]
        trials = 2
    else:
        configs = (
            [("broadcast", {}), ("none", {})]
            + [("random", {"k": k}) for k in (5, 25, 100)]
            + [("centrality", {"k": k}) for k in (5, 25, 100)]
            + [("topk", {"k": k}) for k in (2, 5, 10, 25, 60, 150)]
            + [("parcel", {"lam": l}) for l in
               (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01)]
        )
        trials = args.trials

    out = pathlib.Path(args.out) if args.out else RESULTS / "allocation.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    total = trials * len(configs) * args.agents
    print(f"[{args.model}] {trials} trials x {len(configs)} configs x "
          f"{args.agents} agents = {total} calls")

    with open(out, "a") as fh:
        for t in range(trials):
            recs = run_trial(args.model, key, 1000 + t, args.agents,
                             args.facts, configs, args.workers,
                             not args.homogeneous)
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"  trial {t + 1}/{trials}", file=sys.stderr, flush=True)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
