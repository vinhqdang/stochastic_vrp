"""Dependent multi-agent pipeline on MuSiQue: the conversation history accumulates.

WHY THIS EXPERIMENT
-------------------
allocation_experiment.py and musique_benchmark.py run independent
single-shot agents: each receiver answers once from its own context, and
nothing it produces reaches another receiver. A real multi-agent pipeline
is different. Agent j consumes the outputs of agents k < j, so the context
it must absorb grows with every hop, and the forwarding policy decides
what that growing history costs and whether the final answer survives it.

DESIGN
------
Each MuSiQue instance is a chain of sub-questions in executable order.
Agent j's question is sub-question j with every "#k" replaced by agent k's
extracted answer. That substitution is the dependency: agent j cannot run
before agent k, and a wrong upstream answer propagates downstream.

The six policies differ only in what fills the CONTEXT block (documents)
and the MESSAGES block (what upstream agents said):

  full_history       all documents; the complete transcript of every earlier
                     agent, verbatim. The common "forward everything"
                     practice; its prompt grows with every hop.
  all_docs_answers   all documents; only the answers of the sub-questions
                     this agent references by "#k".
  per_receiver_k3    top-3 documents by lexical overlap with THIS agent's
                     resolved question, so retrieval depends on upstream
                     output; messages as in all_docs_answers.
  per_receiver_k5    as above with top-5.
  seed_set_k3        one global top-3 set for all agents of an instance,
                     ranked from the sub-questions (with "#k" removed) and
                     the full question, so independent of upstream output;
                     messages as in all_docs_answers.
  oracle             the single gold paragraph for this sub-question;
                     messages as in all_docs_answers.

Token counts are the REAL usageMetadata values returned by the API.

USAGE
  python3 pipeline_agents.py --per-hop 2 --out results/pipeline_smoke.jsonl
  python3 pipeline_agents.py                  # full 120-instance run
  python3 pipeline_agents.py --analyze        # tables + summary JSON
"""

import argparse
import json
import os
import pathlib
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from musique_benchmark import _lexical, em, extract  # noqa: E402

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent?key={key}")
HOP_SIZES = (2, 3, 4)
POLICIES = ["full_history", "all_docs_answers", "per_receiver_k3",
            "per_receiver_k5", "seed_set_k3", "oracle", "per_receiver_ce_k2"]
PAIRS = ([(p, "full_history") for p in POLICIES if p != "full_history"]
         + [("per_receiver_ce_k2", "oracle"),
            ("per_receiver_ce_k2", "per_receiver_k3"),
            ("per_receiver_k3", "seed_set_k3"),
            ("per_receiver_k3", "all_docs_answers")])
REF = re.compile(r"#(\d+)")
MAX_RETRIES = 8
MAX_OUTPUT_TOKENS = 800
QUOTA_STOP_STREAK = 3          # consecutive exhausted-429 jobs before stopping
B_RESAMPLES = 10000
PROMPT_TAIL = ("Answer from the documents and messages above. Think briefly "
               "(at most three sentences), then end your reply with the line: "
               "ANSWER: <answer>. If the answer is not available, reply "
               "ANSWER: UNKNOWN.")


# ------------------------------------------------------------------ data

def referenced(q):
    """1-indexed sub-question numbers named by '#k' in q, sorted, unique."""
    return sorted({int(k) for k in REF.findall(q)})


def to_instance(r, k):
    for j, d in enumerate(r["question_decomposition"]):
        for ref in referenced(d["question"]):
            if not 1 <= ref <= j:       # sub-question j+1 may use only #1..#j
                raise ValueError(f"{r['id']}: sub-question {j + 1} refers "
                                 f"to #{ref}, which has not run yet")
    return {
        "id": r["id"], "family": r["id"].split("__")[0], "hops": k,
        "question": r["question"], "answer": r["answer"],
        "aliases": list(r["answer_aliases"] or []),
        "answerable": r.get("answerable"),
        "paragraphs": [{"idx": p["idx"], "title": p["title"],
                        "text": p["paragraph_text"]}
                       for p in r["paragraphs"]],
        "subs": [{"q": d["question"], "gold_idx": d["paragraph_support_idx"]}
                 for d in r["question_decomposition"]],
    }


def sample_instances(per_hop, seed=0):
    from datasets import load_dataset
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    out = []
    for k in HOP_SIZES:
        picked = [r for r in rows
                  if len(r["question_decomposition"]) == k][:per_hop]
        if len(picked) < per_hop:
            sys.exit(f"only {len(picked)} rows with {k} sub-questions")
        out.extend(to_instance(r, k) for r in picked)
    return out


# ----------------------------------------------------- resolution and context

def resolve(q, answers):
    def sub(m):
        k = int(m.group(1))
        if not 1 <= k <= len(answers):
            raise ValueError(f"'#{k}' in {q!r} refers to an agent that has not run")
        return answers[k - 1]
    return REF.sub(sub, q)


def seed_set(inst, k=3):
    """Global top-k set for an instance. Uses no upstream output."""
    sub_qs = [REF.sub("", s["q"]) for s in inst["subs"]]

    def score(p):
        return (sum(_lexical(p, {"q": sq}) for sq in sub_qs)
                + _lexical(p, {"q": inst["question"]}))
    ranked = sorted(inst["paragraphs"], key=score, reverse=True)
    return {p["idx"] for p in ranked[:k]}


_CE = {"model": None, "lock": threading.Lock()}


def ce_rank(paras, q):
    """Rank paragraphs by the ms-marco cross-encoder on the RESOLVED
    question (added for the rebuttal: does better relevance estimation
    close the gap to the oracle when agents depend on each other?)."""
    with _CE["lock"]:
        if _CE["model"] is None:
            from sentence_transformers import CrossEncoder
            _CE["model"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        sc = _CE["model"].predict([(q, p["title"] + ". " + p["text"])
                                   for p in paras], show_progress_bar=False)
    return [p for _, p in sorted(zip(sc, paras), key=lambda x: -x[0])]


def context_for(policy, inst, j, q, gold_idx, seed):
    paras = inst["paragraphs"]
    if policy in ("full_history", "all_docs_answers"):
        chosen = paras
    elif policy == "oracle":
        chosen = [p for p in paras if p["idx"] == gold_idx]
    elif policy.startswith("per_receiver_ce_k"):
        k = int(policy.rsplit("k", 1)[1])
        chosen = ce_rank(paras, q)[:k]
    elif policy.startswith("per_receiver_k"):
        k = int(policy.rsplit("k", 1)[1])
        chosen = sorted(paras, key=lambda p: _lexical(p, {"q": q}),
                        reverse=True)[:k]
    elif policy == "seed_set_k3":
        chosen = [p for p in paras if p["idx"] in seed]
    else:
        raise ValueError(policy)
    chosen = list(chosen)
    random.Random(j).shuffle(chosen)
    return chosen


def messages_for(policy, j, transcript, answers, raw_q):
    if policy == "full_history":
        return "\n\n".join(
            f"Agent {i + 1} was asked: {t['q']}\nAgent {i + 1} replied: {t['reply']}"
            for i, t in enumerate(transcript[:j]))
    return "\n".join(f"Answer to sub-question {k}: {answers[k - 1]}"
                     for k in referenced(raw_q))


def build_prompt(paras, msgs, q):
    ctx = ("\n\n".join(f"[{p['title']}] {p['text']}" for p in paras)
           or "(no documents provided)")
    block = f"Messages from other agents:\n{msgs}\n\n" if msgs else ""
    return f"{ctx}\n\n{block}Question: {q}\n\n{PROMPT_TAIL}"


# ------------------------------------------------------------------ model

class CallFailed(Exception):
    pass


def call_openrouter(model, prompt, timeout=180):
    """Same call through OpenRouter (model 'vendor/name'), same settings.

    Used to continue a run once the Gemini free tier's 500 requests/day
    per model is spent. Token counts are the provider-reported usage.
    """
    key = os.environ["OPENROUTER_API_KEY"]
    payload = json.dumps({
        "model": model, "temperature": 0, "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "usage": {"include": True},
    }).encode()
    attempt = 0
    while True:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            if "choices" not in body:
                raise ValueError("no choices")
            break
        except urllib.error.HTTPError as e:
            transient = e.code == 429 or e.code >= 500
            err = f"HTTP{e.code}"
        except (urllib.error.URLError, socket.timeout, TimeoutError,
                ConnectionError, ValueError) as e:
            transient = True
            err = type(e).__name__
        if not transient or attempt >= MAX_RETRIES:
            raise CallFailed(err)
        time.sleep(min(2 ** (attempt + 1), 60) * (1 + random.random() * 0.25))
        attempt += 1
    try:
        text = body["choices"][0]["message"].get("content") or ""
        u = body["usage"]
        pt, ot = int(u["prompt_tokens"]), int(u["completion_tokens"])
        th = int((u.get("completion_tokens_details") or {})
                 .get("reasoning_tokens") or 0)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise CallFailed(f"parse:{type(e).__name__}") from None
    if not text.strip():
        raise CallFailed("empty")
    return text, pt, ot, th


def call_gemini(model, prompt, key, timeout=180):
    """One generateContent call with exponential backoff on 429/5xx.

    Returns (text, prompt_tokens, output_tokens, thought_tokens) from the
    response's usageMetadata. Retries 429, 5xx and transient network errors
    up to MAX_RETRIES times; anything else, or exhaustion, raises CallFailed
    carrying only the HTTP code or exception name (never the URL, which
    holds the key).
    """
    if "/" in model:
        return call_openrouter(model, prompt, timeout)
    url = GEMINI_URL.format(model=model, key=key)
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0,
                             "maxOutputTokens": MAX_OUTPUT_TOKENS},
    }).encode()
    attempt = 0
    while True:
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            transient = e.code == 429 or e.code >= 500
            err = f"HTTP{e.code}"
        except (urllib.error.URLError, socket.timeout, TimeoutError,
                ConnectionError) as e:
            transient = True
            err = type(e).__name__
        except ValueError:
            raise CallFailed("badjson") from None
        if not transient or attempt >= MAX_RETRIES:
            raise CallFailed(err)
        time.sleep(min(2 ** (attempt + 1), 60) * (1 + random.random() * 0.25))
        attempt += 1
    try:
        cand = body["candidates"][0]
        text = "".join(p.get("text", "")
                       for p in cand.get("content", {}).get("parts", []))
        um = body["usageMetadata"]
        pt = int(um["promptTokenCount"])
        ot = int(um["candidatesTokenCount"])
        th = int(um.get("thoughtsTokenCount") or 0)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise CallFailed(f"parse:{type(e).__name__}") from None
    if not text.strip():
        raise CallFailed(f"empty:{cand.get('finishReason')}")
    return text, pt, ot, th


# ------------------------------------------------------------------ jobs

def run_job(inst, policy, model, key, dump=None):
    """One (instance, policy) job: its agents run sequentially in order."""
    seed = seed_set(inst) if policy == "seed_set_k3" else None
    answers, transcript, agents = [], [], []
    tot_p = tot_o = 0
    err = None
    for j, s in enumerate(inst["subs"]):
        q = resolve(s["q"], answers)
        paras = context_for(policy, inst, j, q, s["gold_idx"], seed)
        msgs = messages_for(policy, j, transcript, answers, s["q"])
        prompt = build_prompt(paras, msgs, q)
        try:
            text, pt, ot, th = call_gemini(model, prompt, key)
        except CallFailed as e:
            err = str(e)
            break
        pred = extract(text) or ""
        answers.append(pred)
        transcript.append({"q": q, "reply": text})
        agents.append({
            "q_resolved": q, "pred": pred,
            "prompt_tokens": pt, "output_tokens": ot, "thought_tokens": th,
            "had_gold_paragraph": any(p["idx"] == s["gold_idx"] for p in paras),
            "n_paragraphs": len(paras),
        })
        tot_p += pt
        tot_o += ot
        if dump:
            dump({"id": inst["id"], "policy": policy, "agent": j + 1,
                  "q_resolved": q, "prompt": prompt, "reply": text})
    final = answers[-1] if (err is None and answers) else None
    correct = None
    if final is not None:
        golds = [inst["answer"], *inst["aliases"]]
        correct = any(em(final, g) for g in golds)
    return {"id": inst["id"], "family": inst["family"], "hops": inst["hops"],
            "policy": policy, "final_pred": final, "gold": inst["answer"],
            "correct": correct, "agents": agents,
            "total_prompt_tokens": tot_p, "total_output_tokens": tot_o,
            "error": err}


def load_done(path):
    done = set()
    if path.exists():
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    done.add((r["id"], r["policy"]))
    return done


def run(args, out):
    key = os.environ.get("GEMINI_API_KEY")
    if not key and "/" not in args.model:
        sys.exit("set GEMINI_API_KEY")
    insts = sample_instances(args.per_hop)
    print(f"sampled {len(insts)} instances: "
          + ", ".join(f"{h}-subq={sum(1 for i in insts if i['hops'] == h)}"
                      for h in HOP_SIZES)
          + f"; answerable=False: "
          f"{sum(1 for i in insts if i['answerable'] is False)}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    jobs = [(inst, pol) for inst in insts for pol in POLICIES
            if (inst["id"], pol) not in done]
    print(f"{len(done)} jobs already recorded; {len(jobs)} to run -> {out}",
          flush=True)

    lock = threading.Lock()
    stop = threading.Event()
    dump_fh = open(args.dump, "a") if args.dump else None

    def dump(rec):
        with lock:
            dump_fh.write(json.dumps(rec) + "\n")
            dump_fh.flush()

    def work(inst, pol):
        if stop.is_set():
            return None          # left unrecorded, so a rerun picks it up
        return run_job(inst, pol, args.model, key,
                       dump if dump_fh else None)

    streak = n_ok = n_err = 0
    with open(out, "a") as fh, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, inst, pol) for inst, pol in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            if rec is None:
                continue
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if rec["error"] is None:
                    n_ok += 1
                    streak = 0
                else:
                    n_err += 1
                    streak = streak + 1 if rec["error"] == "HTTP429" else 0
                    if streak >= QUOTA_STOP_STREAK and not stop.is_set():
                        stop.set()
                        print(f"\nSTOP: {streak} consecutive jobs exhausted "
                              f"429 retries (quota likely spent). Rerun to "
                              f"resume; errored jobs are retried.",
                              file=sys.stderr, flush=True)
            tag = (f"correct={rec['correct']}" if rec["error"] is None
                   else f"ERROR {rec['error']}")
            print(f"[{n}/{len(futs)}] {rec['id']} {rec['policy']}: {tag} "
                  f"prompt_tok={rec['total_prompt_tokens']}", flush=True)
    if dump_fh:
        dump_fh.close()
    print(f"done: {n_ok} ok, {n_err} errors this run", flush=True)


# -------------------------------------------------------------- analysis

def latest_records(path):
    """Last record per (id, policy); a rerun of an errored job supersedes it."""
    last = {}
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            last[(r["id"], r["policy"])] = r
    return last


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(d, seed=0, B=B_RESAMPLES):
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(B))
    return means[round(0.025 * (B - 1))], means[round(0.975 * (B - 1))]


def sign_flip_p(d, seed=0, B=B_RESAMPLES):
    """Two-sided paired sign-flip permutation p, add-one corrected."""
    obs = abs(sum(d))
    if obs == 0:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(B):
        s = sum(x if rng.random() < 0.5 else -x for x in d)
        if abs(s) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (B + 1)


def analyze(out, model, per_hop):
    last = latest_records(out)
    ok = {k: r for k, r in last.items() if r["error"] is None}
    by = {pol: {i: r for (i, p), r in ok.items() if p == pol}
          for pol in POLICIES}
    inst_hops = {r["id"]: r["hops"] for (i, p), r in last.items()}
    print(f"{out.name}: {len({i for i, _ in last})} instances, "
          f"{len(last)} (instance, policy) jobs, {len(ok)} error-free, "
          f"{len(last) - len(ok)} errored")
    print("instances by hops: " + ", ".join(
        f"{h}: {sum(1 for v in inst_hops.values() if v == h)}"
        for h in HOP_SIZES) + "\n")
    summary = {"model": model, "per_hop": per_hop,
               "n_instances": len(inst_hops), "policies": {}, "by_hops": {},
               "per_agent_position": {}, "paired": [],
               "bootstrap": {"B": B_RESAMPLES, "seed": 0},
               "permutation": {"B": B_RESAMPLES, "seed": 0,
                               "p_value": "two-sided, add-one corrected"}}

    # (a) per policy
    full_tok = mean([r["total_prompt_tokens"]
                     for r in by["full_history"].values()])
    print("(a) per policy")
    print(f"{'policy':18s} {'n':>4s} {'err':>4s} {'acc':>7s} "
          f"{'prompt_tok/inst':>16s} {'vs_full':>8s}")
    for pol in POLICIES:
        rs = list(by[pol].values())
        n_err = sum(1 for (i, p), r in last.items()
                    if p == pol and r["error"] is not None)
        acc = mean([float(bool(r["correct"])) for r in rs])
        tok = mean([r["total_prompt_tokens"] for r in rs])
        summary["policies"][pol] = {"n": len(rs), "errors": n_err,
                                    "accuracy": acc,
                                    "mean_prompt_tokens": tok,
                                    "ratio_vs_full_history": tok / full_tok}
        print(f"{pol:18s} {len(rs):4d} {n_err:4d} {acc:7.3f} {tok:16.0f} "
              f"{tok / full_tok:8.3f}")

    # (b) by hop count
    print("\n(b) mean total prompt tokens per instance, by hop count (n)")
    print(f"{'policy':18s} " + " ".join(f"{str(h) + '-hop':>16s}"
                                       for h in HOP_SIZES))
    for pol in POLICIES:
        cells, rec = [], {}
        for h in HOP_SIZES:
            rs = [r for r in by[pol].values() if r["hops"] == h]
            m = mean([r["total_prompt_tokens"] for r in rs])
            rec[str(h)] = {"n": len(rs), "mean_prompt_tokens": m}
            cells.append(f"{m:10.0f} ({len(rs):3d})")
        summary["by_hops"][pol] = rec
        print(f"{pol:18s} " + " ".join(f"{c:>16s}" for c in cells))

    # (c) per-agent position
    print("\n(c) mean prompt tokens per agent, by position (n)")
    for pol in ("full_history", "per_receiver_k3"):
        cells, rec = [], {}
        for pos in range(1, 5):
            xs = [r["agents"][pos - 1]["prompt_tokens"]
                  for r in by[pol].values() if len(r["agents"]) >= pos]
            m = mean(xs)
            rec[str(pos)] = {"n": len(xs), "mean_prompt_tokens": m}
            cells.append(f"{pos}{['st', 'nd', 'rd', 'th'][pos - 1]}: "
                         f"{m:7.0f} ({len(xs):3d})")
        summary["per_agent_position"][pol] = rec
        print(f"{pol:18s} " + "  ".join(cells))

    # (d) paired comparisons
    print("\n(d) paired comparisons on instances error-free under both "
          "(acc diff = A - B)")
    print(f"{'A':18s} {'B':18s} {'pairs':>5s} {'acc_A':>6s} {'acc_B':>6s} "
          f"{'diff':>7s} {'95% CI':>18s} {'p':>7s} {'A>B':>4s} {'B>A':>4s}")
    for A, B in PAIRS:
        common = sorted(i for i in by[A] if i in by[B])
        if not common:
            continue
        ca = [float(bool(by[A][i]["correct"])) for i in common]
        cb = [float(bool(by[B][i]["correct"])) for i in common]
        d = [a - b for a, b in zip(ca, cb)]
        diff = mean(d)
        lo, hi = bootstrap_ci(d)
        p = sign_flip_p(d)
        a_gt = sum(1 for x in d if x > 0)
        b_gt = sum(1 for x in d if x < 0)
        summary["paired"].append({
            "A": A, "B": B, "n_pairs": len(common), "acc_A": mean(ca),
            "acc_B": mean(cb), "diff": diff, "ci95": [lo, hi], "p": p,
            "A_only_correct": a_gt, "B_only_correct": b_gt})
        print(f"{A:18s} {B:18s} {len(common):5d} {mean(ca):6.3f} "
              f"{mean(cb):6.3f} {diff:+7.3f} "
              f"[{lo:+.3f},{hi:+.3f}] {p:7.4f} {a_gt:4d} {b_gt:4d}")

    summary_path = RESULTS / f"summary_{out.stem}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsummary -> {summary_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--per-hop", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None,
                    help="append every prompt and reply to this JSONL "
                         "(inspection only)")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    out = (pathlib.Path(args.out) if args.out
           else RESULTS / f"pipeline_{args.model.replace('/', '-')}.jsonl")
    if args.analyze:
        analyze(out, args.model, args.per_hop)
    else:
        run(args, out)


if __name__ == "__main__":
    main()
