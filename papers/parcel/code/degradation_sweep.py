"""Dense sweep of the low-load region, with a confusability arm.

WHY THIS EXPERIMENT EXISTS
--------------------------
PROJECT.md §11 leaves two modelling questions that no published curve
answers, and both are load-bearing for the theorem:

  1. WHERE IS c*?  The saturation penalty rho is S-shaped -- convex
     below a model's "effective length", concave above. The admission
     price result (§9.4) holds only on the convex region, so c* defines
     where the guarantee applies. Published sweeps are log-spaced
     (NoLiMa: 1K/2K/4K/8K) and give only ~2 points inside that region.
     Nobody samples it densely.

  2. IS THE PENALTY ON TOKENS OR ON CONFUSABILITY?  Five length-matched
     studies say raw token count barely predicts degradation once length
     is controlled, and semantic proximity predicts it strongly. If that
     holds, rho takes confusable load, not c(S) -- which is what §4 now
     assumes.

DESIGN
------
Task: a synthetic k-hop indirection chain. Each instance defines a chain
of code words ending in a numeric value; the question asks the model to
follow the chain. Multi-hop is used deliberately -- the degradation
literature is consistent that multi-hop degrades far faster than
single-span retrieval, so it is where the effect is measurable.

The task is SYNTHETIC and generated fresh per run, on purpose. Public
QA benchmarks (HotpotQA, SQuAD, NIAH) are contaminated in current
models, and a contaminated baseline would confound the very curve we
are trying to measure.

Two distractor arms at matched token budgets:

  confusable -- sentences of IDENTICAL form about unrelated entities
                ("The code for DELTA is ECHO."). Maximum lexical and
                structural proximity to the signal.
  random     -- topically unrelated filler prose of the same length.

Matched length, different confusability, is exactly the contrast that
separates the two candidate arguments for rho. If accuracy tracks token
count, both arms fall together. If it tracks confusability, they
separate -- and §4's model is right.

Signal sentences are placed at random positions and the whole context is
shuffled, so needle-position effects average out rather than confound.

USAGE
  python3 degradation_sweep.py --pilot            # tiny smoke test
  python3 degradation_sweep.py --model X --n 30   # one full sweep
  python3 analyze_sweep.py results/<file>.jsonl   # shape analysis

Results are appended as JSONL, one record per model call, so a run can
be interrupted and resumed without losing work.
"""

import argparse
import json
import os
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://openrouter.ai/api/v1/chat/completions"
RESULTS = pathlib.Path(__file__).parent / "results"

# Rough words -> tokens conversion. Recorded alongside raw word counts so
# a reader can re-derive with a real tokenizer; relative spacing is what
# the shape analysis needs, and that is preserved under any linear map.
TOKENS_PER_WORD = 1.33

CODEWORDS = """ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA
JULIET KILO LIMA MIKE NOVEMBER OSCAR PAPA QUEBEC ROMEO SIERRA TANGO
UNIFORM VICTOR WHISKEY XRAY YANKEE ZULU ANVIL BEACON CANYON DRIFTER
EMBER FURNACE GRANITE HARBOR IVORY JASPER KESTREL LANTERN MARBLE NOMAD
OBSIDIAN PRISM QUARRY RIDGE SUMMIT THICKET UMBRA VERTEX WILLOW ZENITH""".split()

# Topically unrelated filler. Deliberately mundane and non-numeric so it
# cannot be mistaken for signal.
FILLER = """Sedimentary rock forms when mineral particles accumulate and
are compacted over long periods. The process begins with weathering,
which breaks larger formations into smaller fragments. Rivers and wind
transport this material across considerable distances before deposition
occurs. Layers build gradually, and the weight of overlying material
compresses those beneath it. Cementation follows as dissolved minerals
crystallise in the spaces between grains. Geologists read these layers
much as one reads pages, since each records the conditions under which
it formed. Grain size indicates the energy of the transporting medium.
Colour often reflects the presence of iron compounds. Fossils preserved
within the layers help establish relative ages across separated regions.
The study of these sequences underpins much of our understanding of past
climates and the slow rearrangement of continents over geological
time.""".split()


def make_instance(k_hops, rng):
    """Build a k-hop chain plus its distractor-proof answer."""
    chain = rng.sample(CODEWORDS, k_hops + 1)
    value = rng.randint(100, 999)
    sentences = [f"The code for {chain[i]} is {chain[i + 1]}."
                 for i in range(k_hops)]
    sentences.append(f"The value of {chain[-1]} is {value}.")
    question = (f"Starting from {chain[0]}, follow each code to the next "
                f"until you reach a word with a value. What is that value?")
    return sentences, question, value, set(chain)


def confusable_distractors(n_words, used, rng):
    """Sentences of identical form about entities not in the chain."""
    pool = [w for w in CODEWORDS if w not in used]
    out, count = [], 0
    while count < n_words:
        a, b = rng.sample(pool, 2)
        if rng.random() < 0.25:
            s = f"The value of {a} is {rng.randint(100, 999)}."
        else:
            s = f"The code for {a} is {b}."
        out.append(s)
        count += len(s.split())
    return out


def random_distractors(n_words, rng):
    """Topically unrelated filler at a matched word count."""
    out, count = [], 0
    while count < n_words:
        start = rng.randrange(0, max(1, len(FILLER) - 30))
        span = FILLER[start:start + rng.randint(12, 28)]
        s = " ".join(span).rstrip(".,") + "."
        out.append(s)
        count += len(span)
    return out


def build_prompt(k_hops, arm, target_words, rng):
    signal, question, answer, used = make_instance(k_hops, rng)
    if target_words <= 0:
        body = list(signal)
    else:
        distractors = (confusable_distractors(target_words, used, rng)
                       if arm == "confusable"
                       else random_distractors(target_words, rng))
        body = distractors + signal
        rng.shuffle(body)          # signal position randomised, not fixed
    text = " ".join(body)
    # An explicit answer marker keeps grading uniform across terse models
    # and reasoning models that emit a visible chain of thought.
    prompt = (f"{text}\n\n{question}\n\n"
              f"End your reply with the line: ANSWER: <number>")
    return prompt, answer, len(text.split())


def call_model(model, prompt, key, timeout=180, retries=5, max_tokens=400):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL, data=payload,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            content = body["choices"][0]["message"].get("content")
            if not content:
                raise KeyError("empty content")
            return content
        except urllib.error.HTTPError as exc:
            # 429 is routine on the free tier; back off hard and retry.
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1) + random.random() * 3)
                continue
            if attempt == retries - 1:
                return f"__ERROR__ HTTP{exc.code}"
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, KeyError, TimeoutError,
                json.JSONDecodeError, TypeError) as exc:
            if attempt == retries - 1:
                return f"__ERROR__ {type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return "__ERROR__ unreachable"


def graded(response, answer):
    """True iff the reply's stated answer is the chain's value.

    Prefers the explicit ANSWER: marker; falls back to the LAST integer,
    which is the right default for a model that reasons aloud before
    committing. Returns None for a failed call so errors stay separable
    from wrong answers in the analysis.
    """
    if response.startswith("__ERROR__"):
        return None
    clean = response.replace(",", "")
    m = re.search(r"ANSWER:\s*(-?\d+)", clean, re.IGNORECASE)
    if not m:
        nums = re.findall(r"-?\d+", clean)
        if not nums:
            return False
        return int(nums[-1]) == answer
    return int(m.group(1)) == answer


def run(model, levels, arms, n, k_hops, out_path, workers, seed):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("set OPENROUTER_API_KEY")

    jobs = []
    for arm in arms:
        for words in levels:
            for rep in range(n):
                # Per-cell seed: the SAME chain instances are reused across
                # arms at each level, so the arms differ only in distractor
                # kind. Without this the comparison would be confounded by
                # instance difficulty.
                jobs.append((arm, words, rep,
                             random.Random(f"{seed}|{words}|{rep}")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    total = len(jobs)

    def one(job):
        arm, words, rep, rng = job
        prompt, answer, actual = build_prompt(k_hops, arm, words, rng)
        resp = call_model(model, prompt, key)
        return {
            "model": model, "arm": arm, "target_words": words,
            "actual_words": actual,
            "est_tokens": round(actual * TOKENS_PER_WORD),
            "k_hops": k_hops, "rep": rep,
            "answer": answer, "response": resp.strip()[:64],
            "correct": graded(resp, answer),
        }

    with open(out_path, "a") as fh, ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(one, jobs):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  {done}/{total}", file=sys.stderr, flush=True)
    print(f"wrote {done} records to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-4-31b-it:free")
    p.add_argument("--n", type=int, default=30,
                   help="repetitions per (arm, level) cell")
    p.add_argument("--k-hops", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", default="parcel")
    p.add_argument("--out", default=None)
    p.add_argument("--pilot", action="store_true",
                   help="tiny smoke test: 3 levels, n=4")
    args = p.parse_args()

    if args.pilot:
        levels, arms, n = [0, 1500, 6000], ["confusable", "random"], 4
    else:
        # Dense where it matters. Published sweeps jump 1K->2K->4K->8K;
        # the inflection reportedly sits inside that gap, so sample it
        # finely and let the tail stay coarse.
        levels = [0, 150, 300, 500, 750, 1000, 1400, 1900, 2500,
                  3200, 4000, 5000, 6000, 8000]
        arms, n = ["confusable", "random"], args.n

    tag = args.model.split("/")[-1].replace(":", "-")
    out = pathlib.Path(args.out) if args.out else RESULTS / f"sweep_{tag}.jsonl"
    print(f"model={args.model} levels={len(levels)} arms={len(arms)} "
          f"n={n} -> {len(levels) * len(arms) * n} calls")
    run(args.model, levels, arms, n, args.k_hops, out, args.workers, args.seed)


if __name__ == "__main__":
    main()
