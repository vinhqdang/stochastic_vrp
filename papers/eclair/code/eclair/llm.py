"""LLM candidate generation (PROJECT.md §3 Step 1) via OpenRouter.

Diversified generation: candidates are sampled over (model family x
prompt style x temperature). Each candidate is a Python function
`build_model(params) -> cpmpy.Model` generated from a natural-language
spec; the params schema matches the harness's instance generators, so
every probe tier runs on LLM candidates unchanged.

The API key is read from $OPENROUTER_API_KEY or the untracked file
.openrouter_key next to this package (gitignored — never commit it).
Responses are cached in results/llm_cache.json keyed by a hash of
(model, prompt, temperature), so reruns cost no API calls.

Ground-truth proxy label (scoring only, the paper's point is that no
oracle exists in production): a candidate is labelled faithful iff its
optimum matches the spec's brute-forced optimum on LABEL_TRIES micro
instances without error.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import pathlib
import time
import urllib.request

import cpmpy

from .probes import brute_optimum, solve_value

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = ["openrouter/free", "nvidia/nemotron-3-ultra-550b-a55b:free"]
# used automatically when the free models hit their daily usage limits:
FALLBACK_MODEL = "qwen/qwen3.7-flash"
LABEL_TRIES = 12
CACHE_PATH = pathlib.Path(__file__).parent.parent / "results" / "llm_cache.json"


def _api_key():
    import os
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    f = pathlib.Path(__file__).parent.parent / ".openrouter_key"
    if f.exists():
        return f.read_text().strip()
    raise RuntimeError("no OpenRouter key: set OPENROUTER_API_KEY or "
                       "create code/.openrouter_key")


NL_SPECS = {
    "knapsack_conflicts": """\
You must select a subset of n items. Item i has weight w[i] and value
v[i]. The total weight of selected items must not exceed the capacity
cap. Some pairs of items conflict: for each pair (i, j) in the list
`conflicts`, items i and j must not both be selected. Maximize the
total value of the selected items.

params dict schema: {"n": int, "w": list[int] of length n,
"v": list[int] of length n, "cap": int,
"conflicts": list of (i, j) index pairs}.""",
    "set_cover": """\
There are `nelem` elements, numbered 0..nelem-1, and a collection of
sets. Set s covers the elements listed in sets[s] and costs cost[s].
Choose a sub-collection of sets so that every element is covered by at
least one chosen set. Minimize the total cost of the chosen sets.

params dict schema: {"nelem": int, "sets": list of lists of element
ids, "cost": list[int], one cost per set}.""",
    "coloring": """\
Color the n vertices of an undirected graph, using color indices from
0 to n-1, so that the two endpoints of every edge receive different
colors. Minimize the number of colors used. IMPORTANT convention: the
objective value must equal 1 + the maximum color index used (so a
solution using colors {0,1,2} has objective 3).

params dict schema: {"n": int, "edges": list of (u, v) vertex pairs}.""",
    "gen_assignment": """\
Assign each of T tasks to exactly one of M machines. Task t has size
p[t]; machine m has capacity cap[m]: the total size of tasks assigned
to machine m must not exceed cap[m]. Assigning task t to machine m
costs cost[t][m]. Minimize the total assignment cost.

params dict schema: {"T": int, "M": int, "p": list[int] length T,
"cap": list[int] length M, "cost": list of T rows, each of length M}.""",
    "ontime_selection": """\
A single machine processes jobs one at a time. Job j has processing
time p[j], deadline d[j] and weight wt[j]. Select a subset of jobs
that can ALL be completed by their deadlines when the selected jobs
are processed in non-decreasing deadline order (equivalently: a
selected set is feasible if and only if, for every selected job j, the
sum of processing times of selected jobs i with d[i] <= d[j] is at
most d[j]). Maximize the total weight of the selected jobs.

params dict schema: {"n": int, "p": list[int], "d": list[int],
"wt": list[int]}.""",
}

PROMPT_STYLES = [
    ("direct", "Write the model directly."),
    ("stepwise", "First restate the decision variables, constraints and "
     "objective in one short paragraph each, then write the code."),
    ("expert", "You are an expert constraint-programming modeller known "
     "for scrupulously faithful formalizations. Write the model."),
]


def _prompt(spec_name, style_hint, spec_text=None):
    return f"""Translate the following optimization problem into a CPMpy model.

{spec_text if spec_text is not None else NL_SPECS[spec_name]}

{style_hint}

Requirements:
- Provide a single Python function `build_model(params)` that returns a
  `cpmpy.Model` with the objective set (use model.maximize(...) or
  model.minimize(...)).
- Use only the `cpmpy` library (imported as `cp`) and the Python
  standard library. `import cpmpy as cp` is already available; you may
  repeat imports.
- The function must work for ANY params dict matching the schema.
- Reply with ONLY a Python code block."""


def _call(model, prompt, temperature, max_retries=5):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4000,
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    })
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            if not content:                 # some models return None
                raise RuntimeError("empty completion content")
            return content
        except Exception as e:                      # 429s, transient 5xx
            if attempt == max_retries - 1:
                raise
            time.sleep(8 * (attempt + 1))
    raise RuntimeError("unreachable")


def _cached_call(model, prompt, temperature):
    key = hashlib.sha256(
        f"{model}|{temperature}|{prompt}".encode()).hexdigest()[:24]
    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
    if key not in cache or not cache[key].get("response"):
        cache[key] = {"model": model, "temperature": temperature,
                      "response": _call(model, prompt, temperature)}
        CACHE_PATH.parent.mkdir(exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=1))
    return cache[key]["response"]


def _extract_code(text):
    if not text:
        return None
    if "```" not in text:
        return text if "def build_model" in text else None
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        block = parts[i]
        if block.startswith(("python", "py")):
            block = block.split("\n", 1)[1] if "\n" in block else ""
        if "def build_model" in block:
            return block
    return None


class LLMCandidate:
    """Wraps LLM-generated build_model(); same interface as
    mutations.Candidate so every probe runs unchanged."""

    _ids = [10000]

    def __init__(self, spec, fn, meta):
        self.spec = spec
        self.fn = fn
        self.meta = meta                 # model / style / temperature
        self.descriptor = ("llm", meta["model"], meta["style"])
        self._ids[0] += 1
        self.cid = self._ids[0]

    def build(self, params):
        return self.fn(copy.deepcopy(params))


def make_llm_candidate(spec, model, style, temperature, spec_text=None):
    """Generate one candidate; returns (candidate|None, status).
    `spec_text` overrides the default NL specification (ambiguity
    studies)."""
    style_name, hint = style
    meta = {"model": model, "style": style_name, "temperature": temperature}
    prompt = _prompt(spec.name, hint, spec_text)
    try:
        text = _cached_call(model, prompt, temperature)
    except Exception as e:
        try:                    # free-tier quota exhausted -> fallback model
            text = _cached_call(FALLBACK_MODEL, prompt, temperature)
            meta["model"] = f"{FALLBACK_MODEL} (fallback)"
        except Exception:
            return None, f"api_error: {e}"
    code = _extract_code(text)
    if code is None:
        return None, "no_code"
    ns = {"cp": cpmpy, "cpmpy": cpmpy, "math": math, "itertools": itertools}
    try:
        exec(code, ns)                  # user-sanctioned research pipeline
        fn = ns["build_model"]
    except Exception as e:
        return None, f"exec_error: {type(e).__name__}: {e}"
    cand = LLMCandidate(spec, fn, meta)
    import random as _r
    gate_rng = _r.Random(0)
    try:                                # intake gate: must build+solve
        for _ in range(2):
            solve_value(cand.build(spec.gen(gate_rng, "micro")))
    except Exception as e:
        return None, f"runtime_error: {type(e).__name__}: {e}"
    return cand, "ok"


def proxy_label(cand, rng, tries=LABEL_TRIES):
    """Scoring-only faithfulness label vs the brute-forced spec."""
    try:
        for _ in range(tries):
            params = cand.spec.gen(rng, "micro")
            if solve_value(cand.build(params)) != brute_optimum(cand.spec,
                                                                params):
                return False
    except Exception:
        return False
    return True
