"""Stronger relevance scorers for the MuSiQue benchmark, cached on disk.

Three scorers replace the bag-of-words `_lexical` for the rebuttal:

  ce    cross-encoder/ms-marco-MiniLM-L-6-v2, raw logits
  bge   BAAI/bge-reranker-base, raw logits (skipped if it cannot load)
  llm   Gemini gemini-3.5-flash-lite as a relevance judge, one call per
        (row, agent) scoring all of that row's paragraphs, plus a
        deterministic lexical tie-breaker

Every score is cached in results/scores_<name>.json, keyed by
key(par, agent). Builds are resumable: keys already present are skipped.

USAGE
  python3 rebuttal_scorers.py --scorers ce bge llm   # build caches
  python3 rebuttal_scorers.py --report               # delivery-recall table

From other code:
  from rebuttal_scorers import key, load_scorer
  musique_benchmark.SCORER = load_scorer("ce")
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from musique_benchmark import (add_composite_receivers,  # noqa: E402
                               load_instances, _lexical)

RESULTS = HERE / "results"
NAMES = ["ce", "bge", "llm"]
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_MODEL = "BAAI/bge-reranker-base"
GEMINI_MODEL = "gemini-3.5-flash-lite"
N_INSTANCES = 250
FAMILY = "branching"


# ------------------------------------------------------------------ keys

def _sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def key(par, agent):
    """Cache key for one (paragraph, receiver question) pair."""
    return _sha1(agent["q"])[:16] + "|" + _sha1(par["title"] + "\n" + par["text"])[:16]


def cache_path(name):
    return RESULTS / f"scores_{name}.json"


def _read_cache(name):
    p = cache_path(name)
    if not p.exists():
        return {}
    try:
        with open(p) as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        # A torn write should not happen (we use atomic replace), but if
        # it does, keep the file for inspection and start empty.
        print(f"WARNING: {p} unreadable, starting from empty cache",
              file=sys.stderr)
        return {}


def _save_cache(name, data):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = cache_path(name)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, p)


def load_scorer(name):
    """Return (par, agent) -> float backed by the on-disk cache.

    Raises KeyError with a clear message if a pair is not cached. The
    special name "lexical" returns the bag-of-words scorer directly.
    """
    if name == "lexical":
        return _lexical
    if name not in NAMES:
        raise KeyError(f"unknown scorer {name!r}; expected one of {NAMES + ['lexical']}")
    table = _read_cache(name)
    if not table:
        raise KeyError(f"no cache for scorer {name!r} at {cache_path(name)}; "
                       f"run: python3 rebuttal_scorers.py --scorers {name}")

    def scorer(par, agent):
        k = key(par, agent)
        if k not in table:
            raise KeyError(f"scorer {name!r} has no cached score for key {k} "
                           f"(title={par.get('title')!r}); rebuild the cache")
        return table[k]
    return scorer


# ------------------------------------------------------------------ data

def collect_instances():
    rows = add_composite_receivers(load_instances(N_INSTANCES, FAMILY))
    return rows


def unique_pairs(rows):
    """All (key, agent q, paragraph) pairs over every row, deduplicated."""
    seen, out = set(), []
    for row in rows:
        for agent in row["agents"]:
            for par in row["paragraphs"]:
                k = key(par, agent)
                if k not in seen:
                    seen.add(k)
                    out.append((k, agent["q"], par))
    return out


# ------------------------------------------------------------ cross-encoders

def _pair_text(par):
    return par["title"] + ". " + par["text"]


def build_cross_encoder(name, model_id, rows):
    """Resumable CrossEncoder scoring. Returns True on success."""
    try:
        import torch
        from sentence_transformers import CrossEncoder
        print(f"[{name}] loading {model_id} ...", flush=True)
        model = CrossEncoder(model_id)
    except Exception as exc:  # download / load failure: report and go on
        print(f"[{name}] FAILED to load {model_id}: "
              f"{type(exc).__name__}: {str(exc)[:300]}", flush=True)
        return False

    table = _read_cache(name)
    pending = [(k, q, p) for k, q, p in unique_pairs(rows) if k not in table]
    print(f"[{name}] {len(table)} cached, {len(pending)} to score", flush=True)

    chunk = 256
    t0 = time.time()
    for start in range(0, len(pending), chunk):
        batch = pending[start:start + chunk]
        pairs = [(q, _pair_text(p)) for _, q, p in batch]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False,
                               activation_fn=torch.nn.Identity())
        for (k, _, _), s in zip(batch, scores):
            table[k] = float(s)
        _save_cache(name, table)
        done = min(start + chunk, len(pending))
        print(f"[{name}] {done}/{len(pending)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[{name}] complete: {len(table)} scores", flush=True)
    return True


# ---------------------------------------------------------------- LLM judge

PROMPT_HEAD = ("Rate how useful each paragraph is for answering the question, "
               "0 (useless) to 10 (contains the answer). Return a JSON object "
               "mapping each paragraph number (string) to an integer score, "
               "covering all paragraphs.")

MAX_ATTEMPTS = 6
RETRY_CODES = {429, 500, 502, 503, 504}


class ParseError(Exception):
    pass


def build_llm_prompt(agent, paras):
    lines = [f"[{i}] {p['title']}: {p['text']}"
             for i, p in enumerate(paras, 1)]
    return ("Question: " + agent["q"] + "\n\nParagraphs:\n"
            + "\n\n".join(lines) + "\n\n" + PROMPT_HEAD)


def parse_llm_scores(text, m):
    """Parse {"1": 7, ...} robustly. Missing numbers -> 0.

    Raises ParseError when no paragraph number yields a valid score,
    which is treated as a retryable failure rather than an all-zero row.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ParseError("no JSON object in reply")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}")
    if not isinstance(obj, dict):
        raise ParseError("JSON is not an object")
    scores = {}
    for k, v in obj.items():
        m_num = re.search(r"\d+", str(k))
        if not m_num:
            continue
        idx = int(m_num.group(0))
        if not (1 <= idx <= m):
            continue
        try:
            val = int(round(float(str(v).strip().strip('"'))))
        except ValueError:
            continue
        scores[idx] = min(10, max(0, val))
    if not scores:
        raise ParseError("no valid paragraph scores in reply")
    return {i: scores.get(i, 0) for i in range(1, m + 1)}


def gemini_judge(prompt, key_str):
    """One generateContent call with retries. Returns parsed text reply.

    Retries on HTTP 429/5xx, network errors and unparseable replies, with
    exponential backoff. Other HTTP errors fail immediately.
    """
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={key_str}")
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0,
                             "responseMimeType": "application/json",
                             "maxOutputTokens": 2048},
    }).encode()
    last = "unknown"
    for attempt in range(MAX_ATTEMPTS):
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read())
            cand = body["candidates"][0]
            text = "".join(p.get("text", "")
                           for p in cand.get("content", {}).get("parts", []))
            if not text.strip():
                raise ParseError(f"empty reply ({cand.get('finishReason')})")
            return text
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code not in RETRY_CODES:
                return _fail(last, exc)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = f"network {type(exc).__name__}"
        except (ParseError, KeyError, IndexError, ValueError) as exc:
            last = f"reply {type(exc).__name__}: {exc}"
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts: {last}")


def _fail(msg, exc):
    raise RuntimeError(f"non-retryable {msg}: {str(exc)[:200]}")


def llm_job(row_idx, agent_idx, rows, key_str):
    row = rows[row_idx]
    agent = row["agents"][agent_idx]
    paras = row["paragraphs"]
    text = gemini_judge(build_llm_prompt(agent, paras), key_str)
    got = parse_llm_scores(text, len(paras))
    out = {}
    for i, p in enumerate(paras, 1):
        out[key(p, agent)] = got[i] + 0.001 * _lexical(p, agent)
    return out


def build_llm(rows):
    key_str = os.environ.get("GEMINI_API_KEY")
    if not key_str:
        sys.exit("set GEMINI_API_KEY")

    table = _read_cache("llm")
    failures_path = RESULTS / "llm_failures.json"
    failures = []
    if failures_path.exists():
        try:
            failures = json.loads(failures_path.read_text())
        except json.JSONDecodeError:
            failures = []

    todo = []
    for ri, row in enumerate(rows):
        for ai, agent in enumerate(row["agents"]):
            ks = [key(p, agent) for p in row["paragraphs"]]
            if not all(k in table for k in ks):
                todo.append((ri, ai))
    print(f"[llm] {len(rows)} rows, {len(todo)} (row, agent) calls to make; "
          f"{len(table)} scores cached", flush=True)

    lock = threading.Lock()
    n_fail, n_done = 0, 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(llm_job, ri, ai, rows, key_str): (ri, ai)
                for ri, ai in todo}
        for fut in as_completed(futs):
            ri, ai = futs[fut]
            try:
                scores = fut.result()
                with lock:
                    table.update(scores)
            except Exception as exc:
                n_fail += 1
                with lock:
                    failures.append({"row": rows[ri]["id"], "agent": ai,
                                     "error": str(exc)[:300]})
                print(f"[llm] FAILURE row={rows[ri]['id']} agent={ai}: {exc}",
                      file=sys.stderr, flush=True)
            n_done += 1
            if n_done % 20 == 0 or n_done == len(todo):
                with lock:
                    _save_cache("llm", table)
                    failures_path.write_text(json.dumps(failures, indent=1))
                print(f"[llm] {n_done}/{len(todo)} calls, {n_fail} failed, "
                      f"{len(table)} scores", flush=True)
    _save_cache("llm", table)
    failures_path.write_text(json.dumps(failures, indent=1))
    print(f"[llm] done: {n_fail} failed calls this run, "
          f"{len(failures)} recorded in {failures_path}", flush=True)
    return n_fail


# ------------------------------------------------------------------ report

def _rank_hit(rows, scorer_fn, ks=(1, 3, 5)):
    """Delivery recall per receiver type and k.

    Sub-question receivers: 1 if the gold paragraph (agent["needs"]) is in
    the top-k. Composite receivers: 1 only if ALL of agent["needs_all"]
    are in the top-k (all-or-nothing, as multi-hop answers need every hop).
    Ranking is by score descending; ties are broken by a label-free hash
    of the key so that a constant scorer gets no positional advantage.
    """
    sub = {k: [] for k in ks}
    comp = {k: [] for k in ks}
    for row in rows:
        for agent in row["agents"]:
            ranked = sorted(
                row["paragraphs"],
                key=lambda p: (-scorer_fn(p, agent),
                               _sha1(key(p, agent))))
            top_ids = [p["idx"] for p in ranked]
            if agent.get("composite"):
                need = agent["needs_all"]
                for k in ks:
                    comp[k].append(1.0 if all(j in top_ids[:k] for j in need)
                                   else 0.0)
            else:
                for k in ks:
                    sub[k].append(1.0 if agent["needs"] in top_ids[:k] else 0.0)
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return ({k: avg(sub[k]) for k in ks}, len(sub[1]),
            {k: avg(comp[k]) for k in ks}, len(comp[1]))


def report(rows):
    names = [n for n in NAMES if cache_path(n).exists()] + ["lexical"]
    print(f"Instances: {len(rows)}   sub-question receivers: "
          f"{sum(1 for r in rows for a in r['agents'] if not a.get('composite'))}"
          f"   composite receivers: "
          f"{sum(1 for r in rows for a in r['agents'] if a.get('composite'))}")
    hdr = (f"{'scorer':10s} | {'sub-q top1':>10s} {'top3':>6s} {'top5':>6s} |"
           f" {'comp top1':>9s} {'top3':>6s} {'top5':>6s} | {'missing':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        try:
            fn = load_scorer(name)
            missing = 0
        except KeyError as exc:
            print(f"{name:10s} | unavailable: {str(exc)[:90]}")
            continue
        try:
            sub, n_sub, comp, n_comp = _rank_hit(rows, fn)
        except KeyError as exc:
            # Count how many pairs are missing so the row is informative.
            table = _read_cache(name)
            missing = sum(1 for r in rows for a in r["agents"]
                          for p in r["paragraphs"] if key(p, a) not in table)
            print(f"{name:10s} | incomplete cache: {missing} pairs missing")
            continue
        print(f"{name:10s} | {sub[1]:10.3f} {sub[3]:6.3f} {sub[5]:6.3f} |"
              f" {comp[1]:9.3f} {comp[3]:6.3f} {comp[5]:6.3f} | {missing:7d}")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorers", nargs="+", choices=NAMES, default=[],
                    help="build the caches for these scorers")
    ap.add_argument("--report", action="store_true",
                    help="print top-k delivery recall for all cached scorers")
    args = ap.parse_args()
    if not args.scorers and not args.report:
        ap.error("nothing to do: pass --scorers and/or --report")

    rows = collect_instances()
    print(f"{len(rows)} rows, {sum(len(r['agents']) for r in rows)} agents, "
          f"{sum(len(r['paragraphs']) for r in rows)} paragraphs, "
          f"{len(unique_pairs(rows))} unique (agent, paragraph) pairs",
          flush=True)

    for name in args.scorers:
        if name == "ce":
            build_cross_encoder("ce", CE_MODEL, rows)
        elif name == "bge":
            ok = build_cross_encoder("bge", BGE_MODEL, rows)
            if not ok:
                print("[bge] skipped: model unavailable", flush=True)
        elif name == "llm":
            build_llm(rows)

    if args.report:
        report(rows)


if __name__ == "__main__":
    main()
