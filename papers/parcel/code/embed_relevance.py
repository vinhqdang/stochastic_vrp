"""Embedding-based relevance, replacing lexical word overlap.

WHY
---
The first MuSiQue run showed every implementable policy losing accuracy
to full broadcast, while the oracle beat it at 25x fewer tokens. The gap
is a RETRIEVAL failure, not an allocation failure: per-agent top-1 found
the gold paragraph only 44% of the time, because the scorer was bag-of-
words overlap between a short sub-question and a ~114-word paragraph.

That handicaps every policy equally, but it also caps the whole
experiment -- a policy cannot allocate well on top of a scorer that
cannot find the right item. Worse for the paper's purpose, it leaves no
regime where the saturation price could matter, since nothing is worth
sending in the first place.

This module supplies a stronger, still-cheap relevance signal: cosine
similarity between embeddings of the sub-question and the paragraph.
It is still IMPERFECT (no gold labels, no fine-tuning), which is the
point -- a real router has an approximate scorer, not an oracle.

Embeddings are cached on disk by content hash, so a re-run costs
nothing and the generation quota is spent only on the benchmark itself.

QUOTA, measured: the free tier allows 1000 EmbedContent requests per
day PER MODEL, and it counts individual TEXTS rather than batches, so
batching helps latency but not the daily allowance. One model therefore
covers about 45 MuSiQue instances a day (20 paragraphs plus 2 sub-
questions each). The cache is permanent and keyed by model, so coverage
accumulates across days instead of restarting.

Uses `gemini-embedding-001` by default (override with
PARCEL_EMBED_MODEL) via batchEmbedContents, at a reduced output
dimensionality: 3072-d vectors cost ~69 KB each as JSON and blew the
cache to 66 MB for under a thousand paragraphs, while 768 dimensions
rank just as well for this purpose. Values are rounded before storage,
which costs nothing in ranking and shrinks the file again.
"""

import hashlib
import json
import math
import os
import pathlib
import re
import time
import urllib.error
import urllib.request

MODEL = os.environ.get("PARCEL_EMBED_MODEL", "gemini-embedding-001")
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       f"{MODEL}:batchEmbedContents?key={{key}}")
CACHE = pathlib.Path(__file__).parent / "results" / "embed_cache.jsonl"
BATCH = 32
DIM = 768        # ranking quality is flat well below the 3072 default
ROUND = 5        # storage precision; irrelevant to cosine ordering
RETRIES = 25     # free-tier embedding is paced at ~40s between
                 # batches, so patience is the whole strategy


def _key(text):
    """Cache key includes model and width.

    Vectors from different models live in different spaces and are not
    interchangeable, so keying on text alone would silently mix them
    once the model is switched -- which the daily-quota situation makes
    likely.
    """
    tag = f"{MODEL}|{DIM}|{text}"
    return hashlib.sha1(tag.encode("utf-8")).hexdigest()[:20]


def load_keys():
    """Every configured key, in order of use.

    The daily allowance is per Google Cloud PROJECT (the quota metric is
    literally GenerateRequestsPerDayPerProjectPerModel), so a second key
    only adds budget if it belongs to a DIFFERENT project. Keys from the
    same project share one allowance and rotation buys nothing.

    Read from GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
    """
    keys = []
    for name in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}"
                                      for i in range(2, 6)]:
        v = os.environ.get(name)
        if v and v not in keys:
            keys.append(v)
    return keys


class Embedder:
    def __init__(self, key=None):
        self.keys = [key] if key else load_keys()
        if not self.keys:
            raise SystemExit("set GEMINI_API_KEY")
        self.ki = 0
        if len(self.keys) > 1:
            print(f"  {len(self.keys)} keys configured; will rotate on "
                  f"daily-quota exhaustion", flush=True)
        self._load_cache()

    @property
    def key(self):
        return self.keys[self.ki]

    def _next_key(self):
        """Advance to the next key. Returns False when none are left."""
        if self.ki + 1 < len(self.keys):
            self.ki += 1
            print(f"  daily quota spent; switching to key "
                  f"{self.ki + 1}/{len(self.keys)}", flush=True)
            return True
        return False

    def _load_cache(self):
        # Append-only: rewriting the whole map after every batch is
        # quadratic in bytes written, and with a few thousand vectors it
        # dominated the runtime -- two earlier runs were killed by
        # timeouts because of it, not because of the API.
        self.cache = {}
        if CACHE.exists():
            with open(CACHE) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        k, v = json.loads(line)
                        self.cache[k] = v
                    except (json.JSONDecodeError, ValueError):
                        continue      # tolerate a torn final line
        self._fh = None

    def save(self):
        if self._fh:
            self._fh.flush()

    def _append(self, key, vec):
        if self._fh is None:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(CACHE, "a")
        self._fh.write(json.dumps([key, vec]) + "\n")
        self.cache[key] = vec

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def _fetch(self, texts):
        payload = json.dumps({
            "requests": [{"model": f"models/{MODEL}",
                          "content": {"parts": [{"text": t[:8000]}]},
                          "outputDimensionality": DIM}
                         for t in texts]}).encode()
        for attempt in range(RETRIES):
            req = urllib.request.Request(
                URL.format(key=self.key), data=payload,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = json.loads(r.read())
                return [e["values"] for e in body["embeddings"]]
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    # Read the WHOLE body: the quota metric name lives in
                    # the details array, well past the first few hundred
                    # characters, and truncating it meant the per-day case
                    # was never recognised and key rotation never fired.
                    detail = exc.read().decode()
                except Exception:
                    pass
                if exc.code == 429 and attempt < RETRIES - 1:
                    # A per-day exhaustion is not worth waiting out -- the
                    # window is hours, not seconds -- so switch keys if a
                    # different project is configured. Only the per-minute
                    # variety is worth sleeping on, and the API's own
                    # retryDelay says how long (guessing a fixed escalating
                    # sleep previously overshot the caller's timeout and
                    # made a recoverable pause look like a failure).
                    per_day = ("PerDay" in detail
                               or "PerProjectPerDay" in detail)
                    if per_day and self._next_key():
                        continue
                    m = (re.search(r'"retryDelay"\s*:\s*"?(\d+(?:\.\d+)?)s',
                                   detail)
                         or re.search(r"retry in (\d+(?:\.\d+)?)s", detail))
                    time.sleep(min(float(m.group(1)) + 1 if m else 8.0, 40.0))
                    continue
                raise RuntimeError(f"embed HTTP{exc.code}: {detail[:300]}")
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == RETRIES - 1:
                    raise RuntimeError(f"embed failed: {exc}")
                time.sleep(2 ** attempt)
        raise RuntimeError("embed unreachable")

    def embed(self, texts, progress=False):
        """Return one unit-normalised vector per text, using the cache."""
        missing = [t for t in dict.fromkeys(texts) if _key(t) not in self.cache]
        if progress and missing:
            print(f"  embedding {len(missing)} new texts "
                  f"({len(self.cache)} already cached)", flush=True)
        for i in range(0, len(missing), BATCH):
            if progress and i and i % (BATCH * 5) == 0:
                print(f"    {i}/{len(missing)}", flush=True)
            chunk = missing[i:i + BATCH]
            for text, vec in zip(chunk, self._fetch(chunk)):
                n = math.sqrt(sum(v * v for v in vec)) or 1.0
                self._append(_key(text), [round(v / n, ROUND) for v in vec])
            self.save()
        return [self.cache[_key(t)] for t in texts]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def build_scorer(instances, embedder=None):
    """Pre-embed every paragraph and sub-question; return score(par, agent).

    Pre-embedding in one pass keeps the API calls batched and lets the
    policies stay pure functions of a lookup table.
    """
    emb = embedder or Embedder()
    texts = []
    for inst in instances:
        for p in inst["paragraphs"]:
            texts.append(p["title"] + ". " + p["text"])
        for a in inst["agents"]:
            texts.append(a["q"])
    vecs = dict(zip(texts, emb.embed(texts, progress=True)))
    emb.close()

    def score(par, agent):
        pv = vecs.get(par["title"] + ". " + par["text"])
        qv = vecs.get(agent["q"])
        if pv is None or qv is None:
            return 0.0
        # Cosine sits in [-1, 1]; shift to keep densities non-negative so
        # the price comparison stays well defined.
        return max(0.0, cosine(pv, qv))

    return score
