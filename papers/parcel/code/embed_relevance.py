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

Uses `gemini-embedding-001` via batchEmbedContents.
"""

import hashlib
import json
import math
import os
import pathlib
import time
import urllib.error
import urllib.request

MODEL = "gemini-embedding-001"
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       f"{MODEL}:batchEmbedContents?key={{key}}")
CACHE = pathlib.Path(__file__).parent / "results" / "embed_cache.json"
BATCH = 64


def _key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


class Embedder:
    def __init__(self, key=None):
        self.key = key or os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise SystemExit("set GEMINI_API_KEY")
        self.cache = {}
        if CACHE.exists():
            try:
                self.cache = json.loads(CACHE.read_text())
            except json.JSONDecodeError:
                self.cache = {}
        self._dirty = 0

    def save(self):
        if self._dirty:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(self.cache))
            self._dirty = 0

    def _fetch(self, texts):
        payload = json.dumps({
            "requests": [{"model": f"models/{MODEL}",
                          "content": {"parts": [{"text": t[:8000]}]}}
                         for t in texts]}).encode()
        for attempt in range(6):
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
                    detail = exc.read().decode()[:300]
                except Exception:
                    pass
                if exc.code == 429 and attempt < 5:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise RuntimeError(f"embed HTTP{exc.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 5:
                    raise RuntimeError(f"embed failed: {exc}")
                time.sleep(2 ** attempt)
        raise RuntimeError("embed unreachable")

    def embed(self, texts):
        """Return one unit-normalised vector per text, using the cache."""
        missing = [t for t in dict.fromkeys(texts) if _key(t) not in self.cache]
        for i in range(0, len(missing), BATCH):
            chunk = missing[i:i + BATCH]
            for text, vec in zip(chunk, self._fetch(chunk)):
                n = math.sqrt(sum(v * v for v in vec)) or 1.0
                self.cache[_key(text)] = [v / n for v in vec]
                self._dirty += 1
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
    vecs = dict(zip(texts, emb.embed(texts)))
    emb.save()

    def score(par, agent):
        pv = vecs.get(par["title"] + ". " + par["text"])
        qv = vecs.get(agent["q"])
        if pv is None or qv is None:
            return 0.0
        # Cosine sits in [-1, 1]; shift to keep densities non-negative so
        # the price comparison stays well defined.
        return max(0.0, cosine(pv, qv))

    return score
