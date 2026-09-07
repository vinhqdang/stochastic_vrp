# PARCEL — empirical status

Last updated 2026-09-07. Regenerate every table from `code/`; see
`code/README` headers for commands. Raw per-call JSONL is gitignored as
regenerable — the analysed summaries are what the manuscript cites.

---

## 1. Public benchmark: MuSiQue-Ans, paired comparisons

Model `gemini-3.1-flash-lite`, lexical relevance, 4-arm core grid,
69 instances → **n = 138 agent-instances per policy**. Paired McNemar
plus a paired bootstrap over items (`code/paired_test.py`); every policy
answers the same items, so the pairing is what supplies the power.

| comparison | Δacc | 95% CI | p | tokens |
|---|---|---|---|---|
| **oracle vs broadcast** | **+0.051** | [+0.014, +0.094] | **0.039** | **21.9× cheaper** |
| parcel (q=0.3) vs broadcast | −0.029 | [−0.087, +0.029] | 0.454 | 2.6× cheaper |
| topk (k=5) vs broadcast | −0.080 | [−0.145, −0.014] | **0.027** | 5.4× cheaper |

### What this supports

**The saturation premise is confirmed.** Perfect allocation beats full
broadcast *on accuracy* while sending **22× fewer tokens**
(p = 0.039). Sending everything is not merely wasteful; it is worse
than sending the right thing. This is the paper's F1 premise measured
directly rather than imported from the degradation literature, and it
is the first significant result on this benchmark.

**A 2.6× token reduction costs no measurable accuracy.** The two-price
rule's confidence interval spans zero (p = 0.45), so "same task
quality, a third of the tokens" holds.

**Cheaper is not free, though.** Top-k at 5.4× reduction loses 8
accuracy points, and that loss *is* significant (p = 0.027). So the
frontier is real: past some point token savings do cost quality.

### What this does NOT support

The two arms sit at different token levels (2.6× vs 5.4×), so this
grid does **not** establish that the two-price rule beats top-k at a
matched budget. It shows only that the rule sits on the better side of
the accuracy trade at its setting. A matched-budget comparison needs
the 12-arm grid at this sample size, which is the next run.

Earlier n≈36 results are superseded; at that size the intervals ran to
±0.11–0.17 and none of these three comparisons was significant.

---

## 2. Relevance scorer: dense embeddings are WORSE here

**Prediction falsified.** The retrieval bottleneck (top-1 recall of the
gold paragraph was 0.43) was expected to yield to embeddings. It did
not — `gemini-embedding-001` at 768 dimensions is consistently *worse*
than bag-of-words overlap for per-agent retrieval. Same 130 instances,
no API cost to reproduce (`--dry-run`):

| policy | lexical: tokens / recall | embedding: tokens / recall |
|---|---|---|
| topk k=1 | 59 / **0.43** | 40 / **0.25** |
| topk k=3 | 213 / **0.67** | 141 / **0.45** |
| topk k=5 | 410 / **0.79** | 270 / **0.61** |
| topk k=10 | 1008 / **0.92** | 700 / **0.81** |
| centrality k=3 | 254 / **0.45** | 329 / **0.85** |
| parcel q=0.95 | 315 / 0.81 | 398 / 0.72 |
| parcel q=0.3 | 869 / 0.93 | 1227 / 0.90 |

### Why, and why it is interesting rather than merely disappointing

MuSiQue's branching sub-questions are mostly **terse relation
templates**, not natural-language questions:

```
'Charles Edmund Nugent >> conflict'
'The Poor Boob >> screenwriter'
'Greenwood Laboratory School >> located in the administrative territorial entity'
```

Exact entity-token overlap finds the right paragraph; a dense encoder
maps such a string to a topical region and loses the entity identity.

The centrality row is the tell, and it flips the other way: embeddings
take it from 0.45 to **0.85**. Centrality sends one global top-k to
every agent, so it needs only *instance-level topicality* — which
embeddings capture well. Per-agent top-k needs *which sub-question
needs which paragraph* — which embeddings capture badly.

**So the two scorers fail in opposite directions:** embeddings know
what the instance is about but not who needs what; lexical matching
discriminates between agents but misses paraphrase. That is a genuine
finding about relevance signals for multi-receiver allocation, and it
belongs in the paper rather than being tuned away.

The oracle-versus-practical gap therefore remains open, and its cause
is the **query representation**, not the scorer family. A hybrid or
entity-aware scorer is the obvious next attempt; it has not been tried.

---

## 3. Synthetic sweep: at ceiling, uninformative so far

`code/degradation_sweep.py` on `gemini-3.5-flash-lite` stays at 1.00
accuracy through ~1900 tokens of *confusable* distractors, even at 4
reasoning hops. Template-generated distractors do not interfere with
this model at these scales, which is consistent with the
counter-evidence in `PROJECT.md` §11.3 and is why the public benchmark
(whose distractors are author-selected) carries the empirical argument
instead. Either the task needs more reasoning depth or the sweep should
be dropped; it currently measures nothing.

---

## 4. Budget, measured

Free-tier limits, read from quota-violation details rather than
inferred:

- **500 generate requests/day/model/project**
- **1000 embed requests/day/model/project**
- counted per request, **not** per batch

Keys from different Google Cloud projects have independent allowances,
and both clients rotate on per-day exhaustion (`code/degradation_sweep.py`
`KeyRing`, `code/embed_relevance.py`). The 4-arm core grid costs 8 calls
per instance, so one project-model pair buys ~62 instances a day.

**Contamination control:** 0–5% of agent-instances are answerable with
no context at all, and results are reported both with and without them.
Deltas, never absolute EM.
