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

**The saturation premise is supported, though not on one model alone.**
Perfect allocation beats full broadcast *on accuracy* while sending
**22× fewer tokens** (p = 0.039 here). Sending everything is not merely
wasteful; it appears worse than sending the right thing — the paper's
F1 premise measured directly rather than imported from the degradation
literature.

⚠️ **Do not quote this single p-value as establishing it.** The same
comparison on `gemini-3.5-flash-lite` gives an almost identical effect
(+0.050) at p = 0.146 (§1b). The honest statement is a consistent
~5-point effect across two models at borderline power, pooled before
claiming. A referee who sees only the significant model will find the
other one.

**A 2.6× token reduction costs no measurable accuracy.** The two-price
rule's confidence interval spans zero (p = 0.45), so "same task
quality, a third of the tokens" holds.

**Cheaper is not free, though.** Top-k at 5.4× reduction loses 8
accuracy points, and that loss *is* significant (p = 0.027). So the
frontier is real: past some point token savings do cost quality.

Earlier n≈36 results are superseded; at that size the intervals ran to
±0.11–0.17 and none of these three comparisons was significant.

---

## 1b. Matched-budget grid — the mechanism does NOT separate

Model `gemini-3.5-flash-lite`, 12-arm grid, 60 instances complete
(1440/1440 calls), **n = 120 per policy**. This is the run that the
core grid could not answer.

| comparison | Δacc | 95% CI | p | token ratio |
|---|---|---|---|---|
| **topk k=3 vs centrality k=3** | **+0.192** | [+0.100, +0.283] | **0.0004** | 1.1× (matched) |
| parcel q=0.95 vs topk k=3 | +0.092 | [+0.025, +0.158] | **0.019** | 0.7× (parcel spends 1.4× MORE) |
| parcel q=0.7 vs topk k=5 | **+0.000** | [−0.042, +0.042] | 1.000 | 0.8× (parcel spends 1.25× more) |
| parcel q=0.3 vs topk k=10 | +0.017 | [−0.058, +0.092] | 0.824 | 1.2× |
| parcel q=0.3 vs broadcast | −0.042 | [−0.100, +0.008] | 0.227 | 2.7× cheaper |
| topk k=5 vs broadcast | −0.083 | [−0.142, −0.025] | **0.013** | 5.4× cheaper |
| parcel q=0.7 vs broadcast | −0.083 | [−0.150, −0.025] | **0.013** | 4.4× cheaper |
| oracle vs broadcast | +0.050 | [−0.008, +0.108] | 0.146 | 21.4× cheaper |

### The two-price rule does not beat a tuned single price

Across three budget levels the saturation price earns nothing at equal
spend. The cleanest comparison, `parcel q=0.7` against `topk k=5`, is
**exactly zero** (p = 1.000) while parcel spends 25% *more* tokens. The
one significant accuracy win, `q=0.95` over `k=3` at +9.2 points, comes
with **40% more tokens** — it is buying accuracy with budget, not with
the mechanism.

**This is the paper's central mechanism failing to demonstrate value,
and it must be reported as such.** Combined with §10.1 (the rule was
already prior art: Harshaw et al. 2019, Shi & Lai's Algorithm 1, BPS),
the admission price is a workable method with no measured advantage
over per-agent top-k. It cannot be the contribution.

### What IS strongly validated: the multi-receiver structure

**Per-agent allocation beats one global seed set by 19.2 points at
matched cost, p = 0.0004.** The centrality arm is the stand-in for the
classical influence-maximization answer — rank items by aggregate
score, seed the top few to everyone — and it fails exactly as the
theory predicts, by over-concentrating on globally-central items that
no particular receiver needs.

This is the strongest empirical result in the project, and it supports
contribution 2 (§3.2, multi-receiver structure) rather than
contribution 3. It is also the result that most directly earns the
theory: the reason the seed-set formulation fails here is the same
reason the IM toolchain does not transfer.

### Cross-model consistency of the oracle effect

The oracle-over-broadcast effect is **+0.050 here and +0.051 on
`gemini-3.1-flash-lite`** — nearly identical magnitude, same direction,
significant on one model (p = 0.039, n = 138) and not the other
(p = 0.146, n = 120). Treat it as a consistent ~5-point effect at
borderline power, not as an established significant result on a single
model, and pool across models before claiming it.

---

## 1c. Marginal allocation does not beat uniform top-k either

The decomposition (THEORY.md Prop 5) gives an algorithm no baseline can
imitate: build each receiver's value-versus-budget curve, then solve a
resource-allocation DP for the split that equalises marginal value per
token. A global `k` cannot reallocate between receivers; greedy pricing
has no view of what a token would buy elsewhere.

**It ties.** Four value-curve designs were tried, on 130 instances,
scored by delivery recall at matched spend:

| value curve for `v_i(b)` | outcome vs tuned top-k |
|---|---|
| sum of relevance scores | **loses** — near-linear in `b`, so nothing to balance; pours budget into receivers with many mediocre candidates |
| noisy-OR over normalised scores | ties; +4–5pp at large budgets only |
| captured share of own score mass | ties |
| any of the above, with composite receivers | ties (≈11% cheaper at equal recall, +2.5pp at equal tokens) |

Composite receivers were added specifically to create the heterogeneity
Proposition 1 predicts the advantage from: MuSiQue's full multi-hop
question needs **all** its supporting paragraphs (2–4) where a
sub-question receiver needs 1, so a global `k` must starve one or
overfeed the other. Even then, marginal allocation only matches.

### Why — and this is the useful finding

Marginal allocation can only help if the per-receiver value curves are
**distinguishable from label-free signals**. On this benchmark they are
not: every receiver draws from the same pool with a similar score
distribution, so the curves look alike and an equal split is already
near-optimal. Heterogeneity in what receivers *need* (1 vs 3 items)
does not show up as heterogeneity in what can be *estimated* about
them.

That is the third independent line of evidence for the same
conclusion. The pricing rule ties (§1b); marginal allocation ties
(here); and a stronger scorer family makes things *worse* (§2). All
three say the binding constraint is **relevance estimation, not
allocation optimisation**.

The oracle makes the size of it plain: **1.00 recall at 208 tokens**,
where the best practical policy reaches 0.62 at 929. No allocation
algorithm closes that gap, because the information required to allocate
well is not present in the scores. That is a statement about the
problem, not about our algorithms, and it is worth more to the paper
than a three-percent win would have been.

⚠️ **Consequence for the paper: do not claim an algorithmic win.**
The contribution is the separation theorem, the two degeneracy results,
and the measured fact that allocation buys large token savings for free
while the remaining gap is an estimation problem.

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
