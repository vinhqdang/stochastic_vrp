# TRANSVRP — citation verification

Verified 2026-11-15 against each paper's arXiv abs-page `citation_*` meta
tags (title, first author(s), first-posting date). The arXiv search API was
rate-limited, so abs pages were used.

| key | arXiv | verified | not yet verified |
|---|---|---|---|
| liu2026ortransformer | 2609.01933 | title, authors, v2 date; **full text read (18 pp.)** | none |
| kool2018attention | 1803.08475 | title, authors, date | ICLR 2019 venue |
| nazari2018rl | 1802.04240 | title, authors, date | NeurIPS 2018 venue |
| kwon2020pomo | 2010.16011 | title, authors, date | NeurIPS 2020 venue |
| iklassov2024svrptw | 2402.09765 | title, authors, date | publication venue; abstract only seen via search snippet |
| dastpak2026vrpsdo | 2607.16875 | title, authors, date | abstract only seen via search snippet |
| zou2025mapt | 2511.17435 | title, authors, date | "AAAI 2026" venue claim comes from a search snippet |
| ye2024reevo | 2402.01145 | title, authors, date | NeurIPS 2024 venue |
| liu2024eoh | 2401.02051 | title, authors, date | ICML 2024 venue |
| zhang2025afl | 2510.16701 | title, authors, date | "ICLR 2026" venue claim from a search snippet |
| xiu2026vrpllm | 2607.00604 | title, authors, date | content only via snippet; read before citing for claims |
| talebi2025llmformulation | 2508.17200 | title, authors, date | content unread |
| tan2025thinkingwhiledriving | 2512.10610 | title, authors, date | content unread |

Rules: no venue or number is stated in the manuscript unless it sits in the
"verified" column. Papers marked "snippet only" must be opened and read
before any claim about their results goes into text.

Also unresolved: a literature search found **no paper** that combines
transformer RL, stochastic demand *and* pickup-and-delivery with online
recourse. This is a search result, not a proof of absence; repeat with
targeted queries before claiming novelty.

## Weight-space programming (added 2026-11-15, raised by the author)

Seen **only in secondary coverage** (DataCamp, VentureBeat, several DEV
Community posts, AI Socratic), all via search snippets; the primary MIT
CSAIL post/paper was **not located**. Reported claims (unverified): an LLM
("GPT-6 Astra") read Qwen3.5-0.8B weights, found one direction written by
the MLPs of layers 15 and 19 tied to runaway/looping generation, subtracted
it from 64 output columns per layer (131,072 weights), and MATH-500 moved
7.9% -> 48.1%; an independent Hugging Face rerun is reported as 51.0%, and
one report gives 49.45% for the unedited model with thinking disabled, so
the baseline depends on the setup. Reported scope: fault removal, not new
ability; one LLM, models up to 0.8B.
**Do not cite or quantify anything from this until the primary source is
found and read.** No bib entry is created for it.

Related, snippet-only, unread: SHINE hypernetwork (arXiv 2602.06358) and a
universal-hypernetwork paper (arXiv 2604.02215). They generate weights with
a trained network, which is a different mechanism from an LLM editing
weights by reading them.
