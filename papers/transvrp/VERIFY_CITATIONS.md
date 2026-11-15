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

## Weight-space programming (primary source found 2026-11-15)

Primary source, **read via fetch summary (not yet read line by line)**:
Takehi, Teoh, Zhang, Khattab (MIT CSAIL), "Sparks of Weight-Space Programming:
LLMs can natively read & write neural network weights", blog post dated
2026-11-02, response section added 2026-11-12,
https://rikiyat.github.io/blog/weight-space-programming/ . **Blog, not a
peer-reviewed paper and not on arXiv**; no separate MIT paper is linked. Logs
and code: github.com/RikiyaT/lm_understand-logs; edited model on Hugging Face
(RikiyaT/Qwen3.5-0.8B-weight-space-programming). The post says it is "not a
method" but an emerging capability, and that a benchmark/harness is being built
but is not released.

Claims relevant to us (as reported by the post):
- Exp 4 (car-on-ramp): edited 3.5% of 171,664 weights of a failing **PPO policy**;
  success 1,024/1,024 episodes vs 0/256 for PPO. This is the closest analogue to
  repairing a routing policy.
- Exp 8 (Qwen3.5-0.8B): 131,072 weights (0.016%) edited; MATH-500 7.9% -> 48.1%,
  GSM8K 13.8% -> 67.1%, IFEval 21.0% -> 62.3%; max-length hits 93% -> 33%.
  With only 10 or 25 forward passes the gains were far smaller (12.3%, 14.1%).
- Control: Gemma 3 270M, no meaningful gain (6.8% -> at most 7.7%).
- Tool use varies by experiment: some forbid Python, Exp 8 uses a Python kernel
  and forward passes.
- Stated limits: removes an obstacle, cannot add a missing ability; results are
  early; some rest on few seeds or single attempts.
- **No comparison against fine-tuning or other gradient-based editing.**

Still unverified: the figures themselves (only the author-side post, reported
through a summary tool); the independent 51.0% Hugging Face rerun and the
49.45% no-thinking baseline come from secondary coverage. The model name
"GPT-6 Astra" is the post's wording; its identity is not established here.
A bib entry (`takehi2026wsp`, @misc with URL) is added, flagged as a blog post.

Related, snippet-only, unread: SHINE hypernetwork (arXiv 2602.06358) and a
universal-hypernetwork paper (arXiv 2604.02215). They generate weights with
a trained network, which is a different mechanism from an LLM editing
weights by reading them.
