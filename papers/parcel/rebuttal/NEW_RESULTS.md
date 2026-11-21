# Rebuttal-phase experiments (20–21 Nov 2026)

Every number quoted in `REBUTTAL.md`, the script that produces it, and
the committed summary it is read from (all paths relative to
`papers/parcel/code/`). Raw per-call JSONL is gitignored as regenerable,
as elsewhere in this paper.

Model access during the window: the Gemini free tier allows 500
requests/day per model, so the bulk runs called the **same**
`gemini-3.5-flash-lite` through OpenRouter (temperature 0, identical
prompt and extraction). The first 2-hop pipeline jobs ran on the direct
Gemini API before the daily quota ran out. Llama-3.3-70B-Instruct also
ran via OpenRouter. The Gemini LLM-relevance judge in
`rebuttal_scorers.py` could not run (quota) and is not reported.

## R1. Held-out replication, frozen settings — `gemini-3.5-flash-lite`

`musique_benchmark.py --model google/gemini-3.5-flash-lite --grid rebuttal --offset 130 --instances 120`,
analysed by `analyze_heldout.py` (real `countTokens` on the exact
transmitted context; instance-level block bootstrap + permutation test).
Instances 130–249 of the seeded shuffle; every earlier grid used only
the first 130. k=5 and q=0.3 are frozen from the submitted grid.
Summary: `results/summary_musique_google-gemini-3.5-flash-lite_lexical_rebuttal_m1_off130.json`.

| policy | acc | Δ vs broadcast | real tok/receiver | ratio |
|---|---|---|---|---|
| broadcast | 0.771 | — | 2619 | 1.0× |
| density loose q=0.3 (frozen) | 0.725 | −0.046 | 1000 | 2.6× |
| uniform k=5 (frozen) | 0.692 | −0.079 | 490 | 5.3× |
| per-recv CE top-3 | 0.758 | −0.013 | 393 | 6.7× |
| seed set CE k=3 | 0.692 | −0.079 | 363 | 7.2× |
| seed set lexical k=3 | 0.442 | −0.329 | 302 | 8.7× |
| per-recv bge top-2 | 0.721 | −0.050 | 269 | 9.7× |
| per-recv lexical k=3 | 0.633 | −0.138 | 261 | 10.0× |
| per-recv CE top-2 | 0.767 | −0.004 | 248 | 10.6× |
| per-recv CE top-1 | 0.767 | −0.004 | 111 | 23.6× |
| oracle | 0.821 | +0.050 | 110 | 23.9× |
| no context | 0.008 | −0.763 | 0 | — |

Paired, cluster-robust (n = 240 receivers / 120 instances):

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| per-recv lex k3 vs seed lex k3 | +0.192 | [+0.125,+0.258] | <0.001 |
| oracle vs broadcast | +0.050 | [+0.004,+0.096] | 0.049 |
| density q=0.3 vs broadcast | −0.046 | [−0.087,−0.004] | 0.062 |
| uniform k=5 vs broadcast | −0.079 | [−0.133,−0.025] | 0.009 |
| per-recv CE k1 vs broadcast | −0.004 | [−0.046,+0.037] | 1.000 |
| per-recv CE k2 vs broadcast | −0.004 | [−0.046,+0.037] | 1.000 |
| per-recv CE k2 vs seed CE k3 | +0.075 | [+0.033,+0.117] | 0.001 |
| per-recv CE k2 vs per-recv lex k3 | +0.133 | [+0.071,+0.196] | <0.001 |
| per-recv CE k2 vs oracle | −0.054 | [−0.096,−0.017] | 0.011 |

Note: density q=0.3's bootstrap CI excludes zero while its permutation p
is 0.062; the rebuttal quotes the CI and calls the loss "small".

## R2. Second model family — Llama-3.3-70B-Instruct

`musique_benchmark.py --model meta-llama/llama-3.3-70b-instruct --grid rebuttal_x --offset 130 --instances 60`.
Summary: `results/summary_musique_meta-llama-llama-3.3-70b-instruct_lexical_rebuttal_x_m1_off130.json`
(tokens: Gemini `countTokens` on the same contexts, for comparability).

| comparison (n = 120 / 60 inst) | Δ | 95% CI | p |
|---|---|---|---|
| per-recv lex k3 vs seed lex k3 | +0.242 | [+0.158,+0.325] | <0.001 |
| oracle vs broadcast | +0.042 | [−0.017,+0.100] | 0.262 |
| uniform k=5 vs broadcast | −0.067 | [−0.133,0.000] | 0.094 |
| per-recv CE k1 vs broadcast | −0.058 | [−0.117,0.000] | 0.089 |
| per-recv CE k2 vs broadcast (10.8×) | −0.042 | [−0.092,+0.008] | 0.183 |
| per-recv CE k2 vs seed CE k3 | +0.092 | [+0.033,+0.150] | 0.013 |
| per-recv CE k2 vs per-recv lex k3 | +0.117 | [+0.033,+0.200] | 0.015 |

A `gemini-3.5-flash` run was attempted on the free tier and stopped
after 5 calls (quota); it is not used.

## R3. Dependent multi-agent pipeline

`pipeline_agents.py` (120 MuSiQue chains, 40 each with 2/3/4 sub-
questions; "#k" in a sub-question is replaced by agent k's extracted
answer; final answer = last agent's; API-reported prompt tokens).
Summary: `results/summary_pipeline_gemini-3.5-flash-lite.json`.

| policy | acc | prompt tok/instance | vs full history |
|---|---|---|---|
| full_history (all docs + every earlier agent's question and full reply) | 0.408 | 8159 | 1.00 |
| all_docs_answers (all docs + referenced upstream answers only) | 0.433 | 7969 | 0.977 |
| per_receiver_k3 (lexical, on the resolved question) | 0.283 | 1110 | 0.136 |
| per_receiver_k5 | 0.317 | 1838 | 0.225 |
| per_receiver_ce_k2 (cross-encoder, resolved question) | 0.400 | 980 | 0.120 |
| seed_set_k3 (one shared set) | 0.075 | 1036 | 0.127 |
| oracle (gold paragraph + referenced answers) | 0.617 | 598 | 0.073 |

Tokens per instance by number of agents (2/3/4): full_history
4964/8325/11188; per_receiver_k3 692/1168/1471. Per-agent prompt in
full_history by position: 2616, 2684, 2837, 2902.

| comparison (n = 120) | Δ | 95% CI | p |
|---|---|---|---|
| oracle vs full_history | +0.208 | [+0.117,+0.292] | <0.001 |
| per_receiver_ce_k2 vs full_history | −0.008 | [−0.108,+0.083] | 1.000 |
| per_receiver_k3 vs full_history | −0.125 | [−0.225,−0.025] | 0.021 |
| seed_set_k3 vs full_history | −0.333 | [−0.425,−0.242] | <0.001 |
| all_docs_answers vs full_history | +0.025 | [−0.050,+0.100] | 0.679 |
| per_receiver_k3 vs seed_set_k3 | +0.208 | [+0.125,+0.292] | <0.001 |
| per_receiver_ce_k2 vs per_receiver_k3 | +0.117 | [+0.033,+0.200] | 0.013 |
| per_receiver_ce_k2 vs oracle | −0.217 | [−0.308,−0.133] | <0.001 |

Failure mode seen in the logs: UNKNOWN cascades. When an upstream agent
answers UNKNOWN, the resolved question ("UNKNOWN >> country") retrieves
nothing useful downstream.

## R4. Estimation vs. allocation (factorial; no API)

`estimation_vs_allocation.py --instances 250 [--base ce] [--plain]`.
Scores s = α·gold + (1−α)·normalised base score. The table gives
Algorithm 1 (captured-mass curve) minus uniform top-k in delivery recall
at matched tokens, with a 95% instance bootstrap.
Summaries: `results/summary_est_vs_alloc_{lexical,ce}_{composite,plain}.json`.

| α | lexical, composite | CE, composite | lexical, plain |
|---|---|---|---|
| 0 | +0.009 [+.001,+.016] | +0.018 [+.011,+.025] | −0.012 |
| 0.25 | +0.018 | +0.028 | −0.010 |
| 0.5 | +0.015 | +0.032 | −0.024 |
| 0.75 | +0.029 | +0.055 | −0.016 |
| 1 | **+0.200 [+.191,+.210]** | **+0.200** | 0.000 |

At α=0.75 the gold paragraphs already rank first for every receiver,
yet the gain is small. What Algorithm 1 needs is calibrated value
curves, not only correct rankings.

## R5. Stronger scorers and receiver scaling (no API)

`rebuttal_scorers.py --scorers ce bge --report` (250 instances; ranking
by raw score). Top-1/3/5 recall of the gold paragraph for sub-question
receivers: CE (ms-marco-MiniLM-L-6-v2) 0.910/0.938/0.950; bge-reranker-base
0.802/0.912/0.940; lexical 0.580/0.756/0.820. (The paper's 0.43 is
lexical with density ranking, `relevance/tokens`.)

`scaling_ce.py` (fused instances, word-count token proxy), summary
`results/summary_scaling_ce.json`. Delivery recall (tokens/receiver):

| n receivers | seed lex k3 | seed CE k3 | seed CE k=n | per-recv CE k1 | per-recv CE k2 |
|---|---|---|---|---|---|
| 2 | 0.475 (260) | 0.842 (318) | 0.794 (204) | 0.910 (100) | 0.935 (217) |
| 8 | 0.090 (247) | 0.260 (321) | 0.600 (881) | 0.887 (100) | 0.933 (220) |
| 16 | 0.023 (267) | 0.119 (337) | 0.510 (1830) | 0.852 (100) | 0.931 (216) |

## Theory note used in the reply to jkMC

Under a conjunctive utility (the task succeeds only if every receiver
gets its item), Proposition 1(a)'s instance (n items, B = nc) gives
OPT_seed = 0, since one item satisfies one receiver, and
OPT_per-receiver = 1. The ratio is therefore unbounded for n ≥ 2 without
saturation. Proposition 4's decomposition needs additivity across
receivers and does not cover this case.
