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

## Experiment B source verification (EXPERIMENT_B §3.0, revision 3.1, 2026-11-18)

Every value below was read in the file named, with the page of the PDF (or the
file:line of the code). Repositories were cloned read-only into a scratch
directory; no code from them enters this project.

### DQN — Dastpak, Errico & Jabali, C&OR 159:106338 (2023)

- **Read:** arXiv 2109.10200v2 = Les Cahiers du GERAD G-2021-53, revised June
  2022 (39 pp.). The C&OR version of record could not be opened (publisher and
  institutional-repository copies refused); the published abstract names the
  algorithm QN-CO, the preprint DecQN. Hyper-parameters below are the
  preprint's; a check against the version of record is still open.
- **Code:** the spec's github.com/moda707/vrp-sdo is **the wrong repository**:
  it holds only the instances and an instance generator for the 2026 VRP-SDO
  paper (commits 2026-07-17). The 2023 paper's code is
  **github.com/moda707/vrpvcsd_of** (README names the paper; last commit
  fef798c, 2022-11-14; TensorFlow 1.14). Read: `config.ini`, `main.py`,
  `myparser.py`, `qnetwork.py`, `rl.py`, `vrpsd_solver.py`, `Utils.py`.

| item | paper (page) | code (file:line) |
|---|---|---|
| network | MLP, 2 hidden layers, sizes 2/3·(h_in−h_out)+h_out and 1/3·(…)+h_out, ReLU, input = hand-engineered fixed-size observation (p.16, Fig. 3 p.30) | same sizes; Leaky ReLU after hidden 1, **hidden 2 linear**; init N(0, 0.1), bias 0 (qnetwork.py:43-66) |
| encoder | **no graph/attention encoder**: fixed-size observation (heat maps, vehicle features, 15 target customers × 8 features) | rl.py:72-144 |
| double Q | text: Double Q network (p.4, p.16); Algorithm 2 writes max over the target net (p.30) | online argmax, target-net evaluation (rl.py:249-253) |
| minibatch | 32 (p.32) | 128 (config.ini) |
| update rule | with probability t = 0.05 at every decision epoch (p.16, p.32) | probability 0.04 (config.ini; vrpsd_solver.py:181) |
| replay | FIFO, 50,000 (p.16, p.32) | 100,000 (config.ini) |
| learning start | not stated | buffer > 1,000 (vrpsd_solver.py:181) |
| target update | hard copy every d = 1,000 **trials** (episodes) (p.16, p.32) | 10,000 trials (config.ini; rl.py:226) |
| γ | 0.999 (p.32) | 0.999 |
| loss | Huber, δ = 5 (p.16) | Huber δ = 1 → 0.25 linear decay, rewards ÷ 50 (vrpsd_solver.py:76-77; Utils.Norms) |
| optimiser | Adam (p.16) | Adam; **no gradient clipping** |
| lr | linear 1e-3 → 1e-4 over first 2M trials (VRPVCSD, p.18) / 1.5M (VRPSD, p.20) | 1e-3 → 1e-4 between trials/10 and 2·trials/3 (vrpsd_solver.py:71-74) |
| exploration | ε-greedy 1.0 → 0.1 over first 1M (p.18) / 300k (p.20) trials | 1 → 0.1 over 30% of trials (rl.py:195) |
| training length | 5M trials (VRPVCSD, p.18); 3M (VRPSD, p.20); CPU only (p.32) | README example 100k; gpu_job.sh 3M |

Implied replay ratio (samples per stored transition): paper 32·0.05 = 1.6;
code 128·0.04 = 5.1. EXPERIMENT_B's "32 per 4, ratio 8" is not in either.

### RLSM — Herrera, Krach, Ruyssen & Teichmann, Front. Math. Finance 3(1) (2024)

- **Read:** arXiv 2104.13669v4 (1 Dec 2023, 54 pp.).
- **Code:** github.com/HeKrRuTe/OptStopRandNN (cited in the paper, p.13; commit
  0613219, 2024-04-17). Read: `algorithms/backward_induction/RLSM.py`,
  `regression.py`, `utils/randomized_neural_networks.py`, `run/run_algo.py`,
  `run/configs.py`, `backward_induction/LSM.py`.

| item | paper (page) | code |
|---|---|---|
| random layer distribution | A, b i.i.d. standard Gaussian (p.5; p.15 "mean 0, sd 1") | `normal_(0, 1)` for weight and bias (randomized_neural_networks.py:23-26) |
| shared over dates | Algorithm 1 draws once for all dates (p.6); variant "RLSMreinit" redraws per date to cut run-to-run variance (p.30) | one `Reservoir2` per pricer (regression.py:103); `reinit` flag redraws |
| activation | Leaky ReLU for RLSM (p.15), slope **not stated**; Softplus for the Greeks (p.28) | `LeakyReLU(factors[0]/2)` (RLSM.py:43), default factors (1,1,1) (configs.py:58) → **slope 0.5** |
| hidden size | 20 hidden nodes in the main comparison (p.15); K grows with paths in the convergence study (p.23) | default `hidden_size=(20,)` (configs.py:56) |
| last layer | ordinary least squares, closed form, per date (p.5, eq. 5-6) | `np.linalg.lstsq` (regression.py:115); a separate "RLSMRidge" variant exists |
| regularisation | "restricting the hidden size is the simplest and best regularisation" (p.23); ridge not used | — |
| inputs | state, plus payoff as extra input for backward-induction methods (p.15) | `use_payoff_as_input` option |
| training rows | first half of the paths (p.6) | in-the-money rows only (LSM.py:33-35) |

### DOS — Becker, Cheridito & Jentzen, JMLR 20(74):1-25 (2019)

- **Read:** jmlr.org/papers/volume20/18-232/18-232.pdf (25 pp.; PDF pages =
  journal pages). No author code consulted.

| item | value (page) |
|---|---|
| network | F = ψ ∘ a_I ∘ φ ∘ … ∘ a_1, ReLU hidden, logistic output ψ (p.5-6) |
| depth / width | "I = 2 and q1 = q2 = d + 40" (p.14); the notation of p.5 gives I − 1 hidden layers, but two widths are named: read as two hidden layers (ambiguity noted) |
| inputs | (X_n, g(n, X_n)), dimension d + 1 (p.14) |
| training | mini-batch gradient ascent, Xavier init, batch normalisation, Adam (p.9); BN placement not stated |
| batch / steps | 3,000 + d steps, each a fresh batch of 8,192 paths (p.14); fBM example 6,000 steps, batch 2,048 (p.19) |
| learning rate | not stated |
| warm start across dates | not stated |
| hardware | float32, GTX 1080, TensorFlow 1.11 (p.13) |

### PPO code — Kadyrov et al., ORP 15:100351 (2025), ref. [38]

- **Paper:** ref. [38] = github.com/shiralikadyrov/CVRP_uncertainty, "accessed
  21 July 2025" (PDF p.5, reference list p.12). Data availability: "No data was
  used".
- **Code read:** the only file, `CVRP_Traffic_and_Demand_Uncertainty_github.ipynb`
  (commit d088960, 2025-07-21), cells 3-4 (`GNNPolicy`, `train_ppo`).

| item | code | paper (METHODS_ALP_PPO §2.3) |
|---|---|---|
| value network | **none**; baseline = mean of the episode's discounted returns | two linear layers, hidden 64 |
| value-loss coefficient | n/a (no value loss) | not stated |
| advantage | G_t − mean_t(G_t); **no division by std** | "normalise rewards" |
| entropy | **bonus 0.01** (loss = policy − 0.01·entropy) | not mentioned |
| GRU | GRU(64) state updated each step but **not used in the output** (logits = linear(node emb + pooled)) | GRU memory on the pooled embedding |
| policy head | per-node linear score, Xavier init, softmax over masked nodes | linear + softmax |
| transformer | 2 layers, 4 heads, d 64, ff 128 | 2 layers, 4 heads, 64 |
| PPO | γ = 0.5, ε = 0.2, 4 epochs per episode, Adam 1e-4, grad-norm clip 1.0 | same |
| run in notebook | `train_ppo(iter=11, num_customers=30)` (demo) | > 200,000 episodes |

Consequence for T12: advantage normalisation "off" is confirmed by the code;
value coefficient and value activation cannot be settled because the code has
no value network, so they remain [DC]. The paper/code conflicts above (value
network, entropy bonus, unused GRU) are design questions and were reported,
not applied.
