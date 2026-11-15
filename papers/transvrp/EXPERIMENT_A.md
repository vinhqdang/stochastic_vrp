# EXPERIMENT A — set-encoder continuation values vs BATON (two-action handoff)

Status: pre-registration draft, written 2026-11-15 **before any run**. Once the
pilot (section 7.3) passes, the go/no-go table (section 1.2), the margins, the
primary data size and the eligible-route list are frozen. Any later change is
logged with date and reason in `papers/transvrp/RESEARCH_LOG.md`, and that
change is then reported in the paper as a protocol deviation.

Convention: **[DC]** marks a design choice made without evidence (not
validated). The pilot or a stated sensitivity check may revisit it; nothing else may.

Facts measured while writing this spec (CON3/SCA3 'Det' plans, copied code):
80 routes over 20 instances, m ∈ {1..20, 22}; 62 routes have m ≥ 4 and a
peak-breach rate ≥ 2% (13 of the 15 routes with m ≤ 6 never breach). On
CON3-0 route 3 (m = 17), BATON at N = 500 versus N = 2·10^5 differs by 0.73%
of cost (paired SE 0.40% over 2,000 days), and BATON at 2·10^5 is 7.5% above
the oracle. `fit_lsm_general` at N = 2·10^5, m = 17 takes 0.5 s. Torch is not
installed.

---

## 1. Hypotheses and pre-registered go/no-go

### 1.1 Hypotheses

- **H1 (main).** With simulator-scale data (N* = 10^5 paths per route), a
  set-encoder continuation model over the richer online state (section 3,
  arm FULL) has lower expected recourse cost than BATON (v2_lsm) trained on the
  same N* paths.
- **H2 (attribution, state).** Any H1 gain comes from information that BATON
  discards: FULL beats the same network fed only (k, W_k) (arm SCALAR).
- **H3 (attribution, function class).** SCALAR does not beat BATON at
  N* by more than δ, because it has the same information set; the plug-in DP at
  10^6 paths (near-optimal in the (k, W_k) class) bounds it.
- **H0 (prior, from BATON's OTR-2.1 null result, RESULTS_OTR2.md section 8).**
  The richer state adds variance and no usable signal, so the neural policy
  only ties BATON, or loses to it.

Why richer state *can* help here (a mechanism, not evidence): the generator
draws deliveries and pickups from **two independent Gaussian copulas**, each
with ρ = 0.6. W_k = P_k − D_k mixes a pickup factor and a delivery factor. A
high W_k can mean many pickups or few deliveries, and the two predict
different futures when the remaining stops are pickup-heavy or delivery-heavy.
The split P_k, D_k and the standardized residuals are observable online, since
the driver sees d_k and p_k separately (`otr_route_v2` observes `(d_k, p_k)`).
The gain is therefore an empirical question about this generator. Section 10
explains why it may not transfer to other generators.

### 1.2 Go/no-go table (primary analysis, decided before any run)

Primary statistic. For each eligible route r (section 7.1), let C^π_r be the
mean test cost over 20,000 test days, averaged over the 3 replicates. Define
Δ_r = 100 · (C^BATON-eq_r − C^FULL_r) / C^BATON-eq_r at N* = 10^5; positive
means the neural policy is cheaper. Let Δ̄ be the mean of Δ_r over eligible
routes. CI_95 and CI_90 are percentile bootstrap intervals over routes
(10,000 resamples, seed 20261115). Equivalence margin: **δ = 1.0% of BATON
cost [DC]**. That is about one seventh of the measured BATON-to-oracle gap,
and larger than the BATON N = 500 vs 2·10^5 difference.

The paired test is a one-sided Wilcoxon signed-rank test on {Δ_r}. Each
direction is tested at α = 0.025, which is Bonferroni over the two directions,
equivalent to a two-sided 5% test. The primary hypothesis is a single family,
so it gets no further correction. Secondary tests (section 8.3) use Holm.

| branch | rule (applied in this order; exhaustive) | what the paper claims |
|---|---|---|
| **WIN** | Δ̄ ≥ +1.0 **and** CI_95 lower bound > 0 **and** Wilcoxon (neural better) p < 0.025 **and** FULL cheaper on ≥ 60% of eligible routes | "At 10^5 simulated paths a ~8·10^4-parameter set encoder over the online state lowers expected recourse cost by Δ̄% [CI] versus BATON at equal data, on X/Y routes." Attribution is required before the claim is made: H2 (FULL vs SCALAR), H3 (SCALAR vs BATON-eq and DP-1e6), and the boosted-tree rung M11 on the same features. If M11 matches FULL within δ, the claim is "richer state helps; a transformer is not needed". Then, and only then, open the three-action extension and direction B. |
| **LOSE** | Δ̄ ≤ −1.0 **and** CI_95 upper bound < 0 **and** Wilcoxon (neural worse) p < 0.025 **and** FULL dearer on ≥ 60% of routes | "Even at 10^5 paths the neural policy is Δ̄% worse." Diagnose the cause with SCALAR vs DP-1e6. If SCALAR is also > δ worse than DP-1e6, the loss is a training or estimation failure of the neural pipeline, and the paper says it is evidence about this pipeline, not about state richness. Otherwise it is a clean negative result: the extra information does not pay. |
| **TIE** | neither of the above | **Strong tie** iff CI_90 ⊂ (−1.0, +1.0) (TOST at 5%): "BATON's scalar isotonic model is at the efficient complexity point; a set encoder with the richer state is equivalent within ±1% at 10^5 paths, loses at small N (curve), and costs ~10^3× more compute." **Weak tie** otherwise: "no difference ≥ 1% established; inconclusive", with the per-stratum table. The paper does not say "equivalent". |

What follows each branch: WIN goes to section 11's extensions. TIE (strong) is
written up as a negative result in the style of QUASAR's REFRAME_M4, and
direction B is not opened (PROJECT.md section 4 recommendation). LOSE gets the
same write-up plus the diagnosis. In any branch, the data-size curve
(section 8.3) is reported.

---

## 2. Problem statement and training target

### 2.1 Two-action stopping problem (one route)

A route has stops j = 1..m. The demand path has deliveries d_j ≥ 0 and
pickups p_j ≥ 0, with g_j = p_j − d_j and running load W_k = Σ_{j≤k} g_j
(W_0 = 0). Slack is B = Q − Σ_j d̄_j. The **breach time** is
τ = min{k : W_k > B}, with τ = m+1 if no breach. This is the **running peak**;
the endpoint W_m is never used. Price schedules H, E come from
`route_cost_schedules`. Decision epochs: after stop k, for k = 1..m−1, on paths
with τ > k. Actions: continue, or hand off and pay H[k−1] (0-indexed arrays).
A breach at τ ≤ m costs E[τ−1]. Completion costs 0. A breach at stop 1 has no
epoch before it.

The information at epoch k is F_k = σ(p_1, d_1, …, p_k, d_k), plus static route
data (means, prices). A policy is a stopping time σ ≤ m−1 adapted to F_k. Its
cost is H[σ−1] if the policy stops at some σ < τ; otherwise E[τ−1] if τ ≤ m;
otherwise 0. The continuation value is
C_k(x) = E[cost from stop k+1 onward | x_k = x, continue at k, follow π after].
The rule is **hand off iff Ĉ_k(x_k) > H[k−1]**, which is BATON's rule
unchanged. BATON restricts x_k to W_k (one isotonic model per k).

### 2.2 Target choice: (a) Longstaff–Schwartz realized-cost targets

**Decision: (a), LS-style regression on realized future costs**, not (b)
fitted value iteration. Reasons:
1. It is the same estimator family as BATON. The comparison then changes only
   the state and the function class, not the estimator.
2. The targets are realized costs, so the target at stop k never bootstraps on
   the network's own value at stop k+1. Errors do not compound backwards; they
   only affect *decisions*, and only near the boundary Ĉ ≈ H.
3. The targets lie in {0} ∪ {H[j]} ∪ {E[j]}, which are bounded and exact.
   (b) would need a min(H, V̂) target whose errors in the large-E region
   propagate.

Target for a path s alive at epoch k (τ_s > k), given a stopping rule ψ for
later epochs: y_k(s) = H[j−1] if ψ stops s at the first epoch j ∈ {k+1..m−1}
with j < τ_s; otherwise E[τ_s−1] if τ_s ≤ m; otherwise 0. Computed backwards,
exactly as in `fit_lsm_general`. Initialize `future[s] = E[τ_s−1]` if
τ_s ≤ m, else 0. For k = m−1 down to 1: record y_k(s) = future[s] for alive s,
then set `future[s] = H[k−1]` where alive and ψ_k(s) says stop. The only
difference from BATON: ψ_k comes from a **previously trained** network, not
from the model just fitted at step k.

### 2.3 Neural analogue of BATON's `future` update, and optimism control

BATON fits model k on paths and immediately uses it to stop the same paths,
which is in-sample optimism. With isotonic regression at N ≥ 500 this is mild.
A 10^5-parameter network at N = 100 can memorize, so the same move would bias
targets downward. Scheme (approximate policy iteration with **alternating
halves** and **cold starts**) **[DC]**:

- Split the N training paths: the first ⌊0.8N⌋ form the train part (at least
  10 paths) and the last ⌈0.2N⌉ form the inner validation part (at least 20
  paths if N ≥ 100). The train part splits into halves H0 (even indices) and
  H1 (odd indices).
- Round 0: targets on H0 with ψ ≡ never stop (reactive, identical to
  `fit_rollout` targets). Train net_0 from scratch.
- Round 1: targets on H1 with ψ = net_0. Train net_1 from scratch.
- Round 2: targets on H0 with ψ = net_1. Train net_2 from scratch.
- Deployed policy: Ĉ = ½(net_1 + net_2), with the predictions averaged.
- Each net's stopping decisions are applied only to paths it was not trained
  on. One indirect channel remains: net_1 learned from targets that net_0's
  decisions shaped, and net_0 saw H0. This is second-order. It is accepted and
  stated.
- Inner-validation targets for round r are built on the validation paths with
  the same ψ as the round's training targets. They are used only for early
  stopping (section 4.4).
- Rule: if the pilot shows that two more rounds (net_3 on H1, net_4 on H0,
  deploy ½(net_3+net_4)) lower pooled dev cost by > 0.3%, the sweep uses 5
  rounds; otherwise 3. This is decided once, at the pilot.

Reported costs are always on independent test days, so optimism can only hurt
the policy; it cannot flatter the reported number. Consequence of this design:
at small N the neural deploys nets trained on 40% of N each, while BATON uses
100%. This is the honest price of the network's capacity, and it is part of
the result.

---

## 3. State features (exact)

Notation: μp_j = p̄[route[j−1]], μd_j = d̄[route[j−1]], σp_j = CV·μp_j,
σd_j = CV·μd_j (CV = `data.CV` = 0.30), μg_j = μp_j − μd_j, P_k = Σ_{j≤k} p_j,
D_k = Σ_{j≤k} d_j, E0 = E[0] (the largest price on the route), T_MAX = 24
(> max m − 1 = 21). All features are float32. Every feature at epoch k is a
function of (p_1..p_k, d_1..d_k) and static route data only.

**Global vector (G = 12).** Rows marked H are *history* features. They are set
to 0.0 in arm SCALAR; the dimension is kept.

| # | feature | arm |
|---|---|---|
| g1 | W_k / B | both |
| g2 | max_{j≤k} W_j / B (peak so far) | H |
| g3 | k / m | both |
| g4 | (m − k) / m | both |
| g5 | m / 22 | both |
| g6 | H[k−1] / E0 | both |
| g7 | E[k−1] / E0 | both |
| g8 | P_k / B | H |
| g9 | D_k / B | H |
| g10 | z̄p_k = clip(mean over {j ≤ k : μp_j > 0} of (p_j − μp_j)/σp_j, −5, 5); 0 if the set is empty | H |
| g11 | z̄d_k, the same for deliveries | H |
| g12 | (p_k − d_k) / B (last increment) | H |

**Token per remaining stop j = k+1..m (F = 11), all arms.** These depend only
on (k, W_k) and static data:

| # | feature |
|---|---|
| t1–t4 | μp_j/B, μd_j/B, σp_j/B, σd_j/B |
| t5 | (j − k)/m (relative position) |
| t6 | j/m |
| t7 | H[j−1]/E0 if j < m, else 0 |
| t8 | E[j−1]/E0 |
| t9 | 1[j = m] |
| t10 | e_j = (W_k + Σ_{i=k+1..j} μg_i)/B (expected running load) |
| t11 | clip((B − W_k − Σ_{i=k+1..j} μg_i) / max(s_j, 1e-6·B), −5, 5), with s_j = sqrt(Σ_{i=k+1..j}(σp_i² + σd_i²)) |

Tokens j > m are padding (mask = True, PyTorch key_padding convention). t11
assumes independence across stops. It is only a feature, never a model.

**SCALAR arm = information set (k, W_k).** All non-H globals and all tokens are
deterministic functions of (k, W_k) and the route. SCALAR therefore has
exactly BATON's information, with the same network and parameter count. FULL
minus SCALAR measures information. SCALAR minus BATON measures function class
plus training procedure.

Within one route, the remaining-stop set is a deterministic function of k. The
set encoder adds no information over k in per-route training; it supplies an
inductive bias and the dynamic t10/t11. Its information value appears only
when a network is shared across routes, which is out of scope for v1
(section 11). This is stated in the paper.

---

## 4. Architecture, loss, optimization

### 4.1 Network (`ContinuationNet`) [DC on all sizes]

- Embeddings: the global vector goes through Linear(12→64) to give the global
  token; each stop token goes through Linear(11→64). The sequence is
  [global, tok_1..tok_T_MAX]; the global token is never masked. No positional
  encoding: order enters only through t5/t6. The network is permutation
  invariant over the token set given its features.
- Encoder: `nn.TransformerEncoder` with 2 × `TransformerEncoderLayer(d_model=64,
  nhead=4, dim_feedforward=128, dropout=0.0, activation="gelu",
  batch_first=True, norm_first=True)`, `enable_nested_tensor=False`.
- Pooling: concat(h_global, masked mean of the valid token outputs) gives 128
  dims, then Linear(128→64), GELU, Linear(64→1), sigmoid. Output u ∈ (0,1);
  Ĉ = E0 · u.
- Parameter count ≈ 7.7·10^4. Acceptance range is 5·10^4 to 2·10^5.

### 4.2 Loss

MSE between u and y/E0 over all alive (path, epoch) pairs, each weighted
equally. MSE estimates the conditional mean, which is what the stopping rule
needs; a Huber loss would bias it. Targets lie in [0, 1] because H, E ≤ E0 on
these schedules. The implementer asserts this at runtime.

### 4.3 Monotonicity: not enforced [DC]

No monotone head is used in either arm. In FULL, "monotone in W_k with the
other features fixed" is not a coherent constraint: W_k = P_k − D_k and
g8/g9/g10/g11 move with it. Using the same unconstrained class in both arms
keeps the FULL vs SCALAR contrast clean. Consequences: (i) the neural loses
the free regularization that isotonic regression gives BATON, so it should
lose at small N, and the data-size curve shows how much; (ii) a diagnostic is
reported: the fraction of (route, k) on a 101-point W grid in [min, B] where
SCALAR's Ĉ decreases by > 0.01·E0 between consecutive grid points. If this
exceeds 10% at N*, a monotone SCALAR variant is listed as a limitation; it is
not run in v1.

### 4.4 Optimization

- Adam, lr 1e-3, no weight decay, batch 512, gradient-norm clip 1.0.
- Each round: steps S = clip(⌈20 · n_pairs / 512⌉, 100, 3000), where n_pairs is
  the number of alive pairs in the round's half. Minibatches are drawn
  uniformly with replacement by a seeded `numpy.random.Generator`.
- Checkpoint every ⌈S/8⌉ steps. Keep the checkpoint with the lowest
  inner-validation MSE (ties go to the earlier one). There is no early
  termination, so runtime is deterministic.
- At N = 10^5 a round sees ≤ 1.5M samples against ~6·10^5 pairs in a half,
  i.e. ≤ 3 epochs. That is the simulator-scale regime: data is not the limit,
  compute is. At N = 10^6 (subset) a round sees < 1 epoch.
- Determinism: `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`,
  `torch.set_num_threads(1)`, numpy Generator seeds. The run seed is
  `zlib.crc32(f"{inst}|{route_id}|{N}|{rep}|{arm}|{round}".encode())`.
- Framework: PyTorch CPU (`pip install torch --index-url
  https://download.pytorch.org/whl/cpu`). Pin the exact installed version in
  `code/requirements-expA.txt` (numpy, scipy, scikit-learn, pandas, pytest,
  torch).

---

## 5. Data and leakage rules

- **Source:** `data.load_plans`, `data.gen_scenarios`, `data.parse_dethloff`,
  `costs.route_cost_schedules`, exactly as in `data.route_problem`. Plan key
  is `"Det"`.
- **Streams** (base = `data.stable_seed(name)`, an int in [7, 10007]), all via
  `gen_scenarios(dbar, pbar, n, seed)`:

  | purpose | seed argument | size | use |
  |---|---|---|---|
  | train, replicate r ∈ {0,1,2}, chunk c | `[base, 1, r, c]` | 100,000 per chunk | training pools |
  | dev | `[base, 3]` | 5,000 | eligibility, pilot, all design debugging |
  | test | `base + 99991` (int) | 20,000 | final evaluation only |
  | test_std | `base + 99991` (int) | 2,000 | parity with `route_problem` / GOLD only |

  The repo's train seed `base` with n = 500 is not used for this experiment;
  it stays only for the parity tests. The list seeds give independent
  `SeedSequence` streams. Note: `gen_scenarios` draws deliveries then pickups
  from one stream, so `n` changes the pickup draws. Prefixes are therefore
  taken from fixed-size chunks, never by calling with a smaller n.
- **Nested sizes:** the training set for (r, N) is the first N rows of chunk 0
  if N ≤ 10^5, and chunks 0..9 concatenated for N = 10^6. Sizes are nested
  within a replicate.
- **Grid:** N ∈ {100, 500, 2,000, 10,000, 100,000} on all eligible routes;
  N = 10^6 on the 12-route subset (section 7.1). Arm FULL runs at every N;
  arm SCALAR at N ∈ {2,000, 100,000} [DC, compute]. Replicates: 3 per
  (route, N, arm). A replicate means an independent data draw *and* an
  independent init.
- **Equal-data rule:** at every (route, N, r), BATON, tuned threshold, DP and
  OTR-2.1 get exactly the same N paths as the neural arms. BATON uses all N;
  the neural arms carve their inner validation set out of those N
  (section 2.3). No method ever sees dev or test paths during fitting.
- **"BATON given everything" control:** BATON-1e6 and DP-1e6 are fitted on
  N = 10^6 (chunks 0..9 of replicate r) for **every** eligible route and
  replicate. They are cheap (seconds per route, ~0.8 GB peak RAM).
- **Regime note:** BATON at 10^5–10^6 paths is a different regime from the
  repo's N = 500 default (`fit_lsm` docstring: N < 500 is noisy). At large N,
  isotonic regression is close to the exact (k, W_k) conditional mean.
  BATON-eq at N* is therefore the strong version of BATON, not the paper
  version, and the paper must say so.
- **Test-set lock:** no code path evaluates on `test` until the code is frozen
  and its commit hash is recorded in RESEARCH_LOG.md. `sweep.py` refuses
  `--eval test` unless `--frozen-commit <hash>` is given; the hash is written
  into every CSV row.

---

## 6. Baseline ladder (per route, per N, per replicate)

| label | definition | code |
|---|---|---|
| none | never hand off: decide with H = 1e12, bill H (`benchmark.run_baselines`) | costs.py |
| tuned_tau | `fit_otr_peak(g_N)` + `tune_tau_general(g_N, …)`, then `simulate_tau_general` | otr2.py, costs.py |
| baton_eq | `fit_lsm_general(g_N)` | costs.py |
| baton_1e6 | `fit_lsm_general(g_1e6)` | costs.py |
| dp_1e6 | `fit_dp(g_1e6)` (plug-in DP, default bins → 256) | **copy** `svrpspd_wdro/core/dp_exec.py` |
| otr21_eq | `fit_lsm_rich` at N ∈ {10^4, 10^5} (prior null result, rich features, boosted trees), μ_g = μp−μd, σ_g = sqrt(σp²+σd²) | **copy** `svrpspd_wdro/core/otr21.py` |
| nn_full | section 4, arm FULL | new |
| nn_scalar | section 4, arm SCALAR | new |
| oracle | `oracle_costs_general` (clairvoyant, handoff-only) | costs.py |

`dp_exec.py` and `otr21.py` are copied verbatim into `code/transvrp/baton/`.
Both import only `.otr2`, which is already a sibling. `PROVENANCE.md` is
updated with the source commit. DP-1e6 is the best available estimate of the
optimum in BATON's information class, so it is the reference for H3.
OTR-2.1 is the published-internally prior that richer features lose at
historical scale; running it at large N is a cheap check of the same question.

Each entry reports mean cost, handoff rate, fail rate, and the gap-closed
fraction GC = (C_none − C_π)/(C_none − C_oracle). GC is set to NaN when
C_none − C_oracle < 0.01·C_none. H_bill = H for every policy except `none`.

---

## 7. Instances, routes, budget, pilot

### 7.1 Routes

- Instances: CON3-0..9 and SCA3-0..9, plan `Det`, 80 routes.
- **Eligibility** (frozen by M8 on the **dev** set before any neural run):
  m ≥ 4 and dev peak-breach rate ≥ 2%. A test-set proxy suggests ~62 routes
  (31 CON3). If fewer than 40 qualify, stop (section 10).
- Strata by m: short 4–9 (~2 eligible; reported descriptively), medium 10–14
  (~28), long ≥ 15 (~32). Ineligible routes are listed with their (trivial)
  baseline costs and are excluded from all tests.
- **10^6 subset:** 12 routes, 6 medium and 6 long, drawn by
  `numpy.random.default_rng(2026).choice` without replacement from the sorted
  (instance, route_id) eligible list of each stratum. Replicate 0 only.

### 7.2 Budget (CPU, 4 cores, 15 GB; estimates, not measurements)

Cost per training sample forward is about 1.2 MFLOP (≈ 9 tokens × 0.13 MFLOP);
forward plus backward is about 3.6 MFLOP. Assumed single-thread throughput is
~15 GFLOP/s, with 4 jobs in parallel. A capped job (3 rounds × 3000 steps ×
512) is ≈ 1.7·10^13 FLOP, so ≈ 20–25 min. N = 2,000 takes ~4 min; N ≤ 500
~1.5 min.

| block | jobs | est. wall-clock |
|---|---|---|
| FULL, 5 sizes × 3 reps × 62 routes | 930 | ~44 h |
| SCALAR, 2 sizes × 3 reps × 62 routes | 372 | ~22 h |
| FULL + SCALAR at 10^6, 12 routes × 1 rep (incl. ~8M-pair target passes) | 24 | ~5 h |
| all classical baselines incl. 10^6 fits, test eval | — | < 2 h |
| **total** | | **~3 days (uncertainty ×2–3)** |

The pilot replaces these estimates with measured times. If the projected total
exceeds 4 days, apply cuts **in this order** until it fits, all decided before
any test evaluation: (1) run the neural jobs on a GPU (Google Colab via
`google-colab-cli`, no design change); (2) S_max 3000 → 1500, allowed only if
the pilot shows inner-validation MSE at step 1500 within 1% of step 3000;
(3) the 10^6 subset from 12 to 6 routes; (4) replicates 3 → 2 for all neural
cells (the unit stays the route); (5) primary set restricted to CON3 (≈31
routes), which also changes power and must be logged. Hard ceiling: 7 days
wall.

### 7.3 Pilot (CON3-0, its 4 routes, dev set, replicate 0, N ∈ {10^3, 10^4})

The pilot evaluates only on dev, never on test. It must pass all of these
before the sweep:

- **P1** all tests green, including M6 parity and the existing GOLD test
  (`baton` = 83.073, `oracle` = 75.37).
- **P2** determinism: rerunning one cell gives `np.array_equal` test costs.
- **P3** SCALAR at N = 10^4 has pooled dev cost (sum over the 4 routes) within
  [−1%, +3%] of BATON-eq at 10^4. Below −1% means suspected leakage (same
  information set); above +3% means training is broken.
- **P4** on every route and every neural cell: C_oracle ≤ C_π ≤ C_none + 0.5%
  of C_none.
- **P5** wall-clock per job recorded; section 7.2's projection recomputed.
- **P6** round-count rule (section 2.3) and S_max rule (cut 2) evaluated and
  logged.

If P3 fails, at most 3 pilot iterations may change optimization
hyperparameters only (lr ∈ {3e-4, 1e-3, 3e-3}, S_max, batch). Each iteration
is logged, and dev data only is used. The features, architecture, margins and
go/no-go table may not change. After 3 failures, stop and redesign
(section 10).

---

## 8. Statistics

1. **Unit of analysis:** the route. For each (route, policy, N), the cost is
   the mean over 20,000 test days, then the mean over replicates. Every
   comparison is paired: same route, same test days, same training paths.
2. **Variance reporting:** report the mean over replicates first, then over
   routes. Also report (a) the median across routes of the between-replicate
   SD of C_π, and (b) the median across routes of the per-day paired SE of
   C_π − C_BATON-eq. Bootstrap CIs (10,000 resamples over routes, percentile,
   seed 20261115) are for between-route uncertainty.
3. **Secondary family** (Holm–Bonferroni at FWER 0.05; two-sided Wilcoxon
   signed-rank on Δ_r):
   - FULL vs BATON-eq at N ∈ {100, 500, 2,000, 10^4} (4 tests);
   - FULL vs SCALAR at {2,000, 10^5} (2);
   - SCALAR vs BATON-eq at {2,000, 10^5} (2);
   - FULL vs BATON-1e6 at 10^5 (1);
   - FULL vs DP-1e6 at 10^5 (1);
   - SCALAR vs DP-1e6 at 10^5 (1).

   That is 11 tests. The 10^6 subset (12 routes) is descriptive only: means,
   CIs, no tests.
4. **Effect sizes** for each comparison: mean and median Δ_r (%), CI_95,
   win count k/n, matched-pairs rank-biserial correlation, the pooled ratio
   Σ_r C^a_r / Σ_r C^b_r, and mean GC.
5. **Data-size curve:** mean Δ_r (FULL vs BATON-eq) with CI_95 against N, on
   a log axis. The descriptive "crossover N" is the smallest N whose CI_95
   lower bound exceeds −δ. Reported in every branch.
6. **Ties** are judged only by TOST (CI_90 ⊂ (−δ, +δ)). A non-significant
   Wilcoxon test alone never supports "equivalent".
7. **Strata:** the primary table is repeated per stratum (medium, long),
   descriptively.

---

## 9. Implementation plan (modules for the implementer)

All new code goes in `code/transvrp/expa/` (with `__init__.py`). Tests go in
`code/tests/test_expa_<module>.py`. Run with `python -m pytest tests -q`
from `code/`. Neither `transvrp/baton/otr2.py` nor `costs.py` is ever edited.
Use float64 numpy for simulation and float32 torch for the network. Each
module ships with its test, and the test passes before the next module starts.

**M0 — copies.** Copy `svrpspd_wdro/core/dp_exec.py` and
`svrpspd_wdro/core/otr21.py` verbatim to `code/transvrp/baton/` and record the
commit in `PROVENANCE.md`. Add `code/requirements-expA.txt`.
*Test:* `from transvrp.baton.dp_exec import fit_dp`; on CON3-0 route 0 with
`route_problem` data, `fit_dp` returns keys == set(range(1, m)); existing
`test_parity.py` still passes.

**M1 — `expa/paths.py`.**
```python
@dataclass(frozen=True)
class RouteSpec: inst: str; route_id: int; route: list[int]; m: int; B: float
    H: np.ndarray; E: np.ndarray; mu_p: np.ndarray; mu_d: np.ndarray  # route order, shape (m,)
    E0: float
@dataclass
class Paths: p: np.ndarray; d: np.ndarray   # (N, m) float64; property g = p - d
def route_specs(inst: str, plan_key: str = "Det") -> list[RouteSpec]  # skips empty routes, same order as route_problem
def draw_instance(inst: str, purpose: str, rep: int = 0, n_chunks: int = 1) -> tuple[np.ndarray, np.ndarray]  # (dsc, psc), purposes: train/dev/test/test_std
def route_paths(spec: RouteSpec, dsc: np.ndarray, psc: np.ndarray, N: int | None = None) -> Paths  # first N rows
```
Seeds and sizes follow section 5 exactly. Cache with `functools.lru_cache` on
(inst, purpose, rep, n_chunks).
*Tests:* (a) for every route of CON3-0, `route_paths(spec, *draw_instance("CON3-0","test_std")).g`
`np.array_equal` the `g_test` of `route_problem("CON3-0")`, and B, H, E are
equal; (b) nesting: train rep 0, n_chunks=10, first 100,000 rows ==
n_chunks=1 rows (run on CON3-0, one route; mark `@pytest.mark.slow`);
(c) train rep 0 ≠ rep 1 ≠ dev (arrays differ).

**M2 — `expa/features.py`.**
```python
G, F, T_MAX = 12, 11, 24
HISTORY_IDX = (1, 7, 8, 9, 10, 11)        # g2, g8..g12 (0-based)
class RouteCache:                          # precomputed per (spec, Paths): cumulative W, P, D, peak, z-sums/counts
    def __init__(self, spec: RouteSpec, paths: Paths): ...
def build_batch(cache: RouteCache, s_idx: np.ndarray, k: np.ndarray, arm: str
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # glob (n,G), tok (n,T_MAX,F), pad_mask (n,T_MAX) bool
```
`k` is 1-based, with 1 ≤ k ≤ m−1 per sample. The function is vectorized: no
Python loop over samples (a loop over the T_MAX token slots is allowed).
*Tests:* (a) shapes and dtype float32; (b) no peeking: changing p, d at stops
> k leaves the outputs for epoch k bit-identical; (c) `pad_mask.sum(1) ==
T_MAX - (m-k)`; (d) SCALAR: two paths with equal W_k but different (p, d)
histories give identical outputs, while FULL gives different ones;
(e) hand-computed g10 and t11 on a 3-stop toy route match within 1e-6.

**M3 — `expa/targets.py`.**
```python
ChatFn = Callable[[int, np.ndarray], np.ndarray]   # (k 1-based, path indices) -> C_hat
def ls_targets(g: np.ndarray, B: float, H: np.ndarray, E: np.ndarray,
               chat_fn: ChatFn | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # s_idx, k, y over alive pairs
```
This implements section 2.2 (`chat_fn=None` means reactive). It uses the
peak breach time via `otr2._overflow_step`.
*Tests:* (a) **LS parity:** with `cm = fit_lsm_general(g, …)` on CON3-0
(`route_problem` n_train=500) and `chat_fn = baton_chat_fn(cm, g)` (M6), refit
`IsotonicRegression(increasing=True, out_of_bounds="clip")` per k on
(W_k, y_k) and assert its predictions on a 50-point grid equal
`cm[k].predict` (`np.allclose`, atol 1e-9) for all k; (b) every y is in
{0} ∪ set(H) ∪ set(E); (c) every returned pair satisfies τ_s > k;
(d) `chat_fn=None` gives a mean y at k=1 equal to the mean realized `none`
cost over paths alive at 1.

**M4 — `expa/model.py`.**
```python
@dataclass
class NetConfig: d: int = 64; layers: int = 2; heads: int = 4; ff: int = 128
class ContinuationNet(nn.Module):
    def forward(self, glob: Tensor, tok: Tensor, pad_mask: Tensor) -> Tensor  # (n,) in (0,1)
```
*Tests:* (a) 5e4 ≤ n_params ≤ 2e5; (b) permuting the valid tokens of a
sample changes the output by < 1e-5; (c) works with exactly one valid token;
(d) changing padded token values does not change the output.

**M5 — `expa/train.py`.**
```python
@dataclass
class TrainConfig: rounds: int = 3; lr: float = 1e-3; batch: int = 512; s_max: int = 3000
    s_min: int = 100; epochs_factor: int = 20; n_ckpt: int = 8; clip: float = 1.0
class NeuralPolicy:
    nets: list[ContinuationNet]; spec: RouteSpec; arm: str
    def chat_fn(self, paths: Paths) -> ChatFn          # mean over nets of E0*u; batches of 65,536
def train_policy(spec: RouteSpec, paths: Paths, arm: str, rep: int, N: int,
                 cfg: TrainConfig) -> tuple[NeuralPolicy, dict]   # dict: per-round val MSE, steps, seconds
```
This implements sections 2.3 and 4.4: split, halves, cold starts, checkpoint
selection, and deploying the last two nets.
*Tests* (tiny: CON3-0 route 1, N=200, s_max=30, s_min=10): (a) two calls give
`np.array_equal` chat values on 100 dev paths; (b) a different rep gives
different values; (c) round-0 training MSE at the last step < at step 0;
(d) the policy uses only `paths` rows passed in (assert the split sizes
80/20 and halves disjoint).

**M6 — `expa/simulate.py`.**
```python
def simulate_policy(g: np.ndarray, B: float, H: np.ndarray, E: np.ndarray,
                    chat_fn: ChatFn, H_bill: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]     # costs (N,), action int8 (0/1/2)
def baton_chat_fn(cont_models: dict, g: np.ndarray) -> ChatFn  # cont_models[k].predict(cumsum(g)[idx, k-1])
```
The control flow mirrors `_simulate_costs_general` line by line. For
`baton_chat_fn`, a missing key k means "no decision at k": return an array of
`-inf` so the trigger never fires.
*Tests:* (a) **parity:** for all 4 CON3-0 routes (`route_problem`,
n_train=500), `simulate_policy(g_test, …, baton_chat_fn(cm, g_test))` equals
`_simulate_costs_general(g_test, …, cm)` exactly for costs and actions
(`np.array_equal`), also with the `none` setting (H=1e12, H_bill=H);
(b) costs ≥ `oracle_costs_general` per route in mean.

**M7 — `expa/baselines.py`.**
```python
def classical_ladder(spec: RouteSpec, g_fit: np.ndarray, g_eval: np.ndarray,
                     which: tuple[str, ...]) -> dict[str, dict]  # label -> {mean_cost, handoff_rate, fail_rate, costs}
```
Labels follow section 6. `baton_1e6`/`dp_1e6` are produced by calling it with
the 10^6 paths.
*Test:* with `route_problem("CON3-0")` data, the sum over routes of
`baton.mean_cost` rounds to 83.073 and `oracle` to 75.37.

**M8 — `expa/eligibility.py`.**
`def eligible_routes(insts: list[str]) -> pd.DataFrame` (columns inst,
route_id, m, stratum, dev_breach_rate, eligible, in_1e6_subset). It writes
`code/results/expA/eligible_routes.csv` and refuses to overwrite.
*Test:* deterministic on CON3-0 (two calls equal); `eligible` follows the
rule; the subset rule gives 6+6 on the full list (`slow`).

**M9 — `expa/sweep.py`.** CLI:
`python -m transvrp.expa.sweep --stage {smoke,pilot,main,big} --eval {dev,test}
[--frozen-commit HASH] --jobs 4 --out results/expA/<stage>.csv`.
Output is one CSV row per (inst, route_id, m, stratum, N, rep, policy, arm),
with columns mean_cost, handoff_rate, fail_rate, gap_closed, n_eval,
fit_seconds, eval_set, commit. The sweep is resumable (it skips existing
keys). Workers are processes with `torch.set_num_threads(1)`. Per-route
per-day cost arrays are saved as `.npz` under `results/expA/costs/` for the
paired-SE statistic.
*Tests:* `smoke` (1 route, N=100, s_max=20) writes a CSV with all required
columns and the 9 ladder labels (minus `otr21_eq` at N=100); `--eval test`
without `--frozen-commit` exits non-zero.

**M10 — `expa/analysis.py`.**
```python
def route_table(csv: str) -> pd.DataFrame            # mean over reps per (route, N, policy)
def holm(pvals: list[float]) -> list[float]
def bootstrap_ci(x: np.ndarray, level: float, B: int = 10000, seed: int = 20261115) -> tuple[float, float]
def go_no_go(df: pd.DataFrame, delta: float = 1.0) -> dict  # branch in {"WIN","TIE_STRONG","TIE_WEAK","LOSE"} + all numbers
```
It writes `results/expA/summary.md` and `results/expA/curve.csv`.
*Tests:* synthetic tables over 60 routes: neural = BATON·0.97 gives WIN;
BATON·(1 + N(0, 0.002)) gives TIE_STRONG; BATON·1.03 gives LOSE;
BATON·(1 + N(0, 0.03)) gives TIE_WEAK; `holm([0.01, 0.04, 0.03])` equals
`[0.03, 0.06, 0.06]`.

**M11 — `expa/boosted_pd.py` (run only in the WIN branch).** This is OTR-2.1's
`HistGradientBoostingRegressor` recipe (same hyperparameters, monotone in
W_k), trained on the FULL global features g1–g12 via the same LS recursion as
`fit_lsm_rich`. It decides whether the transformer is needed.
*Test:* on CON3-0 at N=2,000, the dev cost is between oracle and none on
every route.

---

## 10. Threats to validity, pitfalls, stop rules

- **Peak vs endpoint labels.** Every target and breach uses the running-peak
  τ (`_overflow_step`). An endpoint label silently zeroes targets on
  collect-then-deliver routes (BATON v1's bug). M3(c) guards this.
- **Optimism bias.** Handled by alternating halves (section 2.3). It is
  residual and second-order. Test days are always fresh, so reported costs are
  unbiased for the deployed policy.
- **Scale of data.** At 10^5–10^6 paths, BATON is close to the exact (k, W)
  conditional mean, and DP-1e6 is near-optimal in that class. A neural win
  must therefore come from information, not from fitting better, and H3 checks
  this. Conversely, at 10^5 the neural is compute-limited (≤ 3 epochs), so a
  tie may reflect the step cap; the 10^6 subset and the cut-2 check bound this.
- **H_bill vs decision H.** Here H_dec = H_bill = H for all learned policies;
  only `none` decides with 1e12. If a standby shadow price λ is later added,
  the features (g6, t7), the targets and the stopping rule must all use H_dec,
  and billing must use H_bill; mixing them double counts the retainer.
- **ρ = 0.6 makes W_k informative about the future.** The common factor
  already raises E[future g | high W_k], so the scalar state captures much of
  the factor. The FULL increment comes only from separating the pickup and
  delivery factors and from heterogeneous means. Under ρ = 0 the history adds
  nothing beyond W_k; under a single shared copula for p and d, the split
  matters less. Results are specific to this generator; this is stated as a
  limitation, and an optional sensitivity at ρ ∈ {0, 0.3} on the 12-route
  subset is allowed after the primary result.
- **Small routes.** Routes with m ≤ 6 never breach, so every policy ties
  there. They are excluded by the eligibility rule, not by results.
  Medium/long strata are reported separately.
- **The oracle is a lower bound, not a target.** It uses the whole future
  path. GC < 1 is expected for every online policy, and GC values are not
  comparable to a "fraction of optimal".
- **Separation from BATON (PROJECT.md section 6).** Neural targets come from
  realized costs under the neural's own policy. BATON's fitted values are
  never inputs or targets; `baton_chat_fn` exists only for parity tests.
- **Per-route training.** The set encoder's ability to share across routes is
  not tested in v1 (section 3 note).
- **Single cost model.** `LastMileCosts` defaults are the only setting tested;
  that is a stated limitation.

**Stop and redesign if any of these occur:**
1. The M6 parity or GOLD tests fail.
2. Any policy's mean cost is below the oracle on any route (bug).
3. SCALAR beats DP-1e6 or BATON-1e6 by > 0.5% pooled on dev. This is
   impossible within the same information set beyond noise, so it signals
   leakage.
4. The pilot still fails P3 after 3 iterations.
5. Fewer than 40 eligible routes.
6. The projected wall-clock exceeds 7 days after all section 7.2 cuts.
7. Inner-validation MSE of the deployed nets rises over rounds on > 25% of
   pilot cells, meaning the policy iteration does not settle.

---

## 11. Out of scope for v1

- The three-action version {continue, handoff, depot-restock}
  (`fit_lsm_actions`/OTR-A). This is the first extension after a WIN, and the
  restock reset makes F_k state-dependent, which needs its own design.
- A network shared across routes and instances (amortization, held-out-route
  generalization).
- LLM weight-space editing (direction D(ii)) and LLM heuristic design (D(i)).
- OR-Transformer-style pathwise or score-gradient training (direction B).
- Any change to planning (ALNS, gates, plans), cost parameters, the generator,
  or the 'SAA'/'WDRO'/other gate plans; only `Det` is used.
- Policy-gradient RL baselines (BATON's RESULTS_OTR2 section 9 already
  reports one).
- City and Salhi–Nagy instances. They are not in the copied data.
