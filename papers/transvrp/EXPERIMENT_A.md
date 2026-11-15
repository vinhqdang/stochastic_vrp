# EXPERIMENT A — learned continuation values vs BATON (two-action handoff)

Status: pre-registration, revision 2 (2026-11-15), **before any neural run**.
Revision 1 was reviewed in `REVIEW_A.md`; the mapping is in section 12. After
the pilot (section 7.3) passes, sections 1, 2 and 8, the eligible-route list
and δ are frozen. Any later change goes into `RESEARCH_LOG.md` with its date
and reason, and the paper reports it as a protocol deviation.

**[DC]** marks a design choice made without evidence. Only the pilot rules
that name it may change it.

"Cost" means **recourse cost** throughout: handoff H, emergency E, 0 on
completion, as in `benchmark.py`. It excludes the fixed fleet and travel cost.
A 1% recourse-cost change is a much smaller share of total route cost.

Facts measured for this revision. They use the copied code with read-only
Python, the **dev** stream (5,000 days, section 5) and BATON only; no neural
policy exists yet.
- 80 'Det' routes over 20 instances, m ∈ {1..20, 22}.
- 13 routes have m ≤ 6; 11 of them never breach (the review's count on 20,000
  days, not re-measured on dev).
- 62 routes are eligible (m ≥ 4, breach ≥ 2%), from all 20 instances:
  18 instances have 3 eligible routes and 2 have 4.
- BATON between-replicate relative SD of route cost (3 independent training
  draws): median 0.61% / p90 2.3% at N = 500; 0.21% / 0.56% at 10^4;
  0.13% / 0.44% at 10^5.
- SD over the 20 instance means of Δ between two independent BATON fits at
  10^5 (rep pairs 0-1, 0-2, 1-2): 0.14%, 0.27%, 0.23%.
- BATON at N = 500 is 2.2% dearer than at 10^5 (instance mean, SD 1.39%; worse
  on 20/20 instances). At 10^4 it is 0.24% dearer (SD 0.20%).
- BATON at 10^5 lies 17.1% above the clairvoyant oracle (median over routes;
  range 6.3-28.9%).
- The PyTorch CPU wheel (torch 2.15.0+cpu, cp313) downloads through the
  sandbox proxy. The environment runs Python 3.13.

---

## 1. Hypotheses, unit of analysis, pre-registered go/no-go

### 1.1 Hypotheses

- **H1 (main).** At N* = 10^5 simulated paths per route, a neural continuation
  model over the richer online state (arm FULL, section 3) has lower recourse
  cost than BATON (v2_lsm) fitted on the same N* paths. The object tested is
  "richer online state + a neural regressor + this training procedure". It is
  not "a transformer": within one route the encoder sees a fixed token set per
  k, so it is a fixed-input network (section 3.4). "Equal data" means equal
  paths. Each neural fold net is fitted on 40% of N; BATON uses 100%
  (section 2).
- **H2 (information).** Any H1 gain comes from information BATON discards:
  FULL beats the same network fed only (k, W_k) (arm SCALAR).
- **H3 (competence).** SCALAR is equivalent to the plug-in DP at 10^6 paths,
  the best available policy in BATON's information class. If it is not, a
  FULL loss is a pipeline deficit, not evidence about the state.
- **H0 (prior; RESULTS_OTR2.md section 8).** Richer features lost to scalar
  isotonic at 1,000 paths on 23/25 routes. Those features were net-g
  residuals, last increment and running max; they did not include the p/d
  split used here.

Mechanism (not evidence): deliveries and pickups come from **two independent
copulas**, each with ρ = 0.6 over all customers of the instance. W_k = P_k − D_k
mixes the two factors. P_k and D_k separately predict the remaining pickups
and deliveries. The driver observes them online (`otr_route_v2` receives
`(d_k, p_k)`). Any gain is specific to this generator (section 10).

### 1.2 Unit of analysis: the instance

The routes of one instance share the training, dev and test days and the
common demand factors, so their Δ values are correlated. **The unit is the
instance (n = 20).** For a contrast "a vs b", define per route

Δ_r(a vs b) = 100 · (C_b,r − C_a,r) / C_b,r (positive means a is cheaper),

where C is the mean over 20,000 test days, then over replicates. Let Δ_i be the
mean of Δ_r over the eligible routes of instance i, and Δ̄ the mean of Δ_i over
the 20 instances. The 95% CI (CI95) is Δ̄ ± t_{19,0.975} · s/√20 (t = 2.093)
and CI90 uses t_{19,0.95} = 1.729, where s is the SD of {Δ_i}. A cluster
bootstrap (resample instances, keep their routes; 10,000 draws, seed 20261115)
is reported as a sensitivity check and is not used for decisions [DC]. The
test is the exact Wilcoxon signed-rank on {Δ_i}, n = 20. Route-level numbers
and "X/Y routes" are descriptive only.

### 1.3 Margin δ = 0.7% of BATON recourse cost (frozen now)

δ comes from refitting noise, not from the oracle gap. The SD of instance-mean
Δ between two independent BATON fits at N* is s_rep. On dev, the RMS over the
three replicate pairs is 0.22%. δ is the larger of 0.5% and 3 · s_rep,
rounded up to 0.1%, which gives 0.7% [DC on the factor 3 and the 0.5% floor].
A difference within ±δ is therefore within about three refitting-noise SDs of
BATON itself. It is also about a third of what BATON gains from 500 to 10^5
paths (2.2%).

Feasibility: the CI90 half-width is 0.387 · s. A strong tie at Δ̄ ≈ 0 needs
s < 1.8%. The instance SD of BATON@500 vs BATON@10^5 (very different fits) is
1.39%, so a strong tie is attainable but not guaranteed. The pilot reports s
for FULL vs BATON-eq on CON3-0 for information only; δ does not change.

### 1.4 Go/no-go table (primary: FULL vs BATON-eq at N* = 10^5)

Rules are applied in this order and are exhaustive. Each direction is tested
at α = 0.025, Bonferroni over two directions. Neither the margin nor the
primary test is adjusted further; secondary tests use Holm (section 8).

| branch | rule | claim template (filled from the data; nothing asserted in advance) |
|---|---|---|
| **WIN** | Δ̄ ≥ δ **and** CI95 lower > 0 **and** Wilcoxon (FULL cheaper) p < 0.025 **and** Δ_i > 0 on ≥ 13/20 instances | "At 10^5 paths, FULL lowers recourse cost by Δ̄% [CI95] versus BATON at equal paths (n = 20 instances; k/20 better)." It must be followed by the attribution sentences of 1.5 (H2, ARCH, M11). |
| **LOSE** | Δ̄ ≤ −δ **and** CI95 upper < 0 **and** Wilcoxon (FULL dearer) p < 0.025 **and** Δ_i < 0 on ≥ 13/20 instances | "FULL is Δ̄% [CI95] dearer than BATON." If H3 is "worse": "…which is attributable to a pipeline deficit of x% (SCALAR vs DP-1e6)". Otherwise: "…although the pipeline matches the scalar optimum, so the richer state does not pay here." |
| **TIE** | neither | **TIE_STRONG** iff CI90 ⊂ (−δ, +δ): "FULL and BATON are equivalent within ±δ [CI90]". If the Wilcoxon test is also significant at 0.025 in one direction, add "although FULL is significantly better/worse by Δ̄%". **TIE_WEAK** otherwise: "No difference of ±δ was established; the comparison is inconclusive (CI95 …)". |

The WIN and LOSE rules deliberately combine significance against 0 with a
point estimate of at least δ ("significant and practically relevant"). They do
not test superiority by margin. The data-size curve (section 8) is reported in
every branch. What follows: WIN opens section 11's extensions. TIE and LOSE
are written up as a negative result, and direction B is not opened
(PROJECT.md section 4).

### 1.5 Attribution rules (secondary, pre-registered)

Every contrast is classified at its stated N by the same four-way rule:

- **better** iff Δ̄ ≥ δ, CI95 lower > 0 and Holm-adjusted Wilcoxon p < 0.05;
- **worse** is the mirror image;
- **equivalent** iff CI90 ⊂ (−δ, δ);
- **inconclusive** otherwise.

If the first and third both hold, "better" or "worse" wins.

| id | contrast @ N | result → claim |
|---|---|---|
| H2 | FULL vs SCALAR @ 10^5 | better → "the gain is information"; equivalent → "the richer state carries no usable information here"; worse or inconclusive → reported as such. If WIN holds but H2 is not "better", the claim becomes "the gain comes from the regressor/procedure within BATON's information". In that case S10/S11 must also be "better", or the WIN is called fragile. |
| H3 | SCALAR vs DP-1e6 @ 10^5 | equivalent → "the pipeline reaches the (k, W) optimum"; worse → "pipeline deficit"; better → leakage audit (stop rule 3), with the claim held back until the audit is clean |
| ARCH | FULL vs MLP @ 10^5 | better → the paper may say "set encoder"; otherwise it never says set encoder or transformer, only "neural regressor" |
| M11 | FULL vs BOOST (gradient-boosted trees on the FULL globals) @ 10^5, run only in WIN; its own α = 0.05, outside Holm | better → "a neural regressor is needed"; equivalent → "trees on the same state suffice" (the paper's claim is then about state, not networks) |

---

## 2. Problem statement and training target

### 2.1 Two-action stopping problem (one route)

- Stops are j = 1..m, with deliveries d_j and pickups p_j, g_j = p_j − d_j, and
  W_k = Σ_{j≤k} g_j.
- B = Q − Σ_j d̄_j, and the breach time is τ = min{k : W_k > B}, with m+1 if
  there is no breach. This is the running peak, from `otr2._overflow_step`;
  the endpoint is never used.
- H and E come from `route_cost_schedules` (0-indexed).
- Decision epochs are after stop k, for k = 1..m−1, on paths with τ > k.
- The information is F_k = σ(p_1, d_1, …, p_k, d_k) plus route data and the
  training paths.
- Cost: H[σ−1] if the policy stops at some σ < τ; otherwise E[τ−1] if τ ≤ m;
  otherwise 0.
- Rule (BATON's, unchanged): hand off at k iff Ĉ_k(x_k) > H[k−1], where Ĉ_k
  estimates the expected recourse cost of continuing at k and following the
  policy afterwards.

### 2.2 Target: Longstaff–Schwartz backward induction, exactly BATON's recursion

**Decision:** use realized-cost LS regression with BATON's single backward
pass over k, not fitted value iteration and not policy iteration.

Reasons:
- BATON's estimator is unchanged except for the regressor, so the comparison
  isolates state and function class.
- One backward pass is exact given the regressors; there is no convergence
  question.
- Targets are realized costs in {0} ∪ {H} ∪ {E}. They never bootstrap on Ĉ at
  k+1, so regression errors affect only decisions near Ĉ ≈ H.

Generic driver (M3). Initialize `future[s] = E[τ_s−1]` if τ_s ≤ m, else 0.
Then for k = m−1 down to 1:
1. Fit the regressor(s) for epoch k on the alive rows (τ_s > k) with target
   `future`.
2. Set `future[s] = H[k−1]` on the alive rows where the predicted Ĉ_k > H[k−1].

With an isotonic regressor on W_k and a single fold, this *is*
`fit_lsm_general`; this is tested in M3.

### 2.3 Neural analogue with cross-fitting (optimism control)

BATON fits model k and uses it to stop the same paths. That is harmless for
isotonic regression at N ≥ 500, but not for a network with ~10^5 parameters.
The neural version uses two folds [DC]:

- **Split:** of the N paths, the first ⌊0.8N⌋ are fitting rows and the last
  ⌈0.2N⌉ are inner-validation rows. Fitting rows split into fold A (even
  index) and fold B (odd index).
- **At each k (backward):** fit net_k^A on A's alive rows and net_k^B on B's,
  each with its own `future` targets.
- **Stop update:** A rows use net_k^B, B rows use net_k^A, and validation rows
  use the mean ½(net_k^A + net_k^B).
- No net ever decides on rows it was fitted on. M3 logs index sets per k and
  asserts disjointness.
- **Deployed:** Ĉ_k = ½(net_k^A + net_k^B).
- Known inconsistency [DC]: the targets of fold A at k reflect later decisions
  by fold-B nets, while deployment uses the mean. Its size is logged as the
  fraction of alive rows on which net_k^A and net_k^B decide differently, per
  stage.
- **Warm start:** net_k^f is initialized from net_{k+1}^f. net_{m−1}^f starts
  from scratch.
- **Few alive rows:** if a fold has fewer than 20 alive rows at k, there is no
  training and net_k^f = net_{k+1}^f. At k = m−1 the net is replaced by a
  constant predictor (mean target, or E[k] if no row is alive), mirroring
  BATON's `_ConstantModel` [DC].
- **Early stopping is per stage:** validation targets at k are built under the
  deployed later policy and are fixed within the stage. The validation MSE is
  therefore comparable across a stage's checkpoints. It is never compared
  across stages.

Reported costs are on fresh test days, so leftover optimism can only hurt the
neural policy; it cannot flatter it.

---

## 3. State features (exact)

### 3.1 Empirical route statistics (no generator knowledge)

μ̂p_j, μ̂d_j, σ̂p_j and σ̂d_j are the sample mean and SD (ddof = 1) of p_j and
d_j over the job's **fitting rows** (80% of N). They are stored with the model
and reused at evaluation. The neural arms never receive the generator's d̄, p̄
or CV. Everything is learned from the same N paths BATON gets, which removes
revision 1's "model knowledge" confound. Plan quantities that BATON also uses
(B from the planned load, H, E) are route data. μ̂g_j = μ̂p_j − μ̂d_j. E0 = E[0],
the largest price on the route. Every feature is clipped to [−5, 5] after
scaling and stored as float32.

### 3.2 Global vector (G = 12)

H marks history features. They are zeroed in SCALAR.

| # | feature | arm |
|---|---|---|
| g1 | W_k / B | all |
| g2 | max_{j≤k} W_j / B | H |
| g3, g4, g5 | k/m, (m−k)/m, m/22 | all |
| g6, g7 | H[k−1]/E0, E[k−1]/E0 | all |
| g8, g9 | P_k/B, D_k/B | H |
| g10 | z̄p_k = mean over {j ≤ k : σ̂p_j > 0} of (p_j − μ̂p_j)/σ̂p_j; 0 if the set is empty | H |
| g11 | z̄d_k, the same for deliveries | H |
| g12 | (p_k − d_k)/B | H |

### 3.3 Token per remaining stop j = k+1..m (F = 11), all arms

| # | feature |
|---|---|
| t1–t4 | μ̂p_j/B, μ̂d_j/B, σ̂p_j/B, σ̂d_j/B |
| t5, t6 | (j−k)/m, j/m |
| t7 | H[j−1]/E0 if j < m, else 0 |
| t8 | E[j−1]/E0 |
| t9 | 1[j = m] |
| t10 | (W_k + Σ_{i=k+1..j} μ̂g_i)/B |
| t11 | (B − W_k − Σ_{i=k+1..j} μ̂g_i) / max(ŝ_j, 1e-6·B), with ŝ_j = sqrt(Σ_{i=k+1..j}(σ̂p_i² + σ̂d_i²)) (an independence approximation, used only as a feature) |

### 3.4 Arms and what they isolate

- **FULL:** all features.
- **SCALAR:** history features zeroed. Every remaining input is a function of
  (k, W_k), route data and the training sample, which is BATON's information
  set. FULL − SCALAR measures information.
- **MLP:** the same information as FULL. Input = the 12 globals plus 3
  route-ahead summaries: min_j t11, (argmin_j t11 − k)/m, and
  min_{j>k, j<m} H[j−1]/E0 (0 if none). Network: 3 × Linear(·, 128) with GELU,
  then Linear(128, 1) and sigmoid; about 35k parameters. FULL − MLP measures
  the set encoder.

Within one route, the token set at k is a deterministic function of k (and
t10/t11 of (k, W_k)). With per-k nets, every stage sees a fixed number of
tokens, so the "set encoder" is a fixed-input network. Only ARCH (1.5) can
license the words "set encoder".

---

## 4. Architecture, loss, optimization

### 4.1 Set encoder (FULL, SCALAR) [DC on all sizes]

- **Inputs:** Linear(12→64) for the global token; Linear(11→64) per stop
  token. The sequence is [global, tok_{k+1..m}]. At stage k every sample has
  exactly m−k stop tokens, so **there is no padding**: all rows are real.
- **Order:** no positional encoding; order enters only via t5/t6.
- **Encoder:** 2 × `TransformerEncoderLayer(d_model=64, nhead=4,
  dim_feedforward=128, dropout=0.0, activation="gelu", batch_first=True,
  norm_first=True)`.
- **Head:** concat(h_global, mean of token outputs), then Linear(128→64),
  GELU, Linear(64→1), sigmoid u; Ĉ = E0 · u.
- **Size:** 76,865 parameters (hand count in the review; acceptance range
  5·10^4 to 2·10^5). For deployment, a route stores 2 folds × (m−1) nets.

### 4.2 Loss and monotonicity

- **Loss:** MSE of u against y/E0 on alive rows of stage k, with targets in
  [0, 1]. H, E ≤ E0 was verified on all 80 routes; assert it at runtime.
- **Monotonicity:** no monotone constraint in any neural arm or in BOOST. In
  FULL, "monotone in W_k with the other features fixed" is incoherent, since
  g2 and g8–g12 move with W_k. All arms share the same unconstrained class.
- **Consequence:** the neural arms lose the free regularization of isotonic
  regression, which is visible at small N.
- **Diagnostic:** for SCALAR at N*, the fraction of (route, k) on a 101-point
  W grid in [min W_k, B] where Ĉ decreases by more than 0.01·E0 between
  neighbours.

### 4.3 Optimization per stage [DC; pilot-tunable within 7.3]

- **Optimizer:** Adam with batch 512 and gradient-norm clip 1.0. lr is 1e-3 at
  the cold stage k = m−1 and 3e-4 at warm stages. The lr follows a cosine
  decay to 10% within each stage.
- **Steps:** S_k = clip(⌈20 · n_k / 512⌉, 50, 400), where n_k is the fold's
  alive rows at k. The cold stage uses a floor of 200 instead of 50. The cap
  binds from n_k ≈ 10^4, i.e. N ≳ 3·10^4.
- **Checkpoints:** the warm-start weights (step 0) plus every ⌈S_k/4⌉ steps.
  Keep the one with the lowest stage-k validation MSE; ties go to the earlier.
  There is no early termination.
- **Minibatches:** drawn uniformly with replacement by a seeded
  `numpy.random.Generator`.
- **Determinism:** `torch.manual_seed`,
  `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`.
- **Seed:** `zlib.crc32(f"{inst}|{route_id}|{N}|{rep}|{fold}|{k}".encode())`.
  It deliberately excludes the arm, so FULL, SCALAR and MLP share init and
  batch order, and H2/ARCH are paired comparisons.
- **Install:** `pip install torch --index-url
  https://download.pytorch.org/whl/cpu`. Pin the installed versions (torch,
  numpy, scipy, scikit-learn, pandas, pytest) in `code/requirements-expA.txt`.

---

## 5. Data, leakage, fit/eval separation

**Streams.** base = `data.stable_seed(name)`; all streams go through
`gen_scenarios(dbar, pbar, n, seed)`:

| purpose | seed | size | use |
|---|---|---|---|
| train, replicate r ∈ {0,1,2}, chunk c ∈ {0..9} | `[base, 1, r, c]` | 100,000 per chunk | fitting |
| dev | `[base, 3]` | 5,000 | eligibility, pilot, δ, all debugging |
| test | `base + 99991` | 20,000 | final evaluation only |
| test_std | `base + 99991` | 2,000 | parity with `route_problem` / GOLD only |

**Prefixes and nesting.** `gen_scenarios` draws deliveries then pickups from
one stream, so a call with a smaller n is not a prefix of a larger one.
Training sets are therefore always prefixes of fixed 10^5 chunks. N ≤ 10^5 uses
the first N rows of chunk 0; N = 10^6 uses chunks 0..9. Sizes are nested within
a replicate.

**Memory.** Generate one chunk at a time, keep only route columns (float64),
and discard the instance arrays. A 10^6 route pool is ≤ 0.35 GB per array, and
the peak is about 0.4 GB plus one chunk's temporaries. Data generation takes
≈ 6 s per chunk per instance.

**Grid.**

| arm | N | replicates | routes |
|---|---|---|---|
| FULL | 100, 500, 2,000, 10^4, 10^5 | 3 | all eligible |
| SCALAR | 2,000, 10^5 | 3 | all eligible |
| MLP | 10^5 | 3 | all eligible |
| FULL, SCALAR | 10^6 | rep 0 | 12-route subset (descriptive) |
| BOOST | 10^5 | 3 | all eligible, WIN only |

**Equal paths.** At every (route, N, r), all classical and neural methods
receive the same N paths. BATON uses all N. Neural fold nets each use 40%, and
their deployed average uses 80% plus 20% for validation.

**"BATON given everything".** `baton_1e6` and `dp_1e6` are fitted on 10^6
paths for every eligible route and replicate. At large N, BATON's isotonic fit
approaches the exact (k, W) conditional mean, a different regime from the
repo's N = 500 default, which the `fit_lsm` docstring calls noisy below 500.
BATON-eq at N* is the strong BATON, not the paper's.

**Fit/eval separation.** `sweep fit` saves every fitted object to
`results/expA/models/`: neural state dicts per (fold, k), route statistics, and
pickled classical models, each with the fit commit. `sweep eval` only loads and
simulates. `--eval test` requires `--frozen-commit HASH`, which must be
recorded in RESEARCH_LOG.md. Rows carry both fit and eval commits. Evaluation
can be rerun without refitting.

---

## 6. Baseline ladder

Per route, per N, per replicate:

| label | definition | source |
|---|---|---|
| none | never hand off: decide with H = 1e12, bill H | `benchmark.run_baselines` |
| tuned_tau | `fit_otr_peak(g_N, B)` + `tune_tau_general(g_N, B, H, E, models)`, then `simulate_tau_general` | otr2.py, costs.py |
| baton_eq | `fit_lsm_general(g_N, B, H, E)` | costs.py |
| baton_1e6 | the same on 10^6 paths | costs.py |
| dp_1e6 | `fit_dp(g_1e6, B, H, E)` (256 bins) | **copy** `svrpspd_wdro/core/dp_exec.py` |
| otr21_eq | `fit_lsm_rich` at N ∈ {10^4, 10^5}, with mu/sig = empirical μ̂g, sqrt(σ̂p² + σ̂d²) | **copy** `svrpspd_wdro/core/otr21.py` |
| nn_full, nn_scalar, nn_mlp | sections 2–4 | new |
| boost | M11 (WIN only) | new |
| oracle | `oracle_costs_general` (clairvoyant, handoff-only lower bound) | costs.py |

DP-1e6 is a 256-bin piecewise-constant estimate of the optimum in BATON's
information class. It carries a resolution bias at the boundary, so it is a
reference, not a hard bound. Report the mean cost, handoff rate, fail rate and
the gap-closed fraction GC = (C_none − C_π)/(C_none − C_oracle), with GC = NaN
when C_none − C_oracle < 0.01·C_none. H_bill = H except for `none`.

---

## 7. Routes, budget, pilot

### 7.1 Routes

- **Eligibility** (frozen by M8 on dev before any neural run): m ≥ 4 and dev
  peak-breach rate ≥ 2%. On dev this gives 62 routes over 20 instances.
  Ineligible routes (18) are listed with baseline costs and excluded from
  tests.
- **Strata** by m (descriptive): short 4–9 (2 routes), medium 10–14 (28), long
  ≥ 15 (32).
- **10^6 subset:** 6 medium then 6 long routes. Each is drawn with a fresh
  `numpy.random.default_rng(2026)` (medium) or `default_rng(2027)` (long)
  `.choice(len(L), 6, replace=False)` over that stratum's eligible list L,
  sorted by (inst, route_id).

### 7.2 Budget

Per training sample at stage k, the forward pass is ≈ (m−k+1) × 0.131 MFLOP
with no padding; forward plus backward is ≈ 3×. Summed over stages for the
mean eligible route (Σ_k (m−k+1) = 117), a capped job is 2 folds × 400 steps ×
512 × 3 × 0.131 MFLOP × 117 ≈ 1.9·10^13 FLOP, about 21 min at an assumed
15 GFLOP/s single-thread. Add +30% for stop updates, validation and feature
building. Estimated minutes per (route, replicate): N ≤ 2,000 ≈ 3.4,
10^4 ≈ 7.8, 10^5 ≈ 27.

| block | job-minutes | wall @ 4 workers |
|---|---|---|
| FULL, 5 sizes × 186 route-replicates | 8,400 | 35 h |
| SCALAR, 2 sizes × 186 | 5,650 | 24 h |
| MLP, 10^5 × 186 (~1/5 cost) | 1,000 | 4 h |
| 10^6 subset, 2 arms × 12 routes (incl. 10^6-row stop updates ≈ +20 min) | 1,200 | 5 h |
| data generation at 10^6 (600 chunks × 6 s) + classical fits + eval | — | 3 h |
| **total** | | **≈ 71 h ≈ 3 days (CPU estimate, uncertainty ×2–3)** |

The pilot replaces these estimates with measured times (P5). **Platform rule:**
if the measured projection is ≤ 4 days on CPU, run on CPU. Otherwise the GPU
run on Google Colab (`google-colab-cli`) is the plan, with the same code and
determinism except for documented GPU nondeterminism (seed-level replicates
remain).

If Colab is unavailable or the projection still exceeds the ceiling, apply
these cuts in order, all before any test evaluation, each logged:
1. S_k cap 400 → 200, only if the pilot shows stage validation MSE at 200
   within 1% of 400 on ≥ 80% of stages.
2. The 10^6 subset from 12 to 6 routes.
3. SCALAR and MLP replicates from 3 to 2.
4. FULL at N = 100 and 500 to 2 replicates.
5. Primary set restricted to CON3 (10 instances). This changes power (the
   CI95 half-width grows by about √2) and must be logged.

**Hard ceiling:** 7 days wall.

### 7.3 Pilot (CON3-0, 4 routes, dev only)

Cells:
- N ∈ {10^3, 10^4} × replicates {0, 1, 2} × {FULL, SCALAR, BATON-eq}.
- One full-size timing cell: route 3 (m = 17), N = 10^5, FULL, replicate 0,
  all stages.
- One 10^6 timing pass: data generation plus stop updates for route 3 with the
  10^5 nets.

Pass criteria:
- **P1:** all tests green, including M3/M6 parity and GOLD (`baton` 83.073,
  `oracle` 75.37).
- **P2:** rerunning one cell gives bit-identical dev costs.
- **P3 (gate):** SCALAR's pooled dev cost at N = 10^4 (mean over 3 replicates)
  ≤ BATON-eq's × (1 + max(2%, 3 × the pooled between-replicate SD of
  BATON-eq)). This rejects a broken pipeline.
- **P3' (flag, not a gate):** SCALAR cheaper than BATON-eq by > 1% triggers a
  leakage audit of the M3 index logs before the sweep.
- **P4:** on every route and cell, C_oracle ≤ C_π ≤ C_none · 1.005.
- **P5:** seconds per stage and per job are recorded, the section 7.2 table is
  recomputed, and the platform rule and cut 1 are applied.
- **P6:** at N = 10^4, the selected checkpoint is the warm-start (no learning)
  on < 50% of stages. Report the fold-disagreement rate.

If P3 or P6 fails, at most 3 iterations may change only lr (cold or warm stage,
each in {3e-4, 1e-3, 3e-3}), S_k bounds and batch size. Each iteration is
logged and uses dev only. Features, architecture, δ and the rules of sections 1
and 8 are frozen.

Note: CON3-0's 4 routes are also in the primary set. Design was tuned on their
dev days, not their test days. This is logged and stated in the paper.

---

## 8. Statistics

1. **Unit:** the instance (section 1.2). Route cost is the mean over 20,000
   test days, then over replicates. All contrasts are paired: same routes,
   test days and training paths.
2. **Variance reporting:**
   - (a) the median over routes of the between-replicate SD of each policy's
     cost;
   - (b) the median over routes of the per-day paired SE of C_π − C_BATON-eq;
   - (c) s, the SD of {Δ_i}, for every contrast.
3. **Secondary family:** Holm–Bonferroni at FWER 0.05, with two-sided exact
   Wilcoxon on the 20 instance means:
   - S1–S4: FULL vs BATON-eq at N ∈ {100, 500, 2,000, 10^4};
   - S5: FULL vs SCALAR @ 10^5 (H2);
   - S6: FULL vs SCALAR @ 2,000;
   - S7: SCALAR vs DP-1e6 @ 10^5 (H3);
   - S8: SCALAR vs BATON-eq @ 2,000;
   - S9: FULL vs MLP @ 10^5 (ARCH);
   - S10: FULL vs BATON-1e6 @ 10^5;
   - S11: FULL vs DP-1e6 @ 10^5.

   That is 11 tests. Equivalence (CI90 ⊂ ±δ) is reported per contrast without
   multiplicity adjustment [DC]. M11 is outside the family. The 10^6 subset is
   descriptive (means and route-level ranges, no tests).
4. **Effect sizes per contrast:** Δ̄, the median Δ_i, CI95, CI90, the number
   of instances better, the matched-pairs rank-biserial correlation, the
   pooled ratio Σ C_a / Σ C_b, and the mean GC.
5. **Data-size curve:** Δ̄(FULL vs BATON-eq) with CI95 against log N. The
   crossover N is the smallest grid N such that CI95 lower > −δ holds at it
   **and at every larger grid N**; "none" if no such N exists. Reported in
   every branch.
6. **Ties** are judged only by CI90 ⊂ ±δ. A non-significant test never
   supports "equivalent".
7. **Strata** (medium and long) are tabulated descriptively at route level.

---

## 9. Implementation plan

Code goes in `code/transvrp/expa/` (with `__init__.py`). Tests go in
`code/tests/test_expa_<module>.py` and run from `code/` with
`python -m pytest tests -q`; slow tests are marked `@pytest.mark.slow`.

- `transvrp/baton/otr2.py` and `costs.py` are never edited.
- Simulation uses numpy float64; the network uses torch float32.
- Finish each module and its tests before starting the next.

**M0 — copies and requirements.**
- Copy `svrpspd_wdro/core/dp_exec.py` and `otr21.py` verbatim to
  `code/transvrp/baton/`. Both import only `.otr2`.
- Record the source commit in `PROVENANCE.md`.
- Add `code/requirements-expA.txt`.

*Tests:*
- `fit_dp` on CON3-0 route 0 (`route_problem` data) returns keys ==
  set(range(1, m)).
- `test_parity.py` still passes.

**M1 — `expa/paths.py`.**
```python
@dataclass(frozen=True)
class RouteSpec:
    inst: str
    route_id: int
    route: tuple[int, ...]
    m: int
    B: float
    H: np.ndarray   # (m,)
    E: np.ndarray   # (m,)
    E0: float

@dataclass
class Paths:
    p: np.ndarray   # (N, m) float64, route order
    d: np.ndarray
    @property
    def g(self) -> np.ndarray: return self.p - self.d

def route_specs(inst: str, plan_key: str = "Det") -> list[RouteSpec]   # same order as route_problem, empty routes skipped
def route_pool(inst: str, purpose: str, rep: int = 0, N: int | None = None) -> dict[int, Paths]   # route_id -> Paths
```
- `purpose` ∈ {train, dev, test, test_std}. For train, `N` selects the first
  N rows of the chunk sequence (chunks generated one at a time).
- Use `functools.lru_cache(maxsize=1)` on the per-chunk generator only.

*Tests:*
- (a) for all CON3-0 routes, `route_pool("CON3-0","test_std")[rid].g` is
  `np.array_equal` to `route_problem`'s `g_test`, and B, H, E are equal.
- (b) (slow) train rep 0: the first 10^5 rows of N = 10^6 equal N = 10^5.
- (c) train rep 0, rep 1 and dev differ.

**M2 — `expa/features.py`.**
```python
G, F = 12, 11
HISTORY_IDX = (1, 7, 8, 9, 10, 11)                  # g2, g8..g12, 0-based
@dataclass
class RouteStats:                                    # empirical, from fitting rows
    mu_p: np.ndarray; mu_d: np.ndarray; sd_p: np.ndarray; sd_d: np.ndarray
    @staticmethod
    def fit(paths: Paths) -> "RouteStats": ...
class RouteCache:                                    # one per (spec, Paths, stats); holds W, P, D, peak, z-sums/counts
    def __init__(self, spec: RouteSpec, paths: Paths, stats: RouteStats): ...
    g: np.ndarray                                    # the exact rows the cache was built from
def stage_batch(cache: RouteCache, k: int, idx: np.ndarray, arm: str
                ) -> tuple[np.ndarray, np.ndarray]   # glob (n,G), tok (n, m-k, F); arm in {full, scalar}
def mlp_batch(cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray   # (n, 15)
```
The batch builders are vectorized: no Python loop over samples.

*Tests:*
- (a) shapes, float32, all values in [−5, 5].
- (b) no peeking: changing p, d at stops > k leaves the stage-k outputs
  bit-identical.
- (c) SCALAR: two rows with equal W_k but different histories give identical
  outputs, while FULL gives different ones.
- (d) hand-computed g10, t11 and the MLP summaries on a 3-stop toy route match
  within 1e-6.
- (e) `RouteStats.fit` uses only the rows passed in.

**M3 — `expa/backward.py`** (the generic LS driver of section 2.2/2.3).
```python
class Predictor(Protocol):
    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray: ...
FitFn = Callable[[int, RouteCache, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Predictor | None], Predictor]
#        (k, cache, fit_idx, y_fit, val_idx, y_val, warm_start) -> fitted predictor for epoch k
def ls_backward(cache: RouteCache, B: float, H: np.ndarray, E: np.ndarray, fit_fn: FitFn,
                folds: tuple[np.ndarray, np.ndarray] | None, val_idx: np.ndarray | None,
                log: list | None = None) -> dict[int, list[Predictor]]
class IsoPredictor:          # wraps sklearn isotonic / _ConstantModel on W_k; also adapts BATON cont_models
    ...
def iso_fit_fn(...) -> FitFn  # exact replica of fit_lsm_general's per-k branches (>=2, ==1, 0 alive)
```
- `folds=None` means BATON mode: fit on all alive rows and stop with the same
  predictor.
- With folds, follow section 2.3.
- `log` receives dicts `{k, fold, fit_rows, applied_rows, disagree_rate}`.

*Tests:*
- (a) **LS parity:** `ls_backward(cache, …, iso_fit_fn, folds=None)` on each
  CON3-0 route (`route_problem` n_train = 500) gives predictions on a 50-point
  W grid equal (atol 1e-9) to `fit_lsm_general` for every k.
- (b) **Leakage:** with folds, the log shows, for every k and fold, that
  `fit_rows` ∩ `applied_rows` = ∅, and that `val_idx` never appears in any
  `fit_rows`.
- (c) all targets lie in {0} ∪ set(H) ∪ set(E), and every fitted row has
  τ > k.
- (d) perturbation: changing the outcomes (p, d after stop k) of a fold-A row
  changes no fold-A predictor at k. Fit and compare with a deterministic
  `iso_fit_fn`.

**M4 — `expa/model.py`.**
```python
@dataclass
class NetConfig: d: int = 64; layers: int = 2; heads: int = 4; ff: int = 128
class SetNet(nn.Module):
    def forward(self, glob: Tensor, tok: Tensor) -> Tensor   # (n,) in (0,1); tok (n, T, F), T >= 1
class MLPNet(nn.Module):
    def forward(self, x: Tensor) -> Tensor                   # (n,) in (0,1)
```
*Tests:*
- (a) SetNet has 5e4 to 2e5 parameters; MLPNet has < 5e4.
- (b) permuting tokens changes the output by < 1e-5.
- (c) T = 1 works.

**M5 — `expa/train.py`.**
```python
@dataclass
class StageConfig: lr_cold: float = 1e-3; lr_warm: float = 3e-4; batch: int = 512
    s_min: int = 50; s_min_cold: int = 200; s_max: int = 400; epochs_factor: int = 20
    n_ckpt: int = 4; clip: float = 1.0; min_rows: int = 20
def neural_fit_fn(arm: str, seed_key: str, cfg: StageConfig, diag: list) -> FitFn
#   returns a NetPredictor; diag gets {k, fold, steps, train_mse_first, train_mse_last, val_mse_per_ckpt, chosen_ckpt, seconds}
def fit_neural(spec: RouteSpec, paths: Paths, arm: str, rep: int, N: int, cfg: StageConfig
               ) -> tuple[dict[int, list[Predictor]], RouteStats, list[dict]]
```
- `fit_neural` splits the paths 80/20, forms folds, fits `RouteStats` on the
  fitting rows and calls `ls_backward`.
- `NetPredictor.predict(cache, k, idx)` builds features from that `cache`, so
  binding to rows is explicit.

*Tests* (tiny: CON3-0 route 1, N = 200, s_max = 10):
- (a) two calls give `np.array_equal` predictions on 100 dev rows.
- (b) a different rep gives different predictions.
- (c) at the cold stage, `train_mse_last` < `train_mse_first`.
- (d) `arm` does not enter the seed: FULL and SCALAR at the same key start
  from identical weights at k = m−1 (compare the initial state dicts).

**M6 — `expa/simulate.py`.**
```python
def simulate_policy(cache: RouteCache, B: float, H: np.ndarray, E: np.ndarray,
                    models: dict[int, list[Predictor]], H_bill: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]     # costs (N,), action int8 0/1/2; uses cache.g
def baton_models(cont_models: dict) -> dict[int, list[Predictor]]   # wraps fit_lsm_general output
```
- The control flow mirrors `_simulate_costs_general` line by line.
- Ĉ is the mean over the list. A missing k means no decision.

*Tests:*
- (a) **Parity:** for all 4 CON3-0 routes (`route_problem` n_train = 500),
  costs and actions are `np.array_equal` to `_simulate_costs_general` with the
  same `cont_models`, also in the `none` setting (H = 1e12, H_bill = H).
- (b) the mean cost is ≥ the oracle mean.

**M7 — `expa/baselines.py`.**
```python
def classical_ladder(spec: RouteSpec, g_fit: np.ndarray, which: tuple[str, ...]) -> dict[str, object]  # fitted models
def eval_classical(spec: RouteSpec, models: dict[str, object], g_eval: np.ndarray) -> dict[str, dict]  # label -> {mean_cost, handoff_rate, fail_rate, costs}
```
- Labels: none, tuned_tau, baton_eq, dp, otr21 and oracle (eval only).
- The caller names a label `baton_1e6`/`dp_1e6` when `g_fit` has 10^6 rows.
- Neural labels are not produced here.

*Test:* on `route_problem("CON3-0")` data, the sums over routes are
`baton_eq` = 83.073 and `oracle` = 75.37 (rounded to 3 decimals).

**M8 — `expa/eligibility.py`.**
```python
def eligible_routes(insts: list[str], out_path: Path, overwrite: bool = False) -> pd.DataFrame
```
- Columns: inst, route_id, m, stratum, dev_breach_rate, eligible,
  in_1e6_subset.
- If `out_path` exists and `overwrite` is False, recompute and assert equality
  with the file instead of writing.

*Tests:*
- Deterministic on CON3-0: two calls on a tmp path give equal frames.
- The rule is applied correctly.
- (slow) on all 20 instances: 62 eligible over 20 instances; the subset has
  6 + 6.

**M9 — `expa/sweep.py`.**
```
python -m transvrp.expa.sweep fit   --stage {smoke,pilot,main,big} --jobs 4
python -m transvrp.expa.sweep eval  --stage ... --eval {dev,test} [--frozen-commit HASH] --jobs 4
python -m transvrp.expa.sweep merge --stage ...
```
- Each job writes its own file `results/expA/<stage>/rows/<key>.csv`, with
  per-day cost arrays in `.npz`. `merge` concatenates them into
  `results/expA/<stage>.csv`. Only one writer per file.
- CSV columns: inst, route_id, m, stratum, N, rep, policy, mean_cost,
  handoff_rate, fail_rate, gap_closed, n_eval, eval_set, fit_seconds,
  fit_commit, eval_commit.
- Fit and eval are resumable: existing keys are skipped.
- Workers are processes with `torch.set_num_threads(1)`.
- The `smoke` stage uses 1 route, N = 100 for all arms, s_max = 10, and
  N_big = 2,000 in place of 10^6.

*Tests:*
- `smoke` fit + eval + merge yields exactly the policies {none, tuned_tau,
  baton_eq, baton_big, dp_big, oracle, nn_full, nn_scalar, nn_mlp}.
- `eval --eval test` without `--frozen-commit` exits non-zero.
- Eval reruns without refitting: model file mtimes are unchanged.

**M10 — `expa/analysis.py`.**
```python
def instance_deltas(df: pd.DataFrame, a: str, b: str, N: int) -> pd.Series   # df columns as M9 CSV; mean over rep, then route, then Δ per instance
def classify(d: np.ndarray, delta: float = 0.7, p_adj: float | None = None) -> str   # better/worse/equivalent/inconclusive (1.5)
def go_no_go(d: np.ndarray, delta: float = 0.7) -> dict      # branch in {WIN, LOSE, TIE_STRONG, TIE_WEAK} + numbers + significance flag
def holm(pvals: list[float]) -> list[float]
```
It writes `results/expA/summary.md` and `curve.csv`.

*Tests* (deterministic constructed 20-vectors; L = `np.linspace(-1, 1, 20)`):
- 2 + 0.1·L → WIN; −2 + 0.1·L → LOSE.
- 0.1·L → TIE_STRONG; 5·L → TIE_WEAK.
- 0.7 + 0.05·L → WIN (Δ̄ exactly at δ is included; compare with
  Δ̄ ≥ δ − 1e-9), while 0.6 + 0.05·L → TIE_STRONG with the significance flag
  "better".
- Tie-free win-count boundary: `np.r_[1.5 + 0.01*np.arange(13), -0.1 -
  0.01*np.arange(7)]` → WIN, and `np.r_[1.5 + 0.01*np.arange(12), -0.1 -
  0.01*np.arange(8)]` → TIE_WEAK (Δ̄ = 0.88, significant, only 12/20
  positive). Use `scipy.stats.wilcoxon(..., method="exact")`; these values were
  checked with scipy 1.18.1.
- 0.3 + 0.05·L → TIE_STRONG with the significance flag "better".
- `holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]`.

**M11 — `expa/boost.py`** (WIN branch only). `HistGradientBoostingRegressor`
with OTR-2.1's hyperparameters (max_iter 120, depth 3, lr 0.1, l2 1.0,
min_samples_leaf 20, random_state 0) but **no monotone constraint**
(section 4.2). It uses the FULL globals g1–g12, is plugged into `ls_backward`
as a `FitFn` with the same folds, and is reported as `boost`.
*Test:* on CON3-0 at N = 2,000, dev cost lies between the oracle and none on
every route.

---

## 10. Threats to validity, pitfalls, stop rules

- **Peak vs endpoint.** All targets and breaches use the running-peak τ;
  M3(c) guards this. An endpoint label zeroes the targets on
  collect-then-deliver routes (BATON v1's bug).
- **Optimism.** Cross-fitting removes direct optimism, and M3(b) proves the
  disjointness. The residual is the fold-mixing inconsistency, which is
  logged. Test days are fresh.
- **Scale of data.** At 10^5–10^6 paths, BATON approaches the (k, W) optimum,
  so a FULL gain must come from information (H2); H3 checks pipeline
  competence. At N* the cap allows ≤ 2·400·512 ≈ 4·10^5 samples per stage
  across the folds, against ~4·10^4 alive rows per fold per stage (about
  5 epochs). This is compute-limited by design; the 10^6 subset and cut 1
  bound the effect.
- **H_bill vs decision H.** Here they are equal for every learned policy. If a
  standby shadow price λ is added, the features (g6, t7), the targets and the
  rule must all use H_dec, and billing must use H_bill.
- **ρ = 0.6.** The common factor already makes W_k informative about the
  future, so FULL's extra value comes from separating the p and d factors.
  Under a single shared copula or ρ = 0 the history adds little. This is
  generator-specific and a stated limitation. An optional ρ ∈ {0, 0.3}
  sensitivity on the 12-route subset is allowed after the primary result.
- **Small routes.** m ≤ 6 routes almost never breach (11/13 never). They are
  excluded by rule.
- **The oracle is a lower bound, not a target.** It sees the whole path, so
  GC < 1 for every online policy. BATON closes far less than the oracle gap
  (median 17% residual).
- **Small-N handicap.** Fold nets use 40% of N, validation uses 20 paths at
  N = 100, and checkpoint selection there is noisy. The N = 100 point mostly
  measures the early-stopping rule, and the paper says so.
- **Separation (PROJECT.md section 6).** Neural targets are the neural's own
  realized costs. BATON's fitted values are never inputs or targets.
  `baton_models` exists for parity tests only.
- **Single cost model and single generator.** `LastMileCosts` defaults and the
  two-copula gamma generator only.

**Stop and redesign if:**
1. M3/M6 parity or GOLD fails.
2. Any policy's mean cost is below the oracle on any route.
3. SCALAR is "better" than DP-1e6 at N* on dev (rule 1.5). This is not
   impossible (DP has bin bias), but it triggers a leakage audit of the M3
   logs. Stop only if the audit finds leakage. Otherwise report it as "DP
   resolution bias".
4. Pilot P3 or P6 still fails after 3 iterations.
5. Fewer than 40 eligible routes, or fewer than 15 instances with one.
6. The projection exceeds 7 days after the platform rule and all cuts.
7. In the main run, the warm-start checkpoint is selected on > 50% of stages
   at N ≥ 10^4 for > 25% of jobs, meaning the nets are not learning.

---

## 11. Out of scope for v1

- The three-action version {continue, handoff, restock} (`fit_lsm_actions`).
  It is the first extension after a WIN and needs its own design, because the
  restock reset makes F_k state-dependent.
- Networks shared across routes or instances (amortization). This is the only
  setting where a set encoder over remaining stops carries information.
- A refit of the final nets on all N paths: a cheap variant, deferred.
- LLM weight-space editing (D(ii)) and LLM heuristic design (D(i)).
- Pathwise or score-gradient training (direction B). Policy-gradient RL
  baselines (RESULTS_OTR2 section 9 already has one).
- Planning changes (ALNS, gates, other plan keys), cost parameters, and the
  generator.
- City and Salhi–Nagy instances, which are not in the copied data.

---

## 12. Revision log (response to REVIEW_A.md)

| # | sev. | decision | reason / change (section) |
|---|---|---|---|
| 1 | blocker | ACCEPTED | The unit is now the instance (n = 20): t-intervals on instance means, exact Wilcoxon n = 20, win count ≥ 13/20, cluster bootstrap as sensitivity; route numbers are descriptive. Feasibility checked with measured instance SDs (1.2, 1.3, 1.4, 8). |
| 2 | major | ACCEPTED (fix i) | Policy iteration is replaced by BATON's one-pass backward-in-k recursion with per-k warm-started nets and 2-fold cross-fitting. The deployed value is the fold mean. Validation MSE is compared only within a stage. The old stop rule 7 is replaced (2.2, 2.3, 4.3, 10). |
| 3 | major | ACCEPTED | Per-k stages have exactly m−k tokens, so there is no padding. The budget is re-derived from Σ_k(m−k+1) = 117 (≈ 71 h estimate). The pilot times a full N = 10^5 cell and a 10^6 pass. The GPU/Colab rule applies when the projection exceeds 4 days. Ordered cuts and a 7-day ceiling are kept. The wheel was checked to download through the proxy (7.2, 7.3). |
| 4 | major | ACCEPTED | M10 tests are deterministic constructed vectors, including boundaries (Δ̄ = δ, win count 12/13, significant-small TIE_STRONG). The df schema is given. H2, H3, ARCH and M11 have rules (1.5). M11 has no monotone constraint, consistent with 4.2. |
| 5 | major | ACCEPTED | Re-measured on dev: 0.73% was not reproducible as a general figure. BATON at 500 vs 10^5 differs by 2.2% (instance SD 1.39%); the replicate SD at 10^5 is 0.13% (median). δ is now 0.7% = max(0.5%, 3 × refit-noise SD 0.22%) (1.3). |
| 6 | major | PARTLY | Generator μ/σ are replaced by empirical statistics from the fitting rows, which resolves the "model knowledge" confound for SCALAR and FULL (3.1). Instead of a SCALAR-minimal arm, an MLP arm with FULL's information isolates the encoder (ARCH). The "set encoder" wording is licensed only by ARCH. The seed excludes the arm. Reason for no SCALAR-minimal arm: with empirical features SCALAR's inputs are functions of BATON's own data, and compute is limited. |
| 7 | major | ACCEPTED | Full-size and 10^6 pilot timing cells. P3 uses 3 replicates and a noise-scaled tolerance, and its "better" side is a flag. Stop rule 3 is reworded (DP bin bias). P2 says dev. Fit and eval are separate stages with saved models (5, 7.3, M9). |
| 8 | major | ACCEPTED | δ is now noise-based (see 5). "Recourse cost" is used throughout. The header numbers were measured on dev with BATON only. |
| 9 | minor | ACCEPTED | Claim texts are conditional templates. Significant-but-small effects carry the "although significantly …" wording. The "significant and ≥ δ" logic is stated explicitly. The crossover N needs the condition at all larger N. |
| 10 | minor | ACCEPTED | Valid dataclasses; `lru_cache(maxsize=1)` with chunked generation; the predictor binds to a `RouteCache` and simulate uses `cache.g`; leakage and perturbation tests in M3; M5 diagnostics include train MSE; M8 `out_path`/`overwrite`; subset rng specified; smoke label set corrected; per-job CSV + merge; M7 returns classical labels only; `fit_otr_peak(g, B)`; feature clipping; section cross-references fixed. |
| 11 | minor | PARTLY | Cosine decay is used. Per-k stages give each epoch equal care, so no 1/n_alive weighting is needed. The 40%/80% data use is stated in H1. Tuning on CON3-0 dev is logged. The final-refit variant is deferred to section 11 (compute). |
| 12 | minor | ACCEPTED | Corrected: 13 routes with m ≤ 6, 11 never breach. The 0.73% figure is removed and the facts are re-measured. The oracle gap is now the 17.1% median (one-route 7.5% removed). The 0.8 GB memory claim is replaced by chunked generation. |

### 12.1 Disclosure: one BATON-only pass on the test stream (added 2026-11-15)

While re-measuring BATON's refitting noise for the margin (finding 5), the first
pass of the measurement script evaluated BATON on the 20,000 test days. Only
BATON was evaluated; no neural model existed and no number from that pass is used
anywhere in this specification (delta, the route list and all thresholds come
from the dev stream). Strictly, the test stream has therefore been touched once.
The paper must state this. Everything neural is evaluated on test days only after
the frozen-commit gate of section 5.
