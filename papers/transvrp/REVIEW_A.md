# Review of EXPERIMENT_A.md (direction A, set-encoder continuation values vs BATON)

Reviewer: independent pass, 2026-11-15. Spec reviewed as written; code read in
`papers/transvrp/code/` (data.py, benchmark.py, baton/costs.py, baton/otr2.py,
tests/test_parity.py) and upstream `svrpspd_wdro/core/dp_exec.py`, `otr21.py`.
Read-only Python was run from `code/` (scripts kept outside the repo). Torch is
not installed, so nothing neural was run; parameter and FLOP counts are by hand.

## Verdict: APPROVE WITH CHANGES

The question is well posed, the baseline ladder is strong, the test-set lock and
dev/test separation are good, and the facts about the existing code are mostly
right. Three things must change before any sweep: the statistical unit ignores
the clustering of routes inside instances (finding 1), the training target is
not the analogue of BATON's backward induction and may leave the neural
policy under-converged (finding 2), and the compute budget omits padded-token
cost and is probably 2.5-3x too low, which puts the plan past the 7-day ceiling
(finding 3). Several statistical rules and tests also need repair (4-8).

## Ranked findings

### 1. [blocker] Routes are not independent units; CIs and tests will be too narrow
Location: section 1.2 (bootstrap over routes, Wilcoxon on {Δ_r}), section 8 items
1-3, M10.
Problem: the 62 eligible routes come from only 20 instances (31 CON3, 31 SCA3).
`gen_scenarios` draws ONE Gaussian copula with ρ = 0.6 across all 50 customers of
an instance (data.py:73-94, 108-112), so every route of an instance sees the same
training, dev and test days and the same common demand factors. Δ_r of routes in
one instance are correlated by construction (common random numbers plus a shared
factor). The bootstrap "over routes" and the Wilcoxon test treat 62 correlated
numbers as 62 independent ones. The effective sample size is closer to 20 than
62, so CI_95, the Wilcoxon p, the 60% rule and above all the TOST (CI_90 inside
±1%) are anti-conservative. A false WIN or a false "strong tie" is the likely
failure.
Fix: make the instance the resampling unit. Use a cluster bootstrap (resample
instances, keep all eligible routes of a drawn instance), and replace Wilcoxon on
routes by Wilcoxon (or sign-flip permutation) on instance-level mean Δ, n = 20.
Keep the route-level numbers as descriptive. State that "X/Y routes" counts are
descriptive. Recompute the strong-tie feasibility: with 20 clusters the CI_90
half-width is roughly 1.7 · SD(Δ̄_inst)/√20, so measure SD of instance-mean Δ in
the pilot and check TOST at δ = 1% is even attainable before freezing δ.

### 2. [major] The target scheme is approximate policy iteration from the reactive policy, not backward induction; convergence is assumed, not checked
Location: section 2.2-2.3, rule "3 rounds, or 5 if pilot gains > 0.3%", stop rule 7.
Problem:
(a) BATON's backward pass uses the already-optimal later-epoch policy, so one
pass is exact. Here a single stopping rule ψ is applied at all later epochs.
Round 0 gives the reactive value, round 1 the value of "greedy w.r.t. reactive"
(π1), round 2 the value of π2 = greedy w.r.t. net_1. Policy iteration on an
m-step stopping problem can need many rounds; two improvement steps are not
shown to reach the optimum. The deployed Ĉ = ½(net_1 + net_2) averages value
functions of two different policies (π1, π2) and is greedy for neither, so it
is not any policy's value. A LOSE (or weak tie) may then reflect under-converged
policy iteration, and the spec's LOSE diagnosis (SCALAR vs DP-1e6) would blame
"training failure of the pipeline" without telling you which part.
(b) Stop rule 7 monitors inner-validation MSE across rounds, but the targets
change with ψ every round, so MSE values from different rounds are not
comparable. A "rising MSE" is not evidence of non-convergence, and a flat MSE is
not evidence of convergence.
(c) The 3-vs-5 round rule changes cost by 5/3 and is not in the budget table.
Fix: (i) preferred: mirror BATON exactly. Train for k = m-1 down to 1 with a
k-conditioned shared net (or the same architecture per k, warm-started from k+1),
using frozen later-epoch nets for ψ and cross-fitting (train on half A, apply the
stop update to half B, swap) to control optimism. This is the same recursion as
`fit_lsm_general`, one pass, no convergence question. (ii) If keeping policy
iteration: iterate until dev cost changes by < 0.1% for two consecutive rounds
(cap 6), deploy the last single net (or the average of the last two nets
trained on the SAME ψ), and use the fraction of (path, epoch) decisions that
flip between rounds as the convergence diagnostic. Replace the stop-rule-7
wording accordingly and put the extra rounds in the budget.

### 3. [major] Compute estimate ignores padded tokens and is probably 2.5-3x low; the 5-round option is unbudgeted
Location: section 7.2, section 4.1, section 4.4.
Problem: tokens are padded to T_MAX = 24 and masked, but PyTorch does not skip
masked positions (`enable_nested_tensor=False` is specified). Every sample costs
25 token-rows, not "≈ 9 tokens". By hand: per token per layer 32,768 MAC =
65.5 kFLOP, two layers 0.131 MFLOP (the spec's 0.13 is right), so forward is
about 3.3 MFLOP and forward+backward about 10 MFLOP per sample, versus the
spec's 3.6. One capped job (9,000 steps of 512) is then about 4.6·10^13 FLOP,
about 50 min at 15 GFLOP/s, not 20-25 min. FULL (44 h) and SCALAR (22 h) become
roughly 120 h and 60 h. With the stated ×2-3 uncertainty the plan exceeds the
7-day hard ceiling on CPU. A second check: the spec's own job sizes sum to
about 29 h for FULL (38 min per route-replicate summed over the five N, ×186
cells / 4 workers), not 44 h, so the table is not internally derived. The claim
"at N = 10^5 a round sees ≤ 3 epochs" is also loose: pairs per half are about
3.5·10^5 for m = 17 (measured alive fractions), so about 4 epochs, and short
routes see more.
Fix: use a per-route token length T = m-1 (not 24) and trim each batch to the
largest valid token count (sort or bucket by k), which cuts cost about 2.5x;
re-derive the table from measured per-step time in the pilot; make the GPU run
the default plan, not cut 1. Include the 5-round (or convergence-loop) cost.
Also check that the CPU wheel can be fetched through the sandbox proxy; if not,
the plan has no neural run at all.

### 4. [major] M10 synthetic test for TIE_WEAK fails about two thirds of the time
Location: section 9, M10 tests.
Problem: neural = BATON·(1 + N(0, 0.03)) gives Δ_r ~ N(0, 3%). With n = 60 the
bootstrap CI_90 half-width is about 0.65%, so for a mean near 0 the interval sits
inside (−1, 1) and the branch is TIE_STRONG. I simulated it: P(strong) ≈ 0.66,
so a correct `go_no_go` fails this test about 66% of seeds. The test is also not
seeded.
Fix: use SD = 8% (half-width about 1.7%, always weak), or a fixed seed plus a
deterministic assertion on a constructed Δ vector. Add boundary tests: Δ̄ =
+1.0 exactly, win fraction 59% vs 61%, CI lower bound 0.0, and a case that is
significantly better by 0.5% (TIE_STRONG) so the wording issue in finding 9 is
visible. Specify the `df` schema `go_no_go` expects (it is not given).

### 5. [major] H2, H3, the attribution step and the M11 comparison have no decision rule; M11 contradicts section 4.3
Location: section 1.1 (H2, H3), section 1.2 WIN row, section 8 item 3 and 6, M11.
Problem:
- H3 says SCALAR "does not beat BATON by more than δ". That is an equivalence
  claim; section 8 item 6 says only TOST supports equivalence, yet the secondary
  family tests SCALAR vs BATON-eq with a two-sided Wilcoxon and Holm. H3 therefore
  has no pre-registered rule.
- The WIN row says attribution "is required before the claim is made" but never
  defines success (FULL vs SCALAR Holm-adjusted p < 0.05 with Δ̄ > 0? by how much?)
  nor what happens if FULL beats BATON but not SCALAR (gain from function class,
  not information).
- "If M11 matches FULL within δ" is undefined (TOST on which statistic, at which
  N, which unit).
- Δ_r is defined only for FULL vs BATON-eq; for FULL vs SCALAR or vs DP-1e6 the
  denominator is unstated.
- M11 trains boosted trees with monotone constraint on W_k "on the FULL global
  features g1-g12". Section 4.3 argues that monotonicity in W_k is not coherent
  once g8-g11 move with W_k. The two sections contradict each other.
Fix: write the attribution rules as a table (statistic, N, test, threshold,
outcome text) alongside 1.2; define Δ_r(a vs b) = 100(C_b − C_a)/C_b generically;
drop the monotone constraint in M11 or keep it only for a labelled variant.

### 6. [major] "SCALAR = BATON's information" is true only asymptotically; the neural arms also receive generator knowledge BATON never sees; the "set encoder" is not what is being tested
Location: section 3 (SCALAR paragraph, t1-t4, t10, t11), section 1.1 H1/H3, section 4.1.
Problem: the information claim is correct as a statement about online
observations (all non-H globals and all tokens are functions of (k, W_k) and
route data; I checked each row). But the route data includes exact μp, μd and
σ = 0.30·μ, and t10/t11 hand the network a Gaussian-approximate breach z-score
built from the true generator parameters. BATON and the DP are distribution
free. So "SCALAR minus BATON = function class plus training procedure" also
contains model knowledge, and at small N that knowledge is a prior, not
information in BATON's sense. SCALAR is run only at N ∈ {2,000, 10^5}, so the
small-N story ("loses at small N") cannot be attributed. Separately, within one
route every token is a deterministic function of k (and t10/t11 of (k, W_k)),
so the encoder is a (k, W_k, history) MLP in disguise; the spec itself says
this in section 3 but the hypotheses and the TIE text still say "set encoder
over the online state". A tie therefore says nothing about transformers.
Fix: add a cheap SCALAR-minimal arm (inputs k/m, m, W_k/B, H/E only; no μ, σ,
t10, t11) at N ∈ {2,000, 10^5} and an MLP arm on the 12 globals plus (t10, t11
at the mean) at N*. State in H1 that the tested object is "richer state plus
model features", and phrase the claim "set encoder" only if the MLP arm loses to
it. Also give FULL and SCALAR the same init seed (the seed currently includes
`arm`, so H2 is an unpaired-init comparison; this adds avoidable noise).

### 7. [major] Pilot does not exercise the expensive path; test lock couples fitting to test evaluation; P2/P3/stop-rule wording
Location: section 7.3, section 5 "Test-set lock", M9, section 10 stop rules 3 and 7.
Problem:
- Pilot N ∈ {10^3, 10^4}: at N = 10^4 steps are about 1,100, below S_max, so the
  capped N = 10^5 job, the 10^5 target pass and the 10^6 pass are never timed.
  P5's projection extrapolates blind.
- P3 pools 4 routes, one replicate, one init, dev 5,000 days. The pooled SE of a
  cost ratio is about 0.3% from test noise alone, plus init variance; the window
  [−1%, +3%] is arbitrary and can fail or pass by noise. The lower bound
  "below −1% = leakage" is not a theorem: BATON-eq at 10^4 uses in-sample greedy
  updates and isotonic step noise, so a neural policy can legitimately be
  1% better.
- Stop rule 3 ("SCALAR beats DP-1e6 or BATON-1e6 by > 0.5% pooled on dev is
  impossible") is not impossible: DP-1e6 is a 256-bin piecewise-constant
  estimator, so it has resolution bias at the decision boundary; the N at which
  it is evaluated is not stated.
- P2 says "test costs" but the pilot never evaluates test.
- `sweep.py --stage main --eval test` fits and tests in one run. A crash or a
  bug in evaluation found on day 2 forces a refit of days of neural training or
  a protocol deviation.
Fix: add one full-size pilot cell (route 3, N = 10^5, all rounds) and one 10^6
target pass for timing; use 3 replicates for P3 and compare against the replicate
SD; make P3 a flag (investigate leakage) rather than a gate; fit-and-save
(nets, cont models) as a separate stage from `--eval test`, so evaluation can be
rerun without refitting and the freeze hash covers the evaluation code only.

### 8. [major] The 1% margin is not justified by the evidence quoted, and "cost" should be named recourse cost
Location: section 1.2 (δ paragraph), header facts.
Problem: the spec says δ "is larger than the BATON N = 500 vs 2·10^5 difference"
(0.73%). I could not reproduce 0.73%. On CON3-0 route 3 with the repo train
seed (500 paths) against a 2·10^5 fit I get 0.22% (paired SE 0.37%, 2,000 days).
With five independent 500-path draws against a 10^5 fit and 20,000 test days
the difference is 0.38, 0.91, 0.91, 0.43, 1.30% (SE about 0.1%). The training-draw
noise of BATON at N = 500 is therefore about 0.4-1.3%, same size as δ, so the
justification is weak. "One seventh of the BATON-oracle gap" (7.5%) is from this
single route. Also, the costs in `benchmark.py` and costs.py are recourse costs
only (H, E, zero on completion); a 1% change of recourse cost is a small share
of total route cost, and the spec calls it "BATON cost".
Fix: freeze δ after the pilot from a measured quantity, e.g. δ = max(1%, 2 ×
between-replicate SD of BATON-eq at N*), and say "recourse cost" throughout.
Report the route-level between-replicate SD as already planned.

### 9. [minor] Wording of branches can contradict the numbers
Location: section 1.2 TIE and LOSE rows.
Problem: the strong-tie text asserts results not yet seen ("loses at small N",
"~10^3× more compute", "BATON's scalar isotonic model is at the efficient
complexity point"). A significant +0.5% neural gain or −0.8% loss with CI_90
inside ±1% is also labelled "equivalent". WIN requires the point estimate ≥ +1
but only the lower bound > 0, which is a point-estimate threshold paired with a
null-0 test (mixed superiority logic). The data-size "crossover N = smallest N
with CI_95 lower bound > −δ" is ambiguous if the curve is non-monotone.
Fix: make the claim texts conditional templates filled from the data; for
significant-but-small effects use "equivalent within ±δ although significantly
better/worse by x%"; state the crossover as the smallest N after which all larger
N satisfy the condition.

### 10. [minor] Implementability gaps and tests that cannot fail
- M1: the `RouteSpec` code block is not valid Python (`class RouteSpec: inst: str; ...`
  followed by indented lines). `functools.lru_cache` on `draw_instance` caches
  arrays of 0.4-0.8 GB at n_chunks = 10 and 3 reps; with 4 workers memory is
  tight (a 10^6 draw peaks at about 2.5 GB with the Z/U temporaries, not 0.8 GB).
  Set `maxsize=1` or cache per process and say so.
- M3/M5: a neural `ChatFn` is bound to a `Paths` object (`NeuralPolicy.chat_fn(paths)`),
  but round r applies ψ to a different half (H1 or the validation part). The
  spec does not say that `train_policy` rebinds `chat_fn` to each subset, nor
  that `simulate_policy`'s `g` must be the same rows as the bound `paths`
  (nothing asserts it).
- M5 test (c) needs training MSE, but the returned dict is "per-round val MSE,
  steps, seconds". Test (d) ("uses only `paths` rows passed in", "split sizes
  80/20, halves disjoint") checks bookkeeping and cannot detect leakage. Add a
  test that records, per round, the index set a net was trained on and the index
  set it is applied to, and asserts they are disjoint; add a perturbation test
  (change rows of H0 after round 0, net_1 must still be a function of H1
  targets that legitimately depend on net_0).
- M8: "refuses to overwrite" contradicts the test "two calls equal". Say which
  path the function writes and add an `out_path`/`overwrite` argument. The 10^6
  subset: one `default_rng(2026)` shared by both strata or two fresh rngs, and
  in which order? Specify.
- M9: smoke test lists "the 9 ladder labels minus otr21_eq at N = 100" but
  `nn_scalar` runs only at N ∈ {2,000, 10^5} and `baton_1e6`/`dp_1e6` need 10^6
  paths; the expected label set is wrong. Four worker processes appending to one
  CSV need one writer or per-job files plus a merge, otherwise rows interleave.
- M7 does not say whether `classical_ladder` returns `nn_*` labels, and
  `fit_otr_peak` takes `(g_hist, B)`, not `g_N` alone (section 6 shorthand).
- Section 8 cross-references: "section 8.3" for the data-size curve is item 5.
- M2: SCALAR zeroing is by `HISTORY_IDX`, which is consistent with the table
  (idx 1, 7-11 = g2, g8-g12). No normalisation/clipping is specified for g1,
  g12, t10 (unbounded); state it or clip.

### 11. [minor] Details that can make the neural look worse (or noisier) than it is
- Constant lr 1e-3 with no decay and 8 checkpoints: the best-of-8 checkpoint on
  a noisy final iterate; MSE weighting is per (path, epoch) pair, so early
  epochs (many alive paths) dominate while the late epochs, where decisions are
  near the boundary and few paths are alive, get little weight. BATON fits each
  k separately with equal care. Allow cosine decay (or Polyak averaging) and a
  per-epoch weight ∝ 1/n_alive(k) as pilot-allowed optimisation changes, or
  report that the neural is handicapped this way.
- At N = 100 the validation part is 20 paths; checkpoint selection on 20 paths
  is extremely noisy, and 100 minimum steps over ~300 pairs is ~170 epochs.
  This is by design (spec says so) but means the N = 100 point measures the
  early-stopping rule more than the model.
- Each deployed net sees 40% of N (halves) versus 100% for BATON, stated as
  "honest price"; it should be reflected in the H1 text (equal data is equal
  paths, not equal effective data), and a variant where the final deployed net is
  refit on all data with the final ψ is a cheap check.
- Pilot hyperparameters (lr, rounds, steps) are chosen on dev days of CON3-0,
  whose 4 routes are also in the primary set. No test leakage, but design is
  tuned on routes that are later evaluated; log it.

### 12. [minor] Numerical claims in the header facts
See the table: "13 of the 15 routes with m ≤ 6" is wrong (there are 13 routes
with m ≤ 6, 11 never breaching on 20,000 days; the 15/13 pair matches m ≤ 8 on
2,000 days). "0.73%" not reproduced (finding 8).

## Items checked and found sound (no action)
- Step indexing: BATON uses 1-indexed k, `cum[:, k-1] = W_k`, `H[k-1]`, `E[τ-1]`;
  the spec's section 2.1/2.2 recursion, the feature indices and the M2 tests use
  the same convention; no off-by-one found. `_overflow_step` returns m+1 for no
  breach as the spec assumes. The breach at stop 1 has no epoch (loop starts at
  k = 1 and evaluates the model only for k ≤ m-1; the oracle also excludes it).
- The cross-fit chain (net_0 on H0; net_1 targets on H1 under net_0; net_2 targets
  on H0 under net_1) never applies a net to its own training paths; the residual
  indirect channel the spec names (net_1's targets were shaped by net_0 that saw
  H0) is the only one. Validation paths are never trained on. For rounds 3-4 the
  same alternation holds.
- Within one route the remaining-stop set is a function of k only, so the claim
  that it adds no information is right (t10/t11 are functions of (k, W_k) and
  add feature engineering only).
- `holm([0.01, 0.04, 0.03]) = [0.03, 0.06, 0.06]` is the correct Holm output;
  the secondary family count (4+2+2+1+1+1 = 11) is right; the go/no-go branches
  are exhaustive and WIN/LOSE/TIE-strong are mutually exclusive.
- Stream seeding: train `[base,1,r,c]`, dev `[base,3]`, test `base+99991` are
  independent `SeedSequence` streams; the 20 instance bases are distinct (no
  `stable_seed` collisions). The `n`-dependence warning is right (see table).

## Claim-verification table

| # | Claim in spec | Status | Evidence |
|---|---|---|---|
| 1 | `fit_lsm_general(g, B, H, E)` backward recursion, `future` initialised to E[τ-1] if τ ≤ m else 0, loop k = m-1..1, fit on alive, stop update `alive & pred > H[k-1]` sets `future = H[k-1]` | CONFIRMED | costs.py:184-209 |
| 2 | Model k is fit on (cum[alive, k-1], future[alive]) with isotonic increasing, clip; n_alive<2 gives a constant | CONFIRMED | costs.py:194-203 |
| 3 | `_simulate_costs_general` control flow: emergency check at every k, no decision at k = m, trigger `predict(Wk[ho_active]) > H[k_idx]`, bills `H_bill[k_idx]`, action codes 0/1/2, missing key means no decision | CONFIRMED | costs.py:232-273 |
| 4 | `H_bill` defaults to H; only `none` decides with 1e12 | CONFIRMED | costs.py:232-233; benchmark.py:23-25 |
| 5 | `oracle_costs_general` = min(min_{k<j} H[k-1], E[j-1]) on breach, 0 otherwise, breach at stop 1 only E | CONFIRMED | costs.py:605-622 |
| 6 | `fit_rollout` targets equal "reactive" targets (no stop update) | CONFIRMED | costs.py:332-364 |
| 7 | `route_cost_schedules(route, D, scale, costs)` returns H,E of shape (m,); H[k-1] and E[k-1] are after/at stop k; H[m-1] never used | CONFIRMED | costs.py:105-138, 123-124 |
| 8 | `E0 = E[0]` is the largest price on the route; H, E ≤ E0 (targets in [0,1]) | CONFIRMED | checked on all 80 routes: max(H) ≤ E[0] and E non-increasing for every route |
| 9 | `route_problem` yields route_id, route, B, H, E, g_train, g_test with seed `base`, test `base+99991`, skips empty routes | CONFIRMED | data.py:115-133 |
| 10 | M1 test (a): `route_paths(spec, *draw_instance("CON3-0","test_std")).g` equals `g_test`; B,H,E equal | CONFIRMED (feasible) | same call `gen_scenarios(dbar,pbar,2000,base+99991)` and `ptr[:, r] - dtr[:, r]` (data.py:124, 133) |
| 11 | `gen_scenarios` draws deliveries then pickups from one stream, so `n` changes the pickup draws | CONFIRMED | data.py:108-112; measured: first 2,000 delivery rows of an N=100,000 draw equal an N=2,000 draw, pickups differ |
| 12 | Deliveries and pickups come from two independent Gaussian copulas, each ρ = 0.6 | CONFIRMED | data.py:73-94 (two `sample_demands` calls); measured corr(d_j, d_i) = 0.594, corr(d_j, p_j) = 0.002; note the copula spans all 50 customers of the instance (shared across routes) |
| 13 | CV = 0.30, σ = CV·μ, gamma marginals | CONFIRMED | data.py:20, 69-70 |
| 14 | `stable_seed(name)` is an int in [7, 10007] | CONFIRMED | data.py:97-98 (range [7, 10006]); no collisions among the 20 instances |
| 15 | `otr_route_v2` observes `(d_k, p_k)` | CONFIRMED | otr2.py:144-186 (`d_k, p_k = observe(k)`) |
| 16 | `fit_lsm` docstring: N < 500 is noisy | CONFIRMED | otr2.py:104-106 |
| 17 | `fit_dp` returns keys 1..m-1, default bins clip(N//30, 8, 256) so 256 at 10^6, models expose `.predict`, plug into `_simulate_costs_general` | CONFIRMED | dp_exec.py:67-153 (every loop iteration assigns `dp_models[k]`; line 87) |
| 18 | `dp_exec.py` and `otr21.py` import only `.otr2` (so copying beside `otr2.py` works) | CONFIRMED | dp_exec.py:32; otr21.py:30 (also needs scikit-learn, already a dependency) |
| 19 | `fit_lsm_rich(g_hist, mu, sig, B, H, E)` boosted trees, monotone `[1,0,0,0]`, same recursion; `μ_g = μp−μd`, `σ_g = sqrt(σp²+σd²)` as inputs | CONFIRMED | otr21.py:56-98 |
| 20 | `fit_otr_peak(g_N)` | WRONG (signature) | otr2.py:356: `fit_otr_peak(g_hist, B)`; minor shorthand |
| 21 | GOLD: baton 83.073, oracle 75.37 on CON3-0, 4 routes | CONFIRMED | tests/test_parity.py:7-14; `pytest tests -q` gives 2 passed |
| 22 | 80 routes over 20 instances, m ∈ {1..20, 22} | CONFIRMED | measured: 80 routes, m values {1..20, 22} |
| 23 | 62 routes have m ≥ 4 and breach rate ≥ 2% (test proxy); 31 are CON3 | CONFIRMED | measured on 20,000 test days: 62 and 31 |
| 24 | Strata: short ~2, medium ~28, long ~32 | CONFIRMED | measured 2 / 28 / 32 |
| 25 | "13 of the 15 routes with m ≤ 6 never breach" | WRONG | 13 routes have m ≤ 6, of which 11 (20,000 days) or 12 (2,000 days) never breach; 15 routes / 13 never match m ≤ 8 on 2,000 days |
| 26 | T_MAX = 24 > max m − 1 = 21 | CONFIRMED | max m = 22 |
| 27 | CON3-0 route 3 has m = 17 | CONFIRMED | route_problem("CON3-0") route ids 0-3 have m = 9, 12, 12, 17 |
| 28 | BATON at N=500 vs 2·10^5 differs by 0.73% of cost, paired SE 0.40% over 2,000 days | UNVERIFIABLE / not reproduced | repo seed (500) vs 2·10^5 fit: 0.22% (SE 0.37%); five independent 500-path draws vs a 10^5 fit with 20,000 test days: 0.38, 0.91, 0.91, 0.43, 1.30% (SE ≈ 0.1%). The SE order matches; the 0.73% is one draw and the "δ larger than this difference" inference is not safe |
| 29 | BATON at 2·10^5 is 7.5% above the oracle on that route | CONFIRMED | measured 7.46% (cost 62.48 vs oracle 58.15, 2,000 days) |
| 30 | `fit_lsm_general` at N = 2·10^5, m = 17 takes 0.5 s | CONFIRMED | measured 0.45 s |
| 31 | Parameter count ≈ 7.7·10^4, within 5·10^4..2·10^5 | CONFIRMED (by hand) | 832 + 768 + 2×33,472 + 8,256 + 65 = 76,865 (embeddings, two pre-norm layers d=64/ff=128, head; torch absent so not run) |
| 32 | Cost ≈ 1.2 MFLOP per training sample forward (≈ 9 tokens × 0.13) | WRONG for the specified model | 0.13 MFLOP per token is right (65.5 kFLOP per layer per token); padded T_MAX = 24 plus the global token means 25 rows are always computed, ≈ 3.3 MFLOP forward |
| 33 | A capped job is ≈ 1.7·10^13 FLOP, ≈ 20-25 min | WRONG (consequence of 32) | with 25 rows ≈ 4.6·10^13 FLOP, ≈ 50 min at the assumed 15 GFLOP/s |
| 34 | At N = 10^5 a round sees ≤ 3 epochs of a half's pairs | WRONG (loose) | measured alive counts for route 3 give ≈ 3.5·10^5 pairs per half, so ≈ 4 epochs at 1.5M samples; shorter routes more |
| 35 | FULL block ≈ 44 h | NOT DERIVABLE from the stated per-job times | summing the spec's own job times gives ≈ 29 h before the padding correction |
| 36 | `holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]` | CONFIRMED | Holm step-down with monotone enforcement |
| 37 | Secondary family = 11 tests | CONFIRMED | 4+2+2+1+1+1 |
| 38 | BATON-1e6 / DP-1e6 cost seconds per route and ~0.8 GB peak | PARTLY | fits are seconds (isotonic 2·10^5 in 0.45 s); data generation costs ≈ 6 s per 10^5-path chunk per instance (≈ 1 h for 20 instances × 3 reps × 10 chunks); peak memory with Z/U temporaries is about 2.5 GB, not 0.8 GB |
| 39 | Step cap binds from n_pairs ≈ 77k (N ≈ 2·10^4) | CONFIRMED (arith.) | S = ⌈20·n_pairs/512⌉ ≥ 3000 when n_pairs ≥ 76,800 |
| 40 | SCALAR has exactly BATON's information (non-H globals and all tokens are functions of (k, W_k) and the route) | CONFIRMED for online observations; qualified | each of g1,g3-g7 and t1-t11 checked; but μ, σ and t10/t11 are generator knowledge BATON does not use (finding 6) |
| 41 | Remaining-stop set adds no information within a route | CONFIRMED | tokens j = k+1..m are a function of k |
| 42 | `HISTORY_IDX = (1,7,8,9,10,11)` matches g2, g8..g12 | CONFIRMED | table in section 3 (0-based) |
| 43 | Pilot P4 ordering C_oracle ≤ C_π pathwise | CONFIRMED | any non-anticipative policy pays at least the clairvoyant minimum on every path |
| 44 | M10 synthetic "BATON·(1+N(0,0.03)) gives TIE_WEAK" | WRONG | simulated: P(CI_90 inside ±1%) ≈ 0.66 for Δ_r ~ N(0,3%), n = 60 |
| 45 | M0 test "fit_dp returns keys == set(range(1, m))" | CONFIRMED | dp_exec.py loop assigns every k in 1..m-1 |
| 46 | RESULTS_OTR2.md section 8 reports OTR-2.1 as a null result with richer features | CONFIRMED | RESULTS_OTR2.md:157-168 (loses on 23/25 routes, −3.2%; features are net-g precision-weighted residual, last increment, running max, not the p/d split) |
| 47 | PROJECT.md section 4 recommends A first and B only if A shows a gap; section 6 on separation | CONFIRMED | PROJECT.md:105-107, 148-152 |
| 48 | M3 test (a) LS parity with `baton_chat_fn` can fail if the recursion is wrong | CONFIRMED (meaningful) | isotonic refit on identical (W_k, y_k) is deterministic; only valid for k with ≥ 2 alive paths (true at N = 500) |
| 49 | M6 parity test (exact array equality) | CONFIRMED (feasible and meaningful) | `baton_chat_fn` uses `np.cumsum(g)` identical to costs.py:235 |
