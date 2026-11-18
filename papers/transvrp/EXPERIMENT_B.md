# EXPERIMENT B — FULL against published algorithms on the two-action handoff

Status: pre-registration, **revision 3 (after REVIEW_B2), 3.1 (source verification)**, 2026-11-18, **before any B
code or run**. Revision 1 was reviewed in `REVIEW_B.md`, revision 2 in `REVIEW_B2.md`
(both: accept with required changes); response tables are appended there and the changes
are listed in section 15. The two sources revision 1 could not read are read in full:
`METHODS_ALP_PPO.md`.
After the pilot, sections 1, 6 and 7 (hypotheses, grids, gates and thresholds) are
frozen; any later change goes into `RESEARCH_LOG.md` with its date and reason and is
reported in the paper as a protocol deviation. Conventions are those of
`EXPERIMENT_A.md` (cited "A §x"): cost = **recourse cost** (handoff H, emergency E, 0 on
completion), unit of analysis = instance, δ = 0.7%. **[DC]** marks a design choice made
without evidence. **[VERIFY]** marks a value to be checked against the cited source under
section 3.0 before tuning. **[SRC p.x]** marks a value read in the source at page x.

Facts this spec builds on (A; `RESEARCH_LOG.md`, 2026-11-18):
- A's test stream was opened at frozen commit c37ad7c. FULL vs BATON-eq at 10^5
  paths: Δ̄ = +0.82% [CI95 +0.65, +0.99], s = 0.365, 19/20 instances: **WIN**.
  H2 (FULL vs SCALAR) better; H3 equivalent; ARCH (MLP) equivalent; M11 (boosted
  trees on FULL globals) equivalent (+0.25%, CI90 [+0.19, +0.32]).
- Measured fit seconds per job at 10^5 (i9-13950HX, 12 workers, CPU): nn_full mean
  2,317 s (max 5,929), nn_mlp 124 s, boost 44 s, classical ladder 33 s. The A
  projection under-estimated FULL by 1.43 (contention); section 9 applies it.
- 62 eligible routes over 20 instances (`results/expA/eligibility_dev.csv`); the
  12-route subset (6 medium, 6 long; 10 CON3 and 2 SCA3 routes from 10 instances) is
  fixed there.

Why B exists. BATON is under review and cannot be cited. In the paper A's `baton_eq` is
Longstaff–Schwartz regression with an isotonic regressor on (k, W_k). A shows FULL
beats it. B asks whether FULL also beats **published methods for the same decision**,
run on the same paths and judged by the same statistics. Wording rule for the paper
(abstract included): THR, RLSM and DOS are "published methods"; ALP-Z is "Zhu et al.'s
ALP specialised to a fixed route"; DQN and PPO are "DQN after Dastpak et al." and "PPO
with Kadyrov et al.'s encoder and settings, adapted to the two-action problem". For DQN
and PPO the paper never writes "converged"; it writes "not labelled budget-limited"
(section 6.3) or "budget-limited".

**Platform (fixed).** DQN trains on the GPU (NVIDIA RTX 5000 Ada, 16 GB; CUDA verified
with torch 2.6.0+cu124), one process at a time unless the pilot measures a gain from two.
Every other arm, every evaluation (DQN models included) and the FULL refit run on the CPU,
12 workers, `torch.set_num_threads(1)`. **GPU nondeterminism:** CUDA training is not
required to be bitwise reproducible. As in A, a replicate is defined by its seed and its
training rows; training noise is measured across reps 0–2. The object that is evaluated is
the hashed model file (section 5.3), and evaluation runs on the CPU, so test_b numbers are
deterministic given that file. Library versions (python, numpy, scipy incl. HiGHS, torch,
CUDA) are written into every payload and every result row.

## 0. Design thresholds needing author sign-off

Values chosen without pilot evidence. Each is frozen after the pilot unless changed here.

| # | item | value | rationale | if altered |
|---|---|---|---|---|
| T1 | tie tolerance, THR and ALP-Z | hand off iff Ĉ ≥ H − 1e-6·E0 | Zhu eq. 29.1 gives ties to the preventive action; LP vertices are tight exactly at handoff states | larger: more handoffs near the threshold; smaller: LP round-off flips decisions |
| T2 | ALP-Z lattice | n_w = 401 | LP ≈ 6.4k variables, ≤ 2.6M nonzeros, minutes in HiGHS | finer: ALP-Z closer to THR, LP slower; coarser: discretisation loss |
| T3 | THR grid | 2,001 points | interpolation error ≪ per-day SE; 4,001 check reported | changes thresholds by < 1 grid step |
| T4 | B-THR | within 3 SE on ≥ 99% of route-reps | 3 SE passes 99.7% by chance if exact | lower share: weaker bug detection |
| T5 | B-ALPZ | identities within 1e-6·E0; deployed decisions = lattice DP on ≥ 99.9% of visited dev states | exact up to float and interpolation edges | looser: masks assembly bugs |
| T6 | P1 | dev cost ≤ best trivial policy + 2 SE | "never hand off" is in every arm's class | tighter: more arms leave the family |
| T7 | bug detectors RLSM / DOS | 90% / 95% of stages with ≥ 100 rows | catch gross errors only | none on claims |
| T8 | tuning tie window | 0.10 points | about one per-day SE of the dev score | wider: cheaper configs win more often |
| T9 | learning curve | P_max = 3; budget-limited if gain > 0.5 points in pass 3 | 3 passes ≥ Kadyrov's 200k episodes | other margin moves the label |
| T10 | DQN update shape | batch 256, 1 update / 32 transitions, target every {8k, 32k} transitions | source replay ratio 8 at feasible cost | more updates: about 8× GPU time |
| T11 | PPO start | initial handoff probability 0.05 | late stages reached (0.95^16 = 44%) | 0.5: stages > 6 starve |
| T12 | PPO unstated items | value coef 0.5, ReLU in value net, no advantage normalisation | not stated in the paper; their code (rev. 3.1) confirms no normalisation and has no value net, so the other two stay [DC] | small |
| T13 | repairs | 1 per arm, bug fixes only | equal budgets | more repairs: tuning by gate |
| T14 | trigger | cuts until projection ≤ 0.8 × 168 h | 20% headroom for repairs | less headroom: no repairs |
| T15 | "coincides with THR" wording | ALP-Z and THR decisions agree on > 99% of dev states | descriptive only | wording only |

---

## 1. Hypotheses, unit, decision rules

### 1.1 Hypotheses

- **HB_a** (one per family arm a). At N* = 10^5 training paths per route, FULL (A's
  nn_full, unchanged configuration) has lower recourse cost than arm a trained on the
  same 10^5 paths.
- **RE-EVAL** (outside the family; renamed from "REP", which was not a replication).
  FULL vs baton_eq at 10^5 on the fresh stream test_b. It re-samples **test days only**:
  the models are refits that pilot Q3 shows to be bitwise A's (or A's own files under
  cut (i)), so it says nothing about training-draw variance. Training variance comes
  from the three training replicates only. A training replication (new reps 3–5) is
  out of scope.

Stated priors (not evidence): THR and ALP-Z use only (k, W_k) and assume independence,
so FULL is expected to beat them by about A's margin or more. RLSM and DOS see FULL's
information; A's ARCH/M11 results suggest TIE. DQN: no prior. PPO: Kadyrov et al.'s
plain PPO never beats the Greedy/Tabu heuristics in their tables (METHODS_ALP_PPO §2.4),
so a weak PPO is expected. For the RL arms a **weak tie is the expected outcome**
(section 8 item 8).

### 1.2 Unit, contrast, intervals

Exactly A §1.2: Δ_r(a vs b) = 100·(C_b,r − C_a,r)/C_b,r (positive: a cheaper),
C = mean over the 20,000 test_b days, then over replicates; Δ_i = mean over the eligible
routes of instance i; n = 20; CI95/CI90 = Δ̄ ± t_{19,q}·s/√20 (t = 2.093 / 1.729);
exact Wilcoxon signed-rank on {Δ_i}; cluster bootstrap (10,000 draws, **seed
20261119**) as sensitivity only. Route numbers are descriptive. 10 of the 20 instances
also contain tuning routes (dev days only; stated in the paper).

### 1.3 Primary family (Holm)

Family F = {FULL vs a @ 10^5 : a ∈ {THR, ALP-Z, RLSM, DOS, DQN, PPO}, a passed every
**bug gate** and the **performance bound P1** of section 7}. ALP-Z is included on the same
terms as every other arm (no distinctness rule). |F| ≤ 6, computed mechanically by
`fidelity.py` and pinned by the `TEST_B FAMILY CLOSED` line (section 5.3) **before any
test_b number exists**. Two-sided exact Wilcoxon p per
contrast, Holm over F at FWER 0.05:

| branch | rule | claim template |
|---|---|---|
| **WIN** | Δ̄ ≥ δ **and** p_Holm < 0.05 with Δ̄ > 0 **and** CI95 lower > 0 **and** Δ_i > 0 on ≥ 13/20 | "At 10^5 paths, FULL lowers recourse cost by Δ̄% [CI95; Holm-level CI] versus ⟨arm, citation⟩ trained on the same paths (k/20 instances)." |
| **LOSE** | mirror image | "⟨arm⟩ is Δ̄% cheaper than FULL." Followed by the section 12 threat check. |
| **TIE_STRONG** | not WIN/LOSE and CI90 ⊂ (−δ, δ) | "FULL and ⟨arm⟩ are equivalent within ±δ [CI90]" (+ "although significantly …" if p_Holm < 0.05). |
| **TIE_WEAK** | otherwise | "No difference of ±δ was established (CI95 …)." Never "FULL = arm". |

The multiplicity gate is p_Holm. CI95 is a consistency condition and is reported with the
**Holm-level interval** (two-sided level 1 − 0.05/|F|) beside it. Two-sided p_Holm <
0.05 equals A's one-sided 0.025 per direction when |F| = 1. Equivalence uses the
unadjusted CI90 (A §8 item 3) [DC]; equivalence statements are **per arm, not
simultaneous**, and the paper says so.

Arm-level qualifier: an arm labelled **budget-limited** (section 6.3) carries that label in
every sentence about it; no sentence "FULL beats ⟨arm⟩" is written without it.

### 1.4 Outside the family (reported, never used to rescue or bury an arm)

- **RE-EVAL** (1.1), A's `go_no_go` unchanged (one-sided 0.025 per direction, no Holm).
  Any branch other than WIN is reported with both A's and B's numbers.
- **D_a**: each evaluated arm vs baton_eq @ 10^5, classified by A §1.5 unadjusted.
- **Descriptive test_b rows** for every arm outside F: gate-dropped and P1-failing
  arms. test_b is B's own stream; evaluating them costs minutes and removes no
  information, the family being already fixed.
- **ALP-Z vs THR agreement**: share of identical test_b decisions and the paired cost
  difference; if decisions agree on > 99% of dev states the paper adds "ALP-Z coincides
  with THR on a fixed route" (descriptive; F unchanged).
- **Independence world** (section 3.8): ladder, THR, ALP-Z and FULL on an
  independence-resampled copy of test_b, 12 subset routes, rep 0, in their own gated
  stages.
- **Compute table**: per arm median/max fit seconds, path-stop reads (steps actually
  taken), gradient updates or LP size, parameters, peak RSS, device, between-rep SD.

### 1.5 Forbidden

Arm selection, re-tuning, re-fitting, gate or family changes after the family line is
committed. Every evaluated arm is reported whatever its result. The gate of section 5
enforces the order mechanically from git history; it prevents accidents and leaves an
audit trail, it cannot stop a deliberate bypass (stated in the paper).

---

## 2. Problem and the common interface

A §2.1 unchanged: fixed route, stops j = 1..m, W_k running load (W_0 = 0), breach τ
when W_τ > B, decision epochs k = 1..m−1 on alive paths, handoff at k costs H[k−1],
emergency E[τ−1], 0 on completion (`simulate.simulate_policy`, simulate.py:40-77).

Every arm deploys through `simulate_policy`: a dict {k: [predictor, ...]}; each
predictor has `predict(cache, k, idx) -> np.ndarray` (backward.py:30) and the route hands
off at k iff the mean over the list exceeds H[k−1].
- **cost predictors** (THR, ALP-Z, RLSM, DQN): an estimate Ĉ_k of the continuation cost.
- **decision surrogates** (DOS, PPO): S = H[k−1] + E0·(π − 0.5), π the stop/handoff
  probability; S > H[k−1] iff π > 0.5. DOS has two fold nets, so the fold mean of π
  decides at 0.5 (as `ls_backward` averages fold predictions, backward.py:170-176). PPO
  has no fold structure (one net per job). Surrogates are never read as costs.

Evaluation caches are `RouteCache(spec, pool, stats)` with the `RouteStats` fitted on the
job's fitting rows (A §3.1), stored in the payload.

---

## 3. Arms

### 3.0 Source-verification rule (before tuning)

Read sources: ALP-Z and PPO (**done**, `METHODS_ALP_PPO.md`), THR (Yang, Mathur &
Ballou 2000; Venkateshan & Mathur 2026). Read in revision 3.1 (2026-11-18,
`VERIFY_CITATIONS.md`): Dastpak, Errico & Jabali (GERAD G-2021-53 / arXiv 2109.10200v2;
the C&OR version was not accessible) and github.com/moda707/vrpvcsd_of (DQN); Herrera et
al. 2024 (arXiv 2104.13669v4) and github.com/HeKrRuTe/OptStopRandNN (RLSM); Becker et al.
2019 (DOS); github.com/shiralikadyrov/CVRP_uncertainty (PPO items "not stated" in the
paper). The implementer fills `VERIFY_CITATIONS.md` with each
[VERIFY] value as stated there or "not stated"; a stated value replaces the default, "not
stated" keeps it. Committed and logged before the pilot; nothing in section 3 changes
after the pilot starts except through the grids of section 6.

Common data use. Classical arms (THR, ALP-Z) use all N paths, as baton_eq. Learned arms
(RLSM, DOS, DQN, PPO) use A's split (`backward.split_80_20`): first 80% fitting rows,
last 20% validation rows used only for in-job checkpoint/model selection. Seeds:
`zlib.crc32(f"{inst}|{route_id}|{N}|{rep}|{arm}|{tag}".encode())`. Torch arms use
`torch.set_num_threads(1)` and `torch.use_deterministic_algorithms(True)` (on CUDA also
`CUBLAS_WORKSPACE_CONFIG=:4096:8`).

### 3.1 THR — threshold DP under independence (Yang, Mathur & Ballou 2000; Venkateshan & Mathur, NRL 2026, doi 10.1002/nav.70103)

**THR is a separate arm, not dp relabelled.** The published threshold DP assumes
independent demands; (stop, load) is then Markov with a load threshold per stop. A's
`dp` regresses realised costs on W_k and absorbs the copula dependence; baton_eq is
already a threshold policy (isotonic Ĉ_k), so "thresholds read off baton_eq" would
duplicate it. One sentence in the paper; no duplicate contrast.

Algorithm (per route, all N rows), as revision 1 (indexing checked by the reviewer
against simulate.py:55-77):
1. S_j = the N values of g_j (empirical marginal of stop j).
2. Grid w_0 < … < w_{2000}, uniform on [w_lo, B], w_lo = min over rows and k of W_k
   minus max_j |min S_j|.
3. V_m ≡ 0. For k = m−1, …, 1: C_k(w_i) = mean over g ∈ S_{k+1} of
   [1{w_i + g > B}·E[k] + 1{w_i + g ≤ B}·V_{k+1}(w_i + g)], V_{k+1}(x) =
   min(H[k], C_{k+1}(x)) for k+1 ≤ m−1 (linear interpolation, clipped at w_lo).
   Row chunks of 2,048 grid points, float64.
4. Deploy `GridPredictor(k -> (w_grid, C_k), tie_tol)` with the common tie rule:
   **hand off iff Ĉ_k(w) ≥ H[k−1] − τ, τ = 1e-6·E0** (T1), implemented by returning
   Ĉ_k + τ to `simulate_policy`'s strict comparison (diagnostics subtract τ). THR and
   ALP-Z use the same rule; for THR it changes decisions only on states within τ of the
   threshold. Report thresholds W*_k = min{w_i : C_k(w_i) ≥ H[k−1] − τ} (+∞ if none).

No tuning (2,001 points) [DC]; a 4,001-point refinement is reported on the G2 route-reps
(section 7). Information class (k, W_k).

### 3.2 ALP-Z — Zhu et al.'s ALP on a fixed route (Zhu, Adulyasak & Rousseau, Networks 87(3):266–288, 2026, doi 10.1002/net.70024)

**What the source does** (METHODS_ALP_PPO §1, pp. 270-277). Single vehicle, independent
discrete demands, failure recourse = outsource the shortfall at b per unit, preventive
decision = restock at the depot, routes re-optimised. The value function is decomposed
into a TSP-path travel bound and an **ALP lower bound on the penalty (recourse)
cost-to-go**: LP 24 maximises the **initial-state** value subject to the Bellman
inequalities for both actions, at **every feasible residual capacity** of each sampled
(location, next customer, unvisited set); the basis (eq. 23.2, p. 275) has an intercept
θ_{l,0,q} and coefficients α_{l,j,q} indexed by the load q. The policy restocks iff the
one-step expected approximate cost of restocking is ≤ that of going direct (eq. 29.1).
So it **is** an ALP continuation-value method; revision 1's reconstruction (affine plus
hinge basis in W, uniform objective over sampled loads) was not their method and is
withdrawn.

**What it becomes on our problem.** With a fixed route, location = stop k and the
unvisited set is fixed by k, so the basis is a **free value per (stage, load)**: exact.
Constraint sampling is degenerate (every stage is visited by every policy, all loads are
always constrained). Their ALP is then the exact LP of the independent-demand DP on a
load lattice. It is implemented as published, with these adaptations (paper lists them):
(1) load lattice instead of integer capacities; (2) handoff and breach are terminal with
fixed prices H[k−1], E[k] (their restock/outsourcing continue); (3) HiGHS instead of
Gurobi; (4) no routing part (no TSP bound, no next-customer choice, no heuristic pool).

LP (per route, all N rows):
- Lattice w_i = w_lo + i·h, i = 0..n_w−1, n_w = 401 [DC], h = (B − w_lo)/(n_w − 1), so
  w_{n_w−1} = B. Each stop's increments are rounded to the nearest multiple of h:
  pmf p_j(r) over integer offsets r (binned empirical marginal).
- Variables f_k(i), k = 1..m−1, i = 0..n_w−1 (f_m ≡ 0).
- Handoff rows: f_k(i) ≤ H[k−1].
- Continue rows: f_k(i) ≤ Σ_r p_{k+1}(r)·[E[k]·1{i + r > n_w−1} +
  1{i + r ≤ n_w−1}·f_{k+1}(max(i + r, 0))], with f_m ≡ 0.
- Objective (their eq. 24.1, the initial state W_0 = 0 written one stop later):
  maximise Σ_r p_1(r)·1{i_0 + r ≤ n_w−1}·f_1(max(i_0 + r, 0)), i_0 the lattice index of
  w = 0.
- Solver: `scipy.optimize.linprog(method="highs-ds")`, fixed options
  (`presolve=True`, default tolerances, seeded `random_seed` option = job seed); status
  must be 0. The optimal vertex is **not unique** off the optimal path (values at states
  reachable only after a handoff-optimal state are free below their exact value); this is
  a property of the published objective. The vertex returned is recorded (sha256 of f).
- Deploy (their eq. 29.1): L_k(w) = Σ_r p_{k+1}(r)·[E[k]·1{w + rh > B} +
  1{w + rh ≤ B}·f̃_{k+1}(w + rh)], f̃ linear interpolation on the lattice, f̃_m ≡ 0;
  **hand off iff L_k(w) ≥ H[k−1] − τ, τ = 1e-6·E0** (T1). Ties are **not** rare here:
  with the initial-state objective, a handoff-optimal state only needs f_k = H ≤ L_k, and
  basic solutions are generically tight (L_k = H up to round-off) exactly on the
  handoff-optimal states the policy visits. A strict "L > H" would continue there, the
  reverse of the exact decision. The preventive action winning ties is Zhu et al.'s
  rule (restock iff L^R ≤ L^D, eq. 29.1) and is the same rule THR uses (3.1).
  Tabulated on THR's 2,001-point grid → `GridPredictor`.

No tuning (the source tunes only the heuristic pool and b, neither of which exists
here) [SRC p.278]. Information class (k, W_k).

**Family membership.** ALP-Z enters F on the same terms as every arm (bug gates and
P1); revision 2's 1% distinctness rule is dropped (it could be triggered by vertex ties
and discretisation, not by a method difference). The cost is at most a Holm factor of
6/5. If ALP-Z and THR agree on > 99% of dev decisions the paper says so (T15, 1.4).

### 3.3 RLSM — randomized neural networks for optimal stopping (Herrera, Krach, Ruyssen & Teichmann, Front. Math. Finance 3(1):31–77, 2024, doi 10.3934/fmf.2023022)

A `FitFn` for A's `ls_backward` with A's folds, exactly as M11 (boost) plugs in.
Signature `(k, cache, fit_idx, y_fit, val_idx, y_val, warm)` (backward.py:34-36). At
stage k and fold f: x = `mlp_batch(cache, k, idx)` (15 inputs, float64); random layer
A_kf ∈ R^{15×K}, b_kf ∈ R^K, i.i.d. N(0, 1) [SRC p.5, p.15: standard Gaussian for A
and b; code `Reservoir2` normal(0, 1)]. Drawn per stage (and fold): the paper's Algorithm 1
draws one layer shared by all dates [SRC p.6], and its "RLSMreinit" variant redraws per date
to average out the draw [SRC p.30; code `reinit=True`]; ours is the per-date variant [DC].
h = LeakyReLU(xA + b), slope 0.5 [code: `LeakyReLU(factors[0]/2)` with default factors
(1, 1, 1), RLSM.py:43, configs.py:58; the paper names Leaky ReLU without a slope, SRC p.15];
θ = argmin ‖[h, 1]θ − y‖² + λ‖θ‖² (the source uses plain least squares, no ridge
[SRC p.5; code `np.linalg.lstsq`, regression.py:115]);
prediction clipped to [0, E0]. Fewer than 20 alive rows: A's constant fallback.
Tuning grid: K ∈ {20, 100, 400} × λ ∈ {1e-6·n, 1e-3·n} (6 configs; the ridge axis
gives the regularisation the published method tunes).

### 3.4 DOS — deep optimal stopping (Becker, Cheridito & Jentzen, JMLR 20(74):1–25, 2019)

A `FitFn` in `ls_backward` with A's folds. Network F_θ(x) ∈ (0, 1) on `mlp_batch`
inputs minimises L(θ) = mean over alive fitting rows of [F·H[k−1] + (1 − F)·y]/E0.
Two hidden layers of width 15 + 40 = 55 [SRC p.14: "I = 2 and q1 = q2 = d + 40"; the
paper's depth notation would give one hidden layer for I = 2, but it names two widths, read
as two hidden layers], ReLU hidden activations [SRC p.5], logistic (sigmoid) output
[SRC p.6], batch normalisation, Xavier init and Adam [SRC p.9; placement of batch
normalisation not stated]; batch 8,192 [SRC p.14, fresh paths per step; 2,048 in the
fractional-Brownian example, p.19]; learning rate not stated in the source; steps S_k = clip(⌈20·n_k/8192⌉, 50,
400); warm start from the same fold's stage-(k+1) net, **cold start when that predictor
is a `ConstantPredictor`** (as `neural_fit_fn`, train.py:222-224); checkpoints as A §4.3
selected by L on validation rows. **Adaptation stated in the paper: trained on a fixed
sample** (the source resamples fresh paths every step). Tuning grid: lr ∈ {1e-3, 3e-4}.

### 3.5 DQN — double deep Q-learning with a set encoder (Dastpak, Errico & Jabali, C&OR 159:106338, 2023, doi 10.1016/j.cor.2023.106338; code github.com/moda707/vrpvcsd_of — revision 3.1:
github.com/moda707/vrp-sdo holds only the instances of the 2026 VRP-SDO paper, see
`VERIFY_CITATIONS.md`)

One network for all stages of a route on FULL features: stage-k globals (12) and tokens
(11 features per remaining stop) from precomputed padded tables (M17), tokens padded to
m−1 with a key-padding mask (`SetNetMasked`). The set encoder stands in for their
graph-attention encoder.

Adaptations (stated in the paper):
- Only Q(s, continue) is learned; Q(s, handoff) = H[k−1] is known route data. Greedy
  action: hand off iff Q̂_cont(s) > H[k−1].
- γ = 1 (finite horizon, undiscounted cost); targets and outputs scaled by E0, sigmoid
  head, Q̂ ∈ (0, E0).
- **Behaviour policy = continue-forced** (fixes REVIEW_B finding 2). Because the handoff
  value is known and handoff transitions carry no information, every fitting path is
  run with action "continue" until breach or completion. Q-learning is off-policy, so
  this loses nothing and gives every stage all its alive transitions: the number stored
  for stage k equals the number of alive fitting (row, k) pairs per pass (exact; bug
  gate B-DQN). ε-greedy is therefore not used. The greedy policy is used only in the
  checkpoint rollouts on validation rows.
- **Minibatch 256 with one update per 32 stored transitions** (T10; replay ratio 8
  samples per stored transition). Source values (revision 3.1): the paper samples a
  minibatch of 32 with probability 0.05 at every decision epoch [SRC p.16, p.32], i.e.
  about one update per 20 transitions, ratio ≈ 1.6; the released code uses batch 128 with
  probability 0.04 (config.ini; vrpsd_solver.py:181), ratio ≈ 5.1. T10's ratio 8 is
  therefore above both, not equal to the source's (open item 14). Target period
  expressed in stored transitions.

Training (environment = the job's fitting rows):
- A pass shuffles the fitting rows (seeded) and runs them in lockstep batches of n_env =
  64 through k = 1..m−1 with the emergency check of `simulate_policy` first at every
  stop. Replay stores (row, k, c, done): done with c = E[k] if W_{k+1} > B; done with
  c = 0 if k+1 = m; else c = 0, not done. Capacity 50,000 [SRC p.32; FIFO, p.16; the
  released code uses 100,000]; learning starts after 1,000 transitions [code: updates
  only while the buffer holds > 1,000, vrpsd_solver.py:181; not stated in the paper].
- Double-Q target y = c + (1 − done)·V′, V′ = H[k] if Q_online(s′) > H[k] else
  Q_target(s′). Huber loss (δ = 1 on the E0 scale) [paper δ = 5 on its reward scale,
  SRC p.16; code δ = 1 decayed linearly to 0.25 on demand/50, vrpsd_solver.py:76-77],
  Adam [SRC p.16], gradient-norm clip 10 [DC; neither paper nor code clips], hard
  target update every T stored transitions [SRC p.16, p.32: hard copy every d trials].
  Double Q confirmed in code (online argmax, target evaluation, rl.py:249-253; the paper's
  Algorithm 2, p.30, writes a single-network max).
- Trained on the GPU (header); the kept model is saved to the CPU and evaluated there.
- Checkpoints at update 0, every 1/8 pass and at every pass end; within a job the kept
  checkpoint is the one with the lowest greedy policy cost on the validation rows (ties to
  the earlier). Per-stage transition counts are written into the payload.

Tuning grid: lr ∈ {1e-4, 3e-4} × T ∈ {8,000, 32,000} stored transitions (source period:
d = 1,000 trials [SRC p.32], one trial = one path here, ≤ m−1 ≤ 16 stored transitions,
so ≤ 16,000 transitions, inside the grid's range; the code's 10,000 trials, config.ini,
would be ≤ 160,000); passes P ∈ {1, 2, 3} from the
learning curve (section 6.3).

### 3.6 PPO — Kadyrov et al.'s PPO on the two-action problem (Kadyrov, Azamov, Abdumajitov & Turan, OR Perspectives 15:100351, 2025, doi 10.1016/j.orp.2025.100351)

Settled by the source (METHODS_ALP_PPO §2, p. 3-4), used verbatim:
- Transformer encoder 4 heads, 2 layers, width 64 [SRC p.3] = A's `SetNet` configuration
  (`NetConfig(d=64, layers=2, heads=4, ff=128)`), with masking (`SetNetMasked`).
- GRU memory, hidden 64, on the pooled embedding [SRC p.3]: h_k = GRU([global, masked
  mean of tokens]_k, h_{k−1}), h_0 = 0, updated at every decision step.
- Policy head: one linear layer on h_k, softmax over {continue, handoff}; no action at
  k = m (masking) [SRC p.4].
- Value network: two linear layers, hidden 64, on h_k [SRC p.4] (activation between them
  not stated: ReLU [DC]).
- PPO clipped surrogate ε = 0.2; **one episode per update, 4 epochs per episode**; Adam,
  constant lr 1e-4; gradient-norm clip 1.0; no schedule; no entropy bonus; advantage =
  discounted return − value baseline (no GAE) [SRC p.4].
- **Episode loop, as in the source (n_env = 1):** take the next fitting path, roll it out
  with the current policy (sampling actions), update on that episode (4 epochs, ratios
  against the log-probabilities stored at rollout), then take the next path. There is no
  lockstep batch of episodes, so no update uses data from a stale policy. Features come
  from the job's `PaddedTables`, so the sequential rollout is a gather plus ≤ 16
  batch-1 forwards. The cost is in section 9.

Adaptations (paper lists them): NNConv graph layers dropped (fixed route; tokens of
remaining stops enter the transformer directly); actions {continue, handoff} instead of
next node; reward −H[k−1]/E0 on handoff, −E[k]/E0 on breach at k+1, 0 otherwise (their
coverage shaping +5 per visit and −0.5 per step is not carried over: it would change the
optimal stopping rule); policy-head bias initialised so that the initial handoff
probability is 0.05 (their actions never end an episode early; ours do); episodes replay
the fitting rows; value-loss coefficient 0.5 [DC] and advantage normalisation off. The
code [38] (read in revision 3.1, commit d088960) centres returns by their episode mean
without dividing by a standard deviation, so "off" is confirmed; it has no value network,
so it states neither a value coefficient nor a value activation (the paper/code
differences are listed in `VERIFY_CITATIONS.md` and open item 14).

Deployment: `PolicyPredictor` returns the surrogate H[k−1] + E0·(π − 0.5) with π the
greedy handoff probability. It caches h_k per row between consecutive calls k−1 → k on
the same cache (simulate_policy calls k in increasing order on shrinking row sets) and
recomputes from k = 1 otherwise; cached and recomputed outputs must be equal (test and
bug gate). Checkpoints as DQN.

Tuning grid: lr ∈ {1e-4 [SRC], 3e-4} × γ ∈ {0.5 [SRC], 1.0 (our undiscounted
objective)}; passes P ∈ {1, 2, 3} from the learning curve (3 passes = 240,000 episodes,
above the source's 200,000 [SRC p.4]). γ = 0.5 is myopic here (a breach can be up to 16
steps ahead, weighted 0.5^16); it is in the grid only because it is the source's value
and is not expected to be chosen.

### 3.7 Reference arms

- `nn_full` @ 10^5, reps 0–2: A's `fit_neural(..., "full", ...)`, A's `StageConfig()`;
  no B tuning of FULL. Under cut (i): A's own model files (section 9.2).
- Ladder @ 10^5, reps 0–2, `ladder_labels = ("none", "baton_eq", "oracle")` (no
  `tuned_tau`).

### 3.8 Independence world (descriptive, fixes REVIEW_B finding 5)

THR and ALP-Z assume independent stops; the generator has a Gaussian copula (ρ = 0.6).
To separate method from mis-specification, on the **12 subset routes, rep 0**:
- Stages with `independent=True` (StageSpec): `b_ladder_ind` (none, baton_eq, oracle),
  `b_thr_ind`, `b_alpz_ind`, `b_full_ind`. Training and evaluation pools have each stop
  column permuted by its own seeded permutation (`crc32("perm|inst|route|j|<purpose>")`);
  marginals unchanged, dependence removed. They are fitted at F and listed in the FROZEN
  line and in the FAMILY line's `independence=[...]` list like every other stage, so the
  gate covers them (5.3).
- THR and ALP-Z use only marginals, so `b_thr_ind` and `b_alpz_ind` models must equal
  `b_thr` and `b_alpz` models of the same jobs (sha256 of the tables; bug check).
- In this world THR is the exact optimum up to grid error, so FULL − THR measures FULL's
  approximation loss where THR's assumption holds.
Reported only as: "THR's realised cost on the real stream exceeds its own-model DP value
by x%; in the independent world FULL is within z% of THR." The DP value under the true
copula is not available and is not claimed. Outside F; no decision depends on it.

---

## 4. Matched budget and compute accounting

### 4.1 Budget

Every arm and replicate r ∈ {0, 1, 2} receives exactly the first 10^5 rows of A's
training stream (rep r, chunk 0) on all 62 eligible routes: **10^5 unique simulated
paths**. Arms differ in how often they read them (LS arms one backward pass; DQN/PPO P
passes); reported, not equalised.

### 4.2 Accounting (written into every payload as `budget`)

`unique_paths`; `path_stop_reads` (classical/LS: N·m; DQN/PPO: **environment steps
actually taken**, summed over passes) and `val_reads` separately; `updates` or `lp_rows`,
`lp_cols`, `lp_nnz` (ALP-Z); `n_params`; `fit_seconds` split into `feature_seconds`,
`train_seconds`, `val_seconds`; `peak_rss_mb`; `device`; DQN `transitions_per_stage`; PPO
`visits_per_stage`.

---

## 5. Data streams, test_b, the gate

### 5.1 Streams

| purpose | seed | size | use in B |
|---|---|---|---|
| train, rep r, chunk 0 | `[base, 1, r, 0]` (A) | 10^5 | fitting (all arms) |
| dev | `[base, 3]` (A) | 5,000 | pilot, tuning, gates, sanity |
| test | `base + 99991` (A) | 20,000 | **never read by B** |
| test_std | `base + 99991` (A), first 2,000 test rows, ungated by A's design (paths.py:11-13) | 2,000 | **never read by B** (a test asserts no `expb` module requests it) |
| **test_b** | `[base, 20261118]` | 20,000 | B's final evaluation only |

`base = data.stable_seed(inst)`. The entropy `[base, 20261118]` is unused anywhere in the
repository (reviewer grep). Because `SeedSequence` pads entropy with zeros, M12 compares
**states**: `SeedSequence(seed).generate_state(4)` of test_b against dev, test/test_std
and train (reps 0–2, chunks 0–9) for all 20 instances, all distinct (slow test); plus the
first 100 rows differ.

### 5.2 Code changes (M12) — every literal "test" check generalised

`TEST_STREAMS = ("test", "test_b")` in paths.py. Exact edits (line numbers at 007b542):
- paths.py:45 `PURPOSES` gains "test_b"; `TEST_B_N = 20_000`.
- paths.py:214-227 `_stream`: `"test_b" -> ([base, 20261118], TEST_B_N)`.
- paths.py:126-134 `GateToken` gains `stream: str` and `stage: str | None`.
- paths.py:163-167 `_require_gate(purpose, gate)`: for purpose ∈ TEST_STREAMS requires
  a token with `gate.stream == purpose` (both branches tested).
- paths.py:137-160 A's `verify_test_gate`: the substring test `frozen not in
  read_text()` (152) becomes the anchored regex `^FROZEN COMMIT: ([0-9a-f]{40})$`
  (multiline) with group == frozen; otherwise unchanged (HEAD == frozen, clean tree).
  A's existing log lines have exactly that form (RESEARCH_LOG.md:131, 174); A's gate
  tests stay green. A `FROZEN COMMIT (test_b): …` line can no longer unlock A's stream.
- New `verify_test_b_gate(stage, fit_commit) -> GateToken` (section 5.3).
- paths.py:313, 331 `route_pool`: gate for purpose ∈ TEST_STREAMS through the stream's
  own verifier; size table gains test_b.
- sweep.py:76 `EVAL_SETS = ("dev", "test", "test_b")`.
- sweep.py:105-117 `StageSpec` gains `test_stream: str = "test"`, `arm_cfg: tuple = ()`,
  `independent: bool = False`. B stages set `test_stream="test_b"`.
- sweep.py:275-285 `check_test_gate`: `if eval_set not in TEST_STREAMS: return`;
  refuse unless `eval_set == stage_spec(stage).test_stream`; dispatch to the stream's
  verifier.
- sweep.py:435-445 `run_eval_job`: the three `eval_set == "test"` checks become
  `eval_set in TEST_STREAMS`; for test_b the fit-commit check compares with the stage's
  FROZEN line (5.3), plus `fit_tree_clean is True`.
- sweep.py:543 `merge_stage`: name `<stage>.test_b.csv`.
- Fit refuses (not merely records) a dirty tree for B stages.
Mutation tests remove each of these edits one at a time; each must fail a test.

### 5.3 The test_b gate and the commit sequence

**Why A's design cannot be reused.** A requires HEAD == frozen; the log line naming the
frozen commit cannot be inside that commit, so A read the log from the working tree
(outside the clean-tree check). B reads the log **from the committed blob**, pins it, and
replaces HEAD equality by hash equality of everything the evaluation depends on.

**Definitions (pre-registered now).**
- `norm(x)`: bytes with CRLF replaced by LF. Every text hash is
  `sha256(norm(git cat-file blob <commit>:<path>))`, i.e. of **git-tracked content**, never
  of the working-tree file (the repository checks out with CRLF on Windows).
- `tree_hash(c, P)`: sha256 of the sorted lines `"<path> <text hash>\n"` over every
  tracked file under the pathspec set P at commit c (`git ls-tree -r --name-only`).
- **CODE** = every tracked file under `papers/transvrp/code/transvrp/` (**default-deny**:
  `__init__.py`, `pilot.py`, `analysis.py`, `expb/fidelity.py`, `expb/analysis_b.py`,
  `expb/gatecheck.py`, any new helper are all inside). Tests (`code/tests/`) are outside.
- OWN(arm) = `code/transvrp/expb/<module>.py` (thr, alp for ALP-Z, rlsm, dos, dqn, ppo).
  EXCL(s) = OWN(a) for every arm a ≠ arm(s): the only files allowed to differ between a
  stage's fit commit and the evaluation commit (a repair of another arm).
- ANALYSIS = `expa/analysis.py`, `expb/analysis_b.py`, `expb/fidelity.py`.
- **Model manifest** per stage: `code/results/expB/<stage>/models.sha256`, one sorted line
  per model file `"<sha256 of raw bytes> <key>.pkl <fit_commit> <fit_tree_clean>"`,
  written by `sweep manifest --stage <s>`. Model pickles stay git-ignored; the manifests
  are committed. Reuse stages (cut (i)) get the same manifest over A's files.
- **Family derivation**: `fidelity.py` ends `fidelity_dev.md` with a fenced block
  ```` ```family-derivation ```` holding JSON: per arm every gate value and verdict, and
  the lists `family`, `descriptive`, `dropped` (`arm:gate`), `references`,
  `independence` computed by the rules of section 7. Nobody types these lists.
- Log lines (read from `norm(git show HEAD:papers/transvrp/RESEARCH_LOG.md)`, anchored,
  multiline; list classes cannot contain delimiters):
  - `^FROZEN COMMIT \(test_b\): ([0-9a-f]{40}) stages=\[([a-z0-9_]+(,[a-z0-9_]+)*)\]$`
  - `^TEST_B FAMILY CLOSED: family=\[([a-z0-9_,]*)\] descriptive=\[([a-z0-9_,]*)\]
    dropped=\[([a-z0-9_:,]*)\] references=\[([a-z0-9_,]*)\] independence=\[([a-z0-9_,]*)\]
    fidelity_sha256=([0-9a-f]{64}) manifests_sha256=([0-9a-f]{64})
    code_sha256=([0-9a-f]{64}) analysis_sha256=([0-9a-f]{64})$` (one line).
    `manifests_sha256` = sha256 of the sorted lines `"<stage> <text hash of its
    manifest>\n"`; `code_sha256` = tree_hash(M, CODE); `analysis_sha256` =
    tree_hash(M, ANALYSIS); M is the manifest commit below.
  - `^TEST_B ANALYSIS PATCH: analysis_sha256=([0-9a-f]{64}) date=(\d{4}-\d{2}-\d{2})
    reason=(.+)$`, allowed only below the FAMILY line.

**`verify_test_b_gate(stage, F)`** (git subprocess, fail closed) passes only if all hold:
1. **Clean and no stray bytecode.** `git status --porcelain --untracked-files=all
   --ignored -- papers/transvrp/code/transvrp papers/transvrp/RESEARCH_LOG.md` shows no
   tracked change, no untracked or ignored `*.py`, `*.pyd`, `*.so`, `*.pth`, no `*.pyc`/
   `*.pyo` outside a `__pycache__/` directory (a source-less `.pyc` there is importable),
   and no `__pycache__/<name>.*.pyc` whose `<name>.py` is not tracked (orphan bytecode).
2. **One pinned FAMILY line.** The log has exactly one FAMILY line, every FROZEN (test_b)
   line above it, ANALYSIS PATCH lines only below it. `git log --format=%H -G
   '^(TEST_B FAMILY CLOSED|FROZEN COMMIT \(test_b\))' -- papers/transvrp/RESEARCH_LOG.md`
   returns exactly **one** commit L (the lines were added once and never edited), and L's
   FAMILY line equals HEAD's byte for byte.
3. `stage` appears in exactly one FROZEN (test_b) line, whose hash is F.
4. `git merge-base --is-ancestor F HEAD`, and `git show F:papers/transvrp/RESEARCH_LOG.md`
   contains no FAMILY line and no FROZEN (test_b) line naming `stage`.
5. **Code.** tree_hash(HEAD, CODE) == `code_sha256` (nothing in CODE changed since M), and
   tree_hash(F, CODE∖EXCL(stage)) == tree_hash(HEAD, CODE∖EXCL(stage)) (evaluated code =
   fitted code, file by file, after CRLF normalisation). Reuse stages: the second
   equality is over `data.py, baton/, expa/{features,model,backward,simulate,train,
   baselines}.py` with F = c37ad7c, plus cut (i) condition 3.
6. **Models.** The manifests at HEAD hash to `manifests_sha256`; every model file of the
   stage exists, hashes to its manifest line, the line's fit commit is F and
   `fit_tree_clean` is True, and the file count equals the stage's planned job count.
7. **Family lists.** `fidelity_dev.md` at HEAD hashes to `fidelity_sha256`; its
   family-derivation block parses, and the five FAMILY lists equal the block's lists as
   sets; arm(stage) ∈ family ∪ descriptive ∪ dropped-arms ∪ references, or stage ∈
   independence.
8. **Libraries.** python, numpy, scipy and torch versions of the running process equal
   those recorded in the stage's payloads (CPU evaluation is then deterministic).
It returns `GateToken(commit=HEAD, stream="test_b", stage=stage, family_sha256=sha256 of
the FAMILY line)`. `run_eval_job` writes into **every test_b row** `family_sha256`,
`eval_commit`, `fit_commit`, `model_sha256` and the library versions.

`analysis_b` refuses unless: every row has eval_set test_b; every row's `family_sha256`
equals the hash of HEAD's pinned FAMILY line; all rows share one `eval_commit` (rule: **no
commit between the first and the last step-10 evaluation**; if one happens, all test_b
rows are deleted and re-evaluated at the new HEAD, which the gate still accepts only if
nothing in CODE changed); each stage's rows carry its FROZEN fit commit; and
tree_hash(HEAD, ANALYSIS) equals `analysis_sha256` or the hash in the last ANALYSIS PATCH
line (each patch line is added once, in its own commit, before the patched analysis runs,
with a reason; the paper reports every patch).

**Numbered sequence** (bash, repository root; `S` = every B stage to evaluate on test_b:
`b_ladder b_full b_thr b_alpz b_rlsm b_dos b_dqn b_ppo b_ladder_ind b_thr_ind b_alpz_ind
b_full_ind`, with `b_full_reuse b_ladder_reuse` replacing `b_full b_ladder` under cut (i);
`C=papers/transvrp/code`):
1. `git pull origin main`; all B code and tests committed; `python -m pytest tests -q`
   green (in `$C`); the clean check of gate item 1 passes.
2. `F=$(git rev-parse HEAD)` — the **fit commit**. Nothing in CODE changes from here on,
   except a repair (step 5).
3. `for s in $S; do python -m transvrp.expa.sweep fit --stage $s --jobs 12
   --eligibility results/expA/eligibility_dev.csv; done` (in `$C`; DQN stages run on the
   GPU process). Payloads record `fit_commit=$F`, `fit_tree_clean=True`, library
   versions; a dirty tree aborts.
4. `for s in $S; do sweep eval --stage $s --eval dev --jobs 12; sweep merge --stage $s
   --eval dev; sweep manifest --stage $s; done`; then `python -m transvrp.expb.fidelity
   --out results/expB/fidelity_dev.md` (bug gates, P1, budget labels, family-derivation
   block).
5. Repair (only if a bug gate fails; section 7): fix in that arm's OWN file + test,
   commit → `F2`; refit only that stage at `F2`; redo step 4 for it. A fix needing any
   other CODE file means every stage is refitted at the new commit.
6. **Per-stage code check**: `for s in $S; do git diff --quiet $F_s HEAD --
   $C/transvrp $(for a in <other arms of s>; do echo ":(exclude)$C/transvrp/expb/$a.py";
   done) || echo "FAIL $s"; done` (with `F_s` = F or F2) must print nothing; the gate
   repeats it with CRLF-normalised hashes.
7. `git add $C/results/expB/*/models.sha256 $C/results/expB/fidelity_dev.md
   $C/results/expB/*.csv`; `git commit` → **M** (the manifest commit). `git diff
   --name-only M~1 M -- $C/transvrp` must be empty.
8. `python -m transvrp.expb.gatecheck family-line --commit M` prints the FROZEN lines
   (from the payloads' fit commits) and the FAMILY line (lists copied from the derivation
   block, the four hashes computed at M). Append the FROZEN lines, a short dev summary, and
   the FAMILY line, in that order, to `papers/transvrp/RESEARCH_LOG.md`.
9. `git add papers/transvrp/RESEARCH_LOG.md`; `git commit` → **L** (only the log
   changes); `git push origin main`. Rerun step 6 against L.
10. At L, clean tree, **no commit until the last evaluation**: `for s in $S; do sweep
    eval --stage $s --eval test_b --frozen-commit <F_s> --jobs 12; sweep merge --stage $s
    --eval test_b; done`.
11. `python -m transvrp.expb.analysis_b`; then commit the results (per-day `.npz` stay
    local with a sha256 manifest, as A).

What each commit holds: F holds the code and no B log lines; F2 (if any) a repair of one
OWN file; M the model manifests, `fidelity_dev.md` and dev CSVs, no CODE change; L only the
log lines. Check 4 proves from git history that the family was closed after the fit; check
2 that it was never edited; checks 5–6 that the evaluated code and model files are the
fitted ones; check 7 that the lists follow from the dev gates. test_b chunks are generated
only through a token, which needs the pinned FAMILY line in HEAD's blob. A deliberate
bypass (editing the gate) remains possible; because the gate's files are in CODE, it would
break check 5 and show in history. The paper says both.

---

## 6. Tuning on dev (fixed now)

### 6.1 Routes, score, rule

- **Tuning routes:** the 12 subset routes, replicate 0, N = 10^5, dev stream. Under cut
  (iii): the 3 medium and 3 long subset routes with the smallest (inst, route_id) in
  `eligibility_dev.csv` order.
- **Grids:** THR none; ALP-Z none; RLSM 6; DOS 2; DQN 4 × P; PPO 4 × P. Each config is
  one stage `b_tune_<arm>_c<i>` (dev only, own models dir; `Job.key` carries no config,
  which is safe because the directory is per stage).
- **Score:** mean over the tuning routes of 100·(C_cfg − C_baton_eq)/C_baton_eq on dev,
  baton_eq fitted on the same rows. FULL is never part of a tuning score.
- **Rule:** lowest score; configs within 0.10 points count as tied; among tied, fewer
  passes, then lower total fit seconds, then lower index. Chosen configs and all scores
  are logged before the main fit. A config that crashes or gives a non-finite cost is
  excluded and logged.

### 6.2 What is not tuned

FULL and baton_eq (A's frozen configurations). No dev contrast FULL vs any arm before
test_b. The author has seen A's dev and test FULL numbers; the priors of 1.1 are the
disclosure.

### 6.3 Learning curves and the pass count (fixes REVIEW_B finding 2)

DQN and PPO tuning runs train to **P_max = 3 passes** for every config and save the
in-job-selected checkpoint at the end of passes 1, 2, 3 (selection restricted to
checkpoints up to that pass). Each is scored on dev, giving a 3-point learning curve per
config; (config, P) is chosen jointly by 6.1. These curves survive every cut.
- Pass-end checkpoints of one 3-pass run are the 1- and 2-pass runs exactly, because no
  setting depends on the total number of passes (constant lr, constant target period, no
  ε schedule, no lr schedule); a test checks this on the toy route.
- **Budget-limited label** (pre-registered): the arm is labelled budget-limited if, for the
  chosen config, score(P=2) − score(P=3) > 0.5 points (still improving), or if cut (v)
  forces the main fit to P_main < P* and score(P_main) − score(P*) > 0.5. Otherwise the
  paper writes "not labelled budget-limited", never "converged": the label rests on rep 0
  and the tuning routes and detects only improvement between passes 2 and 3.
- Pilot Q6 checks stage coverage: DQN per-stage transitions equal alive counts; PPO
  visits per stage logged (every stage with alive rows visited ≥ 1,000 times in pass 1
  on the timing route, else listed as a bug-check item).

---

## 7. Gates (dev, main fit, before the family line)

Computed on dev for the main-fit models (62 routes × reps, chosen configs). None compares
an arm with FULL. Thresholds are [DC] unless the gate is an exact identity. **Bug gates**
detect a broken implementation; **P1** is a performance bound.

**Definition (revision 3.2).** *G2 route-reps* are replicate 0 of every eligible route, i.e. the 62 route-reps (rep = 0) of the main fit. The name "G2" was used in revision 3.1 without a definition; the implementation (`expb/fidelity.py`) already used this set. It is a superset of what any identity needs, and is fixed here before the pilot.

| id | arm(s) | type | gate |
|---|---|---|---|
| B0 | all | bug | 100% of jobs fitted; all dev costs finite; C ≥ C_oracle on every route-rep. |
| B-THR | THR | bug, exact under own model | (a) On independence-resampled dev days, simulated THR cost within 3 per-day SE (SE of the 5,000 permuted days) of its DP value E_{g1}[1{g1 > B}E[0] + 1{g1 ≤ B}V_1(g1)] on ≥ **99%** of route-reps. (b) C_k nondecreasing in w on the grid for every k and job (tolerance 1e-9·E0): exact under independence (cf. Zhu et al. Prop. 1, monotone penalty). (c) 4,001-point refinement: thresholds reported, not gated. |
| B-ALPZ | ALP-Z | bug, exact | (a) LP status optimal on 100% of jobs. (b) LP objective equals the lattice-DP initial value (same lattice and binned pmfs, computed by backward recursion) within 1e-6·E0 on 100% of jobs: the basis is exact on a fixed route, so the LP optimum is V*(s_0). (c) Uniform-weight variant of the same LP equals the lattice DP at every lattice point within 1e-6·E0 on the G2 route-reps (detects constraint-assembly errors off the optimal path). (d) **Deployed decisions**: on every dev state the deployed ALP-Z policy visits, its decision (rule T1) equals the lattice-DP decision (same formula with V* in place of f) except where |L^DP − H| ≤ 1e-6·E0, on ≥ 99.9% of visited states (exceptions listed). Off-path vertex differences of the published objective are reported (share of lattice states with f < V* − 1e-6·E0). |
| B-RLSM | RLSM | bug detector only | Ridge normal-equation residual ≤ 1e-8 relative; validation MSE ≤ constant-predictor MSE on ≥ 90% of stages with ≥ 100 alive validation rows. Not presented as fidelity. |
| B-DOS | DOS | bug detector only | Stage objective on validation rows ≤ min(always stop, never stop) + 1e-6 on ≥ 95% of stages with ≥ 100 alive validation rows. Not presented as fidelity. |
| B-DQN | DQN | bug, exact | Stored transitions per stage = alive fitting (row, k) pairs × passes on 100% of jobs; kept checkpoint finite. |
| B-PPO | PPO | bug, exact | Zero actions at k = m; cached and recomputed `PolicyPredictor` outputs equal on 1,000 dev rows per job; finite. |
| P1 | all | performance | Pooled dev cost (sum over routes, mean over reps) ≤ min(C_none, C_h1) + 2·SE, C_h1 = "hand off at k = 1" (cost E[0] if W_1 > B else H[0]), SE from per-day paired SEs summed in quadrature over routes. Every arm's policy class contains "never hand off", so an arm worse than it beyond noise has not found a policy its own class contains. |

Removed from revision 1 (gameable or tautological): DQN "learns 0.98 × update-0", pooled
calibration within 15%, PPO "mean |π − 0.5| ≥ 0.3", ALP lower bound "at sampled states".
Divergence (last vs best checkpoint) and TD residuals per stage are reported, not gated.

Consequences:
- **Bug gate failure**: at most **one repair per arm** (equal for all arms), a bug fix
  only (with a test; a hyper-parameter change is not a repair, grids are section 6),
  logged with its dev evidence, re-fitting only that arm's stage at a new commit, paid
  from the headroom of section 9 (if the refit does not fit, no repair). Still failing:
  `dropped=[arm: gate]`, evaluated on test_b descriptively.
- **P1 failure**: out of F, evaluated on test_b descriptively, no repair. For PPO this is
  consistent with the source (METHODS_ALP_PPO §2.4) and reported that way.
- The family-derivation block of `fidelity_dev.md` (5.3) is computed from these results only.

Dev sanity, all stages (as A): coverage 20 instances / 62 routes / reps, every row's fit
commit equals its stage's, 0 rows below the oracle, cells above 1.005·C_none listed.

---

## 8. Statistics

1. Unit, pairing, intervals: 1.2. All contrasts paired on routes, test_b days and
   training paths.
2. Family F (1.3): Holm at FWER 0.05 over |F| two-sided exact Wilcoxon p.
3. Per contrast: Δ̄, median Δ_i, CI95, Holm-level CI, CI90, instances better,
   rank-biserial, pooled cost ratio, mean GC (A §6), raw and Holm p, branch, labels.
4. Variance as A §8 item 2: between-replicate SD per arm (in the compute table), per-day
   paired SE vs baton_eq, s per contrast.
5. Strata (medium, long) descriptive.
6. Equivalence only by CI90 ⊂ ±δ, per arm.
7. Compute table (4.2): medians and maxima per arm.
8. Power, stated now: CI90 half-width 0.387·s, CI95 0.468·s (n = 20). At A's s = 0.365:
   0.14 and 0.17. For RL arms s of 1–3% is plausible, so TIE_STRONG is unlikely and a
   WIN needs Δ̄ ≥ 0.7 with s ≲ 2; a TIE_WEAK against them is expected and is reported as
   "no difference established".

---

## 9. Compute budget, pilot, cuts

Machine: i9-13950HX, 64 GB RAM. **Platform (fixed, see header):** DQN on the RTX 5000 Ada
(16 GB, torch 2.6.0+cu124), one GPU process; all other arms, all evaluations and the FULL
refit on 12 CPU workers. **Hard ceiling: 7 days = 168 h wall** for tuning + main fit +
evaluation. **Trigger:** cuts are applied in the order of 9.2 until the pilot-measured
worst-case projection (P = 3 for DQN and PPO, measured per-pass times, sequential sum, no
×2 guess) is ≤ 0.8 × 168 = **134.4 h**. The remaining headroom pays for repairs. The pilot
also measures two concurrent DQN processes on the GPU (tables ≈ 0.9 GB each); if they give
≥ 1.5× the throughput of one, the DQN rows below use the measured rate.

### 9.1 Projection (priors; the pilot replaces them)

Priors per job, contention included:
- FULL measured 38.6 min.
- DQN (GPU) per pass ≈ 31,000 updates of batch 256 (≈ 10^6 transitions × replay ratio 8 /
  256) at ≈ 5 ms (launch-bound) ≈ 3 min. Per job, serially on the GPU process: 5 min to
  build `PaddedTables` + 3 min per pass + 1 min of checkpoint validation, i.e. **9 / 15
  min at P = 1 / 3**. On a serial GPU, wall equals job-hours.
- DQN tuning: the `PaddedTables` of the 12 tuning routes (rep 0) are built once on the
  CPU and cached on disk (12 × ≈ 0.9 GB = 1.0 h), shared by all DQN and PPO tuning
  configs; each of the 48 DQN tuning runs then costs 3 × 3 + 1 = 10 min of GPU.
- PPO (CPU, n_env = 1, section 3.6) per episode ≈ 7 ms rollout (≤ 16 batch-1 forwards)
  + 4 updates × 7 ms ≈ 35 ms, × 1.43 ≈ 50 ms; 80,000 episodes ≈ 1.11 h, + validation
  ≈ **1.15 h per pass and job**.

| block | job-hours | wall |
|---|---|---|
| ladder @ 10^5, 186 jobs (CPU) | 1.7 | 0.2 h |
| nn_full @ 10^5 refit, 186 (CPU) | 120 | 10.0 h |
| THR 2 + ALP-Z 5 + RLSM 3 + DOS 10 min, 186 each (CPU) | 62 | 5.2 h |
| DQN main, 186 jobs, P = 1 / 3 (GPU, serial) | 27.9 / 46.5 | 27.9 / 46.5 h |
| PPO main, 186 jobs, P = 1 / 3 (CPU) | 214 / 642 | 17.8 / 53.5 h |
| tuning tables cache, 12 routes (CPU) + DQN tuning 4 × 12 runs (GPU) | 1.0 + 8.0 | 9.0 h |
| tuning PPO: 4 configs × 12 routes × 3 passes (CPU) | 166 | 13.8 h |
| tuning RLSM 6 × 12, DOS 2 × 12 (CPU) | 7.6 | 0.6 h |
| independence world: `b_full_ind` + `b_ladder_ind`, `b_thr_ind`, `b_alpz_ind` (12 jobs each) | 9.2 | 0.8 h |
| dev + test_b evaluation of all stages, merges, manifests, gates | — | 4 h |
| **total, sequential sum, P = 1 / P = 3** | | **≈ 89 h / ≈ 144 h** |

Reconciliation with REVIEW_B2 §4 (≈ 139 h): revision 2's 134.9 h plus the reviewer's
+4 h for per-config tables gives 138.9 h. Revision 3 instead caches the tuning tables
(DQN tuning 9.0 h instead of 12.2 h, −3.2 h), sets the DQN main row equal to its
job-hours (+3.1 h), runs PPO one episode at a time as the source does (+3.7 h main,
+1.0 h tuning), and adds the three gated independence stages (+0.1 h): 143.6 h.

**Trigger outcome under these priors: cut (i) is expected to apply.** 143.6 h > 134.4 h;
after cut (i) (−10.0 h) the projection is 133.6 h ≤ 134.4 h, leaving 34.4 h of headroom
for repairs. The margin is 0.8 h, so a pilot measurement about 1% slower also triggers
cut (ii) (drop DOS, −2.9 h → 130.7 h). With GPU and CPU work overlapped (GPU 54.5 h; CPU
89.1 h on 11 workers ≈ 97 h) the real wall is about 97 h, but the sequential sum is the
binding number. If CUDA fails in the pilot, DQN falls back to the CPU (≈ 3.2 h per pass
and job, ≈ 271 h before cuts) and all five cuts apply (≈ 76 h).

### 9.2 Ordered cuts (least harm to the competitors first; each logged)

(i) **Reuse A's FULL and ladder models (accepted with the reviewer's conditions).**
nn_full and ladder @ 10^5, reps 0–2, are A's `main`-stage model files (fit commit c37ad7c,
`fit_tree_clean == True`), not refits. Allowed only if all hold:
1. Pilot Q3 on ≥ 6 jobs (one short, one medium, one long route; CON3 and SCA3; reps 0–2;
   including A's slowest job): refit at the pilot commit is `np.array_equal` to A's file
   on all 5,000 dev rows for every fold net and ladder model.
2. `git diff --quiet c37ad7c HEAD -- data.py baton/ expa/{features,model,backward,
   simulate,train,baselines}.py` at the pilot commit and at F (gate check 5); and the
   train pool at B equals A's (sha256 of the first 10^5 rows per instance and rep).
3. Re-evaluating A's 186 nn_full and ladder files on dev with B's code reproduces A's
   committed dev rows (`results/expA/main.csv`) exactly (mean_cost bit-equal); this
   covers the changed evaluation path in sweep.py/paths.py.
4. The stage manifest (path, sha256, fit_commit, fit_tree_clean) of the used files is
   committed in M, before the FAMILY line, and enters `manifests_sha256`; the gate
   verifies each file (5.3 check 6).
5. RE-EVAL is then computed with A's own models and the paper says so.
Saves ≈ 10 h; touches no competitor. If the refit fits the budget, refit (fewer special
cases).
(ii) **Drop DOS** (`dropped=[dos: compute cut]`); saves ≈ 3 h. Counted before stop rule 2.
(iii) **Tune on 6 routes** (3 medium + 3 long, 6.1); halves tuning.
(iv) **Replicates 3 → 2 for every arm, FULL and the ladder included** (reps 0, 1), so all
contrasts stay paired in training reps.
(v) **Main fit of DQN and PPO at P = 1** regardless of P*; the 3-pass tuning curves are
kept and the budget-limited rule of 6.3 applies.
If the projection still exceeds 134.4 h: stop rule 4.

### 9.3 Pilot (dev only; CON3-0's eligible routes)

Cells: every B arm at its first grid point, N = 10^4 × reps {0, 1}; one timing cell per arm
at N = 10^5, route 3 (m = 17), rep 0, DQN for one full pass on CUDA (one and two
concurrent processes) and PPO for one full pass on CPU.
- **Q1** all tests green (A's 237 plus B's).
- **Q2** rerunning one cell per CPU arm gives bit-identical dev costs. DQN on CUDA is not
  required to be bitwise reproducible (seed-level replicates, as in A): its rerun
  difference is reported beside the between-rep SD, and its evaluation (CPU, hashed model
  file) must be bit-identical.
- **Q3** cut (i) condition 1 (6 jobs); a failure disables cut (i) and is logged.
- **Q4** C_oracle ≤ C ≤ 1.005·C_none on every pilot cell, or the violations listed with a
  diagnosis (not a gate).
- **Q5** measured per-pass times **of the real update loop including feature tables and
  the 8+ checkpoint validations** replace 9.1; GPU concurrency measured; cuts applied by
  the trigger.
- **Q6** stage coverage (6.3).
The pilot changes no default or grid. A pilot bug is fixed with a test and the cell
re-run; logged.

---

## 10. Implementation plan

Code in `code/transvrp/expb/` (imports `expa` read-only except M12 and M21). Tests in
`code/tests/test_expb_<module>.py`; slow tests marked. **Mutation check** per module,
recorded in the code review.

**M12 — test_b stream and gate** (paths.py, sweep.py, new `expb/gatecheck.py`; edits
listed in 5.2). `verify_test_b_gate(stage: str, fit_commit: str) -> GateToken`;
`route_pool(..., purpose="test_b", frozen=..., stage=...)`; `StageSpec.test_stream`;
`gatecheck.py`: `norm`, `text_hash(commit, path)`, `tree_hash(commit, paths, exclude)`,
`write_manifest(stage)`, `family_line(commit) -> str` (CLI `family-line`), and the
bytecode scan of gate item 1. Tests run on a temporary git repository built in the test.
Tests: (a) SeedSequence states and first rows distinct (5.1), 20 instances (slow);
(b) test_b refused, one test per gate item: FAMILY line only in the working tree; two
FAMILY lines; a FROZEN line below it; the FAMILY line edited in a second commit (pinning,
item 2); stage in no FROZEN line or in two; F not an ancestor; F's blob already holding the
FAMILY line; a change in any CODE file outside EXCL(stage) (including `pilot.py`,
`__init__.py`, `analysis_b.py`); a source-less `.pyc` outside `__pycache__`; an orphan
`__pycache__` file; a model file whose hash differs from its manifest; a missing model
file; a manifest changed after M; a FAMILY list differing from the derivation block; a
fidelity hash mismatch; a library version mismatch; a dirty log or CODE tree;
(c) a CRLF-only change of a CODE file is accepted (normalisation) and a content change is
refused; (d) an A stage `--eval test_b` and a B stage `--eval test` exit non-zero; a test
token does not unlock test_b chunks and vice versa; a `FROZEN COMMIT (test_b): <h>` line
does not satisfy A's gate for h; (e) A's gate tests green; (f) no `expb` module requests
purpose test or test_std (source scan); (g) every test_b row carries `family_sha256`, and
`analysis_b` refuses rows whose hash differs from HEAD's line, mixed eval commits, and an
analysis hash with no matching ANALYSIS PATCH line.
Mutations: substring instead of anchored match; drop each gate item in turn; allow-list
instead of default-deny; hash working-tree files instead of blobs; no CRLF normalisation;
accept stream mismatch; any `in TEST_STREAMS` edit reverted to `== "test"`; reuse
base+99991; use `[base, 20261118]` for two purposes.

**M13 — `expb/grid.py`, `expb/thr.py`.** As revision 1:
```python
class GridPredictor:
    def __init__(self, w_grid: np.ndarray, c: np.ndarray): ...      # c: (m-1, n_grid)
    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray: ...
def independent_dp(spec: RouteSpec, g: np.ndarray, n_grid: int = 2001
                   ) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]
def fit_thr(spec, paths, N, rep, cfg=None) -> tuple[dict, RouteStats, list, dict]  # models, stats, diag, budget
def thresholds(w_grid, C, H) -> dict[int, float]
def permute_columns(g: np.ndarray, key: str) -> np.ndarray
```
Tests: toy 2-stop route vs hand computation (1e-12); with truly independent increments
THR's simulated cost equals the DP value within 3 SE; `thresholds` matches the first
crossing; permutation preserves column marginals exactly; C_k monotone.
Mutations: E[k−1] for E[k]; V_m = H; no clipping at w_lo.

**M14 — `expb/alp.py` (ALP-Z).**
```python
@dataclass(frozen=True)
class ALPZConfig: n_w: int = 401; objective: str = "initial"   # "initial" (Zhu 24.1) | "uniform" (gate B-ALPZ c)
def lattice_pmf(g: np.ndarray, w_lo: float, h: float) -> tuple[np.ndarray, np.ndarray]   # offsets, probs per stop
def build_lp(spec: RouteSpec, pmfs, lattice, cfg: ALPZConfig) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray, dict]
def lattice_dp(spec, pmfs, lattice) -> np.ndarray                                      # (m-1, n_w) by recursion
def fit_alpz(spec, paths, N, rep, cfg: ALPZConfig) -> tuple[dict, RouteStats, list, dict]
```
Tests: (a) toy route: LP objective = lattice DP initial value (1e-9), uniform variant =
lattice DP everywhere; (b) the deploy rule equals THR's on a lattice-aligned toy where
the two coincide exactly; (c) **tie test**: on a toy where the returned vertex has
L_k = H (to round-off) at a handoff-optimal state, the deployed rule hands off, and the
deployed decisions equal the lattice-DP decisions on every reachable state; (d) LP size
equals the layout count; status 0; (e) deterministic given options. Mutations: maximise →
minimise; drop the handoff rows; next-stage value evaluated at i instead of i + r; breach
test ≥ instead of >; strict `L > H` deploy rule (must fail test c).

**M15 — `expb/rlsm.py`.** `RLSMConfig(K=100, slope=0.5, ridge=1e-6, min_rows=20)`;
`rlsm_fit_fn(seed_key, cfg) -> FitFn`; `fit_rlsm(...)`. Tests and mutations as rev. 1
(OLS reduction with K = 1, slope 1; leakage log; seeds; clipping).

**M16 — `expb/dos.py`.** `DOSConfig(lr, batch=8192, s_min=50, s_max=400, width=55,
n_ckpt=4, min_rows=20)`; `DOSNet`; `dos_fit_fn(seed_key, cfg, diag) -> FitFn`;
`fit_dos(...)`. Tests as rev. 1 plus: cold start when `warm` is a `ConstantPredictor`.
Mutations: swap H and y; surrogate sign; no warm start.

**M17 — `expb/padded.py`.** Fixes REVIEW_B 9.5-9.6.
```python
class SetNetMasked(nn.Module):
    # A's SetNet layers (model.py:31-61) with src_key_padding_mask passed to EACH
    # TransformerEncoderLayer of the ModuleList, a prepended False column for the global
    # token, and the token mean over unpadded tokens only.
    def forward(self, glob: Tensor, tok: Tensor, pad: Tensor) -> Tensor: ...
    def pooled(self, glob, tok, pad) -> Tensor: ...         # (n, 2d), used by PPO's GRU
class PaddedTables:
    # built once per job: for every row in `rows` and k = 1..m-1, globals (G=12) and
    # tokens padded to m-1 (F=11), from one stage_batch(cache, k, rows, "full") call per k
    def __init__(self, cache: RouteCache, rows: np.ndarray): ...
    def batch(self, pos: np.ndarray, ks: np.ndarray) -> tuple[Tensor, Tensor, Tensor]  # glob, tok, pad
```
Memory ≈ 0.9 GB per job at N = 10^5 (stored float32 on the job's device).
Tests: (a) SetNet state dict loaded into SetNetMasked gives SetNet's output on unpadded
batches (atol 1e-6), run with `torch.backends.mha.set_fastpath_enabled(False)` inside a
fixture that restores the previous flag; (b) pad-token values do not change outputs;
(c) a mixed-k batch equals per-k unpadded outputs; (d) importing `expb` leaves A's SetNet
outputs bit-identical (the fast-path flag is never changed globally outside tests).
Mutations: mean over pads; mask polarity inverted; no global-token column in the mask.

**M18 — `expb/dqn.py`.**
```python
@dataclass(frozen=True)
class DQNConfig: lr: float = 1e-4; passes: int = 3; n_env: int = 64; buffer: int = 50_000   # n_env: lockstep paths of the continue-forced behaviour
    batch: int = 256; per_update: int = 32; learn_start: int = 1_000; target_every_tr: int = 8_000
    huber: float = 1.0; clip: float = 10.0; ckpt_per_pass: int = 8; device: str = "cpu"
class QPredictor: ...                    # Q̂_cont from the shared net; predict(cache, k, idx)
def fit_dqn(spec, paths, N, rep, cfg: DQNConfig) -> tuple[dict, RouteStats, list, dict]
```
Tests: transitions match `simulate_policy` dynamics; per-stage counts = alive pairs ×
passes; double-Q target on a 2-state toy; 1-stage route converges to the empirical breach
cost within 2%; only fitting rows are environment rows; determinism; pass-end checkpoints
saved. Mutations: single-Q target; handoff transitions stored; behaviour hands off.

**M19 — `expb/ppo.py`.**
```python
@dataclass(frozen=True)
class PPOConfig: lr: float = 1e-4; gamma: float = 0.5; passes: int = 3; epochs: int = 4
    clip: float = 0.2; max_grad: float = 1.0; vf_coef: float = 0.5; ent_coef: float = 0.0
    init_handoff_p: float = 0.05; ckpt_per_pass: int = 8            # one episode per update (n_env = 1), CPU
class PPONet(nn.Module): ...             # SetNetMasked trunk -> GRU(128->64) -> policy (64->2), value (64->64->1)
class PolicyPredictor: ...               # surrogate, h_k cache
def fit_ppo(spec, paths, N, rep, cfg: PPOConfig) -> tuple[dict, RouteStats, list, dict]
```
Tests: discounted returns with γ = 0.5 and 1 on a hand trajectory; clipped objective on a
hand batch; one episode per update and 4 epochs (update counter); no action at k = m;
initial handoff probability 0.05; cached = recomputed predictor; 1-stage toy converges to
the cheaper action on ≥ 95% of states; determinism. Mutations: no ratio clipping;
advantage sign flipped; reward not scaled by E0; GRU state not carried.

**M20 — `expb/codecs.py`, `expb/registry.py`.** `B_ARMS = ("thr", "alpz", "rlsm",
"dos", "dqn", "ppo")`; `ARM_POLICY_B = {a: a for a in B_ARMS}`; `fit_arm(arm, spec,
paths, N, rep, cfg) -> (models, stats, diag, budget)`; `pack_models(models, arm, E0) ->
dict`, `unpack_models(blob) -> dict[int, list]`. Round trip gives `np.array_equal`
predictions on 1,000 dev rows per arm; a shared DQN/PPO net is stored once per job.
Mutation: unpack builds an untrained net.

**M21 — sweep integration** (sweep.py, as boost was added). Fixes REVIEW_B 9.1-9.2:
- `run_fit_job` (sweep.py:380-400): a branch `job.kind == "neural" and job.arm in B_ARMS`
  **before** the generic neural branch, unpacking **four** values and storing `budget`;
  the stage's `arm_cfg` passes the config.
- `_pack` (289-306): first line `if arm in B_ARMS: return codecs.pack_models(...)`;
  `_unpack` (309-323): first line `if blob["arm"] in B_ARMS: return
  codecs.unpack_models(blob)`.
- `ARM_POLICY.update(ARM_POLICY_B)`; `fields_from_key` already accepts the new tags.
- `StageSpec.independent` permutes train and evaluation pools (3.8).
- Stages: `b_smoke` (1 route, N = 2,000, every arm, tiny configs, dev only), `b_pilot`,
  `b_tune_<arm>_c<i>`, `b_ladder`, `b_full`, `b_<arm>` per arm (eligible routes, reps
  0–2, N = 10^5, chosen config, test_stream "test_b"), `b_full_ind`; under cut (i)
  `b_full_reuse`, `b_ladder_reuse` reading A's files through the manifest. Results under
  `code/results/expB/`.
Tests: `b_smoke` fit + eval + merge gives exactly {none, baton_eq, oracle, nn_full, thr,
alpz, rlsm, dos, dqn, ppo}; eval reruns leave model mtimes unchanged; A's sweep tests
pass; A and boost stages' job lists unchanged (snapshot of `plan_stage` keys).

**M22 — `expb/fidelity.py`.** `gate_report(stage, arm, dev_df, models_dir, rows_dir) ->
dict` per section 7; budget-limited labels; the family-derivation JSON block; writes
`results/expB/fidelity_dev.md`. Tests: each gate just above and below its threshold.
Mutations: ≥ → > at a threshold; P1 against the oracle instead of none.

**M23 — `expb/analysis_b.py`.** Fixes REVIEW_B 9.9-9.10. Never calls
`expa.analysis.write_summary` (eval_set guard analysis.py:486, `FAMILY_SIZE = 11`
analysis.py:38); reuses `instance_deltas`, `holm`, `_ci`, `_wilcoxon_p`, `go_no_go`.
```python
def load_stage_tables(paths: list[Path]) -> pd.DataFrame   # asserts equal test_b day counts, one eval_commit, per-stage fit commits
def family_table(df, family: list[str], delta: float = 0.7) -> pd.DataFrame
def go_no_go_holm(d: np.ndarray, p_holm: float, delta: float = 0.7) -> str   # includes the 13/20 count
def holm_ci(d: np.ndarray, size: int) -> tuple[float, float]
def re_eval(df) -> dict                                                      # A's go_no_go, unadjusted
def compute_table(models_dirs: list[Path]) -> pd.DataFrame
```
Reads family/descriptive/dropped from the FAMILY line in HEAD's committed log. Tests: A's
M10 vectors with p_holm = raw p reproduce A's branches; WIN unadjusted with Holm p 0.06 →
TIE; |F| = 6 Holm on known p; refusal on a test row, on mixed eval commits, on a fidelity
hash mismatch. Mutations: raw p for Holm; 12 wins accepted.

---

## 11. Order of work

1. Revision 2 (this file) → independent re-review of the changed sections → log.
2. Source verification for DQN, RLSM, DOS and the PPO code (3.0) → `VERIFY_CITATIONS.md`
   → log.
3. Implement M12 → M23, tests green before the next module.
4. Independent code review; mutation results recorded.
5. Pilot (9.3) → report → GPU concurrency measured → cuts by the trigger → log.
6. Tuning (6) → chosen configs, learning curves, labels → log.
7–10. Steps 2–11 of section 5.3 (fit, dev gates, manifests, family line, test_b, analysis).

---

## 12. Honest threats, stop rules

- **Our adaptation may be weaker than the authors' own.** Mitigations: sources read
  (ALP-Z, PPO) or to be verified (3.0); grids per arm with learning curves (6); bug gates
  that are exact where possible (7); every adaptation listed (3.2, 3.5, 3.6). Residual:
  sources verified in revision 3.1; the remaining departures from them (DQN replay ratio,
  encoder, lr schedule and episode count; PPO paper/code differences; DOS step count) are
  listed in item 14 and `VERIFY_CITATIONS.md`.
- **ALP-Z is near-identical to THR on a fixed route** (exact basis). It stays in F by
  rule (cost: at most a Holm factor 6/5); agreement is reported and the paper says it
  plainly.
- **Information class / independence.** THR and ALP-Z use (k, W_k) and assume
  independence, which the copula violates by construction. The independence world (3.8)
  measures how much of their loss is mis-specification.
- **Adaptations in the arms' favour** (known handoff value for DQN, continue-forced data,
  PPO's initial bias, exact price schedules, grids for arms only) make a FULL WIN
  conservative. **Against the arms:** fixed sample (no fresh simulation, A §1.1), PPO
  without the routing hybrids that were Kadyrov et al.'s best, DOS on a fixed sample;
  each named in the paper.
- **Budget-limited RL arms**: labelled by 6.3 rather than silently compared.
- **Shared tuning routes**: 12 routes in 10 of the 20 instances (dev days only); stated.
- **Generator and cost model** as A §10.
- **A's test stream not reused**; A's numbers quoted only as A's.

**Stop and redesign if:**
1. Any M12 gate test fails, or RE-EVAL cannot be computed.
2. Fewer than 3 published arms in F after gates, repairs and cut (ii) (the comparison is
   then too thin to be called "published methods"; reported as such).
3. Any policy below the oracle on any dev or test_b route.
4. The projection exceeds 134.4 h after all cuts.
5. Pilot Q2 fails for a CPU arm after one fix attempt, or DQN's CPU evaluation of a
   fixed model file is not bit-identical.

---

## 13. Out of scope

Three-action version (restock); amortised networks across routes; re-planning; DQN/PPO
with fresh simulation beyond the 10^5 paths; the arXiv 2026 Dastpak–Errico–Jabali method
as such (ex-ante outsourcing); Zhu et al.'s routing re-optimisation and Kadyrov et al.'s
routing hybrids (no routing decision on a fixed route); tuning FULL; any change to A's
code beyond M12 and M21; a training replication with new reps.

## 14. Open items for the author

- Sign off the design thresholds of section 0 (no pilot evidence; frozen after the pilot).
- Cut (i) is accepted with conditions and is **expected to apply** under the priors
  (9.1); the pilot decides, and a slower pilot also triggers cut (ii).
- Source verification (revision 3.1) found departures that are design questions, not
  numbers, and were therefore **not** changed: (a) DQN: the source network is an MLP on a
  hand-engineered fixed-size observation, not a graph-attention encoder (3.5's "stands in
  for their graph-attention encoder" is inaccurate); T10's replay ratio 8 exceeds the
  source's ≈ 1.6 (paper) / ≈ 5.1 (code); the source decays lr 1e-3 → 1e-4 and trains 3–5
  million trials, against our constant {1e-4, 3e-4} and ≤ 240,000 paths. (b) PPO: the code
  of [38] has no value network (baseline = episode-mean return), uses an entropy bonus
  0.01, and its GRU state does not reach the policy output, against the paper's value
  network, no entropy term and GRU memory (3.6 follows the paper). (c) DOS: the source
  trains 3,000 + d steps per date on fresh batches; ours ≤ 400 steps on a fixed sample.
  (d) RLSM: the source regularises by limiting K, not by ridge (3.3's grid note).

## 15. Changelog

### Revision 3 → revision 3.1: source verification (2026-11-18)

Step 2 of section 11; details, page and line references in `VERIFY_CITATIONS.md`. Only
[VERIFY] values and their source notes changed; no threshold value changed.
- **RLSM (3.3, M15):** N(0, 1) confirmed; per-stage draw recorded as the paper's
  "reinit" variant; Leaky ReLU slope 0.01 → **0.5** (stated in the authors' code; the
  paper gives no slope); plain least squares in the source noted.
- **DOS (3.4):** two hidden layers of d + 40, ReLU, sigmoid output, batch norm, Xavier,
  Adam, batch 8,192 confirmed (depth-notation ambiguity noted); lr not stated.
- **DQN (3.5):** code URL corrected to github.com/moda707/vrpvcsd_of (vrp-sdo is the
  instance repository of the 2026 paper); "32 per 4, same replay ratio 8" replaced by the
  source values (paper batch 32 at probability 0.05 per epoch, code 128 at 0.04); capacity
  50,000 confirmed (paper); learning start 1,000 confirmed (code); Huber, Adam, hard target
  and double Q sourced; gradient clip marked [DC]; target period sourced as 1,000 trials.
  T10 itself unchanged (open item 14).
- **PPO (3.6, T12):** code read; advantage normalisation off confirmed; value coefficient
  and value activation stay [DC] because the code has no value network. T12's values are
  unchanged; its rationale text is updated.
- Sections 3.0, 12 and 14 updated to the verified state; design departures listed in 14.

### Revision 2 → revision 3 (REVIEW_B2)

- **N1 family pinned:** `fidelity.py` writes a machine-readable family-derivation block;
  the gate checks the FAMILY lists equal it; the FAMILY and FROZEN lines must be added in
  exactly one commit and never edited (`git log -G`); every test_b row records
  `family_sha256`, and `analysis_b` refuses a mismatch (5.3 items 2, 7).
- **N2 model manifests:** a sha256 manifest per stage, committed in M before the FAMILY
  line (L), hashed into `manifests_sha256`, verified per file (5.3 item 6).
- **N3/N3b code hashes:** default-deny CODE set (all git-tracked files under
  `code/transvrp/`, analysis and fidelity modules included) hashed from git blobs with CRLF
  normalised; stray and orphan bytecode refused; library versions checked; analysis
  changes after the family line only through pinned ANALYSIS PATCH lines (5.3).
- **N4 ALP-Z:** deploy rule "hand off iff L ≥ H − 1e-6·E0" for ALP-Z and THR (ties to the
  preventive action, Zhu eq. 29.1); "ties have probability 0" corrected; deployed-decision
  bug gate against the lattice DP; 1% distinctness rule dropped; ALP-Z unconditionally in
  F on the same terms as every arm; agreement with THR reported descriptively (3.1, 3.2, 7).
- **N5 sequence:** per-stage diff (step 6); gated independence stages `b_ladder_ind`,
  `b_thr_ind`, `b_alpz_ind`, `b_full_ind`; no commits between test_b evaluations; CRLF
  normalisation; delimiter-free list classes (3.8, 5.3).
- **N6 PPO:** one episode per update with n_env = 1 (sequential rollouts, no stale-policy
  data); γ = 0.5 myopia noted; "not labelled budget-limited", never "converged" (3.6, 6.3).
- **Compute:** DQN rows equal job-hours on the serial GPU; tuning tables cached; PPO cost
  at n_env = 1; totals ≈ 89 / ≈ 144 h; cut (i) expected to apply (9.1).
- **Platform:** DQN on RTX 5000 Ada (torch 2.6.0+cu124), all else CPU 12 workers; GPU
  nondeterminism handled by seed-level replicates and CPU evaluation of hashed models
  (header, 9.3).
- **Section 0:** design thresholds needing author sign-off.

### Revision 1 → revision 2 (REVIEW_B)

- **Gate (REVIEW_B 1, critical):** log read from the committed blob; HEAD equality
  replaced by ancestor + per-stage diff on SHARED ∪ OWN; anchored regexes for both
  streams (A's too); every literal "test" check generalised with exact line list;
  FROZEN lines name their stages; family/descriptive/dropped enforced per stage;
  fidelity-file hash in the family line; numbered command sequence (5.3); SeedSequence
  state test; test_std excluded.
- **DQN/PPO training (REVIEW_B 2, critical):** DQN continue-forced behaviour (no
  ε-greedy truncation), batch 256 / per-update 32 at the source's replay ratio; PPO
  follows Kadyrov et al. (one episode per update, 4 epochs, lr 1e-4, clip 1.0, γ = 0.5
  grid point, GRU, value net 2×64) with an initial handoff bias; learning curves to 3
  passes on dev in every tuning run; pass count chosen by a dev rule; budget-limited
  label; extra grid axes (DQN target period, PPO γ).
- **Compute (REVIEW_B 3):** table recomputed with tuning tied to the grids, feature
  tables and checkpoint validation included; GPU device rule; cuts reordered (reuse A,
  drop DOS, 6 tuning routes with both strata, reps 3 → 2 for all arms, P = 1 last);
  trigger "apply in order until ≤ 0.8 × ceiling"; repairs paid from headroom.
- **Cut 2 (REVIEW_B 4):** accepted with the five conditions plus a dev re-evaluation
  identity check against A's committed dev rows.
- **ALP (REVIEW_B 5; source now read):** arm redesigned as ALP-Z following Zhu et al.
  (per-load basis, initial-state objective, all loads constrained, eq. 29.1 policy);
  revision 1's affine-hinge reconstruction withdrawn; distinctness rule vs THR;
  independence-world evaluation (3.8); G2-ALP replaced by exact LP-vs-DP identities.
- **Fidelity (REVIEW_B 6, 8):** RLSM ridge axis; DOS "fixed sample" and cold start;
  bug gates separated from the performance bound P1; gameable DQN/PPO gates removed;
  G2-THR at 99% with per-day SE of permuted days; one repair per arm, bug fixes only.
- **Statistics (REVIEW_B 7):** REP → RE-EVAL; Holm-level CI; per-arm equivalence; power
  and expected weak ties; dropped and P1-failing arms evaluated descriptively; tuning
  instances stated.
- **Code (REVIEW_B 9):** run_fit_job four-value branch, `_pack/_unpack` hooks,
  SetNetMasked per-layer masks and fast-path handling, `PaddedTables` instead of a
  mixed-k `padded_batch`, analysis_b never calls `write_summary`, ladder labels.
- **Minor (REVIEW_B 10):** 4,001-point refinement reported; actual steps counted;
  bootstrap seed 20261119; wording rule; stop rule 2 counts cut (ii); seed-reuse mutation.

- **Revision 3.2 (2026-11-18, before the pilot):** defined "G2 route-reps" (section 7); `dos.py` records the never-stop validation objective that gate B-DOS reads (previously missing, which would have dropped DOS for a defect of ours). No threshold of section 0 changed.
