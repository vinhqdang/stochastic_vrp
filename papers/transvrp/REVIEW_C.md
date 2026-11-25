# Review of EXPERIMENT_C.md (revision 1)

Reviewer: independent pass, 2026-11-25. Read: EXPERIMENT_C.md (all); EXPERIMENT_A.md sections 1.3-1.5 (delta, classes);
REVIEW_B2.md (style); RESEARCH_LOG.md (A's M11 entry, B's tuning, cuts, gates); code: `data.py`, `expa/features.py`,
`expa/boost.py`, `expa/paths.py`, `expb/thr.py`, `expb/grid.py`, `expb/rlsm.py`, `expb/stages.py`, selected parts of
`expb/fidelity.py`, `expb/gatecheck.py`, `expb/dos.py`; `results/expA/analysis_test_m11/summary.md`,
`results/expB/*.csv` (fit_seconds), `results/expB/tuning_record.json`, `b_tune_dos_c{0,1}.csv`.
Read-only except this file. One scratch computation (breach rates of the 62 routes under rho = 0, .3, .6, .9, 20,000
fresh draws, seed 12345; script in the scratchpad, not in the repo) and a few pandas/git queries.
`expc/` does not exist, so gate findings are against the spec text plus the B code it generalises.

## Verdict: ACCEPT WITH REQUIRED CHANGES

The question set is the right one and the design reuses B's machinery sensibly. Q1 (nested feature masks) and Q2
(copula worlds) are implementable as written. But two points make the pre-registered *decision rules* unable to
answer Q1 and Q2 (findings 1 and 2: both blocking, both text changes), the model-based rival is specified at a
resolution that will probably make it look worse than it is (finding 4), the learner chosen as stand-in captures
only about 70% of the gain it is supposed to attribute (finding 3), and the test_c scope is under-specified
(finding 6). Nothing needs a redesign. Fix 1 and 2 before the pilot, because the pilot is where sections 1-6 freeze.

## Findings

### 1. [blocking] delta = 0.7% cannot resolve Q1 or Q2: the whole effect being attributed is 0.82%

Location: header ("Conventions are A's"), 1.7 (classes: "better requires Delta >= delta"), Q1 predictions, Q2 text
"inside +-delta at W0".

The rich-state gain that Q1 splits into feature groups is, in A on test, FULL vs BATON = +0.823% [0.65, 0.99]
(RESEARCH_LOG; summary.md). delta = 0.70% (A section 1.3, 3 x refit noise 0.22%) was chosen to separate a *headline*
effect from refit noise. Consequences in C:
- C1 (split vs load) is "better" only if BOOST-F1 recovers >= 0.7/0.82 = 85% of FULL's entire gain over BATON. BOOST
  itself is 0.254% dearer than FULL (M11, [0.17, 0.34], 20/20), i.e. BOOST-F3 recovers about 0.57% in total, so even
  the best-case C1 cannot reach delta. C3 ("smaller than C1, better or inconclusive") can never be "better". C2
  "equivalent" holds for any |difference| below roughly 0.5% (CI90 half-width is 0.39 s, s about 0.1-0.4): it is
  satisfied by almost anything. The family therefore returns "inconclusive" or "equivalent" for every informative
  outcome, whatever the truth.
- Q2: "BATON vs THR and BOOST-F3 vs BATON inside +-delta at W0" is the same trivial band; at W3 the expected gain is
  below delta too. The point estimates with intervals are fine; the classes attached to them are not informative.
- The A/B precedent does not transfer: B's contrasts were between arms differing by 0.16% to 11%.

Fix (both, pre-registered now, [DC] marked):
(a) Primary Q1 estimand = **share of the total gain**: R_x = (C_F0 - C_Fx)/(C_F0 - C_F3) per instance (ratio of
instance-mean costs, or ratio of instance means; bootstrap-by-instance CI95), with classes on R (for example "most":
CI95 lower > 0.5; "little": CI95 upper < 0.2) in addition to the delta classes on Delta; require
the denominator CI to exclude 0 or report "not estimable".
(b) For the Holm family C use a smaller, justified delta_C (for example 0.25%, about the refit-noise RMS 0.22%; with
s about 0.3 the CI90 half-width 0.12% still permits a strong tie). State that delta_C is not A's delta and say why.
(c) Delete the predicted classes that are structurally impossible under the old delta (C1 "better", C3).

### 2. [blocking] The estimand is undefined in W0 (and fragile in W3) as specified

Location: 1.4 ("eligibility is not re-applied"), 1.7, 7. A's Delta_r = 100 (C_b - C_a)/C_b (A section 1.3).

Eligibility (breach >= 2%, m >= 4) was set on W6. My check (20,000 draws, any breach of the route): under rho = 0
the route breach rate is 0 for 2 of the 62 routes and below 0.1% for 6 (min 0.0, median 0.44); under rho = .3 the
minimum is 0.6%, under .6 and .9 it is 2.5% and 4.9%. For those routes C_b is 0 or tiny in W0 (cost equals recourse
cost; B already met zero-SE routes in THR's gate), so Delta_r is 0/0 or dominated by one rare day, and `instance_deltas`
as written yields NaN/inf that propagates into the instance mean. W0 is the anchor of the dose-response curve.

Fix: pre-register for every world (a) the aggregation as instance-level ratio of summed route costs, Delta_i = 100
(sum_r C_b,r - sum_r C_a,r)/sum_r C_b,r over the instance's routes (A's route-mean-then-instance mean is the
fragile part, not the t interval), or normalise route Delta_r by E0 rather than C_b; (b) a rule for instances whose
denominator is below a floor (report, do not drop silently); (c) the world-specific breach table (min/median/number
below 0.1%) as part of the Q2 output, with the number of routes that never breach in the training stream.
Run the same aggregation in W6 to confirm it reproduces A's numbers (G-W6-type unit test).

### 3. [required] BOOST is not a stand-in for FULL at this resolution; C5 mixes learners

Location: 1.2 (justification), 1.7 (C5, C6).

A's M11 shows BOOST dearer than FULL by +0.254% [0.17, 0.34] on 20/20 instances; "equivalent" is the +-0.7% band of
finding 1. Relative to FULL's 0.82% gain that is a 31% shortfall, and RLSM and DQN (B) are closer: +0.16% [0.09,
0.24] and +0.16% [0.11, 0.21]. Hence: (i) the top of the Q1 ladder (BOOST-F3) is not the rich-state policy of the paper;
(ii) C5 (BOOST-F3 vs CDP, "learners vs model-based") can flip sign depending on the learner, and the reader will
read "learner" as FULL; (iii) C6 uses FULL while C1-C5 use BOOST, so the family mixes learner classes. The spec
says "A and B showed ... equivalent to FULL within delta" - true only at the wide delta.

Fix: make **RLSM the co-primary Q1/Q2 learner** (its fit is 43 s/job, as cheap as BOOST, and it is closer to FULL); C1-C3
become a pair (BOOST, RLSM) with the rule "learner-dependent" if the two classes differ. C5 becomes **FULL vs CDP**
(FULL is evaluated on test_c anyway through `c_full_w6`); keep BOOST-F3 vs CDP and RLSM-F3 vs CDP descriptive.
Quote "BOOST recovers about 70% of FULL's gain" in the section 1.2 justification.

### 4. [required] CDP: three accuracy defects and a missing-bias statement

Location: 1.5.

Algebra checked: with z_j = sqrt(rho) Z + sqrt(1-rho) eps_j, the posterior of Z has precision 1 + k rho/(1-rho) and
mean sqrt(rho) sum z/(1 - rho + k rho), variance (1-rho)/(1-rho + k rho). The spec's mu_k and s_k^2 are **correct**.
The conditional marginal construction (z_j given the factor, mapped through the fitted gamma marginal) is correct for
W0-W9 with separate delivery/pickup copulas (generator: two independent `sample_demands` calls, equicorrelation over
all instance customers, any subset has the same rho). "Exact at rho = 0 equals THR" is right up to marginal fit and
grid (rho-hat is about +-0.003, not 0, and marginals are moment-fitted gamma versus THR's empirical resampling).
Defects:
(a) **Node-grid quadrature is under-resolved.** Nodes are spaced 0.5 in zeta; the posterior SD is s_k = sqrt(0.4/(0.4 + 0.6k)) =
0.34 (k = 5), 0.25 (k = 10), 0.19 (k = 17) at rho = 0.6. The Riemann sum of C against a Gaussian of SD s on spacing h has relative
error about exp(-2 pi^2 (s/h)^2): 6% at k = 17, and the posterior at rho = 0.9, k = 10 (s = 0.105) is a single node
(error about 40%, with the nearest node possibly 0.25 = 2.4 s off). CDP is fitted in W6 (and W0) only, but C4/C5 are exactly
about CDP at W6, and the same code in other worlds would be unusable.
(b) **S = 64 Sobol atoms per stop** cannot represent the tail that drives the decision: the handoff region is where
P(breach next stop) is of order H/E (a few percent); the atom granularity is 1/64 = 1.6%. THR uses all N = 10^5 empirical increments
and a 2,001-point grid; CDP uses 64 atoms and 401 grid points, and B already saw THR's own gate fail at the 2,001-point grid
(88.7% < 99%). The G-CDP gate (W0, 4 SE, 95%) will mix this discretisation error with real bugs.
(c) Compute is not the obstacle: 289 pairs x 2,001 grid x 256 atoms = 148M interpolations per stage, about 17 stages: a few minutes
per route.
(d) Direction of the approximation bias is not stated. Averaging DP values that each assume the factor is known gives a value
that is **lower** than the true continuation cost (the policy is credited with future learning it does not have), so CDP
hands off too late in expectation. The paper must say this; the spec only says "ignores what later stops will reveal".

Fix: (a) interpolate C_k(w | zeta) in zeta (cubic) and integrate the posterior with Gauss-Hermite (7-9 points per factor) around
(mu_k, s_k), or use a node spacing at most 0.5 s_min (about 33-65 nodes per factor); (b) S >= 256 (or 1,024) with the Sobol
points common across nodes, w-grid 2,001 as THR; (d) add the sentence on bias direction. Fix the numerical settings now; a
reported sensitivity run (true rho 0.6 plugged in; finer grid) in W6 on dev separates *estimation*, *discretisation* and
*approximation* so that a CDP loss in C4/C5 is attributable (otherwise C5 "worse" cannot be interpreted).

### 5. [required] G-CDP does not test the part that matters, and its SE is ambiguous

Location: section 6.

(a) The gate runs at rho = 0, where the factor posterior is inert: node weights, the 2-D pairing of delivery/pickup nodes, the
per-stage conditional tables and the posterior indexing (decision at k uses scores of stops 1..k, tables for stops k+1..m) are
untested by the gate. The unit test "posterior vs brute-force" covers the algebra only.
(b) "within 4 SE (per route, replicate mean)": SE of what? Both policies are evaluated on the same dev days, so the SE of the
paired difference is the relevant one; when the two policies agree on most days this is tiny and a 1e-3 relative cost difference caused by
discretisation will fail it; if the policies are identical the SE is 0 (the spec excludes zero-SE THR routes but not zero paired SE).
Fix: define SE_paired; pass if |difference| <= max(4 SE_paired, 0.002 E0) [DC value stated now]; define behaviour for SE = 0.
(c) Add an **identity gate in W6**: simulate with the true latent factors available from an explicit-factor generator (test
generator, not the production stream) and deploy the factor-known policy C(. | zeta_true); its expected cost equals the DP's
E_zeta[V_0(zeta)] exactly (up to MC error and grid), a check of tables, conditional marginals and the wiring of both factors at rho > 0.
Also keep B's monotonicity-in-W check for each node table.
(d) Implementation note for MC3: `RouteCache` keeps only g, W, P, D, standardised z-sums (not p_j, d_j). The predictor needs normal scores of each
p_j, d_j; they can be recovered as P_j - P_{j-1}, D_j - D_{j-1} (float64, exact to rounding), or the cache must be extended. Say which,
and test it.

### 6. [required] test_c scope, B's models on test_c, and what is read where

Location: 3, 4, 4.6, 5, 1.7 ("PPO-long vs ... B's PPO").

(a) Section 3 defines test_c per world (`[base, 7, w]`) but section 1.7 only requires W6 for family C. State which
stages are evaluated on test_c (all? W6 only?) and which Q2 table is read from test_c versus dev. If Q2 is dev-only, say so;
dev was used for gates and the pilot, so it is not a clean estimate. If Q2 reads test_c in all five worlds, every Q2 stage is in the
FROZEN line and in `manifests_sha256` (about 30 stages x 124 model sets: say so).
(b) Section 1.7 compares PPO-long with "B's PPO on the subset (rep 0)". B's PPO was evaluated on test_b; on test_c it needs an eval-only
stage with B's `b_ppo/models.sha256` exactly as `c_full_w6`. Add `c_ppo_b_w6` (12 routes, rep 0). Same for B's `b_ladder` if the BATON arm
in the gap-closed statistic should be B's rather than a refit (the refit is deterministic and will agree; say so).
(c) Reuse of B's FULL models: the gate must accept a **fit commit that is not F_C** for this stage (B's gate requires the fit blob at the
stage's fit commit, and `check_libraries` compares versions). Specify the exception and add a test that FULL's predictions on a fixed 1,000-row dev
sample are **bit-identical** under F_C code and under `bf3edab` (MC1/MC2 touch `data.py` and `features.py`, which FULL's predictor imports). Library
versions (torch, numpy, scipy) must equal those recorded in B's payloads or the stage records the difference.
(d) No leakage: test_c is a new seed (`[base, 7, 16]`), FULL's training stream is A's `[base,1,r,0]` and
nothing in B's tuning touched it. I find no leak, but the spec should say "B's FULL models were fitted on A's train stream, no C data enter".
(e) G-W6: the check is "hash of the dev chunk of CON3-0"; extend to one chunk per purpose (train rep 0, dev) for 3 instances, cheap.

### 7. [required] DOS configuration provenance contradicts the log

Location: 1.6.

RESEARCH_LOG 2026-11-20 03:30 says "DOS tuning (c0, c1) is sunk by cut (i) and is not used", and `tuning_record.json` has only dqn, ppo, rlsm. C now
selects DOS c0 from "B's committed dev tuning scores (c0 -0.60, c1 +1.55)". I recomputed them from `b_tune_dos_c{0,1}.csv` on the subset6 routes (3 medium + 3 long):
c0 -0.60, c1 +1.55: **the numbers match**, but c1's mean is driven by two medium routes at +11.2% and +8.4% (c0 is positive nowhere beyond +0.03 on long
routes). The code (`dos.py`, `features.py`, `backward.py`) is unchanged between the tuning commit `155cb19` and HEAD (git diff empty), so the tuning is valid for the
frozen code.
Fix: run `expb.choose` for DOS before F_C so that a committed record (with hashes of the two CSVs and the rule outcome) exists; correct the log sentence
("sunk" -> "used in C, recorded here"); keep the two-route instability as a disclosure in the paper (DOS is not robust at lr 3e-4).
The CSV rows are tracked but `results/expB/b_tune_dos_c*/rows` are untracked: hash them or add them to the manifest.

### 8. [required] Feature sets: nesting is correct; four design problems

Mapping verified against `features._stage_arrays`: g1 = W/B (col 0), g2 = peak (1), g3 = k/m (2), g4 = (m-k)/m (3), g5 = m/22 (4),
g6 = H/E0 (5), g7 = E/E0 (6), g8 = P/B (7), g9 = D/B (8), g10 = zbar_p (9), g11 = zbar_d (10), g12 = g_k/B (11); `HISTORY_IDX = (1,7,8,9,10,11)` is g2, g8..g12.
So F0 = {g1, g3-g7} + s1-s3 is exactly A's SCALAR arm (history globals zeroed). F0 is a function of (k, W_k) and route constants: s1 = min_j (B - W - sum mu_g)/s_hat_j, s2 its
argmin, s3 = min H are all functions of (k, W); so **F0 is in BATON's information class**, and A's results (SCALAR = DP-1e6 equivalent, FULL vs SCALAR = +0.826 against FULL vs BATON = +0.823)
support that. Nesting F0 < F1 < F2 < F3 and F0 < F0n < F3 holds; F3 equals all 12 globals; `mlp_batch` columns 12-14 are s1-s3.
Problems:
(a) Models are fitted **per route and per stage** (`boost.fit_fn(k, ...)`). At fixed k, E[P_k], E[D_k] are constants, so P/B and D/B already
contain the deviations; zbar (g10, g11) differs from them only by weighting stops equally rather than by size. "F2 = standardised deviations" is not new
information, a re-weighting; and since zbar (relative deviations, equal weights) is closer to the sufficient statistic of the factor than P_k (size-weighted),
the spec's prediction "C2 equivalent" may be wrong (zbar can help). Say this in 1.3 and report F2 vs F1 and F3 vs F2 separately instead of C2 = F3 vs F1 alone.
(b) C2 as defined (F3 vs F1) is a composite of g10, g11, g2, g12; add F2 vs F1 (deviations) and F3 vs F2 (peak, last increment) descriptive; add F1 vs F0n (split vs net
history) descriptive: F0n and F1 are not nested and C1 vs C3 are the only comparison.
(c) RLSM masks: with a fixed 15 x K random layer, specify **zero-masking the input columns while keeping the same A, b** (random features of F0 then
live in the span of F3's), not redrawing a d_set x K layer; K = 100 is then the same number of random features for every set, which gives the small sets
more features per input dimension (capacity confound, state it). For BOOST, depth-3, 120 trees is fixed across sets: sets with more features need
interactions that depth 3 may not use. The conclusion "set X carries no value" is therefore a statement about *these learners*; require BOOST and RLSM to agree or
report "learner-dependent".
(d) F0 for BOOST includes s1-s3 and A's BOOST did not; BOOST-F0 vs BATON is reported descriptively (good): it calibrates the estimator handicap (1-D regression through depth-3
trees versus BATON's isotonic fit). Make the pre-stated expectation explicit: BOOST-F0 >= BATON cost; C1 and C3 are within-learner contrasts and are not affected.

### 9. [required] Seeds, WS construction, and the stream-distinctness test

Location: 1.4, 3.

(a) `numpy.random.SeedSequence` pads entropy lists with zeros to the pool size 4, so `[base, 3]` and `[base, 3, 0]` give the **same** state. The spec's new
tuples (`[base, 3, w]`, `[base, 1, r, c, w]`, `[base, 7, w]` with w in {10, 13, 16, 19, 26}) are distinct from A/B (`[base,1,r,c]`, `[base,3]`, `base + 99991`,
`[base, 20261118]`) because w != 0, but the distinctness test must compare **the first 1,000 draws of each stream**, not the tuples, and must state that
W6 train and dev are *by design identical to A's* (so "No C stream equals an A or B stream" is true only for test_c and for the non-W6 worlds).
(b) W6 reproducing A/B bit for bit is implementable: keep the two sequential `sample_demands` calls (deliveries then pickups) and the Cholesky draw
(`rng.standard_normal((N, m)) @ L.T`) unchanged for default arguments; W0 with `rho = 0` goes through the `rng.gamma` branch (different draw path than the copula;
equal in distribution) and W3/W9 through the same Cholesky branch with changed rho. The chunk cache key (`chunk_file`) is keyed by the seed repr and not by the
world: safe only because every world has its own seed; add the world name to the key.
(c) **WS is under-specified.** "One equicorrelated copula over all deliveries and pickups of the instance" needs: a single joint draw of dimension
(m_d + m_p) from one Cholesky (about 100 x 100, fine, 80 MB per chunk); a decision on whether the *same customer's* d_i and p_i get correlation rho
(it follows from "all", but is a strong choice: the net g_j = p_j - d_j then has a much weaker common-factor load than in W6); which RNG call order;
and what ratio of a stop's pickup and delivery means makes WS dependence harmless. Expect, and pre-state, that WS has a **smaller** factor effect on net load than W6
(shared factor cancels in p - d), so a smaller rich-state gain at WS than W6 is an implementation-consistent outcome, not a puzzle; say that WS differs from W6 in two ways
(cross-correlation of p and d and of same-customer p, d), so it is not a clean single-factor contrast.

### 10. [required] Compute: add the estimate; the pilot must exclude PPO-long

Location: 8.

The spec gives no projection ("must stay under 72 h") and no numbers. Estimate from measured per-job times (12 concurrent workers; B's `fit_seconds`: THR 331 s,
RLSM 43 s, BOOST 44 s in A, DOS about 170 s on the tuning routes, ladder 0.7 s per job; 124 jobs = 62 routes x 2 reps):

| item | job-hours | wall (12 workers) |
|---|---|---|
| THR, 5 worlds (331 s x 124 x 5) | 57 | 4.7 h |
| BOOST, 13 stages (about 55 s with token summaries) | 25 | 2.1 h |
| RLSM, 5 stages (43 s) | 7.4 | 0.6 h |
| DOS (about 170 s x 124) | 5.9 | 0.5 h |
| CDP, 2 stages as specified (about 60 s; with the settings of finding 4 about 150 s) | 4 to 10 | 0.3 to 0.9 h |
| ladder, 5 worlds | 1 | 0.1 h |
| stream generation (5 worlds x 20 instances x train/dev/test chunks) | about 1 | 0.1 h |
| evaluation: dev (5,000 days) about 32 stages, test_c (20,000 days), if all stages (B: 11 stages = 1.65 h wall) | about 35 | about 3 h |
| gates and fidelity report (B's THR gate alone took 13.9 h wall; C does not run B-THR, assume 3 to 5 h) | n/a | 3-5 h |
| **subtotal without PPO-long** | about 140 | **about 15 h** |
| **PPO-long, 12 routes, rep 0, 9 passes** | 12 x 17 h | **about 21.5 h** |

B measured 3 passes at 5.65 h mean and 7.2 h maximum on the 12 subset routes (rep 0: 4.3 to 7.2 h), so 9 passes are 12.8 to 21.5 h per route (mean 17 h), wall about 22 h
since all 12 run concurrently on the 12 workers. The spec's "about 18 h" is the mean, not the wall; and 12 PPO jobs occupy the 12-worker pool, so either PPO-long runs
alone (total about 15 + 22 = 37 h) or concurrently with the rest on the other 12 cores of the 24-core machine (then about 22 h plus contention). Either is below 72 h; the
cut order of section 8 will not trigger. The **pilot** ("all C stages ... on CON3-0's routes") includes PPO-long at 9 passes on four routes (m = 9, 12, 12, 17), i.e.
a 10 to 20 h job for information B already has (time is linear in passes); exclude PPO-long from the pilot (run one pass for a smoke check) and state the linear scaling.
Also add the DOS and CDP times to the cut list; if the CDP settings of finding 4 are used, recompute.

### 11. [required] Statistics: family composition, Holm, and independent choices

Location: 1.7, 7.

(a) C1-C3 share BOOST-F0 and C4/C5 share CDP; Holm is valid under arbitrary dependence, so Holm over six is correct. With finding 3, family C becomes C1-C3 for BOOST and for RLSM
(six contrasts) plus C4, C5 (FULL vs CDP), C6: nine; Holm over nine costs a factor 0.67 at the smallest p, acceptable with n = 20 and p-values near 1e-5. Alternatively declare RLSM Q1 descriptive
as spec'd, but then "learner-dependent" cannot be a family-level statement. Choose now.
(b) C4 as CDP vs BATON conflates the DP engine and the information: B found THR (same DP engine as CDP) 13.2% dearer than BATON. Add CDP vs THR (descriptive) so the value of the factor posterior with the
engine held fixed is visible, and BATON vs THR from B as context.
(c) The "share of gap closed" request: A's data show BATON is 17.1% above the clairvoyant oracle (median) and FULL's gain is 0.82%, so FULL closes about 5% of that gap; the
table is correct but needs the framing that most of the gap is irreducible (a clairvoyant oracle sees the future demands). Pre-state the denominator (BATON to oracle, not none to oracle; A's `gap_closed` column measures none-to-oracle and gives 0.758 for FULL).
(d) A dose-response "no trend test" is acceptable; a pre-stated direction ("BOOST-F3 vs BATON and BATON vs THR increase with rho") is a hypothesis without a test; either test it (a simple
monotone contrast on Delta_i across four worlds, Page/Jonckheere on instance means, unadjusted) or label it "pattern, not a result".
(e) 20 instances x 62 routes: the route-level SDs (A's between-replicate SD 0.13% median) are fine; the unit rule is unchanged. Good.

### 12. [suggested] A value-of-information ceiling and a mis-specification test

(a) The referee's natural next question is "how much of the 0.82% is all there is?". The factor-known DP (policy C(. | zeta_true), cost E_zeta V_0, simulated on the explicit-factor test generator of
finding 5c) is the ceiling on what any policy using the factor information can achieve, and costs minutes. Report BATON -> F3 -> CDP -> factor-known -> oracle on dev as a descriptive ladder;
it also explains why the gap to the oracle stays large.
(b) CDP is cheap: run it in **all** copula worlds, W0-W9 and **WS**. WS is the natural mis-specification test (CDP assumes separate factors; the data have one shared factor): the model-free learners should be unaffected and CDP should lose,
which is the real selling point of learning versus model-based. A second dev-only world with a different copula (t copula, nu = 4) or lognormal marginals would make the point without the shared-factor interpretation.
(c) Add RLSM-F0 and RLSM-F3 to all five worlds (about 0.1 h wall each) so the Q2 curve does not rest on the weakest learner (finding 3).

### 13. [suggested] What a Transportation Science referee would still object to

1. **Practical significance.** The headline is +0.82% against 17% above the oracle; even a 4-fold dependence dose (W9) may give a few percent. Say plainly that the contribution is *which information and when*, not a large cost reduction; give the dose-response in cost units per day as well as percent.
2. **Synthetic dependence.** One-parameter equicorrelated Gaussian copula across all customers, fixed CV 0.3 and gamma marginals, synthetic last-mile cost model: no spatial or time structure, no empirical demand data. C addresses sensitivity to rho and one shared-factor variant; the referee will ask for a spatially structured copula (distance-decay correlation) and for real data. A second dependence family (finding 12b) is the cheapest answer.
3. **CDP as strawman.** A model-based rival that (i) gets the generator's true model class, (ii) is described as an approximation that ignores learning, (iii) is coarse (finding 4) will be called a straw man; the sensitivity runs in finding 4 and the factor-known ceiling in 12a neutralise this.
4. **Selection of 62 routes on W6 breach.** The W0/W3 results are on routes selected for dependence-driven breach; report all routes (m >= 4 and the 18 ineligible) in an appendix table at least for the ladder, or state selection explicitly with the breach-by-world table of finding 2.
5. **Single plan.** The 'Det' plans are fixed for all worlds; the online policy is evaluated for plans not optimised for the world (appropriate for execution, but state it).
6. **Estimator versus information.** Q1 identifies information value through two learners; add the coarse check that all four learner families (BOOST, RLSM, DQN, FULL) from A/B give the same ordering for F0 vs F3 (already in hand: ladder from B) to defend against "learner-specific" and note that the F-ladder is within-route, per-stage so "information" means *route-local* information.
7. **Reporting.** Pre-registration discipline is good (dev-only pilot, family line, hashes); keep in the paper that the families were frozen before test_c was read and the one deviation list.

## Remaining required changes (checklist)

1. (1, blocking) Share-of-gain estimand and delta_C (or equivalent) for Q1; drop structurally impossible predictions; same for Q2 wording.
2. (2, blocking) Define Delta in degenerate worlds (instance-level ratio or E0 normalisation), breach-by-world table, NaN rule, unit test reproducing A in W6.
3. (3) RLSM co-primary; C5 = FULL vs CDP; state BOOST recovers about 70% of FULL's gain.
4. (4) CDP: Gauss-Hermite/interpolated zeta integration, S >= 256, 2,001-point grid, bias-direction sentence, sensitivity runs.
5. (5) G-CDP: paired SE with a floor, identity gate at rho > 0 via factor-known simulation, monotonicity per node, cache recovery of p_j, d_j.
6. (6) test_c scope, eval-only stages for B's PPO models, FULL reuse exception and bit-identical prediction test, library versions.
7. (7) Run `choose` for DOS and commit the record, fix the log sentence, hash DOS tuning rows.
8. (8) Feature-set clarifications (per-stage models, zero-masking for RLSM, F2 vs F1 and F3 vs F2, F1 vs F0n), correct the C2 prediction logic.
9. (9) Seed tests compare draws, W6 equality statement, WS construction details, cache key with world, WS expectation.
10. (10) Add the compute table, run PPO-long outside the pilot, state wall about 22 h and the 24-core arrangement.
11. (11) Family composition and Holm count after changes; CDP vs THR descriptive; gap-closed denominator; dose-response hypothesis label.
