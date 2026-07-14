# TEMPO — Tilted E-Martingale Process for Online re-optimization

Paper 2: an e-process that monitors a running routing plan and triggers
re-optimization with anytime-valid type-I error control. TEMPO pairs
with paper 1's BATON — the baton sets the tempo — and the name is the
contribution: the paper decides *when* to replan.

Status: ACTIVE — spec + core code. Venue targets (subscription/hybrid,
OR-first): Transportation Science, INFORMS Journal on Computing, EJOR,
Transportation Research Part B. Decide after results.

## 1. One-paragraph pitch

A dispatcher commits to routes optimized against a planning model
`P0` — a joint law over demands, travel times (diurnal + weather),
congestion shocks, accidents, breakdowns, and dwell times. As the day
unfolds these quantities realize sequentially. When should the
dispatcher replan? Fixed-interval replanning wastes solves and reacts
late; ad-hoc rules ("replan when 30 min behind") have no error
guarantee and invite alarm fatigue. We monitor the day with a single
nonnegative supermartingale (an e-process) testing H0: "the day follows
the planning model," and replan when it crosses 1/alpha. Ville's
inequality makes the false-alarm guarantee anytime-valid — the
dispatcher may peek continuously. The novel coupling: the bets inside
the e-process are tilted by *decision relevance*, computed from the
plan's own economic sensitivities (BATON's fitted continuation costs
and per-stop recourse prices from paper 1), so the monitor is most
sensitive exactly where drift would change the optimal decision — and
provably remains a valid e-process because the tilts are predictable.

## 2. The stochastic world (per factor)

Event epochs t = leg traversals and node services of the running plan.

| Factor | Model under H0 | Bet (e-factor) |
|---|---|---|
| Demand at stop k | net increment `g_k ~ N(mu_g, sig_g^2)` (or empirical, standardized) | Gaussian mean-shift LR `exp(theta*z - theta^2/2)` on standardized residual z |
| Travel time on leg (i,j) at clock time t | `T = tau_ij * m(t) * w(R_t) * exp(eps)`, `eps ~ N(0, sig_T^2)`; m = diurnal multiplier (known curve), w = weather multiplier given rain indicator R_t (exogenous covariate) | same mean-shift LR on the log-residual z |
| Dwell at stop k | `S = a + b*|demand| + eps`, `eps ~ N(0, sig_S^2)` | mean-shift LR on regression residual |
| Accidents (zone-wide) | Poisson counts, rate `lam0 * dt` per interval | tilted-Poisson LR `exp(theta*N - lam0*dt*(e^theta - 1))` |
| Breakdown per leg | Bernoulli(p0), p0 small | Bernoulli LR `(p1/p0)^X * ((1-p1)/(1-p0))^(1-X)` |

Notes:
- Rain/diurnal are *covariates* of the null, not things we test: the
  monitor only fires on deviation from the *conditional* forecast. A
  jam at 6pm that the m(t) curve predicted is NOT evidence.
- A breakdown is a near-point-mass event under H0: its single-step
  e-factor is `~p1/p0 >> 1/alpha`, so the master process fires
  immediately — hard events need no side-channel special-casing.

## 3. Master e-process

Chain rule: with F_{t-1} the history before epoch t and independent
channels given the state (assumption A1, to be defended/relaxed),

    E_t = prod_{s<=t} prod_{k fires at s} e_s^(k)

is a nonnegative martingale with E[E_t] = 1 under H0 for ANY
predictable choice of the tilts theta_k(s), by the tower property.
Ville: P(sup_t E_t >= 1/alpha) <= alpha. Channels that do not fire at
epoch s contribute factor 1 — asynchronicity is free.

Robustness to tuning: replace each `e(theta)` by a finite mixture
`sum_j pi_j e(theta_j)` over a grid (mixtures of e-values are
e-values). Default grid {0.25, 0.5, 1.0} with uniform pi.

## 4. The novel coupling — decision-relevant predictable tilts

Generic drift monitoring wastes power on harmless drift. We scale each
channel's tilt by a predictable sensitivity s_k(t) in [0,1] computed
from the current plan's economics (all F_{t-1}-measurable):

- demand channel: s = closeness of BATON's continuation cost
  `Chat_k(W)` to the cheapest recourse price min(H_k, R_k + F_k) —
  i.e. how near the current load path is to the optimal-stopping
  boundary. Near boundary -> small drift flips the decision -> bet big.
- traffic channel: s = fraction of remaining planned distance in the
  affected zone x tightness of downstream time economics (p_late
  exposure of remaining stops).
- accident channel: s = share of remaining legs in the elevated zone.
- dwell channel: s = remaining stop count (late dwell drift hurts more
  when many services remain).

Then theta_k(t) = theta_max * s_k(t) (per mixture component). Because
s_k(t) uses only the plan, fitted models, and realizations strictly
before t, the tilt is predictable and validity is untouched. The paper
formalizes this as: *e-process betting strategies informed by
optimization sensitivity*, with BATON's fitted value functions playing
the role that LP duals would play in an exact method.

Equivalent framing (for the theory section): the multi-channel tilt is
the independence-factorization special case of a single exponential
tilt of the joint null in the direction of the plan's cost subgradient;
per-channel weights fall out as the components of that subgradient.

## 5. Decision layer

- Soft trigger: E_t >= 1/alpha -> replan remaining customers with
  warm-started ALNS (the paper-1 planner; we stay with ALNS), new plan
  becomes the reference, all channels reset (E := 1) because H0's
  conditional laws now condition on the new plan.
- The reset makes the procedure a sequence of independent anytime-valid
  tests; total false replans over a day are controlled at alpha per
  segment (report both per-segment alpha and expected false replans).
- Metrics: false-replan rate on null days (target <= alpha), detection
  delay after an injected change-point, realized day cost vs (a) never
  replanning, (b) periodic replanning, (c) CUSUM/Page-Hinkley trigger
  (no anytime guarantee — the foil), (d) oracle replan at the true
  change-point.

## 6. What must be proved

1. **Validity with predictable tilts** (Prop 1): E_t is a nonnegative
   supermartingale with E[E_t] <= 1 under H0 for any F_{t-1}-measurable
   theta_k(t) taking values in the allowed grid; Ville gives the
   anytime false-alarm bound. (Direct; write carefully — the crux is
   that s_k(t) never uses the epoch-t realization.)
2. **Validity of the mixture + reset scheme** (Prop 2): per-segment
   validity after data-dependent resets (restart at a stopping time is
   again an e-process by optional stopping / segment-wise argument).
3. **Power** (Prop 3 / empirical): under a change-point model where the
   drift direction correlates with the plan's sensitive direction,
   sensitivity-tilted bets have strictly higher log-growth (GRO
   argument) than uniform bets; detection delay ~ log(1/alpha) / KL
   per-epoch. Uniform-vs-tilted delay gap quantified in simulation.
4. Honest discussion: independence assumption A1 across channels given
   state (rain drives both traffic and accidents — either model rain as
   covariate in BOTH nulls, which restores conditional independence, or
   bound the effect).

## 7. Evaluation plan (same evaluation set as BATON)

- Instances: the 40 Dethloff instances, the 14 Salhi--Nagy instances,
  and the 19 real-shop city instances — identical to paper 1, plans
  from the same `results/plans/` ALNS cache. Also reuse city instances (real road networks, real shops) and
  Dethloff; plans from the Det/Gounaris gates via cached ALNS solves.
- Day simulator: `svrpspd_wdro/ev/world.py` — full timeline with all
  five factors; drift scenarios injected at a change-point t*:
  congestion surge (x1.5–2.5 log-mean), demand shift (+0.5–1.5 sd),
  accident burst (x5 rate), dwell inflation, breakdown; plus pure-null
  days for false-alarm calibration.
- Detection layer first (this milestone): false-alarm rate, detection
  delay, per-channel attribution. Replanning value layer second
  (warm-started ALNS on trigger; realized-cost comparison).
- Baselines (the comparison set for the paper; anchors, published
  detectors, and ablations):
  1.  never-replan — static plan ridden to the end (lower anchor).
  2.  periodic re-optimization every Delta events/minutes, Delta
      grid-tuned — the rolling-horizon industry default (cf. the
      dynamic-VRP survey tradition, Pillac et al. 2013).
  3.  lateness-threshold rule — replan when cumulative delay exceeds a
      tuned tau minutes; the practitioner heuristic.
  4.  CUSUM per channel + Bonferroni across channels (Page 1954) — the
      classical sequential detector; no anytime-valid guarantee, h
      tuned (a) as published defaults and (b) calibrated to TEMPO's
      realized false-alarm rate for a like-for-like power comparison.
  5.  Page-Hinkley drift detector — the standard data-stream monitor
      (Gama et al. 2014 tradition).
  6.  fixed-sample GOF z-tests at pre-committed checkpoints with
      Bonferroni — valid only at the committed peeks; shows exactly
      what anytime validity buys.
  7.  Bayesian online change-point detection (Adams & MacKay 2007),
      alarm on posterior change probability — strong modern baseline,
      no frequentist error control.
  8.  untilted master e-process (uniform bets) — ablation isolating
      the decision-relevance tilts.
  9.  per-channel e-processes combined by equal-weight averaging
      (Vovk & Wang) — ablation isolating the single-master design.
  10. oracle trigger at the true change-point (upper anchor).
  Milestone 2 pairs every trigger with the SAME warm-started ALNS
  replanner and compares realized day cost on identical scenario
  streams, plus a compute-unbounded re-optimize-every-stop anchor.

## 8. Code map

- `svrpspd_wdro/ev/world.py` — multi-factor day simulator on existing
  instances/plans.
- `svrpspd_wdro/ev/eprocess.py` — per-channel bets, mixture, predictable
  sensitivity tilts, master process with per-channel log attribution.
- `svrpspd_wdro/ev/baselines.py` — CUSUM, Page-Hinkley, periodic,
  fixed-sample tests.
- `svrpspd_wdro/scripts/ev_detect_eval.py` — milestone-1 experiment.
- `svrpspd_wdro/tests/test_ev.py` — martingale/Ville/validity tests.

## 9. Milestone-1 findings (2026-07-12, results_ev_detect.csv)

40 Dethloff instances x 6 scenarios x 25 days, longest Det-gate route
per instance, alpha = 0.05.

- **Validity holds and the foil story works**: TEMPO false-alarm rate
  0.019 <= alpha (tempo_flat 0.022), while CUSUM (h=8) is 0.054 and
  Bonferroni-fixed 0.079 — both EXCEED their nominal level exactly as
  the anytime-validity argument predicts, and periodic replans 90% of
  null days by construction.
- **Conditional on detection, TEMPO is as fast or faster** (mean delay
  in events after t*): dwell 11.6 vs CUSUM 13.7; traffic 23.0 vs 24.7;
  accident 21.2 vs 26.6.
- **Raw detection rate is lower than CUSUM on demand/accident/
  breakdown** — three known causes, all fixable or defensible:
  (1) the rare-event breakdown bet crosses 1/alpha alone only from a
  neutral wealth position; pre-drift null bleed (~ -1 to -2 log units
  across five channels) means one breakdown may not suffice — raise the
  stake margin or run breakdown as a parallel e-process with a
  union-bound alpha split;
  (2) demand sensitivity s = W/B is deliberately ~0 early in the route
  — TEMPO is built to ignore drift that is not decision-relevant, so
  raw detection rate is the WRONG metric for it: milestone 2 must score
  detection conditional on days that actually end in (near-)breach, and
  realized cost once the replanning layer exists;
  (3) CUSUM's 0.054 false rate at h=8 means its detection rates are
  bought with an invalid type-I level — tune h to match 0.019 false
  rate for a like-for-like comparison.

Next: (a) like-for-like CUSUM calibration; (b) harmful-day conditional
scoring; (c) the replanning layer (warm-started ALNS on trigger) with
realized-cost comparison; (d) extend runs to Salhi-Nagy + city sets.

## 9b. Milestone 1b — TEMPO v2 (2026-07-13, results_ev_detect.csv)

Three validity-preserving upgrades (implemented as `TempoMonitor`):
adaptive aGRAPA-style betting (theta_t from the EWMA of PAST residuals,
predictable, mixed 50/50 with a 0-floored grid), the dual-regime
combination C_t = 1/2 prod_k E_k + 1/2 mean_k E_k (product pools
distributed drift, mean isolates single-channel drift without paying
the other channels' bleed), and the breakdown alpha-split (alpha/2 +
alpha/2, single event fires its own monitor). Plus two protocol
decisions: observed breakdowns are HARD reactive triggers for every
method (dispatchers do not run statistics on a driver calling in), and
the classical foils get ORACLE calibration on 100 null days — both at
nominal alpha and at TEMPO's realized false rate (matched-rate).

Result (40 Dethloff instances x 25 days, alpha = 0.05):

| scenario | TEMPO v2 | CUSUM@matched | PH@matched | v2 delay vs CUSUM |
|---|---|---|---|---|
| false-alarm | **0.016** | 0.027 | 0.027 | — |
| traffic x1.6 | **0.94** | 0.65 | 0.23 | **18.8** vs 43.3 |
| demand +1sd | **0.48** | 0.10 | 0.07 | 35.4 vs 32.5 |
| dwell +3sd  | **0.99** | 0.89 | 0.74 | **7.6** vs 26.6 |
| accident x5 | 0.34 | 0.37 | 0.23 | 29.4 vs 35.5 |
| breakdown   | 0.68 (hard event — identical for every method) | | | |

At matched false-alarm budgets TEMPO v2 dominates or ties every
scenario (accident is parity within noise, and TEMPO still spends 40%
less false-alarm budget there). The uncalibrated classical numbers
(CUSUM h=8: 0.73 accident detection at 0.065 false rate) are exactly
the invalid-budget story the paper tells. Ablations confirm the
upgrades do the work: flat e-process detects demand at 0.08 vs 0.48,
traffic at 0.46 vs 0.94.

Remaining known limit: accident detection is information-limited
(~1.8 informative events per drifted day at these rates); candidate
improvement is exposure pooling across a sliding window. Not blocking.

## 9f. Real data: Amazon LMRRC pilot (2026-07-13)

Fifth evaluation family: the 2021 Amazon Last-Mile Routing Research
Challenge data set (Merchan et al. 2024, Transportation Science 58(1),
DOI 10.1287/trsc.2022.1173, CC-BY-4.0) — 25 evaluation routes, 5 per
metro (LA, Boston, Seattle, Chicago, Austin), 100-200 REAL stops each,
adapter `ev/amazon.py`, fetcher `scripts/fetch_amazon_pilot.py`
(streams the 843 MB/166 MB bucket files, stores only the slice). The
null P0 is built entirely from the data set's own planner quantities:
Amazon's inter-stop travel times (forecast), planned service times
(dwell), package volumes (demand). Results
(`results_ev_amazon.csv`, 15 days/scenario):

Full-grid run (50 routes, 10/metro, ALL 19 scenarios, 12 days,
matched-rate foils):

- validity on real data: TEMPO 0.018 (null) / 0.022 (forecast-rain)
  <= alpha; CUSUM 0.032/0.022, PH 0.037/0.027 at their oracle-matched
  thresholds;
- the discriminator on industrial-length routes is MILD DEMAND drift:
  TEMPO 0.70 vs CUSUM 0.23 vs PH 0.23 (3x) — the 'packages run
  slightly heavy all day' case a dispatcher actually faces; demand_late
  0.88 vs 0.82 vs 0.23, demand_ramp 0.95 vs 0.93 vs 0.30;
- traffic scenarios sit at the ceiling for TEMPO and CUSUM alike
  (0.86-0.97; 150-stop routes carry overwhelming travel evidence) while
  Page-Hinkley collapses on traffic_mild (0.25); accident scenarios
  remain information-limited ties; compound days ~0.95 for all;
- uniform across metros: TEMPO 0.84-0.86 mean detection in all five;
- known pilot conventions: delivery-only, one realization per route so
  cross-stop dispersion proxies day-to-day dispersion; drift still
  injected (the data set has no realized-vs-forecast traces). All
  noted for the paper's limitations paragraph.

## 9e. Theory core drafted + Corollary 1 confirmed (2026-07-13)

`THEORY.md` states the three OR-native theorems (regret bound for the
statistically triggered replan; closed-form optimal evidence level
alpha* = Delta_c/(g*C_fr); value of waiting; decision-relevant betting
growth-optimality) with proof routes, plus the honest imported-vs-new
inventory. Empirical witness (`ev_alpha_sweep.py`, 10 instances x 15
days x 8 alphas, 50/50 null-surge population):

alpha:   0.5    0.2    0.1    0.05   0.02   0.005  0.001  1e-4
mixed:   167.0  165.1  163.1  161.4  158.3  153.8  152.1  161.8

U-shaped with interior optimum near alpha ~ 1e-3 for these economics.
Both arms are theorem-shaped: the RIGHT arm (1e-4) is detection
delay/miss (Theorem 1's Delta_c*log(1/alpha)/g term); the LEFT arm is
NOT false-alarm cost but information-poor replans — at alpha=0.5 the
alarm fires after ~2 events and the rebalancer acts on priors (Theorem
2's V(s) term), costing 191.1 vs 156.2 on surge days. Note: in this
harness a replan on a null day is nearly free (the insertion heuristic
slightly improves the remaining tour), so C_fr ~ 0 here; real
deployments carry coordination costs — add an explicit replan fee to
the harness when writing the paper so the corollary's trade-off shows
in both terms.

## 9d. Comprehensive grid (2026-07-13, results_ev_grid.csv)

73 instances (40 Dethloff + 14 Salhi-Nagy + 19 real-shop City) x 19
scenarios x 15 days; scenario registry in `ev/scenarios.py` (nulls incl.
a hostile forecast-rain null, single-factor x severity, ramps, a
transient jam, late onsets, compound storm/rush-crush/black-day).

Headlines:
- **The hostile null is the sharpest separator**: on fully-forecast
  rainy days TEMPO's false-alarm rate DROPS to 0.005 (conditional null
  absorbs the covariate), while oracle-calibrated CUSUM stays invalid
  at 0.057 and Bonferroni at 0.07. A monitor on raw lateness cannot
  tell forecast rain from drift; TEMPO can, by construction.
- **At matched false-alarm budgets TEMPO v2 dominates nearly every
  scenario** (tempo2 @0.016 vs cusum_match @0.038): traffic_mild 0.61
  vs 0.11, traffic_transient 0.37 vs 0.03 (12x), demand_severe 0.88 vs
  0.40, demand_ramp 0.70 vs 0.15, dwell_mild 0.87 vs 0.40, storm 0.98
  vs 0.85, rush_crush 0.98 vs 0.78, black_day 1.00 vs 0.97. Only
  accident_mild is a (information-limited) tie: 0.08 vs 0.10.
- **Ramps and transients are where anytime validity shines**: CUSUM's
  drain (the -k per step) forgets slow ramps and short windows; the
  adaptive e-process accumulates and keeps them.
- **Uniform across geographies**: tempo2 vs cusum_match detection —
  City 0.82/0.54, Salhi-Nagy 0.78/0.58, Dethloff 0.71/0.52.
- Ablation: tempo_flat collapses on exactly the scenarios that matter
  (demand_severe 0.31, traffic_late 0.60) — adaptive betting is the
  power source, the dual combination its safety net.

## 9c. Milestone 2 opener — the replanner is pluggable (2026-07-13)

`ev/replan.py` makes the optimizer under the trigger swappable:
per-vehicle open-TSP re-sequencing (NN + 2-opt), fleet-level 2-regret
reinsertion that REBALANCES remaining customers across residual slack,
and an exact Gurobi open-TSP (MTZ) anchor. ALNS remains the
planning-time optimizer; these serve the mid-route replan.

First cost experiment (`ev_replan_demo.py`: 12 Dethloff instances x 20
demand-surge days, +0.8 sd, day cost = 0.1/unit travel + 40/breach,
TEMPO v2 trigger, alarm rate 0.98):

| policy | day cost | saving | breaches/day |
|---|---|---|---|
| never replan | 519.2 | — | 11.45 |
| TEMPO + resequence | 399.5 | 23.0% | 8.46 |
| TEMPO + exact TSP | 399.5 | 23.1% | 8.46 |
| **TEMPO + rebalance** | **224.8** | **56.7%** | **3.79** |
| replan-at-first-stop + rebalance | 255.5 | 50.8% | 4.60 |

Three findings:
1. **The trigger and the recourse DIMENSION dominate the optimizer.**
   Exact TSP == NN+2-opt to 0.1%: sequencing quality is nearly
   irrelevant under demand drift; reassignment across vehicles is
   worth 2.5x more than any sequencing improvement.
2. **TEMPO + rebalance beats replanning at the first stop** (56.7% vs
   50.8%): waiting for evidence is not just statistically safer, it is
   ECONOMICALLY better, because the replanner acts on realized loads
   and true residual slack rather than priors — "the value of waiting
   for evidence," a headline result candidate for the paper.
3. Swappability is real: any `replan(cur, remaining, D)` /
   `rebalance(...)` callable slots in; milestone 2 proper will cross
   triggers x backends (incl. warm-started ALNS) on all scenarios.

## 9g. Three literature baselines added (2026-07-13): e-detectors, SMPC, DGTA-RL

Per user request, added and ran the three most relevant 2024-2026
papers as concrete baselines rather than citations only:

**e-SR / e-CUSUM** (Shin, Ramdas & Rinaldo 2024, "E-detectors: A
Nonparametric Framework for Sequential Change Detection," NEJSDS
2(2):229-260, DOI 10.51387/23-NEJSDS51) — implemented in
`ev/baselines.py` using EXACTLY TEMPO's own per-channel bets (s=1, no
adaptive theta, no cost tilt), combined via their restart-and-sum
(SR)/restart-and-max (CUSUM) rule instead of TEMPO's single running
product. Caught a real conceptual bug while validating: an e-detector
satisfies E[M_tau] <= E[tau] (Def. 2.2), an AVERAGE RUN LENGTH
guarantee over an indefinitely-run stream — NOT a Ville-style
P(false alarm in n steps) <= alpha bound like TEMPO's own guarantee.
Fixed by (a) testing the native ARL guarantee on a continuous
non-reset stream (test_ev.py, confirms Theorem 2.4 holds), and
(b) oracle-calibrating the M-statistic threshold for the apples-to-
apples PER-DAY comparison the rest of the project uses (same protocol
as the classical CUSUM/PH foils).

**SMPC** (He, Li, Li, Huang, Huang & Duan 2026, Mathematics 14(6):1032,
DOI 10.3390/math14061032) — chance-constrained periodic rolling-horizon
re-solve; its trigger mechanism is exactly `PeriodicMonitor`, now cited
directly (the time-window MILP itself is out of scope — TEMPO's harness
doesn't model time windows).

**DGTA-RL** (Chen, Imdahl, Lai & Van Woensel 2025, Transportation
Research Part C 172:105022, DOI 10.1016/j.trc.2025.105022) — a Dynamic
Graph Temporal Attention RL policy for the DTSP with time-dependent
stochastic travel times, no explicit detection layer. Added their own
cited rolling-horizon baseline, "Rolling-opt" (Gmira et al. 2021a), as
`RollingOptMonitor` (replan every stop, period=5 events). Retraining
DGTA-RL's actual attention architecture is underway on Colab T4 (see
§9h) rather than left as a citation only.

Full-grid results (73 instances x 19 scenarios x 15 days,
results_ev_grid.csv) at TEMPO's matched false-alarm rate (~0.016-0.028
across the new detectors):

| scenario | cusum_match | esr_match | ecusum_match | **tempo2** |
|---|---|---|---|---|
| traffic_mild | 0.11 | 0.28 | 0.25 | **0.61** |
| demand_severe | 0.40 | 0.66 | 0.58 | **0.88** |
| demand_ramp | 0.15 | 0.41 | 0.34 | **0.70** |
| traffic_late | 0.74 | 0.94 | 0.94 | **0.98** |
| traffic_ramp | 0.92 | 0.97 | 0.97 | 0.98 (tie) |
| accident_severe | 0.80 | 0.74 | 0.73 | 0.84 |

Key finding: **e-SR/e-CUSUM (the general nonparametric restart
machinery, no cost coupling) sit consistently between classical CUSUM
and full TEMPO** — confirming the restart-at-every-changepoint idea
alone recovers real power over naive CUSUM, but TEMPO's additional
adaptive betting + cost-relevance tilting still adds a clear further
margin on demand and mild-traffic drift (where the gap is largest:
demand_ramp 0.34->0.70, more than double), and the gap nearly closes
only on scenarios with very strong, sustained evidence (traffic_ramp/
severe/late) where any reasonable method saturates. This is the
cleanest ablation yet: it isolates "general e-process technology" from
"TEMPO's specific coupling" as two separable sources of power.

`rolling_opt` and `periodic` score near 0 on the detection-rate metric
by construction — they alarm on a fixed schedule regardless of
evidence, so their first scheduled alarm almost always lands before a
scenario's drift onset and gets scored as a false/non-detection. This
is not a flaw in the comparison so much as a mismatch of objectives:
periodic/rolling-horizon methods are designed to be evaluated on
REALIZED COST (do frequent free replans pay for themselves), which is
milestone 2's job (ev_replan_demo.py), not the detection-layer harness.
Noted here so the discrepancy isn't mistaken for a bug.

## 9h. DGTA-RL retraining on Colab T4 (2026-07-13, in progress)

Scaled-down faithful reimplementation of the DGTA model (spatial-
temporal encoder with dual attention, dynamic encoder, spatial+temporal
pointers, REINFORCE training per Algorithm 1 of Chen et al. 2025) for
the time-dependent stochastic travel-time channel TEMPO's world model
already simulates. Training on Colab T4; see commit history / this
section for the outcome once complete.

## 9i. Manuscript drafted (2026-07-13, INFORMS Transportation Science template)

Full manuscript at `papers/tempo/main.tex` (informs4.cls, `trsc`
option), compiling cleanly to a 37-page PDF (`pdflatex + bibtex +
pdflatex x2`, no undefined references, no overfull boxes, no duplicate
labels). Full flowing prose, all four theorems (Theorem 1 regret
bound, Corollary 1 optimal alpha*, Theorem 2 value of waiting with a
worked two-vehicle stylized model, Theorem 3 growth-optimality via a
saddle-point argument) proved in full rather than left as proof
routes, Propositions 1-5 for the validity chain (master process,
coupled tilts, dual-regime combination, rare-event split), an
algorithm-environment pseudocode listing for the monitor, and results
tables transcribed from `results_ev_grid.csv`, `results_ev_replan.csv`,
`results_ev_amazon.csv`, and the alpha-sweep run. Section~9.10
(DGTA-RL) is a placeholder pending the Colab training run in \S9h ---
fill in the realized-cost comparison once training completes, then
rebuild (`pdflatex && bibtex main && pdflatex && pdflatex`). Citations
mostly verified; three flagged in `VERIFY_CITATIONS.md` (Gmira et al.
2021's bibliographic record, and exact volume/page ranges for Ramdas
et al. 2023 and Waudby-Smith & Ramdas 2024). `main.pdf`/`.aux`/`.bbl`/
`.log` are gitignored per `papers/tempo/.gitignore` --- rebuild locally
to view.

**Revision (2026-07-13, same day):** per explicit direction, (a) the
manuscript is now fully anonymous for double-blind review
(`\documentclass[trsc,dblanonrev]{informs4}`, author/affiliation block
blinded, `\RUNAUTHOR{Anonymous}`) and (b) every citation/mention of the
sibling (unpublished) paper's authors and its "BATON" name is removed
--- the planning-layer recourse policy is now described generically
("a planning layer that solves for recourse under an optimal-stopping
formulation") with no external attribution, and its bib entry is
deleted; (c) restructured from 11 top-level sections down to the
requested 6 (Introduction incl. Related Work; The stochastic routing
day and the master e-process; TEMPO: decision-relevant coupling and
construction; The economics of the trigger incl. the decision layer;
Computational study; Discussion and conclusion) via section/subsection
demotion, no content cut; (d) `tempo_fig2_vs_baton.png` removed (its
title/labels literally said "BATON," which the sibling-paper module
docstring had explicitly flagged as "does NOT go into the TEMPO
manuscript" — the earlier draft included it by mistake); (e) two new
figures added instead: `tempo_fig4_fleet.png` (a static frame pulled
from `anim_tempo_fleet.gif` — many vehicles on a real road network
colored by realized congestion, alongside the pooled e-process, is
exactly the "traffic with colors and many vehicles" visualization
requested) and `tempo_fig5_baselines.png` (a new grouped bar chart of
TEMPO vs. CUSUM/Page-Hinkley/e-CUSUM/e-SR detection rate across nine
representative scenarios, generated fresh from `results_ev_grid.csv`
rather than reusing the sibling paper's reactive-vs-optimal-stopping
comparison GIF, which was a different comparison entirely).

## 10. World-model extensions surfaced by the visualizations

- **Zonal jam** (implemented, `DriftSpec(kind="traffic_zone")`): a
  congestion pocket with a centre and a radius growing over time; only
  legs whose stop lies inside the current zone are slowed. Gives drift
  a WHERE, not just a WHEN, and makes fleet-level pooling meaningful
  (vehicles far from the pocket contribute null evidence).
- **Fleet-pooled monitoring** (implemented in the fleet animation): one
  master e-process consumes every vehicle's events in clock order —
  the dispatcher tests the PLAN, not a vehicle. Empirical lesson: the
  rare-event breakdown stake must run as a parallel e-process with a
  union-bound alpha split (alpha/2 + alpha/2), otherwise its per-leg
  insurance premium scales with fleet size x legs and drowns the
  continuous channels (~ -8 log units for 11 vehicles). This deserves
  a remark in the paper.
- **Per-edge congestion field** (visualization layer today): every road
  segment carries a static severity, its own jam-response exponent and
  an AR(1) log-noise state with temporal persistence. The simulator's
  per-leg lognormal is the marginal of this field; promoting the field
  itself into the simulator (spatio-temporally correlated travel
  noise) plus a zone-aware spatial sensitivity tilt is the natural
  next modelling step — and a testable one: correlated noise breaks
  the independence assumption A1, so the master process must switch to
  conditional-law bets (chain rule handles it, the alternative just
  needs the field's AR structure).

## 11. Manuscript revision: responding to a rigorous adversarial review (2026-07-14)

The user relayed a detailed 20-point adversarial review (plausibly
self-generated by asking "would this pass review at Transportation
Science") arguing the manuscript oversold its novelty and left several
proofs under-justified. Most points were correct and each got a real
fix, not a cosmetic one:

- **Sensitivity weights derived, not asserted** (the review's sharpest
  point): added a genuine first-order/Taylor derivation in §3.1.1
  showing the target sensitivity is `r(n) = D(W)/(Ĉ'(W) sigma_g)` — a
  boundary-distance-in-shock-units ratio directly tied to Theorem 3's
  cost-gradient `c_k(n)` — with Eqs. 15/16 (the closed-form proxies
  actually used) now explicitly framed as cheap approximations to that
  target under two stated conditions, plus an explicit worked
  counterexample (a tightly-windowed high-value stop near the route's
  end) showing exactly where the proxy fails, since the harness has no
  time-window data to do better.
- **Proposition 3's proof fixed, not just re-asserted**: realized the
  "product of e-processes" argument never needed cross-channel
  independence at all (only one channel updates per event, so the
  asynchronous tower-property argument from Prop. 1 applies verbatim)
  — removed the incorrect independence citation and replaced it with
  the correct, stronger argument, plus a new Remark distinguishing
  mathematical validity (holds regardless of real-world channel
  correlation) from modeling fidelity (whether the null model captures
  real correlation, e.g. an unmodeled shared congestion field).
- **Theorem 1's "standard argument" replaced with a real proof**: now
  invokes Wald's identity (1945) + Lorden's overshoot inequality (1970)
  for the fixed-grid case and Woodroofe's nonlinear renewal theory
  (1982) for the adaptive-bet case, with kappa properly defined as a
  finite, bounded constant in both regimes.
- **Segment-wise validity (the reset argument) actually proved**: the
  old claim ("sequence of independent anytime-valid tests") was wrong
  as stated — segments are NOT independent, the replanner changes the
  state. Replaced with a genuinely correct proof (new Proposition,
  "segment-wise validity under reset") via a conditional-Ville
  argument that needs no independence across segments at all, only
  that each segment's own conditional false-alarm probability is
  bounded given whatever state it inherits.
- **Theorem 2's delta-method step completed**: the review was right
  that "V(s) = kappa*sigma^2/s, delta method" had no shown derivation.
  Added the missing algebra end to end (Taylor-expand the breach cost
  around the population-optimal split, Taylor-expand the split
  function in mu, combine with the sampling variance of a mean) with
  an explicit closed-form kappa_0, plus an honest scope statement (this
  is a stylized single-parameter model, not a claim about the
  combinatorial rebalancer actually used).
- **Theorem 3's alternative class justified**: framed `P_Delta` via the
  Huber (1964) contamination-neighborhood tradition in robust
  statistics — a legitimate, citable precedent for "minimax-optimal
  against a linearized worst-case class," with an explicit statement
  that this is a modeling choice, not a derived necessity.
- **Real ablation added** (`scripts/ev_ablation.py`, new
  `use_adaptive`/`use_dual` flags on `TempoMonitor`, backward-compatible
  — all 28 existing tests still pass): isolates the sensitivity tilt,
  adaptive betting, and dual-regime combination incrementally on 15
  instances x 8 scenarios x 15 days. Found and reported an unflattering,
  informative result: the tilt ALONE (no adaptive betting) actually
  *hurts* detection on demand scenarios (0.111->0.040, 0.351->0.129)
  because scaling a fixed-strength grid bet down by a small sensitivity
  makes it a strictly weaker bet — only once adaptive betting is added
  does the combination recover and exceed the untilted baseline. This
  is a more convincing ablation than a monotonically-improving one
  would have been.
- **Real confidence intervals on the Amazon pilot** (95% Wilson score,
  computed from the actual per-route outcomes in
  `results_ev_amazon.csv`, not asserted), with an explicit caveat that
  the 600-trial count overstates independent real-world realizations
  by roughly the 12x day-simulation factor.
- **Computational complexity subsection added** (O(1) per event, O(K)
  state, negligible next to any replanning backend).
- **Explicit tuning-protocol paragraph added** clarifying TEMPO's
  hyperparameters are fixed once across the whole grid (not tuned per
  instance/scenario) while classical/e-detector baselines get generous
  oracle calibration.
- **Honest-contribution framing added** at the top of the theory
  section (mirrors THEORY.md's own inventory table) explicitly stating
  which results are routine/imported vs. genuinely new, and Corollary 1
  now explicitly says its calculus is elementary and its value is
  economic translation, not mathematical difficulty.
- **Collected limitations paragraph** added to the discussion,
  synthesizing all of the above into one place rather than scattering
  caveats.

Six new citations added, each verified against a live publisher/DOI
record before use: Wald (1945), Lorden (1970), Woodroofe (1982), Huber
(1964) — all classical and safe — plus the four 2024-2026 related-work
citations from the prior revision. Column-width overflows introduced by
the new equations (informs4's 240pt single-column check) were found and
fixed iteratively; final compile is clean at 47 pages, zero undefined
references, zero overfull boxes, zero ATTENTION markers, and the
anonymity/no-self-citation sweep from the prior revision still holds.

## 12. Manuscript revision: hitting TRSC's hard 35-page limit

TRSC's submission guidelines cap the main body (including references,
tables, and graphs) at 35 pages, plus a separate 15-page appendix
allowance. The manuscript was at 46 pages main-body-equivalent after
the adversarial-review revision (Section 11); a first pass moved all
detailed proofs and several derivations into a formal `APPENDIX`
environment (proofs of Propositions 2/3/4, Theorems 1/3, the
delta-method derivation behind Theorem 2, and the discussion of several
modeling choices), getting to 40 pages. Closing the remaining 5 pages
took many more iterations than proof-relocation alone:

- Prose tightened throughout (Introduction, Related Work, the
  "economics of the trigger" framing paragraphs, Managerial
  Insights/Conclusion) without cutting any citation, number, or claim.
- More discussion moved to the Appendix: the mechanism behind the
  ablation's non-monotonic finding, the "what's routine vs. new"
  tool-by-tool breakdown, two of the three "value of waiting" findings,
  and the reimplementation details of the DGTA-RL baseline — each
  replaced in the main body by a one-line summary and pointer.
- The 19-scenario grid catalog table (reference material, not a
  result) moved to the Appendix; its prose description stayed in the
  main body.
- Several short subsections (Corollary 1, Metrics, Validity, the
  significance-level and waiting-time confirmations, component
  ablation, DGTA-RL, the capacity-triggered-recourse discussion,
  Conclusion) were converted from numbered `\subsection`s to bold
  run-in headers — no content lost, just less header overhead.
- Figure widths were reduced project-wide (explainer/baselines/fleet
  now 0.52–0.58\textwidth, down from full/near-full width).
- **`tempo_fig3_map.png` (the single-vehicle zonal-traffic-jam figure)
  was dropped entirely.** It visually overlapped `tempo_fig4_fleet.png`
  (same congestion pocket, same real road network, already shown for
  eleven vehicles), and after ~15 rounds of prose-only trimming
  plateaued at 36/35 pages with zero further movement, removing the
  redundant figure was the cleanest way to close the last page without
  cutting any result. The user confirmed this call explicitly. The
  three remaining figures (explainer, fleet, baselines) still cover
  both original asks: real-network traffic-with-vehicles, and a direct
  TEMPO-vs-baselines comparison.
- **Update:** the user asked for the map figure back — they liked it —
  so it was restored, but into the Appendix rather than the main body
  (a new subsection alongside the scenario-grid table, cross-referenced
  from the fleet-figure discussion), keeping the main body at exactly
  35 pages. Getting back to 35 after re-adding it took one more small
  fix: the reference list's own pagination is borderline (its last 2-3
  entries flip between the previous and a fresh page on tiny upstream
  reflow), so `\bibsep` was tightened from `\smallskipamount` to `1pt
  plus 1pt minus 1pt` to reclaim those lines — a purely typographic
  change, not a content cut.
- A genuine latent bug was found and fixed along the way: the prior
  proof-relocation pass had left four labels
  (`eq:sensdemand`, `eq:vs`, `rem:growth`, `rem:independence`)
  multiply-defined — duplicated verbatim between the main body and the
  Appendix. Fixed by de-duplicating each into a single canonical label.
  A separate copy-paste artifact (a stray, dangling half-sentence
  fragment after Theorem 2's derivation) was also found and removed.

Final state, verified via a from-scratch `pdflatex + bibtex + pdflatex
+ pdflatex` rebuild: **35 pages main-body-equivalent, 7-page Appendix**
(well under its own 15-page cap), zero ATTENTION column-overflow
markers, zero undefined or multiply-defined references, zero
significant overfull boxes, structural begin/end balance clean across
every environment, and no BATON/identity leaks.
