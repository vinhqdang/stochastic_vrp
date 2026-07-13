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
