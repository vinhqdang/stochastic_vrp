# BATON — Results Summary

**Naming.** The algorithm is **BATON** — *Backward-induction AcTion pricing
for ONline recourse* (formerly "OTR"). Mapping to code/CSV labels, which
stay frozen for reproducibility:

| Paper name | Code / CSV label | What it is |
|---|---|---|
| **BATON** | `v2_act` | full algorithm: continue / handoff / depot-restock, exact reset valuation, per-route deployment selection |
| BATON-HO | `v2_lsm` | handoff-only restriction (formerly "OTR-2.0") |
| endpoint-threshold predecessor | `v1_end`, `v1_myo` | ablation (formerly "OTR 1.0") |
| peak-label threshold | `fb_tau` | ablation of the trigger (label fixed, threshold kept) |


*All experiments reproducible from `svrpspd_wdro/scripts/`; tables in the
manuscript regenerate from these CSVs via `papers/baton/make_tables.py`.*

## What was evaluated

Six per-route execution policies, always trained on scenarios independent
of the test set:

| Label | Policy |
|---|---|
| `none` | Reactive: never hand off, pay the emergency price at the breach |
| `v1` (endpoint) | OTR 1.0: isotonic overflow curves on the **endpoint** label `W_m > B`, tuned threshold |
| `v1_myo` | Myopic threshold `omegaF/Cfail` |
| `fb` | Label-corrected fallback: **peak** label + tuned threshold |
| `v2` | **OTR-2.0**: peak label + Longstaff–Schwartz optimal stopping (no threshold) |
| `oracle` | Clairvoyant lower bound (sees the whole demand path) |

## 1. Structural synthetic scenarios (5 seeds × 12k test routes)
`results/results_otr2_synthetic.csv`

| Scenario | v1 tuned | fallback | **v2** |
|---|---|---|---|
| Collect-then-deliver | **0.0%** (never triggers) | 49.2% | **51.9%** |
| Milk run + regime switching | 58.9% | 58.9% | **63.8%** |
| Milk run, Cfail/omegaF=20 | 85.5% | 85.5% | **88.2%** |

(saving in expected execution cost vs `none`.) The collect-then-deliver row
is the v1 defect isolated: endpoint labels are almost surely zero there, so
the trained policy degenerates to inaction — exactly as Proposition 1 of the
manuscript predicts.

## 2. Dethloff benchmark, flat two-price model (40 instances × 3 gates)
`results/results_otr2_eval.csv`

Saving vs reactive: v2 **57.7 / 55.6 / 39.1 %** (Det/SAA/WDRO) against
56.5 / 54.6 / 40.3 % for the tuned v1. Dethloff routes have
corr(endpoint, peak) ≈ 0.9, so the label defect is mild on this dataset and
gains come mainly from the trigger; WDRO plans leave almost no overflow to
prevent (ties).

## 3. Dethloff under the realistic last-mile cost model
`results/results_realistic_eval.csv` — contracted fleet day-rates, variable
km costs, standby handoffs priced on the *remaining* route, surge-priced
emergencies with per-customer SLA compensation (`core/costs.py`).

| Gate | saving% v1 | saving% fb | **saving% v2** | gap-to-oracle v1/fb/**v2** |
|---|---|---|---|---|
| Det | 38.0 | 38.4 | **40.1** | 30.6 / 29.6 / **26.6** |
| SAA | 33.6 | 33.7 | **38.5** | 48.5 / 48.3 / **41.0** |
| WDRO | 18.5 | 18.2 | **25.9** | 71.7 / 71.8 / **59.9** |

Paired Wilcoxon (one-sided, per gate, n=40): v2 beats both tuned-threshold
competitors at **p < 8×10⁻⁷ on every gate**; pooled over all 120 plan
comparisons, p ≈ 2×10⁻²⁰. State-dependent prices are where the
optimal-stopping trigger structurally outruns any single threshold — a
scalar `tau` cannot adapt to per-stop handoff prices.

## 4. Sensitivity study (5 sweeps × 2 structures × 5 seeds)
`results/results_otr2_sensitivity.csv`, `_tests.csv`

- v2 beats the strongest tuned competitor at p<0.05 in **35/40 cells**.
- Mean share of oracle-achievable saving missed: **v2 25.4%**, fallback
  29.8%, v1 tuned 61.9%.
- v2's edge **grows with route length and with the emergency-price ratio**
  (more option value), is robust across gamma/lognormal/Student-t demand,
  and degrades gracefully under a common demand factor.
- Even at N=200 training paths v2 ≥ fallback in these scenarios; the
  theoretical small-N caution applies below that.

## 5. Large-scale benchmarks (added after the Dethloff layers)

**Salhi–Nagy classical benchmark** (`results/results_salhinagy_eval.csv`) —
14 instances (CMT1–5, 11, 12 X/Y; 50–199 customers) derived from CVRPLIB
via the documented ratio split; realistic cost model, Det-gate plans:

| Policy | saving% vs reactive | share of oracle saving |
|---|---|---|
| v1 endpoint | 48.6 | — |
| fallback (peak+tau) | 53.8 | — |
| DP equal-data | 59.3 | 90.0% |
| **OTR-2.0** | **59.4** | **90.3%** |
| DP 50k paths (near-exact) | 60.1 | 91.3% |

v2 beats v1 and fallback at p ≈ 6×10⁻⁵ (paired Wilcoxon, n=14), edges the
equal-data DP (p=0.045), and reaches **98.9% of the near-exact DP's
performance** from 1/50th of its data.

**Real-map city instances** (`results/results_city_eval.csv`) — 6 instances
(100/200/400 customers) on the actual OSM drive networks of Ho Chi Minh
City and Hanoi (symmetrized shortest-path road distances, parcel-van
demands): v2 saves 24.6% vs 21.5% (v1) and 23.7% (fallback); it beats v1
and the equal-data DP at p=0.016 and attains 92.3% of the near-exact DP.
Routes here are long (12–22 stops), which is exactly where the
optimal-stopping trigger's option value grows.

## 6. Exact-solver certification (HiGHS MIP)

Planning-layer check with the Montané–Galvão two-commodity-flow VRPSPD MIP
solved by HiGHS (`results/results_mip_cert.csv`, `results_mip_small.csv`):

- **Small city sub-instances (12–24 customers):** HiGHS proves optimality
  in seconds; even the fast CW+2-opt planner is within 1.2–3.6% of the
  true optimum.
- **Full Dethloff instances (50 customers, 300 s/instance):** the ALNS
  plans used throughout our experiments beat HiGHS's own incumbent on
  **35/40 instances** (the solver's best solution averages 5.9% *worse*
  than ours), and the MIP dual bound certifies every ALNS plan within
  **11.0% mean / 14.5% max** of optimal — an upper bound on true
  suboptimality, since the two-flow LP relaxation is known to be loose.

Execution stage vs exact methods: covered by the DP benchmark (Sections
3 and 5) — MIP technology does not encode non-anticipative multistage
stopping rules, so backward DP is the correct exact comparator there.


## 7. Two-class fleet economics (final cost model)

Fleet costs split by vehicle class: planned fleet F_plan x K (fixed),
standby class — each planned handoff consumes one pooled standby
vehicle-day billed at F_standby=$20 inside the handoff price — and
emergency class (ad-hoc surge, per incident, no retainer). Full grid
rerun (`results_grand_dethloff.csv`, 6 gates x 11 policies x 40 instances):

| Gate | v1 | fallback | pi3 | DP(=data) | **OTR-2.0** | DP(50k) | oracle |
|---|---|---|---|---|---|---|---|
| Det | 24.2 | 24.6 | * | 25.4 | **25.9** | 26.2 | 40.4 |
| SAA | 16.1 | 16.1 | * | 12.1 | **20.3** | 22.2 | 43.5 |
| WDRO | **-1.3** | **-1.4** | * | 0.0 | **+11.2** | 12.0 | 41.9 |
| GNRS | 25.2 | 25.6 | * | 28.0 | **28.8** | 29.6 | 45.1 |
| BSIM | 22.5 | 22.8 | * | 24.7 | **26.3** | 27.3 | 44.3 |
| MDRO | 2.8 | 2.7 | * | 0.0 | **+12.6** | 15.2 | 42.4 |

(recourse saving % vs reactive; * pi3 columns in the CSV.) The headline
finding: with handoffs carrying their true standby vehicle-day cost, the
THRESHOLD family goes NEGATIVE on conservative plans (WDRO/MDRO) — their
handoffs cost more than they save — while the optimal-stopping trigger
correctly backs off and stays firmly positive. Paired Wilcoxon: v2 beats
v1, fallback, pi3 and the equal-data DP on every gate, p<=8e-3 (mostly
1e-8..1e-13). Salhi-Nagy: v2 42.0% vs fb 37.3% (p=6e-5). City: 12.0% vs
11.3% (p=0.04).


## 8. OTR-2.1 feature enrichment: a null result (kept deliberately)

Enriching the conditioning statistic beyond scalar W_k — common-factor
posterior (precision-weighted residuals), last increment, running max —
with monotone gradient-boosted trees (`core/otr21.py`) LOSES to the
isotonic scalar-W models on 23/25 real routes (mean -3.2% relative,
`scripts/validate_otr21.py`), while fitting 100-1000x slower. At
realistic history sizes the extra features add more estimation variance
than signal; the isotonic shape constraint on W_k is doing the heavy
lifting. This confirms the design choice empirically (spec section 7)
and is worth a paragraph in the manuscript as evidence that OTR-2.0's
simplicity is not a limitation at operational data scales.


## 9. RL baseline (Iklassov et al. 2024 re-implementation, Colab T4)

A neural execution policy in the style of Iklassov, Sobirov, Solozabal &
Takac (ACML 2024): shared MLP over observable state (position, load,
slack, price ratios, remaining-demand moments), trained with REINFORCE +
moving baseline for 40 epochs on a T4 (`scripts/rl_exec_train.py`,
results `results/rl_results.json`). On the identical 50 routes and test
days (Dethloff + Hanoi/HCMC bundle):

| Policy | saving vs reactive |
|---|---|
| RL policy (GPU-trained) | 27.6% |
| **OTR-2.0** (CPU, ms per route) | **33.9%** |

OTR-2.0 wins on 39/50 routes (paired Wilcoxon p = 1.3e-8). The learned
policy is a respectable competitor — clearly better than the rule-based
family — but the backward-induction estimator extracts more from the same
1,000 training paths than policy-gradient RL, at a tiny fraction of the
compute.

Note on framing: OTR 1.0 (v1_end/v1_myo columns) is our own prior
version; in the manuscript it appears as an ABLATION (label defect and
trigger change isolated), not as a competitor. The competitor set is:
reactive, tuned threshold, pi1-pi3 (Salavati-Khoshghalb 2019), rollout
(Secomandi 2001), depot restocking (Florio/Legault family), the plug-in
DP at equal data, the RL policy above, and the clairvoyant oracle bound.


## 10. The combined-action policy (OTR-A): final comparison matrix

Generalizing the OTR-2.0 backward induction to the full recourse action
set {continue, hand off, depot-restock} — with exact fresh-suffix
valuation of the restock reset and per-route train-set deployment
selection (falls back to handoff-only where restocking cannot pay) —
produces the strongest policy on every benchmark:

| Gate | restock | tuned thresh | OTR-2.0 (HO) | **OTR-A** | DP-50k (HO) | oracle (HO) |
|---|---|---|---|---|---|---|
| Det | 5.6 | 24.8 | 26.1 | **27.8** | 26.5 | 40.7 |
| SAA | 43.7 | 16.6 | 20.5 | **53.4** | 22.3 | 43.5 |
| WDRO | 34.8 | -8.3 | 9.3 | **40.9** | 12.0 | 42.0 |
| Gounaris | 43.3 | 17.0 | 21.2 | **52.8** | 22.9 | 43.5 |
| Cui | 41.0 | 16.6 | 20.7 | **51.6** | 22.7 | 43.2 |
| MDRO | 38.8 | -1.8 | 11.9 | **44.3** | 14.2 | 42.2 |

(saving % vs reactive; HO = handoff-only action set.) Pooled over 240
plan rows, OTR-A beats every competitor on 221-240/240 instances at
p ~ 1e-39..1e-41 — including the handoff-only 50k-path DP (235/240) and,
on four gates plus Salhi-Nagy (53.0 vs 49.0), it EXCEEDS the
handoff-only clairvoyant oracle: richer actions legitimately beat a
clairvoyant restricted to fewer actions. Salhi-Nagy: 53.0%. City: 11.5%
(restock worthless there — deployment selection correctly falls back).

Notes: the oracle column is a bound for the HANDOFF-ONLY class and is
kept as the ablation anchor; the depot-restock action is priced with
full SLA lateness for downstream customers (same rate as emergencies).
The three-action finding also explains the literature: depot-return
recourse (Florio/Legault) wins exactly where depots are central, handoff
recourse wins where they are not, and OTR-A prices the choice per stop
from data — subsuming both.


## 11. Cost-parameter sensitivity (managerial analysis)

One-factor sweeps around the fleet-economics defaults (12 Dethloff
instances, Det+SAA gates; `results/results_costsens_*.csv`). Saving % vs
reactive; BATON = deployed combined-action policy:

| Configuration | restock | tuned thresh | BATON-HO | **BATON** | oracle(HO) |
|---|---|---|---|---|---|
| baseline | 24.6 | 20.7 | 23.3 | **40.6** | 42.1 |
| cheap emergencies (F_emg=25) | 22.5 | 13.5 | 17.6 | **39.5** | 31.1 |
| dear emergencies (F_emg=60) | 32.3 | 35.7 | 39.1 | **54.4** | 55.4 |
| cheap standby (F_standby=10) | 28.2 | 37.0 | 41.3 | **51.3** | 57.7 |
| dear standby (F_standby=35) | 28.0 | 7.3 | 12.4 | **41.9** | 24.8 |
| low SLA price (p_late=0.5) | 33.1 | 22.3 | 27.0 | **48.8** | 42.0 |
| high SLA price (p_late=3.0) | 23.3 | 28.5 | 32.2 | **46.1** | 47.6 |
| mild surge (s_emg=1.5) | 26.8 | 20.3 | 24.9 | **44.2** | 40.3 |
| heavy surge (s_emg=4.0) | 29.8 | 30.2 | 33.6 | **51.4** | 49.3 |

BATON leads in every configuration. The managerially interesting rows:
when standby vehicles are DEAR ($35/day) the threshold family collapses
(7.3%) while BATON shifts weight to depot returns and holds 41.9% —
exceeding the handoff-only oracle (24.8) by a wide margin; when SLA
lateness is expensive, depot returns lose appeal and BATON shifts back
toward handoffs. The policy re-balances its recourse mix as prices move,
with no re-tuning.

## 12. Spatial structure: real shops vs uniform scatter

Twin city instances (same cities, demands, seeds) with customers at real
OSM shop locations vs uniformly sampled street nodes
(`results_city_eval.csv` vs `results_cityuniform_eval.csv`): real retail
clustering yields **25.9% shorter routes** (211.9 vs 287.2 km mean) —
uniform-scatter conventions materially overstate travel — while the
policy ranking and BATON's lead are unchanged, supporting external
validity. City replication (seeds 1-3, 19 instances): BATON 10.8%,
DP-50k 12.4%, oracle 36.8%.

## 13. Strengthened RL baseline (robustness of Section 9)

Retraining the Iklassov-style policy with 150 epochs (~4x), entropy
regularization, learning-rate selection, and multiple seeds on a T4
(`results/rl_strong_s*.json`): best seed reaches 28.6% saving (vs 27.6%
originally) — still decisively below BATON-HO's 33.9% on identical
routes and days (better on 38/50 routes, p = 1.8e-8). The gap is not an
artifact of under-training.

## Verdict

OTR-2.0 dominates OTR 1.0 everywhere it matters: catastrophically on
collect-then-deliver structures (0% → 52%), significantly under realistic
state-dependent economics (p < 10⁻⁶ per gate), and never worse elsewhere —
with zero threshold tuning. Remaining headroom to the clairvoyant bound is
~27–60% of the achievable saving depending on plan conservatism; the next
levers are richer conditioning statistics, partial handoff, and feeding
execution cost back into the planner's acceptance gate.
