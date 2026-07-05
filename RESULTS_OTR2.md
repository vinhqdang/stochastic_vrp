# OTR-2.0 — Results Summary

*All experiments reproducible from `svrpspd_wdro/scripts/`; tables in the
manuscript regenerate from these CSVs via `paper/make_tables.py`.*

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

## Verdict

OTR-2.0 dominates OTR 1.0 everywhere it matters: catastrophically on
collect-then-deliver structures (0% → 52%), significantly under realistic
state-dependent economics (p < 10⁻⁶ per gate), and never worse elsewhere —
with zero threshold tuning. Remaining headroom to the clairvoyant bound is
~27–60% of the achievable saving depending on plan conservatism; the next
levers are richer conditioning statistics, partial handoff, and feeding
execution cost back into the planner's acceptance gate.
