## Seat and angle

Peer Reviewer 3, cross-disciplinary and practice perspective. I read it as a last-mile operations researcher who has worked on fleet and standby-capacity management, pricing and managerial decisions. My questions: are the three-class cost model, the standby pool treatment and the Section 5 findings credible, and are they qualified correctly for practitioners? I checked the claims against main.tex, the tables and macros, `core/costs.py`, `make_city_instances.py`, `results/r1/pool.csv`, `results/r1/fresh.csv` and `results/results_grand_dethloff.csv`. The r1/ folder was not opened.

## Verdicts on round-1 comments

| item | verdict | evidence anchor | residual gap |
|---|---|---|---|
| R1.2 | PARTLY_RESOLVED | §3.1 "Standby versus emergency" (main.tex ~495–510); §4.6 and Table 12 (reserved pool, FCFS, shadow price λ) | The experiment is informative in principle, but its design settles the result. (a) It runs only on Dethloff **SAA** plans and City Det plans. On SAA plans Baton's 95% pool size is 0 on every plan (grand CSV: `v2_act_S` median 0, max 0), so S*=0 there holds by construction. The Dethloff **Det** plans, where Baton needs a median of 3.5 and up to 7 standby vehicles per plan-day, were left out. (b) A reserved vehicle costs the same F_sb=20 as a pay-per-use vehicle-day, so reserving can only pay when expected handoffs exceed about 1. In pool.csv the first reserved vehicle is worth 0.12 per day on Dethloff (max 0.87) and 3.7 in City (max 7.4), far below 20. That makes the holding-cost setting a straw man. (c) "Shadow beats naive by 15.8%" at S=0 compares against a policy that asks for vehicles that do not exist. Details under New issues 1–2. |
| R1.8 | PARTLY_RESOLVED | New "Scope of the conclusions" paragraph (§5) covers time windows, heterogeneous penalties, shared resources and integrated planning | Section 5 still opens with "five findings travel beyond our test bed" (l.1955). Finding 4 (geography) and Finding 5 (pool standby) are not supported as stated (New issues 1, 3). Two things are not listed as out of scope: standby response time and availability, and lateness that does not depend on how long the delay is. |
| R2.M3 | PARTLY_RESOLVED | §4.3 city paragraph; Table 6 (Baton 10.6 = Baton-ho 10.6); abstract; §5 "Limitations" gives boundaries for partial handoff | The paper now states clearly that the return lever is idle on city routes, which is honest. But the causal explanation offered is wrong (New issue 3): the city depots sit at the exact centre, and the model's own detour cost there is below 1 currency unit. |
| R2.M5 | RESOLVED | §5 last paragraph: co-optimisation named the most important direction, with two reasons it was not done | — |
| R2.M6 | RESOLVED | §4.5 "Non-exchangeable days", Table 10; §5 remedy of one fit per day type or a rolling window | Only a known, scheduled mean shift is tested. Unannounced drift and the rolling-window remedy are recommended but not tested, so present them as suggestions. |
| R2.m3 | RESOLVED | §4.1 deliver-only twins (25%/50%, re-planned); Table 6 rows | At 50% deliver-only, thr.-k beats Baton (3.3 vs 3.1). The introduction admits this tie; the abstract (l.29–40) does not. |
| R3.1 | RESOLVED | §3.1 paragraph separating the two classes by contract and customer impact; pool size given as a by-product (p95) in §4.6 | The pool-size figure "median 0, max 7" pools all six gates and hides the Det median of 3.5. The model assumes a standby vehicle meets the route at the next stop with no wait (response time is not modelled). Add that to Scope. |
| R3.2 | RESOLVED | F_sb reinterpreted as the holding cost (§3.1); Ineq. (7) declared non-essential; Table 15 row F_sb=60; reserved-pool case in §4.6 | For the §4.6 caveat on reserved rates, see R1.2. |
| R3.4 | RESOLVED | Table 5 now has a Baton column (69.1/78.5/93.5); Table 7 has a Baton row (33.7) | Table 5's Baton figure rests on an arbitrary flat return price ("half a handoff", with no lateness term). Say so next to the number in the text. |
| R3.5 | RESOLVED | §4.3 "Two reference points lie above Baton…"; Table 13 (City: Baton 12.9 vs DP 12.2 at N=2×10⁴) | — |
| R2.m4 | RESOLVED | Fig. 2 split into fig2a (two panels) and fig2b (Fig. 3); fonts are legible | Fig 2a(a): the spike-day trace keeps going above Q after the breach at stop 12 (about 158 kg at stop 14), which cannot happen physically; cut it at the breach. Figs 2b/2a and Fig 7 use "$" while the text uses "currency units". |

## New issues introduced or exposed by the revision

1. **MAJOR — the §4.6 pool experiment settles its own answer (l.1757–1800, Table 12, Finding 5 l.1978).**
   - Problems: (a) the plan families chosen, (b) equal reserved and on-demand rates, (c) a naive S=0 comparator. Together they make "a reserved pool never pays" and "shadow pricing beats ignoring the cap" close to foregone conclusions.
   - Side result the paper does not state: on Dethloff SAA plans, S=0 shadow (8.3) equals pay-per-use (8.3). The standby lever is worth nothing there once returns are available. That is a practice finding in its own right.
   - The sentence "expected handoffs per plan-day far below one" (l.1784) is false for Det plans.
   - Minimal fix:
     - add Dethloff Det plans to Table 12;
     - report the break-even reserved holding cost (the value of the S-th vehicle) instead of S* alone, and/or sweep the reserved rate below F_sb, for example 5 and 10;
     - replace the naive S=0 comparator with the fallback-menu policy itself;
     - tone down Finding 5 to "at equal reserved and on-demand rates".

2. **MAJOR — the pool that operations would actually run is not tested.** The default model rests on a metro-wide pool "large relative to one plan". The capacitated test caps the pool per plan, which is exactly the setting the paper says is uneconomic. Minimal fix: run one experiment with a pool of S vehicles shared across all plans or instances of a family on the same day, or state explicitly that the metro-wide coupling is untested.

3. **MAJOR — Finding 4 "geography selects the lever" (l.1971) and the §4.3 explanation (l.1465–1466) contradict the paper's own model.**
   - Depots are not the difference. `make_city_instances.py` puts every city depot at the centre (District 1, Hoan Kiem, Midtown, Paris centre, People's Square), so city depots are central too.
   - Detours are not long in cost terms. For Hanoi-100-1 Det routes the detour part of R_k is 0.16–1.0 currency units, against 0.4–7.7 on CMT1X. The per-customer lateness term (1.5·(m−k)) dominates R_k.
   - Congestion does not exist in the model: distances only, no travel times.
   - The likely driver is the load profile. City pickups are 0.2–0.8 of deliveries, so the net load drifts downward. Capacity also differs (150 kg vans).
   - The same numbers weaken the Intro claim that a return is "nearly free downtown and prohibitive at the urban fringe" (l.159). With c_km=0.1 per km, geometry moves the prices only slightly: H_k varies 30.0→28.6 along a city route.
   - Minimal fix: attribute the city result to the pickup/delivery ratio and vehicle size; delete "congested" and "geography selects"; add a short experiment that varies the pickup fraction on the city instances, or a sentence admitting the confound.

4. **MINOR — the cost calibration lacks a source and one ordering looks odd to practitioners (§4.1 "Fleet economics", l.1341).**
   - A standby vehicle-day (20) costs less than a contracted planned vehicle-day (35), even though standby includes on-call availability.
   - The "Southeast-Asian gig-logistics rates" have no citation.
   - Lateness does not depend on how long the delay is: a 1 km and a 20 km detour carry the same p_late.
   - Minimal fix: cite or justify the rates, and list lateness that is independent of delay length in Scope.

5. **MINOR — deployment selection in the showcase example.** Fig 2a(b) and Fig 6 show a handoff of 16 of 18 stops taken at stop 2. On that route the model's own R_2 (about 3 + 1.5·16 + <1 ≈ 27.5) is below H_2 (about 29.7), yet deployment selection disabled the return. The paper's own flagship example is a case where selection removes a cheaper lever. Minimal fix: say so in the caption, or pick a route where the illustration is not open to this objection.

6. **MINOR — Baton-cf is better than the headline policy but sits in a robustness subsection.** Table 9: Det at ρ=0.6 gives 29.2 vs 27.9 (DP350k 29.5); SAA at ρ=0.9 gives 65.9 vs 64.5. Table 11 (City) gives 9.4 vs 5.6. A practitioner will ask which one to deploy. Minimal fix: recommend Baton-cf in Section 5, or explain why Baton stays the headline.

7. **MINOR — Remark 1 (l.873) says "The submitted version of this paper therefore recommended…".** Review history does not belong in the manuscript. Rephrase it as a hypothesis the experiment tests.

8. **MINOR — Figure 7 day (l.1548).** How the "high-demand day" was chosen is not stated, and "routine re-balancing on roughly half the routes" reads as typical. Give the percentile of that day's bill.

9. **MINOR — length and navigation.** About 62 preprint pages, 15 tables, 10 figures and 5 maps. These can move to an appendix or supplement with no loss to the argument:
   - Table 5 (synthetic) and the Table 7 RL block with its architecture paragraph (keep one sentence each);
   - Table 11 (fresh-start biases);
   - Table 14 (timings; keep 27 ms and 1 µs in the text);
   - Table 13 (keep Fig 8);
   - one of Figs 6, 7 and 10;
   - the proof of Prop 1.

   Also shorten the §4.4 heading, and add a one-paragraph roadmap of the "claims → table" mapping at the start of §4.

## Claim/number consistency problems

- **Pool size.** §4.6 says the 95% pool has "median 0 / max 7" for Baton and "1 / 8" for Baton-ho, pooled over all gates. Per gate (grand CSV): Det has median 3.5, max 7 (Baton) and 4 / 8 (Baton-ho); SAA has 0 / 0 (Baton). l.1784 "far below one" holds only for the gates that were tested.
- **Aggregation in Table 11.** City Baton-ho is 9.2 in Table 11 but 10.6 in Table 6; Salhi–Nagy 41.3 vs 42.3; Dethloff SAA 18.9 vs 20.1. Table 11 averages over routes (fresh.csv, 159 city routes), while Table 6 is plan-level. The caption does not say this.
- **Aggregation in Table 8.** City "thr. − Baton-ho" is 0.4, while Table 6 gives 10.6 − 9.7 = 0.9 (route- vs plan-level again). State the unit in the caption.
- **Reference rows.** Table 13 ref row gives DP50k 22.2 (Dethloff SAA) and 12.2 (City); Tables 2 and 6 give 22.1 and 12.1.
- **City return usage.** Table 11 City exact reset (9.9) is above Baton-ho (9.2). Part of "the return rarely pays on real networks" is therefore due to the conservative reset convention. Qualify the §4.3 and Finding 4 wording.
- **Units.** Figs 2a(b), 3 and 7 use "$"; §4.1 and Table 12 use "currency units".
- Checked and consistent: 86–96% (Table 2 ratios run 86.2–95.6%), "four of six gates" above the oracle, 2.1–10.5 pp in Table 3, 22.3–54.3% CVaR reduction, and Table 9 ρ=0.6 = Table 2.

## Recommendation signal

**Minor revision (borderline major on §4.6).**
The handling of R3.x, R2.M5, R2.M6 and R2.m3 is solid, and the scope paragraph is a real improvement. For practitioners, two headline "findings" still overreach. The capacitated-pool test leaves out the plans that actually use standby vehicles and prices reserved capacity at the on-demand rate. The "geography" explanation contradicts the paper's own central city depots and near-zero detour prices. Both can be fixed with modest re-runs and rewording; neither affects the core policy comparison.
