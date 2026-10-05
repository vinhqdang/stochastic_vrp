# Seat: Devil's Advocate (revision 1, BATON)

## Strongest counter-argument (200-300 words)

The paper says BATON is the lowest-cost implementable policy and that its advantage comes from *pricing* a richer action set. The experiments support a weaker claim: the only policy that holds *both* levers wins. On the five conservative gates, almost all of BATON's lead over the field is the depot return. BATON over BATON-ho is +31 to +34 pp (Table 3). Over the best competitor it is +6.5 to +10.5 pp, and that competitor is `restock`. `restock` is a one-parameter trigger (return iff W_k > cB, one scalar c for all stops), allows a single return and has no handoff lever at all (`core/costs.py` `simulate_restock`/`tune_restock`). No competitor combines a handoff rule with a return rule. There is no equal-data three-action DP (DP_N is handoff-only), no three-action rollout, and no two-lever position-dependent threshold. So the comparison never isolates "pricing by backward induction" from simply "having two levers". Among handoff-only policies, the value of pricing is small. On Det, BATON-ho beats thr.-k by 0.3 pp, and Table 8 puts a tuned threshold 0.4 to 1.0 pp behind BATON-ho on Det and city plans.

The size of the two-lever gain is also set by calibration rather than measured. R_k (eq. 6) has no fixed charge, no driver-time charge and no depot-dock charge. H_k carries a 20-unit standby day rate plus a 5-unit dispatch fee. Table 15 sweeps F_sb but never sweeps a return fee. Demand is synthetic throughout, including the "real" cities, where only the geometry is real. And the near-exact yardstick DP³50k shares BATON's reset convention and fresh-start construction. The 86-96% figure is therefore a convergence measure within one modelling choice, and it is not an optimality gap.

## Issue list

1. **CRITICAL, logic/overclaim.** Anchors: abstract (main.tex l.92, "every fixed threshold over-triggers"); Introduction l.182 ("it always over-triggers"); contributions l.219-220 ("any myopic or fixed-threshold rule"). Proposition 2 proves the inclusion only for the *myopic* rule C⁰_k > H_k, which under flat prices is the threshold τ = ω_F/C_fail. A tuned or arbitrary τ can under-trigger. τ→1 never fires, so its stopping region is empty. The paper itself reports tuned thresholds drifting "toward less triggering" (§3.3). Fix: restate as "the myopic threshold (τ = ω_F/C_fail) over-triggers". Either drop "every/any fixed threshold" or prove a separate statement about tuned τ.

2. **MAJOR, experimental design / attribution.** Anchors: Tables 2-3, Table 1 `restock`, §4.2 "first" observation. No two-lever competitor exists, as argued above. The "pricing" attribution is untested. Fix: add at least one of the following. (a) An equal-data three-action plug-in DP (DP³_N). (b) A two-lever heuristic: tuned return trigger per stop plus tuned handoff threshold, with multiple returns allowed. (c) Rollout over a three-action base. Then report BATON against the best two-lever competitor in Table 3.

3. **MAJOR, cherry-picking in headline range.** Anchors: abstract l.101 and Conclusion ("86%–96%"); `make_tables.py` l.183-187. The ratio is computed over the six Dethloff gates only. Other tables give lower values. Table 6: city real shops 10.6/12.3 = 86%, 25% deliver-only 4.9/6.4 = 77%, 50% deliver-only 3.1/6.7 = 46%. Table 9: day-factor SAA 7.4/12.4 = 60%, ρ = 0 SAA −0.1/7.4 < 0. Fix: state the range as "on the Dethloff benchmark" and give the full range across all tables.

4. **MAJOR, overclaim "lowest cost ... every setting up to one tie".** Anchors: abstract; §1 l.~267; §5. Two cases contradict this. Table 6, 50% deliver-only: thr.-k 3.3 against BATON 3.1. Table 9, ρ = 0 SAA: BATON −0.1, below both reactive and thr.-k (0.0). The text calls these ties "within noise", but Tables 6 and 9 carry no CI or paired test. Fix: report paired CIs for Tables 6 and 9. Then either say "ties or loses narrowly in 2 of N settings" or show that the CIs overlap.

5. **MAJOR, confounded managerial claim "geography selects the lever".** Anchors: §4.3 city paragraph; §5 "Fourth". City instances are evaluated only on Det-gate plans (Table 6 caption). The paper's own explanation is that returns become live on plans with slack (§4.2 "second" observation). On Dethloff Det, the return is also nearly worthless (+1.8 pp). The uniform-geometry twin also shows no return value (11.8 vs 11.7). This points to demand structure and plan slack, not road geometry: delivery-dominated vans, pickup fraction U[0.2,0.8], Det plans. Fix: run the city instances under at least SAA/Rob-G plans. Rephrase the claim as demand- and slack-driven unless the new runs support geography.

6. **MAJOR, circularity of the "near-exact" yardstick.** Anchors: §3.6 (DP³50k), §4.5 last sentence, Table 11. DP³50k uses the same conservative W = 0 reset and the same simulate-through-fitted-policy fresh-start value, with bins in place of isotonic regression. The paper admits the exact reset exceeds it. BATON at 2×10⁴ days also exceeds the city references (Table 13). The references are therefore not near-exact. Fix: under ρ = 0, Assumption 1 holds and the demand law is known. Solve the (k, W) DP exactly by numerical convolution, with the exact reset, and report BATON's gap to that value. Stop calling the binned programs "near-exact" otherwise.

7. **MAJOR, missing correct upper bound.** Anchors: §3.6, Tables 2/6/15 `oracle`, R2.m2. The only clairvoyant bound is handoff-only, and BATON "exceeds" it on 4/6 gates. That is explained, but the right bound is still not supplied. Fix: compute a three-action per-day clairvoyant bound (a deterministic DP per test day) and report BATON as a share of it.

8. **MAJOR, calibration drives magnitude.** Anchors: eq. (6), §4.1 fleet economics, Table 15, Table 5 ("depot return priced flat at half a handoff"). The return carries no fixed or driver-time cost, while the handoff carries F_sb + F_ho = 25. The +31-34 pp headline is a direct function of that asymmetry. Fix: add a return fixed fee and dwell sweep to Table 15. Report the break-even fee at which the three-action advantage vanishes.

9. **MAJOR, post-hoc variant and in-sample selection.** Anchors: §3.5 "Deployment selection", BATON-cf, Tables 9 and 11. Deployment selection exists because the three-action fit loses on city routes (5.6 vs 9.2, Table 11). It is an in-sample binary model choice, so "no tuning parameter" is overstated; §4.6's λ is also tuned per plan. BATON-cf is better or equal almost everywhere in Table 9 (e.g. ρ = 0.9 Det: 37.6 vs 35.4), yet it is absent from Tables 2 and 6. Fix: report BATON-cf in Tables 2 and 6, or state openly that it was introduced after the city result. Qualify "needs no tuning" to "no continuous hyper-parameter; one in-sample menu selection".

10. **MINOR, Proposition 3 abstract wording.** Anchors: abstract; §5 "Second". The abstract says over-triggering is "bounded by the probability of interventions on days that would have completed cleanly". That form holds only under flat prices. The general bound (13) also contains the declining-price term, and the experiments use geometric prices. Fix: add "under flat prices", or name both terms.

11. **MINOR, Table 8 bound "holds on 734/735 routes".** A theorem cannot fail. The one violation shows that both sides are estimates. The text also calls Det regret (3.8) the "smallest", yet City Det (2.0) and m ≤ 12 (2.7) are lower. The "all routes" 3.8% is a reactive-cost-weighted ratio (`make_tables.py` l.923) dominated by Det. Separately, the clean-day term exceeds the bound on City (5.7 > 4.8), which makes "90% of the bound" confusing. Fix: explain the violation as estimation error, state the weighting, and correct "smallest".

12. **MINOR, unexplained number drift.** The same quantity takes different values across tables:
    - BATON-ho: 41.3 (Table 11) vs 42.3 (Table 6) on Salhi–Nagy; 18.9 vs 20.1 on Dethloff SAA; 9.2 vs 10.6 on city.
    - DP³50k: 53.5 vs 54.0; 54.6 vs 56.4; 10.7 vs 12.3.
    - N = 1000 in Table 13 vs Table 2: thr. SAA 16.5 vs 16.3; DP50k SAA 22.2 vs 22.1.

    Fix: state the subset or seed behind each table.

13. **MINOR, wording in §4.7.** §4.7 says "the equal-data DP is no better" at N = 100 on SAA, but DP_N (−1.6) beats thr. (−2.4). Fix the wording.

14. **MINOR, unevidenced field claims.** Anchors: §3.3 ("observed in the field", "tuned thresholds drifted upward ... on long routes"), §1 ("the system this work replaces"). No data are shown. Fix: give evidence or mark these as anecdotal.

15. **MINOR, weak monotonicity test.** Anchor: Table 9 `viol.` column. At a nominal one-sided 2.5% level, the rate under independence is only 0.2%. That shows the test has little power, so rates of 0.1% and 0.0% do not show monotonicity. Fix: report a power check, e.g. an injected non-monotone bump.

16. **MINOR, Lean remark scope.** Anchor: §3 intro. The history-conditioned finite-tree versions of Propositions 2-3 do not transfer to policies that observe only W_k, which is what BATON is. Fix: say so explicitly.

## Round-1 items still open in substance

- **R1.4 (PARTLY).** thr.-k and roll.-θ answer the comment for the handoff lever only. The three-action claim still faces no flexible two-lever threshold (issue 2).
- **R1.6 (PARTLY).** Table 3 covers Dethloff only. Tables 6, 9, 13 and 15 have no CIs, and the "tie" claims depend on them. Route-level results are only "provided as supplementary data". The tail analysis in Table 4 omits `restock`, the strongest competitor.
- **R1.2 (PARTLY).** §4.6 is a real improvement. But λ is searched per plan, dispatch is FCFS, and no coordinated allocation is compared, e.g. priority by C_k − H_k.
- **R2.M3 (PARTLY).** The boundaries of the action set and city ≈ BATON-ho are acknowledged. The city conclusion is confounded with the Det gate (issue 5).
- **R2.m2 (PARTLY).** The explanation now appears prominently, but the correct three-action bound is still missing (issue 7).
- **R2.M4 (PARTLY).** Proposition 3 prices the *myopic* rule. The question asked about the threshold family, which is covered only empirically (Table 8, last column), and the abstract still overgeneralises (issue 1).
- **R2.m4:** NOT_ASSESSED (figures not inspected at size).
- Resolved in substance: R1.1, R1.3, R1.5, R1.7, R1.8, R2.M1 (with issue 15 caveat), R2.M2, R2.M5, R2.M6, R2.m1 (discussion only), R2.m3, R3.1-R3.5.

## Ignored alternative explanations or paths

- **Two levers vs pricing.** The gain over competitors may simply be lever availability (issue 2).
- **City result driven by demand, not geography.** The delivery-dominated protocol and Det plans explain "returns rarely pay" as well as road geometry does. The uniform twin supports the demand explanation (issue 5).
- **Dependence as the engine of the headline.** At ρ = 0, SAA savings collapse from 53.7 to −0.1 and Det from 27.9 to 15.5 (Table 9). The headline magnitudes are tied to the ρ = 0.6 benchmark protocol. No real demand log is used to justify that value.
- **Pre-departure action.** On Det plans the risk sits "at the first stop, where no policy can act". A k = 0 decision (lighter loading, pre-emptive split) is an obvious lever that the model excludes by construction.
- **Stronger rollout.** Secomandi-style rollout is implemented over the *reactive* base, which reduces it to the myopic rule. Rollout over thr.-k or BATON-ho as base is the natural stronger competitor.
- **Smoothed position-dependent threshold.** thr.-k's variance could be reduced by parametrising thresholds in remaining distance or remaining stops, instead of m − 1 free levels.

## Observations (non-defects)

- The WDRO gate is now honestly described as SAA at a tighter threshold (§4.1).
- The retraction of the submitted version's 500-path fallback (Remark 1) and the explicit reporting of BATON = BATON-ho on city routes are candid.
- Table 11 is a good response to R2.M2. Biases (i) and (ii) are measured and small, and bias (iii) is identified.
- The "scope of conclusions" paragraph correctly limits claims to relative rankings rather than absolute savings.
- The statistical unit (instance per gate), Holm correction and effect sizes in Table 3 are appropriate.

## Recommendation signal

**Major revision.** The structural contributions are largely sound, and the revision answers most round-1 comments in substance. However, Proposition 2 is overgeneralised in the abstract and introduction. The central attribution ("gains come from pricing a richer action set") is untested because no two-lever competitor exists. And the headline ranges are drawn from the most favourable table. All three are fixable with the targeted experiments and rewordings above, without changing the method.
