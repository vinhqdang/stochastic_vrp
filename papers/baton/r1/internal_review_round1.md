# Internal verification review — BATON revision 1 (before resubmission)

Manuscript: CAOR-D-26-01885, revision 1 (`papers/baton/main.tex`, 62 pp.).
Purpose: check whether each of the 23 round-1 comments is resolved in
substance, find weaknesses introduced by the revision, and catch claim or
number inconsistencies before the revision goes back to the journal.

## Judge record

| Seat | Angle | Recommendation |
|---|---|---|
| Journal-Fit Reviewer | C&OR associate editor: fit, originality, claim proportionality | Minor revision (close to major) |
| Reviewer 1 — Methodology | design and statistics of computational experiments | Minor revision |
| Reviewer 2 — Domain/theory | stochastic VRP, optimal stopping, regression-based ADP, Lean correspondence | Minor revision (close to major on theory text) |
| Reviewer 3 — Practice | last-mile operations, standby capacity, managerial insight, readability | Minor revision (close to major) |
| Devil's Advocate | core-argument attack | Major revision |
| Phase-2B auditor | response letter and portal replies vs manuscript | 20 wording/scope mismatches, 0 wrong numbers |

Provenance, stated plainly:
- The five seats ran in separate fresh contexts, did not see each other's reports, and did not see the response letter (persuasion-blind). They judged the revision from the manuscript, the tables, the code and the raw results.
- A sixth pass, run separately, read the letter and the portal replies and checked every claim against the manuscript.
- All seats run on the **same model family**. The seats are separated by role, not independent reviewers, so correlated blind spots are possible.
- The round-1 reviews were written by the journal's human referees. The panel's own round-1 artifacts (roadmap, adjudication sidecar) therefore do not exist, and the contract-checked re-review mode could not run. This is the evidence-before-persuasion variant of that mode.

Full seat reports are in `internal_review/`.

## Decision

**Minor revision (internal): do not resubmit yet.** The core result
stands and the theory is mathematically sound:
- All four propositions were checked line by line.
- The Lean axiom audit was re-run: the formal proofs use only the standard axioms and contain no `sorry`.
- The headline statistics are computed correctly.
- There is no train/test leakage, and every spot-checked number reproduces from the raw CSVs.

Several claims, however, are broader than the evidence, one new competitor is under-optimised, and three new experiments have design gaps that a round-1 referee would find. Items M1–M7 below must be fixed before resubmission. S1–S3 are strongly recommended because they answer exactly what R1.2, R1.4 and R2.m2 asked.

### Devil's Advocate CRITICAL — adjudicated: VALIDATED

**The claim "every fixed threshold over-triggers" is broader than what is proved.** It appears in the abstract (main.tex l.92), the introduction (l.182 "it always over-triggers") and the contributions (l.219 "any myopic or fixed-threshold rule").
- Proposition 2 proves the inclusion only for the myopic rule `C⁰_k > H_k`, which under flat prices is the break-even threshold `τ = ω_F/C_fail`.
- A tuned `τ` above that level can under-trigger; `τ → 1` never fires.
- The paper itself reports tuned thresholds drifting toward less triggering (§3.3).
- Reviewer 2 and the 2B auditor reached the same finding independently of the DA.
- This blocks acceptance until reworded. It is a wording fix: no result changes.

## Revision response checklist (consensus across seats)

A verdict is PARTLY when at least one seat produced anchored evidence of a residual gap.

| Item | Verdict | Residual gap (anchor) |
|---|---|---|
| R1.1 state sufficiency | RESOLVED* | (k,W_k) sufficiency holds for the handoff problem. Eq. (bellman3) is exact only under independent increments, not under the factor case that Ass. 1 covers; bias (iii) is that gap and the text should say so (R2) |
| R1.2 shared standby | PARTLY | §4.6 uses SAA plans, where BATON already needs no standby. The reserved rate equals the pay-per-use rate, so S\*=0 is almost automatic. The naive baseline requests vehicles that do not exist. Refused routes cannot act at the refusal stop. The metro-wide shared pool is never capped (R1, R3, DA) |
| R1.3 depot-return transition | RESOLVED | Two descriptions of the reset conflict: "deliveries reloaded" (l.463, Alg. 2) vs "pickups unloaded only" (§3.5). Alg. 1 calls F_k "exact" (R2) |
| R1.4 flexible thresholds | PARTLY | thr.-k is under-optimised: it starts from "never act" and runs ≤ 2 sweeps. In sample it does **worse** than the global threshold it contains (SAA 16.3 vs 23.2; WDRO 0.0 vs 20.2 on 10 instances). The "variance" explanation in §4.2 is wrong. No two-lever (handoff + return) competitor exists (R1, DA) |
| R1.5 fairness | RESOLVED | Wording: "grid-tuned competitors cost about the same or more" is false for π3 and restock (5 ms); the online lookup is piecewise linear, not a step function (R1) |
| R1.6 statistics | PARTLY | Table 3 is correct. Tables 6 and 9–12 carry no intervals, yet the text claims "ties within noise". §4.4 still reports a route-level `39/49, p ≤ 10⁻⁷` test over routes that share days (R1, DA, 2B) |
| R1.7 positioning | PARTLY | Props. 2 and 4 are instances of textbook facts; add the monotone-MDP and optimal-stopping antecedents (Serfozo 1976, Puterman 1994 §4.7, Müller–Stoyan 2002). Yang et al. (2000) proved per-stop optimal thresholds but are listed with fixed-threshold heuristics. Minis & Tatarakis (EJOR 2011) is the closest prior model and is missing (verify) (R2, EIC) |
| R1.8 scope | PARTLY | The scope paragraph is good, but finding 4 ("geography selects the lever") and finding 5 (pool) overreach (R3, EIC) |
| R2.M1 dependence | PARTLY | The ρ sweep is done well. The shape test has no demonstrated power: it rejects 0.2% under independence against a nominal 2.5%. It was not run for ρ=0.3 or the day-factor law, where BATON reaches only 60% of DP³ on SAA (R1, R2, DA) |
| R2.M2 fresh-start bias | RESOLVED | The reference F is itself a binned-DP estimate; City bias −1.05% has the opposite sign to claim (i); report MAE (R1, R2) |
| R2.M3 action set / city | PARTLY | Boundaries are well stated. The explanation "detours through congested networks are long" contradicts the model: depots are at the centre, detour cost is 0.16–1.0 units on Hanoi routes, and the model has no congestion. The uniform twin shows the same null value, and cities run on Det plans only. The driver is demand profile and plan slack, not geography (R3, DA) |
| R2.M4 quantify Prop. 2 | PARTLY | Prop. 3 is correct and useful. The abstract generalises to "every fixed threshold" (CRITICAL above). The clean-day form is the flat-price case only (R2, DA) |
| R2.M5 co-optimisation | RESOLVED | — |
| R2.M6 exchangeability | PARTLY | The day-type-specific fit gets 1,000 promotion days against about 200 for the pooled fit. Part of the 30.9 vs 29.4 "recovery" is a five-fold data advantage (R1) |
| R2.m1 RL architectures | RESOLVED | — |
| R2.m2 clairvoyant bound | PARTLY | The explanation is prominent now. No three-action clairvoyant bound is reported, yet the intro says the oracle measures "how much saving is achievable at all" (EIC, DA) |
| R2.m3 zero pickups | RESOLVED | The text calls 3.3 (thr.-k) vs 3.1 (BATON) at 50% deliver-only a "tie"; say "narrowly ahead, p = 0.10" (R1, 2B) |
| R2.m4 Figure 2 | RESOLVED | The Fig. 2a load trace continues above capacity after the breach (R3) |
| R3.1 standby vs emergency | RESOLVED | The pool-size figure "median 0 / max 7" pools all gates and hides the Det median of 3.5 (R3) |
| R3.2 holding cost / Ass. 2 | RESOLVED | — |
| R3.3 lowest-cost claim | RESOLVED* | "Implementable" is now used everywhere. *"Up to one tie" is still overstated (see M3) |
| R3.4 BATON in Tables 3/5 | RESOLVED | — |
| R3.5 city vs DP50k/oracle | RESOLVED | The budget experiment shows BATON *exceeding* the binned references at 20k days, so they are not near-exact (see S7) |

Score: 12 RESOLVED, 11 PARTLY, 0 NOT_RESOLVED.

## Must fix before resubmission

- **M1 — Proposition 2 overclaim (DA CRITICAL).** Say "the myopic (break-even) threshold over-triggers" in the abstract, the introduction (l.182, l.219), the conclusion and §4.4. Present the clean-day bound as the flat-price case, and stop calling the tuned-threshold gap "the same pattern" as Prop. 3.
- **M2 — Headline range scope.** "86–96% of the near-exact program" covers the six Dethloff gates only. Across all tables the range is 46% (50% deliver-only) to 98% (Salhi–Nagy), plus 60% on day-factor SAA. Say "on the Dethloff benchmark" and give the full range once. Also, "six planning regimes" is attached to benchmarks that were run on Det plans only.
- **M3 — "Lowest cost everywhere up to one tie".** thr.-k is ahead at 50% deliver-only (3.3 vs 3.1, Wilcoxon p = 0.10). At SAA ρ = 0, BATON is −0.1 vs thr.-k 0.0 (p = 0.07). Add paired intervals for the key contrast in Tables 6, 9, 10 and 12, and report these two cases as narrow losses or ties with their intervals. Fix the second-tie omission in the introduction.
- **M4 — thr.-k optimiser bug.** Warm-start the coordinate descent from the per-stop cuts of the tuned global threshold (and optionally BATON-ho's boundaries), so that thr.-k ≥ thr. in sample by construction. Re-run Tables 2 and 6 (minutes of compute) and rewrite the §4.2 "variance" sentence to match what the re-run shows.
- **M5 — Manuscript and letter consistency.**
  - The letter says "every table separates reference points", but Tables 6, 9, 11 and 15 do not, or do not label the oracle as handoff-only. Fix the tables or the claim.
  - The letter says "pooled p-values removed", but §4.4 still has the route-level test. Replace it with an instance-level test or an interval.
  - The letter says the exact reset is "stated as the recommended specification", but no such sentence exists. Add it to §4.5, or drop it from the letter.
  - "Highest saving among implementable policies under every law" ignores BATON-cf, which is higher in 8 of 10 rows of Table 9, and Table 9 bolds BATON. Say "BATON or its cf variant", or bold cf.
  - "Regret smallest on deterministic plans (3.8%)": the city row is lowest (2.0%), and the 3.8% is a pooled ratio, not an average over routes.
  - Table 11 pools route costs while Tables 2 and 6 average per instance (Salhi–Nagy BATON-ho 41.3 vs 42.3; DP³ 54.6 vs 56.4 on SAA). State the aggregation in the captions, or unify.
  - Explain the 1/735 route where Prop. 3's bound fails as Monte Carlo error; the "90% clean-day share" is a ratio that exceeds 100% on the city row.
- **M6 — Theory text.**
  - After eq. (bellman3), state that it is the exact DP under independent increments and misspecified under factor dependence, with BATON-cf as the correction.
  - Qualify "(k,W_k) is a sufficient state" as holding for the handoff problem.
  - Replace the Clément et al. convergence claim, which covers a fixed linear basis, with nonparametric LSM results (e.g. Egloff 2005; Zanger 2013, to verify). Add a sentence that the step-k target is under the *fitted* downstream policy, so monotonicity holds only in the limit.
  - Add the classical antecedents under R1.7.
  - Prop. 2's boundary claim also needs Ass. 2.
  - Add one sentence listing the Lean modelling abstractions: abstract kernel, finite history trees (history-measurable rules), and abstract admissible class. State that the finite-tree results cover history-measurable rules, not the deployed W_k-rules.
- **M7 — Geography claim.** Either re-run the city instances on SAA/Rob-G plans (cheap), or reword §4.3 and finding 4 as demand-profile and plan-slack driven. Remove "congested networks", and soften the introduction's "nearly free downtown, prohibitive at the fringe".

## Should fix (targeted experiments; each hours of compute)

- **S1 — Two-lever competitor (R1.4, DA MAJOR).** Add an equal-data three-action plug-in DP (DP³_N, already implemented as `fit_dp_actions_cf`) and/or a per-stop return trigger plus tuned handoff threshold with multiple returns. Report BATON against the best two-lever competitor in Table 3. Without this, "gains come from *pricing* a richer action set" cannot be separated from "having two levers".
- **S2 — Three-action clairvoyant bound (R2.m2).** Compute a deterministic per-day DP over continue/handoff/return on the test days, and report BATON as a share of it.
- **S3 — Pool experiment (R1.2).**
  - Add Det plans, where standby is actually used.
  - Report the break-even holding cost per reserved vehicle, and sweep the reserved rate below F_sb.
  - Compare shadow pricing with λ = F_sb and λ = ∞ rather than only with naive.
  - Let a refused route act at the refusal stop.
  - Soften finding 5 accordingly.
- **S4 — Day-type confound (R2.M6).** Split the pooled 1,000-day history by type (~800/200) for the day-type-specific fit.
- **S5 — Shape-test power (R2.M1).** Inject a known dip and report the detection rate. Run the test for ρ = 0.3 and the day-factor law, and optionally on the three-action target.
- **S6 — Return-fee sensitivity (DA).** Add a fixed return fee and dwell sweep to Table 15, and report the break-even fee.
- **S7 — Reference programs.** Relabel DP50k/DP³50k as "high-data plug-in references in the (k,W_k) state" and document that DP³ uses the state-conditional fresh-start value (§3.6 currently describes the unconditional one). Optionally add an information-relaxation dual bound, or an exact DP under ρ = 0 by numerical convolution.

## Consider (wording, length, tone)

- **Length (62 pp.).** Move Tables 5, 7, 11, 13 and 14, one of Figs. 6/7/10, and the Prop. 1 proof to an online appendix.
- **BATON-cf.** Either report it in Tables 2 and 6 or say openly that it was introduced after the city result. Qualify "no tuning" as "no continuous hyper-parameter; one in-sample menu selection".
- **Small fixes:**
  - Remark 1 refers to "the submitted version".
  - "$341"/"dollars" vs currency units.
  - Garbled phrase at l.901.
  - "Orders of magnitude slower" (l.1645) is unquantified.
  - "[0,1)" should be "[0,1]" in Prop. 1.
  - σ definitions should restrict to k < T.
  - The §4.7 N = 100 wording (DP_N −1.6 beats thr. −2.4).
  - Field-observation claims should be marked anecdotal.
  - The Table 13 N = 1,000 row differs from Tables 2/6 by 0.1–0.3.
  - The tail claim on Det: BATON's CVaR reduction (22.3%) is below the thresholds' (23.6%).
- **Letter tone.** Soften the openings of R3.2 ("The holding cost is in the model") and R3.5 ("Neither is a competitor"). "We have done all of this" overpromises until M5 is fixed.

## What is confirmed sound

- Props. 1–4 are correct as stated, including Prop. 3 under position-dependent prices and Prop. 4 needing only E non-negative and non-increasing.
- The Lean library proves what it states, using only the standard axioms.
- Table 3 is correct: instance-level unit, percentile bootstrap (10⁴ resamples), Holm over 18 comparisons, rank-biserial effect sizes.
- There is no leakage: seeds are disjoint, and all tuning and deployment selection is on training days.
- Every quoted number in the letter matches the tables. The claims about the submitted version (240/240, −8.3%, 50 routes, "single-digit ms", "92–99%", entropy, the Prop. 1 proof error) are true.
- None of the removed phrasings ("GPU-trained", "five orders of magnitude", "eleven competitors", unqualified "lowest cost") survives in the manuscript.
