# ECLAIR — external review ledger

## Review 1 (2026-08-10, pre-submission, LLM reviewer arranged by the author)

Verdict: reject in current form, 3/10, confidence 5/5. Full text held
by the author; the substantive findings and our disposition:

| # | Finding | Our assessment | Disposition |
|---|---------|----------------|-------------|
| 1 | The claimed guarantee controls false REJECTION of faithful models, not false ACCEPTANCE; "certified at level alpha" is wrong; pick-acc 0.890 at alpha=0.05 proves the gap; e-BH controls FDR of the rejected set only | **Correct — the central flaw** | Add a genuine acceptance guarantee: a second, opposite-direction e-process per final pick testing H0': err(m) >= eps on FRESH exact-oracle instances (pass multiplies wealth by 1/(1-eps); Ville gives P(certify a model with err >= eps) <= alpha, selection-robust because the probes are fresh). Reframe survival as "not falsified"; reserve "certified" for picks whose certification process crosses 1/alpha. Rewrite abstract/intro/theory accordingly. |
| 2 | Clopper–Pearson calibration does not establish the history-uniform conditional bound of Assumption 1 (randomness of the bound, aggregation, cross-family transfer, tier-C pool dependence) | **Correct** | Restructure: tier A with the hand-written independent checker has conditional null alarm rate ZERO by construction (proved, no calibration); tier B zero by proof; tier C becomes explicitly assumption-based, stated as a deployment assumption, with the calibration failure probability delta folded into the level (alpha + delta). Empirical transfer results stay as support, not proof. |
| 3 | Repair tournament not implemented; e-BH proposition glosses adaptive family size; pick ignores the e-BH rejection set | **Correct** | Scope the proposition to the static pool as implemented; align the pick with the e-BH set in code; move repair to future work. |
| 4 | Baselines got tier C only while ECLAIR got A+B+C; "same probes" false; 0.0612 vs 0.05 not significant (binomial p≈0.22); pools not identical across policies (shared RNG) | **Correct** | New fair design: identical per-rep pools (dedicated seeds); a fixed-n baseline consuming the SAME probe stream and SAME e-factors as ECLAIR (isolates the accounting); ECLAIR-C vs fixed-C; Wilson CIs on every rate; >=300 reps; delete "breaks its nominal level" unless significant. |
| 5 | LLM experiment is a smoke test (all-null pool, weak proxy labels, intake catches everything) | **Correct** | Reword; strengthen proxy labels (more instances); no guarantee-language for this section. |
| 6 | Oracle problem partly relocated (manual checkers/relations); "faithful" = optimum-value agreement only | **Correct** | Scope statements; add solution-level feasibility checking to the oracle probe; quantify manual effort per family; retitle claims to optimum-value agreement where applicable. |
| 7 | Routing is Kelly-STYLE (plug-in, misspecified), cost-blind wins on the real testbed; Hedge proposition assumes full information unavailable under bandit feedback | **Correct** | Honest rename + report cost-blind as testbed winner prominently; restate the regret result under explicit full-information idealization or with a bandit (EXP3) bound; keep the 20:1 simulation labeled as controlled. |
| 8 | Repro/presentation: pools-identical claim false, "exactly"-language, missing CIs, "two to three alarms" arithmetic slip, solver-seconds naming, pending-verification footnote, "???" in ref 21 | **Correct** | All fixed in the revision. |

Response tracking: this file + the revision commits reference the
finding numbers (R1.1 ... R1.8).

**Revision executed 2026-08-12 (all eight findings addressed):**
R1.1 eps-certification theorem + implementation (certify.py cert
phase; testbed cert-acc 298/299, the one exception inside the stated
eps-tolerance); manuscript reframed screen-then-certify throughout.
R1.2 Assumption restructured (A/B proved, C explicit deployment
assumption; calibration failure delta folded into the level).
R1.3 e-BH scoped to the static pool; pick e-BH-consistent in code;
repair explicitly not claimed. R1.4 fair baseline design (identical
pools, probe-diet groups, same-stream seq-vs-fixed with nested
rejection sets, Wilson CIs, 300 reps); "breaks its nominal level"
withdrawn. R1.5 LLM study reworded as smoke test. R1.6 manual effort
quantified; certification scope stated as optimum-value agreement on
the enumerable distribution. R1.7 routing renamed Kelly-style;
cost-blind parity reported; Hedge proposition restated under explicit
full-information idealization with bandit caveat. R1.8 identical
pools implemented, finite-sample-zero language, CIs added, worked-run
arithmetic fixed, probe-seconds naming, footnote citations resolved
(Li & Hai arXiv:2607.16646; Zadorojniy et al. arXiv:2511.16383),
bib "???" fixed.


---

## Review 2 (2026-08-12, adversarial re-review of the revision)

Verified the revision finding-by-finding against manuscript AND code.
Verdict: R1.2/R1.3/R1.5/R1.7 FIXED; R1.1/R1.4/R1.6/R1.8 PARTIAL, with
two substantive defects and a numbers-hygiene list.

| Finding | Issue raised | Disposition (2026-08-12) |
|---|---|---|
| **R2.1** | Theorem 3's multi-attempt union bound is wrong: k attempts at threshold 1/alpha give level k*alpha, and the selection-robustness argument fails for attempts after the first (the 2nd candidate is chosen *because* the 1st attempt's fresh probes alarmed). Code attempted every survivor. | **FIXED.** Single-attempt rule enforced in `certify.py` (`ranked[:1]`); Theorem 3 restated for one attempt; new Remark 2 explains why, and gives the k/alpha multi-attempt variant. Rerun: certification accuracy 287/287 (was 294/295 under multi-attempt), 10 runs abstain instead of retrying. |
| **R2.2** | Headline "298 of 299 certified outputs faithful" contradicted by shipped JSON (actual 294/295). | **FIXED.** All three occurrences (abstract, S7.1, conclusion) now quote the recomputed post-fix totals: **287 of 287**, with the abstention count stated. |
| **R2.3** | "peeking ... whose interval sits above alpha" — false, lower limit 0.0499 < 0.05. | **FIXED.** Restated as point estimate above alpha with an interval grazing it; called suggestive, not conclusive. |
| **R2.4** | Baseline exposure counts wrong (880/920 vs actual 870/930). | **FIXED** both places. |
| **R2.5** | Stale detection ranges (87-94% in contributions, 92-94% in S7.2 vs 96.4% in the appendix). | **FIXED** to 89-96% / 92-96%. |
| **R2.6** | tab:calib costs (5.2/6.3/12.4 ms) off vs shipped calibration.json. | **FIXED** to 5.1/6.2/12.3 ms. |
| **R2.7** | p1 = 0.473 not derivable from the paper (shrinkage rule undocumented). | **FIXED.** tab:calib gains a p1 (bet) column and the caption states the shrinkage formula and that misspecifying p1 costs power only. |
| **R2.8** | Residual "exactly"-language; "mean solver cost"; code docstrings still "solver-seconds"; worked-run 640 ms vs 633; n approx 290-300 vs 285-301. | **FIXED** (all; docstrings in probes.py/certify.py too). |
| **R2.9** | Abstract quotes bare alpha though tier-C runs carry alpha+delta. | **FIXED.** Abstract carries the delta caveat; a new paragraph in S7 states alpha=0.05 nominal vs 0.075 unconditional for tier-C runs, alpha exactly for the tier-A certification. |
| **R2.10** | R1.6's promised solution-level oracle check was silently dropped. | **FIXED as an explicit narrowing**, not a silent one: a new paragraph states that faithfulness here means optimum-value agreement, that the ambiguity study exhibits the blindness, that solution-level probes fit the framework unchanged, and that they need a candidate/checker variable-mapping contract our formalism-agnostic interface does not impose. Flagged as the most valuable extension. |
| **R2.11** | Arch figure still drew repair as a live component. | **FIXED.** Repair box/arrows dashed and captioned as deliberately-not-claimed future work. |
| **R2.12** | (new, self-caught) alpha-sweep appendix detection at alpha=0.05 (0.924) vs Table 4 (0.964) look contradictory. | **FIXED.** Appendix caption explains those runs are screening-only (no certification phase). |


---

## Review 3 (2026-08-12, second adversarial re-review; 5/10, major revision)

Accepted the falsification-vs-certification split as resolving the
original defect, but raised nine majors. All are correct; all are
addressed.

| # | Finding | Disposition |
|---|---------|-------------|
| **R3.1** | The alpha+delta screening guarantee does not follow: Clopper-Pearson covers ONE binomial parameter under the calibration design; it cannot establish the history-uniform conditional bound of Assumption 1(c). Propagates to e-BH. | **FIXED — claim withdrawn entirely.** Theorem 2 and Proposition 4 are now stated strictly conditionally on Assumption 1, with the caveat propagated verbatim into the e-BH proof. The assumption discussion states that calibration + transfer tests are empirical stress tests, NOT proofs, and names uniform-conditional calibration (stratified worst-case / online recalibration with a risk budget) as the framework's most important open problem. Tier-A/B-only configurations are exempt and remain unconditional. |
| **R3.2** | Title/abstract too broad for a certificate covering optimum-value agreement on the micro distribution. | **FIXED.** Retitled "Anytime-Valid Screening of LLM-Generated Constraint Models, with eps-Certification of Optimum-Value Agreement"; the abstract now names the scope and explicitly excludes feasible-set equivalence, returned solutions, and deployment-scale behaviour. |
| **R3.3** | The new certification path had no direct tests (7 tests; `test_certification_separates` never enabled it). | **FIXED.** Seven new deterministic tests with a mocked Tier-A oracle (14 total, all passing): threshold arithmetic at four (alpha,eps) pairs; certification after exactly `need` passes; alarm blocks certification and hard-falsifies; insufficient-budget abstention; single-attempt rule; e-BH-rejected candidates cannot be certified; and a 400-trial Monte-Carlo check that candidates with true error exactly eps certify at most ~alpha of the time. |
| **R3.4** | Abstention semantics did not match Algorithm 1 (`abstained` keyed to the screening pick). | **FIXED.** Two-stage outputs: `screening_survivor`, `certified_pick`, `screening_abstained`, `certification_abstained`; `abstained` now keyed to the terminal certified output. |
| **R3.5** | Certification probe costs omitted from kill cost. | **FIXED.** Certification probes update `State.spent` (with `cert_spent` recorded separately); Table 4 reports kill cost twice, screening-only and including certification. |
| **R3.6** | Certification denominator selectively conditional; stale "single certified-mutant case" sentence contradicting S7.1. | **FIXED.** Both denominators reported (285/300 unconditional; 0.99/0.99/0.90 conditional availability). The stale sentence is replaced by a statement of what the tolerance permits in general. |
| **R3.7** | LLM and ambiguity studies never ran the certification phase; README stale. | **FIXED.** Both runners now pass eps/cert_budget and were rerun: certificates issued in 5/5 LLM families under all three policies and in 3/3 ambiguity families, every certified model proxy-faithful/canonical. Section text updated; README corrected. |
| **R3.8** | Same-stream "consumes 4.4 probes" is counterfactual (all 13 execute). | **FIXED.** Restated as "would stop after 4.4 probes," with an explicit parenthetical that both arms execute all 13 so the comparison is exactly paired. |
| **R3.9** | Four inferentially incorrect remnants; Figure 1 lacked the certification node. | **FIXED.** All four rewritten; the architecture figure gains an eps-certify node fed from the survivor pick, with alarm/budget routing to abstention, and a caption naming it the only path to an accepted model. |

Smaller items: code README de-staled ("probe-seconds", "Kelly-style",
14 tests); e-BH set documented as the screening-stage set that the
certification attempt was selected from; a reproducibility caveat added
for the timing-derived budget (623-649 ms across runs).

**Rerun outcome under the corrected code (300 runs):** 285 certified,
283 faithful, 15 abstentions --- and the two certified mutants are
near-misses inside the eps = 0.10 tolerance, which the paper now uses
as the concrete illustration of what Theorem 3 does and does not
promise (previously a clean 287/287 run made this point abstractly).

Still open (declared as such in the paper, not claimed):
uniform-conditional calibration for Tier C; solution-level oracle
probes; repair-tournament FDR; deployment-scale certification.
