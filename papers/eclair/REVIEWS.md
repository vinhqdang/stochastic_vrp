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


---

## Review 4 (2026-08-12, third re-review; 6/10 borderline weak accept)

| # | Finding | Disposition |
|---|---------|-------------|
| **R4.1** | "The two certified mutants are near-misses inside the eps-tolerance" is UNSUPPORTED: a mutant label only means disagreement on >=1 screening instance, and a certificate does not prove err<eps (certifying err>=eps is the type-I event bounded by alpha). | **FIXED by measuring.** New `estimate_error_rate()` audits every issued certificate on 400 fresh micro instances with a Wilson interval; the experiment logs each certified candidate's descriptor, err_hat, CI, and below/above-eps classification. New S7.1 "Auditing the certificates" reports the audit, and Table 4 classifies certificates below/above/inconclusive vs eps instead of by construction label. |
| **R4.2** | Stale "calibration failure probability accounted in the level" in contributions; intro "suffices for validity" unqualified. | **FIXED.** Contributions now say unconditional for proved tiers, conditional for the entangled one; the intro carries the uniformity requirement in the same sentence. |
| **R4.3** | "Tier C is safe as a supplement" does not follow. | **FIXED.** Replaced: a valid tier does not immunize an invalid one (sound A/B factors times unsound C factors invalidates the product); tier C is valid alone or in combination only conditionally on Assumption 1(c). Figure 3's caption carries the same reminder. |
| **R4.4** | "Kelly-optimal" contradicts S5's honest "Kelly-style". | **FIXED** in both intro and S4. |
| **R4.5** | Acceptance depends on alpha AND eps, not eps alone. | **FIXED** in deployment guidance. |
| **R4.6** | Theorem 3's novelty should be calibrated (it is the zero-failure binomial rule as an e-process). | **FIXED.** New Remark 3 says so explicitly and locates the contribution in the composition with adaptive screening + fresh-data selection. |
| code | cert budget one-probe overshoot; single-seed MC test; certified-mutant identity not stored; `pick`/`abstained` aliases; "solver time" print. | **ALL FIXED.** Probe-count cap (no overshoot) with validation; MC boundary test over four seeds/1000 runs with pooled ~3-sigma threshold and per-seed ceiling; descriptors + audits logged; aliases removed and all call sites updated; print corrected. |
| presentation | duplicated "On routing:"; Fig 2 axis "solver time"; dense abstract. | **ALL FIXED.** |

## Review 5 (2026-08-12, artifact review; 4/10 reject-as-incomplete)

Procedural, and correct: the committed artifact was inconsistent.

| # | Finding | Disposition |
|---|---------|-------------|
| **R5.1** | `main.pdf` stale w.r.t. `main.tex` — reviewers get the PDF. | **FIXED.** Recompiled; a pre-commit consistency check now verifies PDF mtime > source mtime, that the shipped results file is a publication run, and that the manuscript's headline counts match the JSON. |
| **R5.2** | A 4-replication smoke run had OVERWRITTEN the 100-replication publication results. | **FIXED at the root cause.** `run_experiment.py` now writes atomically (tmp + rename) and any run below `PUBLICATION_REPS = 100` is diverted to `*_smoke<N>.*` filenames with a printed notice, so a smoke run cannot clobber shipped artifacts. Full 100-rep run re-executed with auditing. |
| **R5.3** | The empirical-error claim was not represented in shipped data. | **FIXED.** Shipped `mutation_experiment.json` (n_reps=100) contains the per-certificate audit log; manuscript numbers are read from it: 291 certificates over 300 runs, 290 audited entirely below eps, 0 above, 1 inconclusive. |
| **R5.4** | "Confirms the acceptance guarantee empirically" overclaims; procedural validation and output auditing are different questions; multiplicity. | **FIXED.** Abstract now states only what the audit found. New S7.2 "Two different questions: procedure vs. outputs" separates procedural validation (the mocked-boundary test) from output auditing (the 400-instance audits). Only the two certified mutants get reported intervals, Bonferroni-adjusted to hold simultaneously at 95%; reference-faithful builders have err = 0 identically. |
| **R5.5** | Report certification probe counts/costs. | **FIXED.** New paragraph: a successful attempt is deterministic at 29 fresh tier-A probes (~148 ms, ~24% on top of the 622 ms screening budget); failed attempts stop at the first alarm. |
| code | cert_budget semantics; validate cert_max_probes. | **FIXED** (documented conversion; type/range validation). |

**The decisive new finding, stated as measured:** of 291 certificates,
290 audit entirely below eps = 0.10 and none above. Two certified
models were mutants: one a legitimate near miss (err_hat 0.040,
CI [0.025, 0.064]), one INCONCLUSIVE at n = 400 (err_hat 0.108,
CI [0.081, 0.142]) — the audit cannot say whether it is a near miss
or a genuine false certification, and the paper does not claim
either. At most one possible false certification in 291, against an
alpha = 0.05 budget that would tolerate roughly 15.


---

## Review 6 (2026-08-12, fourth re-review; major revision)

Two of these were outright errors in the paper, not presentation
issues. All six plus the smaller items are fixed.

| # | Finding | Disposition |
|---|---------|-------------|
| **R6.1** | The advertised Bonferroni-adjusted 97.5% intervals were never computed: `estimate_error_rate` hard-coded z = 1.96, so the shipped JSON held ordinary 95% intervals. | **FIXED.** `estimate_error_rate` takes `z`; the experiment audits at z = 2.2414 and records the level in every log entry. Recomputed intervals match the reviewer's arithmetic exactly: 16/400 -> [0.02315, 0.06826]; 43/400 -> [0.07753, 0.14721]. The manuscript now states the level, says the pair holds JOINTLY at 95% by Bonferroni, and presents the "at most one false certification" line as a simultaneous-confidence statement rather than a fact. |
| **R6.2** | Proposition 6's Hedge bound is wrong by a factor of two: rewards in [-L,L] rescaled to [0,1] must be scaled back by the range 2L. | **FIXED.** Bound corrected to L*sqrt(2T ln 3); the proof now shows the rescaling step (2L * sqrt((T/2)ln3) = L*sqrt(2T ln3)) so the constant is checkable. |
| **R6.3** | Table 4's budget (649 ms) was not the shipped run's (622 ms); and screening tests `while budget > 0` then charges afterwards, so it can overshoot. | **FIXED.** All budget figures now track the reported run (632 ms). The allowance is described as a LAUNCH THRESHOLD, not a cap, and realized expenditure is recorded and reported: mean 630-631 ms, max 711 ms across 300 runs - the overshoot is one probe, but one probe can be expensive, and the paper says so. |
| **R6.4** | The claimed pre-commit artifact-consistency check did not exist in the repository. | **FIXED by shipping it.** `code/check_artifacts.py` is tracked and verifies: PDF newer than sources+figures; results are a publication run; provenance present; manuscript counts (runs / certificates / audited-below / above) match the JSON; the quoted screening budget equals the run's and no stale budget figures remain; every quoted certified-mutant interval appears in the audit log. Currently 11/11 pass. |
| **R6.5** | Environment not reproducible; wall-clock costs feed calibration, budget and routing, so versions can change the statistical trajectory. | **FIXED.** `code/requirements.txt` pins cpmpy 0.9.29 / ortools 9.14.6206 / pytest 8.4.2 with the reference platform recorded; every results file now carries a provenance block (python, platform, machine, cpu_count, cpmpy, ortools, git commit, budget semantics). |
| **R6.6** | "Stays below alpha, as it must" overstates the Monte-Carlo test (it accepts alpha+0.02 pooled, 2*alpha per seed; one seed was 0.064). | **FIXED.** Now described as a sanity check, with the actual per-seed rates (0.064, 0.036, 0.044, 0.044; pooled 0.047 over 1000 runs) and the note that the MC standard error (~0.007) cannot resolve 0.05 from 0.06, so one seed above alpha is expected noise. The theorem establishes validity; the simulation only guards against gross implementation error. |
| smaller | Audit seeds not policy-dependent; duplicated assertion; provenance absent from JSON. | **ALL FIXED** (audit seed now includes the policy index; duplicate removed; provenance block added). |

Open editorial question (not a defect): at 36 pages the routing and
baseline material competes with the central line - falsification
followed by a fresh zero-failure acceptance test. Worth an author
decision before submission.
