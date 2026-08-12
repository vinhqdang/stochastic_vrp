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
