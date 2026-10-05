# Audit 2b: response letter and portal replies, BATON R1 (CAOR-D-26-01885)

Scope: response_letter.tex/.pdf and portal_replies.md, checked against main.tex, main_revised.txt, tables/*.tex, macros.tex, make_tables.py, results/r1/*.csv, main_submitted.tex and the submitted-era macros (git c76f6c3). Lean axiom audit re-run (`lake env lean Axioms.lean`): every headline theorem depends only on propext/Classical.choice/Quot.sound and there is no `sorry` in the sources, so the Lean claims in the letter hold.

Coverage (g): all 23 comments (R1.1–R1.8, R2.M1–M6, R2.m1–m4, R3.1–R3.5) have a reply in both documents.

Numbers that check out (a): nearly every quoted number matches the tables and macros. That includes Table 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14 and 15 values, 240/240 and −8.3% in the submitted version, 50→49 RL routes, "single-digit milliseconds", "most comprehensive", "no room for doubt", the old Remark "N ≲ 500", the entropy/learning-rate text, the old Prop. 1 "no worse than the clairvoyant", and "92–99%" in the submitted abstract. The remaining problems are wording, pointers and scope, listed below.

## Mismatches

1. **Letter R1.1, R2.M1; portal R1.1, R2.M1; manuscript §4.5 (main.tex 1674).** Claim: "Baton keeps the highest saving among the implementable policies under every law." Actual: Baton-cf, which is implementable, beats Baton in 8 of the 10 rows of Table 9 (e.g. ρ=0.9 Det 37.6 vs 35.4; ρ=0.3 SAA 36.6 vs 36.3). Table 9 also shows only thr, thr.-k and Baton-ho as competitors, although dependence.csv holds π3, roll.-θ, restock and DPN too (all below Baton, except for ties at ρ=0 SAA). The text says "re-fits and re-tests every policy". Fix: write "the highest saving of all competing policies (Baton-cf, its variant, is higher still)". Add to the Table 9 caption that π3, roll.-θ, restock and DPN were run and stay below Baton.
2. **Portal R2.M1.** Claim: "BATON stays best among implementable policies." This drops the ρ=0 SAA tie that the letter and portal R1.1 both state. Fix: add "(one tie within noise, SAA plans under independence)".
3. **Letter R1.3.** Claim: "We state the exact reset as the recommended specification for deployment." Actual: no such statement exists in §3.5 or §4.5 (grep "recommend" finds only Remark 1). Fix: add one sentence to the end of §4.5 "fresh-start" paragraph, or delete the claim from the letter.
4. **Letter R1.3.** Claim: the published restocking competitor "uses the same reset". Actual: the manuscript says only that it is "restricted to a single return" (§3.5). Fix: state the competitor's reset convention in §4.1/§3.5, or drop the clause.
5. **Letter R1.4.** The body says "We now also say explicitly, after Proposition 2, that the critique applies to fixed and myopic rules". Changes says "after Proposition 3". Actual: the text is the paragraph "Proposition 2 also explains a field observation…" (main.tex after Prop. 3's discussion, §3.3). It says a stop-dependent threshold can represent the boundary but must be searched. It does not literally say "the critique applies to fixed and myopic rules". Fix: use one pointer ("§3.3, paragraph after Proposition 3") and paraphrase what that paragraph actually says.
6. **Letter R1.5 ("separated them visually in every table"), R2.m2 ("Every results table labels the oracle as handoff-only and separates the reference points"), R3.3 ("every table now separates the reference points"); portal R2.m2, R3.3, R3.5.** Actual:
   - Only Tables 1, 2, 3, 13 and 14 (and Table 6 by caption) do this.
   - Table 9 has DP350k and oracle columns with no separator and no "reference"/"handoff-only" label.
   - Table 11 has an unlabeled DP350k column.
   - Table 6 separates the columns in its caption but does not say the oracle is handoff-only.
   - Table 15 has "oracle (HO)" with no separator.

   Fix: add a "reference points" cmidrule group and "(HO)" in make_tables.py for Tables 6, 9, 11 and 15, or change the letter to "the main results tables".
7. **Letter R2.m3; portal R2.m3; manuscript §4.3 (1492), intro (264), conclusion (1946).** Claim: the position-dependent threshold "matches Baton within a few tenths" / "ties". Actual (Table 6, 50% deliver-only): thr.-k 3.3 vs Baton 3.1, so the competitor is numerically ahead. At 25% the gap is 2.4 vs 4.9, but the letter's sentence reads as if it covers both twins. Fix: "at 50% deliver-only the position-dependent threshold saves 3.3% against Baton's 3.1%". Also soften "the new twins show that the conclusions do not depend on it".
8. **Letter R2.M4; manuscript §4.4 (1614, 1621).** Claim: "The myopic regret averages 3.8%… The regret is smallest on deterministic (tightly packed) plans (3.8%)". Actual: the overall figure is a cost-weighted ratio of sums (make_tables.py l.923), not an average over routes. It equals the Dethloff-Det row (3.8) because Det reactive costs dominate. The smallest group is City Det (2.0%). Fix: "smallest on the city routes (2.0%) and short routes (2.7%), 3.8% on Dethloff Det, largest on SAA (6.8%)". Replace "averaged over all routes" with "pooled over all routes (total regret / total reactive cost)".
9. **Abstract (101), intro (266), §4.2 (1390), conclusion (1948); letter Correction 4 ("reports the full range, 86%–96%").** Actual: the ratio is computed over the six Dethloff gates only (make_tables.py l.184). Elsewhere it falls outside the range: Salhi–Nagy 53.0/54.0 = 98%, city 10.6/12.3 = 86%, uniform 90%, 25% deliver-only 77%, 50% deliver-only 3.1/6.7 = 46%. Fix: "86%–96% on the six Dethloff planning gates" in all four places and in Correction 4.
10. **Letter R3.5; portal R3.5.** In one paragraph the letter quotes DP50k on city as 12.1% (Table 6) and as 12.2% (Table 13). Table 13's N=1,000 row also differs from Tables 2/6 without explanation: city Baton 10.5 vs 10.6, thr 10.0 vs 9.7; Dethloff SAA thr 16.5 vs 16.3, DP50k 22.2 vs 22.1. Fix: state in the Table 13 caption that the budget experiment uses its own nested training draws and re-estimated references, or harmonize the values.
11. **Table 11 vs Tables 2/6 (letter R1.3, R2.M2, R2.M3 quote both).** The same quantities differ between the tables, and the caption gives no reason:

    | Benchmark | Baton-ho (T11 / T2,6) | DP350k (T11 / T2,6) |
    |---|---|---|
    | Salhi–Nagy | 41.3 / 42.3 | 53.5 / 54.0 |
    | Dethloff SAA | 18.9 / 20.1 | 54.6 / 56.4 |
    | City | 9.2 / 10.6 | 10.7 / 12.3 |

    Fix: state the aggregation and the route subset in the Table 11 caption.
12. **Letter "Summary of the main changes", item 2.** Pointer "(Section 4.5–Section 4.7)" also covers the deliver-only customers, which are in §4.1/§4.3 (Table 6), and the standby-dearer-than-emergency case, which is in §4.8 (Table 15). Fix: "(Sections 4.3 and 4.5–4.8)".
13. **Letter R1.2 ("report the pool size each policy would need"), Summary item 2 and R1.5 ("single-core timing of every policy", "Table 14 reports … for every policy").** Actual: pool sizes are reported only for Baton and Baton-ho. Table 14 omits π1, π2 and the endpoint ablations; RL appears only in the text. Fix: "for Baton and Baton-ho" and "for every competitor in Table 2 (RL in the text)".
14. **Letter R1.6 Changes ("the pooled p-values have been removed").** Actual: §4.4 still reports "losing on 39/49 routes (p ≤ 10⁻⁷)". This is a Wilcoxon test over routes that share test days (make_tables.py `_wilcox(best_rl[1], v2)`), which contradicts the experimental unit the letter now defends. Fix: drop the p-value, or aggregate to plans and report a paired CI.
15. **Letter R2.M2.** Claim: "Baton-cf … matches or improves on Baton (Table 9, Table 11)". Actual: Table 11 has no Baton-with-selection column, so nothing there supports the claim. In Table 9, at ρ=0 Det, cf is 15.4 vs 15.5. Fix: "matches (within 0.1 point) or improves on Baton in Table 9; Table 11 shows it recovers the city loss without deployment selection".
16. **Abstract (92), intro (220); letter R1.4.** Claims: "every fixed threshold over-triggers relative to the optimal rule" and "the stopping region of any myopic or fixed-threshold rule contains the optimal one (Proposition 2)". Actual: Proposition 2 is proved only for the myopic rule, i.e. the break-even threshold τ = ωF/Cfail. A tuned threshold with larger τ need not contain the optimal region, and §3.3 itself says tuned thresholds drift upward. R1 and R2 are theory readers and are likely to notice. Fix: "the myopic (break-even) threshold over-triggers; no fixed threshold can represent the optimal boundary under per-stop prices".
17. **Figure 5 caption (main.tex 1374), cited in letter R3.3 Changes.** Claim: Baton "exceeds the handoff-only clairvoyant bound … wherever plans leave slack". Actual: WDRO is a conservative (slack) gate, and there Baton scores 41.2 vs the oracle's 42.0. Fix: "on four of the six gates (SAA, Rob-G, Rob-BS, M-DRO)".
18. **Intro (264).** Claim: "the only exception being a tie … on a deliver-only city variant". Actual: §4.5 reports a second tie (ρ=0 SAA: thr.-k 0.0 vs Baton −0.1). Fix: "the only exceptions being two ties within a few tenths of a point, where recourse is worth least".
19. **Portal R3.3.** It quotes "lowest cost of every implementable policy" as literal manuscript text. The actual wording is "lowest expected cost of the implementable policies compared" (abstract), "lowest expected execution cost of every implementable policy" (intro) and "highest saving of every implementable policy" (§4.2). Fix: drop the quotation marks or quote exactly.
20. **Letter R1.4; manuscript §4.2 third observation (1416).** Claim: on the conservative gates thr.-k "does no better than the global threshold (-0.2% on WDRO)". Actual: on WDRO it is 1.1 points better than thr (−1.3), so the example contradicts the claim. Fix: cite SAA (15.4 vs 16.3) or M-DRO (−1.0 vs 0.5).

## Missing or weak replies

- **R2.M3:** the reviewer quoted 10.8% vs 10.7%. The letter answers with 10.6% vs 10.6% and does not say that the numbers moved because of the seed fix (Correction 5). Add a half-sentence.
- **R3.4:** the letter never says that the old Tables 3 and 5 are now Tables 5 and 7. The portal does. Add it to the letter.
- **R1.1/R2.M1:** the shape (violation) test is not reported at ρ=0.3 or under the day factor (Table 9 shows "–"). The reviewer asked for ρ=0.3 explicitly. Either run the test or say why it was skipped.
- **R2.M4:** Proposition 3 is a theorem, yet the bound fails on one of 735 routes. The letter and §4.4 report "734/735" without explaining it (presumably estimation noise). Add a clause.
- **R1.6:** route-level CSVs exist for the main tables (svrpspd_wdro/results/routes/, tracked in git) but not for the R1 experiments (Tables 8–13). Make sure the files are actually uploaded as supplementary material, as promised.
- **R1.6 tail risk:** Table 4 on Det plans shows Baton cutting CVaR95 by 22.3%, less than thr (23.6%) and Baton-ho (23.8%). §4.2 nevertheless says "the gains extend to the tail". Acknowledge this in §4.2.
- **R1.5 comparable compute:** the reply reports times but runs no equal-compute comparison. The argument that the comparison is "generous to the learner" is acceptable, but it is an argument rather than an experiment.
- **R2.m4:** the figure font sizes were not verified in this audit.

## Leftover wording in the manuscript

- None of the flagged phrases remain in main.tex or tables/: "single-digit milliseconds", "GPU", "five orders of magnitude", "eleven competitors", "50 routes", "most comprehensive" and "no room for doubt" are all gone. "Lowest … cost" now always comes with "implementable" (abstract 97, intro 262, conclusion 1945).
- main.tex 901: "keeps the fresh-start value **below** a single number per stop" is a garbled edit. It should read "keeps the fresh-start value a single number per stop".
- main.tex 1521 ("a few hundred dollars") and the Fig. 7 caption at 1548 ("\$341") use dollars, while everything else is in "currency units". Harmonize.
- main.tex 1645: "fitting orders of magnitude slower" (the enriched-statistic negative result) has no reported number behind it. Either give the time or soften the claim.
- main.tex 220 and 92: "any myopic or fixed-threshold rule" / "every fixed threshold" overstate Proposition 2 (see Mismatch 16).
- "general in form" (2011) is retained on purpose, with its boundaries now stated. This is fine.

## Tone

- **R3.2** opens with "The holding cost is in the model". This flatly contradicts the reviewer, and the submitted text never called F_sb a holding cost. Suggested opening: "Thank you; the original text did not make this clear. The standby day rate F_sb … is the holding cost…".
- **R3.5** opens with "Neither is a competitor; both are reference points". This reads as dismissive of a "Why?" question. Suggested: "Thank you; the original tables did not make this distinction clear. Both are reference points rather than competitors…".
- **Opening paragraph:** "We have done all of this." The reply is mostly complete, but Mismatches 3, 6 and 14 are promises the manuscript does not yet fully deliver. Soften to "We have addressed each of these points", or fix the gaps first.
- **R2.m3:** calling 3.3 vs 3.1 in the competitor's favour "a tie" and "matches" reads as spin (Mismatch 7). State the numbers directly.
- The rest is appropriately appreciative and concessive: R1.6 admits the pooling was inappropriate, R3.3 admits the wording was imprecise, and R2.M3 agrees on both counts.
