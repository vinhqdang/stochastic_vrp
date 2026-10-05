# Editorial Decision and Revision Roadmap — Review Round 1 (simulated)

**Manuscript:** *Minimum Weighted Hazard-Exposure Dispatch: Complexity, an Exact Algorithm, and an FPTAS* (revised version, `submission/main.tex`)
**Venue modelled:** Journal of Combinatorial Optimization
**Date:** 2026-10-05
**Mode:** `full` panel, one round, cold read of the revised manuscript (the reviewers did not see the response letter).
**Panel:** five role-separated seats — Journal-Fit (R0), Methodology (R1), Domain (R2), Perspective (R3), Devil's Advocate (R4). Individual reports are in this folder.

> This is a simulated review to find weaknesses before a real reviewer does. It is not a prediction by the journal. Several literature pointers in R2/R3 are marked `[UNVERIFIED]` by the reviewers themselves; none may be cited before it is checked.

---

## Decision: **MAJOR REVISION**

Honest risk note: the panel splits on contribution, not correctness. Two seats (R0, R2) say the paper would be **rejected** if it is resubmitted without a new result or a narrower claim. The paper is correct and unusually candid, but a combinatorial-optimization journal wants something a scheduling theorist does not already know.

| Seat | Recommendation | Confidence | Core objection |
|---|---|---|---|
| R0 Journal-Fit | Major (borderline Reject) | 4 | Contribution small; "network" framing unsupported; related work misses closest areas |
| R1 Methodology | Minor | 5 | No wrong theorem; proof-presentation gaps; reproducibility; case-study protocol |
| R2 Domain | Major (Reject if nothing new) | 4 | No research-level new theorem; Table 2 "new" labels wrong; release-date claim wrong for m=1 |
| R3 Perspective | Major | 4 | Camp Fire optimum is an artefact of the depot-round-trip assumption; ethics; "serving" undefined |
| R4 Devil's Advocate | (no score) | — | C1 and C2 CRITICAL, both validated |

## What the panel agrees on

1. **Every theorem checked is correct** (R1 re-derived all proofs and brute-forced Thm 5 for m=1..3; R4 re-ran the central claims). No panelist found a false statement.
2. **The new content is elementary** (R0 W1, R2 W1–W3/W8, R4 C2). The classification, the m-vehicle greedy, the two-site heuristic counterexamples and the tightness proposition are corollaries, known constructions, or tightness of a deliberately handicapped variant.
3. **The depot-round-trip premise does not fit the paper's own Camp Fire geometry** (R3 W1, R4 C1).
4. **"Polynomial exactly when" needs P≠NP** (R0, R1, R2, R4 — four seats).
5. **Code and the Lean library are "on request" only** (R0, R1, R4).
6. **The paper is honest about what is classical.** This is a strength and every seat says so.

## Disputed points, with arbitration

| Issue | Positions | My verification | Ruling |
|---|---|---|---|
| **Camp Fire: does the depot-spoke model change the answer?** | R3 W1 and R4 C1 say a chained route serves all four communities; R1 only checks the arithmetic of the paper's model | **Confirmed.** Using the paper's own coordinates, speeds and hazard minutes, with great-circle travel between sites (a lower bound on road distance), a chained route serves all four sites on time at both 50 and 80 km/h (weight 38,571 vs the paper's 26,218 / 26,928). The reviewers report it still serves three or four sites at a 1.3 road factor (not re-checked by me). | **Valid, CRITICAL for the case study.** The case study must either adopt a path model or be explicitly demoted to "a worked illustration of the model, not evidence about the fire". |
| **26.6% mean retention (nominal plan)** | R1 gets ≈22% for a literal fixed route and ≈26% only if Concow is skipped when late; R4 gets 21.7% | **The paper's script does produce 0.266** (it is a fixed Concow→Paradise sequence in which a late Concow still consumes its time, and which drops sites deleted by Assumption 1 in a draw). The text does not state this protocol, so independent replication gives a different number. | **Valid as a documentation defect**, not a numerical error. State the evaluation protocol and give Monte-Carlo standard errors. |
| **Arrival reading "applies unchanged"** | R4 M1: true for algorithms, not for the interpretation attached to Thm 9; hardness with constant *hazard* deadlines is not proved | Not independently verified; R4 reports a 400/400 brute-force check of a reduction. | **Valid.** Either prove hardness for constant hazard deadlines under the arrival reading (R4 sketches how) or restrict the interpretive sentence. |
| **Release dates "break" equal-cost tractability** | R2 W5: for m=1 equal-length jobs with release dates and weights are polynomial (Baptiste 1999); Heeger–Molter is unweighted/parallel machines | Not verified by me (reviewer confirmed the paper's existence by web search). R1 independently recalls the same. | **Likely valid.** Rewrite, after checking Baptiste and Heeger–Molter. |
| **"Tight up to 1−ε²"** | R1 W2: the same algorithm provably achieves ≈1/(1+ε); R4 M6: tightness of a variant a completion step repairs | R1 checked the refinement numerically on 4,000 instances (0 violations). Not re-derived by me. | **Valid.** Either strengthen Theorem 4 to ≈1/(1+ε) (then Prop. 8 is exactly tight) or reword. |
| **Severity of the contribution gap** | R1: Minor overall; R2: Critical | Not a factual dispute; a matter of venue standard. | **Treated as the decisive issue.** It sets the decision at Major with Reject risk. |
| **Heterogeneous speeds extension** | R4 M4(c): the matroid greedy extends to vehicles of different speeds (300/300 brute-force); the paper lists this as open | Not verified by me. | **Plausible and valuable** — if it holds up, it is the cheapest route to a result that is not in the current paper. |

## Items that need no new research (do these first)

1. State "NP-hard, hence not polynomial unless P=NP" everywhere "polynomial exactly when" appears (abstract, Thm 9, conclusion); drop "complete classification"; relabel Table 2 "new" items (naive EDD ratio, tightness, classification, m-vehicle greedy) honestly.
2. Fix the release-date sentence (intro item 5, Section 6.1) per Baptiste 1999 / Heeger–Molter, after verifying both.
3. Fix the main-text proof of Thm 4 (wrong inequality, draft-like self-correcting prose) and add the dominance lemma for the value-indexed recursion g; write Eq. (1) and the g-recursion with explicit cases.
4. Union–find remark: path compression alone gives O(n log n); α(n) needs union by rank. Tie handling in Lemma 1.
5. Lean statement: remove the inconsistent scope sentence (Thm 2 and Thm 5 *are* complexity-class/running-time claims), say exactly what is formalised, and either deposit the library with a DOI or drop the sentence. Add the Lean work to the Author Contributions / AI-use statements if it stays.
6. Deposit code and generator parameters (seeds, D, M, hardware) in a public archive; label Table 3's adversarial column as analytic (equals the closed form of Prop. 8); give spreads.
7. State the Camp Fire evaluation protocol (fixed route vs skip-if-late; floor rounding; the deletion rule) and the tipping weight ratio at which "Paradise first" stops being optimal.
8. Remove the stream-of-consciousness in the Appendix B proof steps.

## Items that need real work (decide the strategy)

| # | Item | Options |
|---|---|---|
| A | **A result that is not classical** (R0 W1, R2 W1, R4 C2) | (i) Extend the matroid greedy to vehicles of different speeds and/or give an approximation scheme for MWHED-m with m part of the input (compare multiple-knapsack PTAS); (ii) prove hardness for constant *hazard* deadlines under the arrival reading and a finer dichotomy (agreeable weights, bounded distinct values); (iii) re-scope as an expository/application paper and target a transportation or applied-OR venue, dropping the "new" labels. |
| B | **Camp Fire** (R3 W1–W5, R4 C1/M3) | Add a path/orienteering variant on the same data (the "relocating depot" variant already mentioned), report where the depot-spoke model is and is not appropriate; add a weight-perturbation and correlated-error scenario; define "serve" and on-site time; add a short ethics/equity paragraph on sacrificing low-weight sites. Or demote the case study to a one-paragraph illustration. |
| C | **Positioning** (R0 W2–W3, R2 W4/W6/W7) | Cut the real-time-systems and matroid-frontier paragraphs; add scheduling with rejection / order acceptance, deadline-TSP and orienteering, knapsack-FPTAS (JOCO audience), tardy-processing-time fine-grained results, Karp 1972, Lawler's agreeable-weights result, equal-processing-time scheduling. **Verify every reference before adding it.** |

## Recommended order of work

1. Do the eight "no new research" items (one working session).
2. Decide A (this is a strategy decision for the author, not something to improvise): the heterogeneous-speed extension is the lowest-cost candidate and should be tested first.
3. Rework B or demote the case study.
4. Re-run a second review round (`re-review` mode) on the revised manuscript.

## Bibliography caution

R2 could not confirm several recent entries in the bibliography (listed in `R2_domain.md`, W7) and flagged one possibly wrong venue ("Proceedings of AAMAS 2026" for an arXiv entry submitted 2026-05-31). None was found wrong, but each needs checking against its source before submission.

## Process disclosure

- The Phase 0 panel configuration was set without asking the author to confirm it.
- The sprint-contract machinery (paper-blind pre-commitment, executable conformance and panel checkers), the provenance artifact and the cross-model track were **not** run. All five seats ran on the same model family, so their agreement is not independent evidence; the seats were isolated from each other but share the same blind spots.
- No file of the manuscript was modified by this review.
