# Peer Review Report

## Manuscript Information
- **Title**: Minimum Weighted Hazard-Exposure Dispatch: Complexity, an Exact Algorithm, and an FPTAS (revised manuscript, CSoNet 2026 Journal Track / J. Comb. Optim.)
- **Manuscript ID**: not supplied
- **Review Date**: 2026-10-05
- **Review Round**: first-time cold read of the revised manuscript (simulated panel, FULL mode)

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology / technical correctness)

### Reviewer Identity
Expert in scheduling theory and algorithm analysis (approximation schemes, pseudo-polynomial DP, matroids, NP-hardness reductions) who also audits experimental design and reproducibility. Paradigm: theoretical/algorithmic paper with a computational study and one data-driven illustration.

### Review Focus
Line-by-line check of every lemma, fact, theorem and proposition (statements, proofs, constants, edge cases, complexity-class wording), consistency between statements, pseudocode and worked examples, and the soundness/reproducibility of the numerical study and the Camp Fire case study. The authors' Lean 4 claim is not credited; the written proofs were judged on their own. Everything below was re-derived by hand and, where possible, recomputed in Python (scripts in the review scratchpad: `chk.py`, `fptas.py`, `t1.py`, `cf.py`, `cf2.py`, `cf3.py`).

---

## Overall Assessment

### Recommendation
**Minor Revision** (from a methodology/correctness standpoint). I found no error that invalidates any stated theorem. The remaining issues are proof-presentation gaps, a few overstated or imprecise complexity/classification statements, an FPTAS analysis that is looser than what the same algorithm provably achieves, and reproducibility/disclosure gaps in the experiments and the case study.

### Confidence Score
5 (core expertise: scheduling/approximation/matroid proofs); 4 for the case-study modelling comments (adjacent expertise).

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment
The paper re-derives, in transport vocabulary, the classical results for 1||Σw_jU_j (EDD lemma, NP-hardness from the common-deadline/knapsack case, Lawler–Moore DP, a value-scaling FPTAS), adds an equal-dispatch-time matroid greedy extended to m identical vehicles, two heuristic lower bounds, a tightness proposition for the FPTAS analysis, and a coarse classification by which of (p, w, d) are constant. It is candid that most results are classical. I verified every proof. They are correct as stated; the issues are small: tie handling in Lemma 1, an unproved correctness step for the value-indexed recursion g in Theorem 4 (plus an indicator-times-value notation that is wrong as literally written), a union–find claim that needs union by rank, and "polynomial exactly when" that silently assumes P≠NP. The most substantive technical observation is that the FPTAS guarantee 1−ε is not what the algorithm achieves: a two-line refinement gives ≈1/(1+ε), which is exactly the ratio of the paper's own adversarial family, so Proposition 9's "gap 1−ε²" is an artifact of a loose analysis rather than of the algorithm. The experimental section is honest about its vacuous regimes, but the adversarial column is a closed-form restatement of Proposition 9, code and seeds are not provided, and the Camp Fire sensitivity study has unstated evaluation protocol and limited uncertainty coverage (weights, road factor, error correlation).

---

## Strengths

### S1: All theorem proofs are correct and the matroid proof is genuinely self-contained
Lemma 1, Fact 1, Theorems 2–5 and Propositions 6–9 hold as stated; Theorem 5's counting criterion, augmentation argument (Step 3) and the exchange-style optimality of greedy (Step 4) are correct, including the delicate "greedy examined o_j before g_k" step, and the union–find correctness argument (Step 5, the q* argument) is correct. I re-checked them independently (see "Verified as correct").
**Evidence Anchor**: text: §4.3, Theorem 5 "Let $k$ be the largest integer in $\{0,\dots,n\}$ with $N_B(k)\le N_A(k)$"

### S2: Transparent positioning and honest handling of vacuous regimes
The paper states what is classical, flags the K=1 vacuous case in the FPTAS proof, and in §7.2 explicitly says that ratio 1.000 with weights ≤100 "is *not* evidence about the approximation scheme", then designs the §7.3 study with active scaling. This is good methodological practice.
**Evidence Anchor**: text: §7.2 "this is \emph{not} evidence about the approximation scheme"

### S3: Worked examples and appendices are numerically right and aid checking
Example 1 (W*=23, naive EDD 16), Example 2 (knapsack 16), Table 2 (the whole DP table), Example 4 (greedy 22 / 26), Example 5 (K=1.25, w'=(4,6,2,8)) and Appendix A's 16-subset table all recompute exactly. Verification section §7.1 covers the right cross-checks (n! orders, subset enumeration, exhaustive m-vehicle assignments).
**Evidence Anchor**: table: Table 2 (f(i,t)) and Appendix A Table 6, all cells reproduced

### S4: Case-study separation of measured inputs from assumptions, and disclosure of the reading ambiguity
The arrival-vs-return reading (Remark 1) is formally reduced to the same model with shifted deadlines, and the case study reports how the answer changes under the other reading. The case-study geometry (distances, ignition distances, arrival extrapolation 12.1/14.2 km/h → 13.1, p_i at 50/80 km/h, deadlines, all four rows of both result columns of Table 9) is internally consistent; I recomputed it from the map coordinates and the tables.
**Evidence Anchor**: text: §7.6 "We separate real measurements from disclosed modeling assumptions"

---

## Weaknesses

### W1: Correctness of the value-indexed recursion g (the object the FPTAS runs on) is asserted, not proved; the displayed recursion is wrong as literally written
**Problem**: The proof of Theorem 4 says the recursion for g "by the same argument as in Theorem 3's proof, with the roles of the time and value axes exchanged". That is not quite the same argument. For g one needs an additional *dominance* step: the minimum-total-time feasible S' for scaled value v−w_i' is the best one to extend, because feasibility of S'∪{i} (i appended last in EDD order) depends on S' only through its total time, and a smaller total time can only help; and S' itself must be feasible. This is easy but is the one non-routine point in the construction and is not stated. Second, the displayed formula `g(i,v)=min(g(i-1,v), [g(i-1,v-w_i')+p_i ≤ d_i]·(g(i-1,v-w_i')+p_i))` is wrong as a literal expression: when the indicator is false the product is 0 and the min returns 0. Eq. (1) is protected by the "taken as −∞" sentence, but the g-recursion has no such convention (Algorithm 2 is correct). Third, in the K=1 case the proof says the g-DP "is exactly Theorem 3's dynamic program"; it is the value-indexed dual with unscaled weights, not f, so exactness again rests on the unproved g correctness. A fourth, trivial point: the proof writes "since $w_i\le K(w_i'+1)$" but the inequality actually used is $w_i\ge Kw_i'$ (Appendix B states this correctly).
**Evidence Anchor**: equation: Proof of Theorem 4, display "g(i,v)=\min\Bigl(g(i-1,v),\ [g(i-1,v-w_i')+p_i\le d_i]\cdot(g(i-1,v-w_i')+p_i)\Bigr)"
**Why it matters**: Theorem 4 is the paper's headline approximation result; its proof should be complete, and the K=1 "trivial" case plus the O(n³/ε) table size both rest on g.
**Suggestion**: Add a lemma: "for every i and v, g(i,v) equals the minimum total time of a feasible S⊆{1..i} with scaled value v", proved by induction using (a) EDD order with i last, (b) the dominance remark, and write the recursion with an explicit case (`+∞` if the deadline test fails or v<w_i'). Note that when K=1 one has V=Σw_i ≤ n·w_max < n²/ε, which is why the K=1 table is still polynomial; say this explicitly.
**Severity**: Minor
**Confidence**: 5 — core expertise: DP proofs for scheduling

### W2: The FPTAS guarantee 1−ε is not the guarantee of Algorithm 2; the same algorithm provably achieves ≈1/(1+ε), so Proposition 9's "tight up to 1−ε²" measures a loose analysis, not the algorithm
**Problem**: Besides W(Ŝ) > W*−ε·w_max (the paper's bound), the returned set has scaled value at least that of the single heaviest site, ⌊w_max/K⌋ = ⌊n/ε⌋ (that site is feasible by Assumption 1), so W(Ŝ) ≥ K⌊w_max/K⌋ > w_max − K = w_max(1−ε/n). With W* = x·w_max (x≥1) this gives ratio ≥ max(1−ε/x, (1−ε/n)/x), minimized at x ≈ 1+ε(1−1/n), i.e. ratio ≳ (1−ε/n)/(1+ε(n−1)/n) ≈ 1/(1+ε). I checked both inequalities numerically on 4,000 random instances (zero violations). This is exactly the ratio of the adversarial family in Proposition 9 (the family gives (1+ε(n−1)/n−(n−1)/M)^{-1}). Hence the true worst-case ratio of Algorithm 2 is ≈1/(1+ε), the theorem's 1−ε is weaker by the factor 1−ε² the paper reports, and Proposition 9 as framed ("the guarantee is tight up to a factor 1−ε²", abstract, Table 1, Conclusion) presents as a feature of the problem what is a slack in the proof. The conclusion's "within a factor 1−ε²" is literally true of the *analysis*, but a reader will take it as a property of the scheme.
**Evidence Anchor**: text: Proposition 9 "so the guarantee is tight up to a factor $1-\epsilon^2$"
**Why it matters**: Tightness is one of the paper's three advertised heuristic/analysis contributions; with a two-line refinement it becomes exact tightness, which is a cleaner and stronger statement (and also yields the simple corollary that ε'=ε/(1+ε)… i.e. running with ε gives (1+ε)-approximation).
**Suggestion**: Add the refinement as a strengthening of Theorem 4 (ratio ≥ 1/(1+ε)·(1−ε/n)), then Proposition 9 shows it is asymptotically tight; or state clearly that Prop. 9 concerns the proof technique and that a sharper analysis exists.
**Severity**: Minor
**Confidence**: 4 — core expertise; derivation checked numerically, not formally

### W3: Camp Fire case study: unstated evaluation protocol, uncertainty coverage that cannot test the headline lesson, and results that depend on a dominant weight
**Problem**: (i) The "retained fraction of the drawn optimum" for the nominal plan is not reproducible from the text. I re-implemented the sampler (speed U[40,90], arrival factors U[0.85,1.15]/U[0.70,1.30], delay U[0,20], 5,000 draws): the Paradise-only statistics (98.5%/97.4%), the optimal-set shares (63.7–65.2 / 17.9–18.9 / 14.2–14.6 / 2.6–2.7 against the paper's 65.3/18.4/13.9/2.3, consistent with MC error), the fully-on-time share (20.6–21.6% vs 20.7%) and the fifth percentile 2.7% all reproduce. But the mean retention of the nominal plan, reported as 26.6%, does not under the literal reading "fly to Concow, then Paradise" (≈22–23%); it does (≈26.3–26.9%) only if the crew silently skips Concow when it would be late. So the evaluation uses an unstated runtime rule (an adaptive element), which also qualifies the "brittle nominal optimum" message (a dispatcher with even a skip-if-late rule is much less brittle). (ii) The uncertainty study varies speed, hazard arrival and delay, but not the weights (2010 census populations, the weight vector that determines the answer: Paradise 26,218 exceeds the other three combined, 12,353, so "Paradise is in the optimum in 100% of the draws" is a structural consequence of feasibility plus weight dominance, not a finding), nor the great-circle-to-road factor (only partly absorbed by the speed range), nor correlation between arrival errors (independent factors understate joint-error risk for a single fire model). (iii) Probability estimates carry no Monte Carlo standard errors; "100%" is in fact not exactly 100% (I estimate P(Paradise infeasible) ≈ 3×10⁻⁴; with 5,000 draws that is zero hits with probability ≈ 0.2), so it should be reported as "≥ 99.9%". (iv) The "practical rule" drawn in §6.3 and the Conclusion ("protect the dominant site first") is extrapolated from one 4-site, brute-forceable instance whose dominant weight drives it. (v) Rounding: the deadlines d_i = d^haz + p_i/2 are non-integers; Table 7 rounds half-to-even (83.5→84, 71.5→72, 85.5→86 but 102.5→102). With integer completion times the exact test is C ≤ ⌊d⌋, so ceilings are non-conservative; no result here changes, but the convention should be stated and be floor. (vi) The paper treats the hazard timestamp for Paradise as "spot fires igniting" (not a flame front) while Concow's is fire arrival; the mixed definitions are not discussed.
**Evidence Anchor**: text: §7.6 "retains on average $26.6\%$ of the drawn optimum (fifth percentile $2.7\%$)"
**Why it matters**: The abstract advertises "a Camp Fire case study with a sensitivity analysis" as a contribution and the conclusion draws a decision-style lesson from it. The study is labelled an illustration, which mitigates it, but its claims should be reproducible and its assumptions complete.
**Suggestion**: State the plan-evaluation protocol (fixed route vs skip-if-late), give MC standard errors, add a weight-perturbation scenario (or at least report the tipping weight ratio at which Paradise-first stops being optimal) and a correlated-error scenario, state the rounding convention (floor), print the coordinates/distances used (they are only in the figure), and soften the "practical rule".
**Severity**: Major
**Confidence**: 4 — adjacent field (disaster-response modelling); numerics re-implemented from the description

### W4: Reproducibility: code, seeds and several generator parameters are not available; Lean artefact unavailable
**Problem**: Data Availability says code (and the Lean development) are "available from the corresponding author on reasonable request". No seeds, language, hardware or library versions are given; timings (Tables 4–5) are reported in ms from 5 instances per row with no spread. Several generator parameters needed to rerun §7.3 are missing: the common deadline D of the "strongly correlated" family, the deadline distribution for the random-weight family, and the value of M for the adversarial family (my reconstruction suggests M=10⁶). Table 1's K=1 shares are stated per nominal n, but K uses the post-deletion n (with nominal n=10 and ε=0.1 one has εw_max/n ≤ 1, i.e. K=1 always, yet the table gives 0.54; it only reproduces if n is the number of sites after deleting infeasible ones), which should be said.
**Evidence Anchor**: absence: Data Availability and §7 — expected seeds, language/hardware, repository or archive DOI, and the full generator parameters for each family; checked Data Availability, §7.1–7.6, captions of Tables 3–6
**Why it matters**: For a paper whose empirical component is a verification and a demonstration, replication is cheap to enable and its absence prevents anyone from checking "no violation in 4,500 runs" or the table entries; "on request" is also weak for a journal that encourages deposit.
**Suggestion**: Deposit code, seeds, and the instance generators in a public archive (Zenodo/GitHub with a DOI), list all generator parameters, hardware/software for timings, and (if kept) the Lean development.
**Severity**: Major
**Confidence**: 5 — standard reproducibility practice

### W5: Experimental design gives little independent evidence about the FPTAS; one table is an analytic restatement
**Problem**: (i) The "Adversarial" column of Table 3 (0.669, 0.771, 0.835, 0.910, 0.953) equals the closed form of Proposition 9 at n=100, M=10⁶ to three digits (I computed 0.6689, 0.7711, 0.8348, 0.9100, 0.9529). It is a deterministic instance, not an empirical "worst over instances"; the table should say so. (ii) Random and strongly-correlated families give ≥0.9993 even at ε=0.5, so they test only that the implementation runs. (iii) No competing method is compared: neither the faster published FPTAS the paper cites (Gens–Levner), nor Moore–Hodgson-type baselines, nor an off-the-shelf MIP/CP solver on the 2^n subset formulation. (iv) §7.4 compares two of the authors' own algorithms; with p ≤ 20 the exact DP is faster, as the authors note. (v) Ratios are means over 50 instances with no spread.
**Evidence Anchor**: table: Table 3 "Adversarial" column vs Proposition 9's ratio (1+ε(n−1)/n−(n−1)/M)^{-1}
**Why it matters**: The claims drawn ("the guarantee is not vacuous", cost ∝ 1/ε) are fine, but an "empirical evidence" framing overstates what the study adds beyond the theory.
**Suggestion**: Label the adversarial column as analytic; add instances near the knapsack-hard regime (deadline ≈ Σp/2, correlated w and p), report spread, and compare with at least one published scheme or a CP-SAT baseline.
**Severity**: Minor
**Confidence**: 4

### W6: Complexity-class and classification wording is stronger than what is proved
**Problem**: (a) "Polynomial exactly when dispatch times or weights are constant and weakly NP-hard otherwise" (abstract, Theorem 9, Conclusion). "Exactly" needs P≠NP; Theorem 9(b) proves NP-hardness, and the "iff" is only conditional. (b) "A complete classification" (Intro item 4, Table 1) is a classification of the eight subsets of {p,w,d} being entirely constant, an elementary corollary (the paper concedes "elementary"), and does not address, e.g., two distinct dispatch values (for which, as §2 notes, fixed-parameter results are negative). (c) "Weakly NP-hard" is used for the restrictions; this is correct, but the formal basis (NP-hard, pseudo-polynomial algorithm) should be stated in Theorem 9(b), and the Partition-based proof silently dismisses odd A (answer no). (d) Table 1 labels Theorem 5 for m≥2 as "new (proof)"; unit/equal-time jobs on identical parallel machines with weighted late-job objective (P|p_j=p|Σw_jU_j) is, to my knowledge, classical through the same slot/transversal-matroid view; I could not verify the literature here, so please check scheduling tables (Brucker; Baptiste) before claiming novelty.
**Evidence Anchor**: text: Theorem 9 "Thus an MWHED restriction is polynomial exactly when the dispatch times or the weights are constant"
**Why it matters**: These are statements about the paper's contribution and a precise reader will challenge them.
**Suggestion**: Write "polynomial if…, and NP-hard (hence not polynomial unless P=NP) otherwise"; drop "complete"; add the formal definition of weak NP-hardness in Theorem 9(b); verify and rephrase the novelty claim for m≥2.
**Severity**: Minor
**Confidence**: 4 (a–c) / 3 (d)

---

## Coverage Receipt
Not required; both lists are populated.

---

## Verified as correct (what I actually checked)

- **Lemma 1 (EDD)**: both exchange steps correct (prefix step strictly lowers completion times; adjacent swap keeps both on time since t0+p_i+p_i' ≤ d_i' < d_i). Only gap: ties (d_i = d_i') are not covered by the stated swap argument, though the same inequality works (see Minor 1).
- **Fact 1**: correct (last completion = Σp; partial sums ≤ total).
- **Theorem 2**: reduction from Partition correct (p=w=a, d=A/2; W* ≤ A/2 with equality iff partition; a_i ≤ A/2 guarantees Assumption 1); NP membership correct; "weakly NP-hard" justified jointly with Theorem 3.
- **Theorem 3**: recursion and induction correct; verified the full table f(1..4,t) of Table 2 and the backtracked optimum {1,2,4}=23 by code; O(nP).
- **Theorem 4**: algebra for K>1 correct (Σ_{S*}w' ≥ W*/K−n, W(Ŝ) ≥ W*−Kn = W*−εw_max ≥ (1−ε)W*; uses W* ≥ w_max from Assumption 1; K=1 case and boundary ε w_max/n = 1 fine); table size O(n³/ε) correct in both cases (V ≤ n w_max < n²/ε when K=1); empirical check 0 violations in 4,000 random instances; Example 5 numbers exact.
- **Theorem 5**: Step 1–5 all correct. Counting criterion (both directions, including ceil(r/m) ≤ D), Step 3 (largest k with N_B(k) ≤ N_A(k), existence of x with D_x=k+1, post-insertion bound N_A(j)+1 ≤ N_B(j) ≤ mj), Step 4 (greedy optimality, tie-breaking and strict-inequality step), Step 5 (q* argument incl. q*=n edge case). Greedy vs brute-force over all assignments for m∈{1,2,3} on 1,500 random small instances: 0 mismatches. Example 4 (22 and 26) exact.
- **Proposition 7 (naive EDD / skipping)** and **Proposition 8 (greedy repair)**: instances and ratios correct (including the tie-free comparison (2k−1)/k<2); feasibility invariant of Algorithm 3 correct.
- **Proposition 9**: instance, scaled weights (w_i<K ⇒ w_i'=0), returned set {1}, W* and the displayed ratio all correct; I recomputed the five ratios of Table 3's adversarial column.
- **Theorem 9**: (a) via Theorems 5 and Moore; (b) hardness inclusion correct; consistent with Theorem 2/3 (modulo W6).
- **§8.1 multi-vehicle hardness by blockers**: correct (a served blocker occupies a vehicle exclusively; dropping k blockers loses k(Ω+1) and gains at most Ω overall).
- **§8.3 two-site stochastic example** (6 vs 11) and the robust-feasibility corner-instance claim: correct.
- **Camp Fire**: p_i, d_i (up to rounding), arrival extrapolation, Table 9 for all five algorithms at 50 and 80 km/h, 2.6% gap, return-reading statements, all recompute; sensitivity-study percentages reproduce except the 26.6% mean (W3).
- Table 1 means/worst ratios reproduce in magnitude with the stated generator (my seeds: greedy 0.993, skip 0.936, naive 0.737 at n=10 vs reported 0.993/0.933/0.714); Tables 4–5 cell counts (n·V) and speed-up factors (74× for 256×) are arithmetically consistent.

---

## Detailed Comments

### Title & Abstract
- Abstract's "polynomial exactly when…" and "tight up to a factor 1−ε²" should be qualified (W2, W6). "Classification" in the abstract is accurate; "complete" in the introduction is generous.

### Introduction
- Clear about what is classical. The separation-of-concerns argument (deadlines produced by a hazard model) is an assumption that makes uncertainty in d_i exogenous; it is acknowledged in §8.3.

### Methodology / Research Design
- **Model**: MWHED = 1||Σw_jU_j is correct and the proof via EDD is sound. Remark 1 (arrival reading with shifted deadlines) is correct; note shifted deadlines are generally half-integers, so an explicit rounding convention (floor) is needed for integer data.
- **Algorithms**: Algorithm 1 correct (the `choice` flag is only set when strictly improving, so backtracking is valid; ties are harmless). Algorithm 2 correct; Algorithm 3 correct given the union–find caveat below; Algorithm 4 correct.
- **Union–find remark (Theorem 5, Step 5)**: "with path compression a sequence of n operations costs O(nα(n))" is only true with union by rank/size as well; with path compression alone the amortized cost is O(log n), and with the required linking direction (the lower index must stay the representative) one needs a separately stored label. The overall O(n log n) claim is unaffected (sorting dominates, and path compression alone gives O(n log n) total), so this is a presentation fix.
- **Randomization of m**: the bound is independent of m; for m ≥ n the instance is trivial; say that m can be capped at n.

### Results / Findings
- Section 7 is honest and well organized. Tables 3–6 would benefit from variance, spread and published baselines (W5).

### Discussion
- §8.1: correct, and honest about what is open (conjecture "not verified" flagged as such). The sentence about release dates should be qualified: the quoted hardness is for the number of machines being part of the input (W[2]-hard in m); for fixed m (and m=1) equal-length jobs with release dates are, as I recall from the Baptiste et al. line of work, polynomial; the statement "robust to adding vehicles but not release dates" should be restricted accordingly (please verify against the cited paper).
- §8.3: sound; the corner-instance argument is correct.

### Reproducibility
- See W4.

### Methodological Fallacies Detected
- Survivorship/selection: none. Overfitting: not applicable. Instance-generator dependence: the default generator (d uniform on [1,Σp]) yields slack-rich instances; the paper partly addresses this in §7.5 (tight deadlines) but not the knapsack-hard regime. Endogeneity/uncertainty: the case study treats d_i and weights as exact except where perturbed (W3).

---

## Questions for Authors
1. In the sensitivity study, what exactly is the policy being evaluated for the "nominal 80 km/h plan": fixed route (Concow, then Paradise) or with a skip-if-late rule? A literal fixed route gives ≈22% mean retention in my replication, not 26.6%.
2. How large must the Paradise-to-rest weight ratio be for the Paradise-first conclusion to change under your uncertainty model? Does the brittleness lesson survive if weights are perturbed or if the arrival errors are correlated?
3. Do you agree the algorithm of Theorem 4 achieves ≈1/(1+ε)? If so, would you present Proposition 9 as tightness of the (refined) guarantee?
4. Can code, seeds, and generator parameters be deposited in a public archive for the revision?

---

## Minor Issues

### Language / Grammar / Notation
- Lemma 1 proof: the swap argument assumes strict inversion d_i > d_i'; the same inequality t0+p_i+p_i' ≤ d_i' ≤ d_i covers ties, which Fact 1 and the DP relabelling use implicitly ("any order among equal deadlines"). State it.
- Theorem 4 proof: replace "$w_i\le K(w_i'+1)$" by "$w_i\ge Kw_i'$" in the conversion step (only the latter is used).
- Eq. (1): the indicator-times-value form is incorrect if the product is read literally; use cases. Same for the g-recursion (W1).
- Theorem 2: state the trivial handling of odd A (answer no) alongside a_i > A/2.
- Table 1 (results) and Table 3 caption: report that K=1 shares are for post-deletion n (W4).
- §7.2 asserts a theory-practice agreement ("K=1 for ε=0.1 at every n≥15"); it holds only for post-deletion n.
- §2: the quoted W[1]-hardness statements (Heeger–Hermelin) are used to claim "no f(k)·poly(n) algorithm for k distinct dispatch times (or criticality levels)"; please cite the exact parameterization from the paper, since the hardness there may use several distinct due dates (not verified here).

### Citation Format
- Not within my remit; Reviewer 2 will address.

### Figures and Tables
- Figure 5 (right panel) uses a dual axis where cells and ms coincide visually; this makes proportionality look better than it is. Use two panels or normalized values.
- Table 3: label the adversarial column as a single deterministic instance with the formula value.

### Layout
- None of substance.
