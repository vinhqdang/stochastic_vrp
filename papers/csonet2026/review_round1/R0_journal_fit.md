# Peer Review Report

## Manuscript Information
- **Title**: Minimum Weighted Hazard-Exposure Dispatch: Complexity, an Exact Algorithm, and an FPTAS
- **Manuscript ID**: not provided
- **Review Date**: 2026-10-05
- **Review Round**: Revised manuscript, first read by this reviewer (cold read)

---

## Reviewer Information

### Reviewer Role
Journal-Fit Reviewer (handling Associate Editor perspective, Journal of Combinatorial Optimization)

### Reviewer Identity
Associate Editor for a combinatorial-optimization / scheduling journal. Familiar with single-machine scheduling, knapsack-type approximation schemes and matroid methods. Not reviewing proofs line by line (Reviewer 1's remit).

### Review Focus
Scope fit with JCO, originality relative to the classical literature, significance for JCO readers, structural coherence (title / abstract / contributions / conclusion), and overall quality signal.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision** (borderline: the reject-for-insufficient-contribution subtype was seriously considered; see "Recommendation rationale")
- [ ] Reject

### Confidence Score
4. This is core territory for scheduling, knapsack-type approximation and matroid greedy. I did not verify the Lean 4 formalisation or re-run the authors' code. Everything else I rely on I read in the text, and I spot-checked the hand examples (brute force on the 4-site instance confirms W* = 23; the DP table, the Prop. 3 and Prop. 4 constructions, and the Table 6 / Table 5 arithmetic are consistent).

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements

| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Journal scope fit | JCO aims and scope (combinatorial optimization: complexity, algorithms, scheduling) | MEETS | text: Abstract "weak NP-hardness ... a Lawler--Moore-type pseudo-polynomial dynamic program, and a value-scaling FPTAS" | Complexity, exact DP, FPTAS and matroid greedy for a scheduling problem are squarely in scope. | The "transportation" framing is not what makes it fit; the scheduling content is. | no |
| Originality | Venue expectation of new theoretical results | PARTLY_MEETS (low end) | text: Section 1 "we do not claim a new algorithmic technique for the general problem"; table: Table 2 "Origin" column | The authors state candidly that hardness, DP and FPTAS are classical. What remains new is elementary (a corollary-level classification, a multi-vehicle extension of a classical unit-job greedy that may itself be known, two-site counterexamples, tightness of one implementation's analysis). | I could not search the full scheduling literature; novelty of Thm. 5 (m vehicles) should be checked against Brucker-style textbook results on P\|p_j=1\|sum w_jU_j. | yes |
| Significance and impact | Interest to JCO readers; consequence if claims hold | PARTLY_MEETS | text: Section 4 "The classification is elementary once Theorems 1 and 5 are available; its value is interpretive." | For readers who already know 1\|\|sum w_jU_j, there is little to learn. The "network interpretation" of the boundary is thin because the model has no network. | Pedagogical or modelling value for the disaster-response audience is real but is not a JCO-type contribution on its own. | yes |
| Structural coherence (title -> abstract -> contributions -> conclusion) | Internal consistency, no over-promising | PARTLY_MEETS | text: Title "Complexity, an Exact Algorithm, and an FPTAS"; text: Section 1 "this is the Lawler--Moore dynamic program" | Body and abstract are honest. The title and keywords ("transportation networks") still advertise the classical items and a network dimension the paper does not develop. | Easily repaired. | partly |
| Overall quality / presentation | Clarity, completeness, reproducibility | MEETS (with caveats) | table: Tables 3-7; text: Data Availability "available from the corresponding author on reasonable request" | Clean, verified, carefully caveated. Reproducibility rests on code and a Lean formalisation that are not supplied. | Did not inspect code or formalisation. | no (repairable) |
| Related-work coverage | Positioning among CO communities | PARTLY_MEETS | absence: Section 2 -- expected firefighter-problem and scheduling-with-rejection literatures; checked Section 2 and bibliography | Wildfire OR, real-time systems and matroid items are cited at length, but the closest CO literature on "race a spreading hazard" (the Firefighter Problem) and on scheduling with rejection is absent. | Suggestions are from my own knowledge; authors should verify relevance. | partly |

Judgements are not totalled or averaged; the recommendation follows from the unresolved decision-bearing rows (Originality, Significance).

### Summary Assessment
The paper formalises a single-depot dispatch problem in which a vehicle makes independent round trips to sites with hazard-arrival deadlines and criticality weights, and the objective is the total weight of sites reached in time. It proves (Section 3) that this problem is exactly the classical 1||sum w_jU_j, so weak NP-hardness (via Partition, equal deadlines), the Lawler-Moore pseudo-polynomial DP and a value-scaling FPTAS are inherited, and it re-derives these with full proofs. Its claimed additions are a classification by which of (p, w, d) are constant (Thm. 5), an O(n log n) matroid-greedy algorithm for equal dispatch times with m identical vehicles (Thm. 4), unbounded-ratio examples for simple heuristics and tightness of one FPTAS implementation (Props. 1-3), a verified numerical study, and a four-site Camp Fire illustration.

The manuscript is unusually honest about what is classical and the proofs I sampled are correct and clearly written. The issue for JCO is contribution size. Everything that is hard is classical, and everything that is new is elementary. The "transportation" framing contributes a vocabulary and a case study, not a result. As submitted, the paper reads as a well-executed expository note plus modest extensions. I recommend major revision: either (i) establish genuinely new theory (for example for a version where the order-independence assumption breaks) or (ii) retarget and retitle the paper so that its actual contribution (a verified, accessible treatment plus a few corollaries) is what is being judged, and fix the novelty and interpretation overstatements listed below.

### Recommendation rationale
Unresolved decision-bearing criteria are Originality and Significance (W1, W2). Both are repairable only by adding substance (a non-classical result) or by reframing the claim; neither is repairable by editing alone. The honest disclosure, correct proofs and verified experiments keep this from a flat reject. The remaining weaknesses (W3-W6) are repairable by rewriting.

---

## Strengths

### S1: Transparent, accurate attribution of what is classical
The abstract, Section 1 and Table 2 state that hardness, the DP and the FPTAS are classical and give the original sources. This is exemplary and protects the paper from a novelty-misrepresentation objection on the core results.
**Evidence Anchor**: text: Section 1 "Consequently the basic complexity picture ... is inherited from scheduling theory, and we do not claim a new algorithmic technique"

### S2: Correct, self-contained and readable proofs with worked examples
Lemma 1, the Partition reduction, the DP recursion and the five-step matroid argument (counting criterion, augmentation, greedy optimality, union-find implementation) are complete. The running example is consistent across Examples 1-4 and Appendix A (I confirmed W* = 23 by brute force and re-derived Table 3's rows).
**Evidence Anchor**: table: Table 3 -- row f(4,t) ends at 23 at t = 9 and is "--" at t = 10 because d_4 = 9

### S3: Honest diagnosis of a trap in FPTAS experiments
The paper notes that with weights <= 100 the scaling factor is 1, so a ratio of 1.000 is "guaranteed, not informative", then designs an active-scaling regime and an adversarial family (Prop. 3) where the guarantee is visibly approached.
**Evidence Anchor**: figure: Figure 5 (epsilon_sensitivity.png) left panel -- adversarial-family curve stays above and approaches the 1-epsilon line; completion step sits at 1.0

### S4: Candid limitations and discussion of what breaks under extensions
Section 6 explains precisely which assumption (order-independent p_i, deterministic parameters, no release dates) each extension breaks, and the Camp Fire study is explicitly framed as an illustration with disclosed assumptions (e.g., the arrival versus return reading, Remark 1).
**Evidence Anchor**: text: Section 5.6 "This single instance cannot substitute for the synthetic study's statistical evidence"

### S5: Useful, correct observation on arrival vs return readings of the deadline
Remark 1 reduces the arrival reading to the same model via d_i := d_i^haz + (p_i - a_i), and the case study shows the answer changes. This is a small but practically valuable modelling point.
**Evidence Anchor**: text: Remark 1 "the arrival reading is an instance of Definition 1 with shifted deadlines"

---

## Weaknesses

### W1: The new theoretical content is elementary, and the paper's headline items are classical
**Problem**: By the authors' own account, contributions 1-3 (hardness, Lawler-Moore DP, FPTAS) are classical. Of the claimed additions: Theorem 5 enumerates the eight subsets of {p,w,d}; part (a) is Thm. 4 with m = 1 plus Moore's algorithm, part (b) is Thm. 1. The authors themselves call it "elementary". Propositions 1-2 are two-site examples (the same failure as ratio-greedy on knapsack). Proposition 3 shows tightness of one particular implementation, which drops zero-scaled sites and is repaired by a one-line completion step the paper itself supplies; it says nothing about the problem's approximability. The m-vehicle result (Thm. 4) reduces, via D_i = floor(d_i/p), to unit-time jobs on identical machines with deadlines and weights, a transversal-matroid structure that I believe appears in standard scheduling texts; Table 2 labels it "new (proof)", which is a weak claim at best. Table 2 also labels Props. 1-3 and the classification "new", which overstates them.
**Evidence Anchor**: text: Section 4 "The classification is elementary once Theorems 1 and 4 are available; its value is interpretive."
**Why it matters**: JCO's bar is new results in combinatorial optimization. A reader who knows 1||sum w_jU_j learns essentially nothing new; the paper's value is expository.
**Suggestion**: (a) Check Thm. 4 against the unit-job / parallel-machine literature (e.g., Brucker, Scheduling Algorithms; Lawler's matroid-scheduling results) and either cite it as known or isolate what is genuinely new. (b) Add at least one result that is not a corollary of existing theory. Natural candidates, all already gestured at in Section 6: a hardness / approximation result for the stochastic (scenario) version where Lemma 1 fails, strong NP-hardness or inapproximability for sequence-dependent travel times (the actual routing version), parameterised complexity in the number of distinct (p, w) types beyond citing Heeger-Hermelin, or a competitive/approximation result for a simple heuristic. (c) Re-label Table 2 so "new" is applied only to items that survive (a).
**Severity**: Major
**Confidence**: 4 -- core expertise: scheduling and approximation-scheme literature; novelty of Thm. 4 not exhaustively searched

### W2: Significance claim ("what the transportation framing adds") is not supported by the results
**Problem**: The paper argues that the framing adds a classification with a "network interpretation" (if every site is equally far from the depot, the problem is easy). But the model has no network: p_i is an arbitrary input, and the tractability boundary is purely arithmetic (some vector exactly constant). Exact constancy is a knife-edge; the authors' own related work notes the problem is W[1]-hard in the number of distinct dispatch times (Heeger-Hermelin), so near-equal distances give no tractability guarantee. The "two independent sources of heterogeneity" interpretation is also not what Theorem 1 shows: its reduction sets w_i = p_i, i.e., a single perfectly correlated heterogeneity (it is subset-sum), so "independent" heterogeneity in time and criticality is not needed for hardness. Moreover, the star topology (return to depot after every site) is exactly what removes routing; the motivating examples (a crew defending structures in sequence, a repair crew visiting substations) would normally chain sites, which is the strongly NP-hard deadline-TSP-type problem that a transportation reader would care about, and is left aside.
**Evidence Anchor**: text: Section 4 "Hardness needs two independent sources of heterogeneity, in the dispatch time and in the criticality, which compete for the shared time budget"
**Why it matters**: The paper's differentiator from a relabelling is this interpretive claim; as phrased it is partly contradicted by the paper's own Theorem 1 (w = p) and Section 2 (W[1]-hardness).
**Suggestion**: Reword to what is proved ("hardness needs non-constant p and w; with w = p one heterogeneous vector already suffices"). Either quantify robustness of the easy case (e.g., an approximation guarantee or a bound as p_i varies within a factor 1+delta; this would be a real network-flavoured result) or drop the network-interpretation language. Add a short, explicit justification for the return-to-depot assumption versus chained visits, and state what is known for the chained (sequence-dependent) version.
**Severity**: Major
**Confidence**: 4 -- core expertise: complexity classification and interpretation of reductions

### W3: Related-work coverage misses the closest combinatorial-optimization literatures
**Problem**: Section 2 is long on wildfire OR and real-time systems (Liu-Layland, self-suspending tasks) but does not engage with (i) the Firefighter Problem and its variants, the standard combinatorial model of choosing what to protect before a hazard spreads (hardness and approximability are well studied), (ii) scheduling with rejection (the weighted-late-jobs problem viewed as accept/reject under a deadline), and (iii) deadline-constrained routing (TSP with deadlines, orienteering with time windows) that are the natural routing generalisations. The real-time-systems paragraph is lengthy relative to its relevance (the paper itself concedes the objectives differ fundamentally). I flag these from my own knowledge; authors should verify and select.
**Evidence Anchor**: absence: Section 2 and bibliography -- expected Firefighter Problem, scheduling-with-rejection and deadline-TSP / orienteering references; checked Section 2 ("Positioning" paragraph) and the full reference list
**Why it matters**: JCO readers will immediately ask how MWHED relates to these; positioning only against OR wildfire papers and RTS undersells and misplaces the work.
**Suggestion**: Replace part of the real-time-systems discussion with a compact comparison to the above literatures.
**Severity**: Major
**Confidence**: 3 -- adjacent expertise; coverage judged from memory of the field, not an exhaustive search

### W4: Title, keywords and contribution list do not match the actual contribution
**Problem**: The title leads with "Complexity, an Exact Algorithm, and an FPTAS", all three being classical by the paper's own account; the keyword "transportation networks" is not reflected in any network result; the Introduction presents classical items as contributions 1-3.
**Evidence Anchor**: text: Title "Complexity, an Exact Algorithm, and an FPTAS"
**Why it matters**: Over-promising in the most visible places invites exactly the novelty objection the body text works to avoid.
**Suggestion**: Retitle toward what is actually established (for example, a classification / structural-results framing) and move classical items to a "Preliminaries / classical results revisited" role in the contribution list.
**Severity**: Minor
**Confidence**: 5 -- direct textual inconsistency

### W5: Case study has limited construct validity and adds little evidence
**Problem**: Four sites, one of which dominates by two orders of magnitude (population 26,218 versus 710 / 333 / 11,310), so the "optimum" is Paradise in essentially every scenario and the result is nearly determined by the weights. Weights are census populations "protected" by a single crew arriving before the fire, which is not a credible protection model for towns of 10,000+ people; Magalia and Yankee Hill arrival times are extrapolated isotropically; and the sensitivity distributions are ad hoc. The study demonstrates the arrival/return distinction and brittleness of a nominal optimum, but these are properties of any such model, and the brittleness is attributed to an effect (a small site ahead of a dominant one) visible by inspection. The sensitivity paragraph also does not reconcile "Paradise is in the optimum in 100% of draws" with the statement that the extreme corner makes Paradise infeasible (my simulation of the stated distributions gives about a 3e-4 chance per draw, i.e., about 1.5 expected in 5,000, so the two statements are compatible but the reader is left to wonder).
**Evidence Anchor**: table: Table 6 (campfire-results) -- 26,218 at 50 km/h and 26,928 at 80 km/h versus 710 / 1,043 for the deadline-order rules
**Why it matters**: A real-event section lends transportation credibility, but its evidential weight is small, and JCO readers may see it as decoration.
**Suggestion**: Either shorten it and present it as a worked illustration of Remark 1, or strengthen it (more sites, e.g. sub-community or structure-level weights; multiple fires; documented arrival times only) and report the 100% / corner discrepancy explicitly.
**Severity**: Minor
**Confidence**: 3 -- adjacent expertise for case-study realism

### W6: Reproducibility and verification claims rest on unavailable artifacts
**Problem**: The Data Availability statement says code and a Lean 4 formalisation of the main results ("no unproved steps") are available "on reasonable request". An unverifiable machine-checking claim in a journal paper carries little weight, and code on request is below current Springer Nature expectations for computational claims (Tables 3-8 are the paper's evidence base).
**Evidence Anchor**: text: Data Availability "have also been machine-checked in the Lean 4 proof assistant ... available from the corresponding author on reasonable request"
**Why it matters**: Either claim, if left unsupported, may be read as unsubstantiated; if supported, the Lean artifact is a genuine asset.
**Suggestion**: Deposit code and the formalisation in a public repository with a DOI and cite it, or remove the Lean claim. State which statements the formalisation covers (the text already says it excludes running-time claims).
**Severity**: Minor
**Confidence**: 4 -- journal-policy knowledge

### W7: Unqualified "polynomial exactly when" statement
**Problem**: The abstract and Thm. 5 say the problem is "polynomial exactly when dispatch times or weights are constant and weakly NP-hard otherwise". The "only if" direction is conditional on P != NP; the proof shows NP-hardness for T within {d}, not unconditional non-polynomiality. Also, the statement concerns exact constancy only.
**Evidence Anchor**: text: Abstract "the problem is polynomial exactly when dispatch times or weights are constant and weakly NP-hard otherwise"
**Why it matters**: A precision issue in the headline new theorem.
**Suggestion**: Say "polynomial-time solvable if ... and NP-hard (hence not polynomial unless P = NP) otherwise" in the abstract, theorem and conclusion.
**Severity**: Minor
**Confidence**: 5 -- direct reading of Thm. 5 and its proof

---

## Detailed Comments

### Journal Fit
Topic and methods are within JCO scope (scheduling, complexity, approximation schemes, matroid greedy). Length and style are appropriate and the exposition suits the JCO readership. The fit concern is not scope but contribution level (see Originality). The long wildfire-OR and real-time-systems related-work paragraphs target a transportation audience rather than the combinatorial-optimization readership; if retained for JCO they should be compressed in favour of the CO literatures in W3. The manuscript uses the Springer Nature `sn-jnl` class, consistent with JCO. The Use-of-LLM statement is present.

### Originality
New: (i) corollary-level classification by homogeneity; (ii) m-identical-vehicle extension of the equal-time matroid greedy with a self-contained proof (possibly known, W1); (iii) trivial worst-case examples for three heuristics; (iv) tightness of one FPTAS implementation; (v) an arrival-versus-return observation; (vi) a verified experimental comparison. Classical: hardness, Lawler-Moore DP, value-scaling FPTAS, Moore's algorithm, equal-time unit-job greedy. Source of originality is new combination / reframing, not new method or theory.

### Significance
Low-to-moderate for JCO readers. Useful for a practitioner entering the disaster-response literature, who would otherwise not connect dispatch ordering to 1||sum w_jU_j. If a non-classical result were added (W1), significance would rise sharply, since the discussion in Section 6 already identifies where the interesting problems begin (scenario-dependent deadlines, sequence-dependent travel).

### Structural Coherence
Title, abstract, Introduction and Conclusion are mutually consistent on the facts, and the Conclusion is appropriately restrained. The mismatch is between how the contribution is packaged (title / keywords / "network interpretation") and what is proved (W2, W4). The Introduction is long and repeats the "ordering decision isolated from forecasting" argument; the paper would benefit from cutting roughly a third of Sections 1 and 2. Section 6's expected-deadline counterexample (6 versus 11) is correct and illustrative but is not a result about hardness or approximability.

### Title & Abstract
The abstract is accurate and unusually self-aware but dense (a single long paragraph with seven distinct claims). The phrase "polynomial exactly when" needs the P != NP qualifier (W7). The title overstates (W4).

### Conclusion
Alignment with the body is good. The two "lessons beyond the theorems" (state the deadline reading; nominal optima are brittle) are sensible but rest on a single four-site instance. Future directions are specific and credible; they point to where the paper's real research content should be.

---

## Questions for Authors

1. Is Theorem 4 (m identical vehicles, equal dispatch time) known in the scheduling literature as P|p_j = p|sum w_jU_j (with common deadline-scaling to unit jobs)? If so, what remains new beyond the self-contained proof? If not, please point to the closest known statement.
2. Can the tractability of the equal-dispatch-time case be made robust, for instance an approximation ratio or exact algorithm when the p_i lie within a factor (1 + delta) of each other? Without something of this kind, the "network interpretation" of the boundary is hard to sustain given the W[1]-hardness result you cite.
3. Which result in the paper would you identify as not a direct consequence of existing theory, and would you be willing to restructure the paper (title, contribution list) around it?
4. What is known about the sequence-dependent version (chained visits with travel times between sites)? Does Lemma 1 / the DP have any analogue, and is strong NP-hardness immediate from deadline-TSP results?
5. Will the code and the Lean development be deposited in a public repository, and which statements does the formalisation cover?

---

## Minor Issues

### Language / Grammar
- Section 1 and Section 2 are long and repetitive; several paragraphs (the "separation of concerns" argument, the Liu-Layland comparison) repeat points already made.
- The Introduction's sentence "Together with the equal-cost result below this turns the informal question ... into a precise statement" is stronger than the corollary-level content warrants.

### Citation Format
- Several reference entries are preprints or incomplete (e.g., arXiv:2603.29865, arXiv:2606.01309, SIAM J. Comput. 2025 with no volume or pages); confirm final bibliographic data before publication.
- Both Table 2's "Moore" and "classical" origin labels would be clearer with citations in the table.

### Figures and Tables
- Table 4 (scaling) and Table 5 (cost): clear. Table 3's description "n in {10,15,20,...}" versus the columns shown (10, 20, 50, ...) should be reconciled, and the text should say that n in K = max(1, eps w_max / n) is the post-deletion number of sites; this explains why the K = 1 share at n = 10 is 0.54 for eps = 0.1 even though w_max <= 100.
- Figure 6 (campfire_map.png): the dashed arrow's head sits between the Concow and Yankee Hill markers and is ambiguous; extend it to the Concow marker. The legend says "in the optimum" but the arrow for the second site at 80 km/h is not distinguishable by order of service.
- Figure 5 (epsilon_sensitivity.png): the right panel uses two y-axes whose curves coincide; consider a single normalised axis.

### Layout
- Table 2 (summary) is too narrow for its five columns and wraps awkwardly; consider a landscape layout or removing the "Runtime" column for non-algorithmic rows.
- Algorithms 1-3 (exact DP, FPTAS, greedy repair) differ little in structure; consider merging Algorithms 1 and 2 to save space.
