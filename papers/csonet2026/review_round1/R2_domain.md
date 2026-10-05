# Peer Review Report

## Manuscript Information
- **Title**: Minimum Weighted Hazard-Exposure Dispatch: Complexity, an Exact Algorithm, and an FPTAS (revised manuscript, `papers/csonet2026/submission/main.tex`)
- **Manuscript ID**: not given
- **Review Date**: 2026-10-05
- **Review Round**: simulated panel review, FULL mode, first cold read of the revised manuscript

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
Domain expert in single-machine scheduling with deadlines and due dates, the 1||Σw_jU_j literature (Moore–Hodgson, Lawler–Moore, Sahni, Gens–Levner, Heeger–Hermelin, Hermelin et al.), knapsack-type approximation schemes, unit-job matroid scheduling, and the disaster-response/evacuation scheduling literature.

### Review Focus
Novelty and positioning. Which claimed contributions are new rather than textbook, whether the attributions are accurate, whether the bibliography is sound, whether the classification theorem is correct and new, and whether the transportation framing justifies a combinatorial-optimization journal.

Method note. I read the whole manuscript. I re-derived the proofs of Lemma 1, Theorem 2 (DP), Theorem 3 (FPTAS), Theorem 4 (matroid greedy, Steps 1-5), Propositions 1-2, the Section 6.1 hardness-for-fixed-m argument and the stochastic counter-example. I also re-solved the Camp Fire instance by hand at 50 and 80 km/h. All of these check out. My concerns are not about correctness. I spot-checked bibliography entries against the web in this session. Entries I could confirm are marked "confirmed (this session)". Everything I could not confirm is marked `[UNVERIFIED]` or "verify".

---

## Overall Assessment

### Recommendation
- [x] **Major Revision** (with the caveat below)

Caveat: in its present framing, a research-journal acceptance in Journal of Combinatorial Optimization (JOCO) is not justified on novelty grounds. The revision must either (i) add at least one new theorem with research-level content, or (ii) be honestly repositioned as an application-oriented/expository paper whose claims are limited to what is actually new, and be re-judged against that scope. If neither happens, my recommendation is Reject.

### Confidence Score
4. Core expertise in the scheduling literature. Some references below are from memory and tagged accordingly.

### Summary Assessment
The authors state openly that MWHED is exactly 1||Σw_jU_j. They present weak NP-hardness, the Lawler–Moore DP and a value-scaling FPTS as classical. That honesty is welcome, and the proofs I checked are correct. The consequence is that the paper's own account of its contribution is thin.

The remaining "new" items are not new, or are elementary. The classification by homogeneity (Thm 7) is a three-line corollary of Karp, Moore and the classical equal-processing-time case. It appears in standard complexity tables, and its "iff" boundary is neither robust nor the right one. The extension of the equal-cost greedy to m identical vehicles is the textbook unit-job matroid over m·n slots. The heuristic ratio bounds (Props 1-2) are folklore two-job examples. The FPTAS tightness (Prop 3) is a statement about one rounding analysis of an algorithm that is dominated by known schemes.

The transportation framing is thin. Routing structure is deliberately abstracted away, and the closest transportation literature (deadline-TSP, orienteering, profitable-tour problems, scheduling with rejection) is not discussed. The related-work section is long but padded with tangential citations. It also omits core references: Karp 1972, the equal-processing-time scheduling literature (Baptiste), and the 1||Σp_jU_j fine-grained line. One statement about release dates is incorrect for m=1.

---

## Strengths

### S1: Candid and accurate self-positioning of the classical core
The abstract, Section 1 and Section 2 state that MWHED = 1||Σw_jU_j and that Thm 1-3 are classical or re-derivations. Table 2 labels origins.
**Evidence Anchor**: text: Section 1, "we do not claim a new algorithmic technique for the general problem"

### S2: Correct, self-contained proofs, including a complete matroid argument for the m-vehicle case
Steps 2-4 of Theorem 4 (counting criterion, augmentation via the largest k with N_B(k) ≤ N_A(k), greedy optimality) are correct. The union–find implementation argument (Step 5) is correct. The Section 6.1 blocker construction for fixed m is correct, and its parenthetical on idle vehicles is right.
**Evidence Anchor**: text: Section 4.4, Thm 4 Steps 2-4, "N_S(k)\le mk for all k"

### S3: Carefully verified numerics and an explicit account of when the FPTAS is vacuous
Table 5 explains that K=1 for weights ≤ 100. The adversarial family is built so that the rounding is active. I re-solved the Camp Fire instance and the numbers (26,218 / 26,928; greedy-repair path; EDD-with-skipping 1,043) are correct.
**Evidence Anchor**: table: Table 8 (tab:scaling), adversarial column vs 1−ε

### S4: Honest treatment of the deadline reading (Remark 1) and of model limitations in Section 6
The remark that arrival and return readings give different answers is a useful practical observation. The Section 6 discussion states what breaks under a relocating depot or uncertain parameters.
**Evidence Anchor**: text: Remark 1, "The two readings can give different answers on the same data"

---

## Weaknesses

### W1: No new theorem with research-level content in the main results
**Problem**: Every statement that carries mathematical weight is classical. NP-hardness is Karp 1972. The DP is Lawler–Moore 1969. The FPTAS is Sahni 1976 and Gens–Levner 1981. The equal-p case is unit-job sequencing with profits. The paper concedes this. What the paper calls its own contributions (Thm 7, m-vehicle Thm 4, Props 1-3) are each elementary (see W2, W3, W7). The experiments verify classical algorithms on synthetic data. The Camp Fire illustration has four sites and is solved by inspection.
**Evidence Anchor**: text: Section 1, "the basic complexity picture ... is inherited from scheduling theory, and we do not claim a new algorithmic technique"
**Why it matters**: JOCO requires original results. As written, the paper reads as an expository or application note. A reviewer for a theory venue would see the "second kind" of contribution as a relabeling plus bookkeeping.
**Suggestion**: Either (i) add a result a scheduling theorist would not already know. Candidates: an FPTAS or approximation for MWHED-m with m part of the input (compare multiple-knapsack PTAS results), a genuinely new structural case (e.g. a bounded-ratio or agreeable regime, or p-d correlation induced by the arrival reading of Remark 1), or a rigorous stochastic/robust variant with complexity and approximation results. Or (ii) re-title and re-scope the paper (drop "Complexity, an Exact Algorithm, and an FPTAS" from the title) as an application-oriented study whose novelty claims are limited to the transportation modeling, the deadline-reading analysis, and the case study.
**Severity**: Critical
**Confidence**: 4 — core expertise: single-machine scheduling theory

### W2: The classification theorem is a restatement of standard complexity tables, and its "exact" boundary is not the right one
**Problem**: Thm 7 says: for T ⊆ {p,w,d}, MWHED restricted to constant vectors in T is polynomial iff p ∈ T or w ∈ T. Part (a) is Moore 1968 (w constant) plus the classical equal-processing-time case (p constant). Part (b) is the common-due-date knapsack hardness (Karp 1972, Lawler–Moore 1969). In the standard scheduling classification (Graham et al. notation, Lenstra–Rinnooy Kan–Brucker 1977, Brucker–Knust complexity tables), these are three table entries: 1||ΣU_j, 1|p_j=p|Σw_jU_j and 1|d_j=d|Σw_jU_j. The paper itself calls the proof "elementary" and the value "interpretive".

Further:
(i) The k=1 case of "number of distinct p / w" is already stated in the literature. Heeger and Hermelin's introduction records that the problem is polynomial when p_# or w_# is bounded by a constant, attributes the w_#=1 and p_#=1 cases to Moore and to Peha [verify exact Peha reference], and records FPT results in the due-date count d_#. (Retrieved this session.)
(ii) Constancy is an unstable boundary. The manuscript's own related-work paragraph notes W[1]-hardness in the number of distinct p or w. The boundary is polynomial at k=1, XP at fixed k, W[1]-hard in k. A "classification" that jumps at k=1 says little about practical dispatch networks.
(iii) The structural boundary is not "constant". The paper does not mention that the problem with agreeable weights (p_i < p_j ⇒ w_i ≥ w_j) is polynomially solvable by Lawler's classical algorithm [Lawler 1976; verify bound and statement]. That condition contains both constant-p and constant-w as special cases and is the more natural dichotomy.
(iv) "Polynomial exactly when" is stated unconditionally in the theorem and abstract. It needs "unless P=NP". Also, T ⊆ {d} gives weak NP-hardness but the paper does not say what happens for restrictions such as w_j = p_j (hard, by Thm 1) vs. bounded p_max/p_min.
(v) The "network interpretation" ("every site equally far from the depot") is just the relabeling of p_i = const. No network structure enters the model.

**Evidence Anchor**: text: Section 4.5, "The classification is elementary once Theorems 1 and 4 are available; its value is interpretive."
**Why it matters**: This is the headline "new" contribution (contribution 4, abstract, conclusion). Table 2 labels it "new framing", which is more honest than "new", but the abstract says "the framing adds a classification".
**Suggestion**: Present it as a remark, not a theorem. If kept as a theorem, replace it with a finer dichotomy: agreeable weights (Lawler), bounded distinct values (n^{O(k)}, W[1]-hardness from Heeger–Hermelin), bounded p_max (Õ(n + p_max^3)-type results in Hermelin–Molter–Shabtay [verify exact bound]), and number of distinct deadlines. State all results conditional on P≠NP (or FPT≠W[1]).
**Severity**: Major
**Confidence**: 4 — core expertise; Lawler agreeable result from memory, tagged verify

### W3: The m-vehicle extension is labeled "new" but is the classical unit-job matroid on m·n slots, and the equal-p case has well-known treatments that are not cited
**Problem**: With p_i = p, all completion times are multiples of p, and deadlines become ⌊d_i/p⌋. MWHED-m reduces to P|p_j=1|Σw_jU_j, whose feasible sets are a transversal matroid (a bipartite graph between jobs and m copies of each time slot). Greedy plus a union–find slot assignment is the textbook algorithm. The classical scheduling-matroid treatments I would expect cited are: Edmonds 1971 (greedy on matroids), Lawler 1976 (Combinatorial Optimization: Networks and Matroids, unit-time tasks with deadlines and penalties), and Cormen–Leiserson–Rivest–Stein §16.5, plus Gabow–Tarjan 1985 for the linear-time union–find on the slot path [all verify details]. On equal processing times specifically: Baptiste 1999 (J. Scheduling 2:245–252, confirmed this session) gives polynomial algorithms for 1|r_j,p_j=p|Σw_jU_j. Baptiste–Brucker–Knust–Timkovsky, "Ten notes on equal-processing-time scheduling" (4OR 2004) is a survey of exactly this class [UNVERIFIED venue details]. Table 2 labels "Equal cost, m≥2, identical vehicles" as "new (proof)". The proof presentation is self-contained, but the result is not new. Citing Korte–Vygen for the union–find cost is also weak. Also, path compression alone gives the claimed O(nα(n)) only with union by rank; the O(n log n) total is unaffected.
**Evidence Anchor**: table: Table 2 (tab:summary), row "Equal cost, m≥2 identical vehicles ... new (proof)"
**Why it matters**: One of two claimed "kind two" contributions is an unattributed classical result.
**Suggestion**: Relabel as classical (restated for completeness). Cite the matroid-scheduling and equal-processing-time sources above. Correct the union–find attribution (Tarjan; Gabow–Tarjan).
**Severity**: Major
**Confidence**: 4 — core expertise

### W4: Misattribution and missing seminal references on the classical core
**Problem**:
(a) The NP-hardness of exactly this problem (job sequencing with deadlines and penalties) is in Karp's original 21-problem list. Heeger–Hermelin's introduction says 1||Σw_jU_j was "included in Karp's famous initial list of 21 NP-hard problems" (retrieved this session). The paper credits only Garey–Johnson (for Partition) and does not cite Karp 1972.
(b) The paper's reduction (p_i=w_i=a_i, common deadline A/2) is precisely the subset-sum hardness of 1|d_j=d|Σp_jU_j. That case, w_j = p_j, has its own recent fine-grained line that is entirely missing: Bringmann–Fischer–Hermelin–Shabtay–Wellnitz (tardy processing time, ICALP 2020/Algorithmica), Klein–Polak–Rohwedder (SODA 2023), Fischer–Wennmann (ICALP 2024) [all UNVERIFIED venues]. The Hermelin–Karhi–Pinedo–Shabtay 2021 Annals of OR paper on parameterized algorithms for 1||Σw_jU_j is also missing [UNVERIFIED].
(c) Missing knapsack-FPTAS context for a JOCO audience: Lawler 1979 (Math. OR), Kellerer–Pferschy 1999 (J. Comb. Optim. 3:59–71) and 2004 (J. Comb. Optim. 8:5–11), and the Kellerer–Pferschy–Pisinger monograph [all verify].
(d) The multi-vehicle common-deadline case is the multiple knapsack problem (identical capacities); the Caprara–Kellerer–Pferschy and Chekuri–Khanna PTAS literature is directly relevant to the paper's open question about MWHED-m approximability [verify].
(e) The paper's open conjecture that value-scaling extends to an m-dimensional state: Woeginger's benevolent-DP framework (INFORMS J. Comput. 2000) may already settle FPTAS existence for fixed m [UNVERIFIED applicability; check].
(f) Graham–Lawler–Lenstra–Rinnooy Kan 1979 is not cited for the three-field notation used in Section 2.
(g) The hardness theorem (Thm 1) deserves a sentence acknowledging it is Karp's reduction restricted to w = p.
**Evidence Anchor**: absence: Bibliography (30 entries) — expected Karp 1972 and the fine-grained 1||Σp_jU_j papers; checked reference list, Section 2, Section 4.1
**Why it matters**: Attribution of the core result is the main thing a scheduling reviewer will check. Omitting the original hardness source for a problem whose hardness is the lead claim is a visible gap.
**Suggestion**: Add Karp 1972 [R. M. Karp, "Reducibility among combinatorial problems", in Complexity of Computer Computations, Plenum 1972; the inclusion of job sequencing with penalties is confirmed this session via Heeger–Hermelin]. Add the other references above after verifying them.
**Severity**: Major
**Confidence**: 4 — core expertise; individual references (b)-(e) from memory

### W5: The statement that "release dates break" equal-cost tractability is wrong or misleading for a single vehicle
**Problem**: Intro contribution 5 and Section 6.1 say equal-cost tractability "is therefore robust to adding vehicles but not to adding release dates". For m=1, 1|r_j,p_j=p|Σw_jU_j is polynomial (Baptiste 1999, O(n^7); J. Scheduling 2:245–252, existence and title confirmed this session). The cited hardness (Heeger–Molter, STACS 2025) is for the unweighted problem on parallel machines. Its abstract (retrieved this session) states that P|r_j,p_j=p|ΣU_j is NP-hard and W[2]-hard in m, and that the weighted problem is in XP for the number of machines. So release dates break tractability only for multiple vehicles (m part of the input). The same abstract also notes the weighted problem is XP in p and FPT in (m,p). The paper should also say that for fixed m the problem is polynomial, which is not what the sentence conveys.
**Evidence Anchor**: text: Section 6.1, "The equal-cost tractability is therefore robust to adding vehicles but not to adding release dates"
**Why it matters**: A stated limit of one of the contributions is inaccurate. A reader would infer that single-vehicle equal-cost with release dates is hard.
**Suggestion**: Rewrite: "with release dates, the single-vehicle case remains polynomial (Baptiste 1999); the multi-vehicle case becomes NP-hard even unweighted (Heeger–Molter 2025), while remaining XP in m for the weighted problem".
**Severity**: Major
**Confidence**: 4 — Baptiste and Heeger–Molter claims confirmed against abstracts this session; the O(n^7) bound is from memory

### W6: The "transportation framing" is a relabeling, and the closest transportation and disaster-scheduling literature is missing
**Problem**: The model abstracts away routing. Each site is an independent round trip from the depot with fixed p_i. The paper says this is the key feature that places it in scheduling rather than routing. The consequences:
(i) A crew visiting adjacent threatened sites in one tour is excluded. That setting is deadline-TSP / orienteering with deadlines and time windows, which is the natural transportation problem (Tsitsiklis 1992; Bansal–Blum–Chawla–Meyerson STOC 2004; Chekuri–Korula–Pál; Feillet–Dejax–Gendreau 2005 on TSP with profits; Vansteenwegen et al. 2011 orienteering survey) [all UNVERIFIED venues]. None is cited. The Camp Fire instance itself treats Concow→Paradise as two depot round trips, which contradicts how a crew would actually move.
(ii) The closest model-level neighbors are absent: scheduling with rejection / order acceptance (Shabtay–Gaspar–Kaspi, J. Scheduling 16:3–28, 2013, confirmed this session; Slotnick 2011 [verify]); firefighting scheduling with deteriorating jobs (Rachaniotis and Pappis, Canadian J. Forest Research 36(3):652–658, 2006, confirmed this session); and wildfire resource-allocation models (Donovan and Rideout 2003 [verify]). The paper's wildfire-OR discussion lists only 2023-2026 papers. It claims "no prior work in this literature establishes the knapsack-equivalence hardness result"; Rachaniotis–Pappis-type scheduling models were never compared.
(iii) Under the arrival reading of Remark 1, d_i = d_i^haz + p_i/2 couples d_i to p_i. This undercuts the headline interpretation that heterogeneity in deadlines is "the part produced by the hazard model" and plays no role. The classification interpretation ignores the p-d correlation that the paper's own application induces.
(iv) The tractable cases (all sites equidistant, or all equally critical) occupy a measure-zero corner of any real dispatch instance.
**Evidence Anchor**: text: Section 2, "MWHED instead abstracts the hazard's spread into a per-site deadline"
**Why it matters**: The paper claims the transportation interpretation of the tractability boundary as part of its contribution. As framed, the added value for a combinatorial-optimization journal is the problem name and the notation.
**Suggestion**: Position MWHED explicitly as the "no-routing" lower end of the deadline-routing family, and state what is gained (full complexity picture) and lost (tour structure). Cite the orienteering/deadline-TSP and scheduling-with-rejection literatures. If a transportation contribution is claimed, give a result that depends on network structure, for example a metric depot–site setting where p_i and d_i are linked by geometry (the correlation from W6(iii)).
**Severity**: Major
**Confidence**: 4 — adjacent field for the routing literature (3); core for scheduling with rejection (5)

### W7: The related-work section is padded with tangential citations, and several recent references could not be confirmed
**Problem**: Two long paragraphs compare EDF for periodic and self-suspending real-time tasks (Liu–Layland 1973, Günzel et al. 2022, Wang et al. 2025). The paper itself says the objectives "differ fundamentally", so these add no positioning value. The relevant real-time connection (overload scheduling, maximizing the value or weighted count of deadline-meeting firm tasks) is not cited. The matroid paragraph cites matroid partition (Terao 2025) and multiagent matroid upgrading (Ma et al. 2026), which have no bearing on the unit-job greedy. Koca 2023 (facility location under disruptions) and Wang 2020 (evacuation, two-stage stochastic programming) are used to support the stochastic extension but do not address stochastic scheduling. The relevant stochastic scheduling and stochastic knapsack literature (Pinedo; Dean–Goemans–Vondrák 2008, Math. OR) is absent [verify].
Verification status of recent items:
- Delazeri and Ritt, arXiv:2603.29865: confirmed to exist this session. Its abstract reports strong NP-completeness on planar graphs, so "NP-completeness" in the manuscript is an understatement; check the current revised version (v. Aug 2026).
- Ma et al., arXiv:2606.01309: exists, but arXiv says submitted 2026-05-31; the manuscript's "Proceedings of AAMAS 2026" is not confirmed; verify.
- Chen–Lian–Mao–Zhang (the entry for "A nearly quadratic-time FPTAS for knapsack", SIAM J. Comput. 2025): the bibliography entry has no volume or pages and appears to be the STOC 2024 paper's journal version; verify the venue.
- Antonov et al. (C&OR 185:107281, 2026); Terao (ACM TALG 21(2), 2025); Rostamian et al. (OR Forum 7(1), 2026); Granda et al. (Operational Research 25(1):16, 2025); Wang et al. (IEEE Trans. Comput. 74(7), 2025); Günzel et al. (RTSS 2022); Wang (C&IE 145:106458, 2020); Koca (C&IE 183:109484, 2023): I could not confirm these entries and mark all as `[UNVERIFIED]`. I flag no entry as wrong.
I confirmed the core classical entries are plausible and consistent with my knowledge: Moore 1968, Lawler–Moore 1969, Sahni 1976 (Sahni is described in Heeger–Hermelin's introduction as using 1||Σw_jU_j as an FPTAS example, confirmed this session), Ibarra–Kim 1975, Gens–Levner 1981, Liu–Layland 1973, Heeger–Hermelin ESA 2024, Heeger–Molter STACS 2025, Hermelin–Molter–Shabtay INFORMS J. Comput. 2024, Hejl et al. 2022, Jin ICALP 2019.
**Evidence Anchor**: text: Section 2, "the objectives differ fundamentally: real-time scheduling asks whether every task can be met"
**Why it matters**: The related work reads as citation volume rather than positioning. A scheduling reader will notice the missing 1||Σp_jU_j and equal-processing-time literature immediately, and that crowds out the real gap argument.
**Suggestion**: Cut or compress the real-time and matroid-frontier paragraphs. Replace them with the missing references in W3, W4 and W6. Verify and complete the bibliography entries marked above.
**Severity**: Minor
**Confidence**: 4 — competence basis: citation audit; web-confirmed where noted

### W8: Propositions 1-3 are labeled "new" but are folklore-level, and the FPTAS is dominated by known schemes
**Problem**: Prop 1 is a two-job example (p=(1,2), d=(1,2), w=(1,W)). Prop 2 is the standard density-greedy counterexample for knapsack-type problems (here with deadlines). Prop 3 shows that one floor-based rounding analysis loses a factor between 1−ε and 1/(1+ε). The instance is tight only for this particular scaling (K from w_max, flooring, no completion step), and the paper's own Remark 3 shows a trivial post-processing step defeats it. The O(n^3/ε) scheme is slower than Gens–Levner's, as the manuscript itself acknowledges. Table 2 lists all three as "new".
**Evidence Anchor**: table: Table 2 (tab:summary), rows "Naive EDD ... new", "Weighted greedy repair ... new", "FPTAS analysis ... new"
**Why it matters**: Calling these "new" invites the same reaction as in W1. They are fine as illustrations.
**Suggestion**: Relabel them as illustrative examples. If tightness is kept, state it for the FPTAS-with-completion variant or for the best known scaling, where it would be informative.
**Severity**: Minor
**Confidence**: 4 — core expertise

### W9: Unconditional "polynomial iff" statements and small technical imprecisions
**Problem**: "Polynomial exactly when" (abstract, Thm 7, conclusion) must be conditional on P≠NP. The classification covers only constancy restrictions; the abstract's phrase "a complete classification" overstates. Table 2's FPTAS runtime O(n³/ε) with "classical type" is imprecise (Sahni's and Gens–Levner's bounds should be quoted). "Moore–Hodgson" is cited via Moore 1968 only. The DP runs in O(n·min(P, d_max)); the paper states only O(nP).
**Evidence Anchor**: text: Abstract, "the problem is polynomial exactly when dispatch times or weights are constant"
**Why it matters**: Precision of dichotomy statements matters for a complexity paper.
**Suggestion**: Add "assuming P≠NP", quote exact classical bounds, tone down "complete".
**Severity**: Minor
**Confidence**: 5 — standard complexity conventions

---

## Coverage Receipt
Not required: both Strengths and Weaknesses lists are populated.

---

## Table: Claimed vs. actual novelty

| # | Claim in manuscript | Manuscript's own label | My assessment | Accurate attribution? |
|---|---|---|---|---|
| 1 | Weak NP-hardness via Partition, equal deadlines (Thm 1) | Classical | Classical: Karp 1972 (job sequencing with penalties); Lawler–Moore 1969 context | Partly: Karp 1972 missing (W4) |
| 2 | Pseudo-polynomial DP, O(nP) (Thm 2) | Lawler–Moore | Classical | Yes |
| 3 | Value-scaling FPTAS (Thm 3) | "classical type" | Classical (Sahni 1976; Gens–Levner 1981 faster) | Yes, but dominated (W8) |
| 4 | FPTAS tight to factor 1−ε² (Prop 3) | New | Elementary statement about one rounding analysis; defeated by the paper's own completion step | Over-labeled (W8) |
| 5 | Equal-p, single vehicle, O(n log n) (Thm 4, m=1) | Classical | Classical (unit-job sequencing with profits; matroid greedy) | Yes, but standard references missing (W3) |
| 6 | Equal-p, m identical vehicles (Thm 4, m≥2) | "new (proof)" | Classical: P\|p_j=p\|Σw_jU_j reduces to unit jobs; transversal matroid on m·n slots | No (W3) |
| 7 | Classification by homogeneity (Thm 7) | New framing | Restatement of standard complexity-table entries (Moore; equal-p; Karp); k=1 case already in Heeger–Hermelin intro; boundary not the natural one (agreeable weights) | Over-labeled (W2) |
| 8 | Equal weights, Moore O(n log n) | Moore | Classical | Yes |
| 9 | Naive EDD / EDD-skip unbounded ratio (Prop 1) | New | Folklore two-job example | Over-labeled (W8) |
| 10 | Weighted greedy repair unbounded ratio (Prop 2) | New | Folklore density-greedy counterexample | Over-labeled (W8) |
| 11 | Hardness for every fixed m (Section 6.1 blockers) | Discussion | Correct, short; consistent with multiple-knapsack hardness; not new in substance | Needs MKP references (W4d) |
| 12 | "Release dates break equal-cost tractability" | Discussion/contribution 5 | Incorrect for m=1 (Baptiste 1999); correct only for the multi-vehicle unweighted case (Heeger–Molter 2025) | No (W5) |
| 13 | Camp Fire illustration and sensitivity analysis | Illustration | Numerically correct; four-site; no new theory | n/a (the one part with practical modeling content) |
| 14 | Lean 4 machine-checked proofs (Data Availability) | Statement | Not evaluable; available on request only; not a novelty claim | n/a |

---

## Detailed Comments

### Title & Abstract
The title advertises complexity, an exact algorithm and an FPTAS, all of which the abstract then labels classical. Retitle to match what is new. The abstract's "complete classification" overstates.

### Literature Review
- **Coverage**: strong on 2022-2026 wildfire OR and on parameterized scheduling (Heeger–Hermelin, Hermelin et al., Hejl et al. are good, current picks). Missing: Karp 1972; Baptiste 1999 and equal-processing-time literature; the 1||Σp_jU_j fine-grained line; Lawler 1976 (agreeable); scheduling with rejection/order acceptance; deadline-TSP/orienteering/profitable tours; Rachaniotis–Pappis 2006; stochastic knapsack/scheduling.
- **Integration quality**: enumerative in places (real-time paragraphs, matroid frontier). The gap argument ("no prior work establishes the knapsack-equivalence hardness result") is weak because the knapsack-equivalence is Karp/Lawler–Moore, and because the wildfire-scheduling subliterature was not surveyed.

### Theoretical Framework
Appropriate and correctly used. EDF exchange lemma, Lawler–Moore DP, value-scaling and the matroid greedy are standard and correctly applied.

### Academic Argument Quality
- **Factual accuracy**: see W5 (release dates). Sahni and Gens–Levner attributions are plausible; the O(n³/ε) vs. Gens–Levner comparison should quote their bounds.
- **Argument logic**: "heterogeneous deadlines never cause hardness" is true but follows from the definition of the problem class; it is not evidence about hazard models, particularly under the arrival reading where d_i depends on p_i (W6(iii)).
- **Terminology**: "FPTAS" in Thm 3 is correct. "Weakly NP-hard" for T ⊆ {d} is correct. "NP-complete" for the decision version under the numeric encoding is correct as stated.

### Contribution to the Field
- **Incremental contribution**: a clean, self-contained, correct exposition of a classical problem for a transportation audience, plus a small case study. The modeling observation about arrival vs. return readings is practical and correct.
- **Positioning**: honest in tone, inaccurate in specific labels (Table 2) and incomplete in references.
- **Overclaiming**: moderate: "new" labels in Table 2 (rows 4, 6, 7, 9, 10); "complete classification".

### Missing Key References (leads; do not cite without checking)
- Karp 1972, "Reducibility among combinatorial problems" (inclusion of job sequencing confirmed this session). Verify bibliographic details.
- Baptiste 1999, J. Scheduling 2:245–252 (confirmed this session).
- Baptiste, Brucker, Knust, Timkovsky, "Ten notes on equal-processing-time scheduling", 4OR (2004) `[UNVERIFIED]`.
- Lawler 1976, "Sequencing to minimize the weighted number of tardy jobs", RAIRO (agreeable weights) `[UNVERIFIED claim of O(n log n) agreeable case; the title and RAIRO 10:27-33 citation appeared in this session's search]`.
- Peha, special cases p_#=1 / w_#=1 (as cited in Heeger–Hermelin's introduction) `[UNVERIFIED exact reference]`.
- Hermelin, Karhi, Pinedo, Shabtay, "New algorithms for minimizing the weighted number of tardy jobs on a single machine", Ann. Oper. Res. (2021) `[UNVERIFIED]`.
- Fine-grained tardy processing time: Bringmann et al.; Klein–Polak–Rohwedder; Fischer–Wennmann `[UNVERIFIED venues]`.
- Knapsack FPTAS in JOCO: Kellerer and Pferschy 1999 (J. Comb. Optim. 3:59–71) `[UNVERIFIED]`; Lawler 1979 `[UNVERIFIED]`.
- Scheduling with rejection: Shabtay, Gaspar, Kaspi, J. Scheduling 16:3–28 (2013) (confirmed this session); order acceptance and scheduling: Slotnick 2011 `[UNVERIFIED]`.
- Firefighting scheduling: Rachaniotis and Pappis, Can. J. For. Res. 36(3):652–658 (2006) (confirmed this session).
- Selective routing with deadlines: Tsitsiklis 1992; Bansal–Blum–Chawla–Meyerson 2004; Feillet–Dejax–Gendreau 2005; Vansteenwegen–Souffriau–Van Oudheusden 2011 `[UNVERIFIED venues]`.
- Multiple knapsack PTAS: Caprara–Kellerer–Pferschy; Chekuri–Khanna `[UNVERIFIED]`.
- Woeginger 2000 (benevolent DP), possible FPTAS for fixed m `[UNVERIFIED applicability]`.
- Stochastic knapsack: Dean–Goemans–Vondrák 2008 `[UNVERIFIED]`.
- Graham–Lawler–Lenstra–Rinnooy Kan 1979 (notation).
- Edmonds 1971, Lawler 1976 (book), CLRS §16.5, Gabow–Tarjan 1985 for the matroid greedy and linear-time union–find `[UNVERIFIED details]`.

---

## Questions for Authors
1. Is there any result in the paper you believe is not derivable from Karp 1972, Moore 1968, Lawler–Moore 1969, Sahni 1976 and the unit-job matroid? If so, state it as a theorem separate from the "framing".
2. Under the arrival reading of Remark 1, d_i = d_i^haz + p_i/2 couples deadline and dispatch time. Can you give a structural result that exploits this coupling (for example when d_i^haz is a function of distance from the hazard origin)? That would be a transportation-specific result.
3. Why is the agreeable-weights case (Lawler) not part of the classification, given that it contains both tractable cells of Thm 7? Does your Thm 7 add anything beyond it?
4. For the single-vehicle equal-cost case with release dates, do you agree it is polynomial (Baptiste 1999)? If so, how should the "release dates break it" statement be revised?
5. Is an FPTAS for MWHED-m with fixed m already available from a general DP-to-FPTAS framework, and can you settle your open conjecture in Section 6.1 that way?

---

## Minor Issues

### Language / Grammar
- Section 4.4 and Section 6.1 contain long run-on paragraphs; shorten.

### Citation Format
- Chen–Lian–Mao–Zhang entry lacks volume/pages. Delazeri–Ritt and Ma et al. are arXiv-only entries; the AAMAS 2026 proceedings statement for Ma et al. needs verification.
- Cite Graham et al. 1979 for the three-field notation. Cite Tarjan or Gabow–Tarjan for union–find rather than Korte–Vygen.
- "Moore–Hodgson" is credited to Moore 1968 only.

### Figures and Tables
- Table 2 "Origin" column: reconsider the labels "new", "new (proof)", "new framing" (see the novelty table above).

### Layout
- None specific.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Domain-expert reading of the scheduling literature (this review) | DOES_NOT_MEET | text: Section 1 "we do not claim a new algorithmic technique"; table: Table 2 | Core results classical; claimed-new items elementary or classical (W1-W3, W8) | Lawler agreeable result and some references from memory (tagged verify) | Yes: unresolved and repairable only by adding a result or re-scoping |
| Methodological Rigor | n/a (Reviewer 1's remit) | NOT_ASSESSED | — | Proofs I re-derived are correct; formal rigor is outside my seat | none identified | no |
| Evidence Sufficiency | Domain-expert reading | MEETS | table: Table 8 (tab:scaling) | Claims about classical algorithms are supported by correct proofs and exhaustive checks | Experiments are checks of classical algorithms, not evidence of novelty | no |
| Argument Coherence | Domain-expert reading | PARTLY_MEETS | text: Section 6.1 "robust to adding vehicles but not to adding release dates" | One incorrect claim (W5); p-d coupling undercuts the deadline-heterogeneity interpretation (W6iii) | none identified | Yes (W5, W6) |
| Writing Quality | Domain-expert reading | MEETS | — | Clear and honest; long in places | Not my primary remit | no |
| Literature Integration | Domain-expert reading | PARTLY_MEETS | absence: Bibliography — expected Karp 1972, Baptiste 1999, deadline-TSP/orienteering; checked reference list, Section 2 | Strong recent parameterized-scheduling coverage; core and adjacent-field omissions; padded tangents | Several 2025-2026 items unconfirmed | Yes (W4, W6, W7) |
| Significance & Impact | Domain-expert reading | PARTLY_MEETS | text: Remark 1; Section 5.7 | Practical modeling observation (arrival vs return reading) and case study are useful; theoretical significance limited | Application impact not assessable from the manuscript alone | Yes (tied to W1) |

Recommendation rationale: the unresolved decision-bearing criteria are Originality (W1, repairable by adding a genuine result or re-scoping), Literature Integration (W4, W6, W7, repairable by adding and verifying references) and the Argument Coherence items W5 and W6(iii) (repairable by rewriting). Strengths on writing and honesty do not offset the originality gap.
