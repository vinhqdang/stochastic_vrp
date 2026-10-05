# Seat R2: domain and theory re-review of BATON revision 1

## Seat and angle
Peer Reviewer 2 (domain and theory): stochastic VRP with recourse, optimal stopping, MDPs, regression-based ADP. I checked Section 3 of main.tex (lines 388–1146) line by line. That covers Assumptions 1–2, Propositions 1–4 and their proofs, eq. (bellman3) and the fresh-start estimator. I also checked the Lean library (BatonProofs/*.lean, README table) against the paper's claims, and the r1 CSV/code where a theoretical statement is backed by a number. I did not build the Lean project, so the claim that it compiles rests on the source only.

**Mathematical checks, all correct as stated:**
- The Gaussian-factor sufficiency argument (l. 553–558) holds. With a common loading β and common noise variance, the likelihood depends on Z only through W_k − Σμ_j. The posterior variance does not depend on W_k, so the kernel is Gaussian with an increasing affine mean. It is therefore Markov and stochastically monotone.
- Prop. 4 needs only E ≥ 0 and E non-increasing. The key step is C_{k+1} ≤ C⁰_{k+1} ≤ max_{j≥k+2} E_j ≤ E_{k+1}, and Lean's `C_le_E` confirms that no ordering between H (or R) and E is used.
- The strictness condition of Prop. 2 is correct and sufficient: a positive-probability path reaches a later node where the myopic rule fires, and the strict inequality holds there (`Cn_lt_C0`).
- Prop. 3's identity and bound are correct for general position-dependent prices. The flat-price case analysis (ω, 0, ω − C_fail) is right.
- Prop. 1's bounds and the attainment condition are correct. Lean encodes admissibility as an abstract class, which is fine.

## Verdicts on round-1 comments

| item | verdict | evidence anchor | residual gap |
|---|---|---|---|
| R1.1 | RESOLVED | Ass. 1 is now a stochastically monotone Markov chain (l. 534–540); covered cases and failure cases are spelled out (l. 548–566); ρ-sweep, day factor and shape test in Tab. dependence | Sufficiency is claimed for the three-action problem too, but it fails there under the factor model (New issue 2) |
| R1.3 | RESOLVED | l. 888–909: post-return load, exact state x_k = −D_{≤k}, a conservative convention, repeated returns allowed; eq. (bellman3) l. 913 | "reloading the plan's remaining deliveries" (l. 463–464, Alg. 2 comment) contradicts the "unload pickups only" description (MINOR 7) |
| R1.7 | PARTLY_RESOLVED | Intro l. 211–232 now separates model / structural results / method and claims only the adaptation for the method | Props. 2 and 4 are instances of textbook facts and are presented as contributions without citing the monotone-MDP / monotone-stopping literature (New issue 4) |
| R2.M1 (theory) | PARTLY_RESOLVED | Prop. 4 now rests on stochastic monotonicity, not independence; Tab. dependence gives ρ ∈ {0, 0.3, 0.6, 0.9} and a shape test; l. 861–867 qualifies isotonic use under approximate sufficiency | Handoff side done. The three-action recursion is not exact under the dependence that Ass. 1 claims to cover (New issue 2). The shape test is low-power and covers the handoff menu only (MINOR 9) |
| R2.M2 (theory) | RESOLVED | l. 939–960 splits the bias into three signed components; Tab. fresh quantifies them on Salhi–Nagy (the CMT geometry the reviewer asked about); BATON-cf fixes bias (iii) | The reference "near-exact F_k" is itself a binned-DP estimate; the City out-of-sample bias −1.05% has the opposite sign to claim (i) (New issue 5) |
| R2.M3 | RESOLVED | Conclusion l. 2011–2024 sets the boundaries: partial handoff is out of scope, and the augmented-state extension is sketched; the abstract says returns rarely pay on urban networks; Tab. fresh City row | None material |
| R2.M4 | PARTLY_RESOLVED | Prop. 3 (l. 733) prices the over-triggering; l. 768–779 says when a threshold is "good enough"; Tab. regret evaluates both sides | Abstract, intro and conclusion generalise Prop. 2 from the myopic rule to "every fixed threshold", which is false (New issue 1) |
| R3.2 | RESOLVED | Ass. 2 drops H ≤ E (l. 542–545); l. 476–488 says eq. (prices) is calibration only; l. 500–512 treats the standby day rate as holding cost and points to §pool for a reserved pool | None on the theory side |

## New issues introduced or exposed by the revision

1. **MAJOR: Prop. 2 overclaimed as "every fixed threshold over-triggers".** Anchors: abstract l. 92; intro l. 181–182 and l. 219–221; conclusion l. 1960; results l. 1621–1626.
   - Problem: Prop. 2 covers only the myopic rule C⁰_k > H_k, which under flat prices is τ = ω_F/C_fail. A tuned τ′ > ω_F/C_fail stops on a strict subset of the myopic region, which need not contain the optimal region, so it can under-trigger. The paper itself reports that tuned thresholds drift upward (l. 781–783).
   - The "clean-day probability" bound is the flat-price specialisation. Under geometric prices, eq. (regret) adds a declining-price term.
   - Fix: say "the myopic (break-even) threshold over-triggers". State the clean-day bound as the flat-price case. Stop presenting the tuned-threshold gap (l. 1621) as "the same pattern" of Prop. 3.

2. **MAJOR: eq. (bellman3) is exact only when the post-return increments are independent of the pre-return history.** Anchors: l. 548–558, Prop. 4 l. 820–824, eq. (bellman3) l. 913.
   - Problem: under the second case of Ass. 1 (the Gaussian day factor), the future law after a return at (k, W_k = w) depends on the posterior of Z, i.e. on w. The reset state (k, 0) evaluated through C_k(0) conditions on "W_k = 0", which is the wrong posterior. After a return, (k, W) is not a sufficient state: the sufficient state is (k, W_post, W_pre). Prop. 4 proves monotonicity of the recursion, which is true (Lean `baton_C_monotone` models the reset as a restart of the same kernel), but the recursion is not the value function of the controlled problem in that case. Bias (iii) (l. 945–950) is this gap seen empirically. It is not tied back to the assumption.
   - Fix: after eq. (bellman3), state that it is the exact DP under independent increments and a misspecification under factor dependence. Present BATON-cf as the corresponding correction. Qualify "(k, W_k) is a sufficient state" (l. 548) as holding for the handoff-only problem.

3. **MAJOR: the convergence claim cites the wrong kind of result.** Anchor: l. 855–863.
   - Problem: Clément, Lamberton & Protter (2002) analyse LSM with a fixed finite linear basis. They show convergence to the basis-restricted value as N → ∞, and they treat basis growth separately. They do not cover a nonparametric estimator. "Glivenko–Cantelli" is not the property such proofs need; they need entropy or uniform L² rates.
   - The shape guarantee has a further gap. Prop. 4 gives monotonicity of C_k under **optimal** downstream play, but the step-k regression target is the cost under the **fitted** downstream policy (l. 802–809). That conditional mean need not be monotone: a fitted boundary set too high makes the cost-to-go jump down at the boundary. So the statement "projection onto exactly the class containing the truth" (l. 851–854) holds only in the limit.
   - Fix: cite nonparametric LSM analyses that handle general function classes, e.g. Egloff (2005, Ann. Appl. Probab.) and Zanger (2013, Finance Stoch.); put both in VERIFY_CITATIONS. State consistency as a conjecture or a sketch. Add one sentence on the fitted-policy target.

4. **MAJOR (positioning): the structural results are not placed against known theory.** Anchors: l. 216–224, §2 l. 314–318, Props. 2 and 4.
   - C ≤ C⁰ and "the one-stage look-ahead region contains the optimal region" are standard optimal-stopping facts (Chow–Robbins–Siegmund 1971 is cited, but only for verification).
   - Monotone value functions under stochastically monotone kernels are classical in monotone MDP theory (Serfozo 1976; Puterman 1994 §4.7; Müller & Stoyan 2002). None of these is cited.
   - Yang, Mathur & Ballou (2000) already prove per-stop optimal threshold restocking policies on a fixed route. That is the position-dependent threshold structure of l. 787–790. Yet l. 170–174 lists them with fixed-threshold heuristics.
   - Minis & Tatarakis (EJOR 2011) is missing. It studies stochastic single-vehicle routing with delivery and pickup on a predefined sequence with optimal DP restocking, the closest prior model to this one (verify the citation).
   - Fix: cite these works. Recast Props. 2 and 4 as instantiations whose contribution is the peak/breach structure and the menu generality, not the inequalities themselves.

5. **MINOR: the "near-exact" DP yardsticks are exact only for a Markov approximation.** Anchors: l. 1098–1106; Tab. fresh; abstract ratio 86–96%.
   - Problem: DP_50k and DP³_50k treat W as Markov and use an unconditional F. Under the benchmark ρ = 0.6 copula, which the paper says violates Ass. 1 (l. 560–563), they solve the Markov-approximated, conservative-reset problem. They are also fitted policies, not optimal ones. The City out-of-sample F bias is −1.05% (fitted policy cheaper than the "optimum"), which contradicts the sign of bias (i) unless the reference is not optimal. The exact-reset BATON beats DP³ (57.9 vs 53.5).
   - Fix: call the DPs "high-data plug-in references in the (k, W_k) state". Compute the F-bias columns against a fixed common reference, or explain the sign.

6. **MINOR: Prop. 2's last claim needs Ass. 2.** Anchor: l. 690 and proof l. 717–718. The strict-boundary step uses monotonicity of C⁰_k, which comes from Prop. 4 and so needs E non-increasing and non-negative. Lean's kernel form `C_le_C0` also takes `hE0, hEanti`. Fix: "Under Assumption 1 (and Assumption 2 for the boundary statement)".

7. **MINOR: the reset convention is described inconsistently.** Eq. (restockprice) text (l. 463–464) and the Alg. 2 comment ("deliveries reloaded") say remaining deliveries are reloaded. §3.5 (l. 895–899) says only the pickups are unloaded, leaving L₀ − D_{≤k} on board. The two give different exact states. Fix: choose one description. Also Alg. 1 comments F_k as the "exact fresh-start value", while the text lists three biases; change it to "simulated".

8. **MINOR: the Markov-free claim is correct but nearly empty for the fitted policy.** Anchor: l. 402–409. On the empirical distribution of continuous-demand training days, each history after stop 1 is typically a single path. The history-conditioned C and C⁰ then equal realised costs, and the myopic rule in `Regret.lean` (conditioned on the node) is not the W_k-conditioned rule that is deployed. The README goes further and calls this "exactly the setting of the fitted policy". Fix: say that the finite-tree statements apply to history-measurable rules, and that the deployed W_k rules are covered only under Ass. 1.

9. **MINOR: the shape test is weak evidence.**
   - The violation rate under independence is 0.2%, against a nominal 2.5%. So the null calibration shows the test is conservative, not calibrated.
   - Adjacent-bin z-tests (25 bins) detect only steep local decreases.
   - The code (`_shape_job`) tests the handoff-only cost-to-go, not the three-action or BATON-cf targets.
   - Fix: say so in the caption, or add a global test (e.g. the isotonic-vs-unconstrained LR statistic).

10. **MINOR: the Lean correspondence has gaps that the text does not mention.**
    - The strict-boundary theorem (`Boundary.lean`) assumes C(w̄) < C⁰(w̄) as a hypothesis on continuous real functions. `Cn_lt_C0` derives the strict inequality on finite trees, where continuity is meaningless. The two pieces are not formally connected.
    - Prop. 1 is formalised for flat prices and an abstract admissible class. The step "T−1 a stopping time ⇒ in the class" is not formalised, and the "exactly when" of the proof (l. 659) is only the "if" direction.
    - Prop. 4's kernel operator stands in for the conditional expectation and does not prove that the recursion is the optimal value.
    - Fix: one sentence in §3 listing these modelling abstractions, instead of the blanket "verified formally" (l. 399–400).

11. **MINOR:** "P(T ≤ m) may be arbitrary in [0,1)" (l. 623) should read [0,1]: a deterministic peak followed by W_m ≤ 0 gives 1. The σ⁰ and σ* definitions (l. 585, l. 683) should restrict to k < T, otherwise C_k(W_k) is evaluated above B.

## Claim/number consistency problems
- Tab. dependence: BATON is bold as "highest implementable" (text l. 1673–1674), but BATON-cf is higher at ρ = 0.3/0.6/0.9 Det (22.5 / 29.2 / 37.6 vs 21.9 / 27.9 / 35.4) and at every SAA row. Either bold cf or say "BATON or its cf variant".
- Tab. fresh City out-of-sample F bias of −1.05% vs text claim (i) "pushes F̂ above the optimal F" (l. 940–942): the sign is inconsistent (New issue 5).
- Tab. regret: the bound holds on 734/735 routes (`\regretBoundHolds`), while Prop. 3 is a theorem. The text should attribute the one violation to Monte Carlo error in the separately estimated C⁰ and C.
- `\regretCleanShare` = 90% is computed as Σ clean / Σ bound (make_tables.py l. 934). Because the T = σ⁰+1 term is negative, the clean term exceeds the bound on City Det (5.7 vs 4.8). A "share" above 100% in one row makes "accounts for 90%" misleading; describe it as a ratio.

## Recommendation signal
**Minor revision (close to major on the theory text).** The four propositions are mathematically sound, and the Lean library checks what it states. R1.1, R1.3, R2.M2, R2.M3 and R3.2 are properly addressed.
What remains:
- "Every fixed threshold over-triggers" is false as stated, and it appears in the abstract.
- The sufficiency and exactness of eq. (bellman3) under factor dependence is overstated.
- The Clément et al. convergence claim cites the wrong kind of result.
- Props. 2 and 4 need their standard antecedents cited.

All four are fixable in the text, with no new experiments.
