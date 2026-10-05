# MwhedProofs — Lean 4 verification of the MWHED theory

Machine-checked proofs (Lean 4 + Mathlib, toolchain in `lean-toolchain`,
Mathlib pinned in `lakefile.toml`) of the mathematical content of
`../main.tex`. The full library builds with no `sorry`, no `native_decide`
and no added axioms; every headline theorem depends only on the standard
`propext`, `Classical.choice`, `Quot.sound`.

    cd papers/csonet2026/MwhedProofs
    lake exe cache get      # fetch the Mathlib build cache (optional)
    lake build              # builds all modules, ~minutes with cache
    # audit, e.g.:  #print axioms Mwhed.isOPT_iff_dp_max

## Correspondence: paper <-> Lean

| Paper | Lean (namespace `Mwhed` unless noted) | File |
|---|---|---|
| Definition 1, objective (1), Assumption 1 | `Inst`, `onTimeW`, `W`, `completion`, `IsOrder`, `Feasible`, `IsOPT`, `IndivFeasible` | `Defs` |
| Eq. (1): `W(σ)` = weight of on-time set | `W_eq_weight_onTimeSet` | `Core` |
| Lemma 1 (EDD feasibility) | `lemma1_edd` (via `feasible_iff_thr`) | `Core` |
| `W*` = max weight of a feasible set | `isOPT_iff_max_feasible`, `exists_isOPT` | `Core` |
| Theorem 3 (exact DP, correctness) | `dpTable_eq_bestWeight`, `isOPT_iff_dp_max` | `Dp` |
| Fact 1 (common deadline) | `feasible_const_deadline_iff` | `Hardness` |
| Theorem 2 (PARTITION reduction) | `partition_reduction`, `partition_iff_isOPT`, `partitionInst_*` | `Hardness` |
| Theorem 4 (FPTAS guarantee), both cases | `scaling_loss`, `fptas_case_scaled`, `fptas_case_vacuous`, `fptas_guarantee` | `Fptas` |
| Appendix B value-indexed DP `g(i,v)` | `gTab`, `gTab_eq_iInf` | `Fptas` |
| Proposition 8 (FPTAS tightness) | `tight_isOPT`, `tight_algorithm_output`, `tight_ratio`, `tight_limit_identity` | `Fptas` |
| Theorem 5/6, Steps 1–2 (slots, counting criterion) | `SlotFeasible`, `slotFeasible_iff_count`, `schedulable_iff`, `feasible_iff_slotFeasible` (`Mwhed.EqualDispatch`) | `Matroid` |
| Theorem 5/6, Step 3 (matroid) | `slotFeasible_isIndepFamily` | `Matroid` |
| Theorem 5/6, Step 4 (greedy optimal) | `greedy_optimal`, `equalP_greedy_optimal`, `equalP_greedy_optimal_mwhed` | `Matroid` |
| Step 5 / Algorithm 3 (latest-free-slot rule, logic only) | `algStep_accepts_iff`, `algRun_eq_greedy`, `equalP_algRun_optimal` | `Matroid` |
| Example 5 (m=1: 22, m=2: 26) | `Example5.*` | `Matroid` |
| Proposition 6 (naive EDD / EDD-skip unbounded) | `prop6_naive`, `prop6_skip`, `prop6_unbounded_ratio` | `Heuristics` |
| Proposition 7 (greedy repair, family of Prop. 7) | `prop7_repair`, `prop7_unbounded_ratio` | `Heuristics` |
| Running example (W* = 23, naive EDD = 16), Example 2 | `runInst_isOPT`, `runInst_naiveEDD`, `runInst6_*` | `Examples` |
| Example 4 arithmetic (K = 5/4, scaled weights) | `ex4_K`, `ex4_scaled`, `ex4_bound` | `Fptas` |
| Theorem 9, hardness part (Thm 2 instances lie in every class C_T, T ⊆ {d}) | `classification_b_hardness`, `partitionInst_mem_class` | `Classification` |
| Theorem 9, equal-weight reduction to max cardinality | `isOPT_const_weight_iff` | `Classification` |

## What is NOT formalised

* **Complexity statements.** NP-hardness, weak NP-hardness, NP-membership,
  polynomial/pseudo-polynomial running times (O(nP) for Thm 3,
  O(n log n) for Thm 5, O(n²/ε) for Thm 4) and polynomial-time
  constructibility of the reduction. Lean has no machine model here; what is
  proved is the *mathematical* content: the reduction preserves yes/no
  answers (Thm 2), the DP recurrence computes `W*` (Thm 3), the output of the
  scaled algorithm meets the guarantee (Thm 4).
* The union-find data structure and the sorting step of Algorithm 3
  (its decision rule and correctness are proved, not its data structure).
* Moore's algorithm (a cited classical result, not a claim of the paper).
* The limit `n, M → ∞` in Proposition 8 as a filter statement; the finite-n
  ratio bound and the algebraic limit identity `1/((1+ε)(1−ε)) = 1/(1−ε²)`
  are proved.
* That the DP on the scaled weights of Example 4 selects `{1,2,4}`.
* The equal-p *polynomial* half of Theorem 9(a) beyond Theorem 5 (only the
  hardness and the equal-weight reductions are formal).
* All experiments, the case study and the Python scripts (those are checked
  by `../verify_small.py`, not by Lean).

## Differences from the paper's proofs (same statements)

* Lemma 1 is proved via a threshold characterisation (`Thr`: for every `t`
  the total dispatch time of sites with `d ≤ t` is at most `t`) rather than
  the paper's two exchange arguments.
* Theorem 5 Step 4 is proved by induction on the list with iterated matroid
  augmentation instead of the paper's sorted-comparison argument.
* Site indices in the examples are 0-based. Greedy ties are broken by input
  order. `D_i = ⌊d_i/p⌋` is not capped at `n` (sites with `D_i = 0` are
  simply never feasible).
* Real-valued guarantees (Thm 4) are stated over `ℝ`; `−∞` in the DP is
  `⊥ : WithBot ℕ`.

`.lake` is a build directory (git-ignored); in this repository checkout it
is a symlink to a shared Mathlib build.
