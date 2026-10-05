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

Numbers are those of the revised manuscript (items are numbered in order of
appearance; the submitted version had Lemma 1, Theorems 2-5, Propositions
6-8, Theorem 9, which are now Lemma 2, Theorems 3, 5, 7, 8, Propositions 9,
10, 11, Theorem 12). Names are in namespace `Mwhed` unless noted.

| Paper | Lean | File |
|---|---|---|
| Definition 1, objective (1), Assumption 1 | `Inst`, `onTimeW`, `W`, `completion`, `IsOrder`, `Feasible`, `IsOPT`, `IndivFeasible` | `Defs` |
| Eq. (1): `W(sigma)` = weight of on-time set | `W_eq_weight_onTimeSet` | `Core` |
| Lemma 2 (EDD feasibility, ties arbitrary) | `lemma1_edd` (via `feasible_iff_thr`) | `Core` |
| `W*` = max weight of a feasible set | `isOPT_iff_max_feasible`, `exists_isOPT` | `Core` |
| Theorem 5 (exact DP, correctness) | `dpTable_eq_bestWeight`, `isOPT_iff_dp_max`; eq. (2) with cases: `dpTable_succ`, `dpTable_succ_cases` | `Dp`, `DpLemmas` |
| Fact 1 (common deadline) | `feasible_const_deadline_iff` | `Hardness` |
| Theorem 3 (PARTITION reduction; odd `A` and `a_i > A/2` handled) | `partition_reduction`, `partition_reduction_full`, `partition_iff_isOPT`, `partition_no_of_odd`, `partition_no_of_big`, `partitionInst_*` | `Hardness`, `DpLemmas` |
| Proposition 4 (constant hazard-arrival time, arrival reading) | `feasible_iff_mid` (eq. mid), `hazard_partition_reduction`, `hazard_partition_iff_isOPT`, `hazardInst_*` | `HazardHardness` |
| Lemma 6 (value-indexed table `g`, eq. g with cases) | `lem_g_zero_zero`, `lem_g_zero_pos`, `lem_g_succ`, `lem_g_succ_zero_scaled`, `gSpec_eq_gTab`; recursion `gTab`, `gTab_eq_iInf` | `DpLemmas`, `Fptas` |
| Theorem 7 (FPTAS), classical bound | `scaling_loss`, `fptas_case_scaled`, `fptas_case_vacuous`, `fptas_guarantee` | `Fptas` |
| Theorem 7 (FPTAS), refined constant `rho_{n,eps}` | `fptasRef_F1`, `fptasRef_F2`, `fptasRef_F3`, `fptasRef_core`, `fptas_refined`, `rhoRef_chain`, `inv_lt_rhoRef` | `FptasRefined` |
| Proposition 11 (the refined guarantee is tight; both families) | `tight1_algOutput_iff`, `tight1_bound`, `tight2_algOutput_iff`, `tight2_bound`, `tendsto_tightB1`, `tendsto_tightB2`, `prop_tight`; earlier first-family lemmas `tight_*` | `FptasRefined`, `Fptas` |
| Theorem 8 (equal-size sites), identical vehicles: slots, counting criterion | `SlotFeasible`, `slotFeasible_iff_count`, `schedulable_iff`, `feasible_iff_slotFeasible` (`Mwhed.EqualDispatch`) | `Matroid` |
| Theorem 8, identical vehicles: matroid, greedy | `slotFeasible_isIndepFamily`, `greedy_optimal`, `equalP_greedy_optimal`, `equalP_greedy_optimal_mwhed` | `Matroid` |
| Theorem 8, **vehicles with different dispatch times** (a)-(c): `C(t)`, counting criterion, matroid, greedy, link to schedules, identical-time special case | `cap`, `card_slots`, `slotFeasible_iff_count`, `slotFeasible_isIndepFamily`, `greedy_slotFeasible_optimal`, `schedulable_iff`, `speeds_greedy_optimal`, `cap_const`, `slotFeasible_const_iff` (`Mwhed.Speeds`) | `MatroidSpeeds` |
| Algorithm 3 decision rule (latest-free-slot, logic only) | `algStep_accepts_iff`, `algRun_eq_greedy`, `equalP_algRun_optimal` | `Matroid` |
| Example 5 (m=1: 22, m=2: 26) | `Example5.*` | `Matroid` |
| Proposition 9 (naive EDD / EDD-skip unbounded) | `prop6_naive`, `prop6_skip`, `prop6_unbounded_ratio` | `Heuristics` |
| Proposition 10 (greedy repair) | `prop7_repair`, `prop7_unbounded_ratio` | `Heuristics` |
| Running example (W* = 23, naive EDD = 16), Example 2 | `runInst_isOPT`, `runInst_naiveEDD`, `runInst6_*` | `Examples` |
| Example 4 arithmetic | `ex4_K`, `ex4_scaled`, `ex4_bound` | `Fptas` |
| Theorem 12, hardness part (the Theorem 3 instances lie in every class `C_T`, `T` subset of `{d}`) | `classification_b_hardness`, `partitionInst_mem_class` | `Classification` |
| Theorem 12, equal-weight reduction to max cardinality | `isOPT_const_weight_iff` | `Classification` |

(Names such as `prop6_*`, `prop7_*` and `lemma1_edd` keep the numbering of the
first version of the manuscript.)

## What is NOT formalised

* **Complexity statements.** NP-hardness, weak NP-hardness, NP-membership,
  polynomial/pseudo-polynomial running times (O(nP) for Thm 5, O(n^3/eps)
  for Thm 7, O(m + n log n) for Thm 8) and polynomial-time constructibility
  of the reductions. Lean has no machine model here; what is proved is the
  *mathematical* content: the reductions preserve yes/no answers (Thm 3,
  Prop 4), the DP recurrences compute `W*` (Thm 5, Lemma 6), the output of
  the scaled algorithm meets its guarantee (Thm 7) and that guarantee is
  attained (Prop 11).
* The heap-based slot generation, the "only the n earliest slots matter"
  fact, and the union-find data structure of Algorithm 3 (its decision
  rule and correctness are proved, not its data structure or cost).
* **Proposition 1** (the depot round-trip model is conservative; a
  triangle-inequality argument) and Remark 1 (arrival vs return reading).
* Moore's algorithm (a cited classical result, not a claim of the paper).
* The limit `n -> infinity` of `rho_{n,eps}` as a filter statement (the
  exact limits as `M -> infinity` for each family are proved).
* That the DP on the scaled weights of Example 4 selects `{1,2,4}`.
* All experiments, the case study and the Python scripts (checked by
  `../verify_small.py` and `../verify_extensions.py`, not by Lean).

## Differences from the paper's proofs (same statements)

* Lemma 2 is proved via a threshold characterisation (`Thr`: for every `t`
  the total dispatch time of sites with `d ≤ t` is at most `t`) rather than
  the paper's direct prefix argument.
* Theorem 8 Step 4 is proved by induction on the list with iterated matroid
  augmentation instead of the paper's sorted-comparison argument.
* `SlotFeasible` in `MatroidSpeeds` uses an `Option (Fin m × ℕ)`-valued
  assignment so that it is correct for `m = 0`; the Proposition 4 reduction
  needs no evenness hypothesis on `A` (an odd `A` makes both sides false).
* `fptasRef_core` needs neither optimality of the compared set nor that
  `h` is heaviest, only that `{h}` is feasible and `K = ε w_h / n`; the
  wrapper `fptas_refined` supplies the heaviest site.
* Site indices in the examples are 0-based. Greedy ties are broken by input
  order. `D_i = ⌊d_i/p⌋` is not capped at `n` (sites with `D_i = 0` are
  simply never feasible).
* Real-valued guarantees (Thm 4) are stated over `ℝ`; `−∞` in the DP is
  `⊥ : WithBot ℕ`.

`.lake` is a build directory (git-ignored); in this repository checkout it
is a symlink to a shared Mathlib build.
