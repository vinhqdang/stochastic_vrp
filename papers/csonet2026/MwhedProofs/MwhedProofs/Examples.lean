import MwhedProofs.Defs
import MwhedProofs.Hardness
import MwhedProofs.Heuristics

set_option linter.unusedSectionVars false

/-!
# Machine-checked running examples (Examples 1 and 2, Appendix A)

Sites `1,2,3,4` of the paper are `0,1,2,3 : Fin 4` here.

* `runInst`  : the running instance `p=(2,3,1,4)`, `d=(4,6,2,9)`, `w=(5,8,3,10)`.
* `runInst_isOPT`      : `W* = 23` (Example 1), by exhaustive enumeration of all
  dispatch orders (every order of `Fin 4` is a list `[a,b,c,d]`).
* `runInst_opt_order`  : the optimal order serves sites `{1,2,4}` first, i.e.
  `[0,1,3,2]`, with completion times `2,5,9`.
* `runInst_naiveEDD`   : naive earliest-deadline-first order (`3,1,2,4`) has
  on-time weight `16`; `runInst_naiveEDD_order` records the order itself.
* `runInst_feasible_iff` : Appendix A -- a set is feasible iff it is not the
  full set; `runInst_table` lists `Σp` and `W` for all `16` subsets;
  `runInst_weight_le` : every feasible set has weight at most `23`.
* `runInst6`, `runInst6_isOPT`, `runInst6_knapsack` : Example 2 (common
  deadline `D = 6`): `W* = 16`, attained at `{1,2,3}`, and equal to the 0/1
  knapsack optimum with sizes `p`, values `w`, capacity `6` (via Fact 1).

Every statement is closed by `decide` (kernel evaluation over `Fin 4`).
-/

namespace Mwhed

/-- The running instance of Example 1. -/
def runInst : Inst (Fin 4) where
  p := ![2, 3, 1, 4]
  d := ![4, 6, 2, 9]
  w := ![5, 8, 3, 10]
  p_pos := by intro i; fin_cases i <;> simp

/-- Every dispatch order of four sites is a list `[a,b,c,d]`. -/
theorem isOrder_fin4 {σ : List (Fin 4)} (h : IsOrder σ) :
    ∃ a b c d : Fin 4, σ = [a, b, c, d] := by
  have hl := isOrder_length h
  simp only [Fintype.card_fin] at hl
  match σ, hl with
  | [a, b, c, d], _ => exact ⟨a, b, c, d, rfl⟩

/-- Assumption 1 holds for the running instance. -/
theorem runInst_indivFeasible : IndivFeasible runInst := by
  intro i; fin_cases i <;> simp [runInst]

/-- **Example 1**: `W* = 23`. -/
theorem runInst_isOPT : IsOPT runInst 23 := by
  have key : ∀ a b c d : Fin 4, IsOrder [a, b, c, d] → W runInst [a, b, c, d] ≤ 23 := by
    unfold IsOrder; decide
  refine ⟨⟨[0, 1, 3, 2], by unfold IsOrder; decide, by decide⟩, fun σ hσ => ?_⟩
  obtain ⟨a, b, c, d, rfl⟩ := isOrder_fin4 hσ
  exact key a b c d hσ

/-- The optimal order of Example 1 serves sites `1,2,4` (indices `0,1,3`) and then
the sacrificed site `3` (index `2`); completion times of the first three are `2,5,9`. -/
theorem runInst_opt_order :
    IsOrder ([0, 1, 3, 2] : List (Fin 4)) ∧ W runInst [0, 1, 3, 2] = 23 ∧
      completion runInst [0, 1, 3, 2] 0 = 2 ∧ completion runInst [0, 1, 3, 2] 1 = 5 ∧
      completion runInst [0, 1, 3, 2] 3 = 9 := by
  refine ⟨by unfold IsOrder; decide, by decide, by decide, by decide, by decide⟩

/-- Example 1: the naive earliest-deadline-first order is `3,1,2,4`
(indices `2,0,1,3`) ... -/
theorem runInst_naiveEDD_order : eddSort runInst [0, 1, 2, 3] = [2, 0, 1, 3] := by decide

/-- ... with completion times `1,3,6,10`, and on-time weight `3 + 5 + 8 = 16`
(site `4` completes at `10 > 9` and is exposed); the total weight is `26`. -/
theorem runInst_naiveEDD :
    naiveEDDValue runInst [0, 1, 2, 3] = 16 ∧
      completion runInst [2, 0, 1, 3] 2 = 1 ∧ completion runInst [2, 0, 1, 3] 0 = 3 ∧
      completion runInst [2, 0, 1, 3] 1 = 6 ∧ completion runInst [2, 0, 1, 3] 3 = 10 ∧
      weight runInst Finset.univ = 26 := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide⟩

/-- Feasibility of a set on four sites, unfolded to an enumeration of orders. -/
theorem feasible_fin4_iff (I : Inst (Fin 4)) (S : Finset (Fin 4)) :
    Feasible I S ↔ ∃ a b c d : Fin 4, IsOrder [a, b, c, d] ∧
      ∀ i ∈ S, completion I [a, b, c, d] i ≤ I.d i := by
  constructor
  · rintro ⟨σ, hσ, hS⟩
    obtain ⟨a, b, c, d, rfl⟩ := isOrder_fin4 hσ
    exact ⟨a, b, c, d, hσ, hS⟩
  · rintro ⟨a, b, c, d, h, hS⟩
    exact ⟨_, h, hS⟩

/-- **Appendix A**: exactly one subset of the running instance is infeasible, the full set. -/
theorem runInst_feasible_iff (S : Finset (Fin 4)) : Feasible runInst S ↔ S ≠ Finset.univ := by
  rw [feasible_fin4_iff]
  revert S
  unfold IsOrder
  decide

/-- Every feasible set of the running instance has weight at most `23`. -/
theorem runInst_weight_le (S : Finset (Fin 4)) (h : Feasible runInst S) :
    weight runInst S ≤ 23 := by
  rw [runInst_feasible_iff] at h
  revert S
  decide

/-- **Appendix A, Table 2**: total dispatch time and weight of all `16` subsets
(feasibility of the `15` proper subsets is `runInst_feasible_iff`). -/
theorem runInst_table :
    (time runInst ∅ = 0 ∧ weight runInst ∅ = 0) ∧
    (time runInst {0} = 2 ∧ weight runInst {0} = 5) ∧
    (time runInst {1} = 3 ∧ weight runInst {1} = 8) ∧
    (time runInst {2} = 1 ∧ weight runInst {2} = 3) ∧
    (time runInst {3} = 4 ∧ weight runInst {3} = 10) ∧
    (time runInst {0, 1} = 5 ∧ weight runInst {0, 1} = 13) ∧
    (time runInst {0, 2} = 3 ∧ weight runInst {0, 2} = 8) ∧
    (time runInst {0, 3} = 6 ∧ weight runInst {0, 3} = 15) ∧
    (time runInst {1, 2} = 4 ∧ weight runInst {1, 2} = 11) ∧
    (time runInst {1, 3} = 7 ∧ weight runInst {1, 3} = 18) ∧
    (time runInst {2, 3} = 5 ∧ weight runInst {2, 3} = 13) ∧
    (time runInst {0, 1, 2} = 6 ∧ weight runInst {0, 1, 2} = 16) ∧
    (time runInst {0, 1, 3} = 9 ∧ weight runInst {0, 1, 3} = 23) ∧
    (time runInst {0, 2, 3} = 7 ∧ weight runInst {0, 2, 3} = 18) ∧
    (time runInst {1, 2, 3} = 8 ∧ weight runInst {1, 2, 3} = 21) ∧
    (time runInst Finset.univ = 10 ∧ weight runInst Finset.univ = 26) := by
  decide

/-- The full set is infeasible because site `4` would complete at `10 > 9`
even in the best (deadline) order. -/
theorem runInst_full_infeasible : ¬ Feasible runInst Finset.univ := by
  rw [runInst_feasible_iff]; simp

/-! ### Example 2: the same sites under the common deadline `D = 6` -/

/-- Same `p` and `w` as the running instance, all deadlines equal to `6`. -/
def runInst6 : Inst (Fin 4) := { runInst with d := fun _ => 6 }

/-- Example 2: the optimum under the common deadline `6` is `16`. -/
theorem runInst6_isOPT : IsOPT runInst6 16 := by
  have key : ∀ a b c d : Fin 4, IsOrder [a, b, c, d] → W runInst6 [a, b, c, d] ≤ 16 := by
    unfold IsOrder; decide
  refine ⟨⟨[0, 1, 2, 3], by unfold IsOrder; decide, by decide⟩, fun σ hσ => ?_⟩
  obtain ⟨a, b, c, d, rfl⟩ := isOrder_fin4 hσ
  exact key a b c d hσ

/-- Example 2 as a knapsack: feasibility is the capacity constraint (Fact 1), the
best feasible set is `{1,2,3}` (time `6`, weight `16`), and the alternatives
`{1,4}` (time `6`, weight `15`) and `{3,4}` (time `5`, weight `13`) do worse. -/
theorem runInst6_knapsack :
    (∀ S, Feasible runInst6 S ↔ time runInst6 S ≤ 6) ∧
    (∀ S : Finset (Fin 4), time runInst6 S ≤ 6 → weight runInst6 S ≤ 16) ∧
    (time runInst6 {0, 1, 2} = 6 ∧ weight runInst6 {0, 1, 2} = 16) ∧
    (time runInst6 {0, 3} = 6 ∧ weight runInst6 {0, 3} = 15) ∧
    (time runInst6 {2, 3} = 5 ∧ weight runInst6 {2, 3} = 13) := by
  refine ⟨feasible_const_deadline_iff runInst6 (D := 6) (fun _ => rfl), ?_, ?_, ?_, ?_⟩
  · decide
  · decide
  · decide
  · decide

end Mwhed
