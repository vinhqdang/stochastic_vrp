import MwhedProofs.Defs

/-!
# Core structural results (Section 3 of the paper)

* `W_eq_weight_onTimeSet`      : `W σ` is the weight of the on-time set.
* `lemma1_edd`                 : Lemma 1 (earliest-deadline-first feasibility).
* `isOPT_iff_max_feasible`     : `W*` is the maximum weight of a feasible set
  (the reduction of the optimisation problem to choosing a subset `S`).
-/

namespace Mwhed

variable {ι : Type*} [DecidableEq ι] [Fintype ι] (I : Inst ι)

theorem W_eq_weight_onTimeSet {σ : List ι} (hσ : σ.Nodup) :
    W I σ = weight I (onTimeSet I σ) := by
  sorry

/-- **Lemma 1** of the paper. -/
theorem lemma1_edd (S : Finset ι) :
    Feasible I S ↔ AllOnTime I 0 (edd I S) := by
  sorry

theorem isOPT_iff_max_feasible (v : ℕ) :
    IsOPT I v ↔
      (∃ S, Feasible I S ∧ weight I S = v) ∧
        ∀ S, Feasible I S → weight I S ≤ v := by
  sorry

theorem exists_isOPT : ∃ v, IsOPT I v := by
  sorry

end Mwhed
