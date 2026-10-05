import MwhedProofs.Defs
import MwhedProofs.Core
import MwhedProofs.Hardness

set_option linter.unusedSectionVars false

/-!
# Classification by homogeneity (Theorem 9 of the paper), formalisable part

For `T ⊆ {p, w, d}` let `C_T` be the class of MWHED instances in which each data
vector named in `T` is constant (`InClass T`).

**(b) Hardness half** (`classification_b_hardness`, `partitionInst_mem_class`):
if `T ⊆ {d}` then `C_T ⊇ C_{d}` contains every instance produced by the
reduction of Theorem 2 (`Hardness.lean`; those instances have constant deadlines
and `p = w`), so the Partition many-one equivalence of Theorem 2 is a many-one
reduction *into* `C_T`.  As in `Hardness.lean`, polynomial-time constructibility
and the complexity-class vocabulary ("NP-hard", "weakly NP-hard") are *not*
formalised, and "weakly" (the pseudo-polynomial dynamic program of Theorem 3)
is not part of this file.

**(a) Equal weights** (`weight_eq_mul_card`, `isOPT_const_weight_iff`,
`isOPT_const_weight_exists_card`): if every weight equals `c` then
`weight S = c * card S`, so maximising on-time weight is maximising the *number*
of on-time sites; `IsOPT I (c * k)` iff `k` is the maximum cardinality of a
feasible set.  This is the formal content of the paper's remark that the case
reduces to `1 || ∑ U_j`.  That this problem is solved in `O(n log n)` time by
Moore's algorithm is a classical result and is **not** formalised here (nor is
any running-time claim).  The other half of (a), `p ∈ T` (all dispatch times
equal), is exactly Theorem 5 (equal dispatch cost; matroid greedy), which is
proved in its own file and deliberately not redone here; its running time is
likewise not formalised.

`isOPT_const_weight_iff` relies on `isOPT_iff_max_feasible` from `Core.lean`.
-/

namespace Mwhed

/-- The three data vectors of an MWHED instance. -/
inductive Het
  | p | w | d
  deriving DecidableEq

/-- The class `C_T`: every data vector named in `T` is constant. -/
def Inst.InClass {ι : Type*} (T : Set Het) (I : Inst ι) : Prop :=
  (Het.p ∈ T → I.ConstDispatch) ∧ (Het.w ∈ T → I.ConstWeight) ∧ (Het.d ∈ T → I.ConstDeadline)

/-- `C_∅` is the class of all instances. -/
theorem inClass_empty {ι : Type*} (I : Inst ι) : I.InClass ∅ :=
  ⟨fun h => h.elim, fun h => h.elim, fun h => h.elim⟩

/-- Fewer constraints give a bigger class: `C_{T'} ⊆ C_T` for `T ⊆ T'`. -/
theorem Inst.InClass.mono {ι : Type*} {T T' : Set Het} (hT : T ⊆ T') {I : Inst ι}
    (h : I.InClass T') : I.InClass T :=
  ⟨fun hp => h.1 (hT hp), fun hw => h.2.1 (hT hw), fun hd => h.2.2 (hT hd)⟩

/-- An instance with constant deadline lies in `C_T` for every `T ⊆ {d}`. -/
theorem inClass_of_constDeadline {ι : Type*} {T : Set Het} (hT : T ⊆ {Het.d}) {I : Inst ι}
    (h : I.ConstDeadline) : I.InClass T := by
  refine ⟨fun hp => ?_, fun hw => ?_, fun _ => h⟩
  · exact absurd (hT hp) (by simp)
  · exact absurd (hT hw) (by simp)

/-- The reduction instances of Theorem 2 lie in `C_T` for every `T ⊆ {d}`. -/
theorem partitionInst_mem_class {n : ℕ} (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) {T : Set Het}
    (hT : T ⊆ {Het.d}) : (partitionInst a ha).InClass T :=
  inClass_of_constDeadline hT (partitionInst_constDeadline a ha)

/-- **Theorem 9(b)**, hardness half.  For every `T ⊆ {d}` the Partition problem
many-one reduces to MWHED restricted to `C_T`: every positive vector `a` with
even sum yields an instance in `C_T` (with in addition `p = w`) such that
Partition has a solution iff some dispatch order has on-time weight `≥ A/2`
(Theorem 2).  Polynomial-time constructibility and the NP-hardness conclusion
itself are not formalised. -/
theorem classification_b_hardness {T : Set Het} (hT : T ⊆ {Het.d}) {n : ℕ}
    (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) (heven : 2 ∣ ∑ i, a i) :
    ∃ I : Inst (Fin n), I.InClass T ∧ I.WeightEqDispatch ∧
      ((∃ S : Finset (Fin n), 2 * ∑ i ∈ S, a i = ∑ i, a i) ↔
        ∃ σ : List (Fin n), IsOrder σ ∧ (∑ i, a i) / 2 ≤ W I σ) :=
  ⟨partitionInst a ha, partitionInst_mem_class a ha hT,
    partitionInst_weightEqDispatch a ha, partition_reduction a ha heven⟩

/-! ### Equal weights -/

section EqualWeights

variable {ι : Type*} [DecidableEq ι] (I : Inst ι)

/-- With all weights equal to `c`, the weight of a set is `c` times its cardinality. -/
theorem weight_eq_mul_card {c : ℕ} (hw : ∀ i, I.w i = c) (S : Finset ι) :
    weight I S = c * S.card := by
  simp [weight, hw, Finset.sum_const, mul_comm]

variable [Fintype ι]

/-- **Theorem 9(a), `w ∈ T`**: with all weights equal to `c > 0`, `W* = c * k`
iff `k` is the maximum cardinality of a feasible set (maximising on-time weight is
maximising the number of on-time sites, i.e. the `1 || ∑ U_j` problem). -/
theorem isOPT_const_weight_iff {c : ℕ} (hc : 0 < c) (hw : ∀ i, I.w i = c) (k : ℕ) :
    IsOPT I (c * k) ↔
      (∃ S, Feasible I S ∧ S.card = k) ∧ ∀ S, Feasible I S → S.card ≤ k := by
  rw [isOPT_iff_max_feasible]
  simp only [weight_eq_mul_card I hw]
  constructor
  · rintro ⟨⟨S, hS, h⟩, hle⟩
    exact ⟨⟨S, hS, Nat.eq_of_mul_eq_mul_left hc h⟩,
      fun S' hS' => Nat.le_of_mul_le_mul_left (hle S' hS') hc⟩
  · rintro ⟨⟨S, hS, h⟩, hle⟩
    exact ⟨⟨S, hS, by rw [h]⟩, fun S' hS' => Nat.mul_le_mul_left c (hle S' hS')⟩

/-- Every optimal value of an equal-weight instance is `c * k` for the maximum
cardinality `k` of a feasible set. -/
theorem isOPT_const_weight_exists_card {c : ℕ} (hc : 0 < c) (hw : ∀ i, I.w i = c) {v : ℕ}
    (hv : IsOPT I v) :
    ∃ k, v = c * k ∧ (∃ S, Feasible I S ∧ S.card = k) ∧ ∀ S, Feasible I S → S.card ≤ k := by
  obtain ⟨⟨S, hS, hSv⟩, -⟩ := (isOPT_iff_max_feasible I v).1 hv
  rw [weight_eq_mul_card I hw] at hSv
  subst hSv
  exact ⟨S.card, rfl, (isOPT_const_weight_iff I hc hw S.card).1 hv⟩

end EqualWeights

end Mwhed
