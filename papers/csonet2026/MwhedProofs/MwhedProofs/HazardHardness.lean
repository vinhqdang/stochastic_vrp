import MwhedProofs.Defs
import MwhedProofs.Core

/-!
# Constant hazard-arrival time (arrival reading): Proposition `prop:consth`

Formalisation of Proposition `prop:consth` (ITEM 2 of the theory-fixes note): for
instances with a *constant hazard-arrival time* `H`, i.e.

* `p i` even, `w i = p i / 2` and `d i = H + p i / 2` for every site `i`,

MWHED stays (weakly) NP-hard.

**Contents (namespace `Mwhed`).**

* `ArrivalConst I H` : the instance class (`p` even, `w = p/2`, `d = H + p/2`).
* `feasible_iff_mid`  : the feasibility characterisation (`eq:mid`).  For `S ≠ ∅`,
  `S` is feasible iff `∑_{i∈S} p i - (max_{i∈S} p i)/2 ≤ H`.  Formalised without
  subtraction or halving, as `2 * ∑_{i∈S} p i ≤ 2 * H + max_{i∈S} p i`
  (the same inequality, multiplied by two; `max` is `S.sup' hne I.p`).
* `hazardInst a ha`   : the instance on `Option ι` built from Partition data
  `a : ι → ℕ` (`ha : ∀ i, 0 < a i`): the dominant site `none` has
  `p = 2A+2`, `w = A+1`, and site `some i` has `p = 2 a i`, `w = a i`; with
  `H = 2A+1` and `d = H + p/2`, where `A = ∑ a`.  (`hazardInst_arrivalConst` shows it
  lies in the class `ArrivalConst _ (2A+1)`, so `w = p/2` at every site.)
* `hazard_partition_reduction` : `(∃ S ⊆ univ, 2 * ∑_{S} a = A) ↔ ∃ σ, IsOrder σ ∧
  3 * A + 2 ≤ 2 * W σ`, the many-one equivalence of the paper's claim
  "`W* ≥ 3A/2 + 1` iff the Partition instance is a yes-instance" (stated with
  `2 * W ≥ 3A + 2` to avoid halving).
* `hazard_partition_iff_isOPT` : the same in terms of the optimal value `v` (any `v`
  with `IsOPT _ v`; uses `isOPT_iff_max_feasible` of `Core.lean`).

**What is and is not formalised.**  Only the *equivalence of yes/no answers* of the
Partition question and the decision question "`W* ≥ 3A/2 + 1`?" on the produced
instance is machine-checked, together with the closed-form feasibility test.
Polynomial-time constructibility of the instance (evident: `2n + 2` numbers of
bit-length `O(log A)`), the complexity classes (NP-hardness) and the statement that
the problem is only *weakly* hard (Theorem 3, pseudo-polynomial DP) are **not**
formalised.  The paper's side remark "if `A` is odd, output a fixed no-instance" is
not needed: the equivalence holds for every positive vector `a` (for odd `A` both
sides are false), so no evenness hypothesis on `A` appears.  Assumption 1
(`p i ≤ d i`) holds for the produced instance (`hazardInst_indivFeasible`).
-/

set_option linter.unusedSectionVars false

namespace Mwhed

open Finset

/-! ## The instance class and the feasibility test -/

/-- The class of instances with constant hazard-arrival time `H` (arrival reading):
`p i` even, `w i = p i / 2`, `d i = H + p i / 2`. -/
def ArrivalConst {ι : Type*} (I : Inst ι) (H : ℕ) : Prop :=
  ∀ i, 2 ∣ I.p i ∧ I.w i = I.p i / 2 ∧ I.d i = H + I.p i / 2

section Mid

variable {ι : Type*} [DecidableEq ι] [Fintype ι] (I : Inst ι)

/-- **Equation `eq:mid`.**  For instances with `d i = H + p i / 2` and `p i` even, a
nonempty set `S` is feasible iff `∑_{S} p - (max_{S} p)/2 ≤ H`, i.e. (multiplying by two,
which avoids subtraction and halving)
`2 * ∑_{i∈S} p i ≤ 2 * H + max_{i∈S} p i`. -/
theorem feasible_iff_mid {H : ℕ} (heven : ∀ i, 2 ∣ I.p i)
    (hd : ∀ i, I.d i = H + I.p i / 2) (S : Finset ι) (hne : S.Nonempty) :
    Feasible I S ↔ 2 * time I S ≤ 2 * H + S.sup' hne I.p := by
  rw [feasible_iff_thr]
  obtain ⟨k, hkS, hkmax⟩ := exists_max_image S I.p hne
  have hsup : S.sup' hne I.p = I.p k := by
    apply le_antisymm
    · exact sup'_le _ _ hkmax
    · exact le_sup' _ hkS
  rw [hsup]
  have hk2 : 2 * (I.p k / 2) = I.p k := Nat.mul_div_cancel' (heven k)
  have hdk : ∀ i, I.d i ≤ I.d k ↔ I.p i / 2 ≤ I.p k / 2 := by
    intro i; rw [hd, hd]; omega
  constructor
  · intro h
    have h1 := h (I.d k)
    have hfilt : S.filter (fun i => I.d i ≤ I.d k) = S := by
      apply filter_true_of_mem
      intro i hi
      rw [hdk]
      exact Nat.div_le_div_right (hkmax i hi)
    rw [hfilt, hd k] at h1
    unfold time
    omega
  · intro h t
    by_cases hdkt : I.d k ≤ t
    · have hfilt : S.filter (fun i => I.d i ≤ t) = S := by
        apply filter_true_of_mem
        intro i hi
        have : I.d i ≤ I.d k := by
          rw [hdk]; exact Nat.div_le_div_right (hkmax i hi)
        omega
      rw [hfilt]
      have h2 : ∑ i ∈ S, I.p i ≤ I.d k := by
        rw [hd k]
        unfold time at h
        omega
      omega
    · -- sites with deadline `≤ t < d k` are strictly smaller than `k`
      rcases (S.filter (fun i => I.d i ≤ t)).eq_empty_or_nonempty with hT | ⟨i1, hi1⟩
      · rw [hT]; simp
      · have hsub : S.filter (fun i => I.d i ≤ t) ⊆ S.erase k := by
          intro i hi
          rw [mem_filter] at hi
          rw [mem_erase]
          refine ⟨?_, hi.1⟩
          rintro rfl
          exact hdkt hi.2
        have hle := sum_le_sum_of_subset (f := I.p) hsub
        have herase : ∑ i ∈ S.erase k, I.p i + I.p k = ∑ i ∈ S, I.p i := by
          rw [add_comm]; exact add_sum_erase S I.p hkS
        have hH : H ≤ t := by
          have := (mem_filter.1 hi1).2
          rw [hd] at this
          omega
        unfold time at h
        omega

end Mid

/-! ## The reduction from Partition -/

section Reduction

variable {ι : Type*} [DecidableEq ι] [Fintype ι]

/-- The MWHED instance of Proposition `prop:consth`, on the sites `Option ι`
(`none` is the dominant site `0`): with `A = ∑ a`,
`p none = 2A+2`, `w none = A+1`, `p (some i) = 2 a i`, `w (some i) = a i`,
`H = 2A+1` and `d = H + p/2`. -/
def hazardInst (a : ι → ℕ) (ha : ∀ i, 0 < a i) : Inst (Option ι) where
  p := fun x => match x with
    | none => 2 * (∑ j, a j) + 2
    | some i => 2 * a i
  d := fun x => (2 * (∑ j, a j) + 1) + (match x with
    | none => 2 * (∑ j, a j) + 2
    | some i => 2 * a i) / 2
  w := fun x => match x with
    | none => (∑ j, a j) + 1
    | some i => a i
  p_pos := fun x => by
    cases x with
    | none => simp
    | some i => have := ha i; simp; omega

variable (a : ι → ℕ) (ha : ∀ i, 0 < a i)

/-- The constructed instance lies in the class `ArrivalConst` with `H = 2A + 1`: all
`p i` are even, `w i = p i / 2` for *every* site (including the dominant one), and
`d i = H + p i / 2`. -/
theorem hazardInst_arrivalConst : ArrivalConst (hazardInst a ha) (2 * (∑ j, a j) + 1) := by
  intro x
  cases x with
  | none => refine ⟨⟨∑ j, a j + 1, by simp [hazardInst]; ring⟩, ?_, rfl⟩; simp [hazardInst]
  | some i => refine ⟨⟨a i, by simp [hazardInst]⟩, ?_, rfl⟩; simp [hazardInst]

/-- Assumption 1 (`p i ≤ d i`) holds for the constructed instance. -/
theorem hazardInst_indivFeasible : IndivFeasible (hazardInst a ha) := by
  intro x
  cases x with
  | none => simp [hazardInst]; omega
  | some i =>
    have : a i ≤ ∑ j, a j := single_le_sum (f := a) (fun _ _ => Nat.zero_le _) (mem_univ i)
    simp [hazardInst]; omega

/-- Sums over a finset of `Option ι` split into the `none` part and the part over the
`some` sites of the finset. -/
theorem sum_option_split (S : Finset (Option ι)) (f : Option ι → ℕ) :
    ∑ x ∈ S, f x = (if none ∈ S then f none else 0) +
      ∑ i ∈ univ.filter (fun i => some i ∈ S), f (some i) := by
  have h1 : ∑ x ∈ S, f x = ∑ x, (if x ∈ S then f x else 0) := by
    rw [sum_ite_mem, univ_inter]
  rw [h1, Fintype.sum_option, sum_filter]

/-- The `some`-part of `S`. -/
private def J (S : Finset (Option ι)) : Finset ι := univ.filter (fun i => some i ∈ S)

private theorem weight_split (S : Finset (Option ι)) :
    weight (hazardInst a ha) S =
      (if none ∈ S then (∑ j, a j) + 1 else 0) + ∑ i ∈ J S, a i := by
  unfold weight
  rw [sum_option_split]
  rfl

private theorem time_split (S : Finset (Option ι)) :
    time (hazardInst a ha) S =
      (if none ∈ S then 2 * (∑ j, a j) + 2 else 0) + 2 * ∑ i ∈ J S, a i := by
  unfold time
  have h2 : 2 * ∑ i ∈ J S, a i = ∑ i ∈ J S, 2 * a i := mul_sum _ _ _
  rw [sum_option_split, h2]
  rfl

private theorem sup_none (S : Finset (Option ι)) (hS : none ∈ S) (hne : S.Nonempty) :
    S.sup' hne (hazardInst a ha).p = 2 * (∑ j, a j) + 2 := by
  apply le_antisymm
  · apply sup'_le
    intro x _
    cases x with
    | none => exact le_rfl
    | some i =>
      have : a i ≤ ∑ j, a j := single_le_sum (f := a) (fun _ _ => Nat.zero_le _) (mem_univ i)
      show 2 * a i ≤ _
      omega
  · exact le_sup' (hazardInst a ha).p hS

/-- Feasibility of a set containing the dominant site: `2 ∑_{J} a ≤ A`. -/
private theorem feasible_none_iff (S : Finset (Option ι)) (hS : none ∈ S) :
    Feasible (hazardInst a ha) S ↔ 2 * ∑ i ∈ J S, a i ≤ ∑ j, a j := by
  have hne : S.Nonempty := ⟨none, hS⟩
  rw [feasible_iff_mid (hazardInst a ha) (H := 2 * (∑ j, a j) + 1)
    (fun x => (hazardInst_arrivalConst a ha x).1) (fun x => (hazardInst_arrivalConst a ha x).2.2)
    S hne, time_split a ha, sup_none a ha S hS hne]
  simp only [hS, ↓reduceIte]
  omega

/-- **Proposition `prop:consth`, reduction correctness.**  For positive integers `a`
(with sum `A`), Partition has a solution iff the constructed instance (constant
hazard-arrival time `H = 2A+1`, `w = p/2`) has a dispatch order of on-time weight at least
`3A/2 + 1`, stated as `3 * A + 2 ≤ 2 * W σ`. -/
theorem hazard_partition_reduction :
    (∃ S : Finset ι, 2 * ∑ i ∈ S, a i = ∑ i, a i) ↔
      ∃ σ : List (Option ι), IsOrder σ ∧
        3 * (∑ i, a i) + 2 ≤ 2 * W (hazardInst a ha) σ := by
  classical
  have hfeas : (∃ σ : List (Option ι), IsOrder σ ∧
        3 * (∑ i, a i) + 2 ≤ 2 * W (hazardInst a ha) σ) ↔
      ∃ S, Feasible (hazardInst a ha) S ∧
        3 * (∑ i, a i) + 2 ≤ 2 * weight (hazardInst a ha) S := by
    constructor
    · rintro ⟨σ, hσ, h⟩
      refine ⟨onTimeSet (hazardInst a ha) σ, feasible_onTimeSet _ hσ, ?_⟩
      rw [← W_eq_weight_onTimeSet _ hσ.1]; exact h
    · rintro ⟨S, hS, h⟩
      obtain ⟨σ, hσ, hsub⟩ := exists_order_of_feasible _ hS
      refine ⟨σ, hσ, h.trans ?_⟩
      rw [W_eq_weight_onTimeSet _ hσ.1]
      exact Nat.mul_le_mul_left _ (sum_le_sum_of_subset hsub)
  rw [hfeas]
  constructor
  · rintro ⟨I, hI⟩
    refine ⟨insert none (I.image some), ?_, ?_⟩
    · have hJ : J (insert none (I.image some)) = I := by
        ext i; simp [J]
      rw [feasible_none_iff a ha _ (mem_insert_self _ _), hJ]
      omega
    · have hJ : J (insert none (I.image some)) = I := by
        ext i; simp [J]
      rw [weight_split a ha, hJ]
      simp only [mem_insert_self, ↓reduceIte]
      omega
  · rintro ⟨S, hS, hw⟩
    by_cases hnone : none ∈ S
    · refine ⟨J S, ?_⟩
      have h1 := (feasible_none_iff a ha S hnone).1 hS
      rw [weight_split a ha] at hw
      simp only [hnone, ↓reduceIte] at hw
      omega
    · exfalso
      rw [weight_split a ha] at hw
      simp only [hnone, ↓reduceIte] at hw
      have : ∑ i ∈ J S, a i ≤ ∑ i, a i := sum_le_sum_of_subset (subset_univ _)
      omega

/-- The decision question in terms of the optimal value: for any `v` with `IsOPT _ v`,
Partition is a yes-instance iff `W* = v ≥ 3A/2 + 1` (i.e. `3 * A + 2 ≤ 2 * v`). -/
theorem hazard_partition_iff_isOPT {v : ℕ} (hv : IsOPT (hazardInst a ha) v) :
    (∃ S : Finset ι, 2 * ∑ i ∈ S, a i = ∑ i, a i) ↔ 3 * (∑ i, a i) + 2 ≤ 2 * v := by
  rw [hazard_partition_reduction a ha]
  obtain ⟨⟨S0, hS0, hv0⟩, hmax⟩ := (isOPT_iff_max_feasible _ v).1 hv
  constructor
  · rintro ⟨σ, hσ, h⟩
    have : W (hazardInst a ha) σ ≤ v := hv.2 σ hσ
    omega
  · intro h
    obtain ⟨σ, hσ, hsub⟩ := exists_order_of_feasible _ hS0
    refine ⟨σ, hσ, ?_⟩
    rw [W_eq_weight_onTimeSet _ hσ.1]
    have h' : weight (hazardInst a ha) S0 ≤
        weight (hazardInst a ha) (onTimeSet (hazardInst a ha) σ) :=
      sum_le_sum_of_subset hsub
    omega

end Reduction

end Mwhed
