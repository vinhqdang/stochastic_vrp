import MwhedProofs.Defs

set_option linter.unusedSectionVars false

/-!
# Hardness (Section 4.1 of the paper): Fact 1 and Theorem 2

* `feasible_const_deadline_iff` : **Fact 1** (feasibility under a common
  deadline): if `d i = D` for every site then `Feasible I S ↔ time I S ≤ D`.
* `partitionInst`               : the instance built by the reduction of
  **Theorem 2**, `p = w = a`, `d = A/2` (constant).
* `partition_reduction`         : many-one equivalence
  `(∃ S, 2 * ∑_{i∈S} a i = A) ↔ ∃ σ, IsOrder σ ∧ A/2 ≤ W σ`.
* `partition_W_le`, `partition_isOPT_le`, `partition_isOPT_ge_iff` :
  `W* ≤ A/2` always, so `W* ≥ A/2 ↔ W* = A/2`.
* `partition_iff_isOPT`         : Partition is a yes-instance iff
  `IsOPT (partitionInst a) (A/2)`.
* `partitionInst_constDeadline`, `partitionInst_weightEqDispatch`,
  `partitionInst_indivFeasible` : the constructed instance lies in the classes
  "`d` constant" and "`p = w`" and satisfies Assumption 1 (when every
  `a i ≤ A/2`, the paper's WLOG).

**What is and is not formalised.**  The polynomial-time constructibility of
the reduction is evident from the definition of `partitionInst` (a copy of the
input vector together with the single number `(∑ a)/2`) and is not formalised;
neither NP-membership of the decision problem nor the complexity classes
`NP`, `NP-hard`, `NP-complete` are formalised (Mathlib has no practical
framework for them).  What is machine-checked is the *many-one equivalence*
of the Partition decision problem with the MWHED decision problem
"is `W* ≥ A/2`?" on the produced instance, which is the mathematical content
of Theorem 2.  The paper's side remark "if some `a i > A/2` output a fixed
no-instance" is not needed for the equivalence (it is true for every
positive vector with even sum); the hypothesis `a i ≤ A/2` is used only to
establish Assumption 1 (`partitionInst_indivFeasible`).

This file depends only on `Defs.lean`; in particular it does not use the
(separately proved) `Core.lean` lemmas.
-/

namespace Mwhed

/-! ### Homogeneity classes (used here and in `Classification.lean`) -/

/-- Class "`d` constant": all hazard deadlines coincide. -/
def Inst.ConstDeadline {ι : Type*} (I : Inst ι) : Prop := ∃ D, ∀ i, I.d i = D

/-- Class "`p` constant": all dispatch times coincide. -/
def Inst.ConstDispatch {ι : Type*} (I : Inst ι) : Prop := ∃ q, ∀ i, I.p i = q

/-- Class "`w` constant": all criticality weights coincide. -/
def Inst.ConstWeight {ι : Type*} (I : Inst ι) : Prop := ∃ c, ∀ i, I.w i = c

/-- Class "`p = w`": weight equals dispatch time at every site. -/
def Inst.WeightEqDispatch {ι : Type*} (I : Inst ι) : Prop := ∀ i, I.w i = I.p i

section Completion

variable {ι : Type*} [DecidableEq ι] (I : Inst ι)

/-- Recursion equation for `completion` on a cons. -/
theorem completion_cons (a : ι) (τ : List ι) (i : ι) :
    completion I (a :: τ) i = if i = a then I.p a else I.p a + completion I τ i := by
  unfold completion
  by_cases h : i = a
  · subst h; simp
  · have h' : a ≠ i := fun e => h e.symm
    simp [h, h']
    ring

/-- A site of `L` completes no later than the total time of `L`. -/
theorem completion_le_sum {L : List ι} {i : ι} (hi : i ∈ L) :
    completion I L i ≤ (L.map I.p).sum := by
  induction L with
  | nil => simp at hi
  | cons a τ ih =>
    rw [completion_cons]
    by_cases h : i = a
    · simp [h]
    · have hτ : i ∈ τ := by simpa [h] using hi
      have := ih hτ
      simp [h]
      omega

/-- Appending sites after `L` does not change the completion time of a site of `L`. -/
theorem completion_append_left {L : List ι} (M : List ι) {i : ι} (hi : i ∈ L) :
    completion I (L ++ M) i = completion I L i := by
  induction L with
  | nil => simp at hi
  | cons a τ ih =>
    rw [List.cons_append, completion_cons, completion_cons]
    by_cases h : i = a
    · simp [h]
    · have hτ : i ∈ τ := by simpa [h] using hi
      simp [h, ih hτ]

variable [Fintype ι]

/-- For every set `S` there is a dispatch order in which every site of `S` completes
by time `time I S` (serve `S` first). -/
theorem exists_order_completion_le (S : Finset ι) :
    ∃ σ : List ι, IsOrder σ ∧ ∀ i ∈ S, completion I σ i ≤ time I S := by
  classical
  refine ⟨S.toList ++ (Finset.univ.toList.filter (fun x => x ∉ S)), ⟨?_, ?_⟩, ?_⟩
  · rw [List.nodup_append]
    refine ⟨Finset.nodup_toList _, (Finset.nodup_toList _).filter _, ?_⟩
    intro x hx y hy hxy
    subst hxy
    simp at hx hy
    exact hy hx
  · intro i
    by_cases h : i ∈ S
    · simp [h]
    · simp [h]
  · intro i hi
    have hi' : i ∈ S.toList := by simpa using hi
    rw [completion_append_left I _ hi']
    have := completion_le_sum I hi'
    have hs : (S.toList.map I.p).sum = time I S := by
      simp [time]
    omega

/-- If `S` is nonempty and all its members occur in `σ`, some member of `S`
completes no earlier than the total time of `S`. -/
theorem exists_completion_ge (σ : List ι) (S : Finset ι) (hne : S.Nonempty)
    (hS : ∀ i ∈ S, i ∈ σ) : ∃ j ∈ S, time I S ≤ completion I σ j := by
  induction σ generalizing S with
  | nil =>
    obtain ⟨i, hi⟩ := hne
    simpa using hS i hi
  | cons a τ ih =>
    by_cases ha : a ∈ S
    · have hsum : time I S = I.p a + time I (S.erase a) := by
        unfold time; rw [Finset.add_sum_erase _ _ ha]
      by_cases hne' : (S.erase a).Nonempty
      · obtain ⟨j, hj, hjle⟩ := ih (S.erase a) hne' (by
          intro i hi
          have hia := Finset.ne_of_mem_erase hi
          have := hS i (Finset.mem_of_mem_erase hi)
          simpa [hia] using this)
        have hja : j ≠ a := Finset.ne_of_mem_erase hj
        refine ⟨j, Finset.mem_of_mem_erase hj, ?_⟩
        rw [completion_cons]
        simp [hja]
        omega
      · have : S.erase a = ∅ := Finset.not_nonempty_iff_eq_empty.mp hne'
        refine ⟨a, ha, ?_⟩
        rw [completion_cons, hsum, this]
        simp [time]
    · obtain ⟨j, hj, hjle⟩ := ih S hne (by
        intro i hi
        have hia : i ≠ a := fun e => ha (e ▸ hi)
        have := hS i hi
        simpa [hia] using this)
      have hja : j ≠ a := fun e => ha (e ▸ hj)
      refine ⟨j, hj, ?_⟩
      rw [completion_cons]
      simp [hja]
      omega

/-- **Fact 1** of the paper (feasibility under a common deadline).  If
`d i = D` for every site, a set `S` can be served entirely on time iff its total
dispatch time is at most `D`. -/
theorem feasible_const_deadline_iff {D : ℕ} (hD : ∀ i, I.d i = D) (S : Finset ι) :
    Feasible I S ↔ time I S ≤ D := by
  constructor
  · rintro ⟨σ, ⟨_, hmem⟩, hS⟩
    rcases S.eq_empty_or_nonempty with rfl | hne
    · simp [time]
    · obtain ⟨j, hj, hle⟩ := exists_completion_ge I σ S hne (fun i _ => hmem i)
      have := hS j hj
      rw [hD] at this
      omega
  · intro h
    obtain ⟨σ, hσ, hc⟩ := exists_order_completion_le I S
    exact ⟨σ, hσ, fun i hi => by rw [hD]; exact (hc i hi).trans h⟩

end Completion

/-! ### The on-time weight lemmas for the reduction instances -/

section OnTime

variable {ι : Type*} [DecidableEq ι] (J : Inst ι)

/-- Upper bound for instances with constant deadline `D` and `w = p`: starting
at time `t`, the on-time weight of any list is at most `D - t`. -/
theorem onTimeW_le_of_const {D : ℕ} (hD : ∀ i, J.d i = D) (hw : ∀ i, J.w i = J.p i)
    (σ : List ι) : ∀ t, onTimeW J t σ ≤ D - t := by
  induction σ with
  | nil => intro t; simp [onTimeW]
  | cons i σ ih =>
    intro t
    have := ih (t + J.p i)
    simp only [onTimeW, hD, hw]
    split_ifs with h
    · omega
    · omega

/-- Lower bound: if the total time of a prefix `L` fits before the common
deadline, every site of `L` is on time and contributes its weight. -/
theorem onTimeW_append_ge {D : ℕ} (hD : ∀ i, J.d i = D) (hw : ∀ i, J.w i = J.p i)
    (L M : List ι) : ∀ t, t + (L.map J.p).sum ≤ D → (L.map J.p).sum ≤ onTimeW J t (L ++ M) := by
  induction L with
  | nil => intro t _; simp
  | cons i L ih =>
    intro t ht
    simp only [List.map_cons, List.sum_cons] at ht
    have := ih (t + J.p i) (by omega)
    simp only [List.cons_append, onTimeW, hD, hw, List.map_cons, List.sum_cons]
    have hc : t + J.p i ≤ D := by omega
    simp only [hc, if_true]
    omega

/-- On a duplicate-free list, the on-time weight is the weight of some subset of its sites. -/
theorem onTimeW_eq_sum_subset {σ : List ι} (hσ : σ.Nodup) :
    ∀ t, ∃ S : Finset ι, S ⊆ σ.toFinset ∧ onTimeW J t σ = ∑ i ∈ S, J.w i := by
  induction σ with
  | nil => intro t; exact ⟨∅, by simp, by simp [onTimeW]⟩
  | cons a τ ih =>
    intro t
    rw [List.nodup_cons] at hσ
    obtain ⟨S', hS', hv⟩ := ih hσ.2 (t + J.p a)
    by_cases h : t + J.p a ≤ J.d a
    · have ha : a ∉ S' := fun e => hσ.1 (by simpa using hS' e)
      refine ⟨insert a S', ?_, ?_⟩
      · rw [List.toFinset_cons]; exact Finset.insert_subset_insert _ hS'
      · rw [Finset.sum_insert ha]; simp [onTimeW, h, hv]
    · refine ⟨S', ?_, ?_⟩
      · rw [List.toFinset_cons]; exact hS'.trans (Finset.subset_insert _ _)
      · simp [onTimeW, h, hv]

end OnTime

/-! ### The reduction of Theorem 2 -/

section Reduction

variable {n : ℕ}

/-- The MWHED instance of the reduction in Theorem 2: `p = w = a` and the
constant deadline `d = A/2` where `A = ∑ a`.  Polynomial-time constructibility
is evident (a copy of `a` and one extra number). -/
def partitionInst (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) : Inst (Fin n) where
  p := a
  d := fun _ => (∑ i, a i) / 2
  w := a
  p_pos := ha

/-- The constructed instance has a constant deadline: it lies in the class `C_{d}`. -/
theorem partitionInst_constDeadline (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) :
    (partitionInst a ha).ConstDeadline := ⟨(∑ i, a i) / 2, fun _ => rfl⟩

/-- The constructed instance has `w = p`. -/
theorem partitionInst_weightEqDispatch (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) :
    (partitionInst a ha).WeightEqDispatch := fun _ => rfl

/-- Assumption 1 (`p i ≤ d i`) holds for the constructed instance when every
`a i ≤ A/2` -- the paper's WLOG (otherwise Partition is trivially a no-instance). -/
theorem partitionInst_indivFeasible (a : Fin n → ℕ) (ha : ∀ i, 0 < a i)
    (hle : ∀ i, a i ≤ (∑ j, a j) / 2) : IndivFeasible (partitionInst a ha) :=
  fun i => hle i

/-- `W σ ≤ A/2` for every dispatch list `σ`: no dispatch order can exceed `A/2`. -/
theorem partition_W_le (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) (σ : List (Fin n)) :
    W (partitionInst a ha) σ ≤ (∑ i, a i) / 2 := by
  have := onTimeW_le_of_const (partitionInst a ha) (D := (∑ i, a i) / 2)
    (fun _ => rfl) (fun _ => rfl) σ 0
  simpa [W] using this

/-- **Theorem 2** (reduction correctness).  For positive integers `a` with even
sum `A`, Partition has a solution iff the constructed MWHED instance has a
dispatch order of on-time weight at least `A/2`. -/
theorem partition_reduction (a : Fin n → ℕ) (ha : ∀ i, 0 < a i)
    (heven : 2 ∣ ∑ i, a i) :
    (∃ S : Finset (Fin n), 2 * ∑ i ∈ S, a i = ∑ i, a i) ↔
      ∃ σ : List (Fin n), IsOrder σ ∧ (∑ i, a i) / 2 ≤ W (partitionInst a ha) σ := by
  constructor
  · rintro ⟨S, hS⟩
    set J := partitionInst a ha
    refine ⟨S.toList ++ (Finset.univ.toList.filter (fun x => x ∉ S)), ⟨?_, ?_⟩, ?_⟩
    · rw [List.nodup_append]
      refine ⟨Finset.nodup_toList _, (Finset.nodup_toList _).filter _, ?_⟩
      intro x hx y hy hxy
      subst hxy
      simp at hx hy
      exact hy hx
    · intro i
      by_cases h : i ∈ S
      · simp [h]
      · simp [h]
    · have hs : (S.toList.map J.p).sum = ∑ i ∈ S, a i := by
        simp [J, partitionInst]
      have := onTimeW_append_ge J (D := (∑ i, a i) / 2) (fun _ => rfl) (fun _ => rfl)
        S.toList (Finset.univ.toList.filter (fun x => x ∉ S)) 0 (by omega)
      simp only [W]
      omega
  · rintro ⟨σ, ⟨hnd, _⟩, hW⟩
    have hle := partition_W_le a ha σ
    obtain ⟨S, -, hS⟩ := onTimeW_eq_sum_subset (partitionInst a ha) hnd 0
    refine ⟨S, ?_⟩
    have hS' : W (partitionInst a ha) σ = ∑ i ∈ S, a i := hS
    omega

/-- Every optimal value of the constructed instance is at most `A/2`. -/
theorem partition_isOPT_le (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) {v : ℕ}
    (hv : IsOPT (partitionInst a ha) v) : v ≤ (∑ i, a i) / 2 := by
  obtain ⟨⟨σ, _, hσ⟩, _⟩ := hv
  rw [← hσ]; exact partition_W_le a ha σ

/-- Hence `W* ≥ A/2` iff `W* = A/2`. -/
theorem partition_isOPT_ge_iff (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) {v : ℕ}
    (hv : IsOPT (partitionInst a ha) v) : (∑ i, a i) / 2 ≤ v ↔ v = (∑ i, a i) / 2 :=
  ⟨fun h => le_antisymm (partition_isOPT_le a ha hv) h, fun h => h.ge⟩

/-- Partition is a yes-instance iff the optimal on-time weight of the constructed
instance equals `A/2` (i.e. `W* ≥ A/2`, the decision question of Theorem 2). -/
theorem partition_iff_isOPT (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) (heven : 2 ∣ ∑ i, a i) :
    (∃ S : Finset (Fin n), 2 * ∑ i ∈ S, a i = ∑ i, a i) ↔
      IsOPT (partitionInst a ha) ((∑ i, a i) / 2) := by
  rw [partition_reduction a ha heven]
  constructor
  · rintro ⟨σ, hσ, hW⟩
    have hle := partition_W_le a ha σ
    exact ⟨⟨σ, hσ, le_antisymm hle hW⟩, fun τ _ => partition_W_le a ha τ⟩
  · rintro ⟨⟨σ, hσ, hW⟩, _⟩
    exact ⟨σ, hσ, hW.ge⟩

end Reduction

end Mwhed
