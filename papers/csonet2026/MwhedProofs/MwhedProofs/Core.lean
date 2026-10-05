import MwhedProofs.Defs

/-!
# Core structural results (Section 3 of the paper)

* `W_eq_weight_onTimeSet`      : `W σ` is the weight of the on-time set.
* `lemma1_edd`                 : Lemma 1 (earliest-deadline-first feasibility).
* `isOPT_iff_max_feasible`     : `W*` is the maximum weight of a feasible set
  (the reduction of the optimisation problem to choosing a subset `S`).
* `exists_isOPT`               : an optimal value exists.

Besides the statements above, this file proves the *threshold characterisation*
of feasibility (`feasible_iff_thr`): `S` is feasible iff for every `t` the sites
of `S` with deadline `≤ t` have total dispatch time `≤ t`.  It is the route
taken for Lemma 1 (instead of the two exchange arguments of the paper) and it is
what makes the dynamic program of `Dp.lean` tractable in the presence of ties
between deadlines.
-/

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

namespace Mwhed

section Aux

variable {ι : Type*} [DecidableEq ι] (I : Inst ι)

/-! ### Completion times -/

theorem completion_cons_self (j : ι) (τ : List ι) :
    completion I (j :: τ) j = I.p j := by
  simp [completion, List.takeWhile_cons]

theorem completion_cons_ne {j i : ι} (τ : List ι) (h : i ≠ j) :
    completion I (j :: τ) i = I.p j + completion I τ i := by
  simp [completion, List.takeWhile_cons, h.symm]
  ring

/-- "On time, but only the sites of `S` are required to be": the recursive
form of `∀ i ∈ S, completion ≤ d`, with service starting at time `t`. -/
def OnTimeOn (S : Finset ι) : ℕ → List ι → Prop
  | _, [] => True
  | t, i :: σ => (i ∈ S → t + I.p i ≤ I.d i) ∧ OnTimeOn S (t + I.p i) σ

theorem onTimeOn_iff_completion (S : Finset ι) :
    ∀ (σ : List ι) (t : ℕ), σ.Nodup →
      (OnTimeOn I S t σ ↔ ∀ i ∈ σ, i ∈ S → t + completion I σ i ≤ I.d i) := by
  intro σ
  induction σ with
  | nil => intro t _; simp [OnTimeOn]
  | cons j τ ih =>
    intro t hnd
    rw [List.nodup_cons] at hnd
    obtain ⟨hj, hτ⟩ := hnd
    rw [OnTimeOn, ih _ hτ]
    constructor
    · rintro ⟨h1, h2⟩ i hi hiS
      rcases List.mem_cons.1 hi with rfl | hi'
      · rw [completion_cons_self]; exact h1 hiS
      · have : i ≠ j := fun h => hj (h ▸ hi')
        rw [completion_cons_ne I τ this]
        have := h2 i hi' hiS
        omega
    · intro h
      refine ⟨fun hjS => ?_, fun i hi hiS => ?_⟩
      · have := h j (List.mem_cons_self) hjS
        rwa [completion_cons_self] at this
      · have hne : i ≠ j := fun h => hj (h ▸ hi)
        have := h i (List.mem_cons_of_mem _ hi) hiS
        rw [completion_cons_ne I τ hne] at this
        omega

theorem allOnTime_imp_onTimeOn (S : Finset ι) :
    ∀ (σ : List ι) (t : ℕ), AllOnTime I t σ → OnTimeOn I S t σ := by
  intro σ
  induction σ with
  | nil => intro t _; trivial
  | cons j τ ih => intro t h; exact ⟨fun _ => h.1, ih _ h.2⟩

theorem onTimeOn_append (S : Finset ι) :
    ∀ (A B : List ι) (t : ℕ), OnTimeOn I S t (A ++ B) ↔
      OnTimeOn I S t A ∧ OnTimeOn I S (t + (A.map I.p).sum) B := by
  intro A
  induction A with
  | nil => intro B t; simp [OnTimeOn]
  | cons j τ ih =>
    intro B t
    simp only [List.cons_append, OnTimeOn, ih, List.map_cons, List.sum_cons]
    rw [show t + (I.p j + (τ.map I.p).sum) = t + I.p j + (τ.map I.p).sum by ring]
    tauto

theorem onTimeOn_of_forall_not_mem (S : Finset ι) :
    ∀ (B : List ι) (t : ℕ), (∀ i ∈ B, i ∉ S) → OnTimeOn I S t B := by
  intro B
  induction B with
  | nil => intro t _; trivial
  | cons j τ ih =>
    intro t h
    exact ⟨fun hj => absurd hj (h j List.mem_cons_self),
      ih _ (fun i hi => h i (List.mem_cons_of_mem _ hi))⟩

/-! ### Threshold characterisation -/

/-- Total dispatch time of the sites of `S` with deadline `≤ t` that occur in
the list `σ`. -/
def sumLE (S : Finset ι) (t : ℕ) (σ : List ι) : ℕ :=
  ((σ.filter (fun i => decide (i ∈ S ∧ I.d i ≤ t))).map I.p).sum

/-- The threshold condition: for every `t`, the sites of `S` whose deadline is
at most `t` have total dispatch time at most `t`. -/
def Thr (S : Finset ι) : Prop :=
  ∀ t : ℕ, ∑ i ∈ S.filter (fun i => I.d i ≤ t), I.p i ≤ t

theorem sumLE_eq_zero_or (S : Finset ι) :
    ∀ (σ : List ι) (t0 : ℕ), OnTimeOn I S t0 σ → ∀ t,
      sumLE I S t σ = 0 ∨ t0 + sumLE I S t σ ≤ t := by
  intro σ
  induction σ with
  | nil => intro t0 _ t; left; simp [sumLE]
  | cons j τ ih =>
    intro t0 h t
    obtain ⟨h1, h2⟩ := h
    have ih' := ih _ h2 t
    by_cases hc : j ∈ S ∧ I.d j ≤ t
    · have e : sumLE I S t (j :: τ) = I.p j + sumLE I S t τ := by
        simp [sumLE, List.filter_cons, hc]
      have := h1 hc.1
      have hp := I.p_pos j
      right
      rcases ih' with h0 | h0
      · rw [e, h0]; omega
      · rw [e]; omega
    · have e : sumLE I S t (j :: τ) = sumLE I S t τ := by
        simp only [sumLE, List.filter_cons]
        rw [if_neg (by simpa using hc)]
      rw [e]
      rcases ih' with h0 | h0
      · left; exact h0
      · right; have := I.p_pos j; omega

theorem sumLE_full [Fintype ι] (S : Finset ι) {σ : List ι} (hσ : IsOrder σ) (t : ℕ) :
    sumLE I S t σ = ∑ i ∈ S.filter (fun i => I.d i ≤ t), I.p i := by
  unfold sumLE
  rw [← List.sum_toFinset _ (hσ.1.filter _), List.toFinset_filter]
  apply Finset.sum_congr _ (fun _ _ => rfl)
  ext i
  simp [hσ.2 i]

theorem thr_of_feasible [Fintype ι] {S : Finset ι} (h : Feasible I S) : Thr I S := by
  obtain ⟨σ, hσ, hS⟩ := h
  intro t
  have h1 : OnTimeOn I S 0 σ := by
    rw [onTimeOn_iff_completion I S σ 0 hσ.1]
    intro i _ hi
    simpa using hS i hi
  rcases sumLE_eq_zero_or I S σ 0 h1 t with h | h
  · rw [sumLE_full I S hσ] at h; omega
  · rw [sumLE_full I S hσ] at h; omega

/-! ### The earliest-deadline-first list -/

theorem edd_perm (S : Finset ι) : (edd I S).Perm S.toList := by
  unfold edd
  exact List.perm_insertionSort _ _

theorem mem_edd {S : Finset ι} {i : ι} : i ∈ edd I S ↔ i ∈ S := by
  rw [(edd_perm I S).mem_iff]; simp

theorem nodup_edd (S : Finset ι) : (edd I S).Nodup :=
  (edd_perm I S).nodup_iff.2 (Finset.nodup_toList S)

theorem pairwise_edd (S : Finset ι) : (edd I S).Pairwise (fun a b => I.d a ≤ I.d b) := by
  unfold edd
  exact List.pairwise_insertionSort _ _

theorem toFinset_edd (S : Finset ι) : (edd I S).toFinset = S := by
  ext i; simp [mem_edd]

theorem sum_filter_list {τ : List ι} (hτ : τ.Nodup) (q : ι → Prop) [DecidablePred q] :
    ((τ.filter (fun i => decide (q i))).map I.p).sum = ∑ i ∈ τ.toFinset.filter q, I.p i := by
  rw [← List.sum_toFinset _ (hτ.filter _), List.toFinset_filter]
  simp

/-- A list sorted by deadline all of whose "prefix loads" fit is served on time. -/
theorem allOnTime_of_sorted :
    ∀ (τ : List ι) (t0 : ℕ), τ.Pairwise (fun a b => I.d a ≤ I.d b) →
      (∀ i ∈ τ, t0 + ((τ.filter (fun j => decide (I.d j ≤ I.d i))).map I.p).sum ≤ I.d i) →
      AllOnTime I t0 τ := by
  intro τ
  induction τ with
  | nil => intro t0 _ _; trivial
  | cons j ρ ih =>
    intro t0 hs h
    rw [List.pairwise_cons] at hs
    have hj := h j List.mem_cons_self
    have hfj : ((j :: ρ).filter (fun k => decide (I.d k ≤ I.d j))).map I.p
        = I.p j :: ((ρ.filter (fun k => decide (I.d k ≤ I.d j))).map I.p) := by
      simp [List.filter_cons]
    rw [hfj, List.sum_cons] at hj
    refine ⟨by omega, ih _ hs.2 ?_⟩
    intro i hi
    have hi' := h i (List.mem_cons_of_mem _ hi)
    have hji : I.d j ≤ I.d i := hs.1 i hi
    have : ((j :: ρ).filter (fun k => decide (I.d k ≤ I.d i))).map I.p
        = I.p j :: ((ρ.filter (fun k => decide (I.d k ≤ I.d i))).map I.p) := by
      simp [List.filter_cons, hji]
    rw [this, List.sum_cons] at hi'
    omega

theorem allOnTime_edd_of_thr (S : Finset ι) (h : Thr I S) : AllOnTime I 0 (edd I S) := by
  apply allOnTime_of_sorted I _ _ (pairwise_edd I S)
  intro i hi
  have hiS := (mem_edd I).1 hi
  rw [sum_filter_list I (nodup_edd I S), toFinset_edd]
  simpa using h (I.d i)

theorem feasible_of_allOnTime_edd [Fintype ι] (S : Finset ι)
    (h : AllOnTime I 0 (edd I S)) : Feasible I S := by
  classical
  refine ⟨edd I S ++ (Finset.univ \ S).toList, ⟨?_, ?_⟩, ?_⟩
  · rw [List.nodup_append]
    refine ⟨nodup_edd I S, Finset.nodup_toList _, ?_⟩
    intro a ha b hb hab
    subst hab
    simp only [Finset.mem_toList, Finset.mem_sdiff] at hb
    exact hb.2 ((mem_edd I).1 ha)
  · intro i
    by_cases hi : i ∈ S
    · exact List.mem_append_left _ ((mem_edd I).2 hi)
    · exact List.mem_append_right _ (by simp [hi])
  · have hnd : (edd I S ++ (Finset.univ \ S).toList).Nodup := by
      rw [List.nodup_append]
      refine ⟨nodup_edd I S, Finset.nodup_toList _, ?_⟩
      intro a ha b hb hab
      subst hab
      simp only [Finset.mem_toList, Finset.mem_sdiff] at hb
      exact hb.2 ((mem_edd I).1 ha)
    have h1 : OnTimeOn I S 0 (edd I S ++ (Finset.univ \ S).toList) := by
      rw [onTimeOn_append]
      refine ⟨allOnTime_imp_onTimeOn I S _ _ h, onTimeOn_of_forall_not_mem I S _ _ ?_⟩
      intro i hi
      simp only [Finset.mem_toList, Finset.mem_sdiff] at hi
      exact hi.2
    intro i hi
    have := (onTimeOn_iff_completion I S _ 0 hnd).1 h1 i
      (List.mem_append_left _ ((mem_edd I).2 hi)) hi
    simpa using this

/-- **Threshold characterisation of feasibility**: a set `S` can be served on
time iff for every `t` the sites of `S` with deadline `≤ t` have total dispatch
time `≤ t`; equivalently iff the EDD list of `S` is on time. -/
theorem feasible_iff_thr [Fintype ι] (S : Finset ι) : Feasible I S ↔ Thr I S :=
  ⟨thr_of_feasible I, fun h => feasible_of_allOnTime_edd I S (allOnTime_edd_of_thr I S h)⟩

end Aux

variable {ι : Type*} [DecidableEq ι] [Fintype ι] (I : Inst ι)

/-- `onTimeW` as a weight of an on-time set, for a start time `t`. -/
theorem onTimeW_eq_sum :
    ∀ (σ : List ι) (t : ℕ), σ.Nodup →
      onTimeW I t σ = ∑ i ∈ σ.toFinset.filter (fun i => t + completion I σ i ≤ I.d i), I.w i := by
  intro σ
  induction σ with
  | nil => intro t _; simp [onTimeW]
  | cons j τ ih =>
    intro t hnd
    rw [List.nodup_cons] at hnd
    obtain ⟨hj, hτ⟩ := hnd
    rw [onTimeW, ih _ hτ]
    have hfin : (j :: τ).toFinset = insert j τ.toFinset := by simp
    have hjτ : j ∉ τ.toFinset := by simpa using hj
    have key : ∀ i ∈ τ.toFinset,
        (t + completion I (j :: τ) i ≤ I.d i ↔ t + I.p j + completion I τ i ≤ I.d i) := by
      intro i hi
      have : i ≠ j := fun h => hj (h ▸ List.mem_toFinset.1 hi)
      rw [completion_cons_ne I τ this]
      constructor <;> intro h <;> omega
    rw [hfin, Finset.filter_insert, Finset.filter_congr key]
    have hjf : j ∉ τ.toFinset.filter (fun i => t + I.p j + completion I τ i ≤ I.d i) :=
      fun h => hjτ (Finset.mem_filter.1 h).1
    by_cases hc : t + I.p j ≤ I.d j
    · have hc' : t + completion I (j :: τ) j ≤ I.d j := by
        rwa [completion_cons_self]
      rw [if_pos hc', if_pos hc, Finset.sum_insert hjf]
    · have hc' : ¬ t + completion I (j :: τ) j ≤ I.d j := by
        rwa [completion_cons_self]
      rw [if_neg hc', if_neg hc]
      simp

theorem W_eq_weight_onTimeSet {σ : List ι} (hσ : σ.Nodup) :
    W I σ = weight I (onTimeSet I σ) := by
  unfold W weight onTimeSet
  rw [onTimeW_eq_sum I σ 0 hσ]
  simp

/-- **Lemma 1** of the paper: a set `S` is feasible (some dispatch order serves
all of `S` on time) iff the earliest-deadline-first list of `S`, served first
from time `0`, has every site on time. -/
theorem lemma1_edd (S : Finset ι) :
    Feasible I S ↔ AllOnTime I 0 (edd I S) := by
  constructor
  · intro h; exact allOnTime_edd_of_thr I S (thr_of_feasible I h)
  · exact feasible_of_allOnTime_edd I S

/-- Every dispatch order's on-time set is feasible. -/
theorem feasible_onTimeSet {σ : List ι} (hσ : IsOrder σ) : Feasible I (onTimeSet I σ) :=
  ⟨σ, hσ, fun i hi => (Finset.mem_filter.1 hi).2⟩

theorem exists_order : ∃ σ : List ι, IsOrder σ :=
  ⟨Finset.univ.toList, Finset.nodup_toList _, fun i => by simp⟩

/-- A feasible set is contained in the on-time set of some dispatch order. -/
theorem exists_order_of_feasible {S : Finset ι} (h : Feasible I S) :
    ∃ σ : List ι, IsOrder σ ∧ S ⊆ onTimeSet I σ := by
  obtain ⟨σ, hσ, hS⟩ := h
  exact ⟨σ, hσ, fun i hi => Finset.mem_filter.2 ⟨by simp [hσ.2 i], hS i hi⟩⟩

theorem isOPT_iff_max_feasible (v : ℕ) :
    IsOPT I v ↔
      (∃ S, Feasible I S ∧ weight I S = v) ∧
        ∀ S, Feasible I S → weight I S ≤ v := by
  have hW : ∀ σ : List ι, IsOrder σ → W I σ = weight I (onTimeSet I σ) :=
    fun σ hσ => W_eq_weight_onTimeSet I hσ.1
  have hmono : ∀ S T : Finset ι, S ⊆ T → weight I S ≤ weight I T :=
    fun S T h => Finset.sum_le_sum_of_subset h
  constructor
  · rintro ⟨⟨σ, hσ, hv⟩, hmax⟩
    refine ⟨⟨onTimeSet I σ, feasible_onTimeSet I hσ, by rw [← hW σ hσ, hv]⟩, ?_⟩
    intro S hS
    obtain ⟨σ', hσ', hsub⟩ := exists_order_of_feasible I hS
    calc weight I S ≤ weight I (onTimeSet I σ') := hmono _ _ hsub
      _ = W I σ' := (hW σ' hσ').symm
      _ ≤ v := hmax σ' hσ'
  · rintro ⟨⟨S, hS, hv⟩, hmax⟩
    obtain ⟨σ, hσ, hsub⟩ := exists_order_of_feasible I hS
    refine ⟨⟨σ, hσ, ?_⟩, fun σ' hσ' => ?_⟩
    · apply le_antisymm
      · rw [hW σ hσ]; exact hmax _ (feasible_onTimeSet I hσ)
      · rw [hW σ hσ, ← hv]; exact hmono _ _ hsub
    · rw [hW σ' hσ']; exact hmax _ (feasible_onTimeSet I hσ')

theorem exists_isOPT : ∃ v, IsOPT I v := by
  classical
  have hne : (Finset.univ.filter (fun S : Finset ι => Feasible I S)).Nonempty := by
    obtain ⟨σ, hσ⟩ := exists_order (ι := ι)
    exact ⟨onTimeSet I σ, by simpa using feasible_onTimeSet I hσ⟩
  obtain ⟨S, hS, hmax⟩ := Finset.exists_max_image _ (fun S => weight I S) hne
  refine ⟨weight I S, (isOPT_iff_max_feasible I _).2 ⟨⟨S, by simpa using hS, rfl⟩, ?_⟩⟩
  intro T hT
  exact hmax T (by simpa using hT)

end Mwhed
