import MwhedProofs.Core

/-!
# The FPTAS (Section 4.3 of the paper)

* Part A : approximation guarantee of value scaling (Theorem 4 / Appendix B).
* Part B : correctness of the value-indexed table `g` (recursion in the proof of Theorem 4).
* Part C : tightness (Proposition 8).
* Part D : Example 4.
-/

open Finset

namespace Mwhed

/-! ## Part A: approximation guarantee -/

section PartA

variable {ι : Type*}

/-- Scaled weight `w'_i = ⌊w_i / K⌋`. -/
noncomputable def scaledW (K : ℝ) (w : ι → ℕ) (i : ι) : ℕ := ⌊(w i : ℝ) / K⌋₊

/-- Step 1 of Appendix B, left half: `K w'_i ≤ w_i`. -/
theorem mul_scaledW_le {K : ℝ} (hK : 0 < K) (w : ι → ℕ) (i : ι) :
    K * (scaledW K w i : ℝ) ≤ w i := by
  have h : (scaledW K w i : ℝ) ≤ (w i : ℝ) / K := Nat.floor_le (by positivity)
  rw [le_div_iff₀ hK] at h
  linarith

/-- Step 1 of Appendix B, right half: `w_i < K (w'_i + 1)`. -/
theorem lt_mul_scaledW_succ {K : ℝ} (hK : 0 < K) (w : ι → ℕ) (i : ι) :
    (w i : ℝ) < K * ((scaledW K w i : ℝ) + 1) := by
  have h : (w i : ℝ) / K < (scaledW K w i : ℝ) + 1 := Nat.lt_floor_add_one _
  rw [div_lt_iff₀ hK] at h
  linarith

/-- `Ŝ` is a maximiser of the scaled weight over the family `F` of feasible sets. -/
def IsScaledMax (F : Finset ι → Prop) (K : ℝ) (w : ι → ℕ) (Sh : Finset ι) : Prop :=
  F Sh ∧ ∀ S, F S → ∑ i ∈ S, scaledW K w i ≤ ∑ i ∈ Sh, scaledW K w i

/-- **Generic scaling lemma** (Steps 2-4 of Appendix B).  For any family `F`, if `Ŝ`
maximises the scaled weight over `F`, then every `S ∈ F` satisfies
`weight(Ŝ) ≥ weight(S) − K |S|`. -/
theorem scaling_loss {F : Finset ι → Prop} {w : ι → ℕ} {K : ℝ} (hK : 0 < K)
    {Sh : Finset ι} (hSh : IsScaledMax F K w Sh) {S : Finset ι} (hS : F S) :
    (∑ i ∈ S, (w i : ℝ)) - K * S.card ≤ ∑ i ∈ Sh, (w i : ℝ) := by
  have h1 : ∑ i ∈ S, (w i : ℝ) ≤ ∑ i ∈ S, K * ((scaledW K w i : ℝ) + 1) :=
    sum_le_sum fun i _ => (lt_mul_scaledW_succ hK w i).le
  have h2 : ∑ i ∈ S, K * ((scaledW K w i : ℝ) + 1)
      = K * (∑ i ∈ S, (scaledW K w i : ℝ)) + K * S.card := by
    simp [← mul_sum, sum_add_distrib, mul_add, mul_comm]
  have h3 : (∑ i ∈ S, (scaledW K w i : ℝ)) ≤ ∑ i ∈ Sh, (scaledW K w i : ℝ) := by
    exact_mod_cast hSh.2 S hS
  have h4 : K * (∑ i ∈ Sh, (scaledW K w i : ℝ)) ≤ ∑ i ∈ Sh, (w i : ℝ) := by
    rw [mul_sum]; exact sum_le_sum fun i _ => mul_scaledW_le hK w i
  nlinarith [mul_le_mul_of_nonneg_left h3 hK.le]

/-- `weight(Ŝ) ≥ weight(S*) − K·n`. -/
theorem scaling_loss_card [Fintype ι] {F : Finset ι → Prop} {w : ι → ℕ} {K : ℝ} (hK : 0 < K)
    {Sh : Finset ι} (hSh : IsScaledMax F K w Sh) {S : Finset ι} (hS : F S) :
    (∑ i ∈ S, (w i : ℝ)) - K * Fintype.card ι ≤ ∑ i ∈ Sh, (w i : ℝ) := by
  have := scaling_loss hK hSh hS
  have hc : (S.card : ℝ) ≤ Fintype.card ι := by exact_mod_cast card_le_univ S
  nlinarith [mul_le_mul_of_nonneg_left hc hK.le]

/-- The scaling factor `K = max (1, ε w_max / n)` of Algorithm 2. -/
noncomputable def fptasK [Fintype ι] (ε : ℝ) (w : ι → ℕ) : ℝ :=
  max 1 (ε * ((univ.sup w : ℕ) : ℝ) / Fintype.card ι)

/-- **Theorem 4, case `ε w_max/n > 0` (scaling active)**, generic form: with
`K = ε w_max / n`, and `w_max ≤ W*`, we get `weight(Ŝ) ≥ (1-ε) W*`. -/
theorem fptas_case_scaled [Fintype ι] {F : Finset ι → Prop} {w : ι → ℕ} {ε : ℝ}
    (hε : 0 < ε) (hK : 0 < ε * ((univ.sup w : ℕ) : ℝ) / Fintype.card ι)
    {Sh : Finset ι} (hSh : IsScaledMax F (ε * ((univ.sup w : ℕ) : ℝ) / Fintype.card ι) w Sh)
    {Sstar : Finset ι} (hSs : F Sstar)
    (hwmax : ((univ.sup w : ℕ) : ℝ) ≤ ∑ i ∈ Sstar, (w i : ℝ)) :
    (1 - ε) * ∑ i ∈ Sstar, (w i : ℝ) ≤ ∑ i ∈ Sh, (w i : ℝ) := by
  have h := scaling_loss_card hK hSh hSs
  have hn : (Fintype.card ι : ℝ) ≠ 0 := by
    intro h0; rw [h0, div_zero] at hK; exact lt_irrefl _ hK
  have hKn : ε * ((univ.sup w : ℕ) : ℝ) / Fintype.card ι * Fintype.card ι
      = ε * ((univ.sup w : ℕ) : ℝ) := div_mul_cancel₀ _ hn
  nlinarith [mul_le_mul_of_nonneg_left hwmax hε.le]

/-- **Theorem 4, case `K = 1`**: the scaling is vacuous, `w' = w`, and `Ŝ` is optimal. -/
theorem fptas_case_vacuous {F : Finset ι → Prop} {w : ι → ℕ}
    {Sh : Finset ι} (hSh : IsScaledMax F 1 w Sh) {S : Finset ι} (hS : F S) :
    ∑ i ∈ S, w i ≤ ∑ i ∈ Sh, w i := by
  have : ∀ i, scaledW 1 w i = w i := by intro i; simp [scaledW]
  simpa [this] using hSh.2 S hS

/-- **Theorem 4 (generic form).**  With `K = max(1, ε w_max / n)`, if `Ŝ` maximises the
scaled weight over `F`, `S*` attains the best weight over `F`, and `w_max ≤ W*`, then
`weight(Ŝ) ≥ (1-ε) W*`. -/
theorem fptas_generic [Fintype ι] {F : Finset ι → Prop} {w : ι → ℕ} {ε : ℝ}
    (hε0 : 0 < ε) (hε1 : ε < 1) {Sh : Finset ι} (hSh : IsScaledMax F (fptasK ε w) w Sh)
    {Sstar : Finset ι} (hSs : F Sstar) (hopt : ∀ S, F S → ∑ i ∈ S, w i ≤ ∑ i ∈ Sstar, w i)
    (hwmax : (univ.sup w : ℕ) ≤ ∑ i ∈ Sstar, w i) :
    (1 - ε) * ∑ i ∈ Sstar, (w i : ℝ) ≤ ∑ i ∈ Sh, (w i : ℝ) := by
  rcases le_or_gt (ε * ((univ.sup w : ℕ) : ℝ) / Fintype.card ι) 1 with h | h
  · have hK : fptasK ε w = 1 := by simp [fptasK, h]
    rw [hK] at hSh
    have h1 := fptas_case_vacuous hSh hSs
    have h2 := hopt Sh hSh.1
    have h3 : ∑ i ∈ Sstar, w i = ∑ i ∈ Sh, w i := le_antisymm h1 h2
    have h4 : ∑ i ∈ Sstar, (w i : ℝ) = ∑ i ∈ Sh, (w i : ℝ) := by exact_mod_cast h3
    rw [h4]
    have : (0 : ℝ) ≤ ∑ i ∈ Sh, (w i : ℝ) := sum_nonneg fun _ _ => Nat.cast_nonneg _
    nlinarith
  · have hK : fptasK ε w = ε * ((univ.sup w : ℕ) : ℝ) / Fintype.card ι := by
      simp [fptasK, h.le]
    rw [hK] at hSh
    exact fptas_case_scaled hε0 (by linarith) hSh hSs (by exact_mod_cast hwmax)

end PartA

/-! ### Instantiation with MWHED feasibility -/

section Inst

variable {ι : Type*} [DecidableEq ι] [Fintype ι] (I : Inst ι)

/-- Serving a single individually feasible site first is a feasible set. -/
theorem feasible_singleton {i : ι} (hi : I.p i ≤ I.d i) : Feasible I {i} := by
  refine ⟨i :: (univ.erase i).toList, ⟨?_, ?_⟩, ?_⟩
  · simp [Finset.nodup_toList]
  · intro j
    by_cases h : j = i
    · simp [h]
    · simp [h]
  · intro j hj
    have : j = i := by simpa using hj
    subst this
    simpa [completion] using hi

/-- Under Assumption 1, `w_max ≤ W*` (given that `W*` bounds every feasible weight). -/
theorem wmax_le_of_indivFeasible (hI : IndivFeasible I) {v : ℕ}
    (hv : ∀ S, Feasible I S → weight I S ≤ v) : univ.sup I.w ≤ v := by
  refine Finset.sup_le fun i _ => ?_
  have := hv {i} (feasible_singleton I (hI i))
  simpa [weight] using this

/-- **Theorem 4 (FPTAS guarantee), Core-free form.**  `v` bounds the weight of every
feasible set and is attained by a feasible set.  If `Ŝ` is a feasible set of maximal
scaled value for `K = max(1, ε w_max/n)`, then `weight(Ŝ) ≥ (1-ε) v`. -/
theorem fptas_guarantee' (hI : IndivFeasible I) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε < 1)
    {Sh : Finset ι} (hSh : IsScaledMax (Feasible I) (fptasK ε I.w) I.w Sh) {v : ℕ}
    (hach : ∃ S, Feasible I S ∧ weight I S = v) (hv : ∀ S, Feasible I S → weight I S ≤ v) :
    (1 - ε) * (v : ℝ) ≤ (weight I Sh : ℝ) := by
  obtain ⟨S, hS, hSv⟩ := hach
  have := fptas_generic hε0 hε1 hSh hS (w := I.w)
    (fun T hT => by simpa [weight, ← hSv] using hv T hT)
    (by simpa [weight, ← hSv] using wmax_le_of_indivFeasible I hI hv)
  simpa [weight, ← hSv] using this

/-- **Theorem 4 (FPTAS guarantee)**: with `W*` the optimum (`IsOPT`), `weight(Ŝ) ≥ (1-ε) W*`
for every feasible `Ŝ` maximising the scaled weights `⌊w_i / K⌋`, `K = max(1, ε w_max / n)`. -/
theorem fptas_guarantee (hI : IndivFeasible I) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε < 1)
    {Sh : Finset ι} (hSh : IsScaledMax (Feasible I) (fptasK ε I.w) I.w Sh) {v : ℕ}
    (hv : IsOPT I v) : (1 - ε) * (v : ℝ) ≤ (weight I Sh : ℝ) := by
  obtain ⟨h1, h2⟩ := (isOPT_iff_max_feasible I v).1 hv
  exact fptas_guarantee' I hI hε0 hε1 hSh h1 h2

end Inst


/-! ## Part B: correctness of the value-indexed table `g`

`gTab I w' L v` is the paper's `g(i, v)` for the list `L = [1, …, i]` of sites sorted by
deadline: the least total dispatch time of a subset of `L` that is served in the order of `L`
(earliest deadline first), has every site on time from time `0`, and has scaled weight
`Σ w' = v`.  `⊤ = ∞` when no such subset exists.

The table is *defined* by the recursion of the proof of Theorem 4,
`g(i,v) = min (g(i-1,v), [g(i-1,v-w'_i) + p_i ≤ d_i] · (g(i-1,v-w'_i) + p_i))`
(see `gTab_nil`, `gTab_append_singleton`), and `gTab_eq_iInf` proves that it equals the
minimum above.  The list sortedness is not needed for this statement (it is only needed
to identify "sublist of `L` with `AllOnTime`" with "feasible set", i.e. Lemma 1). -/

section PartB

set_option linter.unusedSectionVars false

variable {ι : Type*} [DecidableEq ι] (I : Inst ι) (w' : ι → ℕ)

open scoped Classical in
/-- The table for the *reversed* list: the head of the argument is the site `i` with
the largest deadline (the last one served). -/
noncomputable def gRev : List ι → ℕ → ℕ∞
  | [], v => if v = 0 then 0 else ⊤
  | x :: r, v =>
      min (gRev r v)
        (if w' x ≤ v ∧ gRev r (v - w' x) + (I.p x : ℕ∞) ≤ (I.d x : ℕ∞)
          then gRev r (v - w' x) + (I.p x : ℕ∞) else ⊤)

/-- `g(i, v)` for the deadline-sorted list `L`. -/
noncomputable def gTab (L : List ι) (v : ℕ) : ℕ∞ := gRev I w' L.reverse v

/-- Base case `g(0, v)`: `0` for `v = 0`, `∞` otherwise. -/
theorem gTab_nil (v : ℕ) : gTab I w' [] v = if v = 0 then 0 else ⊤ := by
  simp [gTab, gRev]

open scoped Classical in
/-- The recursion of the proof of Theorem 4. -/
theorem gTab_append_singleton (L : List ι) (x : ι) (v : ℕ) :
    gTab I w' (L ++ [x]) v =
      min (gTab I w' L v)
        (if w' x ≤ v ∧ gTab I w' L (v - w' x) + (I.p x : ℕ∞) ≤ (I.d x : ℕ∞)
          then gTab I w' L (v - w' x) + (I.p x : ℕ∞) else ⊤) := by
  simp only [gTab, List.reverse_append, List.reverse_cons, List.reverse_nil, List.nil_append,
    List.singleton_append, gRev]
  congr

theorem allOnTime_append (t : ℕ) (S T : List ι) :
    AllOnTime I t (S ++ T) ↔ AllOnTime I t S ∧ AllOnTime I (t + (S.map I.p).sum) T := by
  induction S generalizing t with
  | nil => simp [AllOnTime]
  | cons i S ih =>
    simp only [List.cons_append, AllOnTime, ih, List.map_cons, List.sum_cons]
    rw [show t + (I.p i + (S.map I.p).sum) = t + I.p i + (S.map I.p).sum by ring]
    tauto

private theorem sublist_snoc {S L : List ι} {x : ι} (h : S.Sublist (L ++ [x])) :
    S.Sublist L ∨ ∃ S', S'.Sublist L ∧ S = S' ++ [x] := by
  rcases List.sublist_append_iff.1 h with ⟨l₁, l₂, rfl, h₁, h₂⟩
  rcases List.sublist_singleton.1 h₂ with rfl | rfl
  · left; simpa using h₁
  · right; exact ⟨l₁, h₁, rfl⟩

/-- Every feasible sublist of value `v` costs at least `g(L, v)`. -/
theorem gTab_le (L : List ι) :
    ∀ (v : ℕ) (S : List ι), S.Sublist L → AllOnTime I 0 S → (S.map w').sum = v →
      gTab I w' L v ≤ (((S.map I.p).sum : ℕ) : ℕ∞) := by
  induction L using List.reverseRecOn with
  | nil =>
    intro v S hS _ hv
    obtain rfl := List.sublist_nil.1 hS
    simp at hv
    subst hv
    simp [gTab_nil]
  | append_singleton L x ih =>
    intro v S hS hon hv
    rw [gTab_append_singleton]
    rcases sublist_snoc hS with h | ⟨S', h', rfl⟩
    · exact (min_le_left _ _).trans (ih v S h hon hv)
    · rw [allOnTime_append] at hon
      obtain ⟨hon', hon2⟩ := hon
      simp only [AllOnTime, zero_add, and_true] at hon2
      simp only [List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
        List.sum_nil, add_zero] at hv ⊢
      have hle : w' x ≤ v := by omega
      have hsub : v - w' x = (S'.map w').sum := by omega
      have ih' := ih (v - w' x) S' h' hon' hsub.symm
      have hc : gTab I w' L (v - w' x) + (I.p x : ℕ∞) ≤ (I.d x : ℕ∞) :=
        (add_le_add ih' le_rfl).trans (by exact_mod_cast hon2)
      refine (min_le_right _ _).trans ?_
      simp only [hle, hc, and_self, ↓reduceIte]
      rw [Nat.cast_add]
      exact add_le_add ih' le_rfl

/-- The value `g(L, v)` is `∞` or is attained by a feasible sublist of value `v`. -/
theorem gTab_attained (L : List ι) :
    ∀ v : ℕ, gTab I w' L v = ⊤ ∨
      ∃ S : List ι, S.Sublist L ∧ AllOnTime I 0 S ∧ (S.map w').sum = v ∧
        gTab I w' L v = (((S.map I.p).sum : ℕ) : ℕ∞) := by
  induction L using List.reverseRecOn with
  | nil =>
    intro v
    by_cases hv : v = 0
    · right; exact ⟨[], List.Sublist.refl _, by simp [AllOnTime], by simp [hv], by simp [gTab_nil, hv]⟩
    · left; simp [gTab_nil, hv]
  | append_singleton L x ih =>
    intro v
    rw [gTab_append_singleton]
    rcases min_choice (gTab I w' L v)
        (if w' x ≤ v ∧ gTab I w' L (v - w' x) + (I.p x : ℕ∞) ≤ (I.d x : ℕ∞)
          then gTab I w' L (v - w' x) + (I.p x : ℕ∞) else ⊤) with h | h
    · rw [h]
      rcases ih v with h1 | ⟨S, hS, hon, hv, hg⟩
      · left; exact h1
      · right
        exact ⟨S, hS.trans (List.sublist_append_left _ _), hon, hv, hg⟩
    · rw [h]
      split_ifs with hc
      · obtain ⟨hle, hc⟩ := hc
        rcases ih (v - w' x) with h1 | ⟨S, hS, hon, hv, hg⟩
        · rw [h1] at hc; simp at hc
        · right
          refine ⟨S ++ [x], hS.append (List.Sublist.refl _), ?_, ?_, ?_⟩
          · rw [allOnTime_append]
            refine ⟨hon, ?_⟩
            simp only [AllOnTime, zero_add, and_true]
            rw [hg] at hc
            exact_mod_cast hc
          · simp only [List.map_append, List.sum_append, List.map_cons, List.map_nil,
              List.sum_cons, List.sum_nil, add_zero]
            omega
          · rw [hg]
            simp
      · left; rfl

/-- **Correctness of the value-indexed DP (Task B).**  For every list `L` and value `v`,
`g(L, v)` is the minimum, over sublists `S` of `L` that are all on time from time `0`
(served in the order of `L`) and have scaled weight `Σ w' = v`, of the total dispatch time
`Σ p`; it is `∞` exactly when there is no such sublist. -/
theorem gTab_eq_iInf (L : List ι) (v : ℕ) :
    gTab I w' L v =
      ⨅ S : {S : List ι // S.Sublist L ∧ AllOnTime I 0 S ∧ (S.map w').sum = v},
        (((S.1.map I.p).sum : ℕ) : ℕ∞) := by
  apply le_antisymm
  · exact le_iInf fun S => gTab_le I w' L v S.1 S.2.1 S.2.2.1 S.2.2.2
  · rcases gTab_attained I w' L v with h | ⟨S, hS, hon, hv, hg⟩
    · rw [h]; exact le_top
    · rw [hg]; exact iInf_le (fun S : {S : List ι // S.Sublist L ∧ AllOnTime I 0 S ∧
        (S.map w').sum = v} => (((S.1.map I.p).sum : ℕ) : ℕ∞)) ⟨S, hS, hon, hv⟩

/-- Pointwise form of `gTab_eq_iInf`: `g(L, v) = t` iff some feasible sublist of value `v`
costs `t` and none costs less. -/
theorem gTab_eq_coe_iff (L : List ι) (v t : ℕ) :
    gTab I w' L v = (t : ℕ∞) ↔
      (∃ S : List ι, S.Sublist L ∧ AllOnTime I 0 S ∧ (S.map w').sum = v ∧ (S.map I.p).sum = t) ∧
      ∀ S : List ι, S.Sublist L → AllOnTime I 0 S → (S.map w').sum = v → t ≤ (S.map I.p).sum := by
  constructor
  · intro h
    rcases gTab_attained I w' L v with h1 | ⟨S, hS, hon, hv, hg⟩
    · rw [h] at h1; exact absurd h1 (by simp)
    · refine ⟨⟨S, hS, hon, hv, ?_⟩, fun T hT hon' hv' => ?_⟩
      · rw [h] at hg; exact_mod_cast hg.symm
      · have := gTab_le I w' L v T hT hon' hv'
        rw [h] at this; exact_mod_cast this
  · rintro ⟨⟨S, hS, hon, hv, hS'⟩, hmin⟩
    apply le_antisymm
    · have := gTab_le I w' L v S hS hon hv; rw [hS'] at this; exact this
    · rcases gTab_attained I w' L v with h1 | ⟨T, hT, hon', hv', hg⟩
      · rw [h1]; exact le_top
      · rw [hg]; exact_mod_cast hmin T hT hon' hv'

end PartB


/-! ## Part C: tightness of the analysis (Proposition 8)

The family of the proposition, over the sites `Fin n` (site `0` of this file is the paper's
site `1`): `p_i = 1`, `d_i = n`, `w_0 = M`, `w_i = ⌈K⌉ - 1` for `i ≠ 0`, `K = ε M / n`. -/

section PartC

/-- `K = ε M / n`. -/
noncomputable def tightK (ε : ℝ) (n M : ℕ) : ℝ := ε * M / n

/-- The weight `⌈K⌉ - 1` of the light sites. -/
noncomputable def tightC (ε : ℝ) (n M : ℕ) : ℕ := ⌈tightK ε n M⌉₊ - 1

/-- The instance of Proposition 8. -/
noncomputable def tightInst (ε : ℝ) (n M : ℕ) : Inst (Fin n) where
  p _ := 1
  d _ := n
  w i := if i.val = 0 then M else tightC ε n M
  p_pos _ := Nat.one_pos

/-- Hypotheses of Proposition 8: `ε ∈ (0,1)`, `n ≥ 2`, `M ≥ 2n/ε`. -/
structure TightHyp (ε : ℝ) (n M : ℕ) : Prop where
  hn : 2 ≤ n
  hε0 : 0 < ε
  hε1 : ε < 1
  hM : 2 * (n : ℝ) / ε ≤ M

namespace TightHyp

variable {ε : ℝ} {n M : ℕ}

theorem n_pos (h : TightHyp ε n M) : 0 < n := by have := h.hn; omega

theorem n_pos_real (h : TightHyp ε n M) : (0 : ℝ) < n := by exact_mod_cast h.n_pos

theorem M_pos_real (h : TightHyp ε n M) : (0 : ℝ) < M := by
  have h1 : 0 < 2 * (n : ℝ) / ε := by have := h.n_pos_real; have := h.hε0; positivity
  exact lt_of_lt_of_le h1 h.hM

/-- `K = εM/n ≥ 2` (in particular the scaling is active). -/
theorem K_ge_two (h : TightHyp ε n M) : 2 ≤ tightK ε n M := by
  have h1 : 2 * (n : ℝ) ≤ M * ε := (div_le_iff₀ h.hε0).1 h.hM
  unfold tightK
  rw [le_div_iff₀ h.n_pos_real]
  linarith

theorem K_pos (h : TightHyp ε n M) : 0 < tightK ε n M := by have := h.K_ge_two; linarith

/-- `⌈K⌉ - 1`, as a real, is `⌈K⌉ - 1`, and it is `< K` and `≥ K - 1`. -/
theorem C_cast (h : TightHyp ε n M) :
    (tightC ε n M : ℝ) = (⌈tightK ε n M⌉₊ : ℝ) - 1 := by
  have h1 : 1 ≤ ⌈tightK ε n M⌉₊ := Nat.one_le_iff_ne_zero.2
    (Nat.pos_iff_ne_zero.1 (Nat.ceil_pos.2 h.K_pos))
  unfold tightC
  rw [Nat.cast_sub h1]; simp

theorem C_lt_K (h : TightHyp ε n M) : (tightC ε n M : ℝ) < tightK ε n M := by
  rw [h.C_cast]
  have := Nat.ceil_lt_add_one h.K_pos.le
  linarith

theorem K_sub_one_le_C (h : TightHyp ε n M) : tightK ε n M - 1 ≤ (tightC ε n M : ℝ) := by
  rw [h.C_cast]
  have := Nat.le_ceil (tightK ε n M)
  linarith

end TightHyp

private theorem length_takeWhile_lt {α : Type*} [DecidableEq α] {σ : List α} {i : α}
    (hi : i ∈ σ) : (σ.takeWhile (· ≠ i)).length < σ.length := by
  have h := List.takeWhile_append_dropWhile (p := fun x => decide (x ≠ i)) (l := σ)
  have hnot : i ∉ σ.takeWhile (fun x => decide (x ≠ i)) := by
    intro hm
    have := List.mem_takeWhile_imp hm
    simp at this
  have hdw : i ∈ σ.dropWhile (fun x => decide (x ≠ i)) := by
    rw [← h] at hi
    rcases List.mem_append.1 hi with h1 | h1
    · exact absurd h1 hnot
    · exact h1
  have hpos : 0 < (σ.dropWhile (fun x => decide (x ≠ i))).length :=
    List.length_pos_of_mem hdw
  have hlen := congrArg List.length h
  rw [List.length_append] at hlen
  omega

variable {ε : ℝ} {n M : ℕ}

/-- **All sets are feasible** (total time `n`, every deadline `n`). -/
theorem tight_feasible (S : Finset (Fin n)) : Feasible (tightInst ε n M) S := by
  refine ⟨List.finRange n, ⟨List.nodup_finRange n, List.mem_finRange⟩, fun i _ => ?_⟩
  have hlt := length_takeWhile_lt (List.mem_finRange i)
  rw [List.length_finRange] at hlt
  simp only [completion, tightInst, List.map_const', List.sum_replicate, smul_eq_mul, mul_one]
  omega

theorem tight_indivFeasible (hn : 1 ≤ n) : IndivFeasible (tightInst ε n M) := by
  intro i; simpa [tightInst] using hn

theorem tight_weight_univ (h : TightHyp ε n M) :
    weight (tightInst ε n M) univ = M + (n - 1) * tightC ε n M := by
  have hz : (⟨0, h.n_pos⟩ : Fin n) ∈ (univ : Finset (Fin n)) := mem_univ _
  unfold weight
  rw [← Finset.add_sum_erase _ _ hz]
  have : ∑ i ∈ univ.erase (⟨0, h.n_pos⟩ : Fin n), (tightInst ε n M).w i
      = ∑ i ∈ univ.erase (⟨0, h.n_pos⟩ : Fin n), tightC ε n M := by
    refine Finset.sum_congr rfl fun i hi => ?_
    have : i.val ≠ 0 := by
      intro h0
      exact (Finset.mem_erase.1 hi).1 (Fin.ext h0)
    simp [tightInst, this]
  rw [this, Finset.sum_const, Finset.card_erase_of_mem hz]
  simp [tightInst]

/-- **`W* = M + (n-1)(⌈K⌉-1)`** (Core-free: the whole ground set is feasible and heaviest). -/
theorem tight_opt_core_free (h : TightHyp ε n M) :
    (∃ S, Feasible (tightInst ε n M) S ∧ weight (tightInst ε n M) S = M + (n - 1) * tightC ε n M) ∧
    ∀ S, Feasible (tightInst ε n M) S → weight (tightInst ε n M) S ≤ M + (n - 1) * tightC ε n M := by
  refine ⟨⟨univ, tight_feasible _, tight_weight_univ h⟩, fun S _ => ?_⟩
  rw [← tight_weight_univ h]
  exact Finset.sum_le_sum_of_subset (Finset.subset_univ S)

/-- **`W* = M + (n-1)(⌈K⌉-1)`** in the sense of `IsOPT` (via Core's `isOPT_iff_max_feasible`). -/
theorem tight_isOPT (h : TightHyp ε n M) :
    IsOPT (tightInst ε n M) (M + (n - 1) * tightC ε n M) :=
  (isOPT_iff_max_feasible _ _).2 (tight_opt_core_free h)

/-- Every site `i ≠ 0` has scaled weight `0`. -/
theorem tight_scaled_light (h : TightHyp ε n M) {i : Fin n} (hi : i.val ≠ 0) :
    scaledW (tightK ε n M) (tightInst ε n M).w i = 0 := by
  unfold scaledW
  rw [Nat.floor_eq_zero]
  simp only [tightInst, hi, ite_false]
  rw [div_lt_one h.K_pos]
  exact h.C_lt_K

/-- Site `0` has scaled weight `⌊n/ε⌋`. -/
theorem tight_scaled_heavy (h : TightHyp ε n M) {i : Fin n} (hi : i.val = 0) :
    scaledW (tightK ε n M) (tightInst ε n M).w i = ⌊(n : ℝ) / ε⌋₊ := by
  unfold scaledW
  simp only [tightInst, hi, ite_true]
  congr 1
  have := h.M_pos_real; have := h.n_pos_real; have := h.hε0
  unfold tightK
  field_simp

theorem tight_floor_pos (h : TightHyp ε n M) : 0 < ⌊(n : ℝ) / ε⌋₊ := by
  apply Nat.floor_pos.2
  have := h.n_pos_real; have := h.hε0; have := h.hε1; have := h.hn
  rw [le_div_iff₀ h.hε0]
  have : (2 : ℝ) ≤ n := by exact_mod_cast h.hn
  nlinarith

/-- The scaled value of a set `S`: `⌊n/ε⌋` if it contains site `0`, else `0`. -/
theorem tight_scaled_sum (h : TightHyp ε n M) (S : Finset (Fin n)) :
    ∑ i ∈ S, scaledW (tightK ε n M) (tightInst ε n M).w i =
      if (⟨0, h.n_pos⟩ : Fin n) ∈ S then ⌊(n : ℝ) / ε⌋₊ else 0 := by
  have : ∀ i : Fin n, scaledW (tightK ε n M) (tightInst ε n M).w i =
      if i = (⟨0, h.n_pos⟩ : Fin n) then ⌊(n : ℝ) / ε⌋₊ else 0 := by
    intro i
    by_cases hi : i.val = 0
    · have hz : i = ⟨0, h.n_pos⟩ := Fin.ext hi
      simp only [hz, ↓reduceIte]; exact tight_scaled_heavy h (i := ⟨0, h.n_pos⟩) rfl
    · have : i ≠ ⟨0, h.n_pos⟩ := fun e => hi (by simp [e])
      simp [this, tight_scaled_light h hi]
  simp only [this]
  rw [Finset.sum_ite_eq']

/-- **The scaled-value maximisers are exactly the sets containing site `0`.** -/
theorem tight_scaledMax_iff (h : TightHyp ε n M) (S : Finset (Fin n)) :
    IsScaledMax (Feasible (tightInst ε n M)) (tightK ε n M) (tightInst ε n M).w S ↔
      (⟨0, h.n_pos⟩ : Fin n) ∈ S := by
  have hp := tight_floor_pos h
  constructor
  · rintro ⟨_, hmax⟩
    by_contra hz
    have := hmax univ (tight_feasible _)
    rw [tight_scaled_sum h, tight_scaled_sum h] at this
    simp only [hz, mem_univ, ↓reduceIte] at this
    omega
  · intro hz
    refine ⟨tight_feasible _, fun T _ => ?_⟩
    rw [tight_scaled_sum h, tight_scaled_sum h]
    simp only [hz, ite_true]
    split_ifs <;> omega

/-- Total dispatch time of a set is its cardinality (`p_i = 1`). -/
theorem tight_time (S : Finset (Fin n)) : time (tightInst ε n M) S = S.card := by
  simp [time, tightInst]

/-- **What Algorithm 2 returns on the family** (`v*` maximal, then minimum time `g(n,v*)`):
a scaled-value maximiser of minimum total dispatch time is exactly `{0}`. -/
theorem tight_algorithm_output (h : TightHyp ε n M) (S : Finset (Fin n)) :
    (IsScaledMax (Feasible (tightInst ε n M)) (tightK ε n M) (tightInst ε n M).w S ∧
      ∀ T, IsScaledMax (Feasible (tightInst ε n M)) (tightK ε n M) (tightInst ε n M).w T →
        time (tightInst ε n M) S ≤ time (tightInst ε n M) T) ↔
      S = {(⟨0, h.n_pos⟩ : Fin n)} := by
  constructor
  · rintro ⟨hS, hmin⟩
    have hz := (tight_scaledMax_iff h S).1 hS
    have h1 := hmin {(⟨0, h.n_pos⟩ : Fin n)} ((tight_scaledMax_iff h _).2 (by simp))
    rw [tight_time, tight_time] at h1
    simp only [card_singleton] at h1
    symm
    apply Finset.eq_of_subset_of_card_le (by simpa using hz)
    simpa using h1
  · rintro rfl
    refine ⟨(tight_scaledMax_iff h _).2 (by simp), fun T hT => ?_⟩
    have hz := (tight_scaledMax_iff h T).1 hT
    rw [tight_time, tight_time]
    simpa using Finset.card_pos.2 ⟨_, hz⟩

/-- The set returned by Algorithm 2, `{0}`, has weight `M`. -/
theorem tight_returned_weight (h : TightHyp ε n M) :
    weight (tightInst ε n M) {(⟨0, h.n_pos⟩ : Fin n)} = M := by
  simp [weight, tightInst]

/-- **Proposition 8, ratio bound.**  `M / W* ≤ 1 / (1 + ε (n-1)/n - (n-1)/M)`, where the
denominator is positive and `W* = M + (n-1)(⌈K⌉-1)`. -/
theorem tight_ratio (h : TightHyp ε n M) :
    0 < 1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M ∧
    (M : ℝ) / ((M + (n - 1) * tightC ε n M : ℕ) : ℝ) ≤
      1 / (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M) := by
  have hn := h.n_pos_real
  have hMp := h.M_pos_real
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast h.n_pos
  have hε0 := h.hε0
  -- M > n
  have hMn : (n : ℝ) < M := by
    have h1 : 2 * (n : ℝ) ≤ M * ε := (div_le_iff₀ h.hε0).1 h.hM
    nlinarith [h.hε1]
  have hD : 0 < 1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M := by
    have h1 : ((n : ℝ) - 1) / M < 1 := by
      rw [div_lt_one hMp]; linarith
    have h2 : 0 ≤ ε * ((n : ℝ) - 1) / n := by
      apply div_nonneg _ hn.le
      exact mul_nonneg hε0.le (by linarith)
    linarith
  refine ⟨hD, ?_⟩
  -- W* ≥ M * D
  have hC := h.K_sub_one_le_C
  have hcast : ((M + (n - 1) * tightC ε n M : ℕ) : ℝ) = M + ((n : ℝ) - 1) * tightC ε n M := by
    rw [Nat.cast_add, Nat.cast_mul, Nat.cast_sub h.n_pos]; simp
  have hlow : (M : ℝ) * (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M) ≤
      ((M + (n - 1) * tightC ε n M : ℕ) : ℝ) := by
    rw [hcast]
    have hK : tightK ε n M = ε * M / n := rfl
    have e : (M : ℝ) * (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M)
        = M + ((n : ℝ) - 1) * (tightK ε n M - 1) := by
      rw [hK]; field_simp; ring
    rw [e]
    have : (0 : ℝ) ≤ (n : ℝ) - 1 := by linarith
    nlinarith [mul_le_mul_of_nonneg_left hC this]
  have hpos : 0 < (M : ℝ) * (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M) := mul_pos hMp hD
  calc (M : ℝ) / ((M + (n - 1) * tightC ε n M : ℕ) : ℝ)
      ≤ (M : ℝ) / ((M : ℝ) * (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M)) :=
        div_le_div_of_nonneg_left hMp.le hpos hlow
    _ = 1 / (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M) := by
        field_simp

theorem tight_fptasK (h : TightHyp ε n M) :
    fptasK ε (tightInst ε n M).w = tightK ε n M := by
  have hz : (⟨0, h.n_pos⟩ : Fin n) ∈ (univ : Finset (Fin n)) := mem_univ _
  have hKM : tightK ε n M < M := by
    have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast h.n_pos
    have h1 : tightK ε n M ≤ ε * M := div_le_self (mul_nonneg h.hε0.le h.M_pos_real.le) hn1
    nlinarith [h.hε1, h.M_pos_real]
  have hsup : (univ.sup (tightInst ε n M).w : ℕ) = M := by
    apply le_antisymm
    · refine Finset.sup_le fun i _ => ?_
      by_cases hi : i.val = 0
      · simp [tightInst, hi]
      · have : (tightC ε n M : ℝ) < M := (h.C_lt_K).trans hKM
        simp only [tightInst, hi, ite_false]
        exact_mod_cast this.le
    · have := Finset.le_sup (f := (tightInst ε n M).w) hz
      simpa [tightInst] using this
  unfold fptasK
  rw [hsup, Fintype.card_fin]
  exact max_eq_right (by have := h.K_ge_two; unfold tightK at this; linarith)

/-- **Sanity check against Theorem 4**: the returned set `{0}` of weight `M` does satisfy
the guarantee `(1-ε) W* ≤ M` (so Proposition 8 is consistent with the FPTAS bound). -/
theorem tight_guarantee_holds (h : TightHyp ε n M) :
    (1 - ε) * ((M + (n - 1) * tightC ε n M : ℕ) : ℝ) ≤ M := by
  have hSh : IsScaledMax (Feasible (tightInst ε n M)) (fptasK ε (tightInst ε n M).w)
      (tightInst ε n M).w {(⟨0, h.n_pos⟩ : Fin n)} := by
    rw [tight_fptasK h]
    exact (tight_scaledMax_iff h _).2 (by simp)
  have := fptas_guarantee' (tightInst ε n M) (tight_indivFeasible (by have := h.hn; omega))
    h.hε0 h.hε1 hSh (tight_opt_core_free h).1 (tight_opt_core_free h).2
  rw [tight_returned_weight h] at this
  exact this

/-- **Proposition 8, comparison of the ratio with the guarantee**: for `ε ∈ (0,1)`,
`(1/(1+ε)) / (1-ε) = 1/(1-ε²)`, and `1 - ε ≤ 1/(1+ε)` (so the guarantee is below the
limit of the ratio bound, the gap being the factor `1 - ε²`). -/
theorem tight_limit_identity {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε < 1) :
    (1 / (1 + ε)) / (1 - ε) = 1 / (1 - ε ^ 2) := by
  have h1 : (1 + ε) ≠ 0 := by linarith
  have h2 : (1 - ε) ≠ 0 := by linarith
  have h3 : (1 - ε ^ 2) ≠ 0 := by nlinarith
  field_simp
  ring

theorem tight_guarantee_lt_limit {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε < 1) :
    1 - ε < 1 / (1 + ε) := by
  rw [lt_div_iff₀ (by linarith)]
  nlinarith

end PartC


/-! ## Part D: Example 4 (the FPTAS on the running instance) -/

section PartD

/-- Weights of the running instance (Example 1), sites `1..4` as `0..3`. -/
def exW : Fin 4 → ℕ := ![5, 8, 3, 10]

theorem ex4_wmax : (Finset.univ.sup exW : ℕ) = 10 := by decide

/-- `K = max(1, 0.5 · 10 / 4) = 1.25`. -/
theorem ex4_K : fptasK (1 / 2 : ℝ) exW = 5 / 4 := by
  unfold fptasK
  rw [ex4_wmax, Fintype.card_fin]
  norm_num

/-- Scaled weights `(⌊5/1.25⌋, ⌊8/1.25⌋, ⌊3/1.25⌋, ⌊10/1.25⌋) = (4, 6, 2, 8)`. -/
theorem ex4_scaled : (fun i => scaledW (5 / 4 : ℝ) exW i) = ![4, 6, 2, 8] := by
  funext i
  fin_cases i <;> simp [scaledW, exW] <;> norm_num

/-- The guaranteed bound `(1 - ε) W* = 11.5` for `W* = 23`, `ε = 1/2`. -/
theorem ex4_bound : (1 - (1 / 2 : ℝ)) * 23 = 23 / 2 := by norm_num

end PartD

end Mwhed
