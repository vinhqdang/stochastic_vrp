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
  simp [gTab, gRev]

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
        (add_le_add_left ih' _).trans (by exact_mod_cast hon2)
      refine (min_le_right _ _).trans ?_
      rw [if_pos ⟨hle, hc⟩]
      push_cast
      exact add_le_add_left ih' _

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
      · have := gTab_le I w' L T hS.length_le |> fun _ => gTab_le I w' L v T hT hon' hv'
        rw [h] at this; exact_mod_cast this
  · rintro ⟨⟨S, hS, hon, hv, hS'⟩, hmin⟩
    apply le_antisymm
    · have := gTab_le I w' L v S hS hon hv; rw [hS'] at this; exact this
    · rcases gTab_attained I w' L v with h1 | ⟨T, hT, hon', hv', hg⟩
      · rw [h1]; exact le_top
      · rw [hg]; exact_mod_cast hmin T hT hon' hv'

end PartB

end Mwhed
