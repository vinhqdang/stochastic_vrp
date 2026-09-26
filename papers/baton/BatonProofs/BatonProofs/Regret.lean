import Mathlib

/-!
Propositions 2 and 3 of the BATON manuscript on finite decision trees.

Setting. A day is a finite probability tree. A `node H n p c` is the
situation after serving a stop without breach, with handoff price `H`;
nature then moves to child `i` with probability `p i`. A `leaf e` ends the
day with cost `e`: the emergency price of a breach, or `0` after a clean
completion. Nodes are *histories*, so the results below need no Markov
assumption; the Markov case of the paper, where values depend on the
history only through `(k, W_k)`, is a special case. Finite trees are also
exactly the setting of the fitted policy, which works on a finite set of
training days.

Everything is restricted to the handoff lever, as in Section 3.3.

* `V`   optimal expected cost (value) from a node;
* `Cn`  continuation value at a node under optimal play (paper: `C_k`);
* `C0`  cost of continuing and never acting again (paper: `C⁰_k`);
* `J0`  expected cost of the myopic rule "hand off iff `H < C0`";
* `Lm`, `Lfrom`, `Lafter` the clairvoyant cost of the rest of the day.
-/

open Finset

namespace Baton.Tree

inductive DTree
  | leaf (e : ℝ)
  | node (H : ℝ) (n : ℕ) (p : Fin n → ℝ) (c : Fin n → DTree)

open DTree

/-- Well-formed probability tree. -/
def WF : DTree → Prop
  | leaf _ => True
  | node _ _ p c => (∀ i, 0 ≤ p i) ∧ (∑ i, p i) = 1 ∧ ∀ i, WF (c i)

noncomputable def V : DTree → ℝ
  | leaf e => e
  | node H _ p c => min H (∑ i, p i * V (c i))

/-- Continuation value under optimal play. -/
noncomputable def Cn : DTree → ℝ
  | leaf e => e
  | node _ _ p c => ∑ i, p i * V (c i)

/-- Cost of continuing and never acting again. -/
noncomputable def C0 : DTree → ℝ
  | leaf e => e
  | node _ _ p c => ∑ i, p i * C0 (c i)

/-- Expected cost of the myopic rule: hand off iff `H < C0`. -/
noncomputable def J0 : DTree → ℝ
  | leaf e => e
  | node H n p c => if H < C0 (node H n p c) then H else ∑ i, p i * J0 (c i)

/-- Clairvoyant cost of a subtree when the cheapest handoff price seen so
far on the path is `a`: path by path, the minimum of `a`, of the handoff
prices along the path and of the terminal cost. -/
noncomputable def Lm : ℝ → DTree → ℝ
  | a, leaf e => min a e
  | a, node H _ p c => ∑ i, p i * Lm (min a H) (c i)

/-- Clairvoyant cost from a subtree, a handoff being allowed at its root
and later. -/
noncomputable def Lfrom : DTree → ℝ
  | leaf e => e
  | node H _ p c => ∑ i, p i * Lm H (c i)

/-- Clairvoyant cost of the rest of the day after the root's decision
(paper: `L_k`): the cheapest *later* handoff before the breach, or the
breach, or nothing on a clean day. -/
noncomputable def Lafter : DTree → ℝ
  | leaf e => e
  | node _ _ p c => ∑ i, p i * Lfrom (c i)

/-! ### Elementary facts about weighted sums -/

section Sums
variable {n : ℕ} (p : Fin n → ℝ)

lemma wsum_le (hp : ∀ i, 0 ≤ p i) {f g : Fin n → ℝ} (h : ∀ i, f i ≤ g i) :
    ∑ i, p i * f i ≤ ∑ i, p i * g i :=
  Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (h i) (hp i)

lemma wsum_const (hs : ∑ i, p i = 1) (a : ℝ) : ∑ i, p i * a = a := by
  rw [← Finset.sum_mul, hs, one_mul]

/-- `min a (𝔼 x) ≥ 𝔼 (min a x)`: the minimum is concave. -/
lemma wsum_min_le (hp : ∀ i, 0 ≤ p i) (hs : ∑ i, p i = 1) (a : ℝ) (x : Fin n → ℝ) :
    ∑ i, p i * min a (x i) ≤ min a (∑ i, p i * x i) := by
  refine le_min ?_ ?_
  · calc ∑ i, p i * min a (x i) ≤ ∑ i, p i * a := wsum_le p hp fun i => min_le_left _ _
      _ = a := wsum_const p hs a
  · exact wsum_le p hp fun i => min_le_right _ _

lemma wsum_lt (hp : ∀ i, 0 ≤ p i) {f g : Fin n → ℝ} (h : ∀ i, f i ≤ g i)
    (hlt : ∃ i, 0 < p i ∧ f i < g i) : ∑ i, p i * f i < ∑ i, p i * g i := by
  obtain ⟨j, hpj, hj⟩ := hlt
  exact Finset.sum_lt_sum (fun i _ => mul_le_mul_of_nonneg_left (h i) (hp i))
    ⟨j, Finset.mem_univ _, mul_lt_mul_of_pos_left hj hpj⟩

lemma wsum_affine (hs : ∑ i, p i = 1) (ω : ℝ) (x : Fin n → ℝ) :
    ∑ i, p i * (ω * (1 - x i)) = ω - ω * ∑ i, p i * x i := by
  have h : ∀ i, p i * (ω * (1 - x i)) = ω * p i - ω * (p i * x i) := fun i => by ring
  simp_rw [h, Finset.sum_sub_distrib, ← Finset.mul_sum, hs, mul_one]

lemma wsum_sub (f g : Fin n → ℝ) :
    ∑ i, p i * f i - ∑ i, p i * g i = ∑ i, p i * (f i - g i) := by
  rw [← Finset.sum_sub_distrib]; congr 1; ext i; ring

end Sums

/-! ### Proposition 2 -/

/-- Optimal play never costs more than never acting again. -/
theorem V_le_C0 : ∀ t : DTree, WF t → V t ≤ C0 t
  | leaf e, _ => le_rfl
  | node H n p c, ⟨hp, _, hc⟩ => by
    simp only [V, C0]
    exact le_trans (min_le_right _ _) (wsum_le p hp fun i => V_le_C0 (c i) (hc i))

/-- **Proposition 2 (inclusion).** `C_k ≤ C⁰_k` at every node. -/
theorem Cn_le_C0 (H : ℝ) (n : ℕ) (p : Fin n → ℝ) (c : Fin n → DTree)
    (h : WF (node H n p c)) : Cn (node H n p c) ≤ C0 (node H n p c) := by
  obtain ⟨hp, _, hc⟩ := h
  simp only [Cn, C0]
  exact wsum_le p hp fun i => V_le_C0 (c i) (hc i)

/-- Consequently the optimal stopping region is inside the myopic one:
wherever the optimal rule hands off, so does the myopic rule. -/
theorem opt_stop_imp_myopic_stop (H : ℝ) (n : ℕ) (p : Fin n → ℝ) (c : Fin n → DTree)
    (h : WF (node H n p c)) (hstop : H < Cn (node H n p c)) :
    H < C0 (node H n p c) :=
  lt_of_lt_of_le hstop (Cn_le_C0 H n p c h)

/-- The myopic rule fires at this node, or at a later node reached with
positive probability before any breach. -/
def Fires : DTree → Prop
  | leaf _ => False
  | node H n p c => H < C0 (node H n p c) ∨ ∃ i, 0 < p i ∧ Fires (c i)

theorem V_lt_C0_of_fires : ∀ t : DTree, WF t → Fires t → V t < C0 t
  | leaf _, _, hf => hf.elim
  | node H n p c, hw, hf => by
    obtain ⟨hp, hs, hc⟩ := hw
    rcases hf with hH | ⟨j, hpj, hj⟩
    · exact lt_of_le_of_lt (by simp only [V]; exact min_le_left _ _) hH
    · have hsum : ∑ i, p i * V (c i) < ∑ i, p i * C0 (c i) :=
        wsum_lt p hp (fun i => V_le_C0 (c i) (hc i))
          ⟨j, hpj, V_lt_C0_of_fires (c j) (hc j) hj⟩
      simp only [V, C0]
      exact lt_of_le_of_lt (min_le_right _ _) hsum

/-- **Proposition 2 (strictness).** If, with positive probability, the
path survives to a later stop at which the myopic rule itself fires, the
optimal continuation value is *strictly* below the never-act cost. -/
theorem Cn_lt_C0 (H : ℝ) (n : ℕ) (p : Fin n → ℝ) (c : Fin n → DTree)
    (h : WF (node H n p c)) (hreach : ∃ i, 0 < p i ∧ Fires (c i)) :
    Cn (node H n p c) < C0 (node H n p c) := by
  obtain ⟨hp, _, hc⟩ := h
  obtain ⟨j, hpj, hj⟩ := hreach
  simp only [Cn, C0]
  exact wsum_lt p hp (fun i => V_le_C0 (c i) (hc i))
    ⟨j, hpj, V_lt_C0_of_fires (c j) (hc j) hj⟩

/-! ### The clairvoyant lower bound -/

theorem Lm_le_min_V : ∀ (t : DTree) (a : ℝ), WF t → Lm a t ≤ min a (V t)
  | leaf e, a, _ => le_rfl
  | node H n p c, a, ⟨hp, hs, hc⟩ => by
    simp only [Lm, V]
    calc ∑ i, p i * Lm (min a H) (c i)
        ≤ ∑ i, p i * min (min a H) (V (c i)) :=
          wsum_le p hp fun i => Lm_le_min_V (c i) (min a H) (hc i)
      _ ≤ min (min a H) (∑ i, p i * V (c i)) := wsum_min_le p hp hs _ _
      _ = min a (min H (∑ i, p i * V (c i))) := min_assoc _ _ _

/-- No policy beats the clairvoyant on average. -/
theorem Lfrom_le_V : ∀ t : DTree, WF t → Lfrom t ≤ V t
  | leaf e, _ => le_rfl
  | node H n p c, ⟨hp, hs, hc⟩ => by
    simp only [Lfrom, V]
    calc ∑ i, p i * Lm H (c i) ≤ ∑ i, p i * min H (V (c i)) :=
          wsum_le p hp fun i => Lm_le_min_V (c i) H (hc i)
      _ ≤ min H (∑ i, p i * V (c i)) := wsum_min_le p hp hs _ _

/-- `C_k(W_k) ≥ 𝔼[L_k | 𝓕_k]`. -/
theorem Lafter_le_Cn (H : ℝ) (n : ℕ) (p : Fin n → ℝ) (c : Fin n → DTree)
    (h : WF (node H n p c)) : Lafter (node H n p c) ≤ Cn (node H n p c) := by
  obtain ⟨hp, _, hc⟩ := h
  simp only [Lafter, Cn]
  exact wsum_le p hp fun i => Lfrom_le_V (c i) (hc i)

/-! ### Proposition 3 -/

/-- Exact regret of the myopic rule, accumulated at the nodes where it
hands off: `H - V = (H - C_k) 1{σ⁰ < σ*}` there. -/
noncomputable def Rex : DTree → ℝ
  | leaf _ => 0
  | node H n p c =>
      if H < C0 (node H n p c) then H - min H (∑ i, p i * V (c i))
      else ∑ i, p i * Rex (c i)

/-- The bound of Proposition 3: `(H - L_k) 1{σ⁰ < σ*}` accumulated where
the myopic rule hands off. -/
noncomputable def Rbd : DTree → ℝ
  | leaf _ => 0
  | node H n p c =>
      if H < C0 (node H n p c) then
        (if H ≤ ∑ i, p i * V (c i) then 0 else H - Lafter (node H n p c))
      else ∑ i, p i * Rbd (c i)

/-- Where the myopic rule continues, so does the optimal rule. -/
theorem V_eq_Cn_of_myopic_continues (H : ℝ) (n : ℕ) (p : Fin n → ℝ) (c : Fin n → DTree)
    (h : WF (node H n p c)) (hcont : ¬ H < C0 (node H n p c)) :
    V (node H n p c) = ∑ i, p i * V (c i) := by
  have h1 := Cn_le_C0 H n p c h
  simp only [Cn] at h1
  simp only [V]
  exact min_eq_right (le_trans h1 (not_lt.mp hcont))

/-- **Proposition 3 (identity).** The excess expected cost of the myopic
rule equals the accumulated `(H - C_k) 1{σ⁰ < σ*}`. -/
theorem regret_eq : ∀ t : DTree, WF t → J0 t - V t = Rex t
  | leaf e, _ => by simp [J0, V, Rex]
  | node H n p c, hw => by
    by_cases hm : H < C0 (node H n p c)
    · simp only [J0, Rex, hm, ↓reduceIte, V]
    · have hV := V_eq_Cn_of_myopic_continues H n p c hw hm
      obtain ⟨hp, hs, hc⟩ := hw
      rw [hV]
      simp only [J0, Rex, hm, ↓reduceIte]
      rw [wsum_sub]
      exact Finset.sum_congr rfl fun i _ => by rw [regret_eq (c i) (hc i)]

theorem Rex_nonneg : ∀ t : DTree, WF t → 0 ≤ Rex t
  | leaf _, _ => le_rfl
  | node H n p c, ⟨hp, hs, hc⟩ => by
    simp only [Rex]
    split_ifs
    · exact sub_nonneg.mpr (min_le_left _ _)
    · exact Finset.sum_nonneg fun i _ => mul_nonneg (hp i) (Rex_nonneg (c i) (hc i))

theorem Rex_le_Rbd : ∀ t : DTree, WF t → Rex t ≤ Rbd t
  | leaf _, _ => le_rfl
  | node H n p c, hw => by
    have hL := Lafter_le_Cn H n p c hw
    obtain ⟨hp, hs, hc⟩ := hw
    simp only [Cn] at hL
    simp only [Rex, Rbd]
    split_ifs with h1 h2
    · rw [min_eq_left h2, sub_self]
    · rw [min_eq_right (le_of_lt (not_le.mp h2))]
      linarith
    · exact wsum_le p hp fun i => Rex_le_Rbd (c i) (hc i)

/-- **Proposition 3.** `0 ≤ 𝔼 J(σ⁰) - 𝔼 J(σ*) = 𝔼[(H - C)1{σ⁰<σ*}]
≤ 𝔼[(H - L)1{σ⁰<σ*}]`. -/
theorem price_of_overtriggering (t : DTree) (h : WF t) :
    0 ≤ J0 t - V t ∧ J0 t - V t = Rex t ∧ J0 t - V t ≤ Rbd t := by
  refine ⟨?_, regret_eq t h, ?_⟩
  · rw [regret_eq t h]; exact Rex_nonneg t h
  · rw [regret_eq t h]; exact Rex_le_Rbd t h

/-! ### Flat prices -/

/-- Flat prices: every handoff costs `ω ≥ 0`; every terminal cost is `0`
(clean completion) or at least `ω` (a breach). -/
def Flat (ω : ℝ) : DTree → Prop
  | leaf e => e = 0 ∨ ω ≤ e
  | node H _ _ c => H = ω ∧ ∀ i, Flat ω (c i)

/-- Probability that the day completes cleanly. -/
noncomputable def clean : DTree → ℝ
  | leaf e => if e = 0 then 1 else 0
  | node _ _ p c => ∑ i, p i * clean (c i)

theorem Lm_flat (ω : ℝ) (hω : 0 ≤ ω) :
    ∀ t : DTree, WF t → Flat ω t → ω * (1 - clean t) ≤ Lm ω t
  | leaf e, _, hf => by
    simp only [Lm, clean]
    by_cases h0 : e = 0
    · simp [h0, hω]
    · have hle : ω ≤ e := hf.resolve_left h0
      simp [h0, hle]
  | node H n p c, ⟨hp, hs, hc⟩, ⟨hH, hfc⟩ => by
    simp only [Lm, clean, hH, min_self]
    calc ω * (1 - ∑ i, p i * clean (c i))
        = ∑ i, p i * (ω * (1 - clean (c i))) := by rw [wsum_affine p hs]; ring
      _ ≤ ∑ i, p i * Lm ω (c i) :=
          wsum_le p hp fun i => Lm_flat ω hω (c i) (hc i) (hfc i)

theorem Lfrom_flat (ω : ℝ) (hω : 0 ≤ ω) :
    ∀ t : DTree, WF t → Flat ω t → ω * (1 - clean t) ≤ Lfrom t
  | leaf e, _, hf => by
    simp only [Lfrom, clean]
    by_cases h0 : e = 0
    · simp [h0]
    · have hle : ω ≤ e := hf.resolve_left h0
      simp [h0, hle]
  | node H n p c, hw, hf => by
    have h := Lm_flat ω hω (node H n p c) hw hf
    obtain ⟨hH, _⟩ := hf
    simp only [Lm, hH, min_self] at h
    simp only [Lfrom, hH]
    exact h

/-- The flat-price bound: `ω · P(σ⁰ < σ*, T = ∞)` accumulated. -/
noncomputable def Rflat (ω : ℝ) : DTree → ℝ
  | leaf _ => 0
  | node H n p c =>
      if H < C0 (node H n p c) then
        (if H ≤ ∑ i, p i * V (c i) then 0 else ω * ∑ i, p i * clean (c i))
      else ∑ i, p i * Rflat ω (c i)

theorem Rbd_le_Rflat (ω : ℝ) (hω : 0 ≤ ω) :
    ∀ t : DTree, WF t → Flat ω t → Rbd t ≤ Rflat ω t
  | leaf _, _, _ => le_rfl
  | node H n p c, ⟨hp, hs, hc⟩, ⟨hH, hfc⟩ => by
    simp only [Rbd, Rflat]
    split_ifs with h1 h2
    · exact le_rfl
    · -- ω - Lafter ≤ ω · P(clean)
      have hL : ∀ i, ω * (1 - clean (c i)) ≤ Lfrom (c i) :=
        fun i => Lfrom_flat ω hω (c i) (hc i) (hfc i)
      simp only [Lafter, hH]
      have : ∑ i, p i * (ω * (1 - clean (c i))) ≤ ∑ i, p i * Lfrom (c i) :=
        wsum_le p hp hL
      have e1 : ∑ i, p i * (ω * (1 - clean (c i))) = ω - ω * ∑ i, p i * clean (c i) :=
        wsum_affine p hs ω _
      linarith
    · exact wsum_le p hp fun i => Rbd_le_Rflat ω hω (c i) (hc i) (hfc i)

theorem Rflat_le_clean (ω : ℝ) (hω : 0 ≤ ω) :
    ∀ t : DTree, WF t → Rflat ω t ≤ ω * clean t
  | leaf _, _ => by
    simp only [Rflat, clean]; split_ifs <;> linarith
  | node H n p c, ⟨hp, hs, hc⟩ => by
    simp only [Rflat, clean]
    split_ifs
    · exact mul_nonneg hω (Finset.sum_nonneg fun i _ =>
        mul_nonneg (hp i) (clean_nonneg (c i) (hc i)))
    · exact le_rfl
    · calc ∑ i, p i * Rflat ω (c i) ≤ ∑ i, p i * (ω * clean (c i)) :=
            wsum_le p hp fun i => Rflat_le_clean ω hω (c i) (hc i)
        _ = ω * ∑ i, p i * clean (c i) := by
            rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ => by ring
where
  clean_nonneg : ∀ t : DTree, WF t → 0 ≤ clean t
    | leaf e, _ => by simp only [clean]; split_ifs <;> norm_num
    | node _ _ p c, ⟨hp, _, hc⟩ => Finset.sum_nonneg fun i _ =>
        mul_nonneg (hp i) (clean_nonneg (c i) (hc i))

/-- **Proposition 3, flat prices.** The regret of the myopic rule is at
most `ω · P(σ⁰ < σ*, T = ∞) ≤ ω · P(T = ∞)`. -/
theorem price_flat (ω : ℝ) (hω : 0 ≤ ω) (t : DTree) (hw : WF t) (hf : Flat ω t) :
    J0 t - V t ≤ Rflat ω t ∧ Rflat ω t ≤ ω * clean t :=
  ⟨le_trans (price_of_overtriggering t hw).2.2 (Rbd_le_Rflat ω hω t hw hf),
   Rflat_le_clean ω hω t hw⟩

end Baton.Tree
