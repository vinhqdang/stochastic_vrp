import Mathlib

/-!
Proposition 1 (endpoint-label bias) of the BATON manuscript.

Setting. A finite set `Ω` of days (scenarios) with probabilities `q`;
day `ω` has running load `W ω k` after stop `k = 1, …, m`, slack `B`,
and breach time `T ω` = first stop with `W ω k > B` (`m + 1` if none,
standing for `∞`). A handoff rule is a map `σ : Ω → ℕ` (hand off after
stop `σ ω`; `σ ω ≥ m` means never). Prices are flat: a handoff costs `ω₀`,
a breach `Cf`, with `0 ≤ ω₀ ≤ Cf`.

Admissible rules. The optimum `V⋆` is the infimum of the expected cost
over a class `𝒜` of admissible (non-anticipative) rules. The proof only
needs that every admissible rule acts no earlier than after stop 1, and
that the two constant rules "never hand off" and "hand off after stop 1"
are admissible; constant rules are adapted to every filtration, so this
holds for the class of stopping times of the paper.
-/

open Finset

set_option linter.unusedSectionVars false

namespace Baton.Endpoint

variable {Ω : Type*}

section Setup
variable (m : ℕ) (W : Ω → ℕ → ℝ) (B : ℝ)

open Classical in
/-- Breach time: the first stop `k ∈ [1, m]` with `W k > B`, else `m + 1`. -/
noncomputable def T (ω : Ω) : ℕ :=
  if h : ∃ k, 1 ≤ k ∧ k ≤ m ∧ B < W ω k then Nat.find h else m + 1

theorem one_le_T (ω : Ω) : 1 ≤ T m W B ω := by
  unfold T
  split_ifs with h
  · exact (Nat.find_spec h).1
  · omega

theorem T_le_m_iff (ω : Ω) : T m W B ω ≤ m ↔ ∃ k, 1 ≤ k ∧ k ≤ m ∧ B < W ω k := by
  unfold T
  split_ifs with h
  · exact ⟨fun _ => h, fun _ => (Nat.find_spec h).2.1⟩
  · exact ⟨fun h' => by omega, fun h' => absurd h' h⟩

/-- **Proposition 1 (inclusion).** An endpoint overflow is a peak
overflow. -/
theorem endpoint_subset_peak (hm : 1 ≤ m) (ω : Ω) (h : B < W ω m) : T m W B ω ≤ m :=
  (T_le_m_iff m W B ω).mpr ⟨m, hm, le_rfl, h⟩

variable [Fintype Ω]

/-- Probability of an event. -/
noncomputable def prob (q : Ω → ℝ) (A : Ω → Prop) [DecidablePred A] : ℝ :=
  ∑ ω, if A ω then q ω else 0

open Classical in
theorem prob_endpoint_le_peak (hm : 1 ≤ m) (q : Ω → ℝ) (hq : ∀ ω, 0 ≤ q ω) :
    prob q (fun ω => B < W ω m) ≤ prob q (fun ω => T m W B ω ≤ m) := by
  unfold prob
  refine Finset.sum_le_sum fun ω _ => ?_
  by_cases h : B < W ω m
  · simp [h, endpoint_subset_peak m W B hm ω h]
  · simp only [h, ite_false]; split_ifs <;> simp [hq ω]

/-- Collect-then-deliver structure: if every day ends with `W m ≤ 0 < B`,
the endpoint label is identically zero ... -/
theorem endpoint_label_zero (hB : 0 < B) (hW : ∀ ω, W ω m ≤ 0) (ω : Ω) :
    ¬ B < W ω m := fun h => by linarith [hW ω]

/-- ... so every empirical frequency of the endpoint label over any set of
training days is zero, and a trigger `p̂ > τ` with `τ > 0` never fires:
the endpoint-trained policy is the reactive policy. -/
theorem endpoint_frequency_zero (hB : 0 < B) (hW : ∀ ω, W ω m ≤ 0)
    (S : Finset Ω) (τ : ℝ) (hτ : 0 < τ) :
    ¬ τ < (∑ ω ∈ S, (if B < W ω m then (1 : ℝ) else 0)) / S.card := by
  have : (∑ ω ∈ S, (if B < W ω m then (1 : ℝ) else 0)) = 0 :=
    Finset.sum_eq_zero fun ω _ => by simp [endpoint_label_zero m W B hB hW ω]
  rw [this, zero_div]
  exact not_lt.mpr hτ.le

end Setup

section Bounds
variable [Fintype Ω] (m : ℕ) (T : Ω → ℕ) (q : Ω → ℝ) (ω₀ Cf : ℝ)

/-- Realised cost of the handoff rule `σ` on day `ω` under flat prices. -/
noncomputable def J (σ : Ω → ℕ) (ω : Ω) : ℝ :=
  if T ω ≤ m ∧ T ω ≤ σ ω then Cf
  else if σ ω < m ∧ σ ω < T ω then ω₀ else 0

noncomputable def EJ (σ : Ω → ℕ) : ℝ := ∑ ω, q ω * J m T ω₀ Cf σ ω

/-- The clairvoyant lower bound `Cf·1{T=1} + ω₀·1{2≤T≤m}`, pathwise. -/
noncomputable def lb (ω : Ω) : ℝ :=
  if T ω = 1 then Cf else if 2 ≤ T ω ∧ T ω ≤ m then ω₀ else 0

variable {m T q ω₀ Cf}

theorem J_ge_lb (hm1 : 1 ≤ m) (hT : ∀ ω, 1 ≤ T ω) (hω : 0 ≤ ω₀) (hωC : ω₀ ≤ Cf)
    (σ : Ω → ℕ) (hσ : ∀ ω, 1 ≤ σ ω) (ω : Ω) : lb m T ω₀ Cf ω ≤ J m T ω₀ Cf σ ω := by
  have h1 := hT ω
  have h2 := hσ ω
  unfold lb J
  split_ifs <;> first | exact le_rfl | exact hωC | exact hω | (exfalso; omega)

theorem EJ_ge (hm1 : 1 ≤ m) (hq : ∀ ω, 0 ≤ q ω) (hT : ∀ ω, 1 ≤ T ω) (hω : 0 ≤ ω₀) (hωC : ω₀ ≤ Cf)
    (σ : Ω → ℕ) (hσ : ∀ ω, 1 ≤ σ ω) :
    ∑ ω, q ω * lb m T ω₀ Cf ω ≤ EJ m T q ω₀ Cf σ :=
  Finset.sum_le_sum fun ω _ =>
    mul_le_mul_of_nonneg_left (J_ge_lb hm1 hT hω hωC σ hσ ω) (hq ω)

/-- Cost of "never hand off" (the reactive policy). -/
theorem J_never (ω : Ω) :
    J m T ω₀ Cf (fun _ => m) ω = if T ω ≤ m then Cf else 0 := by
  unfold J
  dsimp only
  split_ifs <;> first | rfl | omega

/-- Cost of "hand off after stop 1" (needs a decision epoch: `m ≥ 2`). -/
theorem J_one (hm : 2 ≤ m) (hT : ∀ ω, 1 ≤ T ω) (ω : Ω) :
    J m T ω₀ Cf (fun _ => 1) ω = if T ω = 1 then Cf else ω₀ := by
  have := hT ω
  unfold J
  dsimp only
  split_ifs <;> first | rfl | omega

set_option linter.unusedVariables false in
/-- **Proposition 1 (bounds).** Let `V⋆` be the infimum of the expected
cost over an admissible class containing "never" and "after stop 1", all
of whose rules act after stop 1 or later. The loss of the endpoint-trained
(= reactive) policy, `Δ = V_react − V⋆`, satisfies
`max(0, (Cf−ω₀)·P(2≤T≤m) − ω₀·P(T=∞)) ≤ Δ ≤ (Cf−ω₀)·P(2≤T≤m)`. -/
theorem endpoint_loss_bounds (hm : 2 ≤ m) (hq : ∀ ω, 0 ≤ q ω)
    (hT : ∀ ω, 1 ≤ T ω) (hω : 0 ≤ ω₀) (hωC : ω₀ ≤ Cf)
    (𝒜 : Set (Ω → ℕ)) (hnever : (fun _ => m) ∈ 𝒜) (hone : (fun _ => 1) ∈ 𝒜)
    (hadm : ∀ σ ∈ 𝒜, ∀ ω, 1 ≤ σ ω) :
    let Vstar := ⨅ σ : 𝒜, EJ m T q ω₀ Cf σ.1
    let Δ := EJ m T q ω₀ Cf (fun _ => m) - Vstar
    let P1 := ∑ ω, if T ω = 1 then q ω else 0
    let Pmid := ∑ ω, if 2 ≤ T ω ∧ T ω ≤ m then q ω else 0
    let Pinf := ∑ ω, if m < T ω then q ω else 0
    0 ≤ Δ ∧ (Cf - ω₀) * Pmid - ω₀ * Pinf ≤ Δ ∧ Δ ≤ (Cf - ω₀) * Pmid := by
  intro Vstar Δ P1 Pmid Pinf
  have hm1 : 1 ≤ m := by omega
  have hne : Nonempty 𝒜 := ⟨⟨_, hnever⟩⟩
  have hbdd : BddBelow (Set.range fun σ : 𝒜 => EJ m T q ω₀ Cf σ.1) :=
    ⟨∑ ω, q ω * lb m T ω₀ Cf ω, by
      rintro _ ⟨σ, rfl⟩; exact EJ_ge hm1 hq hT hω hωC σ.1 (hadm σ.1 σ.2)⟩
  have hlow : ∑ ω, q ω * lb m T ω₀ Cf ω ≤ Vstar :=
    le_ciInf fun σ => EJ_ge hm1 hq hT hω hωC σ.1 (hadm σ.1 σ.2)
  have hup_never : Vstar ≤ EJ m T q ω₀ Cf (fun _ => m) :=
    ciInf_le hbdd ⟨_, hnever⟩
  have hup_one : Vstar ≤ EJ m T q ω₀ Cf (fun _ => 1) :=
    ciInf_le hbdd ⟨_, hone⟩
  -- closed forms
  have e_never : EJ m T q ω₀ Cf (fun _ => m) = Cf * P1 + Cf * Pmid := by
    unfold EJ
    simp only [J_never, P1, Pmid, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ω _ => ?_
    have := hT ω
    split_ifs <;> first | (exfalso; omega) | ring
  have e_one : EJ m T q ω₀ Cf (fun _ => 1) = Cf * P1 + ω₀ * Pmid + ω₀ * Pinf := by
    unfold EJ
    simp only [J_one hm hT, P1, Pmid, Pinf, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ω _ => ?_
    have := hT ω
    split_ifs <;> first | (exfalso; omega) | ring
  have e_lb : ∑ ω, q ω * lb m T ω₀ Cf ω = Cf * P1 + ω₀ * Pmid := by
    simp only [lb, P1, Pmid, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ω _ => ?_
    split_ifs <;> first | (exfalso; omega) | ring
  refine ⟨?_, ?_, ?_⟩
  · show 0 ≤ EJ m T q ω₀ Cf (fun _ => m) - Vstar
    linarith
  · show (Cf - ω₀) * Pmid - ω₀ * Pinf ≤ EJ m T q ω₀ Cf (fun _ => m) - Vstar
    rw [e_never]; rw [e_one] at hup_one; linarith
  · show EJ m T q ω₀ Cf (fun _ => m) - Vstar ≤ (Cf - ω₀) * Pmid
    rw [e_never]; rw [e_lb] at hlow; linarith

/-- **Proposition 1 (attainment).** If some admissible rule pays the
clairvoyant cost on every day (a handoff at `T − 1` is admissible when
the breach is announced one stop ahead), the upper bound is attained. -/
theorem endpoint_loss_attained (hm1 : 1 ≤ m) (hq : ∀ ω, 0 ≤ q ω)
    (hT : ∀ ω, 1 ≤ T ω) (hω : 0 ≤ ω₀) (hωC : ω₀ ≤ Cf)
    (𝒜 : Set (Ω → ℕ)) (hadm : ∀ σ ∈ 𝒜, ∀ ω, 1 ≤ σ ω)
    (σc : Ω → ℕ) (hσc : σc ∈ 𝒜) (hclair : ∀ ω, J m T ω₀ Cf σc ω = lb m T ω₀ Cf ω) :
    (⨅ σ : 𝒜, EJ m T q ω₀ Cf σ.1) = ∑ ω, q ω * lb m T ω₀ Cf ω := by
  have hbdd : BddBelow (Set.range fun σ : 𝒜 => EJ m T q ω₀ Cf σ.1) :=
    ⟨∑ ω, q ω * lb m T ω₀ Cf ω, by
      rintro _ ⟨σ, rfl⟩; exact EJ_ge hm1 hq hT hω hωC σ.1 (hadm σ.1 σ.2)⟩
  have hne : Nonempty 𝒜 := ⟨⟨_, hσc⟩⟩
  refine le_antisymm ?_ (le_ciInf fun σ => EJ_ge hm1 hq hT hω hωC σ.1 (hadm σ.1 σ.2))
  calc (⨅ σ : 𝒜, EJ m T q ω₀ Cf σ.1) ≤ EJ m T q ω₀ Cf σc := ciInf_le hbdd ⟨_, hσc⟩
    _ = ∑ ω, q ω * lb m T ω₀ Cf ω := by unfold EJ; simp [hclair]

end Bounds

end Baton.Endpoint
