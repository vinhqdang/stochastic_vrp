import Mathlib

/-!
Proposition 2, last claim: strict inclusion of the stopping regions near
the myopic boundary.

Let `C` (optimal continuation value) and `C0` (never-act cost) be
continuous functions of the load, with `C0` nondecreasing, and let `H` be
the handoff price. Let `wbar` be the myopic boundary, the infimum of
`{w | H < C0 w}`. If `C wbar < C0 wbar` (the strictness condition of
Proposition 2 at the boundary), then `C0 wbar = H`, and on a
right-neighbourhood of `wbar` the myopic rule hands off (`H < C0 w`) while
the optimal rule continues (`C w < H`).
-/

open Filter Topology Set

namespace Baton.Boundary

theorem myopic_boundary (C C0 : ℝ → ℝ) (H wbar : ℝ)
    (hC : Continuous C) (hC0 : Continuous C0) (hmono : Monotone C0)
    (hglb : IsGLB {w | H < C0 w} wbar) (hstrict : C wbar < C0 wbar) :
    C0 wbar = H ∧ ∃ ε > 0, ∀ w, wbar < w → w < wbar + ε → C w < H ∧ H < C0 w := by
  -- every load above the boundary is myopically stopped
  have above : ∀ w, wbar < w → H < C0 w := by
    intro w hw
    obtain ⟨w', hw'S, -, hw'lt⟩ := hglb.exists_between hw
    exact lt_of_lt_of_le hw'S (hmono hw'lt.le)
  -- C0 wbar ≤ H: otherwise the region would extend to the left of wbar
  have hle : C0 wbar ≤ H := by
    by_contra hgt
    push Not at hgt
    have hev : ∀ᶠ w in 𝓝 wbar, H < C0 w := hC0.continuousAt.eventually (lt_mem_nhds hgt)
    have hev' : ∀ᶠ w in 𝓝[<] wbar, H < C0 w := hev.filter_mono nhdsWithin_le_nhds
    have hlt : ∀ᶠ w in 𝓝[<] wbar, w < wbar := self_mem_nhdsWithin
    obtain ⟨w, hwS, hwlt⟩ := (hev'.and hlt).exists
    exact absurd (hglb.1 hwS) (not_le.mpr hwlt)
  -- C0 wbar ≥ H: limit from the right of values above H
  have hge : H ≤ C0 wbar := by
    have ht : Tendsto C0 (𝓝[>] wbar) (𝓝 (C0 wbar)) :=
      (hC0.tendsto wbar).mono_left nhdsWithin_le_nhds
    have hev : ∀ᶠ w in 𝓝[>] wbar, H ≤ C0 w :=
      eventually_nhdsWithin_of_forall fun w hw => (above w hw).le
    exact ge_of_tendsto ht hev
  have heq : C0 wbar = H := le_antisymm hle hge
  refine ⟨heq, ?_⟩
  -- C < H near wbar by continuity of C
  have hCw : C wbar < H := heq ▸ hstrict
  have hev : ∀ᶠ w in 𝓝 wbar, C w < H := hC.continuousAt.eventually (gt_mem_nhds hCw)
  obtain ⟨ε, hε, hball⟩ := Metric.eventually_nhds_iff.mp hev
  refine ⟨ε, hε, fun w hw1 hw2 => ⟨hball ?_, above w hw1⟩⟩
  rw [Real.dist_eq, abs_lt]
  constructor <;> linarith

end Baton.Boundary
