import MwhedProofs.Fptas

/-!
# The refined FPTAS guarantee and its tightness (Theorem `thm:fptas`(b) and
# Proposition `prop:tight` of the revised paper)

Everything here is *additional* to `Fptas.lean` (which is not modified).

## Part R: the refined guarantee

Notation of the paper (case `K > 1`, `n ≥ 2`): `y = n/ε`, `N = ⌊y⌋`, `f = y - N`,
`r_i = w_i - K w'_i ∈ [0,K)`, `h` a site with `K = ε w_h / n` (the heaviest site).

* `resid`, `resid_nonneg`, `resid_lt`   : the rounding residuals `r_i ∈ [0, K)`.
* `fptasRef_F1`  (F1) : `K · v̂ ≤ W(Ŝ)`.
* `fptasRef_F2`  (F2) : `w'_h ≤ v̂` (the singleton `{h}` is feasible).
* `fptasRef_F3`  (F3) : `W(S) - W(Ŝ) ≤ Σ_{i ∈ S \ Ŝ} r_i` for every feasible `S`.
* `refinedMax n ε = max {(n-1)/y, (n-2+f)/N}`, `rhoRef n ε = 1/(1 + refinedMax n ε)`.
* `fptasRef_core` : `W(S) ≤ (1 + refinedMax n ε) · W(Ŝ)` for every feasible `S`
  (the case analysis `h ∈ Ŝ`, `h ∈ A`, `h ∉ A ∪ Ŝ`).
* `rhoRef_ge`, `rhoRef_gt` : `ρ ≥ N/(N+n-1) > 1/(1+ε) > 1-ε`.
* `fptas_refined`  : the packaged Theorem 4 over MWHED (cases (a) `K = 1` and (b) `K > 1`),
  with `ρ·W* ≤ W(Ŝ)`, `W*/(1+ε) < W(Ŝ)`, `(1-ε) W* < W(Ŝ)`.

Differences from the LaTeX: the core lemma does not use that `h` is *heaviest* nor that `S`
is optimal (the proof never needs it): it holds for every feasible `S` and every site `h`
with `{h}` feasible and `K = ε w_h / n`.  The heaviest-site hypothesis enters only in the
wrapper (`fptas_refined`), where `K = fptasK ε w` forces `K = ε w_max / n`.
-/

open Finset

namespace Mwhed

section Refined

variable {ι : Type*}

/-- The rounding residual `r_i = w_i - K w'_i`. -/
noncomputable def resid (K : ℝ) (w : ι → ℕ) (i : ι) : ℝ := (w i : ℝ) - K * (scaledW K w i : ℝ)

theorem resid_nonneg {K : ℝ} (hK : 0 < K) (w : ι → ℕ) (i : ι) : 0 ≤ resid K w i := by
  have := mul_scaledW_le hK w i
  unfold resid; linarith

theorem resid_lt {K : ℝ} (hK : 0 < K) (w : ι → ℕ) (i : ι) : resid K w i < K := by
  have := lt_mul_scaledW_succ hK w i
  unfold resid; linarith

/-- **(F1)** `W(Ŝ) ≥ K · v̂`, where `v̂ = Σ_{Ŝ} w'`. -/
theorem fptasRef_F1 {K : ℝ} (hK : 0 < K) (w : ι → ℕ) (Sh : Finset ι) :
    K * ∑ i ∈ Sh, (scaledW K w i : ℝ) ≤ ∑ i ∈ Sh, (w i : ℝ) := by
  rw [mul_sum]; exact sum_le_sum fun i _ => mul_scaledW_le hK w i

/-- **(F2)** If `{h}` is feasible and `Ŝ` maximises the scaled value, `v̂ ≥ w'_h`. -/
theorem fptasRef_F2 {F : Finset ι → Prop} {w : ι → ℕ} {K : ℝ} {Sh : Finset ι}
    (hSh : IsScaledMax F K w Sh) {h : ι} (hh : F {h}) :
    scaledW K w h ≤ ∑ i ∈ Sh, scaledW K w i := by
  simpa using hSh.2 {h} hh

variable [DecidableEq ι]

/-- **(F3)** `W(S) - W(Ŝ) ≤ Σ_{i ∈ S \ Ŝ} r_i` for every feasible `S`. -/
theorem fptasRef_F3 {F : Finset ι → Prop} {w : ι → ℕ} {K : ℝ} (hK : 0 < K)
    {Sh : Finset ι} (hSh : IsScaledMax F K w Sh) {S : Finset ι} (hS : F S) :
    (∑ i ∈ S, (w i : ℝ)) - ∑ i ∈ Sh, (w i : ℝ) ≤ ∑ i ∈ S \ Sh, resid K w i := by
  have hs : ∀ (f : ι → ℝ), ∑ i ∈ S, f i = ∑ i ∈ S \ Sh, f i + ∑ i ∈ S ∩ Sh, f i := by
    intro f; rw [add_comm]; exact (sum_inter_add_sum_sdiff S Sh f).symm
  have ht : ∀ (f : ι → ℝ), ∑ i ∈ Sh, f i = ∑ i ∈ Sh \ S, f i + ∑ i ∈ S ∩ Sh, f i := by
    intro f; rw [add_comm, inter_comm]; exact (sum_inter_add_sum_sdiff Sh S f).symm
  have h3 : (∑ i ∈ S, (scaledW K w i : ℝ)) ≤ ∑ i ∈ Sh, (scaledW K w i : ℝ) := by
    exact_mod_cast hSh.2 S hS
  rw [hs, ht] at h3
  have h4 : ∑ i ∈ S \ Sh, (scaledW K w i : ℝ) ≤ ∑ i ∈ Sh \ S, (scaledW K w i : ℝ) := by linarith
  have h5 : K * ∑ i ∈ Sh \ S, (scaledW K w i : ℝ) ≤ ∑ i ∈ Sh \ S, (w i : ℝ) := fptasRef_F1 hK w _
  have h6 : ∑ i ∈ S \ Sh, resid K w i
      = ∑ i ∈ S \ Sh, (w i : ℝ) - K * ∑ i ∈ S \ Sh, (scaledW K w i : ℝ) := by
    simp only [resid, sum_sub_distrib, mul_sum]
  rw [hs (fun i => (w i : ℝ)), ht (fun i => (w i : ℝ)), h6]
  nlinarith [mul_le_mul_of_nonneg_left h4 hK.le]


/-- `max {(n-1)/y, (n-2+f)/N}` with `y = n/ε`, `N = ⌊y⌋`, `f = y - N`. -/
noncomputable def refinedMax (n : ℕ) (ε : ℝ) : ℝ :=
  max (((n : ℝ) - 1) / ((n : ℝ) / ε))
      (((n : ℝ) - 2 + ((n : ℝ) / ε - ⌊(n : ℝ) / ε⌋₊)) / ⌊(n : ℝ) / ε⌋₊)

/-- The sharp constant `ρ_{n,ε} = (1 + max {(n-1)/y, (n-2+f)/N})⁻¹` of Theorem 4(b). -/
noncomputable def rhoRef (n : ℕ) (ε : ℝ) : ℝ := (1 + refinedMax n ε)⁻¹

/-- **Core of Theorem 4(b)** (facts F1-F3 and the case analysis `h ∈ Ŝ`, `h ∈ A`,
`h ∉ A ∪ Ŝ`): for every feasible `S`,
`W(S) ≤ (1 + max {(n-1)/y, (n-2+f)/N}) · W(Ŝ)`.
Here `K = ε w_h / n > 1` for a site `h` with `{h}` feasible (in the paper `h` is a heaviest
site). -/
theorem fptasRef_core [Fintype ι] {F : Finset ι → Prop} {w : ι → ℕ} {ε K : ℝ}
    (hn : 2 ≤ Fintype.card ι) (hε0 : 0 < ε) (hε1 : ε < 1) {h : ι}
    (hK : K = ε * (w h : ℝ) / Fintype.card ι) (hK1 : 1 < K) (hFh : F {h})
    {Sh : Finset ι} (hSh : IsScaledMax F K w Sh) {S : Finset ι} (hS : F S) :
    ∑ i ∈ S, (w i : ℝ) ≤ (1 + refinedMax (Fintype.card ι) ε) * ∑ i ∈ Sh, (w i : ℝ) := by
  have hnR : (2 : ℝ) ≤ (Fintype.card ι : ℝ) := by exact_mod_cast hn
  have hK0 : 0 < K := by linarith
  have hy_pos : 0 < (Fintype.card ι : ℝ) / ε := by positivity
  have hy2 : 2 < (Fintype.card ι : ℝ) / ε := by
    rw [lt_div_iff₀ hε0]; nlinarith
  have hwh : (w h : ℝ) = K * ((Fintype.card ι : ℝ) / ε) := by
    rw [hK]; field_simp
  have hN1 : 1 ≤ ⌊(Fintype.card ι : ℝ) / ε⌋₊ := (Nat.one_le_floor_iff _).2 (by linarith)
  have hNpos : (0 : ℝ) < (⌊(Fintype.card ι : ℝ) / ε⌋₊ : ℝ) := by exact_mod_cast hN1
  have hf0 : 0 ≤ (Fintype.card ι : ℝ) / ε - (⌊(Fintype.card ι : ℝ) / ε⌋₊ : ℝ) := by
    have := Nat.floor_le hy_pos.le; linarith
  have hf1 : (Fintype.card ι : ℝ) / ε - (⌊(Fintype.card ι : ℝ) / ε⌋₊ : ℝ) < 1 := by
    have := Nat.lt_floor_add_one ((Fintype.card ι : ℝ) / ε); linarith
  generalize hn' : Fintype.card ι = n at *
  generalize hy : (n : ℝ) / ε = y at *
  generalize hN : ⌊y⌋₊ = N at *
  generalize hf : y - (N : ℝ) = f at *
  have hwh' : scaledW K w h = N := by
    unfold scaledW
    rw [hwh, mul_div_cancel_left₀ _ hK0.ne', hN]
  have hrh : resid K w h = K * f := by
    unfold resid; rw [hwh, hwh', ← hf]; ring
  have hx0 : 0 ≤ ∑ i ∈ Sh, (w i : ℝ) := sum_nonneg fun _ _ => Nat.cast_nonneg _
  have hxN : K * N ≤ ∑ i ∈ Sh, (w i : ℝ) := by
    have h1 := fptasRef_F1 hK0 w Sh
    have h2 : (N : ℝ) ≤ ∑ i ∈ Sh, (scaledW K w i : ℝ) := by
      have := fptasRef_F2 hSh hFh
      rw [hwh'] at this; exact_mod_cast this
    nlinarith [mul_le_mul_of_nonneg_left h2 hK0.le]
  have hShne : Sh.Nonempty := by
    by_contra hne
    rw [not_nonempty_iff_eq_empty] at hne
    subst hne
    have : 0 < K * N := mul_pos hK0 hNpos
    simp at hxN
    linarith
  have hF3 := fptasRef_F3 hK0 hSh hS
  have hdisj : Disjoint (S \ Sh) Sh := sdiff_disjoint
  have hAcard : (S \ Sh).card + Sh.card ≤ n := by
    rw [← card_union_of_disjoint hdisj, ← hn']; exact card_le_univ _
  have hShcard : 1 ≤ Sh.card := hShne.card_pos
  have hsumK : ∀ B : Finset ι, ∑ i ∈ B, resid K w i ≤ K * B.card := by
    intro B
    calc ∑ i ∈ B, resid K w i ≤ ∑ i ∈ B, K := sum_le_sum fun i _ => (resid_lt hK0 w i).le
      _ = K * B.card := by simp [mul_comm]
  have hm : refinedMax n ε = max ((n - 1) / y) ((n - 2 + f) / N) := by
    rw [refinedMax, hy, hN, hf]
  rw [hm]
  have e1 : (1 + max (((n : ℝ) - 1) / y) (((n : ℝ) - 2 + f) / N)) * ∑ i ∈ Sh, (w i : ℝ)
      = ∑ i ∈ Sh, (w i : ℝ) + max (((n : ℝ) - 1) / y) (((n : ℝ) - 2 + f) / N) *
          ∑ i ∈ Sh, (w i : ℝ) := by ring
  rw [e1]
  by_cases hhS : h ∈ Sh
  · have hxy : K * y ≤ ∑ i ∈ Sh, (w i : ℝ) := by
      rw [← hwh]
      exact single_le_sum (f := fun i => (w i : ℝ)) (fun _ _ => Nat.cast_nonneg _) hhS
    have hR : ∑ i ∈ S \ Sh, resid K w i ≤ K * ((n : ℝ) - 1) := by
      refine (hsumK _).trans ?_
      apply mul_le_mul_of_nonneg_left _ hK0.le
      have : ((S \ Sh).card : ℝ) + 1 ≤ n := by exact_mod_cast (by omega : (S \ Sh).card + 1 ≤ n)
      linarith
    have ha0 : 0 ≤ ((n : ℝ) - 1) / y := by apply div_nonneg <;> linarith
    have h1 : K * ((n : ℝ) - 1) ≤ ((n : ℝ) - 1) / y * ∑ i ∈ Sh, (w i : ℝ) := by
      have h0 := mul_le_mul_of_nonneg_left hxy ha0
      have e : ((n : ℝ) - 1) / y * (K * y) = K * ((n : ℝ) - 1) := by field_simp
      linarith
    have h2 : ((n : ℝ) - 1) / y * ∑ i ∈ Sh, (w i : ℝ) ≤
        max (((n : ℝ) - 1) / y) (((n : ℝ) - 2 + f) / N) * ∑ i ∈ Sh, (w i : ℝ) :=
      mul_le_mul_of_nonneg_right (le_max_left _ _) hx0
    linarith
  · have hR : ∑ i ∈ S \ Sh, resid K w i ≤ K * ((n : ℝ) - 2 + f) := by
      by_cases hhA : h ∈ S \ Sh
      · rw [← add_sum_erase _ _ hhA, hrh]
        have h1 := hsumK ((S \ Sh).erase h)
        have hApos : 0 < (S \ Sh).card := card_pos.2 ⟨h, hhA⟩
        have hc : (((S \ Sh).erase h).card : ℝ) + 2 ≤ n := by
          rw [card_erase_of_mem hhA]
          exact_mod_cast (by omega : (S \ Sh).card - 1 + 2 ≤ n)
        nlinarith
      · have hsub : (S \ Sh) ∪ Sh ⊆ univ.erase h := by
          intro x hx
          rw [mem_erase]
          refine ⟨?_, mem_univ _⟩
          rintro rfl
          rcases mem_union.1 hx with hx | hx
          · exact hhA hx
          · exact hhS hx
        have hc1 := card_le_card hsub
        rw [card_union_of_disjoint hdisj, card_erase_of_mem (mem_univ _), card_univ, hn'] at hc1
        have hc : ((S \ Sh).card : ℝ) + 2 ≤ n := by
          exact_mod_cast (by omega : (S \ Sh).card + 2 ≤ n)
        have h1 := hsumK (S \ Sh)
        have : K * (((S \ Sh).card : ℝ)) ≤ K * ((n : ℝ) - 2) :=
          mul_le_mul_of_nonneg_left (by linarith) hK0.le
        nlinarith
    have hb0 : 0 ≤ ((n : ℝ) - 2 + f) / N := by apply div_nonneg <;> linarith
    have h1 : K * ((n : ℝ) - 2 + f) ≤ ((n : ℝ) - 2 + f) / N * ∑ i ∈ Sh, (w i : ℝ) := by
      have h0 := mul_le_mul_of_nonneg_left hxN hb0
      have e : ((n : ℝ) - 2 + f) / N * (K * N) = K * ((n : ℝ) - 2 + f) := by field_simp
      linarith
    have h2 : ((n : ℝ) - 2 + f) / N * ∑ i ∈ Sh, (w i : ℝ) ≤
        max (((n : ℝ) - 1) / y) (((n : ℝ) - 2 + f) / N) * ∑ i ∈ Sh, (w i : ℝ) :=
      mul_le_mul_of_nonneg_right (le_max_right _ _) hx0
    linarith

end Refined

/-! ### The constant `ρ_{n,ε}` -/

section Rho

variable {n : ℕ} {ε : ℝ}

/-- Elementary facts on `y = n/ε`, `N = ⌊y⌋`. -/
theorem rho_facts (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) :
    2 < (n : ℝ) / ε ∧ 1 ≤ ⌊(n : ℝ) / ε⌋₊ ∧ ((⌊(n : ℝ) / ε⌋₊ : ℕ) : ℝ) ≤ (n : ℝ) / ε ∧
      (n : ℝ) / ε < ((⌊(n : ℝ) / ε⌋₊ : ℕ) : ℝ) + 1 := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hy2 : 2 < (n : ℝ) / ε := by rw [lt_div_iff₀ hε0]; nlinarith
  exact ⟨hy2, (Nat.one_le_floor_iff _).2 (by linarith), Nat.floor_le (by linarith),
    Nat.lt_floor_add_one _⟩

theorem refinedMax_nonneg (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) : 0 ≤ refinedMax n ε := by
  obtain ⟨hy2, -, -, -⟩ := rho_facts hn hε0 hε1
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  unfold refinedMax
  refine le_max_of_le_left (div_nonneg (by linarith) (by linarith))

/-- `max {(n-1)/y, (n-2+f)/N} ≤ (n-1)/N`. -/
theorem refinedMax_le (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) :
    refinedMax n ε ≤ ((n : ℝ) - 1) / ⌊(n : ℝ) / ε⌋₊ := by
  obtain ⟨hy2, hN1, hNy, hyN⟩ := rho_facts hn hε0 hε1
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hNpos : (0 : ℝ) < ⌊(n : ℝ) / ε⌋₊ := by exact_mod_cast hN1
  unfold refinedMax
  refine max_le ?_ ?_
  · exact div_le_div_of_nonneg_left (by linarith) hNpos hNy
  · exact div_le_div_of_nonneg_right (by linarith) hNpos.le

theorem rhoRef_pos (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) : 0 < rhoRef n ε := by
  have := refinedMax_nonneg hn hε0 hε1
  unfold rhoRef; positivity

/-- `ρ ≥ N/(N+n-1)`. -/
theorem rhoRef_ge (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) :
    (⌊(n : ℝ) / ε⌋₊ : ℝ) / (⌊(n : ℝ) / ε⌋₊ + n - 1) ≤ rhoRef n ε := by
  obtain ⟨hy2, hN1, hNy, hyN⟩ := rho_facts hn hε0 hε1
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hNpos : (0 : ℝ) < ⌊(n : ℝ) / ε⌋₊ := by exact_mod_cast hN1
  have h1 := refinedMax_le hn hε0 hε1
  have h0 := refinedMax_nonneg hn hε0 hε1
  unfold rhoRef
  have e : (⌊(n : ℝ) / ε⌋₊ : ℝ) / (⌊(n : ℝ) / ε⌋₊ + n - 1)
      = (1 + ((n : ℝ) - 1) / ⌊(n : ℝ) / ε⌋₊)⁻¹ := by
    field_simp; ring
  rw [e]
  exact inv_anti₀ (by linarith) (by linarith)

/-- `N/(N+n-1) > 1/(1+ε)`. -/
theorem inv_lt_floor_ratio (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) :
    1 / (1 + ε) < (⌊(n : ℝ) / ε⌋₊ : ℝ) / (⌊(n : ℝ) / ε⌋₊ + n - 1) := by
  obtain ⟨hy2, hN1, hNy, hyN⟩ := rho_facts hn hε0 hε1
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hNpos : (0 : ℝ) < ⌊(n : ℝ) / ε⌋₊ := by exact_mod_cast hN1
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  have hyε : (n : ℝ) / ε * ε = n := by field_simp
  have : (((n : ℝ) / ε) - 1) * ε < (⌊(n : ℝ) / ε⌋₊ : ℝ) * ε :=
    mul_lt_mul_of_pos_right (by linarith) hε0
  nlinarith

/-- `1/(1+ε) > 1-ε`. -/
theorem one_sub_lt_inv_one_add (hε0 : 0 < ε) (hε1 : ε < 1) : 1 - ε < 1 / (1 + ε) := by
  rw [lt_div_iff₀ (by linarith)]; nlinarith

/-- **`ρ_{n,ε} ≥ N/(N+n-1) > 1/(1+ε) > 1-ε`** (the closed-form consequence of Theorem 4(b)). -/
theorem rhoRef_chain (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) :
    (⌊(n : ℝ) / ε⌋₊ : ℝ) / (⌊(n : ℝ) / ε⌋₊ + n - 1) ≤ rhoRef n ε ∧
    1 / (1 + ε) < (⌊(n : ℝ) / ε⌋₊ : ℝ) / (⌊(n : ℝ) / ε⌋₊ + n - 1) ∧
    1 - ε < 1 / (1 + ε) :=
  ⟨rhoRef_ge hn hε0 hε1, inv_lt_floor_ratio hn hε0 hε1, one_sub_lt_inv_one_add hε0 hε1⟩

theorem inv_lt_rhoRef (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) : 1 / (1 + ε) < rhoRef n ε :=
  lt_of_lt_of_le (inv_lt_floor_ratio hn hε0 hε1) (rhoRef_ge hn hε0 hε1)

end Rho

/-! ### The refined theorem over a generic family and over MWHED -/

section Wrapper

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Theorem 4(b), generic family.**  `K = fptasK ε w = ε w_max / n > 1`, `n ≥ 2`, `{h}` feasible for
a heaviest site `h`: every feasible `S` has `W(S) ≤ (1 + max {(n-1)/y, (n-2+f)/N}) · W(Ŝ)`. -/
theorem fptasRef_generic {F : Finset ι → Prop} {w : ι → ℕ} {ε : ℝ}
    (hn : 2 ≤ Fintype.card ι) (hε0 : 0 < ε) (hε1 : ε < 1)
    {h : ι} (hmax : ∀ i, w i ≤ w h) (hFh : F {h})
    (hgt : (Fintype.card ι : ℝ) < ε * ((univ.sup w : ℕ) : ℝ))
    {Sh : Finset ι} (hSh : IsScaledMax F (fptasK ε w) w Sh) {S : Finset ι} (hS : F S) :
    ∑ i ∈ S, (w i : ℝ) ≤ (1 + refinedMax (Fintype.card ι) ε) * ∑ i ∈ Sh, (w i : ℝ) := by
  have hsup : univ.sup w = w h := le_antisymm (Finset.sup_le fun i _ => hmax i) (Finset.le_sup (mem_univ h))
  have hnpos : (0 : ℝ) < Fintype.card ι := by
    have : 0 < Fintype.card ι := by omega
    exact_mod_cast this
  rw [hsup] at hgt
  have hKgt : 1 < ε * (w h : ℝ) / Fintype.card ι := by
    rw [lt_div_iff₀ hnpos]; linarith
  have hK : fptasK ε w = ε * (w h : ℝ) / Fintype.card ι := by
    unfold fptasK; rw [hsup]; exact max_eq_right hKgt.le
  exact fptasRef_core hn hε0 hε1 hK (by rw [hK]; exact hKgt) hFh hSh hS

/-- **Theorem 4 (FPTAS), refined form, over MWHED** (`n ≥ 2`, Assumption 1).  Let `Ŝ` be a feasible
set maximising the scaled value for `K = max {1, ε w_max / n}`, and `W* = v`.

* (a) if `ε w_max ≤ n` (`K = 1`), `W(Ŝ) = W*`;
* (b) if `ε w_max > n`: `W* ≤ (1 + max{(n-1)/y,(n-2+f)/N}) W(Ŝ)`, i.e. `ρ_{n,ε} W* ≤ W(Ŝ)`, and
  `W*/(1+ε) < W(Ŝ)`, `(1-ε) W* < W(Ŝ)`. -/
theorem fptas_refined (I : Inst ι) (hI : IndivFeasible I) (hn : 2 ≤ Fintype.card ι)
    {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε < 1)
    {Sh : Finset ι} (hSh : IsScaledMax (Feasible I) (fptasK ε I.w) I.w Sh) {v : ℕ}
    (hv : IsOPT I v) :
    (ε * ((univ.sup I.w : ℕ) : ℝ) ≤ Fintype.card ι → weight I Sh = v) ∧
    ((Fintype.card ι : ℝ) < ε * ((univ.sup I.w : ℕ) : ℝ) →
      (v : ℝ) ≤ (1 + refinedMax (Fintype.card ι) ε) * (weight I Sh : ℝ) ∧
      rhoRef (Fintype.card ι) ε * v ≤ (weight I Sh : ℝ) ∧
      (v : ℝ) / (1 + ε) < (weight I Sh : ℝ) ∧
      (1 - ε) * (v : ℝ) < (weight I Sh : ℝ)) := by
  obtain ⟨⟨S0, hS0, hv0⟩, hvmax⟩ := (isOPT_iff_max_feasible I v).1 hv
  have hne : (univ : Finset ι).Nonempty := by
    rw [← Finset.card_pos, Finset.card_univ]; omega
  obtain ⟨h, -, hh⟩ := Finset.exists_mem_eq_sup univ hne I.w
  have hmax : ∀ i, I.w i ≤ I.w h := fun i => hh ▸ Finset.le_sup (mem_univ i)
  have hnpos : (0 : ℝ) < Fintype.card ι := by
    have : 0 < Fintype.card ι := by omega
    exact_mod_cast this
  constructor
  · intro hle
    have hK : fptasK ε I.w = 1 := by
      unfold fptasK
      exact max_eq_left ((div_le_one hnpos).2 hle)
    rw [hK] at hSh
    apply le_antisymm (hvmax Sh hSh.1)
    rw [← hv0]
    exact fptas_case_vacuous hSh hS0
  · intro hgt
    have hFh : Feasible I {h} := feasible_singleton I (hI h)
    have hcore := fptasRef_generic hn hε0 hε1 hmax hFh hgt hSh hS0
    have hvS : (v : ℝ) = ∑ i ∈ S0, (I.w i : ℝ) := by
      rw [← hv0]; simp [weight]
    have hW : (weight I Sh : ℝ) = ∑ i ∈ Sh, (I.w i : ℝ) := by simp [weight]
    have hm := refinedMax_nonneg hn hε0 hε1
    have hmain : (v : ℝ) ≤ (1 + refinedMax (Fintype.card ι) ε) * (weight I Sh : ℝ) := by
      rw [hvS, hW]; exact hcore
    have hvpos : (0 : ℝ) < v := by
      have h1 : I.w h ≤ v := by simpa [weight] using hvmax {h} hFh
      have h2 : (0 : ℝ) < I.w h := by
        have : (0 : ℝ) < ε * ((univ.sup I.w : ℕ) : ℝ) := lt_trans hnpos hgt
        rw [hh] at this
        have h3 : (0 : ℝ) < ((I.w h : ℕ) : ℝ) := by
          by_contra hc
          push Not at hc
          nlinarith
        exact h3
      have : (I.w h : ℝ) ≤ v := by exact_mod_cast h1
      linarith
    have hrho : rhoRef (Fintype.card ι) ε * v ≤ (weight I Sh : ℝ) := by
      unfold rhoRef
      rw [inv_mul_le_iff₀ (by linarith)]
      exact hmain
    have hlt := inv_lt_rhoRef hn hε0 hε1
    refine ⟨hmain, hrho, ?_, ?_⟩
    · have : (v : ℝ) / (1 + ε) < rhoRef (Fintype.card ι) ε * v := by
        rw [div_eq_mul_one_div, mul_comm]
        exact mul_lt_mul_of_pos_right hlt hvpos
      linarith
    · have h1 : (1 - ε) < rhoRef (Fintype.card ι) ε :=
        lt_trans (one_sub_lt_inv_one_add hε0 hε1) hlt
      have : (1 - ε) * (v : ℝ) < rhoRef (Fintype.card ι) ε * v :=
        mul_lt_mul_of_pos_right h1 hvpos
      linarith

end Wrapper

/-! ## Part T: tightness (Proposition `prop:tight`)

Setting (as in `Fptas.lean`, Part C): `ε ∈ (0,1)`, `n ≥ 2`, `M ≥ 2n/ε` (`TightHyp`), `K = εM/n`
(`tightK`), `y = n/ε`, `N = ⌊y⌋`, light weight `⌈K⌉ - 1` (`tightC`).

* **Family 1** is `tightInst` of `Fptas.lean` (`p = 1`, `d = n`, `w_1 = M`, `w_i = ⌈K⌉-1`): the
  algorithm output is `{1}` (`tight_algorithm_output`, re-exported as `tight1_algOutput_iff`) and
  `tight1_bound` : `W(output) ≤ B1(M) · W*` with `B1(M) = 1/(1 + (n-1)/y - (n-1)/M)`.
* **Family 2** is `tight2Inst`: site `b` = `(1,1,⌈KN⌉)`, site `h` = `(2,2,M)`, `n-2` sites
  `(1,n,⌈K⌉-1)`.  `tight2_algOutput_iff` : the algorithm output (a minimum-time maximiser of the
  scaled value) is exactly `{b}`; `tight2_bound` : `W(output) ≤ B2(M) · W*` with
  `B2(M) = (εN/n + 1/M)/(1 + (n-2)ε/n - (n-2)/M)`.
* `tendsto_tightB1`, `tendsto_tightB2` : `B1 → 1/(1+(n-1)/y)`, `B2 → 1/(1+(n-2+f)/N)`
  (exact filter limits as `M → ∞`).
* `prop_tight` : for every `η > 0` there is `M₀` such that for every `M ≥ M₀` there is an instance
  (Family 1 or Family 2, according to which term of `refinedMax` is larger) satisfying
  Assumption 1 on which the algorithm output `Ŝ` has `W(Ŝ) ≤ (ρ_{n,ε} + η) · W*`.
-/

section Tight

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- `Ŝ` is what Algorithm 2 returns: a feasible maximiser of the scaled value that has minimum total
dispatch time among all such maximisers (`v*` maximal, then the entry `g(n, v*)`). -/
def IsAlgOutput (I : Inst ι) (K : ℝ) (S : Finset ι) : Prop :=
  IsScaledMax (Feasible I) K I.w S ∧
    ∀ T, IsScaledMax (Feasible I) K I.w T → time I S ≤ time I T

end Tight

section Tight1

variable {ε : ℝ} {n M : ℕ}

/-- `B1(M) = 1 / (1 + ε(n-1)/n - (n-1)/M)`; note `ε(n-1)/n = (n-1)/y`. -/
noncomputable def tightB1 (ε : ℝ) (n M : ℕ) : ℝ :=
  1 / (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M)

/-- Family 1: the output of Algorithm 2 is exactly `{1}` (the paper's site 1, here `0`). -/
theorem tight1_algOutput_iff (h : TightHyp ε n M) (S : Finset (Fin n)) :
    IsAlgOutput (tightInst ε n M) (tightK ε n M) S ↔ S = {(⟨0, h.n_pos⟩ : Fin n)} :=
  tight_algorithm_output h S

/-- Family 1, ratio bound: `W({1}) ≤ B1(M) · W*` for every optimal value `W*`. -/
theorem tight1_bound (h : TightHyp ε n M) {v : ℕ} (hv : IsOPT (tightInst ε n M) v) :
    (weight (tightInst ε n M) {(⟨0, h.n_pos⟩ : Fin n)} : ℝ) ≤ tightB1 ε n M * v := by
  obtain ⟨hD, hr⟩ := tight_ratio h
  obtain ⟨-, hvmax⟩ := (isOPT_iff_max_feasible _ v).1 hv
  have hvW : M + (n - 1) * tightC ε n M ≤ v := by
    have := hvmax univ (tight_feasible _)
    rwa [tight_weight_univ h] at this
  have hMpos := h.M_pos_real
  have hWM : (M : ℝ) ≤ ((M + (n - 1) * tightC ε n M : ℕ) : ℝ) := by
    exact_mod_cast Nat.le_add_right _ _
  have hWpos : (0 : ℝ) < ((M + (n - 1) * tightC ε n M : ℕ) : ℝ) := lt_of_lt_of_le hMpos hWM
  rw [div_le_iff₀ hWpos] at hr
  rw [tight_returned_weight h]
  unfold tightB1
  have hvW' : ((M + (n - 1) * tightC ε n M : ℕ) : ℝ) ≤ (v : ℝ) := by exact_mod_cast hvW
  have hpos : (0 : ℝ) ≤ 1 / (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M) :=
    (one_div_pos.2 hD).le
  calc (M : ℝ) ≤ 1 / (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M) *
        ((M + (n - 1) * tightC ε n M : ℕ) : ℝ) := hr
    _ ≤ 1 / (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M) * v :=
        mul_le_mul_of_nonneg_left hvW' hpos

end Tight1

section Tight2

variable {ε : ℝ} {n M : ℕ}

/-- The weight `⌈K N⌉` of site `b` in Family 2. -/
noncomputable def tight2B (ε : ℝ) (n M : ℕ) : ℕ := ⌈tightK ε n M * ⌊(n : ℝ) / ε⌋₊⌉₊

/-- **Family 2** of Proposition 8 (sites `Fin n`; `0` is `b`, `1` is `h`, the rest are the `z_j`):
`b = (1,1,⌈KN⌉)`, `h = (2,2,M)`, `z_j = (1,n,⌈K⌉-1)`. -/
noncomputable def tight2Inst (ε : ℝ) (n M : ℕ) : Inst (Fin n) where
  p i := if i.val = 1 then 2 else 1
  d i := if i.val = 0 then 1 else if i.val = 1 then 2 else n
  w i := if i.val = 0 then tight2B ε n M else if i.val = 1 then M else tightC ε n M
  p_pos i := by split_ifs <;> omega

/-- The site `b`. -/
def siteB (hn : 2 ≤ n) : Fin n := ⟨0, by omega⟩

/-- The site `h`. -/
def siteH (hn : 2 ≤ n) : Fin n := ⟨1, by omega⟩

theorem siteB_ne_siteH (hn : 2 ≤ n) : siteB hn ≠ siteH hn := by
  intro e; have := congrArg Fin.val e; simp [siteB, siteH] at this

theorem TightHyp.K_mul_y (h : TightHyp ε n M) : tightK ε n M * ((n : ℝ) / ε) = M := by
  have := h.hε0; have := h.n_pos_real
  unfold tightK; field_simp

theorem TightHyp.K_lt_M (h : TightHyp ε n M) : tightK ε n M < M := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast h.n_pos
  have h1 : tightK ε n M ≤ ε * M := div_le_self (mul_nonneg h.hε0.le h.M_pos_real.le) hn1
  nlinarith [h.hε1, h.M_pos_real]

theorem tight2B_le (h : TightHyp ε n M) : tight2B ε n M ≤ M := by
  unfold tight2B
  apply Nat.ceil_le.2
  have hy : 0 ≤ (n : ℝ) / ε := by have := h.hε0; have := h.n_pos_real; positivity
  have hN : (⌊(n : ℝ) / ε⌋₊ : ℝ) ≤ (n : ℝ) / ε := Nat.floor_le hy
  calc tightK ε n M * (⌊(n : ℝ) / ε⌋₊ : ℝ) ≤ tightK ε n M * ((n : ℝ) / ε) :=
        mul_le_mul_of_nonneg_left hN h.K_pos.le
    _ = M := h.K_mul_y

theorem tight2B_scaled (h : TightHyp ε n M) :
    ⌊(tight2B ε n M : ℝ) / tightK ε n M⌋₊ = ⌊(n : ℝ) / ε⌋₊ := by
  have hK := h.K_pos
  have hK2 := h.K_ge_two
  have hN0 : (0 : ℝ) ≤ tightK ε n M * (⌊(n : ℝ) / ε⌋₊ : ℝ) := by positivity
  rw [Nat.floor_eq_iff (by positivity)]
  constructor
  · rw [le_div_iff₀ hK]
    have := Nat.le_ceil (tightK ε n M * (⌊(n : ℝ) / ε⌋₊ : ℝ))
    unfold tight2B; nlinarith
  · rw [div_lt_iff₀ hK]
    have := Nat.ceil_lt_add_one hN0
    unfold tight2B; nlinarith

theorem tight2H_scaled (h : TightHyp ε n M) :
    ⌊(M : ℝ) / tightK ε n M⌋₊ = ⌊(n : ℝ) / ε⌋₊ := by
  congr 1
  rw [div_eq_iff h.K_pos.ne', mul_comm]
  exact h.K_mul_y.symm

theorem tight2_scaled_sum (h : TightHyp ε n M) (S : Finset (Fin n)) :
    ∑ i ∈ S, scaledW (tightK ε n M) (tight2Inst ε n M).w i =
      (if siteB h.hn ∈ S then ⌊(n : ℝ) / ε⌋₊ else 0) +
        (if siteH h.hn ∈ S then ⌊(n : ℝ) / ε⌋₊ else 0) := by
  have key : ∀ i : Fin n, scaledW (tightK ε n M) (tight2Inst ε n M).w i =
      (if i = siteB h.hn then ⌊(n : ℝ) / ε⌋₊ else 0) +
        (if i = siteH h.hn then ⌊(n : ℝ) / ε⌋₊ else 0) := by
    intro i
    by_cases h0 : i.val = 0
    · have hb : i = siteB h.hn := Fin.ext h0
      have hh : i ≠ siteH h.hn := by
        intro e; have := congrArg Fin.val e; simp [siteH] at this; omega
      have hs : scaledW (tightK ε n M) (tight2Inst ε n M).w i = ⌊(n : ℝ) / ε⌋₊ := by
        unfold scaledW
        simp only [tight2Inst, h0, ↓reduceIte]
        exact tight2B_scaled h
      rw [hs]; simp [hb, hh, siteB_ne_siteH h.hn, (siteB_ne_siteH h.hn).symm]
    · by_cases h1 : i.val = 1
      · have hh : i = siteH h.hn := Fin.ext h1
        have hb : i ≠ siteB h.hn := by
          intro e; have := congrArg Fin.val e; simp [siteB] at this; omega
        have hs : scaledW (tightK ε n M) (tight2Inst ε n M).w i = ⌊(n : ℝ) / ε⌋₊ := by
          unfold scaledW
          simp only [tight2Inst, h1, ↓reduceIte, h0]
          exact tight2H_scaled h
        rw [hs]; simp [hb, hh, siteB_ne_siteH h.hn, (siteB_ne_siteH h.hn).symm]
      · have hb : i ≠ siteB h.hn := fun e => h0 (by simp [e, siteB])
        have hh : i ≠ siteH h.hn := fun e => h1 (by simp [e, siteH])
        have hs : scaledW (tightK ε n M) (tight2Inst ε n M).w i = 0 := by
          unfold scaledW
          simp only [tight2Inst, h0, h1, ↓reduceIte]
          rw [Nat.floor_eq_zero, div_lt_one h.K_pos]
          exact h.C_lt_K
        rw [hs]; simp [hb, hh, siteB_ne_siteH h.hn, (siteB_ne_siteH h.hn).symm]
  simp only [key, Finset.sum_add_distrib, Finset.sum_ite_eq']

/-- `b` and `h` are incompatible (`1 + 2 > 2`). -/
theorem tight2_not_both (h : TightHyp ε n M) {S : Finset (Fin n)}
    (hS : Feasible (tight2Inst ε n M) S) : ¬ (siteB h.hn ∈ S ∧ siteH h.hn ∈ S) := by
  rintro ⟨hb, hh⟩
  have hthr := thr_of_feasible _ hS 2
  have hsub : ({siteB h.hn, siteH h.hn} : Finset (Fin n)) ⊆
      S.filter (fun i => (tight2Inst ε n M).d i ≤ 2) := by
    intro x hx
    simp only [mem_insert, mem_singleton] at hx
    rcases hx with rfl | rfl
    · exact Finset.mem_filter.2 ⟨hb, by simp [tight2Inst, siteB]⟩
    · exact Finset.mem_filter.2 ⟨hh, by simp [tight2Inst, siteH]⟩
  have := Finset.sum_le_sum_of_subset (f := (tight2Inst ε n M).p) hsub
  rw [Finset.sum_pair (siteB_ne_siteH h.hn)] at this
  have e1 : (tight2Inst ε n M).p (siteB h.hn) = 1 := by simp [tight2Inst, siteB]
  have e2 : (tight2Inst ε n M).p (siteH h.hn) = 2 := by simp [tight2Inst, siteH]
  rw [e1, e2] at this
  omega

theorem tight2_feasible_singleton_b (h : TightHyp ε n M) :
    Feasible (tight2Inst ε n M) {siteB h.hn} :=
  feasible_singleton _ (by simp [tight2Inst, siteB])

theorem tight2_indivFeasible (h : TightHyp ε n M) : IndivFeasible (tight2Inst ε n M) := by
  intro i
  have hn := h.hn
  simp only [tight2Inst]
  split_ifs <;> omega

theorem tight2_d_cases (h : TightHyp ε n M) {x : Fin n} (hx : x ≠ siteB h.hn) :
    x = siteH h.hn ∨ (tight2Inst ε n M).d x = n := by
  have h0 : x.val ≠ 0 := fun e => hx (Fin.ext e)
  by_cases h1 : x.val = 1
  · exact Or.inl (Fin.ext h1)
  · right; simp [tight2Inst, h0, h1]

/-- Everything except `b` is feasible (`h` completes at `2`, the `z_j` at `3,…,n`). -/
theorem tight2_feasible_erase (h : TightHyp ε n M) :
    Feasible (tight2Inst ε n M) (univ.erase (siteB h.hn)) := by
  have hn := h.hn
  have hne := siteB_ne_siteH h.hn
  have hmem : siteH h.hn ∈ univ.erase (siteB h.hn) := mem_erase.2 ⟨hne.symm, mem_univ _⟩
  have hp : ∑ x ∈ univ.erase (siteB h.hn), (tight2Inst ε n M).p x = n := by
    have e : ∀ x : Fin n, (tight2Inst ε n M).p x = 1 + (if x = siteH h.hn then 1 else 0) := by
      intro x
      by_cases hx : x = siteH h.hn
      · simp [hx, tight2Inst, siteH]
      · have : x.val ≠ 1 := fun e => hx (Fin.ext e)
        simp [tight2Inst, hx, this]
    simp only [e, Finset.sum_add_distrib, Finset.sum_const, Finset.sum_ite_eq', hmem,
      card_erase_of_mem (mem_univ _), card_univ, Fintype.card_fin, ↓reduceIte, smul_eq_mul, mul_one]
    omega
  rw [feasible_iff_thr]
  intro t
  by_cases ht2 : t < 2
  · have : (univ.erase (siteB h.hn)).filter (fun i => (tight2Inst ε n M).d i ≤ t) = ∅ := by
      apply Finset.filter_eq_empty_iff.2
      intro x hx
      have hxb := (mem_erase.1 hx).1
      rcases tight2_d_cases h hxb with rfl | hd
      · simp [tight2Inst, siteH]; omega
      · rw [hd]; omega
    rw [this]; simp
  · by_cases htn : t < n
    · have hsub : (univ.erase (siteB h.hn)).filter (fun i => (tight2Inst ε n M).d i ≤ t)
          ⊆ {siteH h.hn} := by
        intro x hx
        rw [mem_filter] at hx
        have hxb := (mem_erase.1 hx.1).1
        rcases tight2_d_cases h hxb with rfl | hd
        · simp
        · rw [hd] at hx; omega
      calc ∑ i ∈ (univ.erase (siteB h.hn)).filter (fun i => (tight2Inst ε n M).d i ≤ t),
            (tight2Inst ε n M).p i
          ≤ ∑ i ∈ ({siteH h.hn} : Finset (Fin n)), (tight2Inst ε n M).p i :=
            Finset.sum_le_sum_of_subset hsub
        _ ≤ t := by simp [tight2Inst, siteH]; omega
    · calc ∑ i ∈ (univ.erase (siteB h.hn)).filter (fun i => (tight2Inst ε n M).d i ≤ t),
            (tight2Inst ε n M).p i
          ≤ ∑ i ∈ univ.erase (siteB h.hn), (tight2Inst ε n M).p i :=
            Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
        _ ≤ t := by rw [hp]; omega

theorem tight2_weight_erase (h : TightHyp ε n M) :
    weight (tight2Inst ε n M) (univ.erase (siteB h.hn)) = M + (n - 2) * tightC ε n M := by
  have hn := h.hn
  have hne := siteB_ne_siteH h.hn
  have hmem : siteH h.hn ∈ univ.erase (siteB h.hn) := mem_erase.2 ⟨hne.symm, mem_univ _⟩
  unfold weight
  rw [← Finset.add_sum_erase _ _ hmem]
  have : ∑ x ∈ (univ.erase (siteB h.hn)).erase (siteH h.hn), (tight2Inst ε n M).w x
      = ∑ x ∈ (univ.erase (siteB h.hn)).erase (siteH h.hn), tightC ε n M := by
    refine Finset.sum_congr rfl fun x hx => ?_
    have h1 := (mem_erase.1 hx).1
    have h2 := (mem_erase.1 (mem_erase.1 hx).2).1
    have h0 : x.val ≠ 0 := fun e => h2 (Fin.ext e)
    have h1' : x.val ≠ 1 := fun e => h1 (Fin.ext e)
    simp [tight2Inst, h0, h1']
  rw [this, Finset.sum_const, card_erase_of_mem hmem, card_erase_of_mem (mem_univ _), card_univ,
    Fintype.card_fin]
  have e : n - 1 - 1 = n - 2 := by omega
  rw [e]
  simp [tight2Inst, siteH]

/-- The scaled-value maximisers are exactly the feasible sets containing `b` or `h`. -/
theorem tight2_scaledMax_iff (h : TightHyp ε n M) (S : Finset (Fin n)) :
    IsScaledMax (Feasible (tight2Inst ε n M)) (tightK ε n M) (tight2Inst ε n M).w S ↔
      Feasible (tight2Inst ε n M) S ∧ (siteB h.hn ∈ S ∨ siteH h.hn ∈ S) := by
  have hp := tight_floor_pos h
  constructor
  · rintro ⟨hF, hmax⟩
    refine ⟨hF, ?_⟩
    have h1 := hmax {siteB h.hn} (tight2_feasible_singleton_b h)
    rw [tight2_scaled_sum h, tight2_scaled_sum h] at h1
    by_contra hc
    rw [not_or] at hc
    have hne' : siteH h.hn ∉ ({siteB h.hn} : Finset (Fin n)) := by
      simpa using (siteB_ne_siteH h.hn).symm
    simp only [mem_singleton_self, hne', hc.1, hc.2, ↓reduceIte, add_zero] at h1
    omega
  · rintro ⟨hF, hbh⟩
    refine ⟨hF, fun T hT => ?_⟩
    have hnb := tight2_not_both h hT
    rw [tight2_scaled_sum h, tight2_scaled_sum h]
    by_cases hb : siteB h.hn ∈ S <;> by_cases hh : siteH h.hn ∈ S <;>
      by_cases hb' : siteB h.hn ∈ T <;> by_cases hh' : siteH h.hn ∈ T <;>
      simp [hb, hh, hb', hh'] at hnb hbh hF ⊢

theorem tight2_time_singleton_b (h : TightHyp ε n M) :
    time (tight2Inst ε n M) {siteB h.hn} = 1 := by
  simp [time, tight2Inst, siteB]

theorem tight2_weight_b (h : TightHyp ε n M) :
    weight (tight2Inst ε n M) {siteB h.hn} = tight2B ε n M := by
  simp [weight, tight2Inst, siteB]

/-- **What Algorithm 2 returns on Family 2**: the minimum-time maximiser of the scaled value is
exactly `{b}` (scaled maximisers have value `N` and contain `b` or `h`; `{b}` has time `1`,
every set containing `h` has time `≥ 2`, every other maximiser strictly contains `b`). -/
theorem tight2_algOutput_iff (h : TightHyp ε n M) (S : Finset (Fin n)) :
    IsAlgOutput (tight2Inst ε n M) (tightK ε n M) S ↔ S = {siteB h.hn} := by
  have hbmax : IsScaledMax (Feasible (tight2Inst ε n M)) (tightK ε n M) (tight2Inst ε n M).w
      {siteB h.hn} :=
    (tight2_scaledMax_iff h _).2 ⟨tight2_feasible_singleton_b h, Or.inl (mem_singleton_self _)⟩
  have hcard : ∀ T : Finset (Fin n), T.card ≤ time (tight2Inst ε n M) T := by
    intro T
    unfold time
    rw [Finset.card_eq_sum_ones]
    exact Finset.sum_le_sum fun i _ => (tight2Inst ε n M).p_pos i
  constructor
  · rintro ⟨hS, hmin⟩
    have hS' := (tight2_scaledMax_iff h S).1 hS
    have h1 := hmin {siteB h.hn} hbmax
    rw [tight2_time_singleton_b h] at h1
    have hbS : siteB h.hn ∈ S := by
      rcases hS'.2 with hb | hh
      · exact hb
      · exfalso
        have : 2 ≤ time (tight2Inst ε n M) S := by
          calc 2 = (tight2Inst ε n M).p (siteH h.hn) := by simp [tight2Inst, siteH]
            _ ≤ time (tight2Inst ε n M) S := by
              unfold time
              exact Finset.single_le_sum (f := (tight2Inst ε n M).p) (fun _ _ => Nat.zero_le _) hh
        omega
    symm
    apply Finset.eq_of_subset_of_card_le (by simpa using hbS)
    have := hcard S
    simp only [card_singleton]; omega
  · rintro rfl
    refine ⟨hbmax, fun T hT => ?_⟩
    have hT' := (tight2_scaledMax_iff h T).1 hT
    rw [tight2_time_singleton_b h]
    have hpos : 0 < T.card := Finset.card_pos.2 (by
      rcases hT'.2 with hb | hh
      · exact ⟨_, hb⟩
      · exact ⟨_, hh⟩)
    have := hcard T; omega

theorem tight2_fptasK (h : TightHyp ε n M) :
    fptasK ε (tight2Inst ε n M).w = tightK ε n M := by
  have hz : siteH h.hn ∈ (univ : Finset (Fin n)) := mem_univ _
  have hsup : (univ.sup (tight2Inst ε n M).w : ℕ) = M := by
    apply le_antisymm
    · refine Finset.sup_le fun i _ => ?_
      by_cases h0 : i.val = 0
      · simp only [tight2Inst, h0, ↓reduceIte]; exact tight2B_le h
      · by_cases h1 : i.val = 1
        · simp [tight2Inst, h0, h1]
        · simp only [tight2Inst, h0, h1, ↓reduceIte]
          have : (tightC ε n M : ℝ) < M := h.C_lt_K.trans h.K_lt_M
          exact_mod_cast this.le
    · have := Finset.le_sup (f := (tight2Inst ε n M).w) hz
      simpa [tight2Inst, siteH] using this
  unfold fptasK
  rw [hsup, Fintype.card_fin]
  exact max_eq_right (by have := h.K_ge_two; unfold tightK at this; linarith)

/-- `D2(M) = 1 + (n-2)ε/n - (n-2)/M`. -/
noncomputable def tight2D (ε : ℝ) (n M : ℕ) : ℝ :=
  1 + ((n : ℝ) - 2) * ε / n - ((n : ℝ) - 2) / M

/-- `B2(M) = (εN/n + 1/M) / D2(M)`. -/
noncomputable def tightB2 (ε : ℝ) (n M : ℕ) : ℝ :=
  (ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n + 1 / M) / tight2D ε n M

theorem tight2D_pos (h : TightHyp ε n M) : 0 < tight2D ε n M := by
  have hn := h.n_pos_real
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast h.hn
  have hMp := h.M_pos_real
  have hMn : (n : ℝ) < M := by
    have h1 : 2 * (n : ℝ) ≤ M * ε := (div_le_iff₀ h.hε0).1 h.hM
    nlinarith [h.hε1]
  have h1 : ((n : ℝ) - 2) / M < 1 := by rw [div_lt_one hMp]; linarith
  have h2 : 0 ≤ ((n : ℝ) - 2) * ε / n :=
    div_nonneg (mul_nonneg (by linarith) h.hε0.le) hn.le
  unfold tight2D; linarith

/-- **Family 2, ratio bound**: `W({b}) ≤ B2(M) · W*` for every optimal value `W*`. -/
theorem tight2_bound (h : TightHyp ε n M) {v : ℕ} (hv : IsOPT (tight2Inst ε n M) v) :
    (weight (tight2Inst ε n M) {siteB h.hn} : ℝ) ≤ tightB2 ε n M * v := by
  have hn := h.n_pos_real
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast h.hn
  have hMp := h.M_pos_real
  have hDpos := tight2D_pos h
  obtain ⟨-, hvmax⟩ := (isOPT_iff_max_feasible _ v).1 hv
  have hvW : M + (n - 2) * tightC ε n M ≤ v := by
    have := hvmax _ (tight2_feasible_erase h)
    rwa [tight2_weight_erase h] at this
  have hC := h.K_sub_one_le_C
  have hcast : ((M + (n - 2) * tightC ε n M : ℕ) : ℝ) = M + ((n : ℝ) - 2) * tightC ε n M := by
    rw [Nat.cast_add, Nat.cast_mul, Nat.cast_sub h.hn]; simp
  have hlow : (M : ℝ) * tight2D ε n M ≤ ((M + (n - 2) * tightC ε n M : ℕ) : ℝ) := by
    rw [hcast]
    have e : (M : ℝ) * tight2D ε n M = M + ((n : ℝ) - 2) * (tightK ε n M - 1) := by
      unfold tight2D tightK; have := h.hε0; field_simp; ring
    rw [e]
    nlinarith [mul_le_mul_of_nonneg_left hC (by linarith : (0 : ℝ) ≤ (n : ℝ) - 2)]
  have hvW' : ((M + (n - 2) * tightC ε n M : ℕ) : ℝ) ≤ (v : ℝ) := by exact_mod_cast hvW
  have hwb : (weight (tight2Inst ε n M) {siteB h.hn} : ℝ) ≤
      tightK ε n M * (⌊(n : ℝ) / ε⌋₊ : ℝ) + 1 := by
    rw [tight2_weight_b h]
    unfold tight2B
    have h0 : (0 : ℝ) ≤ tightK ε n M * (⌊(n : ℝ) / ε⌋₊ : ℝ) := by
      have := h.K_pos; positivity
    exact (Nat.ceil_lt_add_one h0).le
  have hnum : tightK ε n M * (⌊(n : ℝ) / ε⌋₊ : ℝ) + 1
      = M * (ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n + 1 / M) := by
    unfold tightK; have := h.hε0; field_simp
  have hnn : 0 ≤ ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n + 1 / M := by
    have := h.hε0; positivity
  have hB2 : tightB2 ε n M * ((M : ℝ) * tight2D ε n M)
      = M * (ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n + 1 / M) := by
    unfold tightB2; field_simp
  have hB2nn : 0 ≤ tightB2 ε n M := div_nonneg hnn hDpos.le
  calc (weight (tight2Inst ε n M) {siteB h.hn} : ℝ)
      ≤ tightK ε n M * (⌊(n : ℝ) / ε⌋₊ : ℝ) + 1 := hwb
    _ = tightB2 ε n M * ((M : ℝ) * tight2D ε n M) := by rw [hnum, hB2]
    _ ≤ tightB2 ε n M * ((M + (n - 2) * tightC ε n M : ℕ) : ℝ) :=
        mul_le_mul_of_nonneg_left hlow hB2nn
    _ ≤ tightB2 ε n M * v := mul_le_mul_of_nonneg_left hvW' hB2nn

end Tight2

/-! ### Limits and the proposition -/

section TightLimit

open Filter

variable {ε : ℝ} {n : ℕ}

theorem tendsto_tightB1 (hn : 2 ≤ n) (hε0 : 0 < ε) :
    Tendsto (fun M : ℕ => tightB1 ε n M) atTop
      (nhds (1 / (1 + ε * ((n : ℝ) - 1) / n))) := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (by omega : 1 ≤ n)
  have hpos : 0 < 1 + ε * ((n : ℝ) - 1) / n :=
    add_pos_of_pos_of_nonneg one_pos
      (div_nonneg (mul_nonneg hε0.le (by linarith)) (by linarith))
  have h0 : Tendsto (fun M : ℕ => ((n : ℝ) - 1) / M) atTop (nhds 0) :=
    tendsto_const_div_atTop_nhds_zero_nat _
  have h1 := (tendsto_const_nhds (x := (1 : ℝ))).div
    ((tendsto_const_nhds (x := 1 + ε * ((n : ℝ) - 1) / n)).sub h0) (by simpa using hpos.ne')
  have h2 : Tendsto (fun M : ℕ => 1 / (1 + ε * ((n : ℝ) - 1) / n - ((n : ℝ) - 1) / M))
      atTop (nhds (1 / (1 + ε * ((n : ℝ) - 1) / n - 0))) := h1
  simpa [tightB1] using h2

theorem tendsto_tightB2 (hn : 2 ≤ n) (hε0 : 0 < ε) :
    Tendsto (fun M : ℕ => tightB2 ε n M) atTop
      (nhds ((ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n) / (1 + ((n : ℝ) - 2) * ε / n))) := by
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hpos : 0 < 1 + ((n : ℝ) - 2) * ε / n :=
    add_pos_of_pos_of_nonneg one_pos
      (div_nonneg (mul_nonneg (by linarith) hε0.le) (by linarith))
  have h0 : Tendsto (fun M : ℕ => ((n : ℝ) - 2) / M) atTop (nhds 0) :=
    tendsto_const_div_atTop_nhds_zero_nat _
  have h0' : Tendsto (fun M : ℕ => (1 : ℝ) / M) atTop (nhds 0) :=
    tendsto_const_div_atTop_nhds_zero_nat _
  have hden := (tendsto_const_nhds (x := 1 + ((n : ℝ) - 2) * ε / n)).sub h0
  have hnum := (tendsto_const_nhds (x := ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n)).add h0'
  have h1 := hnum.div hden (by simpa using hpos.ne')
  have h2 : Tendsto (fun M : ℕ => (ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n + 1 / M) /
      (1 + ((n : ℝ) - 2) * ε / n - ((n : ℝ) - 2) / M)) atTop
      (nhds ((ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n + 0) / (1 + ((n : ℝ) - 2) * ε / n - 0))) := h1
  simpa [tightB2, tight2D] using h2

/-- Algebra: `1/(1 + ε(n-1)/n) = (1 + (n-1)/y)⁻¹`, the first term of `ρ`. -/
theorem limit1_eq (hn : 2 ≤ n) (hε0 : 0 < ε) :
    1 / (1 + ε * ((n : ℝ) - 1) / n) = (1 + ((n : ℝ) - 1) / ((n : ℝ) / ε))⁻¹ := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  rw [one_div]
  congr 1
  field_simp

/-- Algebra: `(εN/n)/(1 + (n-2)ε/n) = (1 + (n-2+f)/N)⁻¹`, the second term of `ρ`. -/
theorem limit2_eq (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) :
    (ε * (⌊(n : ℝ) / ε⌋₊ : ℝ) / n) / (1 + ((n : ℝ) - 2) * ε / n)
      = (1 + ((n : ℝ) - 2 + ((n : ℝ) / ε - ⌊(n : ℝ) / ε⌋₊)) / ⌊(n : ℝ) / ε⌋₊)⁻¹ := by
  obtain ⟨hy2, hN1, -, -⟩ := rho_facts hn hε0 hε1
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hNpos : (0 : ℝ) < ⌊(n : ℝ) / ε⌋₊ := by exact_mod_cast hN1
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have h1 : (1 : ℝ) + ((n : ℝ) - 2 + ((n : ℝ) / ε - ⌊(n : ℝ) / ε⌋₊)) / ⌊(n : ℝ) / ε⌋₊
      = ((n : ℝ) - 2 + (n : ℝ) / ε) / ⌊(n : ℝ) / ε⌋₊ := by
    field_simp; ring
  have hd : (0 : ℝ) < (n : ℝ) - 2 + (n : ℝ) / ε := by linarith
  have h3 : (0 : ℝ) < 1 + ((n : ℝ) - 2) * ε / n := by
    have : 0 ≤ ((n : ℝ) - 2) * ε / n := div_nonneg (mul_nonneg (by linarith) hε0.le) hn0.le
    linarith
  rw [h1, inv_div, div_eq_div_iff (ne_of_gt h3) hd.ne']
  field_simp
  ring

/-- Eventually-statement for Family 1. -/
theorem tight1_eventually (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) {η : ℝ} (hη : 0 < η) :
    ∃ M0 : ℕ, ∀ M : ℕ, M0 ≤ M →
      TightHyp ε n M ∧ tightB1 ε n M < (1 + ((n : ℝ) - 1) / ((n : ℝ) / ε))⁻¹ + η := by
  have ht := tendsto_tightB1 hn hε0
  rw [limit1_eq hn hε0] at ht
  have hev : ∀ᶠ M : ℕ in atTop, tightB1 ε n M < (1 + ((n : ℝ) - 1) / ((n : ℝ) / ε))⁻¹ + η :=
    ht.eventually (gt_mem_nhds (lt_add_of_pos_right _ hη))
  obtain ⟨M0, hM0⟩ := eventually_atTop.1 (hev.and (eventually_ge_atTop ⌈2 * (n : ℝ) / ε⌉₊))
  refine ⟨M0, fun M hM => ?_⟩
  obtain ⟨h1, h2⟩ := hM0 M hM
  exact ⟨⟨hn, hε0, hε1, (Nat.le_ceil _).trans (by exact_mod_cast h2)⟩, h1⟩

/-- Eventually-statement for Family 2. -/
theorem tight2_eventually (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) {η : ℝ} (hη : 0 < η) :
    ∃ M0 : ℕ, ∀ M : ℕ, M0 ≤ M →
      TightHyp ε n M ∧ tightB2 ε n M <
        (1 + ((n : ℝ) - 2 + ((n : ℝ) / ε - ⌊(n : ℝ) / ε⌋₊)) / ⌊(n : ℝ) / ε⌋₊)⁻¹ + η := by
  have ht := tendsto_tightB2 hn hε0
  rw [limit2_eq hn hε0 hε1] at ht
  have hev : ∀ᶠ M : ℕ in atTop, tightB2 ε n M <
      (1 + ((n : ℝ) - 2 + ((n : ℝ) / ε - ⌊(n : ℝ) / ε⌋₊)) / ⌊(n : ℝ) / ε⌋₊)⁻¹ + η :=
    ht.eventually (gt_mem_nhds (lt_add_of_pos_right _ hη))
  obtain ⟨M0, hM0⟩ := eventually_atTop.1 (hev.and (eventually_ge_atTop ⌈2 * (n : ℝ) / ε⌉₊))
  refine ⟨M0, fun M hM => ?_⟩
  obtain ⟨h1, h2⟩ := hM0 M hM
  exact ⟨⟨hn, hε0, hε1, (Nat.le_ceil _).trans (by exact_mod_cast h2)⟩, h1⟩

/-- **Proposition 8 (the refined guarantee is asymptotically tight).**  For `ε ∈ (0,1)`, `n ≥ 2` and
every `η > 0` there is `M₀` such that for every `M ≥ M₀` there is an `n`-site instance satisfying
Assumption 1 (Family 1 if `(n-1)/y ≥ (n-2+f)/N`, Family 2 otherwise), with the scaling factor
`K = fptasK ε w` of the algorithm, on which the output `Ŝ` of Algorithm 2 (a minimum-time maximiser of
the scaled value) satisfies `W(Ŝ) ≤ (ρ_{n,ε} + η) · W*` for every optimal value `W*`. -/
theorem prop_tight (hn : 2 ≤ n) (hε0 : 0 < ε) (hε1 : ε < 1) {η : ℝ} (hη : 0 < η) :
    ∃ M0 : ℕ, ∀ M : ℕ, M0 ≤ M → ∃ (I : Inst (Fin n)) (Sh : Finset (Fin n)),
      IndivFeasible I ∧ IsAlgOutput I (fptasK ε I.w) Sh ∧
      ∀ v : ℕ, IsOPT I v → (weight I Sh : ℝ) ≤ (rhoRef n ε + η) * v := by
  by_cases hab : ((n : ℝ) - 2 + ((n : ℝ) / ε - ⌊(n : ℝ) / ε⌋₊)) / ⌊(n : ℝ) / ε⌋₊ ≤
      ((n : ℝ) - 1) / ((n : ℝ) / ε)
  · -- Family 1 realises the first term
    have hρ : rhoRef n ε = (1 + ((n : ℝ) - 1) / ((n : ℝ) / ε))⁻¹ := by
      unfold rhoRef refinedMax; rw [max_eq_left hab]
    obtain ⟨M0, hM0⟩ := tight1_eventually hn hε0 hε1 hη
    refine ⟨M0, fun M hM => ?_⟩
    obtain ⟨hT, hlt⟩ := hM0 M hM
    refine ⟨tightInst ε n M, {(⟨0, hT.n_pos⟩ : Fin n)}, tight_indivFeasible (by omega), ?_, ?_⟩
    · rw [tight_fptasK hT]; exact (tight1_algOutput_iff hT _).2 rfl
    · intro v hv
      refine (tight1_bound hT hv).trans ?_
      rw [hρ]
      exact mul_le_mul_of_nonneg_right hlt.le (Nat.cast_nonneg _)
  · -- Family 2 realises the second term
    push Not at hab
    have hρ : rhoRef n ε =
        (1 + ((n : ℝ) - 2 + ((n : ℝ) / ε - ⌊(n : ℝ) / ε⌋₊)) / ⌊(n : ℝ) / ε⌋₊)⁻¹ := by
      unfold rhoRef refinedMax; rw [max_eq_right hab.le]
    obtain ⟨M0, hM0⟩ := tight2_eventually hn hε0 hε1 hη
    refine ⟨M0, fun M hM => ?_⟩
    obtain ⟨hT, hlt⟩ := hM0 M hM
    refine ⟨tight2Inst ε n M, {siteB hT.hn}, tight2_indivFeasible hT, ?_, ?_⟩
    · rw [tight2_fptasK hT]; exact (tight2_algOutput_iff hT _).2 rfl
    · intro v hv
      refine (tight2_bound hT hv).trans ?_
      rw [hρ]
      exact mul_le_mul_of_nonneg_right hlt.le (Nat.cast_nonneg _)

end TightLimit

end Mwhed
