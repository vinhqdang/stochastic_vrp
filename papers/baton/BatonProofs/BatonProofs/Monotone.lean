import Mathlib

/-!
Proposition 4 (monotone continuation value) of the BATON manuscript.

Setting. The running load after stop `k` is a Markov chain on `ℝ`; its
one-step transition from stop `k` to stop `k+1` acts on functions through
the expectation operator `P k`. We keep exactly the three properties of
that operator that the proof uses:

* positivity:   `f ≤ g` pointwise  ⇒  `P f ≤ P g` pointwise;
* normalisation: `P` maps a constant function to the same constant;
* stochastic monotonicity (Assumption 1): `P` maps nondecreasing
  functions to nondecreasing functions.

The expectation operator `f ↦ (w ↦ 𝔼[f(W_{k+1}) | W_k = w])` of any
stochastically monotone Markov kernel has these properties on bounded
functions, and every function the recursion feeds to `P` is bounded
(between `0` and the largest emergency price).

The recursion is the Bellman recursion of Section 3 for an arbitrary
recourse menu whose actions are priced by a number that does not depend on
the current load: the value after stop `k < m` is
`V k w = min (C k w) (act k (C k))`, where `act k (C k)` is the price of
the cheapest action (for BATON, `min (H k) (R k + C k 0)`; a reset to any
fixed level `x₀` is priced by `C k x₀`). `C k` is the continuation value
`C k w = P k φ_{k+1} w` with `φ_{k+1} x = E (k+1)` if `B < x` (breach)
and `V (k+1) x` otherwise, and `V m ≡ 0`.

Assumption 2 is only that the emergency schedule `E` is non-negative and
non-increasing; no ordering between the handoff (or return) price and the
emergency price is assumed.
-/

namespace Baton.Monotone

/-- The expectation operator of one transition of the load chain,
reduced to the three properties used by the proof. -/
structure Kernel where
  op : (ℝ → ℝ) → ℝ → ℝ
  pos : ∀ f g : ℝ → ℝ, (∀ x, f x ≤ g x) → ∀ w, op f w ≤ op g w
  const : ∀ (c : ℝ) (w : ℝ), op (fun _ => c) w = c
  stoch_mono : ∀ f : ℝ → ℝ, Monotone f → Monotone (op f)

variable (m : ℕ) (B : ℝ) (E : ℕ → ℝ) (P : ℕ → Kernel) (act : ℕ → (ℝ → ℝ) → ℝ)

/-- Value function after stop `k` (Bellman recursion, backwards from `m`). -/
noncomputable def V (k : ℕ) : ℝ → ℝ :=
  if h : m ≤ k then fun _ => 0
  else
    fun w => min ((P k).op (fun x => if B < x then E (k + 1) else V (k + 1) x) w)
      (act k ((P k).op (fun x => if B < x then E (k + 1) else V (k + 1) x)))
termination_by m - k

/-- Integrand of the continuation value: emergency price on a breach,
value function otherwise. -/
noncomputable def phi (k : ℕ) (x : ℝ) : ℝ :=
  if B < x then E (k + 1) else V m B E P act (k + 1) x

/-- Continuation value after stop `k`. -/
noncomputable def C (k : ℕ) : ℝ → ℝ := (P k).op (phi m B E P act k)

variable {m B E P act}

theorem V_of_ge {k : ℕ} (h : m ≤ k) (w : ℝ) : V m B E P act k w = 0 := by
  rw [V]; simp [h]

theorem V_of_lt {k : ℕ} (h : k < m) (w : ℝ) :
    V m B E P act k w = min (C m B E P act k w) (act k (C m B E P act k)) := by
  rw [V]; simp only [Nat.not_le.mpr h, dite_false]; rfl

theorem V_le_C {k : ℕ} (h : k < m) (w : ℝ) : V m B E P act k w ≤ C m B E P act k w := by
  rw [V_of_lt h]; exact min_le_left _ _

/-- The continuation value never exceeds the next emergency price: the
cost of never acting again is at most the largest emergency price still
reachable. Uses only that `E` is non-negative and non-increasing. -/
theorem C_le_E (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k) :
    ∀ d k, k + 1 + d = m → ∀ w, C m B E P act k w ≤ E (k + 1) := by
  intro d
  induction d with
  | zero =>
    intro k hk w
    have hphi : ∀ x, phi m B E P act k x ≤ E (k + 1) := by
      intro x
      unfold phi
      split_ifs
      · exact le_rfl
      · rw [V_of_ge (by omega)]; exact hE0 _
    calc C m B E P act k w ≤ (P k).op (fun _ => E (k + 1)) w := (P k).pos _ _ hphi w
      _ = E (k + 1) := (P k).const _ _
  | succ d ih =>
    intro k hk w
    have hphi : ∀ x, phi m B E P act k x ≤ E (k + 1) := by
      intro x
      unfold phi
      split_ifs
      · exact le_rfl
      · calc V m B E P act (k + 1) x ≤ C m B E P act (k + 1) x := V_le_C (by omega) x
          _ ≤ E (k + 1 + 1) := ih (k + 1) (by omega) x
          _ ≤ E (k + 1) := hEanti _
    calc C m B E P act k w ≤ (P k).op (fun _ => E (k + 1)) w := (P k).pos _ _ hphi w
      _ = E (k + 1) := (P k).const _ _

theorem V_le_E (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k)
    (k : ℕ) (x : ℝ) : V m B E P act (k + 1) x ≤ E (k + 1) := by
  by_cases h : m ≤ k + 1
  · rw [V_of_ge h]; exact hE0 _
  · calc V m B E P act (k + 1) x ≤ C m B E P act (k + 1) x := V_le_C (by omega) x
      _ ≤ E (k + 1 + 1) := C_le_E hE0 hEanti (m - (k + 2)) (k + 1) (by omega) x
      _ ≤ E (k + 1) := hEanti _

/-- **Proposition 4.** Under Assumption 1 (stochastically monotone
kernels) and Assumption 2 (emergency prices non-negative and
non-increasing), every continuation value `C k`, `k < m`, is
nondecreasing in the load, for every recourse menu priced independently
of the load. -/
theorem C_monotone (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k) :
    ∀ d k, k + 1 + d = m → Monotone (C m B E P act k) := by
  intro d
  induction d with
  | zero =>
    intro k hk
    apply (P k).stoch_mono
    intro x y hxy
    unfold phi
    rw [V_of_ge (show m ≤ k + 1 by omega), V_of_ge (show m ≤ k + 1 by omega)]
    split_ifs with h1 h2 h2
    · exact le_rfl
    · exact absurd (lt_of_lt_of_le h1 hxy) h2
    · exact hE0 _
    · exact le_rfl
  | succ d ih =>
    intro k hk
    apply (P k).stoch_mono
    have hC : Monotone (C m B E P act (k + 1)) := ih (k + 1) (by omega)
    have hV : Monotone (V m B E P act (k + 1)) := by
      intro x y hxy
      rw [V_of_lt (show k + 1 < m by omega), V_of_lt (show k + 1 < m by omega)]
      exact min_le_min_right _ (hC hxy)
    intro x y hxy
    unfold phi
    split_ifs with h1 h2 h2
    · exact le_rfl
    · exact absurd (lt_of_lt_of_le h1 hxy) h2
    · exact V_le_E hE0 hEanti k x
    · exact hV hxy

/-- Convenience form: `C k` is nondecreasing for every `k < m`. -/
theorem C_monotone' (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k)
    {k : ℕ} (hk : k < m) : Monotone (C m B E P act k) :=
  C_monotone hE0 hEanti (m - (k + 1)) k (by omega)

/-! ### Instances of the menu -/

/-- BATON's three-action menu: hand off at `H k`, or return to the depot
at `R k` and restart from the reset level `x₀` (the paper's convention is
`x₀ = 0`). -/
def batonMenu (H R : ℕ → ℝ) (x₀ : ℝ) : ℕ → (ℝ → ℝ) → ℝ :=
  fun k Ck => min (H k) (R k + Ck x₀)

theorem baton_C_monotone (H R : ℕ → ℝ) (x₀ : ℝ)
    (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k) {k : ℕ} (hk : k < m) :
    Monotone (C m B E P (batonMenu H R x₀) k) :=
  C_monotone' hE0 hEanti hk

/-- The empty menu ("never act again"): pricing the only alternative at
`E (k+1)`, which is never cheaper than continuing by `C_le_E`, turns the
recursion into `V = C`. Its continuation value is the paper's `C⁰`. -/
def neverAct (E : ℕ → ℝ) : ℕ → (ℝ → ℝ) → ℝ := fun k _ => E (k + 1)

theorem neverAct_V_eq_C (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k)
    {k : ℕ} (hk : k < m) (w : ℝ) :
    V m B E P (neverAct E) k w = C m B E P (neverAct E) k w := by
  rw [V_of_lt hk]
  exact min_eq_left (C_le_E hE0 hEanti (m - (k + 1)) k (by omega) w)

theorem C0_monotone (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k)
    {k : ℕ} (hk : k < m) : Monotone (C m B E P (neverAct E) k) :=
  C_monotone' hE0 hEanti hk

/-! ### Proposition 2 (inclusion), kernel form

The continuation value under optimal play never exceeds the cost of
continuing and never acting again, whatever the (load-independent) menu. -/

theorem C_le_C0 (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k) :
    ∀ d k, k + 1 + d = m → ∀ w,
      C m B E P act k w ≤ C m B E P (neverAct E) k w := by
  intro d
  induction d with
  | zero =>
    intro k hk w
    apply (P k).pos
    intro x
    unfold phi
    rw [V_of_ge (show m ≤ k + 1 by omega), V_of_ge (show m ≤ k + 1 by omega)]
  | succ d ih =>
    intro k hk w
    apply (P k).pos
    intro x
    unfold phi
    split_ifs
    · exact le_rfl
    · calc V m B E P act (k + 1) x ≤ C m B E P act (k + 1) x := V_le_C (by omega) x
        _ ≤ C m B E P (neverAct E) (k + 1) x := ih (k + 1) (by omega) x
        _ = V m B E P (neverAct E) (k + 1) x := (neverAct_V_eq_C hE0 hEanti (by omega) x).symm

/-- **Proposition 2 (inclusion).** The optimal stopping region at stop `k`
is contained in the myopic one: if continuing is dearer than the action
price under optimal play, it is dearer still under "never act again". -/
theorem stop_region_subset (hE0 : ∀ k, 0 ≤ E k) (hEanti : ∀ k, E (k + 1) ≤ E k)
    {k : ℕ} (hk : k < m) (a w : ℝ) (hstop : a < C m B E P act k w) :
    a < C m B E P (neverAct E) k w :=
  lt_of_lt_of_le hstop (C_le_C0 hE0 hEanti (m - (k + 1)) k (by omega) w)

end Baton.Monotone
