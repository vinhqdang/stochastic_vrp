import Mathlib

/-!
# Spoke vs. chained route (Proposition `prop:spoke`)

Formalises, in an arbitrary pseudo-metric space, the comparison between the
depot round-trip ("spoke") model and a route chained directly from site to
site, under the *arrival reading* of protection.

Setting.  `X` is a pseudo-metric space (symmetry and triangle inequality are
exactly the axioms used), `o : X` the depot, `pos : ι → X` the site positions,
`s > 0` the crew speed, `a i = dist o (pos i) / s` the one-way travel time and
`p i = 2 * a i` the round-trip dispatch time.  A visiting order is a
`List ι`.  For a prefix `σ` (sites visited before) and a next site `i`:

* `roundTripArrival σ i = (σ.map p).sum + a i`  (the paper's `A_j`);
* `chainArrival σ i = walk o (σ ++ [i])`, the travel time of the route
  `o → σ₁ → … → σ_last → i` that goes directly from each site to the next
  (the paper's `B_j = a_{i_1} + Σ_{l<j} δ(i_l,i_{l+1})/s`).

Results (all for `ℝ`, no further hypotheses besides `s > 0`):

* `chainArrival_le_roundTripArrival`: `B ≤ A` for every prefix and next site;
* `chainArrival_le_roundTripArrival_getElem`: the same for the `j`-th element
  of an order, `σ.take j` being the prefix;
* `protected_chain_of_protected_roundTrip`: for any hazard times `h`, if the
  `j`-th site is reached by the round-trip model by `h`, it is reached by the
  chained route by `h` (the "hence" part of the proposition);
* star case: if `δ(i,j) = δ(i,o) + δ(o,j)` for distinct sites (sites on
  different branches of a star centred at the depot) and the order has no
  repeated site, then `B = A` (`chainArrival_eq_roundTripArrival_of_star`).
-/

namespace Mwhed.Spoke

variable {X ι : Type*} [PseudoMetricSpace X]

/-- One-way travel time depot → site `i`, speed `s`. -/
noncomputable def a (o : X) (pos : ι → X) (s : ℝ) (i : ι) : ℝ := dist o (pos i) / s

/-- Round-trip dispatch time `p i = 2 a i`. -/
noncomputable def p (o : X) (pos : ι → X) (s : ℝ) (i : ι) : ℝ := 2 * a o pos s i

/-- Travel time of the route that starts at `x` and visits the sites of the list
in order, going directly from each site to the next. -/
noncomputable def walk (pos : ι → X) (s : ℝ) : X → List ι → ℝ
  | _, [] => 0
  | x, i :: σ => dist x (pos i) / s + walk pos s (pos i) σ

/-- Round-trip model: arrival time at `i` when the sites of the prefix `σ` were
served before (each by a full depot round trip). This is `A_j`. -/
noncomputable def roundTripArrival (o : X) (pos : ι → X) (s : ℝ) (σ : List ι) (i : ι) : ℝ :=
  (σ.map (p o pos s)).sum + a o pos s i

/-- Chained route: arrival time at `i` after visiting the prefix `σ` directly
from site to site, starting at the depot. This is `B_j`. -/
noncomputable def chainArrival (o : X) (pos : ι → X) (s : ℝ) (σ : List ι) (i : ι) : ℝ :=
  walk pos s o (σ ++ [i])

section
variable (o : X) (pos : ι → X) {s : ℝ}

/-- Generalised bound from an arbitrary start point `x`. -/
theorem walk_append_le (hs : 0 < s) (x : X) (σ : List ι) (i : ι) :
    walk pos s x (σ ++ [i]) ≤
      dist x o / s + (σ.map (p o pos s)).sum + a o pos s i := by
  induction σ generalizing x with
  | nil =>
    simp only [List.nil_append, walk, List.map_nil, List.sum_nil, add_zero, a]
    rw [← add_div]
    exact div_le_div_of_nonneg_right (dist_triangle _ _ _) hs.le
  | cons k τ ih =>
    have h1 := ih (pos k)
    have h2 : dist x (pos k) ≤ dist x o + dist o (pos k) := dist_triangle _ _ _
    have h3 : dist x (pos k) / s ≤ (dist x o + dist o (pos k)) / s :=
      div_le_div_of_nonneg_right h2 hs.le
    rw [add_div] at h3
    simp only [List.cons_append, walk, List.map_cons, List.sum_cons, p, a] at h1 ⊢
    rw [dist_comm (pos k) o] at h1
    linarith

/-- **Proposition `prop:spoke`, pointwise form.** `B ≤ A`. -/
theorem chainArrival_le_roundTripArrival (hs : 0 < s) (σ : List ι) (i : ι) :
    chainArrival o pos s σ i ≤ roundTripArrival o pos s σ i := by
  have := walk_append_le o pos hs o σ i
  simpa [chainArrival, roundTripArrival] using this

/-- Same, for the `j`-th element of an order (prefix = first `j` elements). -/
theorem chainArrival_le_roundTripArrival_getElem (hs : 0 < s) (ord : List ι)
    (j : ℕ) (hj : j < ord.length) :
    chainArrival o pos s (ord.take j) ord[j] ≤
      roundTripArrival o pos s (ord.take j) ord[j] :=
  chainArrival_le_roundTripArrival o pos hs _ _

/-- **Proposition `prop:spoke`, corollary.** With hazard times `h`, every site
reached in time by the round-trip model with order `ord` is also reached in time
by the chained route with the same order. -/
theorem protected_chain_of_protected_roundTrip (hs : 0 < s) (h : ι → ℝ) (ord : List ι)
    (j : ℕ) (hj : j < ord.length)
    (hA : roundTripArrival o pos s (ord.take j) ord[j] ≤ h ord[j]) :
    chainArrival o pos s (ord.take j) ord[j] ≤ h ord[j] :=
  (chainArrival_le_roundTripArrival_getElem o pos hs ord j hj).trans hA

/-- All indices at once: `A_j ≤ h_{i_j} → B_j ≤ h_{i_j}`. -/
theorem protected_chain_forall (hs : 0 < s) (h : ι → ℝ) (ord : List ι) :
    ∀ (j : ℕ) (hj : j < ord.length),
      roundTripArrival o pos s (ord.take j) ord[j] ≤ h ord[j] →
      chainArrival o pos s (ord.take j) ord[j] ≤ h ord[j] :=
  fun j hj hA => protected_chain_of_protected_roundTrip o pos hs h ord j hj hA

/-- Star case, generalised: if from `x` and between distinct sites distances add
through the depot, the chained bound is an equality. -/
theorem walk_append_eq_of_star
    (hstar : ∀ i j, i ≠ j → dist (pos i) (pos j) = dist (pos i) o + dist o (pos j))
    (x : X) (σ : List ι) (i : ι) (hnd : (σ ++ [i]).Nodup)
    (hx : ∀ k ∈ σ ++ [i], dist x (pos k) = dist x o + dist o (pos k)) :
    walk pos s x (σ ++ [i]) =
      dist x o / s + (σ.map (p o pos s)).sum + a o pos s i := by
  induction σ generalizing x with
  | nil =>
    have := hx i (by simp)
    simp [walk, a, this, add_div]
  | cons k τ ih =>
    rw [List.cons_append, List.nodup_cons] at hnd
    have hk := hx k (by simp)
    have hk' : ∀ m ∈ τ ++ [i], dist (pos k) (pos m) = dist (pos k) o + dist o (pos m) := by
      intro m hm
      exact hstar k m (fun e => hnd.1 (e ▸ hm))
    have := ih (pos k) hnd.2 hk'
    simp only [List.cons_append, walk, List.map_cons, List.sum_cons, p, a] at this ⊢
    rw [this, hk, dist_comm (pos k) o]
    ring

/-- **Star case.** If sites on different branches of a star centred at the depot
satisfy `δ(i,j) = δ(i,o) + δ(o,j)` and the order has no repeated site, then
`B = A`: chaining saves nothing. -/
theorem chainArrival_eq_roundTripArrival_of_star
    (hstar : ∀ i j, i ≠ j → dist (pos i) (pos j) = dist (pos i) o + dist o (pos j))
    (σ : List ι) (i : ι) (hnd : (σ ++ [i]).Nodup) :
    chainArrival o pos s σ i = roundTripArrival o pos s σ i := by
  have := walk_append_eq_of_star (s := s) o pos hstar o σ i hnd (by simp)
  simpa [chainArrival, roundTripArrival] using this

end

end Mwhed.Spoke
