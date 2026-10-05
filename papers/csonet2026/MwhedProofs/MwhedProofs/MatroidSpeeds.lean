import MwhedProofs.Matroid

/-!
# Equal-size sites on vehicles with different dispatch times (Theorem `thm:speeds`)

Formalisation of the theorem "Equal-size sites, vehicles with different dispatch
times" (`thm:speeds`, ITEM 1 of the theory-fixes note), parts (a), (b) and the
*correctness* half of (c).

**Model.**  There are `m` vehicles; vehicle `v : Fin m` needs time `pv v > 0` for
*every* site (vehicle-dependent, site-independent), so the `j`-th site served by `v`
(`j ≥ 1`) completes at time `j * pv v`.  Site `i` has deadline `d i` and weight
`w i`, and is on time iff it completes by `d i`.  Setting `pv ≡ p` gives the identical
vehicle model of Theorem 5 (`Matroid.lean`).

**Contents (namespace `Mwhed.Speeds`).**

* `cap pv t = ∑ v, ⌊t / pv v⌋` : the capacity function `C(t)`; `card_slots` shows it
  is the number of slots `(v, j)`, `j ≥ 1`, `j * pv v ≤ t`.
* `SlotFeasible pv d S` : Step 1, `S` can be injectively assigned to slots `(v, j)`
  with `j * pv v ≤ d i`.
* `slotFeasible_iff_count` : **(a)** `S` is slot-feasible iff
  `|{i ∈ S : d i ≤ t}| ≤ C(t)` for every `t` (Hall's condition for nested
  neighbourhoods; Step 2).
* `slotFeasible_isIndepFamily` : **(b)** the slot-feasible sets form a matroid
  (Step 3; only `C 0 = 0` is used by the augmentation argument).
* `greedy_slotFeasible_optimal` : **(c)** the greedy set is slot-feasible and has
  maximum weight among slot-feasible sets (Step 4, via `EqualDispatch.greedy_optimal`).
* `ValidSchedule`, `OnTime`, `Schedulable`, `schedulable_iff` : the scheduling
  semantics (each vehicle serves a list of sites back to back from time `0`; the
  `j`-th, 1-based, site of `v` completes at `j * pv v`), and the Step 1 equivalence
  "`S` can be served entirely on time by some valid schedule iff `S` is slot-feasible".
* `speeds_greedy_optimal` : **(e)** the headline theorem (an `IsGreatest` statement
  for the maximum on-time weight over valid schedules), mirroring
  `EqualDispatch.equalP_greedy_optimal`.
* `cap_const`, `slotFeasible_const_iff`, `schedulable_const_iff` : **(f)** the
  identical-time theorem is the special case `pv ≡ p`: `C t = m * (t / p)` and
  slot-feasibility coincides with `EqualDispatch.SlotFeasible (fun i => d i / p) m`.
* `slotFeasible_iff_count_range` : the counting condition need only be checked for
  `t ≤ max d` (so slot-feasibility is decidable once positivity is known).

**What is not formalised.**  Part (c) of the LaTeX theorem also asserts an
`O(m + n log n)` implementation (Step 5: heap of slot times, binary search for
`D_i`, union-find); running times are not formalised, and neither is the
"only the `n` earliest slots matter" reduction.  Everything about correctness
(a), (b), greedy optimality and the link to schedules is machine-checked.

All statements assume `0 < pv v` for every vehicle (as in the theorem); the
weights are natural numbers, as everywhere in the library.
-/

set_option linter.unusedSectionVars false

namespace Mwhed.Speeds

open Finset Mwhed.EqualDispatch

variable {ι : Type*} [DecidableEq ι] {m : ℕ}

/-! ## Capacity function and slots -/

/-- The capacity function `C(t) = ∑_v ⌊t / p_v⌋`. -/
def cap (pv : Fin m → ℕ) (t : ℕ) : ℕ := ∑ v, t / pv v

theorem cap_zero (pv : Fin m → ℕ) : cap pv 0 = 0 := by simp [cap]

theorem cap_mono (pv : Fin m → ℕ) {s t : ℕ} (h : s ≤ t) : cap pv s ≤ cap pv t :=
  sum_le_sum fun v _ => Nat.div_le_div_right h

/-- Slot-feasibility (Step 1): the sites of `S` can be mapped injectively to slots
`(v, j)` (vehicle `v`, position `j ≥ 1`) with slot time `j * pv v ≤ d i`.  (The
assignment `f` is `Option`-valued, `some (v, j)` on `S` and arbitrary elsewhere, so that
the definition is also correct for `m = 0`, where there is no vehicle index.) -/
def SlotFeasible (pv : Fin m → ℕ) (d : ι → ℕ) (S : Finset ι) : Prop :=
  ∃ f : ι → Option (Fin m × ℕ), Set.InjOn f S ∧
    ∀ i ∈ S, ∃ s, f i = some s ∧ 1 ≤ s.2 ∧ s.2 * pv s.1 ≤ d i

/-- The (finite) set of slots `(v, j)`, `j ≥ 1`, of time `j * pv v ≤ t`. -/
def slots (pv : Fin m → ℕ) (t : ℕ) : Finset (Fin m × ℕ) :=
  ((univ : Finset (Fin m)).sigma fun v => Icc 1 (t / pv v)).image (fun s => (s.1, s.2))

theorem mem_slots {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v) {t : ℕ} {s : Fin m × ℕ} :
    s ∈ slots pv t ↔ 1 ≤ s.2 ∧ s.2 * pv s.1 ≤ t := by
  obtain ⟨v, j⟩ := s
  simp only [slots, mem_image, mem_sigma, mem_univ, mem_Icc, true_and]
  constructor
  · rintro ⟨⟨v', j'⟩, ⟨h1, h2⟩, h3⟩
    simp only [Prod.mk.injEq] at h3
    obtain ⟨rfl, rfl⟩ := h3
    exact ⟨h1, (Nat.le_div_iff_mul_le (hp _)).1 h2⟩
  · rintro ⟨h1, h2⟩
    exact ⟨⟨v, j⟩, ⟨h1, (Nat.le_div_iff_mul_le (hp _)).2 h2⟩, rfl⟩

/-- The slots of time `≤ t`, wrapped in `some` (the codomain of the assignments). -/
def oslots (pv : Fin m → ℕ) (t : ℕ) : Finset (Option (Fin m × ℕ)) :=
  (slots pv t).image some

theorem card_oslots (pv : Fin m → ℕ) (t : ℕ) : (oslots pv t).card = (slots pv t).card :=
  card_image_of_injective _ (Option.some_injective _)

/-- `C(t)` is the number of slots of time at most `t`. -/
theorem card_slots (pv : Fin m → ℕ) (t : ℕ) : (slots pv t).card = cap pv t := by
  unfold slots cap
  rw [card_image_of_injective, card_sigma]
  · simp
  · intro a b h
    simp only [Prod.mk.injEq] at h
    exact Sigma.ext h.1 (heq_of_eq h.2)

/-! ## (a) The counting criterion (Steps 1-2) -/

theorem slotFeasible_empty (pv : Fin m → ℕ) (d : ι → ℕ) : SlotFeasible pv d (∅ : Finset ι) :=
  ⟨fun _ => none, by simp, by simp⟩

theorem slotFeasible_mono {pv : Fin m → ℕ} {d : ι → ℕ} {A B : Finset ι} (hAB : A ⊆ B)
    (hB : SlotFeasible pv d B) : SlotFeasible pv d A := by
  obtain ⟨f, hinj, hf⟩ := hB
  exact ⟨f, hinj.mono (by exact_mod_cast hAB), fun i hi => hf i (hAB hi)⟩

/-- Necessity: the sites of `S` with `d i ≤ t` need distinct slots of time `≤ t`. -/
theorem count_le_of_slotFeasible {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v) {d : ι → ℕ}
    {S : Finset ι} (h : SlotFeasible pv d S) (t : ℕ) : N d S t ≤ cap pv t := by
  obtain ⟨f, hinj, hf⟩ := h
  rw [← card_slots, ← card_oslots]
  apply card_le_card_of_injOn f
  · intro i hi
    simp only [coe_filter, Set.mem_ofPred_eq] at hi
    obtain ⟨s, hs, h1, h2⟩ := hf i hi.1
    rw [mem_coe, oslots, mem_image]
    exact ⟨s, (mem_slots hp).2 ⟨h1, h2.trans hi.2⟩, hs.symm⟩
  · exact hinj.mono (by intro i hi; exact (mem_filter.1 hi).1)

/-- Sufficiency (induction on `|S|`, processing the site of latest deadline last):
Hall's condition for the nested neighbourhoods `{slots of time ≤ d i}`. -/
theorem count_imp_slotFeasible {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v) (d : ι → ℕ) :
    ∀ (n : ℕ) (S : Finset ι), S.card = n → (∀ t, N d S t ≤ cap pv t) →
      SlotFeasible pv d S := by
  intro n
  induction n with
  | zero =>
    intro S hS _
    rw [card_eq_zero] at hS
    subst hS
    exact slotFeasible_empty pv d
  | succ n ih =>
    intro S hS hk
    have hne : S.Nonempty := by rw [← card_pos]; omega
    obtain ⟨i0, hi0, hmax⟩ := exists_max_image S d hne
    have hcard' : (S.erase i0).card = n := by rw [card_erase_of_mem hi0]; omega
    have hk' : ∀ t, N d (S.erase i0) t ≤ cap pv t := fun t =>
      (N_mono (erase_subset _ _) t).trans (hk t)
    obtain ⟨f', hinj', hf'⟩ := ih (S.erase i0) hcard' hk'
    have hN : S.card ≤ cap pv (d i0) := by
      have := hk (d i0)
      rwa [N_eq_card hmax] at this
    have hlt : ((S.erase i0).image f').card < (oslots pv (d i0)).card := by
      rw [card_image_of_injOn hinj', card_oslots, card_slots]
      omega
    obtain ⟨s', hs', hsn⟩ := exists_mem_notMem_of_card_lt_card hlt
    obtain ⟨s, hs, rfl⟩ := mem_image.1 hs'
    rw [mem_slots hp] at hs
    refine ⟨Function.update f' i0 (some s), ?_, ?_⟩
    · intro a ha b hb hab
      by_cases ha0 : a = i0
      · by_cases hb0 : b = i0
        · rw [ha0, hb0]
        · exfalso
          have hbS : b ∈ S.erase i0 := mem_erase.2 ⟨hb0, hb⟩
          simp only [ha0, Function.update_self, Function.update_of_ne hb0] at hab
          exact hsn (mem_image.2 ⟨b, hbS, hab.symm⟩)
      · by_cases hb0 : b = i0
        · exfalso
          have haS : a ∈ S.erase i0 := mem_erase.2 ⟨ha0, ha⟩
          simp only [hb0, Function.update_self, Function.update_of_ne ha0] at hab
          exact hsn (mem_image.2 ⟨a, haS, hab⟩)
        · have haS : a ∈ S.erase i0 := mem_erase.2 ⟨ha0, ha⟩
          have hbS : b ∈ S.erase i0 := mem_erase.2 ⟨hb0, hb⟩
          simp only [Function.update_of_ne ha0, Function.update_of_ne hb0] at hab
          exact hinj' haS hbS hab
    · intro i hi
      by_cases h0 : i = i0
      · subst h0
        simp only [Function.update_self]
        exact ⟨s, rfl, hs⟩
      · have hiS : i ∈ S.erase i0 := mem_erase.2 ⟨h0, hi⟩
        simp only [Function.update_of_ne h0]
        exact hf' i hiS

/-- **Theorem `thm:speeds` (a).**  A set `S` can be served entirely on time (in the
slot formulation of Step 1) iff `|{i ∈ S : d i ≤ t}| ≤ C(t)` for every `t ≥ 0`. -/
theorem slotFeasible_iff_count {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v) (d : ι → ℕ)
    (S : Finset ι) : SlotFeasible pv d S ↔ ∀ t, N d S t ≤ cap pv t :=
  ⟨fun h t => count_le_of_slotFeasible hp h t, count_imp_slotFeasible hp d _ S rfl⟩

/-- The criterion only has to be checked for `t ≤ max d`. -/
theorem slotFeasible_iff_count_range {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v) (d : ι → ℕ)
    (S : Finset ι) :
    SlotFeasible pv d S ↔ ∀ t ∈ range (S.sup d + 1), N d S t ≤ cap pv t := by
  rw [slotFeasible_iff_count hp]
  refine ⟨fun h t _ => h t, fun h t => ?_⟩
  by_cases ht : t ≤ S.sup d
  · exact h t (mem_range.2 (by omega))
  · have h1 : N d S t = S.card := N_eq_card (fun i hi => (le_sup (f := d) hi).trans (by omega))
    have h2 : N d S (S.sup d) = S.card := N_eq_card (fun i hi => le_sup (f := d) hi)
    have := h (S.sup d) (mem_range.2 (by omega))
    rw [h1]
    rw [h2] at this
    exact this.trans (cap_mono pv (by omega))

/-- Positivity of the dispatch times makes slot-feasibility decidable. -/
@[instance_reducible] def decidableSlotFeasible {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v) (d : ι → ℕ) :
    DecidablePred (SlotFeasible pv d) := fun S =>
  decidable_of_iff _ (slotFeasible_iff_count_range hp d S).symm

/-! ## (b) The slot-feasible sets form a matroid (Step 3) -/

/-- **Augmentation, for an arbitrary capacity function `C` with `C 0 = 0`.**  Take the
largest `k` with `N_B(k) ≤ N_A(k)`; then `B` has more sites with deadline `k+1` than
`A`, and any such site can be added to `A`.  Only `C 0 = 0` is used (the argument of
the paper uses monotonicity of `C`, which here is not even needed). -/
theorem count_augment (C : ℕ → ℕ) (hC0 : C 0 = 0) {d : ι → ℕ} {A B : Finset ι}
    (hA' : ∀ t, N d A t ≤ C t) (hB' : ∀ t, N d B t ≤ C t) (hlt : A.card < B.card) :
    ∃ x ∈ B, x ∉ A ∧ ∀ t, N d (insert x A) t ≤ C t := by
  set K := (A ∪ B).sup d with hK
  have hAK : ∀ i ∈ A, d i ≤ K := fun i hi => le_sup (f := d) (mem_union_left _ hi)
  have hBK : ∀ i ∈ B, d i ≤ K := fun i hi => le_sup (f := d) (mem_union_right _ hi)
  let P : ℕ → Prop := fun k => N d B k ≤ N d A k
  have hP0 : P 0 := by
    have := hB' 0
    rw [hC0] at this
    show N d B 0 ≤ N d A 0
    omega
  have hPK : ¬ P K := by
    show ¬ N d B K ≤ N d A K
    rw [N_eq_card hAK, N_eq_card hBK]
    omega
  set k := Nat.findGreatest P K with hk
  have hkP : P k := Nat.findGreatest_spec (Nat.zero_le _) hP0
  have hkK : k ≤ K := Nat.findGreatest_le K
  have hkK' : k < K := by
    rcases hkK.lt_or_eq with h | h
    · exact h
    · exact absurd (h ▸ hkP) hPK
  have hgt : ∀ j, k < j → N d A j < N d B j := by
    intro j hj
    by_cases hjK : j ≤ K
    · have := Nat.findGreatest_is_greatest hj hjK
      show N d A j < N d B j
      exact not_le.1 this
    · rw [N_eq_card (fun i hi => (hAK i hi).trans (by omega)),
        N_eq_card (fun i hi => (hBK i hi).trans (by omega))]
      exact hlt
  have h1 := hgt (k + 1) (by omega)
  have h2 : N d B k ≤ N d A k := hkP
  rw [N_succ, N_succ] at h1
  have hcard : (A.filter (fun i => d i = k + 1)).card <
      (B.filter (fun i => d i = k + 1)).card := by omega
  obtain ⟨x, hxB, hxA⟩ := exists_mem_notMem_of_card_lt_card hcard
  rw [mem_filter] at hxB hxA
  have hxA' : x ∉ A := fun h => hxA ⟨h, hxB.2⟩
  refine ⟨x, hxB.1, hxA', ?_⟩
  intro j
  rw [N_insert hxA']
  split_ifs with hj
  · have := hgt j (by omega)
    have := hB' j
    omega
  · have := hA' j
    omega

/-- **Theorem `thm:speeds` (b).**  The slot-feasible sets are the independent sets of a
matroid. -/
theorem slotFeasible_isIndepFamily {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v) (d : ι → ℕ) :
    IsIndepFamily (SlotFeasible pv d) where
  empty := slotFeasible_empty pv d
  mono := fun hAB hB => slotFeasible_mono hAB hB
  aug := by
    intro A B hA hB hlt
    obtain ⟨x, hxB, hxA, hx⟩ := count_augment (cap pv) (cap_zero pv)
      ((slotFeasible_iff_count hp d A).1 hA) ((slotFeasible_iff_count hp d B).1 hB) hlt
    exact ⟨x, hxB, hxA, (slotFeasible_iff_count hp d _).2 hx⟩

/-! ## (c) Greedy is optimal (Step 4) -/

/-- **Theorem `thm:speeds` (c), correctness.**  Let `L` list every site once by
non-increasing weight; then the greedy set (keep a site iff the kept set stays
slot-feasible) is slot-feasible and has maximum weight among slot-feasible sets. -/
theorem greedy_slotFeasible_optimal {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v) (d w : ι → ℕ)
    [DecidablePred (SlotFeasible pv d)]
    (L : List ι) (hnd : L.Nodup) (hall : ∀ i, i ∈ L)
    (hsort : L.Pairwise (fun a b => w b ≤ w a)) :
    SlotFeasible pv d (greedy (SlotFeasible pv d) L) ∧
      ∀ O, SlotFeasible pv d O →
        ∑ i ∈ O, w i ≤ ∑ i ∈ greedy (SlotFeasible pv d) L, w i :=
  greedy_optimal (slotFeasible_isIndepFamily hp d) w (fun i => Nat.zero_le _) L hnd hall hsort

/-! ## (d) Scheduling semantics -/

/-- A schedule of `m` vehicles: vehicle `v` serves the list `σ v` back to back from time
`0`, each vehicle serves distinct sites, and each site is on at most one vehicle.  (This
is the same notion as `EqualDispatch.ValidSchedule`; it does not mention dispatch
times.) -/
abbrev ValidSchedule {m : ℕ} (σ : Fin m → List ι) : Prop :=
  Mwhed.EqualDispatch.ValidSchedule σ

/-- Site `i` is on time: it is the `(j+1)`-st site (0-based index `j`) of some vehicle `v`,
so it completes at time `(j+1) * pv v`, and `(j+1) * pv v ≤ d i`. -/
def OnTime (pv : Fin m → ℕ) (d : ι → ℕ) (σ : Fin m → List ι) (i : ι) : Prop :=
  ∃ (v : Fin m) (j : ℕ), (σ v)[j]? = some i ∧ (j + 1) * pv v ≤ d i

/-- `S` can be served entirely on time by some valid schedule. -/
def Schedulable (pv : Fin m → ℕ) (d : ι → ℕ) (S : Finset ι) : Prop :=
  ∃ σ : Fin m → List ι, ValidSchedule σ ∧ ∀ i ∈ S, OnTime pv d σ i

/-- The set of sites that are on time under the schedule `σ`. -/
noncomputable def onTimeSet [Fintype ι] (pv : Fin m → ℕ) (d : ι → ℕ)
    (σ : Fin m → List ι) : Finset ι := by
  classical exact Finset.univ.filter (fun i => OnTime pv d σ i)

theorem mem_onTimeSet [Fintype ι] {pv : Fin m → ℕ} {d : ι → ℕ} {σ : Fin m → List ι} {i : ι} :
    i ∈ onTimeSet pv d σ ↔ OnTime pv d σ i := by
  classical
  unfold onTimeSet
  simp

/-- **Step 1 of the proof of `thm:speeds`**: `S` can be served entirely on time by some
valid schedule iff it is slot-feasible (no gaps: shift the sites of each vehicle to
consecutive positions). -/
theorem schedulable_iff {pv : Fin m → ℕ} (d : ι → ℕ) (S : Finset ι) :
    Schedulable pv d S ↔ SlotFeasible pv d S := by
  constructor
  · rintro ⟨σ, -, hon⟩
    rcases S.eq_empty_or_nonempty with rfl | ⟨i0, hi0⟩
    · exact slotFeasible_empty _ _
    have : Nonempty (Fin m) := by
      obtain ⟨v, -⟩ := hon i0 hi0
      exact ⟨v⟩
    choose! v j hvj using hon
    refine ⟨fun i => some (v i, j i + 1), ?_, ?_⟩
    · intro a ha b hb hab
      simp only [Option.some.injEq, Prod.mk.injEq] at hab
      have h1 := (hvj a ha).1
      have h2 := (hvj b hb).1
      rw [hab.1, hab.2 |> Nat.succ_injective] at h1
      rw [h1] at h2
      exact Option.some.inj h2
    · intro i hi
      exact ⟨_, rfl, by simp, (hvj i hi).2⟩
  · rintro ⟨f, hinj, hf⟩
    choose! s hs h1 h2 using hf
    have key : ∀ u : Fin m, ∃ ℓ : List ι, ℓ.Nodup ∧
        (∀ i, i ∈ ℓ ↔ i ∈ S.filter (fun i => (s i).1 = u)) ∧
        ∀ i ∈ S.filter (fun i => (s i).1 = u), ∃ j, ℓ[j]? = some i ∧ j + 1 ≤ (s i).2 := by
      intro u
      refine exists_ordered_list (fun i => (s i).2) _ _ rfl ?_ ?_
      · intro a ha b hb hab
        simp only [coe_filter, Set.mem_ofPred_eq] at ha hb
        apply hinj ha.1 hb.1
        rw [hs a ha.1, hs b hb.1]
        exact congrArg some (Prod.ext (by rw [ha.2, hb.2]) hab)
      · intro i hi
        exact (h1 i (mem_filter.1 hi).1)
    choose ℓ hnd hmem hidx using key
    refine ⟨ℓ, ⟨hnd, ?_⟩, ?_⟩
    · intro u v huv i hu hv
      rw [hmem] at hu hv
      have h1 := (mem_filter.1 hu).2
      have h2 := (mem_filter.1 hv).2
      exact huv (h1.symm.trans h2)
    · intro i hi
      obtain ⟨j, hj, hjl⟩ := hidx (s i).1 i (mem_filter.2 ⟨hi, rfl⟩)
      exact ⟨(s i).1, j, hj, (Nat.mul_le_mul_right _ hjl).trans (h2 i hi)⟩

/-! ## (e) Headline: correctness of the greedy for vehicles with different dispatch times -/

/-- **Theorem `thm:speeds`, correctness of (a)-(c) in scheduling vocabulary.**  Let vehicle `v`
have dispatch time `pv v > 0` for every site, let `L` list every site once by non-increasing
weight, and let `G` be the set the greedy algorithm returns over the slot-feasible family.
Then `w(G)` is the maximum on-time weight over all valid `m`-vehicle schedules, and it is
attained. -/
theorem speeds_greedy_optimal [Fintype ι] {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v)
    (d w : ι → ℕ) [DecidablePred (SlotFeasible pv d)]
    (L : List ι) (hnd : L.Nodup) (hall : ∀ i, i ∈ L)
    (hsort : L.Pairwise (fun a b => w b ≤ w a)) :
    IsGreatest {x : ℕ | ∃ σ : Fin m → List ι, ValidSchedule σ ∧
        x = ∑ i ∈ onTimeSet pv d σ, w i}
      (∑ i ∈ greedy (SlotFeasible pv d) L, w i) := by
  obtain ⟨hG, hopt⟩ := greedy_slotFeasible_optimal hp d w L hnd hall hsort
  set G := greedy (SlotFeasible pv d) L with hGdef
  obtain ⟨σ, hσ, hon⟩ := (schedulable_iff d G).2 hG
  refine ⟨⟨σ, hσ, ?_⟩, ?_⟩
  · apply le_antisymm
    · exact sum_le_sum_of_subset (fun i hi => mem_onTimeSet.2 (hon i hi))
    · refine hopt _ ((schedulable_iff d _).1 ?_)
      exact ⟨σ, hσ, fun i hi => mem_onTimeSet.1 hi⟩
  · rintro x ⟨σ', hσ', rfl⟩
    exact hopt _ ((schedulable_iff d _).1 ⟨σ', hσ', fun i hi => mem_onTimeSet.1 hi⟩)

/-- The same, stated as "greedy returns a schedulable set of maximum weight among all
schedulable sets". -/
theorem speeds_greedy_optimal_sets {pv : Fin m → ℕ} (hp : ∀ v, 0 < pv v)
    (d w : ι → ℕ) [DecidablePred (SlotFeasible pv d)]
    (L : List ι) (hnd : L.Nodup) (hall : ∀ i, i ∈ L)
    (hsort : L.Pairwise (fun a b => w b ≤ w a)) :
    Schedulable pv d (greedy (SlotFeasible pv d) L) ∧
      ∀ S : Finset ι, Schedulable pv d S →
        ∑ i ∈ S, w i ≤ ∑ i ∈ greedy (SlotFeasible pv d) L, w i := by
  obtain ⟨hG, hopt⟩ := greedy_slotFeasible_optimal hp d w L hnd hall hsort
  exact ⟨(schedulable_iff d _).2 hG, fun S hS => hopt S ((schedulable_iff d S).1 hS)⟩

/-! ## (f) The identical-time theorem is the special case `pv ≡ p` -/

/-- For `pv ≡ p`, the capacity function is `C t = m * ⌊t / p⌋`. -/
theorem cap_const (m p t : ℕ) : cap (fun _ : Fin m => p) t = m * (t / p) := by
  simp [cap]

/-- For `pv ≡ p` (with `p > 0`), slot-feasibility is exactly the slot-feasibility of
`Matroid.lean` with `D i = ⌊d i / p⌋` (and the same counting criterion
`N_S(k) ≤ m k`). -/
theorem slotFeasible_const_iff {p : ℕ} (hp : 0 < p) (d : ι → ℕ) (S : Finset ι) :
    SlotFeasible (fun _ : Fin m => p) d S ↔
      Mwhed.EqualDispatch.SlotFeasible (fun i => d i / p) m S := by
  rw [slotFeasible_iff_count (fun _ => hp), Mwhed.EqualDispatch.slotFeasible_iff_count]
  constructor
  · intro h k
    have h1 := h ((k + 1) * p - 1)
    rw [cap_const] at h1
    have hq : ((k + 1) * p - 1) / p = k := by
      have : (k + 1) * p - 1 = (p - 1) + k * p := by
        rw [Nat.add_mul, one_mul]; omega
      rw [this, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt (by omega)]
      simp
    rw [hq] at h1
    refine le_trans (le_of_eq ?_) h1
    unfold N
    congr 1
    apply filter_congr
    intro i _
    have hpos : 1 ≤ (k + 1) * p := Nat.mul_pos (by omega) hp
    show d i / p ≤ k ↔ d i ≤ (k + 1) * p - 1
    constructor
    · intro h
      have h2 : d i / p < k + 1 := by omega
      have := (Nat.div_lt_iff_lt_mul hp).1 h2
      omega
    · intro h
      have : d i < (k + 1) * p := by omega
      have := (Nat.div_lt_iff_lt_mul hp).2 this
      omega
  · intro h t
    rw [cap_const]
    refine le_trans ?_ (h (t / p))
    apply card_le_card
    intro i hi
    rw [mem_filter] at hi ⊢
    exact ⟨hi.1, Nat.div_le_div_right hi.2⟩

/-- Scheduling form of the special case: with `pv ≡ p`, `S` is schedulable by
`m` vehicles iff it is `EqualDispatch.Schedulable p d m`. -/
theorem schedulable_const_iff {p : ℕ} (hp : 0 < p) (d : ι → ℕ) (S : Finset ι) :
    Schedulable (fun _ : Fin m => p) d S ↔ Mwhed.EqualDispatch.Schedulable p d m S := by
  rw [schedulable_iff, slotFeasible_const_iff hp, ← Mwhed.EqualDispatch.schedulable_iff hp]

end Mwhed.Speeds
