import MwhedProofs.Defs

/-!
# Equal dispatch times: the matroid greedy (Section 4.4, Theorem 5)

Formalisation of Theorem 5 of `../main.tex` (equal dispatch cost is tractable,
for any number `m` of identical vehicles).

* (a) `SlotFeasible`, `N`      : Step 1 (slots) and the counting function of Step 2.
* (b) `slotFeasible_iff_count` : Step 2, `S` feasible iff `N_S(k) ≤ m k` for all `k`.
* (c) `slotFeasible_isIndepFamily` : Step 3, the feasible sets form a matroid.
* (d) `greedy`, `greedy_optimal`   : Step 4, greedy is optimal on any such family.
* (e) `Schedulable`, `schedulable_iff` : Step 1, link to the scheduling semantics.
* (f) `equalP_greedy_optimal`  : the headline theorem (correctness).
-/

set_option linter.unusedSectionVars false

namespace Mwhed.EqualDispatch

open Finset

variable {ι : Type*} [DecidableEq ι]

/-! ## (a) Slots -/

/-- Step 1 of the proof of Theorem 5.  `D i` is the latest feasible position
(`⌊d_i / p⌋` in the paper).  A set `S` is *slot-feasible* if its sites can be
assigned to pairwise distinct slots `(v, j)` (vehicle `v < m`, position `j`)
with `1 ≤ j ≤ D i`. -/
def SlotFeasible (D : ι → ℕ) (m : ℕ) (S : Finset ι) : Prop :=
  ∃ f : ι → ℕ × ℕ, Set.InjOn f S ∧ ∀ i ∈ S, (f i).1 < m ∧ 1 ≤ (f i).2 ∧ (f i).2 ≤ D i

/-- The counting function `N_S(k) = |{i ∈ S : D_i ≤ k}|` of Step 2. -/
def N (D : ι → ℕ) (S : Finset ι) (k : ℕ) : ℕ := (S.filter (fun i => D i ≤ k)).card

theorem slotFeasible_empty (D : ι → ℕ) (m : ℕ) : SlotFeasible D m (∅ : Finset ι) :=
  ⟨fun _ => (0, 0), by simp, by simp⟩

theorem slotFeasible_mono {D : ι → ℕ} {m : ℕ} {A B : Finset ι} (hAB : A ⊆ B)
    (hB : SlotFeasible D m B) : SlotFeasible D m A := by
  obtain ⟨f, hinj, hf⟩ := hB
  exact ⟨f, hinj.mono (by exact_mod_cast hAB), fun i hi => hf i (hAB hi)⟩

theorem N_mono {D : ι → ℕ} {A B : Finset ι} (hAB : A ⊆ B) (k : ℕ) : N D A k ≤ N D B k :=
  card_le_card (filter_subset_filter _ hAB)

theorem N_eq_card {D : ι → ℕ} {S : Finset ι} {k : ℕ} (h : ∀ i ∈ S, D i ≤ k) :
    N D S k = S.card := by
  unfold N
  rw [filter_true_of_mem h]

theorem N_le_card (D : ι → ℕ) (S : Finset ι) (k : ℕ) : N D S k ≤ S.card := card_filter_le _ _

theorem N_insert {D : ι → ℕ} {A : Finset ι} {x : ι} (hx : x ∉ A) (k : ℕ) :
    N D (insert x A) k = N D A k + if D x ≤ k then 1 else 0 := by
  unfold N
  rw [filter_insert]
  split_ifs with h
  · rw [card_insert_of_notMem (by simp [hx])]
  · simp

theorem N_succ (D : ι → ℕ) (S : Finset ι) (k : ℕ) :
    N D S (k + 1) = N D S k + (S.filter (fun i => D i = k + 1)).card := by
  unfold N
  rw [← card_union_of_disjoint]
  · congr 1
    ext i
    simp only [mem_filter, mem_union]
    by_cases hi : i ∈ S <;> simp [hi]
    omega
  · rw [disjoint_left]
    intro i hi hi'
    simp only [mem_filter] at hi hi'
    omega

private theorem if_pos' {c : Prop} [Decidable c] {α : Sort*} {a b : α} (h : c) : (if c then a else b) = a :=
  by simp [h]

private theorem if_neg' {c : Prop} [Decidable c] {α : Sort*} {a b : α} (h : ¬ c) : (if c then a else b) = b :=
  by simp [h]

/-! ## (b) Step 2: the counting criterion -/

theorem count_le_of_slotFeasible {D : ι → ℕ} {m : ℕ} {S : Finset ι}
    (h : SlotFeasible D m S) (k : ℕ) : N D S k ≤ m * k := by
  obtain ⟨f, hinj, hf⟩ := h
  have : (S.filter (fun i => D i ≤ k)).card ≤ ((range m) ×ˢ (Icc 1 k)).card := by
    apply card_le_card_of_injOn f
    · intro i hi
      simp only [coe_filter, Set.mem_ofPred_eq] at hi
      obtain ⟨h1, h2, h3⟩ := hf i hi.1
      simp only [coe_product, coe_range, coe_Icc, Set.mem_prod, Set.mem_Iio, Set.mem_Icc]
      exact ⟨h1, h2, h3.trans hi.2⟩
    · exact hinj.mono (by intro i hi; exact (mem_filter.1 hi).1)
  simpa [N] using this

theorem count_imp_slotFeasible (D : ι → ℕ) (m : ℕ) :
    ∀ (n : ℕ) (S : Finset ι), S.card = n → (∀ k, N D S k ≤ m * k) → SlotFeasible D m S := by
  intro n
  induction n with
  | zero =>
    intro S hS _
    rw [card_eq_zero] at hS
    subst hS
    exact slotFeasible_empty D m
  | succ n ih =>
    intro S hS hk
    have hne : S.Nonempty := by rw [← card_pos]; omega
    obtain ⟨i0, hi0, hmax⟩ := exists_max_image S D hne
    have hcard' : (S.erase i0).card = n := by rw [card_erase_of_mem hi0]; omega
    have hk' : ∀ k, N D (S.erase i0) k ≤ m * k := fun k =>
      (N_mono (erase_subset _ _) k).trans (hk k)
    obtain ⟨f', hinj', hf'⟩ := ih (S.erase i0) hcard' hk'
    have hN : S.card ≤ m * D i0 := by
      have := hk (D i0)
      rwa [N_eq_card hmax] at this
    have hlt : ((S.erase i0).image f').card < ((range m) ×ˢ (Icc 1 (D i0))).card := by
      rw [card_image_of_injOn hinj', card_product, card_range, Nat.card_Icc, Nat.add_sub_cancel]
      omega
    obtain ⟨s, hs, hsn⟩ := exists_mem_notMem_of_card_lt_card hlt
    simp only [mem_product, mem_range, mem_Icc] at hs
    refine ⟨Function.update f' i0 s, ?_, ?_⟩
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
        exact ⟨hs.1, hs.2.1, hs.2.2⟩
      · have hiS : i ∈ S.erase i0 := mem_erase.2 ⟨h0, hi⟩
        simp only [Function.update_of_ne h0]
        exact hf' i hiS

/-- **Step 2 of Theorem 5.**  A set is slot-feasible iff `N_S(k) ≤ m k` for every
`k`.  (The case `k = 0` says no site has `D i = 0`, which handles that case.) -/
theorem slotFeasible_iff_count (D : ι → ℕ) (m : ℕ) (S : Finset ι) :
    SlotFeasible D m S ↔ ∀ k, N D S k ≤ m * k :=
  ⟨fun h k => count_le_of_slotFeasible h k, count_imp_slotFeasible D m _ S rfl⟩

/-- Step 2 exactly as in the paper: all `D i ≥ 1` and `k ≥ 1`. -/
theorem slotFeasible_iff_count_pos (D : ι → ℕ) (m : ℕ) (S : Finset ι)
    (hD : ∀ i ∈ S, 1 ≤ D i) :
    SlotFeasible D m S ↔ ∀ k, 1 ≤ k → N D S k ≤ m * k := by
  rw [slotFeasible_iff_count]
  refine ⟨fun h k _ => h k, fun h k => ?_⟩
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · have : N D S 0 = 0 := by
      unfold N
      rw [card_eq_zero, filter_eq_empty_iff]
      intro i hi
      have := hD i hi
      omega
    simp [this]
  · exact h k hk

/-- The criterion only has to be checked for `k ≤ max D` (so it is decidable). -/
theorem slotFeasible_iff_count_range (D : ι → ℕ) (m : ℕ) (S : Finset ι) :
    SlotFeasible D m S ↔ ∀ k ∈ range (S.sup D + 1), N D S k ≤ m * k := by
  rw [slotFeasible_iff_count]
  refine ⟨fun h k _ => h k, fun h k => ?_⟩
  by_cases hk : k ≤ S.sup D
  · exact h k (mem_range.2 (by omega))
  · have h1 : N D S k = S.card := N_eq_card (fun i hi => (le_sup (f := D) hi).trans (by omega))
    have h2 : N D S (S.sup D) = S.card := N_eq_card (fun i hi => le_sup (f := D) hi)
    have := h (S.sup D) (mem_range.2 (by omega))
    rw [h1]
    rw [h2] at this
    exact this.trans (Nat.mul_le_mul_left _ (by omega))

instance (D : ι → ℕ) (m : ℕ) (S : Finset ι) : Decidable (SlotFeasible D m S) :=
  decidable_of_iff _ (slotFeasible_iff_count_range D m S).symm

/-! ## (c) Step 3: the slot-feasible sets form a matroid -/

/-- The three matroid (independence) axioms (i)-(iii) of Section 4.4. -/
structure IsIndepFamily (I : Finset ι → Prop) : Prop where
  empty : I ∅
  mono : ∀ {A B : Finset ι}, A ⊆ B → I B → I A
  aug : ∀ {A B : Finset ι}, I A → I B → A.card < B.card →
    ∃ x ∈ B, x ∉ A ∧ I (insert x A)

/-- **Augmentation** (Step 3 (iii)), following the paper: take the largest `k`
with `N_B(k) ≤ N_A(k)`; then `B` has more sites with `D = k+1` than `A`. -/
theorem slotFeasible_augment {D : ι → ℕ} {m : ℕ} {A B : Finset ι}
    (hA : SlotFeasible D m A) (hB : SlotFeasible D m B) (hlt : A.card < B.card) :
    ∃ x ∈ B, x ∉ A ∧ SlotFeasible D m (insert x A) := by
  have hA' := (slotFeasible_iff_count D m A).1 hA
  have hB' := (slotFeasible_iff_count D m B).1 hB
  set K := (A ∪ B).sup D with hK
  have hAK : ∀ i ∈ A, D i ≤ K := fun i hi => le_sup (f := D) (mem_union_left _ hi)
  have hBK : ∀ i ∈ B, D i ≤ K := fun i hi => le_sup (f := D) (mem_union_right _ hi)
  let P : ℕ → Prop := fun k => N D B k ≤ N D A k
  have hP0 : P 0 := by
    have := hB' 0
    simp only [Nat.mul_zero, nonpos_iff_eq_zero] at this
    show N D B 0 ≤ N D A 0
    omega
  have hPK : ¬ P K := by
    show ¬ N D B K ≤ N D A K
    rw [N_eq_card hAK, N_eq_card hBK]
    omega
  set k := Nat.findGreatest P K with hk
  have hkP : P k := Nat.findGreatest_spec (Nat.zero_le _) hP0
  have hkK : k ≤ K := Nat.findGreatest_le K
  have hkK' : k < K := by
    rcases hkK.lt_or_eq with h | h
    · exact h
    · exact absurd (h ▸ hkP) hPK
  -- for every `j > k`, `N_B(j) > N_A(j)`
  have hgt : ∀ j, k < j → N D A j < N D B j := by
    intro j hj
    by_cases hjK : j ≤ K
    · have := Nat.findGreatest_is_greatest hj hjK
      show N D A j < N D B j
      exact not_le.1 this
    · rw [N_eq_card (fun i hi => (hAK i hi).trans (by omega)),
        N_eq_card (fun i hi => (hBK i hi).trans (by omega))]
      exact hlt
  have h1 := hgt (k + 1) (by omega)
  have h2 : N D B k ≤ N D A k := hkP
  rw [N_succ, N_succ] at h1
  have hcard : (A.filter (fun i => D i = k + 1)).card < (B.filter (fun i => D i = k + 1)).card := by
    omega
  obtain ⟨x, hxB, hxA⟩ := exists_mem_notMem_of_card_lt_card hcard
  rw [mem_filter] at hxB hxA
  have hxA' : x ∉ A := fun h => hxA ⟨h, hxB.2⟩
  refine ⟨x, hxB.1, hxA', ?_⟩
  rw [slotFeasible_iff_count]
  intro j
  rw [N_insert hxA']
  split_ifs with hj
  · have := hgt j (by omega)
    have := hB' j
    omega
  · have := hA' j
    omega

theorem slotFeasible_isIndepFamily (D : ι → ℕ) (m : ℕ) : IsIndepFamily (SlotFeasible D m) where
  empty := slotFeasible_empty D m
  mono := fun hAB hB => slotFeasible_mono hAB hB
  aug := fun hA hB h => slotFeasible_augment hA hB h

/-! ## (d) Step 4: greedy is optimal on any matroid -/

/-- The greedy algorithm started from `S₀`: process the list left to right and keep
`x` iff the kept set plus `x` is still independent. -/
def greedyFrom (I : Finset ι → Prop) [DecidablePred I] (S₀ : Finset ι) (L : List ι) :
    Finset ι :=
  L.foldl (fun S x => if I (insert x S) then insert x S else S) S₀

/-- The greedy algorithm: `greedy I L` for a list `L` of the elements, typically
sorted by non-increasing weight. -/
def greedy (I : Finset ι → Prop) [DecidablePred I] (L : List ι) : Finset ι :=
  greedyFrom I ∅ L

section Greedy

variable {I : Finset ι → Prop} [DecidablePred I]

theorem greedyFrom_nil (S₀ : Finset ι) : greedyFrom I S₀ [] = S₀ := rfl

theorem greedyFrom_cons_of_indep {S₀ : Finset ι} {x : ι} (L : List ι) (h : I (insert x S₀)) :
    greedyFrom I S₀ (x :: L) = greedyFrom I (insert x S₀) L := by
  simp [greedyFrom, h]

theorem greedyFrom_cons_of_not {S₀ : Finset ι} {x : ι} (L : List ι) (h : ¬ I (insert x S₀)) :
    greedyFrom I S₀ (x :: L) = greedyFrom I S₀ L := by
  simp [greedyFrom, h]

/-- Invariants of greedy: independent, contains the start, uses only list elements. -/
theorem greedyFrom_spec :
    ∀ (L : List ι) (S₀ : Finset ι), I S₀ →
      I (greedyFrom I S₀ L) ∧ S₀ ⊆ greedyFrom I S₀ L ∧
        greedyFrom I S₀ L ⊆ S₀ ∪ L.toFinset := by
  intro L
  induction L with
  | nil => intro S₀ h; exact ⟨h, subset_rfl, by simp [greedyFrom_nil]⟩
  | cons x L ih =>
    intro S₀ h
    by_cases hx : I (insert x S₀)
    · rw [greedyFrom_cons_of_indep L hx]
      obtain ⟨h1, h2, h3⟩ := ih _ hx
      refine ⟨h1, (subset_insert _ _).trans h2, ?_⟩
      intro y hy
      have := h3 hy
      simp only [mem_union, mem_insert, List.toFinset_cons] at this ⊢
      tauto
    · rw [greedyFrom_cons_of_not L hx]
      obtain ⟨h1, h2, h3⟩ := ih _ h
      refine ⟨h1, h2, ?_⟩
      intro y hy
      have := h3 hy
      simp only [mem_union, mem_insert, List.toFinset_cons] at this ⊢
      tauto

/-- Iterated augmentation: an independent set `A` can be extended by elements of an
independent set `B` to an independent set of size at least `|B|`. -/
theorem exists_extend (hI : IsIndepFamily I) (B : Finset ι) (hB : I B) :
    ∀ (n : ℕ) (A : Finset ι), I A → B.card ≤ A.card + n →
      ∃ T, I T ∧ A ⊆ T ∧ T ⊆ A ∪ B ∧ B.card ≤ T.card := by
  intro n
  induction n with
  | zero => intro A hA h; exact ⟨A, hA, subset_rfl, subset_union_left, by omega⟩
  | succ n ih =>
    intro A hA h
    by_cases hAB : A.card < B.card
    · obtain ⟨x, hxB, hxA, hxI⟩ := hI.aug hA hB hAB
      have hc : (insert x A).card = A.card + 1 := card_insert_of_notMem hxA
      obtain ⟨T, hT, h1, h2, h3⟩ := ih (insert x A) hxI (by omega)
      refine ⟨T, hT, (subset_insert _ _).trans h1, h2.trans ?_, h3⟩
      intro y hy
      simp only [mem_union, mem_insert] at hy ⊢
      rcases hy with (rfl | hy) | hy
      · exact Or.inr hxB
      · exact Or.inl hy
      · exact Or.inr hy
    · exact ⟨A, hA, subset_rfl, subset_union_left, by omega⟩

section Weights

variable {α : Type*} [AddCommMonoid α] [PartialOrder α] [IsOrderedAddMonoid α]

/-- The exchange step: `T` is `O` with at most one element replaced by `x`,
and `x` is at least as heavy as whatever was removed. -/
theorem sum_exchange (w : ι → α) (hw : ∀ i, 0 ≤ w i) {O T : Finset ι} {x : ι}
    (hxO : x ∉ O) (hxT : x ∈ T) (hTO : T ⊆ insert x O) (hcard : O.card ≤ T.card)
    (hy : ∀ y ∈ O, y ∉ T → w y ≤ w x) : ∑ i ∈ O, w i ≤ ∑ i ∈ T, w i := by
  have hT0 : T.erase x ⊆ O := by
    intro y hy'
    rw [mem_erase] at hy'
    have := hTO hy'.2
    rw [mem_insert] at this
    tauto
  have hsplit : w x + ∑ i ∈ T.erase x, w i = ∑ i ∈ T, w i := add_sum_erase T w hxT
  have hc : (T.erase x).card + 1 = T.card := by
    rw [card_erase_of_mem hxT]; have := card_pos.2 ⟨x, hxT⟩; omega
  have hR : (O \ T.erase x).card ≤ 1 := by
    rw [card_sdiff_of_subset hT0]; omega
  have hsd := sum_sdiff (f := w) hT0
  have hnn : 0 ≤ w x := hw x
  rcases Nat.lt_or_ge (O \ T.erase x).card 1 with h0 | h1
  · have : O \ T.erase x = ∅ := card_eq_zero.1 (by omega)
    rw [this, sum_empty, zero_add] at hsd
    rw [← hsd, ← hsplit]
    exact le_add_of_nonneg_left hnn
  · obtain ⟨y, hyR⟩ := card_eq_one.1 (le_antisymm hR h1)
    rw [hyR, sum_singleton] at hsd
    have hyR' : y ∈ O \ T.erase x := by rw [hyR]; exact mem_singleton_self y
    rw [mem_sdiff, mem_erase] at hyR'
    have hyx : y ≠ x := fun h => hxO (h ▸ hyR'.1)
    have hyT : y ∉ T := fun h => hyR'.2 ⟨hyx, h⟩
    rw [← hsd, ← hsplit]
    exact add_le_add (hy y hyR'.1 hyT) le_rfl

/-- Main induction for Step 4 (generalised to an arbitrary independent start set `S₀`):
among independent `O` with `S₀ ⊆ O ⊆ S₀ ∪ L`, greedy from `S₀` has maximum weight. -/
theorem greedyFrom_opt (hI : IsIndepFamily I) (w : ι → α) (hw : ∀ i, 0 ≤ w i) :
    ∀ (L : List ι) (S₀ : Finset ι), L.Nodup → L.Pairwise (fun a b => w b ≤ w a) →
      I S₀ → (∀ y ∈ L, y ∉ S₀) →
      ∀ O, I O → S₀ ⊆ O → O ⊆ S₀ ∪ L.toFinset →
        ∑ i ∈ O, w i ≤ ∑ i ∈ greedyFrom I S₀ L, w i := by
  intro L
  induction L with
  | nil =>
    intro S₀ _ _ _ _ O _ h1 h2
    have : O = S₀ := subset_antisymm (by simpa using h2) h1
    simp [this, greedyFrom_nil]
  | cons x L ih =>
    intro S₀ hnd hsort hS₀ hdisj O hO hS₀O hOsub
    rw [List.nodup_cons] at hnd
    rw [List.pairwise_cons] at hsort
    have hxS₀ : x ∉ S₀ := hdisj x (by simp)
    by_cases hx : I (insert x S₀)
    · rw [greedyFrom_cons_of_indep L hx]
      have hdisj' : ∀ y ∈ L, y ∉ insert x S₀ := by
        intro y hy hy'
        rw [mem_insert] at hy'
        rcases hy' with rfl | hy'
        · exact hnd.1 hy
        · exact hdisj y (List.mem_cons_of_mem _ hy) hy'
      by_cases hxO : x ∈ O
      · apply ih (insert x S₀) hnd.2 hsort.2 hx hdisj' O hO
        · intro y hy; rw [mem_insert] at hy; rcases hy with rfl | hy
          · exact hxO
          · exact hS₀O hy
        · intro y hy
          have := hOsub hy
          simp only [mem_union, List.toFinset_cons, mem_insert] at this ⊢
          tauto
      · obtain ⟨T, hT, hT1, hT2, hT3⟩ :=
          exists_extend hI O hO (O.card) (insert x S₀) hx (by omega)
        have hTO : T ⊆ insert x O := by
          refine hT2.trans ?_
          intro y hy
          simp only [mem_union, mem_insert] at hy ⊢
          rcases hy with (rfl | hy) | hy
          · exact Or.inl rfl
          · exact Or.inr (hS₀O hy)
          · exact Or.inr hy
        have hxT : x ∈ T := hT1 (mem_insert_self _ _)
        have hle : ∑ i ∈ O, w i ≤ ∑ i ∈ T, w i := by
          refine sum_exchange w hw hxO hxT hTO hT3 ?_
          intro y hyO hyT
          have hy1 := hOsub hyO
          have hyS : y ∉ S₀ := fun h => hyT (hT1 (mem_insert_of_mem h))
          have hyx : y ≠ x := fun h => hxO (h ▸ hyO)
          simp only [mem_union, List.toFinset_cons, mem_insert, List.mem_toFinset] at hy1
          have hyL : y ∈ L := by tauto
          exact hsort.1 y hyL
        refine hle.trans (ih (insert x S₀) hnd.2 hsort.2 hx hdisj' T hT hT1 ?_)
        intro y hy
        have := hT2 hy
        simp only [mem_union, mem_insert] at this ⊢
        rcases this with (rfl | hy) | hy
        · exact Or.inl (Or.inl rfl)
        · exact Or.inl (Or.inr hy)
        · have := hOsub hy
          simp only [mem_union, List.toFinset_cons, mem_insert, List.mem_toFinset] at this
          rcases this with hy' | rfl | hy'
          · exact Or.inl (Or.inr hy')
          · exact absurd hy hxO
          · exact Or.inr (List.mem_toFinset.2 hy')
    · rw [greedyFrom_cons_of_not L hx]
      have hxO : x ∉ O := fun h => hx (hI.mono (by
        intro y hy; rw [mem_insert] at hy; rcases hy with rfl | hy
        · exact h
        · exact hS₀O hy) hO)
      refine ih S₀ hnd.2 hsort.2 hS₀ (fun y hy => hdisj y (List.mem_cons_of_mem _ hy)) O hO hS₀O ?_
      intro y hy
      have := hOsub hy
      simp only [mem_union, List.toFinset_cons, mem_insert] at this ⊢
      rcases this with h | rfl | h
      · exact Or.inl h
      · exact absurd hy hxO
      · exact Or.inr h

/-- **Step 4 of Theorem 5, for an arbitrary matroid.**  If `L` lists every element
once, by non-increasing weight, then `greedy I L` is independent and has maximum
weight among independent sets. -/
theorem greedy_optimal (hI : IsIndepFamily I) (w : ι → α) (hw : ∀ i, 0 ≤ w i)
    (L : List ι) (hnd : L.Nodup) (hall : ∀ i, i ∈ L)
    (hsort : L.Pairwise (fun a b => w b ≤ w a)) :
    I (greedy I L) ∧ ∀ O, I O → ∑ i ∈ O, w i ≤ ∑ i ∈ greedy I L, w i := by
  refine ⟨(greedyFrom_spec L ∅ hI.empty).1, fun O hO => ?_⟩
  exact greedyFrom_opt hI w hw L ∅ hnd hsort hI.empty (by simp) O hO (empty_subset _)
    (fun i _ => by simp [hall i])

end Weights

end Greedy

/-! ## (e) Scheduling semantics of MWHED-`m` with equal dispatch times -/

/-- A schedule of `m` identical vehicles: vehicle `v` serves the list `σ v` of
sites in order.  Each vehicle serves distinct sites and each site is on at most
one vehicle. -/
def ValidSchedule {m : ℕ} (σ : Fin m → List ι) : Prop :=
  (∀ v, (σ v).Nodup) ∧ ∀ u v, u ≠ v → ∀ i, i ∈ σ u → i ∉ σ v

/-- Site `i` is served on time: it is the `(j+1)`-st site (0-based index `j`) of some
vehicle, so it completes at time `(j+1) * p`, and `(j+1) * p ≤ d i`. -/
def OnTime (p : ℕ) (d : ι → ℕ) {m : ℕ} (σ : Fin m → List ι) (i : ι) : Prop :=
  ∃ (v : Fin m) (j : ℕ), (σ v)[j]? = some i ∧ (j + 1) * p ≤ d i

/-- `S` can be served entirely on time by `m` vehicles, all dispatch times `= p`. -/
def Schedulable (p : ℕ) (d : ι → ℕ) (m : ℕ) (S : Finset ι) : Prop :=
  ∃ σ : Fin m → List ι, ValidSchedule σ ∧ ∀ i ∈ S, OnTime p d σ i

/-- The set of sites that are on time under the schedule `σ`. -/
noncomputable def onTimeSet [Fintype ι] (p : ℕ) (d : ι → ℕ) {m : ℕ} (σ : Fin m → List ι) :
    Finset ι := by
  classical exact Finset.univ.filter (fun i => OnTime p d σ i)

theorem mem_onTimeSet [Fintype ι] {p : ℕ} {d : ι → ℕ} {m : ℕ} {σ : Fin m → List ι} {i : ι} :
    i ∈ onTimeSet p d σ ↔ OnTime p d σ i := by
  classical
  unfold onTimeSet
  simp

/-- "Shift to consecutive positions" (the last sentence of Step 1): sites with
distinct positive keys `g` can be listed so that the site at 0-based index `j`
has `j + 1 ≤ g`. -/
theorem exists_ordered_list (g : ι → ℕ) :
    ∀ (n : ℕ) (T : Finset ι), T.card = n → Set.InjOn g T → (∀ i ∈ T, 1 ≤ g i) →
      ∃ ℓ : List ι, ℓ.Nodup ∧ (∀ i, i ∈ ℓ ↔ i ∈ T) ∧
        ∀ i ∈ T, ∃ j, ℓ[j]? = some i ∧ j + 1 ≤ g i := by
  intro n
  induction n with
  | zero =>
    intro T hT _ _
    rw [card_eq_zero] at hT
    subst hT
    exact ⟨[], List.nodup_nil, by simp, by simp⟩
  | succ n ih =>
    intro T hT hinj hpos
    have hne : T.Nonempty := by rw [← card_pos]; omega
    obtain ⟨i0, hi0, hmax⟩ := exists_max_image T g hne
    have hc : (T.erase i0).card = n := by rw [card_erase_of_mem hi0]; omega
    obtain ⟨ℓ, hnd, hmem, hidx⟩ := ih (T.erase i0) hc
      (hinj.mono (by exact_mod_cast erase_subset _ _))
      (fun i hi => hpos i (mem_of_mem_erase hi))
    have hlen : ℓ.length = n := by
      have : ℓ.toFinset = T.erase i0 := by ext i; simp [hmem]
      rw [← List.toFinset_card_of_nodup hnd, this, hc]
    have hi0ℓ : i0 ∉ ℓ := by rw [hmem]; simp
    have hbound : T.card ≤ g i0 := by
      have : T.card ≤ (Icc 1 (g i0)).card := by
        apply card_le_card_of_injOn g
        · intro i hi
          simp only [coe_Icc, Set.mem_Icc]
          exact ⟨hpos i hi, hmax i hi⟩
        · exact hinj
      simpa using this
    refine ⟨ℓ ++ [i0], ?_, ?_, ?_⟩
    · rw [List.nodup_append]
      refine ⟨hnd, List.nodup_singleton _, ?_⟩
      intro a ha b hb
      simp only [List.mem_singleton] at hb
      subst hb
      rintro rfl
      exact hi0ℓ ha
    · intro i
      rw [List.mem_append, hmem, List.mem_singleton, mem_erase]
      constructor
      · rintro (⟨_, h⟩ | rfl)
        · exact h
        · exact hi0
      · intro h
        by_cases h0 : i = i0
        · exact Or.inr h0
        · exact Or.inl ⟨h0, h⟩
    · intro i hi
      by_cases h0 : i = i0
      · subst h0
        exact ⟨ℓ.length, by simp, by omega⟩
      · obtain ⟨j, hj, hjg⟩ := hidx i (mem_erase.2 ⟨h0, hi⟩)
        have hjl : j < ℓ.length := (List.getElem?_eq_some_iff.1 hj).1
        exact ⟨j, by rw [List.getElem?_append_left hjl]; exact hj, hjg⟩

theorem schedulable_iff {p : ℕ} (hp : 0 < p) (d : ι → ℕ) (m : ℕ) (S : Finset ι) :
    Schedulable p d m S ↔ SlotFeasible (fun i => d i / p) m S := by
  constructor
  · rintro ⟨σ, -, hon⟩
    rcases S.eq_empty_or_nonempty with rfl | ⟨i0, hi0⟩
    · exact slotFeasible_empty _ _
    have : Nonempty (Fin m) := by
      obtain ⟨v, -⟩ := hon i0 hi0
      exact ⟨v⟩
    choose! v j hvj using hon
    refine ⟨fun i => ((v i : ℕ), j i + 1), ?_, ?_⟩
    · intro a ha b hb hab
      simp only [Prod.mk.injEq] at hab
      have hv : v a = v b := Fin.ext hab.1
      have h1 := (hvj a ha).1
      have h2 := (hvj b hb).1
      rw [hv, hab.2 |> Nat.succ_injective] at h1
      rw [h1] at h2
      exact Option.some.inj h2
    · intro i hi
      refine ⟨(v i).isLt, by simp, ?_⟩
      show j i + 1 ≤ d i / p
      exact (Nat.le_div_iff_mul_le hp).2 (hvj i hi).2
  · rintro ⟨f, hinj, hf⟩
    have key : ∀ u : Fin m, ∃ ℓ : List ι, ℓ.Nodup ∧ (∀ i, i ∈ ℓ ↔ i ∈ S.filter (fun i => (f i).1 = u.val)) ∧
        ∀ i ∈ S.filter (fun i => (f i).1 = u.val), ∃ j, ℓ[j]? = some i ∧ j + 1 ≤ (f i).2 := by
      intro u
      refine exists_ordered_list (fun i => (f i).2) _ _ rfl ?_ ?_
      · intro a ha b hb hab
        simp only [coe_filter, Set.mem_ofPred_eq] at ha hb
        apply hinj ha.1 hb.1
        exact Prod.ext (by rw [ha.2, hb.2]) hab
      · intro i hi
        exact (hf i (mem_filter.1 hi).1).2.1
    choose ℓ hnd hmem hidx using key
    refine ⟨ℓ, ⟨hnd, ?_⟩, ?_⟩
    · intro u v huv i hu hv
      rw [hmem] at hu hv
      have := (mem_filter.1 hu).2
      have := (mem_filter.1 hv).2
      exact huv (Fin.ext (by omega))
    · intro i hi
      obtain ⟨h1, h2, h3⟩ := hf i hi
      obtain ⟨j, hj, hjl⟩ := hidx (⟨(f i).1, h1⟩ : Fin m) i (mem_filter.2 ⟨hi, rfl⟩)
      refine ⟨⟨(f i).1, h1⟩, j, hj, ?_⟩
      exact (Nat.le_div_iff_mul_le hp).1 (hjl.trans h3)

/-! ## (f) Headline: correctness of the matroid greedy -/

/-- **Theorem 5 (correctness), for any number `m` of identical vehicles.**
Let all dispatch times be the constant `p > 0`, let `L` list every site once by
non-increasing weight, and let `G` be the set the greedy algorithm returns over the
slot-feasible family (`D i = ⌊d i / p⌋`).  Then `w(G)` is the optimum of MWHED-`m`:
it is the on-time weight of some valid `m`-vehicle schedule and no valid schedule
has larger on-time weight. -/
theorem equalP_greedy_optimal [Fintype ι] {p : ℕ} (hp : 0 < p) (d w : ι → ℕ) (m : ℕ)
    (L : List ι) (hnd : L.Nodup) (hall : ∀ i, i ∈ L)
    (hsort : L.Pairwise (fun a b => w b ≤ w a)) :
    IsGreatest {x : ℕ | ∃ σ : Fin m → List ι, ValidSchedule σ ∧
        x = ∑ i ∈ onTimeSet p d σ, w i}
      (∑ i ∈ greedy (SlotFeasible (fun i => d i / p) m) L, w i) := by
  obtain ⟨hG, hopt⟩ := greedy_optimal (slotFeasible_isIndepFamily (fun i => d i / p) m) w
    (fun i => Nat.zero_le _) L hnd hall hsort
  set G := greedy (SlotFeasible (fun i => d i / p) m) L with hGdef
  obtain ⟨σ, hσ, hon⟩ := (schedulable_iff hp d m G).2 hG
  refine ⟨⟨σ, hσ, ?_⟩, ?_⟩
  · apply le_antisymm
    · exact sum_le_sum_of_subset (fun i hi => mem_onTimeSet.2 (hon i hi))
    · refine hopt _ ((schedulable_iff hp d m _).1 ?_)
      exact ⟨σ, hσ, fun i hi => mem_onTimeSet.1 hi⟩
  · rintro x ⟨σ', hσ', rfl⟩
    exact hopt _ ((schedulable_iff hp d m _).1 ⟨σ', hσ', fun i hi => mem_onTimeSet.1 hi⟩)

/-- The same, stated as "greedy returns a set of maximum weight among all sets that can
be served on time". -/
theorem equalP_greedy_optimal_sets {p : ℕ} (hp : 0 < p) (d w : ι → ℕ) (m : ℕ)
    (L : List ι) (hnd : L.Nodup) (hall : ∀ i, i ∈ L)
    (hsort : L.Pairwise (fun a b => w b ≤ w a)) :
    Schedulable p d m (greedy (SlotFeasible (fun i => d i / p) m) L) ∧
      ∀ S : Finset ι, Schedulable p d m S →
        ∑ i ∈ S, w i ≤ ∑ i ∈ greedy (SlotFeasible (fun i => d i / p) m) L, w i := by
  obtain ⟨hG, hopt⟩ := greedy_optimal (slotFeasible_isIndepFamily (fun i => d i / p) m) w
    (fun i => Nat.zero_le _) L hnd hall hsort
  exact ⟨(schedulable_iff hp d m _).2 hG, fun S hS => hopt S ((schedulable_iff hp d m S).1 hS)⟩

/-! ## Link to `Mwhed.Feasible` (`m = 1`) -/

theorem length_takeWhile_ne (σ : List ι) (i : ι) (h : i ∈ σ) :
    (σ.takeWhile (· ≠ i)).length = σ.idxOf i := by
  induction σ with
  | nil => simp at h
  | cons a t ih =>
    by_cases hai : a = i
    · subst hai; simp
    · have hi : i ∈ t := by simpa [Ne.symm hai] using h
      simpa [hai] using ih hi

/-- For a constant dispatch time `p`, the completion time of `i` in `σ` is
`(index + 1) * p`. -/
theorem completion_const (I : Mwhed.Inst ι) {p : ℕ} (hpc : ∀ i, I.p i = p) (σ : List ι)
    (i : ι) (h : i ∈ σ) : Mwhed.completion I σ i = (σ.idxOf i + 1) * p := by
  unfold Mwhed.completion
  have : (σ.takeWhile (· ≠ i)).map I.p = (σ.takeWhile (· ≠ i)).map (fun _ => p) := by
    congr 1; funext x; exact hpc x
  rw [this, List.map_const', List.sum_replicate, length_takeWhile_ne σ i h, hpc i]
  simp [add_mul]

/-- **MWHED (`m = 1`) with constant dispatch time**: `Mwhed.Feasible` of `Defs.lean`
coincides with slot-feasibility with one vehicle and `D i = ⌊d i / p⌋`. -/
theorem feasible_iff_slotFeasible [Fintype ι] (I : Mwhed.Inst ι) {p : ℕ} (hp : 0 < p)
    (hpc : ∀ i, I.p i = p) (S : Finset ι) :
    Mwhed.Feasible I S ↔ SlotFeasible (fun i => I.d i / p) 1 S := by
  rw [← schedulable_iff hp I.d 1 S]
  constructor
  · rintro ⟨σ, ⟨hnd, hall⟩, hon⟩
    refine ⟨fun _ => σ, ⟨fun _ => hnd, fun u v huv => absurd (Subsingleton.elim u v) huv⟩, ?_⟩
    intro i hi
    have hlt : σ.idxOf i < σ.length := List.idxOf_lt_length_iff.2 (hall i)
    refine ⟨0, σ.idxOf i, ?_, ?_⟩
    · rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    · have := hon i hi
      rwa [completion_const I hpc σ i (hall i)] at this
  · rintro ⟨σ, ⟨hnd, -⟩, hon⟩
    set u := σ 0 with hu
    set ℓ : List ι := u ++ (Finset.univ.filter (fun i => i ∉ u)).toList with hℓ
    have hndℓ : ℓ.Nodup := by
      rw [hℓ, List.nodup_append]
      refine ⟨hnd 0, Finset.nodup_toList _, ?_⟩
      intro a ha b hb
      rw [Finset.mem_toList, Finset.mem_filter] at hb
      rintro rfl
      exact hb.2 ha
    have hallℓ : ∀ i, i ∈ ℓ := by
      intro i
      rw [hℓ, List.mem_append, Finset.mem_toList, Finset.mem_filter]
      by_cases hi : i ∈ u <;> simp [hi]
    refine ⟨ℓ, ⟨hndℓ, hallℓ⟩, ?_⟩
    intro i hi
    obtain ⟨v, j, hj, hjd⟩ := hon i hi
    have hv : v = 0 := Subsingleton.elim _ _
    subst hv
    have hjl : j < u.length := (List.getElem?_eq_some_iff.1 hj).1
    have hjℓ : ℓ[j]? = some i := by rw [hℓ, List.getElem?_append_left hjl]; exact hj
    have hjℓ' : j < ℓ.length := (List.getElem?_eq_some_iff.1 hjℓ).1
    have hidx : ℓ.idxOf i = j := by
      have hge : ℓ[j] = i := by
        rw [List.getElem?_eq_getElem hjℓ'] at hjℓ; exact Option.some.inj hjℓ
      rw [← hge]
      exact hndℓ.idxOf_getElem j hjℓ'
    rw [completion_const I hpc ℓ i (hallℓ i), hidx]
    exact hjd

/-- **Theorem 5 for MWHED (`m = 1`), in the vocabulary of `Defs.lean`.**  For an instance
with constant dispatch time, the greedy set `G` is feasible and has maximum weight among
feasible sets (the characterisation of `W*`, cf. `isOPT_iff_max_feasible` in `Core.lean`). -/
theorem equalP_greedy_optimal_mwhed [Fintype ι] (I : Mwhed.Inst ι) {p : ℕ} (hp : 0 < p)
    (hpc : ∀ i, I.p i = p) (L : List ι) (hnd : L.Nodup) (hall : ∀ i, i ∈ L)
    (hsort : L.Pairwise (fun a b => I.w b ≤ I.w a)) :
    Mwhed.Feasible I (greedy (SlotFeasible (fun i => I.d i / p) 1) L) ∧
      ∀ S, Mwhed.Feasible I S →
        Mwhed.weight I S ≤ Mwhed.weight I (greedy (SlotFeasible (fun i => I.d i / p) 1) L) := by
  obtain ⟨hG, hopt⟩ := greedy_optimal (slotFeasible_isIndepFamily (fun i => I.d i / p) 1) I.w
    (fun i => Nat.zero_le _) L hnd hall hsort
  exact ⟨(feasible_iff_slotFeasible I hp hpc _).2 hG,
    fun S hS => hopt S ((feasible_iff_slotFeasible I hp hpc S).1 hS)⟩

/-! ## Example 5 of the paper (`p = 2`, `d = (3,3,5,9)`, `w = (9,4,7,6)`) -/

namespace Example5

/-- Deadlines `d = (3,3,5,9)` (sites `1..4` of the paper are `0..3` here). -/
def d : Fin 4 → ℕ := ![3, 3, 5, 9]
/-- Weights `w = (9,4,7,6)`. -/
def w : Fin 4 → ℕ := ![9, 4, 7, 6]
/-- The greedy order: by decreasing weight, sites `1,3,4,2`. -/
def L : List (Fin 4) := [0, 2, 3, 1]

/-- The latest feasible positions are `D = (1,1,2,4)`. -/
theorem D_eq : (fun i => d i / 2) = ![1, 1, 2, 4] := by
  funext i; fin_cases i <;> rfl

/-- One vehicle: greedy keeps sites `1,3,4`, weight `9 + 7 + 6 = 22`. -/
theorem greedy_m1 : greedy (SlotFeasible (fun i => d i / 2) 1) L = {0, 2, 3} := by decide

theorem weight_m1 : ∑ i ∈ greedy (SlotFeasible (fun i => d i / 2) 1) L, w i = 22 := by
  rw [greedy_m1]; decide

/-- Exhaustive search over all `16` subsets confirms that `22` is optimal. -/
theorem exhaustive_m1 :
    ∀ S : Finset (Fin 4), SlotFeasible (fun i => d i / 2) 1 S → ∑ i ∈ S, w i ≤ 22 := by decide

/-- Two vehicles: all four sites are served, weight `26`. -/
theorem greedy_m2 : greedy (SlotFeasible (fun i => d i / 2) 2) L = Finset.univ := by decide

theorem weight_m2 : ∑ i ∈ greedy (SlotFeasible (fun i => d i / 2) 2) L, w i = 26 := by
  rw [greedy_m2]; decide

theorem exhaustive_m2 :
    ∀ S : Finset (Fin 4), SlotFeasible (fun i => d i / 2) 2 S → ∑ i ∈ S, w i ≤ 26 := by decide

/-- The optimum of MWHED-1 on the example is `22`, obtained from the headline theorem. -/
theorem optimum_m1 :
    IsGreatest {x : ℕ | ∃ σ : Fin 1 → List (Fin 4), ValidSchedule σ ∧
      x = ∑ i ∈ onTimeSet 2 d σ, w i} 22 := by
  have h := equalP_greedy_optimal (p := 2) (by norm_num) d w 1 L (by decide) (by decide)
    (by decide)
  rwa [weight_m1] at h

/-- The optimum of MWHED-2 on the example is `26`. -/
theorem optimum_m2 :
    IsGreatest {x : ℕ | ∃ σ : Fin 2 → List (Fin 4), ValidSchedule σ ∧
      x = ∑ i ∈ onTimeSet 2 d σ, w i} 26 := by
  have h := equalP_greedy_optimal (p := 2) (by norm_num) d w 2 L (by decide) (by decide)
    (by decide)
  rwa [weight_m2] at h

end Example5

/-! ## (g) Step 5: the "latest free slot" implementation

Only the *logic* of Algorithm 3 is formalised: the occupancy array `occ q` (number of
used slots at position `q`) and the function `findFree` (the largest position
`q ≤ D i` with a free slot, `0` if none).  The union-find data structure, which merely
computes `findFree` in near-constant amortised time, and the running time are out of
scope. -/

section Impl

/-- `findFree m occ D`: the largest position `1 ≤ q ≤ D` with `occ q < m`, or `0`. -/
def findFree (m : ℕ) (occ : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | q + 1 => if occ (q + 1) < m then q + 1 else findFree m occ q

theorem findFree_spec (m : ℕ) (occ : ℕ → ℕ) (D : ℕ) :
    (findFree m occ D = 0 ∧ ∀ r, 1 ≤ r → r ≤ D → m ≤ occ r) ∨
    (1 ≤ findFree m occ D ∧ findFree m occ D ≤ D ∧ occ (findFree m occ D) < m ∧
      ∀ r, findFree m occ D < r → r ≤ D → m ≤ occ r) := by
  induction D with
  | zero => left; refine ⟨rfl, fun r h1 h2 => by omega⟩
  | succ D ih =>
    by_cases h : occ (D + 1) < m
    · right
      have hf : findFree m occ (D + 1) = D + 1 := by simp [findFree, h]
      rw [hf]
      exact ⟨by omega, le_rfl, h, fun r h1 h2 => by omega⟩
    · have hm : m ≤ occ (D + 1) := not_lt.1 h
      have hf : findFree m occ (D + 1) = findFree m occ D := by simp [findFree, h]
      rw [hf]
      rcases ih with ⟨h0, h1⟩ | ⟨h0, h1, h2, h3⟩
      · left
        refine ⟨h0, fun r hr1 hr2 => ?_⟩
        by_cases hr : r = D + 1
        · subst hr; exact hm
        · exact h1 r hr1 (by omega)
      · right
        refine ⟨h0, by omega, h2, fun r hr1 hr2 => ?_⟩
        by_cases hr : r = D + 1
        · subst hr; exact hm
        · exact h3 r hr1 (by omega)

/-- One step of Algorithm 3: find the latest free slot `q`; if `q ≥ 1` keep the site
and use one slot of position `q`. -/
def algStep (D : ι → ℕ) (m : ℕ) (s : Finset ι × (ℕ → ℕ)) (i : ι) : Finset ι × (ℕ → ℕ) :=
  if 1 ≤ findFree m s.2 (D i) then
    (insert i s.1, Function.update s.2 (findFree m s.2 (D i)) (s.2 (findFree m s.2 (D i)) + 1))
  else s

/-- Algorithm 3 (without the union-find): returns the kept set. -/
def algRun (D : ι → ℕ) (m : ℕ) (L : List ι) : Finset ι :=
  (L.foldl (algStep D m) (∅, fun _ => 0)).1

/-- The invariant of the kept set `A` and occupancy array `occ`: there is a slot
assignment `f` of `A` whose number of used slots at position `q` is `occ q`, and each
kept site `j` was given the *latest* free slot: all positions in `(pos j, D j]` are full. -/
def Inv (D : ι → ℕ) (m : ℕ) (A : Finset ι) (occ : ℕ → ℕ) : Prop :=
  ∃ f : ι → ℕ × ℕ, Set.InjOn f A ∧ (∀ i ∈ A, (f i).1 < m ∧ 1 ≤ (f i).2 ∧ (f i).2 ≤ D i) ∧
    (∀ q, occ q = (A.filter (fun j => (f j).2 = q)).card) ∧
    (∀ j ∈ A, ∀ r, (f j).2 < r → r ≤ D j → m ≤ occ r)

theorem Inv.slotFeasible {D : ι → ℕ} {m : ℕ} {A : Finset ι} {occ : ℕ → ℕ}
    (h : Inv D m A occ) : SlotFeasible D m A := by
  obtain ⟨f, hinj, hb, -, -⟩ := h
  exact ⟨f, hinj, hb⟩

theorem inv_empty (D : ι → ℕ) (m : ℕ) : Inv D m (∅ : Finset ι) (fun _ => 0) :=
  ⟨fun _ => (0, 0), by simp, by simp, by simp, by simp⟩

theorem inv_insert {D : ι → ℕ} {m : ℕ} {A : Finset ι} {occ : ℕ → ℕ} {i : ι}
    (hinv : Inv D m A occ) (hi : i ∉ A) {q : ℕ} (hq1 : 1 ≤ q) (hqD : q ≤ D i)
    (hqocc : occ q < m) (hr : ∀ r, q < r → r ≤ D i → m ≤ occ r) :
    Inv D m (insert i A) (Function.update occ q (occ q + 1)) := by
  obtain ⟨f, hinj, hb, hocc, hlat⟩ := hinv
  -- a free vehicle at position `q`
  have hfree : ∃ v, v < m ∧ ∀ j ∈ A, f j ≠ (v, q) := by
    have hlt : ((A.filter (fun j => (f j).2 = q)).image (fun j => (f j).1)).card <
        (range m).card := by
      calc _ ≤ (A.filter (fun j => (f j).2 = q)).card := card_image_le
        _ = occ q := (hocc q).symm
        _ < (range m).card := by simpa using hqocc
    obtain ⟨v, hv, hvn⟩ := exists_mem_notMem_of_card_lt_card hlt
    refine ⟨v, mem_range.1 hv, fun j hj h => hvn ?_⟩
    exact mem_image.2 ⟨j, mem_filter.2 ⟨hj, by rw [h]⟩, by rw [h]⟩
  obtain ⟨v, hvm, hvfree⟩ := hfree
  refine ⟨Function.update f i (v, q), ?_, ?_, ?_, ?_⟩
  · intro a ha b hb' hab
    rw [coe_insert, Set.mem_insert_iff] at ha hb'
    rcases ha with rfl | ha <;> rcases hb' with rfl | hb'
    · rfl
    · exfalso
      have hbi : b ≠ a := fun h => hi (h ▸ hb')
      simp only [Function.update_self, Function.update_of_ne hbi] at hab
      exact hvfree b hb' hab.symm
    · exfalso
      have hai : a ≠ b := fun h => hi (h ▸ ha)
      simp only [Function.update_self, Function.update_of_ne hai] at hab
      exact hvfree a ha hab
    · have hai : a ≠ i := fun h => hi (h ▸ ha)
      have hbi : b ≠ i := fun h => hi (h ▸ hb')
      simp only [Function.update_of_ne hai, Function.update_of_ne hbi] at hab
      exact hinj ha hb' hab
  · intro j hj
    rw [mem_insert] at hj
    rcases hj with rfl | hj
    · simp only [Function.update_self]
      exact ⟨hvm, hq1, hqD⟩
    · have hji : j ≠ i := fun h => hi (h ▸ hj)
      simp only [Function.update_of_ne hji]
      exact hb j hj
  · intro r
    have hcongr : A.filter (fun j => (Function.update f i (v, q) j).2 = r) =
        A.filter (fun j => (f j).2 = r) := by
      apply filter_congr
      intro j hj
      have hji : j ≠ i := fun h => hi (h ▸ hj)
      simp [Function.update_of_ne hji]
    rw [filter_insert]
    have hfi : (Function.update f i (v, q) i).2 = q := by simp
    by_cases hrq : r = q
    · subst hrq
      rw [if_pos' hfi, card_insert_of_notMem (by simp [hi]), hcongr, Function.update_self, ← hocc]
    · rw [if_neg' (by rw [hfi]; exact fun h => hrq h.symm), hcongr, Function.update_of_ne hrq,
        hocc r]
  · intro j hj r hr1 hr2
    rw [mem_insert] at hj
    rcases hj with rfl | hj
    · simp only [Function.update_self] at hr1
      have hrq : r ≠ q := by omega
      rw [Function.update_of_ne hrq]
      exact hr r hr1 hr2
    · have hji : j ≠ i := fun h => hi (h ▸ hj)
      simp only [Function.update_of_ne hji] at hr1
      have := hlat j hj r hr1 hr2
      by_cases hrq : r = q
      · subst hrq
        simp only [Function.update_self]
        omega
      · rw [Function.update_of_ne hrq]
        exact this

/-- The rejection half of Step 5 (the argument with `q*`): if every position
`1..D i` is full then `insert i A` violates the counting criterion. -/
theorem inv_reject {D : ι → ℕ} {m : ℕ} (hm : 1 ≤ m) {A : Finset ι} {occ : ℕ → ℕ} {i : ι}
    (hinv : Inv D m A occ) (hi : i ∉ A) (hfull : ∀ r, 1 ≤ r → r ≤ D i → m ≤ occ r) :
    ¬ SlotFeasible D m (insert i A) := by
  intro hfeas
  obtain ⟨f, hinj, hb, hocc, hlat⟩ := hinv
  have hex : ∃ r, D i < r ∧ occ r < m := by
    refine ⟨A.sup D + D i + 1, by omega, ?_⟩
    rw [hocc, card_eq_zero.2 ?_]
    · omega
    · rw [filter_eq_empty_iff]
      intro j hj hjr
      have := (hb j hj).2.2
      have := le_sup (f := D) hj
      omega
  classical
  obtain ⟨hr0a, hr0b⟩ := Nat.find_spec hex
  set r0 := Nat.find hex with hr0
  have hmin : ∀ r, r < r0 → ¬ (D i < r ∧ occ r < m) := fun r hr => Nat.find_min hex hr
  have hfull' : ∀ r, 1 ≤ r → r < r0 → m ≤ occ r := by
    intro r h1 h2
    by_cases hr : r ≤ D i
    · exact hfull r h1 hr
    · have := hmin r h2
      rw [not_and, not_lt] at this
      exact this (by omega)
  have hclaim : ∀ j ∈ A, (f j).2 < r0 → D j < r0 := by
    intro j hj hjr
    by_contra hcon
    have := hlat j hj r0 hjr (by omega)
    omega
  set k := r0 - 1 with hk
  have hS1 : (A.filter (fun j => (f j).2 < r0)).card =
      ∑ r ∈ Icc 1 k, (A.filter (fun j => (f j).2 = r)).card := by
    rw [card_eq_sum_card_fiberwise (f := fun j => (f j).2) (t := Icc 1 k)]
    · apply sum_congr rfl
      intro r hr
      rw [mem_Icc] at hr
      congr 1
      ext j
      simp only [mem_filter]
      constructor
      · rintro ⟨⟨hj, _⟩, h⟩; exact ⟨hj, h⟩
      · rintro ⟨hj, h⟩; exact ⟨⟨hj, by omega⟩, h⟩
    · intro j hj
      have hj' := mem_filter.1 (mem_coe.1 hj)
      have := (hb j hj'.1).2.1
      simp only [coe_Icc, Set.mem_Icc]
      exact ⟨this, by omega⟩
  have hsum : m * k ≤ ∑ r ∈ Icc 1 k, (A.filter (fun j => (f j).2 = r)).card := by
    calc m * k = ∑ r ∈ Icc 1 k, m := by simp [mul_comm]
      _ ≤ _ := by
        apply sum_le_sum
        intro r hr
        rw [mem_Icc] at hr
        rw [← hocc r]
        exact hfull' r hr.1 (by omega)
  have hNA : m * k ≤ N D A k := by
    refine hsum.trans (hS1 ▸ card_le_card ?_)
    intro j hj
    rw [mem_filter] at hj ⊢
    exact ⟨hj.1, by have := hclaim j hj.1 hj.2; omega⟩
  have := count_le_of_slotFeasible hfeas k
  rw [N_insert hi] at this
  rw [if_pos' (show D i ≤ k by omega)] at this
  omega

theorem algStep_spec {D : ι → ℕ} {m : ℕ} (hm : 1 ≤ m) {A : Finset ι} {occ : ℕ → ℕ} {i : ι}
    (hinv : Inv D m A occ) (hi : i ∉ A) :
    (SlotFeasible D m (insert i A) ↔ 1 ≤ findFree m occ (D i)) ∧
    (1 ≤ findFree m occ (D i) →
      Inv D m (algStep D m (A, occ) i).1 (algStep D m (A, occ) i).2) ∧
    (¬ 1 ≤ findFree m occ (D i) → algStep D m (A, occ) i = (A, occ)) := by
  have hspec := findFree_spec m occ (D i)
  by_cases hq : 1 ≤ findFree m occ (D i)
  · have : (1 ≤ findFree m occ (D i) ∧ findFree m occ (D i) ≤ D i ∧
        occ (findFree m occ (D i)) < m ∧
        ∀ r, findFree m occ (D i) < r → r ≤ D i → m ≤ occ r) := by
      rcases hspec with ⟨h0, -⟩ | h
      · omega
      · exact h
    obtain ⟨h1, h2, h3, h4⟩ := this
    have hnew := inv_insert hinv hi h1 h2 h3 h4
    have hstep : algStep D m (A, occ) i = (insert i A,
        Function.update occ (findFree m occ (D i)) (occ (findFree m occ (D i)) + 1)) := by
      simp [algStep, hq]
    refine ⟨⟨fun _ => hq, fun _ => hnew.slotFeasible⟩, fun _ => ?_, fun h => absurd hq h⟩
    rw [hstep]; exact hnew
  · have h0 : findFree m occ (D i) = 0 := by omega
    have hfull : ∀ r, 1 ≤ r → r ≤ D i → m ≤ occ r := by
      rcases hspec with ⟨-, h⟩ | h
      · exact h
      · omega
    refine ⟨⟨fun h => absurd h (inv_reject hm hinv hi hfull), fun h => absurd h hq⟩,
      fun h => absurd h hq, fun _ => ?_⟩
    simp [algStep, hq]

theorem algRun_from {D : ι → ℕ} {m : ℕ} (hm : 1 ≤ m) :
    ∀ (L : List ι) (A : Finset ι) (occ : ℕ → ℕ), L.Nodup → (∀ y ∈ L, y ∉ A) →
      Inv D m A occ →
      (L.foldl (algStep D m) (A, occ)).1 = greedyFrom (SlotFeasible D m) A L := by
  intro L
  induction L with
  | nil => intro A occ _ _ _; rfl
  | cons x L ih =>
    intro A occ hnd hdisj hinv
    rw [List.nodup_cons] at hnd
    have hxA : x ∉ A := hdisj x (by simp)
    obtain ⟨h1, h2, h3⟩ := algStep_spec hm hinv hxA
    rw [List.foldl_cons]
    by_cases hq : 1 ≤ findFree m occ (D x)
    · have hfe : SlotFeasible D m (insert x A) := h1.2 hq
      rw [greedyFrom_cons_of_indep L hfe]
      have hinv' := h2 hq
      have hstep : (algStep D m (A, occ) x).1 = insert x A := by simp [algStep, hq]
      have hdisj' : ∀ y ∈ L, y ∉ insert x A := by
        intro y hy hy'
        rw [mem_insert] at hy'
        rcases hy' with rfl | hy'
        · exact hnd.1 hy
        · exact hdisj y (List.mem_cons_of_mem _ hy) hy'
      have key := ih (algStep D m (A, occ) x).1 (algStep D m (A, occ) x).2 hnd.2
        (by rw [hstep]; exact hdisj') hinv'
      rw [← hstep]
      exact key
    · have hfe : ¬ SlotFeasible D m (insert x A) := fun h => hq (h1.1 h)
      rw [greedyFrom_cons_of_not L hfe, h3 hq]
      exact ih A occ hnd.2 (fun y hy => hdisj y (List.mem_cons_of_mem _ hy)) hinv

/-- **Step 5 of Theorem 5.**  Algorithm 3's "latest free slot" rule keeps exactly the
sites that the greedy algorithm over the slot-feasible family keeps; i.e. it keeps `i`
exactly when (kept set) `∪ {i}` is slot-feasible.  (Needs `m ≥ 1`.) -/
theorem algRun_eq_greedy {D : ι → ℕ} {m : ℕ} (hm : 1 ≤ m) (L : List ι) (hnd : L.Nodup) :
    algRun D m L = greedy (SlotFeasible D m) L :=
  algRun_from hm L ∅ (fun _ => 0) hnd (by simp) (inv_empty D m)

/-- Per-step form: with `A` the kept set so far, `i ∉ A` is accepted by the
latest-free-slot rule iff `insert i A` is slot-feasible. -/
theorem algStep_accepts_iff {D : ι → ℕ} {m : ℕ} (hm : 1 ≤ m) {A : Finset ι} {occ : ℕ → ℕ}
    {i : ι} (hinv : Inv D m A occ) (hi : i ∉ A) :
    1 ≤ findFree m occ (D i) ↔ SlotFeasible D m (insert i A) :=
  (algStep_spec hm hinv hi).1.symm

end Impl

/-- **Theorem 5, assembled.**  The set kept by Algorithm 3's latest-free-slot rule
(processing the sites by non-increasing weight) can be served on time by `m ≥ 1` vehicles
and has maximum weight among all sets that can. -/
theorem equalP_algRun_optimal {p : ℕ} (hp : 0 < p) (d w : ι → ℕ) {m : ℕ} (hm : 1 ≤ m)
    (L : List ι) (hnd : L.Nodup) (hall : ∀ i, i ∈ L)
    (hsort : L.Pairwise (fun a b => w b ≤ w a)) :
    Schedulable p d m (algRun (fun i => d i / p) m L) ∧
      ∀ S : Finset ι, Schedulable p d m S →
        ∑ i ∈ S, w i ≤ ∑ i ∈ algRun (fun i => d i / p) m L, w i := by
  rw [algRun_eq_greedy hm L hnd]
  exact equalP_greedy_optimal_sets hp d w m L hnd hall hsort

namespace Example5

/-- Algorithm 3's rule reproduces the worked example: sites `1,3,4` are kept (`m = 1`). -/
theorem algRun_m1 : algRun (fun i => d i / 2) 1 L = {0, 2, 3} := by decide

theorem algRun_m2 : algRun (fun i => d i / 2) 2 L = Finset.univ := by decide

end Example5

end Mwhed.EqualDispatch
