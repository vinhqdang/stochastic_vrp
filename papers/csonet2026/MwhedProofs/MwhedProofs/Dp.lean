import MwhedProofs.Core

/-!
# The exact dynamic program (Theorem 3 of the paper)

Paper statement: *MWHED can be solved exactly in `O(n·P)` time and space,
`P = ∑ pᵢ`*.  The proof in the paper has two parts: (a) a correctness argument
for the recursion (eq. `dp`) defining the table `f(i,t)`, and (b) a count of the
table entries.  Only (a) is formalised here; (b) is a statement about running
time and is not expressible in Lean's logic without fixing a machine model
(the table has `(n+1)(P+1)` entries and every entry is computed from two others,
visible in the definition of `dpStep`).

Correspondence:

* `dpInit`, `dpStep`, `dpRow`  — the table.  `dpStep` is the right-hand side of
  eq. `dp` (with the convention that the second term is `-∞` unless
  `p i ≤ t ≤ d i`), `dpInit` is the row `f(0,·)` (`f(0,0)=0`, `f(0,t>0)=-∞`),
  and `dpRow L t` is `f(|L|, t)` for the list `L` of sites sorted by
  non-decreasing deadline.  Values live in `WithBot ℕ`, `⊥ = -∞`.
* `dpTable L i t = dpRow (L.take i) t` is the paper's `f(i,t)`, and
  `dpTable_succ` is eq. `dp` verbatim.
* `dpRow_eq_bestWeight` — correctness of the table: `f(i,t)` is the maximum of
  the total weight over subsets `S` of the first `i` sites with total dispatch
  time exactly `t` that are *feasible* (served in EDD order from time 0, all on
  time), and `⊥` if there is none.
* `isOPT_iff_dp_max` — "the optimal value is `max_{t=0..P} f(n,t)`": `v` is the
  optimal value `W*` of the instance iff `v = max_{t ∈ {0..P}} f(n,t)` for the
  table over the EDD list of all sites.
* `dpTable_eq_bestWeight` — the same for the paper's indexed `f(i,t)`.
-/

namespace Mwhed

section Table

variable {ι : Type*} (I : Inst ι)

/-- The row `f(0,·)`: `f(0,0) = 0`, `f(0,t) = -∞` for `t > 0`. -/
def dpInit : ℕ → WithBot ℕ := fun t => if t = 0 then 0 else ⊥

/-- One step of the recursion, eq. `dp`:
`f(i,t) = max (f(i-1,t), [t ≤ d_i] · (f(i-1,t-p_i) + w_i))`, the second term being
`-∞` (`⊥`) unless `p_i ≤ t ≤ d_i`. -/
def dpStep (i : ι) (g : ℕ → WithBot ℕ) (t : ℕ) : WithBot ℕ :=
  max (g t) (if I.p i ≤ t ∧ t ≤ I.d i then g (t - I.p i) + ((I.w i : ℕ) : WithBot ℕ) else ⊥)

/-- The table row after processing the sites of `L` in the given order. -/
def dpRow (L : List ι) : ℕ → WithBot ℕ :=
  L.foldl (fun g i => dpStep I i g) dpInit

/-- The paper's `f(i,t)`: the row after the first `i` sites of `L`. -/
def dpTable (L : List ι) (i t : ℕ) : WithBot ℕ := dpRow I (L.take i) t

theorem dpRow_nil : dpRow I ([] : List ι) = dpInit := rfl

theorem dpRow_append_singleton (L : List ι) (i : ι) :
    dpRow I (L ++ [i]) = dpStep I i (dpRow I L) := by
  simp [dpRow, List.foldl_append]

/-- Equation `dp` of the paper, for the table `dpTable`. -/
theorem dpTable_succ (L : List ι) (i : ℕ) (hi : i < L.length) (t : ℕ) :
    dpTable I L (i + 1) t = dpStep I L[i] (dpTable I L i) t := by
  have h : L.take (i + 1) = L.take i ++ [L[i]] := by
    rw [List.take_succ, List.getElem?_eq_getElem hi]; rfl
  unfold dpTable
  rw [h, dpRow_append_singleton]

end Table

section Correctness

variable {ι : Type*} [DecidableEq ι] [Fintype ι] (I : Inst ι)

open Classical in
/-- `f(L,t)` as specified in the paper: the maximum weight of a feasible subset
of `L` (feasible: served in EDD order from time `0`, everyone on time) of total
dispatch time exactly `t`; `⊥ = -∞` if there is no such subset. -/
noncomputable def bestWeight (L : List ι) (t : ℕ) : WithBot ℕ :=
  (L.toFinset.powerset.filter (fun S => time I S = t ∧ AllOnTime I 0 (edd I S))).sup
    (fun S => ((weight I S : ℕ) : WithBot ℕ))

open Classical in
/-- The same with feasibility expressed through the threshold characterisation. -/
noncomputable def bestWeightThr (L : List ι) (t : ℕ) : WithBot ℕ :=
  (L.toFinset.powerset.filter (fun S => time I S = t ∧ Thr I S)).sup
    (fun S => ((weight I S : ℕ) : WithBot ℕ))

theorem bestWeight_eq_thr (L : List ι) (t : ℕ) : bestWeight I L t = bestWeightThr I L t := by
  classical
  unfold bestWeight bestWeightThr
  congr 1
  apply Finset.filter_congr
  intro S _
  rw [← lemma1_edd, feasible_iff_thr]

/-- `(max_S f S) + c = max_S (f S + c)` in `WithBot ℕ`. -/
theorem sup_coe_add {α : Type*} (s : Finset α) (f : α → ℕ) (c : ℕ) :
    s.sup (fun x => (((c + f x : ℕ)) : WithBot ℕ)) =
      s.sup (fun x => ((f x : ℕ) : WithBot ℕ)) + (c : WithBot ℕ) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    rw [Finset.sup_insert, Finset.sup_insert, ih]
    push_cast
    rw [add_comm (c : WithBot ℕ) (f a : WithBot ℕ)]
    exact (max_add_add_right (f a : WithBot ℕ) _ (c : WithBot ℕ))

/-- The threshold condition for `insert i S`, when `i` has the largest deadline. -/
theorem thr_insert_iff {S : Finset ι} {i : ι} (hi : i ∉ S) (hd : ∀ j ∈ S, I.d j ≤ I.d i) :
    Thr I (insert i S) ↔ Thr I S ∧ time I S + I.p i ≤ I.d i := by
  have hall : (insert i S).filter (fun x => I.d x ≤ I.d i) = insert i S := by
    apply Finset.filter_true_of_mem
    intro x hx
    rcases Finset.mem_insert.1 hx with rfl | hx
    · exact le_rfl
    · exact hd x hx
  have hsum : ∑ x ∈ insert i S, I.p x = time I S + I.p i := by
    rw [Finset.sum_insert hi, time]; ring
  constructor
  · intro h
    refine ⟨fun t => le_trans ?_ (h t), ?_⟩
    · exact Finset.sum_le_sum_of_subset
        (Finset.filter_subset_filter _ (Finset.subset_insert _ _))
    · have := h (I.d i)
      rwa [hall, hsum] at this
  · rintro ⟨h1, h2⟩ t
    by_cases ht : I.d i ≤ t
    · have : (insert i S).filter (fun x => I.d x ≤ t) = insert i S := by
        apply Finset.filter_true_of_mem
        intro x hx
        rcases Finset.mem_insert.1 hx with rfl | hx
        · exact ht
        · exact le_trans (hd x hx) ht
      rw [this, hsum]; omega
    · rw [Finset.filter_insert, if_neg ht]
      exact h1 t

theorem time_insert {S : Finset ι} {i : ι} (hi : i ∉ S) :
    time I (insert i S) = time I S + I.p i := by
  rw [time, Finset.sum_insert hi, time]; ring

theorem weight_insert {S : Finset ι} {i : ι} (hi : i ∉ S) :
    weight I (insert i S) = I.w i + weight I S := by
  rw [weight, Finset.sum_insert hi, weight]

/-- Correctness of the recursion (eq. `dp`), in threshold form. -/
theorem dpRow_eq_bestWeightThr :
    ∀ (L : List ι), L.Nodup → L.Pairwise (fun a b => I.d a ≤ I.d b) →
      ∀ t, dpRow I L t = bestWeightThr I L t := by
  classical
  intro L
  induction L using List.reverseRecOn with
  | nil =>
    intro _ _ t
    have hT : Thr I (∅ : Finset ι) := by intro t; simp
    unfold bestWeightThr
    have h0 : (([] : List ι).toFinset.powerset.filter (fun S => time I S = t ∧ Thr I S))
        = if t = 0 then {∅} else ∅ := by
      ext S
      by_cases ht : t = 0
      · subst ht
        simp only [List.toFinset_nil, Finset.powerset_empty, Finset.mem_filter,
          Finset.mem_singleton, if_true]
        constructor
        · rintro ⟨h, _⟩; exact h
        · rintro rfl; exact ⟨rfl, by simp [time], hT⟩
      · simp only [List.toFinset_nil, Finset.powerset_empty, Finset.mem_filter,
          Finset.mem_singleton, if_neg ht, Finset.notMem_empty, iff_false]
        rintro ⟨rfl, h, _⟩
        exact ht (by simpa [time] using h.symm)
    rw [h0]
    by_cases ht : t = 0
    · simp [dpRow_nil, dpInit, ht, weight]
    · simp [dpRow_nil, dpInit, ht]
  | append_singleton L' i ih =>
    intro hnd hpw t
    rw [List.nodup_append] at hnd
    rw [List.pairwise_append] at hpw
    have ih' := ih hnd.1 hpw.1
    have hiA : i ∉ L'.toFinset := by
      intro h
      exact hnd.2.2 i (List.mem_toFinset.1 h) i (List.mem_singleton_self i) rfl
    have hdA : ∀ j ∈ L'.toFinset, I.d j ≤ I.d i := fun j hj =>
      hpw.2.2 j (List.mem_toFinset.1 hj) i (List.mem_singleton_self i)
    have htf : (L' ++ [i]).toFinset = insert i L'.toFinset := by
      ext x; simp
    rw [dpRow_append_singleton]
    simp only [dpStep, ih']
    unfold bestWeightThr
    rw [htf, Finset.powerset_insert, Finset.filter_union, Finset.sup_union, Finset.filter_image,
      Finset.sup_image]
    change _ = _ ⊔ _
    -- the powerset members are subsets of `L'.toFinset`
    have hmem : ∀ S ∈ L'.toFinset.powerset, i ∉ S ∧ ∀ j ∈ S, I.d j ≤ I.d i := by
      intro S hS
      have hS' := Finset.mem_powerset.1 hS
      exact ⟨fun h => hiA (hS' h), fun j hj => hdA j (hS' hj)⟩
    by_cases hc : I.p i ≤ t ∧ t ≤ I.d i
    · rw [if_pos hc]
      have hfilt : L'.toFinset.powerset.filter
            (fun S => time I (insert i S) = t ∧ Thr I (insert i S)) =
          L'.toFinset.powerset.filter (fun S => time I S = t - I.p i ∧ Thr I S) := by
        apply Finset.filter_congr
        intro S hS
        obtain ⟨h1, h2⟩ := hmem S hS
        rw [time_insert I h1, thr_insert_iff I h1 h2]
        constructor
        · rintro ⟨h3, h4, h5⟩; exact ⟨by omega, h4⟩
        · rintro ⟨h3, h4⟩; exact ⟨by omega, h4, by omega⟩
      have hs : (L'.toFinset.powerset.filter
            (fun S => time I (insert i S) = t ∧ Thr I (insert i S))).sup
            ((fun S => ((weight I S : ℕ) : WithBot ℕ)) ∘ insert i) =
          (L'.toFinset.powerset.filter (fun S => time I S = t - I.p i ∧ Thr I S)).sup
            (fun S => ((weight I S : ℕ) : WithBot ℕ)) + ((I.w i : ℕ) : WithBot ℕ) := by
        rw [← sup_coe_add, hfilt]
        apply Finset.sup_congr rfl
        intro S hS
        have hS' := (hmem S (Finset.mem_filter.1 hS).1).1
        simp [Function.comp, weight_insert I hS']
      rw [hs]
    · rw [if_neg hc]
      have hfilt : L'.toFinset.powerset.filter
            (fun S => time I (insert i S) = t ∧ Thr I (insert i S)) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        intro S hS ⟨h3, h4⟩
        obtain ⟨h1, h2⟩ := hmem S hS
        rw [time_insert I h1] at h3
        rw [thr_insert_iff I h1 h2] at h4
        exact hc ⟨by omega, by omega⟩
      rw [hfilt]
      simp

/-- **Correctness of the table** (the induction in the proof of Theorem 3).
For a list `L` of distinct sites sorted by non-decreasing deadline, the table
entry `f(L,t)` is the maximum, over subsets `S` of the sites of `L` that are
feasible (the EDD list of `S`, served from time `0`, has every site on time) and
have total dispatch time exactly `t`, of the total weight of `S`; it is `⊥ = -∞`
when there is no such subset. -/
theorem dpRow_eq_bestWeight (L : List ι) (hnd : L.Nodup)
    (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) (t : ℕ) :
    dpRow I L t = bestWeight I L t := by
  rw [dpRow_eq_bestWeightThr I L hnd hs t, bestWeight_eq_thr]

/-- The paper's `f(i,t)` (`i` sites, time exactly `t`) is the maximum weight of a
feasible subset of the first `i` sites of `L` of total time `t`. -/
theorem dpTable_eq_bestWeight (L : List ι) (hnd : L.Nodup)
    (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) (i t : ℕ) :
    dpTable I L i t = bestWeight I (L.take i) t :=
  dpRow_eq_bestWeight I _ (hnd.sublist (List.take_sublist _ _)) (hs.sublist (List.take_sublist _ _)) t

/-- **Theorem 3 (exact DP), headline statement.**  Let `L = edd univ` be the
list of all sites sorted by non-decreasing deadline and `P = ∑ pᵢ`.  Then `v` is
the optimal value `W*` of the instance iff `v = max_{t=0..P} f(n,t)`, where
`f(n,t) = dpRow L t` is the table of eq. `dp` (in `WithBot ℕ`). -/
theorem isOPT_iff_dp_max (v : ℕ) :
    IsOPT I v ↔
      (v : WithBot ℕ) =
        (Finset.range (time I Finset.univ + 1)).sup (fun t => dpRow I (edd I Finset.univ) t) := by
  classical
  have hM : (Finset.range (time I Finset.univ + 1)).sup (fun t => dpRow I (edd I Finset.univ) t) =
      (Finset.range (time I Finset.univ + 1)).sup (fun t =>
        (Finset.univ.filter (fun S : Finset ι => time I S = t ∧ AllOnTime I 0 (edd I S))).sup
          (fun S => ((weight I S : ℕ) : WithBot ℕ))) := by
    apply Finset.sup_congr rfl
    intro t _
    rw [dpRow_eq_bestWeight I _ (nodup_edd I _) (pairwise_edd I _)]
    unfold bestWeight
    rw [toFinset_edd, Finset.powerset_univ]
  rw [hM]
  have fwd : ∀ w, IsOPT I w → (w : WithBot ℕ) =
      (Finset.range (time I Finset.univ + 1)).sup (fun t =>
        (Finset.univ.filter (fun S : Finset ι => time I S = t ∧ AllOnTime I 0 (edd I S))).sup
          (fun S => ((weight I S : ℕ) : WithBot ℕ))) := by
    intro w hw
    obtain ⟨⟨S, hS, hSw⟩, hmax⟩ := (isOPT_iff_max_feasible I w).1 hw
    apply le_antisymm
    · rw [← hSw]
      have hSt : time I S < time I Finset.univ + 1 :=
        Nat.lt_succ_of_le (Finset.sum_le_sum_of_subset (Finset.subset_univ S))
      refine le_trans ?_ (Finset.le_sup (f := fun t =>
        (Finset.univ.filter (fun S : Finset ι => time I S = t ∧ AllOnTime I 0 (edd I S))).sup
          (fun S => ((weight I S : ℕ) : WithBot ℕ))) (Finset.mem_range.2 hSt))
      exact Finset.le_sup (f := fun S => ((weight I S : ℕ) : WithBot ℕ))
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl, (lemma1_edd I S).1 hS⟩)
    · apply Finset.sup_le
      intro t _
      apply Finset.sup_le
      intro S hS
      have := hmax S ((lemma1_edd I S).2 (Finset.mem_filter.1 hS).2.2)
      exact_mod_cast this
  constructor
  · exact fwd v
  · intro h
    obtain ⟨w, hw⟩ := exists_isOPT I
    have := fwd w hw
    rw [← h] at this
    have hvw : v = w := (by exact_mod_cast this : w = v).symm
    rwa [hvw]

end Correctness
