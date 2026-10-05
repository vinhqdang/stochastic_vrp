import MwhedProofs.Fptas
import MwhedProofs.Dp
import MwhedProofs.Hardness

/-!
# Presentation lemmas for the DP recursions and the Partition preprocessing
# (items 4(a), 4(b), 4(e) of the revision)

Everything here is additional to `Fptas.lean`, `Dp.lean`, `Hardness.lean` (not modified).

## Part G: the value-indexed table (Lemma `lem:g`, ITEM 4(a))

`Fptas.lean` already proves `gTab_eq_iInf` at the level of *sublists* of a list `L`
that are served in the order of `L`.  Here the bridge to *feasible sets* is added:

* `feasible_of_allOnTime_list`  : a duplicate-free list that is on time from `0` is a feasible set.
* `allOnTime_filter_of_feasible` : conversely a feasible set, listed in the (deadline-sorted)
  order of `L`, is on time.
* `gSpec L i v` : the paper's `g(i,v)`, defined *semantically*: the minimum
  (`⨅` in `ℕ∞`, `⊤ = +∞` for the empty family) of `time S` over the feasible sets
  `S ⊆ {1..i}` (the first `i` sites of the deadline-sorted list `L`) of scaled value `v`.
* `gSpec_eq_gTab`  : for `L` duplicate-free and sorted by deadline, `gSpec` is `gTab`.
* `lem_g_zero_zero`, `lem_g_zero_pos`, `lem_g_succ` : the statement of Lemma `lem:g`, i.e.
  `g(0,0) = 0`, `g(0,v) = +∞` and the explicit-case recursion (eq:g), including the
  `+∞` case.  `lem_g_succ_zero_scaled` : zero-scaled sites are never selected.

## Part F: the f-recursion with cases (eq:dp, ITEM 4(b))

`Dp.lean` defines `dpStep` literally with the `p_i ≤ t ≤ d_i` guard and `-∞` otherwise
(`dpTable_succ`).  Added: `dpTable_zero` (`f(0,0)=0`, `f(0,t>0)=-∞`), `dpTable_succ_cases`
(the displayed two-case equation, with `-∞ + w = -∞`), and `dpTable_eq_feasible_sup`
(`f(i,t)` is the maximum weight of a *feasible* `S ⊆ {1..i}` of time exactly `t`).

## Part P: Theorem 2 preprocessing (ITEM 4(e))

`partition_no_of_odd`, `partition_no_of_big` (`a_i > A/2`), `noInst` (one site `p=d=w=1`,
optimum `1 < 2`), and the combined `partition_reduction_full`.
-/

namespace Mwhed

section PartG

variable {ι : Type*} [DecidableEq ι]

/-- A duplicate-free list of sites that is on time from time `0` is a feasible set. -/
theorem feasible_of_allOnTime_list [Fintype ι] (I : Inst ι) {τ : List ι} (hτ : τ.Nodup)
    (h : AllOnTime I 0 τ) : Feasible I τ.toFinset := by
  classical
  have hnd : (τ ++ (Finset.univ \ τ.toFinset).toList).Nodup := by
    rw [List.nodup_append]
    refine ⟨hτ, Finset.nodup_toList _, ?_⟩
    intro a ha b hb hab
    subst hab
    simp only [Finset.mem_toList, Finset.mem_sdiff, List.mem_toFinset] at hb
    exact hb.2 ha
  refine ⟨τ ++ (Finset.univ \ τ.toFinset).toList, ⟨hnd, ?_⟩, ?_⟩
  · intro i
    by_cases hi : i ∈ τ
    · exact List.mem_append_left _ hi
    · exact List.mem_append_right _ (by simp [hi])
  · have h1 : OnTimeOn I τ.toFinset 0 (τ ++ (Finset.univ \ τ.toFinset).toList) := by
      rw [onTimeOn_append]
      refine ⟨allOnTime_imp_onTimeOn I _ _ _ h, onTimeOn_of_forall_not_mem I _ _ _ ?_⟩
      intro i hi
      simp only [Finset.mem_toList, Finset.mem_sdiff, List.mem_toFinset] at hi
      simpa using hi.2
    intro i hi
    have hi' : i ∈ τ := List.mem_toFinset.1 hi
    have := (onTimeOn_iff_completion I τ.toFinset _ 0 hnd).1 h1 i
      (List.mem_append_left _ hi') hi
    simpa using this

/-- A feasible set, listed in the order of a deadline-sorted duplicate-free list `L`
(`L.filter (· ∈ S)`), is on time from time `0`. -/
theorem allOnTime_filter_of_feasible [Fintype ι] (I : Inst ι) {L : List ι} (hnd : L.Nodup)
    (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) {S : Finset ι} (hF : Feasible I S) :
    AllOnTime I 0 (L.filter (fun i => decide (i ∈ S))) := by
  have hthr := thr_of_feasible I hF
  have hndf : (L.filter (fun i => decide (i ∈ S))).Nodup := hnd.filter _
  apply allOnTime_of_sorted I _ _ (hs.filter _)
  intro i hi
  have hiS : i ∈ S := by simpa using (List.mem_filter.1 hi).2
  rw [sum_filter_list I hndf]
  have hsub : (L.filter (fun i => decide (i ∈ S))).toFinset.filter (fun j => I.d j ≤ I.d i)
      ⊆ S.filter (fun j => I.d j ≤ I.d i) := by
    intro j hj
    rw [Finset.mem_filter, List.mem_toFinset, List.mem_filter] at hj
    exact Finset.mem_filter.2 ⟨by simpa using hj.1.2, hj.2⟩
  calc ∑ j ∈ (L.filter (fun i => decide (i ∈ S))).toFinset.filter (fun j => I.d j ≤ I.d i), I.p j
      ≤ ∑ j ∈ S.filter (fun j => I.d j ≤ I.d i), I.p j := Finset.sum_le_sum_of_subset hsub
    _ ≤ I.d i := hthr (I.d i)

/-- The paper's `g(i, v)`, defined semantically: the least total dispatch time of a feasible set
`S ⊆ {1,…,i}` (the first `i` sites of the deadline-sorted list `L`) of scaled value `Σ w' = v`;
`⊤ = +∞` when no such set exists (`⨅ ∅ = ⊤`). -/
noncomputable def gSpec [Fintype ι] (I : Inst ι) (w' : ι → ℕ) (L : List ι) (i v : ℕ) : ℕ∞ :=
  ⨅ S : {S : Finset ι // S ⊆ (L.take i).toFinset ∧ Feasible I S ∧ ∑ j ∈ S, w' j = v},
    (time I S.1 : ℕ∞)

/-- `gSpec` (feasible *sets*) coincides with `gTab` (on-time *sublists* of the sorted list). -/
theorem gSpec_eq_gTab [Fintype ι] (I : Inst ι) (w' : ι → ℕ) {L : List ι} (hnd : L.Nodup)
    (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) (i v : ℕ) :
    gSpec I w' L i v = gTab I w' (L.take i) v := by
  have hnd' : (L.take i).Nodup := hnd.sublist (List.take_sublist _ _)
  have hs' : (L.take i).Pairwise (fun a b => I.d a ≤ I.d b) := hs.sublist (List.take_sublist _ _)
  generalize L.take i = M at hnd' hs'
  unfold gSpec
  apply le_antisymm
  · -- `gSpec ≤ gTab`
    rcases gTab_attained I w' M v with h | ⟨S, hS, hon, hv, hg⟩
    · rw [h]; exact le_top
    · rw [hg]
      have hSnd : S.Nodup := hnd'.sublist hS
      have hmem : S.toFinset ⊆ M.toFinset ∧ Feasible I S.toFinset ∧
          ∑ j ∈ S.toFinset, w' j = v := by
        refine ⟨?_, feasible_of_allOnTime_list I hSnd hon, ?_⟩
        · intro x hx; exact List.mem_toFinset.2 (hS.subset (List.mem_toFinset.1 hx))
        · rw [List.sum_toFinset _ hSnd]; exact hv
      have := iInf_le (fun T : {S : Finset ι // S ⊆ M.toFinset ∧ Feasible I S ∧
        ∑ j ∈ S, w' j = v} => (time I T.1 : ℕ∞)) ⟨S.toFinset, hmem⟩
      refine this.trans (le_of_eq ?_)
      simp only [time]
      rw [List.sum_toFinset _ hSnd]
  · -- `gTab ≤ gSpec`
    refine le_iInf fun T => ?_
    obtain ⟨hTsub, hTF, hTv⟩ := T.2
    have hSnd : (M.filter (fun i => decide (i ∈ T.1))).Nodup := hnd'.filter _
    have hSfin : (M.filter (fun i => decide (i ∈ T.1))).toFinset = T.1 := by
      ext x
      simp only [List.mem_toFinset, List.mem_filter, decide_eq_true_eq]
      constructor
      · exact fun h => h.2
      · intro hx; exact ⟨List.mem_toFinset.1 (hTsub hx), hx⟩
    have hon := allOnTime_filter_of_feasible I hnd' hs' hTF
    have hv : ((M.filter (fun i => decide (i ∈ T.1))).map w').sum = v := by
      rw [← List.sum_toFinset _ hSnd, hSfin]; exact hTv
    have := gTab_le I w' M v _ List.filter_sublist hon hv
    rw [← List.sum_toFinset _ hSnd, hSfin] at this
    exact this

variable {I : Inst ι} {w' : ι → ℕ}

/-- **Lemma `lem:g`, base**: `g(0,0) = 0`. -/
theorem lem_g_zero_zero [Fintype ι] (I : Inst ι) (w' : ι → ℕ) {L : List ι} (hnd : L.Nodup)
    (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) : gSpec I w' L 0 0 = 0 := by
  rw [gSpec_eq_gTab I w' hnd hs]; simp [gTab_nil]

/-- **Lemma `lem:g`, base**: `g(0,v) = +∞` for `v > 0`. -/
theorem lem_g_zero_pos [Fintype ι] (I : Inst ι) (w' : ι → ℕ) {L : List ι} (hnd : L.Nodup)
    (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) {v : ℕ} (hv : 0 < v) :
    gSpec I w' L 0 v = ⊤ := by
  rw [gSpec_eq_gTab I w' hnd hs]; simp [gTab_nil, hv.ne']

/-- **Lemma `lem:g`, recursion (eq:g)**, with the explicit cases.  For the deadline-sorted
duplicate-free list `L` of sites and `i < |L|`, `g(i+1, v)` (the least time of a feasible
`S ⊆ {1..i+1}` with scaled value `v`) equals
`min {g(i,v), g(i,v-w'_{i+1}) + p_{i+1}}` if `v ≥ w'_{i+1}` and
`g(i,v-w'_{i+1}) + p_{i+1} ≤ d_{i+1}` (in `ℕ∞`, so `+∞` fails the test), and `g(i,v)` otherwise. -/
theorem lem_g_succ [Fintype ι] (I : Inst ι) (w' : ι → ℕ) {L : List ι} (hnd : L.Nodup)
    (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) {i : ℕ} (hi : i < L.length) (v : ℕ) :
    gSpec I w' L (i + 1) v =
      if w' L[i] ≤ v ∧ gSpec I w' L i (v - w' L[i]) + (I.p L[i] : ℕ∞) ≤ (I.d L[i] : ℕ∞) then
        min (gSpec I w' L i v) (gSpec I w' L i (v - w' L[i]) + (I.p L[i] : ℕ∞))
      else gSpec I w' L i v := by
  have h : L.take (i + 1) = L.take i ++ [L[i]] := by
    rw [List.take_succ, List.getElem?_eq_getElem hi]; rfl
  rw [gSpec_eq_gTab I w' hnd hs, gSpec_eq_gTab I w' hnd hs, gSpec_eq_gTab I w' hnd hs, h,
    gTab_append_singleton]
  split_ifs with hc
  · rfl
  · simp

/-- Zero-scaled sites are never selected: if `w'_{i+1} = 0` then `g(i+1, v) = g(i, v)`. -/
theorem lem_g_succ_zero_scaled [Fintype ι] (I : Inst ι) (w' : ι → ℕ) {L : List ι}
    (hnd : L.Nodup) (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) {i : ℕ} (hi : i < L.length)
    (hw : w' L[i] = 0) (v : ℕ) : gSpec I w' L (i + 1) v = gSpec I w' L i v := by
  rw [lem_g_succ I w' hnd hs hi v, hw]
  split_ifs with hc
  · exact min_eq_left le_self_add
  · rfl

end PartG

/-! ## Part F: the f-recursion with cases -/

section PartF

variable {ι : Type*} (I : Inst ι)

/-- `f(0, 0) = 0` and `f(0, t) = -∞` for `t > 0`. -/
theorem dpTable_zero (L : List ι) (t : ℕ) :
    dpTable I L 0 t = if t = 0 then 0 else ⊥ := by
  simp [dpTable, dpRow, dpInit]

/-- `-∞ + w = -∞` (the convention of eq:dp). -/
theorem bot_add_weight (w : ℕ) : (⊥ : WithBot ℕ) + ((w : ℕ) : WithBot ℕ) = ⊥ :=
  WithBot.bot_add _

/-- **Equation (eq:dp) with cases.**  `f(i+1,t) = max {f(i,t), f(i,t-p_{i+1}) + w_{i+1}}` if
`p_{i+1} ≤ t ≤ d_{i+1}` and `f(i,t)` otherwise (`WithBot ℕ`, `⊥ = -∞`). -/
theorem dpTable_succ_cases (L : List ι) (i : ℕ) (hi : i < L.length) (t : ℕ) :
    dpTable I L (i + 1) t =
      if I.p L[i] ≤ t ∧ t ≤ I.d L[i] then
        max (dpTable I L i t) (dpTable I L i (t - I.p L[i]) + ((I.w L[i] : ℕ) : WithBot ℕ))
      else dpTable I L i t := by
  rw [dpTable_succ I L i hi t]
  unfold dpStep
  split_ifs <;> simp

variable [DecidableEq ι] [Fintype ι]

open Classical in
/-- **Semantics of the f-table.**  For `L` duplicate-free and sorted by deadline, `f(i,t)` is the
maximum weight of a *feasible* set `S ⊆ {1..i}` of total dispatch time exactly `t` (`⊥` if there is
none), the feasibility being that of Definition 1 (Lemma 1 translates the EDD form of `Dp.lean`). -/
theorem dpTable_eq_feasible_sup (L : List ι) (hnd : L.Nodup)
    (hs : L.Pairwise (fun a b => I.d a ≤ I.d b)) (i t : ℕ) :
    dpTable I L i t =
      ((L.take i).toFinset.powerset.filter (fun S => time I S = t ∧ Feasible I S)).sup
        (fun S => ((weight I S : ℕ) : WithBot ℕ)) := by
  rw [dpTable_eq_bestWeight I L hnd hs i t]
  unfold bestWeight
  congr 1
  apply Finset.filter_congr
  intro S _
  rw [lemma1_edd]

end PartF

/-! ## Part P: Theorem 2, preprocessing of the Partition data -/

section PartP

variable {n : ℕ}

/-- Odd total: Partition is a no-instance. -/
theorem partition_no_of_odd (a : Fin n → ℕ) (hodd : ¬ 2 ∣ ∑ i, a i) :
    ¬ ∃ S : Finset (Fin n), 2 * ∑ i ∈ S, a i = ∑ i, a i := by
  rintro ⟨S, hS⟩
  exact hodd ⟨_, hS.symm⟩

/-- Some `a_i > A/2`: every subset containing `i` sums to more than `A/2` and every other subset to
less than `A/2`, so Partition is a no-instance. -/
theorem partition_no_of_big (a : Fin n → ℕ) {i : Fin n} (hi : ∑ j, a j < 2 * a i) :
    ¬ ∃ S : Finset (Fin n), 2 * ∑ i ∈ S, a i = ∑ i, a i := by
  rintro ⟨S, hS⟩
  by_cases h : i ∈ S
  · have : a i ≤ ∑ j ∈ S, a j := Finset.single_le_sum (fun _ _ => Nat.zero_le _) h
    omega
  · have : ∑ j ∈ insert i S, a j ≤ ∑ j, a j :=
      Finset.sum_le_sum_of_subset (Finset.subset_univ _)
    rw [Finset.sum_insert h] at this
    omega

/-- The fixed instance output in the degenerate cases: one site with `p = d = w = 1`. -/
def noInst : Inst (Fin 1) where
  p _ := 1
  d _ := 1
  w _ := 1
  p_pos _ := Nat.one_pos

theorem noInst_indivFeasible : IndivFeasible noInst := fun _ => le_rfl

/-- Its optimum is `1` (so `W* < 2 = k`). -/
theorem noInst_isOPT : IsOPT noInst 1 := by
  refine ⟨⟨[0], ⟨by simp, fun i => by simp [Subsingleton.elim i 0]⟩, by simp [W, onTimeW, noInst]⟩,
    fun σ hσ => ?_⟩
  rw [W_eq_weight_onTimeSet noInst hσ.1]
  calc weight noInst (onTimeSet noInst σ) ≤ weight noInst Finset.univ :=
        Finset.sum_le_sum_of_subset (Finset.subset_univ _)
    _ = 1 := by simp [weight, noInst]

/-- The condition under which the reduction outputs the fixed no-instance: `A` odd or some
`a_i > A/2`. -/
def PartitionBad (a : Fin n → ℕ) : Prop :=
  ¬ 2 ∣ ∑ i, a i ∨ ∃ i, ∑ j, a j < 2 * a i

theorem partition_no_of_bad (a : Fin n → ℕ) (h : PartitionBad a) :
    ¬ ∃ S : Finset (Fin n), 2 * ∑ i ∈ S, a i = ∑ i, a i := by
  rcases h with h | ⟨i, hi⟩
  · exact partition_no_of_odd a h
  · exact partition_no_of_big a hi

/-- No dispatch order of the fixed instance reaches the threshold `k = 2`. -/
theorem noInst_not_ge_two : ¬ ∃ σ : List (Fin 1), IsOrder σ ∧ 2 ≤ W noInst σ := by
  rintro ⟨σ, hσ, h⟩
  have := noInst_isOPT.2 σ hσ
  omega

/-- **Theorem 2 with the preprocessing** (ITEM 4(e)).  For positive `a`:

* if `A` is odd or some `a_i > A/2`, Partition is a no-instance, and the fixed instance `noInst`
  (one site `p = d = w = 1`) satisfies Assumption 1 and has optimum `1 < k = 2`;
* otherwise `p = w = a`, `d = A/2` satisfies Assumption 1 (`a_i ≤ A/2`; `a_i = A/2` allowed)
  and Partition is a yes-instance iff some dispatch order has on-time weight `≥ k = A/2`. -/
theorem partition_reduction_full (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) :
    (PartitionBad a →
      (¬ ∃ S : Finset (Fin n), 2 * ∑ i ∈ S, a i = ∑ i, a i) ∧
      IndivFeasible noInst ∧ IsOPT noInst 1 ∧ ¬ ∃ σ : List (Fin 1), IsOrder σ ∧ 2 ≤ W noInst σ) ∧
    (¬ PartitionBad a →
      IndivFeasible (partitionInst a ha) ∧
      ((∃ S : Finset (Fin n), 2 * ∑ i ∈ S, a i = ∑ i, a i) ↔
        ∃ σ : List (Fin n), IsOrder σ ∧ (∑ i, a i) / 2 ≤ W (partitionInst a ha) σ)) := by
  refine ⟨fun h => ⟨partition_no_of_bad a h, noInst_indivFeasible, noInst_isOPT,
    noInst_not_ge_two⟩, fun h => ?_⟩
  unfold PartitionBad at h
  push_neg at h
  obtain ⟨heven, hbig⟩ := h
  have heven' : 2 ∣ ∑ i, a i := by simpa using heven
  refine ⟨partitionInst_indivFeasible a ha fun i => ?_, partition_reduction a ha heven'⟩
  have := hbig i
  omega

end PartP

end Mwhed
