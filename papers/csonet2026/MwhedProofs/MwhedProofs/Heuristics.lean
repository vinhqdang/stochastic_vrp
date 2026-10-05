import MwhedProofs.Defs

set_option linter.unusedSectionVars false

/-!
# Simple dispatch rules and their unbounded ratios (Section 4.5 of the paper)

Three rules on a list of sites (for an instance `I` over any index type `ι`):

* **naive EDD** (`naiveEDDValue`): serve every site in earliest-deadline-first
  order; a late site still consumes its dispatch time; the value is the
  on-time weight `W` of that order;
* **EDD with skipping** (`skipKept`, `skipValue`): process the sites in
  deadline order and skip any site that would complete late;
* **weighted greedy repair** (Algorithm 4; `greedyRepairKept`,
  `greedyRepairValue`): process the sites in deadline order (after deleting
  the sites with `p i > d i`); add each site to the kept set and, while the
  total dispatch time exceeds the current deadline, drop the kept site of
  smallest weight-to-time ratio `w/p`.  The `while` loop is the structurally
  recursive `dropLoop` with explicit fuel (the current size of the kept list,
  which is exactly enough: every pass removes one site).  Ratios are compared
  by cross-multiplication in `ℕ` (`w k / p k < w j / p j ↔ w k * p j < w j * p k`);
  ties in the arg-min are broken by taking the first minimal site in the kept
  list (the paper's `argmin` is unspecified on ties).  Sorting by deadline is
  Mathlib's stable `List.insertionSort`; ties among equal deadlines are broken by
  input order (the paper leaves them arbitrary; the instances below have distinct
  deadlines).

Results:

* `prop6_*`      : **Proposition 6** -- `p=(1,2), d=(1,2), w=(1,W)` has `OPT = W`
  while naive EDD and EDD with skipping obtain `1`; hence for every `r ∈ (0,1)`
  both are strictly below `r · W*` (`prop6_unbounded_ratio`).
* `prop7_*`      : **Proposition 7** -- `p=(1,k), d=(1,k), w=(2,2k-1)` has
  greedy repair weight `2` (kept set `[0]`) and `OPT = 2k-1`; the ratio
  `2/(2k-1)` is below any `r > 0` for large `k` (`prop7_unbounded_ratio`).

For the proof of `OPT` on the two-site instances we also provide a small toolkit
for enumerating dispatch orders on `Fin n` (`isOrder_length`, `isOrder_fin2`),
reused in `Examples.lean`.  Everything here depends only on `Defs.lean`.
-/

namespace Mwhed

/-! ### Enumerating dispatch orders on a small index type -/

section Toolkit

variable {ι : Type*} [DecidableEq ι] [Fintype ι]

/-- A dispatch order has exactly one entry per site. -/
theorem isOrder_length {σ : List ι} (h : IsOrder σ) : σ.length = Fintype.card ι := by
  obtain ⟨hnd, hmem⟩ := h
  have : σ.toFinset = Finset.univ := by
    ext i; simp [hmem i]
  rw [← List.toFinset_card_of_nodup hnd, this, Finset.card_univ]

/-- The dispatch orders of two sites are `[0, 1]` and `[1, 0]`. -/
theorem isOrder_fin2 {σ : List (Fin 2)} : IsOrder σ ↔ σ = [0, 1] ∨ σ = [1, 0] := by
  constructor
  · intro h
    have hl := isOrder_length h
    simp only [Fintype.card_fin] at hl
    match σ, hl, h with
    | [a, b], _, h =>
      have key : ∀ a b : Fin 2, IsOrder [a, b] → [a, b] = [0, 1] ∨ [a, b] = [1, 0] := by
        unfold IsOrder; decide
      exact key a b h
  · rintro (rfl | rfl) <;> (unfold IsOrder; decide)

/-- Characterisation of `IsOPT` on two sites. -/
theorem isOPT_fin2 (I : Inst (Fin 2)) (v : ℕ) :
    IsOPT I v ↔ (v = W I [0, 1] ∨ v = W I [1, 0]) ∧ W I [0, 1] ≤ v ∧ W I [1, 0] ≤ v := by
  have h01 : IsOrder ([0, 1] : List (Fin 2)) := isOrder_fin2.2 (Or.inl rfl)
  have h10 : IsOrder ([1, 0] : List (Fin 2)) := isOrder_fin2.2 (Or.inr rfl)
  unfold IsOPT
  constructor
  · rintro ⟨⟨σ, hσ, hv⟩, hle⟩
    refine ⟨?_, hle _ h01, hle _ h10⟩
    rcases isOrder_fin2.1 hσ with rfl | rfl
    · exact Or.inl hv.symm
    · exact Or.inr hv.symm
  · rintro ⟨hv | hv, h1, h2⟩
    · exact ⟨⟨_, h01, hv.symm⟩, fun σ hσ => by
        rcases isOrder_fin2.1 hσ with rfl | rfl <;> omega⟩
    · exact ⟨⟨_, h10, hv.symm⟩, fun σ hσ => by
        rcases isOrder_fin2.1 hσ with rfl | rfl <;> omega⟩

end Toolkit

/-! ### The three rules -/

section Rules

variable {ι : Type*} [DecidableEq ι] (I : Inst ι)

/-- Earliest-deadline-first order of a list of sites (stable insertion sort). -/
def eddSort (L : List ι) : List ι := L.insertionSort (fun a b => I.d a ≤ I.d b)

/-- **Naive EDD**: serve every site in deadline order (late sites still consume
their dispatch time) and count the on-time weight. -/
def naiveEDDValue (L : List ι) : ℕ := W I (eddSort I L)

/-- **EDD with skipping**: the sites kept, when serving `σ` from time `t`
and skipping every site that would complete after its deadline. -/
def skipKept : ℕ → List ι → List ι
  | _, [] => []
  | t, i :: σ => if t + I.p i ≤ I.d i then i :: skipKept (t + I.p i) σ else skipKept t σ

/-- Value (total weight of the sites kept) of EDD with skipping. -/
def skipValue (L : List ι) : ℕ := ((skipKept I 0 (eddSort I L)).map I.w).sum

/-- Sanity check: the kept sites of EDD with skipping are all on time, so
their on-time weight (when served consecutively from time `t`) is their total weight. -/
theorem onTimeW_skipKept (σ : List ι) :
    ∀ t, onTimeW I t (skipKept I t σ) = ((skipKept I t σ).map I.w).sum := by
  induction σ with
  | nil => intro t; simp [skipKept, onTimeW]
  | cons i σ ih =>
    intro t
    by_cases h : t + I.p i ≤ I.d i
    · simp [skipKept, h, onTimeW, ih]
    · simp [skipKept, h, ih]

/-- Index of the kept site of smallest weight-to-time ratio (first minimal one),
starting from the candidate `j`.  `w k / p k < w j / p j` is tested as
`w k * p j < w j * p k`. -/
def argminRatio : ι → List ι → ι
  | j, [] => j
  | j, k :: ks => if I.w k * I.p j < I.w j * I.p k then argminRatio k ks else argminRatio j ks

/-- The `while` loop of Algorithm 4 (lines 4-7), with explicit fuel: while the total
dispatch time of the kept list exceeds `di` and the list is nonempty, drop the kept
site of smallest weight-to-time ratio. -/
def dropLoop (di : ℕ) : ℕ → List ι → List ι
  | 0, kept => kept
  | _ + 1, [] => []
  | f + 1, j :: js =>
    if ((j :: js).map I.p).sum > di then
      dropLoop di f ((j :: js).erase (argminRatio I j js))
    else j :: js

/-- The `for` loop of Algorithm 4: `kept` is the current kept list, the second
argument the remaining sites (in deadline order). -/
def repairKept : List ι → List ι → List ι
  | kept, [] => kept
  | kept, i :: rest =>
    repairKept (dropLoop I (I.d i) (kept ++ [i]).length (kept ++ [i])) rest

/-- **Weighted greedy repair** (Algorithm 4): delete sites with `p i > d i`,
sort by deadline, run the repair loop from the empty kept set. -/
def greedyRepairKept (L : List ι) : List ι :=
  repairKept I [] ((eddSort I L).filter (fun i => I.p i ≤ I.d i))

/-- Value (total weight) of the kept set returned by greedy repair. -/
def greedyRepairValue (L : List ι) : ℕ := ((greedyRepairKept I L).map I.w).sum

/-- The dispatch order realised by greedy repair: the kept sites first, in
increasing-deadline order, followed by the remaining sites. -/
def greedyRepairOrder (L : List ι) : List ι :=
  greedyRepairKept I L ++ (eddSort I L).filter (fun i => i ∉ greedyRepairKept I L)

end Rules

/-! ### Proposition 6 -/

/-- The instance of Proposition 6: `p = (1,2)`, `d = (1,2)`, `w = (1,W)`. -/
def prop6Inst (W : ℕ) : Inst (Fin 2) where
  p := ![1, 2]
  d := ![1, 2]
  w := ![1, W]
  p_pos := by intro i; fin_cases i <;> simp

theorem prop6_indivFeasible (W : ℕ) : IndivFeasible (prop6Inst W) := by
  intro i; fin_cases i <;> simp [prop6Inst]

/-- **Proposition 6**, optimum: `W* = W` for `W ≥ 2` (indeed for `W ≥ 1`). -/
theorem prop6_isOPT {W : ℕ} (hW : 2 ≤ W) : IsOPT (prop6Inst W) W := by
  rw [isOPT_fin2]
  simp [Mwhed.W, onTimeW, prop6Inst]
  omega

/-- **Proposition 6**, naive EDD obtains weight `1` (for either input order of the two sites). -/
theorem prop6_naive (W : ℕ) :
    naiveEDDValue (prop6Inst W) [0, 1] = 1 ∧ naiveEDDValue (prop6Inst W) [1, 0] = 1 := by
  simp [naiveEDDValue, eddSort, Mwhed.W, onTimeW, prop6Inst, List.insertionSort,
    List.orderedInsert]

/-- **Proposition 6**, EDD with skipping obtains weight `1`. -/
theorem prop6_skip (W : ℕ) :
    skipValue (prop6Inst W) [0, 1] = 1 ∧ skipValue (prop6Inst W) [1, 0] = 1 := by
  simp [skipValue, skipKept, eddSort, prop6Inst, List.insertionSort, List.orderedInsert]

/-- **Proposition 6** (full statement): for every `r ∈ (0,1)` there is an MWHED
instance satisfying Assumption 1 on which both naive EDD and EDD with skipping
obtain on-time weight strictly less than `r · W*`. -/
theorem prop6_unbounded_ratio (r : ℝ) (hr0 : 0 < r) (hr1 : r < 1) :
    ∃ I : Inst (Fin 2), IndivFeasible I ∧ ∃ v : ℕ, IsOPT I v ∧
      (naiveEDDValue I [0, 1] : ℝ) < r * v ∧ (skipValue I [0, 1] : ℝ) < r * v := by
  obtain ⟨W, hW⟩ : ∃ W : ℕ, 1 / r < W := exists_nat_gt _
  have h1 : 1 < 1 / r := by rw [lt_div_iff₀ hr0]; linarith
  have hW2 : 2 ≤ W := by
    have : (1 : ℝ) < W := h1.trans hW
    exact_mod_cast this
  have hlt : (1 : ℝ) < r * W := by
    rw [div_lt_iff₀ hr0] at hW; linarith
  refine ⟨prop6Inst W, prop6_indivFeasible W, W, prop6_isOPT hW2, ?_, ?_⟩
  · rw [(prop6_naive W).1]; exact_mod_cast hlt
  · rw [(prop6_skip W).1]; exact_mod_cast hlt

/-! ### Proposition 7 -/

/-- The instance of Proposition 7: `p = (1,k)`, `d = (1,k)`, `w = (2, 2k-1)`. -/
def prop7Inst (k : ℕ) (hk : 1 ≤ k) : Inst (Fin 2) where
  p := ![1, k]
  d := ![1, k]
  w := ![2, 2 * k - 1]
  p_pos := by intro i; fin_cases i <;> simp; try omega

theorem prop7_indivFeasible {k : ℕ} (hk : 1 ≤ k) : IndivFeasible (prop7Inst k hk) := by
  intro i; fin_cases i <;> simp [prop7Inst]

/-- **Proposition 7**, optimum: `W* = 2k-1` for `k ≥ 2`. -/
theorem prop7_isOPT {k : ℕ} (hk : 2 ≤ k) : IsOPT (prop7Inst k (by omega)) (2 * k - 1) := by
  rw [isOPT_fin2]
  have h1 : ¬ (1 + k ≤ k) := by omega
  have h2 : ¬ (k + 1 ≤ 1) := by omega
  simp [Mwhed.W, onTimeW, prop7Inst, h1, h2]
  omega

/-- **Proposition 7**, the kept set of weighted greedy repair is `{site 0}`: site `1`
(ratio `(2k-1)/k < 2`) is the one discarded. -/
theorem prop7_kept {k : ℕ} (hk : 2 ≤ k) :
    greedyRepairKept (prop7Inst k (by omega)) [0, 1] = [0] := by
  have h1 : 1 ≤ k := by omega
  have h2 : ¬ (k ≤ 1) := by omega
  have h3 : ¬ (1 + k ≤ k) := by omega
  have h0 : 0 < k := by omega
  have h5 : ¬ (k < 1) := by omega
  have h6 : k < 1 + k := by omega
  simp [greedyRepairKept, repairKept, dropLoop, argminRatio, eddSort, prop7Inst,
    List.insertionSort, List.orderedInsert, h0, h1, h5, h6]

/-- **Proposition 7**: weighted greedy repair returns on-time weight `2`
(its kept set `{0}` has weight `2`, and the dispatch order it realises has `W = 2`). -/
theorem prop7_repair {k : ℕ} (hk : 2 ≤ k) :
    greedyRepairValue (prop7Inst k (by omega)) [0, 1] = 2 ∧
      W (prop7Inst k (by omega)) (greedyRepairOrder (prop7Inst k (by omega)) [0, 1]) = 2 := by
  have h1 : 1 ≤ k := by omega
  have h2 : ¬ (k ≤ 1) := by omega
  have h3 : ¬ (1 + k ≤ k) := by omega
  have hker := prop7_kept hk
  constructor
  · simp only [greedyRepairValue, hker]; simp [prop7Inst]
  · simp only [greedyRepairOrder, hker]
    simp [eddSort, prop7Inst, Mwhed.W, onTimeW, List.insertionSort,
      List.orderedInsert, h1, h3]

/-- **Proposition 7** (full statement): the ratio `2/(2k-1)` tends to `0`; for every
`r > 0` some `k ≥ 2` has greedy repair strictly below `r · W*`. -/
theorem prop7_unbounded_ratio (r : ℝ) (hr : 0 < r) :
    ∃ (k : ℕ) (hk : 2 ≤ k), IndivFeasible (prop7Inst k (by omega)) ∧
      IsOPT (prop7Inst k (by omega)) (2 * k - 1) ∧
      (greedyRepairValue (prop7Inst k (by omega)) [0, 1] : ℝ) <
        r * ((2 * k - 1 : ℕ) : ℝ) := by
  obtain ⟨m, hm⟩ : ∃ m : ℕ, 2 / r < m := exists_nat_gt _
  refine ⟨m + 2, by omega, prop7_indivFeasible _, prop7_isOPT (by omega), ?_⟩
  rw [(prop7_repair (k := m + 2) (by omega)).1]
  rw [div_lt_iff₀ hr] at hm
  have : ((2 * (m + 2) - 1 : ℕ) : ℝ) = 2 * m + 3 := by
    rw [show 2 * (m + 2) - 1 = 2 * m + 3 by omega]; push_cast; ring
  rw [this]
  push_cast
  nlinarith

end Mwhed
