import Mathlib

/-!
# MWHED: core definitions

Formal model of Minimum Weighted Hazard-Exposure Dispatch (Definition 1 of
`../main.tex`).  Sites are indexed by a finite type `ι`; a *dispatch order* is
a list of sites; the completion time of the `k`-th site served is the sum of
the dispatch times of the first `k` sites; a site is *on time* if its
completion time is at most its deadline.

Everything in this file is a definition or an elementary lemma; the theorems
of the paper live in the other files of the library.
-/

namespace Mwhed

/-- An MWHED instance over the sites `ι`: round-trip dispatch times `p`
(positive), hazard-arrival deadlines `d`, criticality weights `w`. -/
structure Inst (ι : Type*) where
  p : ι → ℕ
  d : ι → ℕ
  w : ι → ℕ
  p_pos : ∀ i, 0 < p i

section Defs

variable {ι : Type*} [DecidableEq ι] (I : Inst ι)

/-- On-time weight of the dispatch list `σ`, when the vehicle starts serving it
at time `t`.  The head is served first and completes at `t + p`. -/
def onTimeW : ℕ → List ι → ℕ
  | _, [] => 0
  | t, i :: σ => (if t + I.p i ≤ I.d i then I.w i else 0) + onTimeW (t + I.p i) σ

/-- `W σ`: the objective `W(σ)` of equation (1) of the paper, for a dispatch
list `σ` that starts at time `0`. -/
def W (σ : List ι) : ℕ := onTimeW I 0 σ

/-- Every site of the list `σ` completes on time when service starts at `t`. -/
def AllOnTime : ℕ → List ι → Prop
  | _, [] => True
  | t, i :: σ => t + I.p i ≤ I.d i ∧ AllOnTime (t + I.p i) σ

/-- Completion time of the site `i` in the list `σ` (service starts at `0`):
the total dispatch time of everything up to and including `i`.  Meaningful for
`i ∈ σ` and `σ.Nodup`. -/
def completion (σ : List ι) (i : ι) : ℕ :=
  ((σ.takeWhile (· ≠ i)).map I.p).sum + I.p i

/-- Total weight of a set of sites. -/
def weight (S : Finset ι) : ℕ := ∑ i ∈ S, I.w i

/-- Total dispatch time of a set of sites. -/
def time (S : Finset ι) : ℕ := ∑ i ∈ S, I.p i

/-- The sites of `σ` that are on time when service starts at `0`. -/
def onTimeSet (σ : List ι) : Finset ι :=
  σ.toFinset.filter (fun i => completion I σ i ≤ I.d i)

/-- A *dispatch order* (Definition 1): a permutation of all the sites. -/
def IsOrder [Fintype ι] (σ : List ι) : Prop := σ.Nodup ∧ ∀ i, i ∈ σ

/-- A set `S` of sites is *feasible* if some dispatch order serves every site
of `S` on time. -/
def Feasible [Fintype ι] (S : Finset ι) : Prop :=
  ∃ σ : List ι, IsOrder σ ∧ ∀ i ∈ S, completion I σ i ≤ I.d i

/-- `v` is the optimal value `W*` of the instance. -/
def IsOPT [Fintype ι] (v : ℕ) : Prop :=
  (∃ σ : List ι, IsOrder σ ∧ W I σ = v) ∧ ∀ σ : List ι, IsOrder σ → W I σ ≤ v

/-- The earliest-deadline-first list of a set `S` (ties in an arbitrary but
fixed way): `S` sorted by non-decreasing deadline. -/
noncomputable def edd (S : Finset ι) : List ι :=
  S.toList.insertionSort (fun a b => I.d a ≤ I.d b)

/-- Assumption 1: every site is individually feasible. -/
def IndivFeasible : Prop := ∀ i, I.p i ≤ I.d i

end Defs

end Mwhed
