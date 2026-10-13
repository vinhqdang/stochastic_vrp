# QUASAR M4 — cheapest-feasible move search with estimated feasibility

Status labels: PROVED (complete proof below), CITED (taken from the
literature, see `VERIFY_CITATIONS.md` until checked), CONJECTURE.

## 1. Problem

An ALNS repair step for a removed customer, or any move generator, produces
candidates 1, …, M sorted by deterministic cost c_1 ≤ c_2 ≤ … ≤ c_M (the
travel cost of the move is known exactly). Candidate j has an unknown risk
p_j = P(F_{r_j} > Q) of the moved route r_j. The chance gate accepts j iff
p_j ≤ α. Risks can only be estimated, so we use an ε-relaxed target.

**Definition (ε-cheapest feasible, ε-CF).** Let 0 < ε < α and
k* = min{ j : p_j ≤ α − ε } (the first clearly feasible candidate; k* = ∞ if
none). An output j is *correct* if p_j ≤ α + ε and j ≤ k*. If k* = ∞ the
output "none" is also correct.

So the algorithm never returns a clearly infeasible candidate and never
skips a clearly feasible cheaper one; candidates in the band (α − ε, α + ε]
may be returned or skipped.

**Access model.** A select oracle U : |j⟩|0⟩ ↦ |j⟩ A_j|0⟩ with
A_j|0⟩ = √(1−p_j)|ψ_j0⟩|0⟩ + √p_j|ψ_j1⟩|1⟩ (the M2 oracle with the route
selected by the index register), and U†. Cost unit: one application of U or
U†. Classically: one sample of the overflow indicator of one candidate.
v̄ denotes a known bound with p_j(1−p_j) ≤ v̄ for all j (v̄ ≤ 1/4).

**CRN variant.** Fix a reference candidate ρ (for example the cheapest
position of the same customer, M3 class 3) and the difference
D_j = 1[F_{r_j} > Q] − 1[F_{r_ρ} > Q] on a shared scenario register (M2
difference oracle), with p_j = p_ρ + E[D_j], E[D_j] = a_j⁺ − a_j⁻,
a_j^± = P(D_j = ±1). v̄_D bounds a_j^±(1 − a_j^±).

## 2. Building block: a coherent threshold test

**Lemma 1 (PROVED, standard).** For η ∈ (0, 1/2) there is a unitary B_j
using O(T(ε, v̄) · log(1/η)) applications of A_j and A_j†, with
T(ε, v) = π√v/ε + π/√ε up to a constant, whose output qubit, measured, is
1 with probability q_j satisfying q_j ≥ 1 − η if p_j ≤ α − ε and q_j ≤ η if
p_j > α + ε (no condition in the band).

*Proof.* Canonical amplitude estimation with M ≥ c(√v̄/ε + 1/√ε) Grover
iterates returns p̂ with |p̂ − p_j| < ε with probability ≥ 8/π²
(Brassard–Høyer–Mosca–Tapp, error ≤ 2π√(p(1−p))/M + π²/M²). Run it
r = O(log 1/η) times in parallel registers, compute the median coherently
(classical reversible arithmetic on the outputs) and set the output qubit to
[median ≤ α]. If p_j ≤ α − ε the median lies within ε of p_j except with
probability ≤ η (Chernoff), hence ≤ α; symmetrically for p_j > α + ε. Nothing
is uncomputed: the garbage stays entangled with the output, which is allowed
because B_j is only used as a state-preparation unitary below. ∎

**CRN version.** Estimate p_ρ once (outside superposition) to ±ε/2 with
failure probability δ/4: cost O(T(ε/2, v̄) log(1/δ)). Then B_j^D tests
â_j⁺ − â_j⁻ ≤ α − p̂_ρ with tolerance ε/2, each a^± estimated to ±ε/4:
O(T(ε/4, v̄_D) log(1/η)) applications of the difference oracle. Lemma 1
holds with p_j replaced by p̂_ρ + E[D_j] and ε by ε/2 (the ±ε/2 error of p̂_ρ
is absorbed by the margin).

## 3. Algorithm and upper bound

**Prefix test.** For m ≥ 1 let A'_m prepare (1/√m)Σ_{j≤m}|j⟩ and apply the
select version of B_j. The "good" subspace is output qubit = 1; its weight is
a_m = (1/m)Σ_{j≤m} q_j. TEST(m): run amplitude amplification with the
exponential schedule of Boyer–Brassard–Høyer–Tapp for at most
K_m = c√m iterations, repeated R = O(log 1/δ') times; if some run measures
output 1, return YES with witness w = the measured index, else NO.

**Lemma 2 (PROVED).** Let η ≤ δ'/(c' m R). Then:
(a) if [1..m] contains an index with p_j ≤ α − ε, TEST(m) returns YES with
probability ≥ 1 − δ';
(b) whenever TEST(m) returns YES, the witness satisfies p_w ≤ α + ε except
with probability ≤ δ'.

*Proof.* (a) Such an index has q_j ≥ 1 − η, so a_m ≥ (1−η)/m and the BBHT
schedule with K_m = c√m iterations succeeds with constant probability per
run; R runs give ≥ 1 − δ'. (b) Split the good-subspace weight into the bad
part β = (1/m)Σ_{p_j > α+ε} q_j ≤ η and the rest. After k amplification
iterates the probability of measuring output 1 *and* a bad index is at most
(2k+1)² β (the amplitude on any subspace of the good space grows at most by
a factor (2k+1) — sin((2k+1)θ) ≤ (2k+1) sin θ applied to the bad component,
which evolves in the same two-dimensional rotation as the whole good part).
Summing over R runs of at most K_m iterates: ≤ R (2K_m+1)² η ≤ δ' for the
stated η. ∎

**Algorithm FIRST-FEASIBLE.**
1. Exponential phase: for m = 1, 2, 4, …, M run TEST(m); let hi be the first
   m with YES and w its witness; set lo = hi/2 (lo = 0 if hi = 1). If no YES
   up to M, output "none".
2. Bisection: while hi > lo + 1: mid = ⌊(lo+hi)/2⌋; if TEST(mid) = YES then
   hi = mid and w = its witness, else lo = mid.
3. Output w.

**Theorem 1 (PROVED).** With δ' = δ/(2⌈log₂ M⌉ + 2) and η as in Lemma 2,
FIRST-FEASIBLE outputs a correct ε-CF answer with probability ≥ 1 − δ, using

  O( √k* · log k* · log(log k*/δ) · log(k*/δ) · (√v̄/ε + 1/√ε) )

applications of U (k* < ∞; with k* = ∞ replace √k* log k* by √M log M). With
CRN the factor (√v̄/ε + 1/√ε) becomes (√v̄_D/ε + 1/√ε), plus one stand-alone
estimate of p_ρ, O((√v̄/ε + 1/√ε) log 1/δ).

*Proof.* Correctness: all TEST calls behave as in Lemma 2 except with
probability ≤ (2⌈log₂ M⌉+2)δ' = δ (union bound). Then (i) the exponential
phase stops at hi ≤ 2k* (TEST(m) is YES for every m ≥ k*); (ii) every NO at
m implies [1..m] has no clearly feasible index, i.e. k* > m; (iii) every YES
witness has p_w ≤ α + ε and w ≤ m. Invariant: lo = 0 or TEST(lo) = NO (so
k* > lo), and w is a YES witness of [1..hi]. At termination hi = lo + 1, so
w ≤ hi = lo + 1 ≤ k*, and p_w ≤ α + ε. Cost: at most ⌈log₂ 2k*⌉ tests of
size ≤ 2k* in phase 1 and ⌈log₂ k*⌉ in phase 2, each costing
R · K_m · O(T log 1/η) = O(√k* · log(1/δ') · T · log(k*/δ)). ∎

*Remarks.* The logarithmic factors are not optimised: known techniques for
searching with bounded-error subroutines (Høyer–Mosca–de Wolf) and for
finding the first marked element in O(√k) (Kothari; Dürr–Heiligman–Høyer–
Mhalla) remove some of them for well-defined predicates; the band
(α−ε, α+ε], where our test is not a function, is why the proof above uses
witnesses rather than citing those results directly (CITED items to check).
With variable per-candidate costs T_j (heterogeneous v_j), variable-time
search (Ambainis) suggests Õ(√(Σ_{j≤k*} T_j²)) (CONJECTURE for this
ambiguous-predicate setting).

## 4. Classical counterpart and lower bounds

**Proposition 2 (PROVED).** Sequential scan — test j = 1, 2, … with
N = O(v̄ log(k*/δ)/ε² + log(k*/δ)/ε) samples each and return the first
accepted — is correct with probability ≥ 1 − δ and costs
O(k* (v̄/ε² + 1/ε) log(k*/δ)) samples; with CRN v̄ becomes Var(D) (direct
signed mean) plus one estimate of p_ρ. *Proof:* Bernstein's inequality per
test and a union bound over the at most k* tests. ∎

**Proposition 3 (lower bounds).**
(a) Classical, CITED: on the family where exactly one index among the first
k has p = α − ε and the others p = α + 2ε, any algorithm that is correct
with probability ≥ 2/3 needs Ω(k α(1−α)/ε²) samples in the worst case (the
best-arm / good-arm identification lower bounds; Mannor–Tsitsiklis type
change-of-measure argument).
(b) Quantum, PROVED from cited bounds: Ω(√k) (unstructured search, by taking
ε fixed) and Ω(√(α(1−α))/ε) (approximate counting lower bound, k = 1), hence
Ω(√k + √(α(1−α))/ε).
(c) CONJECTURE: Ω(√k · √(α(1−α))/ε), which would make Theorem 1 optimal up
to logarithms; it would follow from a composition theorem for the adversary
bound applied to OR_k composed with the promise problem "p ≤ α−ε vs
p ≥ α+2ε" if such a theorem holds for partial inner functions (to check).

**Corollary 4 (query advantage).** On the hard family the ratio of quantum
to classical query counts is Õ(ε/√(k α(1−α))) = Õ(1/√N_classical): the
quadratic speedup survives the search over candidates, and CRN multiplies
the per-test cost by √(v̄_D/v̄) (quantum) and v̄_D/v̄ (classical).

## 5. Experiment M4b (planned)

On real repair steps (route r, removed customer j, all insertion positions
sorted by insertion cost, ρ = 0.6 copula, α = 0.10): measure the distribution
of k*_ε, the share of steps with k* = 1 (the cheapest position is already
feasible: no search is needed and the speedup is nil), v and Var(D) relative
to the cheapest position, and the predicted query counts of Theorem 1 and
Proposition 2 with and without CRN, with the log factors stated explicitly.
