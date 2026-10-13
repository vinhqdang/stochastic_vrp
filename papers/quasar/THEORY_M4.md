# QUASAR M4 — cheapest-feasible move search with estimated feasibility

Status labels: PROVED (complete proof below), CITED (taken from the
literature, see `VERIFY_CITATIONS.md`), CONJECTURE.
Revision 2 (2026-10-13) after the review of revision 1: union bound with
two failure modes per test, the CRN failure term, a non-adaptive
amplification run with an explicit attempt count, corrected log factors,
direct proofs of both lower bounds.

## 1. Problem

An ALNS repair step for a removed customer, or any move generator, produces
candidates 1, …, M sorted by deterministic cost c_1 ≤ c_2 ≤ … ≤ c_M (the
travel cost of the move is known exactly). Candidate j has an unknown risk
p_j = P(F_{r_j} > Q) of the moved route r_j. The chance gate accepts j iff
p_j ≤ α. Risks can only be estimated, so we use an ε-relaxed target.

**Definition (ε-cheapest feasible, ε-CF).** Let 0 < ε ≤ α/2 and
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
v̄ denotes a known bound with p_j(1−p_j) ≤ v̄ for every j with p_j ≤ 2α; it
must be uniform in j because the test below is applied in superposition.
Candidates with p_j > 2α need no variance bound (Lemma 1), so a single
candidate near p = 1/2 does not force v̄ = 1/4.

**CRN variant.** Fix a reference candidate ρ (for example the cheapest
position of the same customer, M3 class 3) and the difference
D_j = 1[F_{r_j} > Q] − 1[F_{r_ρ} > Q] on a shared scenario register (M2
difference oracle), with p_j = p_ρ + E[D_j], E[D_j] = a_j⁺ − a_j⁻,
a_j^± = P(D_j = ±1). Quantum side: v̄_D bounds a_j^±(1 − a_j^±) (two
amplitudes are estimated). Classical side: σ̄_D² bounds
Var(D_j) = a_j⁺ + a_j⁻ − (a_j⁺ − a_j⁻)² (one signed mean is estimated).

## 2. Building block: a coherent threshold test

**Lemma 1 (PROVED, standard).** For η ∈ (0, 1/2) there is a unitary B_j
using O(T(ε, v̄) · log(1/η)) applications of A_j and A_j†, with
T(ε, v) = √v/ε + 1/√ε + 1/√α up to a constant, whose output qubit, measured,
is 1 with probability q_j satisfying q_j ≥ 1 − η if p_j ≤ α − ε and q_j ≤ η
if p_j > α + ε (no condition in the band).

*Proof.* Canonical amplitude estimation with M Grover iterates returns p̂
with |p̂ − p_j| ≤ 2π√(p_j(1−p_j))/M + π²/M² with probability ≥ 8/π²
(Brassard–Høyer–Mosca–Tapp). Take M ≥ c(√v̄/ε + 1/√ε + 1/√α).
If p_j ≤ 2α the error is < ε for c large enough. If p_j > 2α then
p_j − α > p_j/2 and the error is ≤ 2π√p_j/M + π²/M² < p_j/2 because
M ≥ c/√α ≥ c/√(p_j/2); so p̂ > α. Run it r = O(log 1/η) times in parallel
registers, compute the median coherently (reversible arithmetic on the
outputs) and set the output qubit to [median ≤ α]. Since 8/π² > 1/2, the
median errs only if at least half the runs err, probability ≤ η (Hoeffding).
If p_j ≤ α − ε a correct median is < α; if p_j > α + ε it is > α. Nothing is
uncomputed: the garbage stays entangled with the output qubit, whose
marginal is still q_j (deferred measurement), and B_j is used below only as
a state preparation with a flag qubit, which amplitude amplification
allows. ∎

**CRN version.** Estimate p_ρ once (outside superposition) to ±ε/2 with
failure probability δ/4: cost O(T(ε/2, v̄) log(1/δ)). Conditional on
|p̂_ρ − p_ρ| < ε/2, the test B_j^D outputs 1 iff p̂_ρ + â_j⁺ − â_j⁻ ≤ α, with
each â_j^± within ε/4 (median of O(log 1/η) amplitude estimates, each on the
difference oracle, O(T(ε/4, v̄_D)) applications). The total error is below
ε/2 + ε/4 + ε/4 = ε, so q_j ≥ 1 − η if p_j ≤ α − ε and q_j ≤ η if
p_j > α + ε: Lemma 1 holds with the same ε.

## 3. Algorithm and upper bound

**Prefix test.** For m ≥ 1 let A'_m prepare (1/√m)Σ_{j≤m}|j⟩ and apply the
select version of B_j. The "good" subspace is output qubit = 1; its weight is
a_m = (1/m)Σ_{j≤m} q_j = sin²θ. Let K_m = ⌈√(2m)⌉. One *run* is two
attempts: measure after 0 amplification iterates, then after k iterates with
k uniform in {0, …, K_m − 1}. TEST(m) makes R = ⌈ln(1/δ')/ln(4/3)⌉ runs; if
some attempt measures output 1, return YES with witness w = the measured
index, else NO. A run applies B or B† at most 1 + (2K_m − 1) = 2K_m times.

**Lemma 2 (PROVED).** Let η ≤ δ'/(R(1 + (2K_m − 1)²)). Then:
(a) if [1..m] contains an index with p_j ≤ α − ε, TEST(m) returns YES with
probability ≥ 1 − δ';
(b) the probability that TEST(m) returns YES with a witness having
p_w > α + ε is at most δ'.

*Proof.* (a) Such an index has q_j ≥ 1 − η ≥ 1/2, so a_m ≥ 1/(2m). If
a_m ≥ 3/4 the first attempt succeeds with probability ≥ 3/4. Otherwise
sin 2θ = 2√(a_m(1 − a_m)) ≥ √a_m ≥ 1/√(2m), so K_m ≥ 1/sin 2θ and, by
Lemma 2 of Boyer–Brassard–Høyer–Tapp, the second attempt succeeds with
probability ≥ 1/4. Each run therefore fails with probability ≤ 3/4, and R
runs all fail with probability ≤ δ'.
(b) Let |G⟩ and |B⟩ be the normalised projections of the initial state on
output = 1 and output = 0. The Grover iterate preserves span{|G⟩, |B⟩}, so
after k iterates the state is sin((2k+1)θ)|G⟩ + cos((2k+1)θ)|B⟩. Only |G⟩
carries output 1, and the weight of bad indices (p_j > α + ε) in |G⟩ is
β/a_m with β = (1/m)Σ_{bad} q_j ≤ η. Hence
P(output 1 and bad index) = sin²((2k+1)θ) β/a_m ≤ (2k+1)² sin²θ β/a_m
= (2k+1)² β, using |sin nx| ≤ n|sin x|. Over one run (k = 0 and
k ≤ K_m − 1) this is ≤ (1 + (2K_m − 1)²) η, and over R runs ≤ δ'. ∎

**Algorithm FIRST-FEASIBLE.**
1. Exponential phase: for m = 1, 2, 4, …, 2^{⌈log₂ M⌉−1} and then m = M,
   run TEST(m); let hi be the first m with YES, w its witness, and lo the
   previously tested size (lo = 0 if hi = 1). If no YES up to M, output
   "none".
2. Bisection: while hi > lo + 1: mid = ⌊(lo+hi)/2⌋; if TEST(mid) = YES then
   hi = mid and w = its witness, else lo = mid.
3. Output w.

**Theorem 1 (PROVED).** Let δ' = δ/(4⌈log₂ M⌉ + 4) and η as in Lemma 2.
FIRST-FEASIBLE outputs a correct ε-CF answer with probability ≥ 1 − δ. With
L = log(log M/δ) and T = √v̄/ε + 1/√ε + 1/√α it uses

  O( √k* · log k* · L · log(k* L/δ) · T )

applications of U if k* < ∞, and O(√M · L · log(M L/δ) · T) if k* = ∞.
With CRN, T becomes √v̄_D/ε + 1/√ε + 1/√α, δ' becomes
(3/4)δ/(4⌈log₂ M⌉ + 4), and one stand-alone estimate of p_ρ is added,
O((√v̄/ε + 1/√ε) log 1/δ).

*Proof.* Correctness. At most 2⌈log₂ M⌉ + 1 tests are run (⌈log₂ M⌉ + 1 in
phase 1, at most ⌈log₂ M⌉ in phase 2). A test misbehaves if it returns NO
although its prefix contains an index with p_j ≤ α − ε, or returns YES with
a bad witness; each has probability ≤ δ' (Lemma 2), so all tests behave
except with probability ≤ 2(2⌈log₂ M⌉ + 1)δ' ≤ δ (with CRN, plus δ/4 for
p̂_ρ). Then (i) every NO at m implies k* > m; (ii) every YES witness has
p_w ≤ α + ε and w ≤ m; (iii) TEST(m) is YES for every m ≥ k*, so phase 1
stops at hi ≤ min(2k*, M) (if 2^t < M ≤ 2^{t+1} the last size M is
< 2^{t+1} ≤ 2k*). Invariant: lo = 0 or TEST(lo) = NO (so
k* > lo), and w is a YES witness of [1..hi]. At termination hi = lo + 1, so
w ≤ hi = lo + 1 ≤ k*, and p_w ≤ α + ε. If k* = ∞ a YES can only return a
band witness, which is correct, and "none" is correct as well.
Cost. A test of size m costs R · 2K_m · O(T log 1/η) = O(√m · L ·
log(mL/δ) · T), since R = O(log 1/δ') = O(L) and log(1/η) = O(log(mR/δ')).
Phase 1 sums √m over a geometric sequence up to hi₀ ≤ 2k*, which is
O(√k*) times one test's log factors; phase 2 runs ⌈log₂ hi₀⌉ tests of size
≤ hi₀, which gives the factor √k* log k*. If k* = ∞ only phase 1 runs and
the geometric sum gives O(√M). ∎

*Remarks.* (1) The factor log k* comes only from the bisection; it is not
optimal. Searching for the first marked element costs O(√p) expected
queries for a well-defined predicate (stated as a remark in Kothari 2014,
§2.2), and bounded-error search loses no log factor for total Boolean
predicates (Høyer–Mosca–de Wolf 2003). Neither covers our predicate, which
is undefined in the band (α−ε, α+ε]; this is why the proof above uses
witnesses. (2) An adaptive failure budget (δ'_i ∝ δ/(i+1)² for the i-th
test) replaces log M by log k* inside L. (3) With heterogeneous per-candidate
costs T_j (variance v_j), variable-time search (Ambainis 2010) suggests
Õ(√(Σ_{j≤k*} T_j²)) (CONJECTURE in this ambiguous-predicate setting; the
cited theorem assumes ideal subroutines). (4) T ≈ √v̄/ε whenever v̄ ≥ ε and
ε ≤ α.

## 4. Classical counterpart and lower bounds

**Proposition 2 (PROVED).** Sequential scan: for j = 1, 2, …, draw
N = ⌈(2v̄ + 2ε/3) ln(2k̄/δ)/ε²⌉ samples of candidate j, accept j iff p̂_j ≤ α,
and return the first accepted index (or "none" after M). Here k̄ is any
known upper bound on the number of tests (k̄ = M in general). The scan is
correct with probability ≥ 1 − δ and costs at most min(k*, M) · N samples,
that is O(k* (v̄/ε² + 1/ε) log(M/δ)), or O(M (v̄/ε² + 1/ε) log(M/δ)) if
k* = ∞. With CRN: estimate p_ρ to ε/2 once and E[D_j] to ε/2 per candidate,
with v̄ replaced by σ̄_D² (D ∈ {−1, 0, 1} has range 2, so Bernstein applies
with the range constant doubled).
*Proof.* Let E_j be the event |p̂_j − p_j| < ε for the test of j, if run;
by Bernstein, P(not E_j) ≤ δ/k̄. The scan runs test j only if all earlier
tests rejected, and under E_{k*} it accepts k*, so only tests j ≤ k* run.
Under the E_j, an accepted j has p_j < α + ε, and a rejection of any j < k*
is allowed by the definition. A union bound over the at most k̄ tests
finishes the proof. For candidates with p_j > 2α the Bernstein bound with
v̄ is not needed: the test rejects them as soon as N ≥ c/α, as in
Lemma 1. ∎

**Proposition 3 (lower bounds).** Let ε ≤ α/2 and α ≤ 1/4.
(a) Classical (PROVED, change of measure). On the family {P_i : i ≤ k}
where p_i = α − ε and p_j = α + 2ε for j ≠ i (j ≤ k), any algorithm that is
correct with probability ≥ 2/3 on every P_i has
max_i E_i[N] ≥ c k α(1 − α)/ε² for k ≥ 16, with an absolute constant c > 0.
(b) Quantum (PROVED). Any quantum algorithm correct with probability ≥ 2/3
uses Ω(√k + √α/ε) applications of U, U†.
(c) CONJECTURE: Ω(√k · √(α(1−α))/ε), which would make Theorem 1 optimal up
to logarithms. It would follow from a composition theorem for the
adversary bound applied to OR_k composed with the promise problem
"p ≤ α − ε vs p ≥ α + 2ε". Composition theorems for the adversary bound
are stated for Boolean functions (Reichardt 2011; Lee–Mittal–Reichardt–
Špalek–Szegedy 2011), not for promise inner problems of this kind, so we do
not claim it.

*Proof of (a).* On P_i the only correct output is i: every other index has
p_j = α + 2ε > α + ε, "none" is wrong because k* = i, and j > k* is wrong.
Let N* = max_i E_i[N]. Truncate the algorithm at T = 6N* samples (output
anything on truncation); by Markov it is still correct with probability
≥ 2/3 − 1/6 = 1/2 on every P_i, and it stops after ≤ T samples on every
instance. Let P₀ have all k arms at α + 2ε (not in the family). Since
Σ_i P₀(out = i) ≤ 1, fewer than k/2 indices have P₀(out = i) > 2/k; since
Σ_i E₀[N_i] ≤ T, fewer than k/4 indices have E₀[N_i] > 4T/k. Take an i with
neither. P₀ and P_i differ only in arm i, so the transportation lemma
(Kaufmann–Cappé–Garivier 2016, Lemma 1) gives
E₀[N_i] · KL(α + 2ε ‖ α − ε) ≥ kl(P₀(out = i), P_i(out = i))
≥ kl(2/k, 1/2) ≥ 0.3 for k ≥ 16. With q = α − ε ∈ [α/2, α], the
χ²-bound gives KL(α + 2ε ‖ q) ≤ (3ε)²/(q(1 − q)) ≤ 18ε²/(α(1 − α)). Hence
4T/k ≥ 0.3 α(1−α)/(18ε²), i.e. N* = T/6 ≥ k α(1−α)/(1440 ε²). ∎
The same bound holds with the arms placed among the first k of any longer
list, so it applies with k = k*.

*Proof of (b).* Ω(√k): take p_j ∈ {0, 1} with at most one index at 0
(feasible) and all others at 1; this is search for a unique marked element,
one application of U is simulated by one query to the marked-set oracle,
and the BBBV lower bound applies (for every ε). Ω(√α/ε): take k = 1 and the
two instances p₁ = α − ε (only "1" is correct) and p₁ = α + 2ε (only "none"
is correct). Realise A_θ = exp(−iθY) on one qubit with sin²θ = p. Then
‖A_θ − A_θ'‖ ≤ |θ − θ'|, also for A†, controlled versions and tensor
products with the identity, so after T applications the two final states
differ in norm by ≤ T|θ − θ'|. Success ≥ 2/3 on both requires a trace
distance ≥ 1/3, so T ≥ 1/(3|θ − θ'|). Since dθ/dp = 1/(2√(p(1 − p))) and
p ∈ [α/2, 2α] ⊂ (0, 1/2], |θ − θ'| ≤ 3ε/(2√((α/2)(1/2))) = 3ε/√α. Hence
T ≥ √α/(9ε). (The counting bound of Nayak–Wu is for a membership oracle; the
direct argument above avoids the reduction.) ∎

**Corollary 4 (query comparison).** For ε ≤ α/2, α ≤ 1/4 and α(1−α) ≥ ε, the
ratio of the Theorem 1 upper bound (with v̄ = O(α)) to the Proposition 3(a)
lower bound is at most Õ(ε/√(k* α(1−α))) = Õ(1/√N_cl), where
N_cl = k* α(1−α)/ε² is the classical lower bound; the log factors on the
two sides differ. This is consistent with a quadratic advantage; that the
quantum bound is optimal is Proposition 3(c). With CRN the quantum per-test
cost scales with √v̄_D and the classical one with σ̄_D².

## 5. Experiment M4b

On real repair steps (route plans of the BATON cache, removed customer c,
all insertion positions in all routes sorted by insertion cost, ρ = 0.6
copula, α = 0.10, 2·10⁵ common scenarios): measure the distribution of
k*_ε, the share of steps with k* = 1 (the cheapest position is already
feasible: no search is needed and the speedup is nil), the share with
k* = ∞, the band sizes, v̄ and the CRN quantities relative to the cheapest
candidate, and the predicted query counts of Theorem 1 and Proposition 2
with and without CRN, both with the explicit constants of Lemma 2 and in
the core form (√k* T against k* v̄/ε²). Code: `quantum/search.py`,
`scripts/run_quasar_m4b.py`.
