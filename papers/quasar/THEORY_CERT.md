# QUASAR — simultaneous risk certification of a routing plan

Status labels as in THEORY_M4.md: PROVED, CITED (see VERIFY_CITATIONS.md),
CONJECTURE. Draft 1, 2026-10-13. Context: REFRAME_M4.md.

## 1. Problem

A plan (produced by any classical solver) has d routes. Route j fails on a
day ξ if its peak load exceeds capacity: X_j(ξ) = 1[F_{r_j}(ξ) > Q], with
risk p_j = E X_j. The days are random through the demand model (gamma
marginals, one-factor Gaussian copula per demand type), so the X_j are
correlated through the shared factors.

**Definition (ε-certification).** Given a target α, a margin 0 < ε < α and a
failure level δ, output a set A ⊆ {1..d} of certified routes such that, with
probability at least 1 − δ, simultaneously for all j:
(soundness) j ∈ A ⇒ p_j ≤ α + ε/2, and (completeness) p_j ≤ α − ε ⇒ j ∈ A.

Routes far from the target are easy: write g_j = max(ε, |p_j − α|) for the
gap of route j.

**Locality (the structural fact used throughout).** X_j depends only on the
demands of the L_j customers of route j and on the copula factors. The M2
oracle A_j for route j therefore acts on 2bL_j demand qubits plus the
factor register. The M2 oracle has O(L_j log² L_j) Toffolis
(DESIGN_M2.md) for independent demands; with the copula it needs a factor
register (the bin-averaged grid of M3) and factor-controlled state
preparation of the L_j demand registers, whose cost grows with the grid
size, and the grid adds a discretisation bias that must be charged to ε
(M3 budget ≈ 3·10⁻⁴). A classical sample of X_j costs Θ(L_j) operations.

**Where the claim lives.** In the BATON demand model P(F_r > Q) has an exact
dynamic program (plus quadrature over the factor), which beats any sampling
method; there the experiments are a validated testbed. The claim of this
note is for a route-level stochastic simulator without such structure (a
local digital twin: demand, dwell, travel and breakdown processes), where
the only access is sampling one route-day. Two cost models follow:

- **Local model (L).** One application of A_j or A_j† costs L_j units; one
  classical sample of route j costs L_j units; a classical sample of the
  whole plan costs Σ_j L_j = n units and yields all d indicators.
- **Black-box model (S).** Only a whole-day simulator is available; one call
  (classical sample, or one application of a state preparation producing
  all d indicators coherently) costs one unit.

The routing problem has the local structure, so model L is the relevant one;
model S is the setting of the multivariate estimators (Huggins et al. 2022;
Tang 2025) and is discussed in §5.

## 2. Building block: a consistent interval from canonical amplitude estimation

**Fact 1 (CITED, Brassard–Høyer–Mosca–Tapp, Thm 12).** For integers M ≥ 1
and k ≥ 1, canonical amplitude estimation with M applications of the
Grover iterate (2M + 1 applications of A_j or A_j† in total) outputs p̂ with

  |p̂ − p| ≤ h_{M,k}(p) := 2πk √(p(1−p))/M + π²k²/M²

with probability at least s_k, where s_1 = 8/π² and s_k = 1 − 1/(2(k−1)) for
k ≥ 2.

**Lemma 1 (PROVED).** Let p̂ be the median of r independent runs, each
satisfying Fact 1 with success probability ≥ s_k > 1/2. Then
|p̂ − p| ≤ h_{M,k}(p) except with probability at most
β(r, k) := P(Bin(r, 1 − s_k) ≥ ⌈r/2⌉).
*Proof.* If more than half of the runs are within h of p, so is their
median. ∎

**Lemma 2 (PROVED).** Given p̂, let I = {q ∈ [0,1] : |p̂ − q| ≤ h_{M,k}(q)}.
(i) I is an interval. (ii) If |p̂ − p| ≤ h_{M,k}(p) then p ∈ I. (iii) In that
case, if 0 < g ≤ |p − α| and 2 h_{M,k}(v̄_g) < g, with
v̄_g := max{q(1−q) : q ∈ [0,1], |q − p| ≤ g} (here and below h_{M,k} is
written as a function of the variance v = q(1−q), so its largest value is
h_{M,k}(1/4), attained at q = 1/2),
then I lies strictly on the same side of α as p.
*Proof.* (i) f(q) = h_{M,k}(q) − |p̂ − q| is concave on [0,1] (√(q(1−q)) is
concave and −|·| is concave), so I = {f ≥ 0} is convex. (ii) is the
hypothesis. (iii) Suppose I contains a point on the other side of α. By (i)
it contains every point between p and that point, in particular q₀ with
|q₀ − p| = g. Then g = |q₀ − p| ≤ |q₀ − p̂| + |p̂ − p| ≤ h(q₀) + h(p)
≤ 2 h(v̄_g) < g, a contradiction. ∎

## 3. Algorithm Q-CERT and its cost

For each route j independently (in the local model the routes use disjoint
oracles; the factor register is re-prepared by each A_j):

```
for t = 0, 1, 2, ...:
    M_t = 2^t;  η_t = 6δ / (π² d (t+1)²);  choose (k_t, r_t) with β(r_t, k_t) ≤ η_t
    p̂ = median of r_t canonical AE runs with M_t Grover applications each
    I = consistent interval of p̂ for h_{M_t, k_t}
    if sup I ≤ α:  certify j, stop
    if inf I > α:  reject j, stop
    if h_{M_t,k_t}(v_I) ≤ ε/2, v_I = max_{q ∈ I} q(1−q):
                     certify j iff p̂ ≤ α, stop                 (adaptive precision floor)
```

In Fact 1 the parameter k is a parameter of the analysis, not of the
circuit: one run with M Grover applications satisfies the k-bound with
probability s_k for every k simultaneously. A larger k needs fewer runs r
for the same median failure β(r, k) but gives a wider interval. The value k is
fixed before the run (it must not depend on the outcomes); r_t is then the
smallest r with β(r, k) ≤ η_t. The experiments report the best fixed
k ∈ {1, …, 20} per instance set. A run costs 2M_t + 1 applications of A_j
or A_j† whatever k is.

**Theorem 1 (PROVED).** Q-CERT is an ε-certification with probability at
least 1 − δ. Route j stops no later than the first round t with
2 h_{M_t,k_t}(v̄_{g'_j}) < g'_j, g'_j = max(|p_j − α|, ε/2), and its cost in
applications of A_j, A_j† is Σ_{s ≤ t} r_s (2 M_s + 1). Consequently the
cost of route j is O(log(d log(1/g_j)/δ) · (√v̄_j/g_j + 1/√g_j)), and in the
local model the total cost is Σ_j L_j C_j with C_j the cost of route j.

*Proof.* Failure events: by Lemma 1 the round-t estimate of route j is good
except with probability η_t; Σ_j Σ_t η_t = (6δ/π²) Σ_t (t+1)^{-2} = δ, so
all estimates of all routes are good with probability ≥ 1 − δ. On that
event: by Lemma 2(ii) p_j ∈ I at every round, so "sup I ≤ α" implies
p_j ≤ α (sound) and "inf I > α" implies p_j > α ≥ α − ε (allowed). At the
precision floor p_j ∈ I, so |p̂ − p_j| ≤ h(p_j(1−p_j)) ≤ h(v_I) ≤ ε/2: a
certification has p_j ≤ p̂ + ε/2 ≤ α + ε/2, and a route with p_j ≤ α − ε has
p̂ ≤ α − ε/2 < α and is certified. Stopping: with γ_j = |p_j − α| > 0,
Lemma 2(iii) (g = γ_j) shows the interval has left α no later than the
first round with 2h(v̄_{γ_j}) < γ_j; and the same convexity argument with
g = ε/2 shows I ⊆ [p_j − ε/2, p_j + ε/2], hence v_I ≤ v̄_{ε/2} and the
floor fires, no later than the first round with 2h(v̄_{ε/2}) < ε/2. So route
j stops no later than the first round with 2h(v̄_{g'_j}) < g'_j,
g'_j = max(γ_j, ε/2). The O(·) form follows from h_{M,k} ≈ 2πk√v/M and the
doubling of M_t (the last round costs at most twice the minimal one, the
geometric sum at most twice the last). ∎

**Classical counterpart (C-CERT).** The same doubling scheme with n_t
samples of route j per round and the empirical-Bernstein interval of
Audibert, Munos & Szepesvári (2009) (constants √(2v ln(3/η)/n) +
3 ln(3/η)/n) or the exact Clopper–Pearson interval. With the Bernstein form
the stopping condition is 2(√(2 v̄ ln(3/η)/n) + 3 ln(3/η)/n) < g, so the
cost of route j is O(log(d/δ) (v̄_j/g_j² + 1/g_j)) samples, each costing L_j
in model L. In the experiments the classical side is given the exact
Clopper–Pearson stopping sample size with no doubling loss (favouring
classical).

**Shared-sample classical (C-SHARED).** One whole-plan sample (cost n in
model L) yields all d indicators; the number of samples is set by the
hardest route, max_j n*_j. It is better than C-CERT only if the routes have
similar gaps.

## 4. Lower bounds (per route)

**Proposition 2 (PROVED, from THEORY_M4 Prop. 3).** Deciding for a single
route whether p ≤ α − ε or p ≥ α + 2ε (the k = 1 case of THEORY_M4 Prop. 3)
needs Ω(α(1−α)/ε²) classical samples and Ω(√α/ε) applications of A, A†
(α ≤ 1/4, ε ≤ α/2). Hence Q-CERT and C-CERT are optimal per route up to the
log factor, and the per-route query ratio quantum/classical is
Θ̃(g_j/√v̄_j) for routes with v̄_j ≥ g_j.

Whether the d-route problem in model L needs the SUM of the per-route lower
bounds (a direct-sum statement for promise problems with separate oracles)
is left as a CONJECTURE (classically it holds by an independence argument;
quantumly a direct-sum theorem for the adversary bound of partial functions
would be needed).

## 5. Model S (black box) for comparison

In model S, per-route estimation costs d times more than in model L. The
simultaneous estimators of Huggins et al. (2022) (cost 2√d log(d/δ)/ε, ℓ∞,
variance-agnostic) and Tang (2025, preprint; O(√(Σ_j v_j) log(d/δ)/ε),
constants not given) apply, against O(v log(d/δ)/ε²) shared classical
samples; the measured ratios are in REFRAME_M4.md. Model S is reported but
is not the claim of the paper.

## 6. What is new, after review

Not new (do not claim): adaptive amplitude estimation with confidence
intervals and explicit constants (Grinko et al. 2021, Suzuki et al. 2020,
Aaronson–Rall 2020, Nakaji 2020, Montanaro 2015, Kothari–O'Donnell 2023);
the concavity of the inversion interval. Q-CERT is a careful instance of
these, used as the measuring instrument.

Candidate contributions: (1) the certification formulation of a whole plan
with one-sided thresholds, where most routes are far from the target; (2)
the locality reduction (route-level oracles) that removes the √d or √tr Σ
of the black-box simultaneous estimators, with the factor-register cost
and bias accounted; (3) the measured crossover on 148 region instances,
algorithm against algorithm and ideal against ideal, with the classical
rivals that apply to a route-level simulator (sequential tests, conditional
Monte Carlo over the factor, importance sampling) — and the explicit
statement that with an exact model (DP) classical wins outright.

Per-route lower bounds (Prop. 2) are minimax bounds at a gap level in the
black-box sampling model; the k = 1 case of THEORY_M4 Prop. 3(a) needs its
own two-point argument (Prop. 3(a) is proved for k ≥ 16).
