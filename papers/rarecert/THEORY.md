# RARECERT — theory (R1, draft 1, 2026-10-14)

Labels: PROVED (proof below), PROVED-SKETCH (argument complete in outline,
details to write), CONJECTURE, CITED.

## 1. Model

Route r = (1, …, L). Inputs: independent standard normals W_d, W_p and
E_{d,i}, E_{p,i} (i = 1..L); u = (W_d, E_d, W_p, E_p) ∈ ℝ^{2L+2}. Latent
normals Z_{d,i} = √ρ W_d + √(1−ρ) E_{d,i}, Z_{p,i} likewise with W_p.
Demands d_i = G_{d,i}⁻¹(Φ(Z_{d,i})), p_i = G_{p,i}⁻¹(Φ(Z_{p,i})), with
G a gamma law of shape κ_s = 1/cv² and scale θ = mean/κ_s. Loads
L_k = D_k + P_k, D_k = Σ_{i>k} d_i, P_k = Σ_{i≤k} p_i (k = 0..L); peak
F = max_k L_k; overflow A_c = {F > c}.

Block matrices. For an index set S, let C_S = ρ 11ᵀ + (1−ρ)I (|S|×|S|)
and Θ_S = diag(θ_i, i ∈ S). Define the block scale
  λ(S) = λ_max(Θ_S^{1/2} C_S Θ_S^{1/2}),
the largest root of the secular equation 1 = ρ Σ_{i∈S} θ_i/(λ − (1−ρ)θ_i)
(λ > (1−ρ) max θ_i). Stage scales: λ_k^d = λ({i > k}; delivery θ),
λ_k^p = λ({i ≤ k}; pickup θ), λ_k = max(λ_k^d, λ_k^p), and
λ* = max_k λ_k = max(λ_0^d, λ_L^p) (λ^d decreases and λ^p increases in k,
by interlacing of principal submatrices).

## 2. Structure

**Proposition 1 (PROVED).** max(D_0, P_L) ≤ F ≤ D_0 + P_L, with D_0 =
L_0 (all deliveries on board) and P_L = L_L (all pickups on board).
*Proof.* L_0 = D_0 and L_L = P_L are among the stage loads; D_k ≤ D_0 and
P_k ≤ P_L for every k. ∎

**Lemma 2 (quantile asymptotics, PROVED-SKETCH).** For the gamma law with
scale θ and shape κ_s, x(z) := G⁻¹(Φ(z)) satisfies
x(z) = θ z²/2 + O(log(2 + z)) for z ≥ 0 and 0 ≤ x(z) ≤ x(0) for z ≤ 0.
*Sketch.* −log(1 − G(x)) = x/θ − (κ_s − 1) log(x/θ) + O(1) and
−log Φ̄(z) = z²/2 + log z + O(1); invert. ∎

**Theorem 1 (large deviations, PROVED-SKETCH).** As c → ∞ with the route
fixed: log P(D_k > c) ~ −c/λ_k^d, log P(P_k > c) ~ −c/λ_k^p,
log P(L_k > c) ~ −c/λ_k, and log P(F > c) ~ −c/λ*.
*Sketch.* Upper bound: by Lemma 2, D_k ≤ ½ Z_+ᵀ Θ Z_+ + R with R a sum of
O(log) terms, and Z_+ᵀΘZ_+ ≤ ZᵀΘZ. For t < 1/λ, E exp(t·½ ZᵀΘZ) =
det(I − tΘ^{1/2}CΘ^{1/2})^{-1/2} < ∞ (Z ~ N(0, C) on the block), so Chernoff
gives log P(½ZᵀΘZ > c(1−η)) ≤ −t c(1−η) + O(1); P(R > ηc) decays faster
than any exponential in c; let t ↑ 1/λ, η ↓ 0. For L_k = D_k + P_k the
blocks are independent (disjoint inputs), so the moment generating
functions multiply and the rate is the smaller of the two, i.e. 1/λ_k.
Lower bound: let v be the Perron vector of Θ^{1/2}CΘ^{1/2} (positive), u*
the corresponding input direction; the event contains a ball around
(1+η)√(2c/λ)·u* of radius O(1), whose Gaussian mass is
exp(−(1+η)²c/λ − O(√c)). For F use Proposition 1: P(max(D_0, P_L) > c) ≤
P(F > c) ≤ P(D_0 + P_L > c), both with rate 1/λ*. ∎

So, on the logarithmic scale, a route overflows either at the start
(delivery block, scale λ_0^d) or at the end (pickup block, λ_L^p); middle
stages have strictly smaller scales whenever every customer has positive
mean demand, but for long routes λ_k is close to λ* for k near 0 or L.

**Proposition 3 (relative mode mass, PROVED from Theorem 1).**
P(L_k > c)/P(F > c) = exp(−c(1/λ_k − 1/λ*)(1 + o(1))). In particular, at a
service level γ = P(F > c), the second end-mode carries relative mass
γ^{(λ*/λ_other − 1)(1+o(1))}, λ_other = min(λ_0^d, λ_L^p).

This is the practical criterion: with ratio ϱ = λ_other/λ* close to 1
(balanced deliveries and pickups, Northern Vietnam), the neglected mode
holds a non-vanishing share of the probability at every γ; with ϱ ≤ 1/2
(Guangdong, pickups ≈ 2.7 × deliveries) it holds at most γ^{1}.
Measured (QUASAR A, 120 routes): single-shift relative error median
0.1–0.2% for ϱ ≤ 0.5, median 3–6% and up to 40% for ϱ ∈ (0.75, 1].

## 3. Single mean shift

Mean-shift IS with shift θ ∈ ℝ^{2L+2}: q = N(θ, I), weight
w(u) = exp(−⟨u, θ⟩ + |θ|²/2). Second moment
m₂(θ) = E_p[w 1_A] = e^{|θ|²}(2π)^{-n/2} ∫_A e^{−|u+θ|²/2} du. The estimator
is logarithmically efficient if log m₂ ~ 2 log P(A).

**Lemma 4 (one block, PROVED).** Let h(u) = ½ uᵀMu on a block with
M = AᵀΘA, A ≥ 0 entrywise, λ1 > λ2 its two largest eigenvalues,
κ = λ1/λ2, and h₊(u) = ½|Θ^{1/2}(Au)_+|² ≤ h(u). Let θ = r v1 with
r² = 2c/λ1 (v1 the Perron vector). If κ ≥ 4, every u with h₊(u) ≥ c
satisfies |u + θ|² ≥ 4|θ|².
*Proof.* Write u = a v1 + u_⊥, b = |u_⊥|. If a ≥ 0: c ≤ h₊(u) ≤ h(u) =
½(λ1 a² + u_⊥ᵀMu_⊥) ≤ ½(λ1 a² + λ2 b²), so b² ≥ κ(r² − a²); then
|u + θ|² = (a + r)² + b² ≥ f(a) := (1 − κ)a² + 2ar + (1+κ)r², concave for
κ ≥ 1, so on [0, r] its minimum is min(f(0), f(r)) = min((1+κ)r², 4r²) =
4r² when κ ≥ 3 (and for a ≥ r, |u+θ|² ≥ (a+r)² ≥ 4r²). If a < 0:
A v1 ≥ 0 gives (Au)_+ ≤ (Au_⊥)_+, so c ≤ h₊(u_⊥) ≤ ½λ2 b², b² ≥ κr², and
|u+θ|² ≥ b² ≥ κ r² ≥ 4r² when κ ≥ 4. ∎
(For the one-factor block, λ2 ≤ (1−ρ) max θ_i, so κ ≥ 4 holds as soon as
ρ Σθ_i ≳ 3(1−ρ) max θ_i — e.g. two equal customers at ρ = 0.6.)

**Theorem 2 (single shift: when it works, when it fails, PROVED-SKETCH).**
(a) If the maximal scale λ* is attained by one end mode only, the block of
that mode has κ ≥ 4, and every other stage block has scale λ_k ≤ λ*/3, the
shift at the dominating point θ* = √(2c/λ*) u* is logarithmically
efficient.
(b) If the other end mode (disjoint inputs, so orthogonal to θ*) has
scale λ_other > λ*/3, the shift θ* is not logarithmically efficient:
log m₂(θ*) − 2 log P(A) ≥ c(3/λ* − 1/λ_other)(1 + o(1)) → ∞.
*Sketch.* (a) Points of the dominant mode: Lemma 4 gives the exponent
½|u+θ*|² − |θ*|² ≥ |θ*|² = 2c/λ*. Points of other blocks have
|u|² ≥ 2c/λ_k ≥ 6c/λ*, and ⟨u, θ*⟩ ≥ −(...) — needs the non-negativity of
the inner product on the relevant region (to write; holds for the
disjoint end mode, where it is 0). (b) Evaluate the integrand of m₂ at the
other mode's dominating point θ_o ⊥ θ*: ½|θ_o + θ*|² − |θ*|² =
c/λ_other − c/λ*; a ball of radius O(1) around it lies in A, so
log m₂ ≥ −c/λ_other + 2c/λ* − O(√c), while 2 log P(A) = −2c/λ*(1+o(1)). ∎

(b) is asymptotic: at finite c the extra second moment comes from a region
that the proposal almost never visits, so the sample variance does not
show it and the estimate is biased low by about the mode mass of
Proposition 3 with a misleadingly small standard error — the failure
measured on the balanced routes.

## 4. Stage-mixture importance sampling (closed form)

**Estimator SMIS.** Components θ_m = √(2c/λ_m) u_m for a set of modes m
(each end mode; each stage k whose relative mass exp(−c(1/λ_k − 1/λ*))
exceeds a cut η, with u_m the Perron direction of its dominant block), a
defensive component N(0, I), weights a_m ∝ exp(−c/λ_m) (normalised, with a
floor). All quantities are closed form (secular equations); no pilot
simulation.

**Theorem 3 (PROVED-SKETCH under conditions).** If (i) every block of every
retained mode has κ ≥ 4, (ii) within every retained stage one block
dominates the other by a factor ≥ 3, and (iii) every stage with
λ_k > λ*/3 is retained, then SMIS is logarithmically efficient.
*Sketch.* w ≤ w_m/a_m on each mode region, so
m₂ ≤ Σ_m a_m^{-1} E_p[w_m 1_{A_m}] + (stages not retained); each term is
bounded as in Theorem 2(a); non-retained stages have scale ≤ λ*/3 and
contribute exponentially less. ∎

**Proposition 4 (split points, PROVED).** Condition (ii) cannot be dropped
for a finite mixture: if a stage has λ_k^d = λ_k^p = λ_k (a within-stage
tie) with λ_k ≥ λ*(1 − η) and η < 0.26, the mixture of the two block
directions is not logarithmically efficient. (Points of the stage event
with energy split between the two blocks at angle φ have exponent
E + 2√E·max(cos φ, sin φ)·… − … < 2c/λ*; computation in the notes.)
A continuous arc mixture over φ ∈ [0, π/2] restores efficiency up to a
polynomial factor (CONJECTURE; discretise with O(√c) angles).

## 5. Plan level

All routes of an instance share the factors (W_d, W_p). The plan overflow
event ∪_r {F_r > c_r} is again a union of stage blocks, now over all
routes, with the same scales; Theorem 1 gives the rate max over routes,
Proposition 3 the relative masses, and SMIS extends with one component per
retained (route, stage, block). (Corollary, PROVED-SKETCH.)

## 6. Open items

1. Full proofs of Lemma 2, Theorem 1 (uniformity in the O(log) terms),
   Theorem 2(a) inner-product step, Theorem 3.
2. Proposition 4 computation written out; arc mixture.
3. Non-asymptotic statement: the bias of a single shift at finite c in
   terms of the mode mass — what practitioners see.
4. Check against Glasserman & Li (2005), Glasserman & Wang (1997, Ann.
   Appl. Probab., counterexamples to single-shift IS), Sadowsky & Bucklew
   (1990), Dupuis & Wang (2007): which statements are special cases.
