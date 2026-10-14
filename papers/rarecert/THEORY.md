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
Lower bound: let v be the Perron vector of Θ^{1/2}CΘ^{1/2} (positive,
ρ > 0) and u* = BᵀΘ^{1/2}v/√λ the input direction (Z = Bu, BBᵀ = C), for
which ½(Bsu*)ᵀΘ(Bsu*) = λs²/2; the event contains a ball around
(1+η)√(2c/λ)·u* of radius O(1), whose Gaussian mass is
exp(−(1+η)²c/λ − O(√c)). For F use Proposition 1: P(max(D_0, P_L) > c) ≤
P(F > c) ≤ P(D_0 + P_L > c), both with rate 1/λ*. ∎

So, on the logarithmic scale, a route overflows either at the start
(delivery block, scale λ_0^d) or at the end (pickup block, λ_L^p); middle
stages have strictly smaller scales whenever every customer has positive
mean demand, but for long routes λ_k is close to λ* for k near 0 or L.

**Proposition 3 (relative mode mass, PROVED from Theorem 1, for a fixed
gap λ_k < λ*; as the gap closes, prefactors decide and the statement is
silent).**
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

Mean-shift IS with shift τ ∈ ℝ^{2L+2}: q = N(τ, I), weight
w(u) = exp(−⟨u, τ⟩ + |τ|²/2). Second moment m₂(τ) = E_p[w 1_A]; the
estimator is logarithmically efficient if log m₂ ~ 2 log P(A).

**Lemma 4 (one block, PROVED).** Let h(u) = ½ uᵀMu on a block with
M = AᵀΘA, A ≥ 0 entrywise, λ1 > λ2 its two largest eigenvalues,
κ = λ1/λ2, and h₊(u) = ½|Θ^{1/2}(Au)_+|² ≤ h(u). Let τ = r v1 with
r² = 2c/λ1 (v1 the Perron vector). If κ ≥ 4, every u with h₊(u) ≥ c
satisfies |u + τ|² ≥ 4|τ|².
*Proof.* Write u = a v1 + u_⊥, b = |u_⊥|. If a ≥ 0: c ≤ h₊(u) ≤ h(u) =
½(λ1 a² + u_⊥ᵀMu_⊥) ≤ ½(λ1 a² + λ2 b²), so b² ≥ κ(r² − a²); then
|u + τ|² = (a + r)² + b² ≥ f(a) := (1 − κ)a² + 2ar + (1+κ)r², concave for
κ ≥ 1, so on [0, r] its minimum is min((1+κ)r², 4r²) = 4r² when κ ≥ 3 (and
for a ≥ r, |u+τ|² ≥ (a+r)² ≥ 4r²). If a < 0: A v1 ≥ 0 gives
(Au)_+ ≤ (Au_⊥)_+, so c ≤ h₊(u_⊥) ≤ ½λ2 b², b² ≥ κr², and
|u+τ|² ≥ b² ≥ κ r² ≥ 4r² when κ ≥ 4. ∎
For the one-factor block λ2 ≤ (1−ρ) max θ_i (Weyl) and
λ1 ≥ ρΣθ_i + (1−ρ) min θ_i, so κ ≥ 4 holds when
ρΣθ_i + (1−ρ) min θ_i ≥ 4(1−ρ) max θ_i (two equal customers at ρ = 0.6
give κ = 4 exactly).

Write ϱ = λ_other/λ* ≤ 1 for the ratio of the weaker to the stronger end
scale, and τ* = √(2c/λ*) u* for the dominating point of the dominant end
block, u* = Bᵀ Θ^{1/2} v/√λ* in input space (Z = Bu, BBᵀ = C; all
components positive).

**Theorem 2 (single shift, PROVED-SKETCH).**
(b) If ϱ > 1/4, the shift τ* is not logarithmically efficient:
log m₂(τ*) − 2 log P(A) ≥ c(4/λ* − 1/λ_other)(1 + o(1)) → ∞.
*Proof.* τ* acts on the dominant block's inputs only, and the other end
event (say {P_L > c}) depends only on the other block's inputs, so
m₂ ≥ E_p[w 1{P_L > c}] = E_p[w]·P(P_L > c) = e^{|τ*|²} P(P_L > c)
= exp(2c/λ* − c/λ_other + o(c)), while 2 log P(A) = −2c/λ*(1+o(1)). ∎
(a) If ϱ ≤ 1/4 and the dominant end block has λ2 ≤ λ*/4, the shift τ* is
logarithmically efficient; no condition on the middle stages is needed.
*Sketch.* A ⊆ {D_0 + P_L > c} (Proposition 1) and, up to the O(log)
remainders of Lemma 2, D_0 + P_L ≤ h₊ for the block-diagonal form of the
two end blocks, whose top eigenvalue is λ* and second eigenvalue is
max(λ_other, λ2(dominant)) ≤ λ*/4; Lemma 4 then gives
½|u+τ*|² − |τ*|² ≥ |τ*|² on A. The remainder set B (where the O(log)
terms exceed ηc) is handled by Cauchy–Schwarz,
E_p[w 1_B] ≤ e^{3|τ|²/2}·P(B)^{1/2}, and P(B) is super-exponentially
small. ∎

Variance and bias are different matters at finite c. The extra second
moment in (b) comes from a region the proposal almost never visits, so the
sample variance does not show it; what a user sees is a low bias of about
the missing mode mass, γ^{1/ϱ − 1} on the log scale (Proposition 3), with
a misleadingly small standard error. That bias is negligible unless ϱ is
close to 1 — the measured split: ϱ ≤ 0.5 gives 0.1–0.2% error, ϱ > 0.75
gives 3–40%.

## 4. Stage-mixture importance sampling (closed form)

**Estimator SMIS.** Components τ_m = dominating points of a set of modes m
(both end blocks; each stage block whose relative mass
exp(−c(1/λ_m − 1/λ*)) exceeds a cut η), computed exactly (minimum-norm
point of {L_k ≥ c} with the exact quantile transform, started on the
Perron ray), a defensive component N(0, I) with a constant weight a_0, and
weights a_m ∝ exp(−|τ_m|²/2) on the rest. No pilot simulation.

**Theorem 3 (PROVED-SKETCH under conditions).** With a constant defensive
weight, SMIS is logarithmically efficient if (i) every retained block has
κ ≥ 4 and (ii) within every retained stage one block dominates the other
by a factor ≥ 4; stages with λ_k ≤ λ*/2 need not be retained (they are
covered by the defensive component: E_p[1_{A_k}/a_0] ≤ P(A_k)/a_0 and
P(A_k) ≤ e^{−2c/λ*(1+o(1))} when λ_k ≤ λ*/2).

**Proposition 4 (within-stage ties, COMPUTED, to be written as a proof).**
If a stage has its two blocks tied, λ_k^d = λ_k^p = xλ*, a mixture with
only the two block directions of that stage is not logarithmically
efficient when: x > 0.707 (equal weights), x > √2 − 1 ≈ 0.414 (weights
∝ e^{−c/λ}), x > 1/2 (with a defensive component); the worst points sit at
angle 45° between the blocks. A continuous arc of components restores
efficiency up to a polynomial factor (CONJECTURE). In the linear regime
the stage where the delivery and pickup loads cross has
λ ≈ λ*ϱ/(1+ϱ) ≤ λ*/2, so ties matter only for heterogeneous routes.

**Positioning (after review).** Not new importance-sampling theory: the
single-block statement is the dominating-point setting of Sadowsky &
Bucklew (1990); Theorem 2(b) is an instance of the multiple-paths failure
of Glasserman & Wang (1997); mixtures are subsolution constructions in the
sense of Dupuis & Wang; the quadratic-form structure matches delta–gamma
VaR (Glasserman, Heidelberger & Shahabuddin 2000, 2002). New for routing:
the secular-equation scales and the two-end-mode structure of the
pickup-and-delivery peak, the sharp ϱ > 1/4 criterion with the γ^{1/ϱ−1}
bias diagnostic, the plan-level extension, and the pilot-free closed-form
mixture.

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
4. Finite-c regime: for cv ≈ 0.3 the gamma quantile is closer to
   Wilson–Hilferty than to θz²/2 at γ = 1e-3…1e-6, so asymptotic scales
   may mispredict; SMIS uses exact dominating points to avoid this.
5. Lemma 2 refinement: x(z) = θz²/2 + θ(2κ_s − 1) log z + O(1).
6. Review record: Theorem 2 threshold corrected from 1/3 to 1/4
   (E_p[w] = e^{|τ|²} on the other block was missed in draft 1).
