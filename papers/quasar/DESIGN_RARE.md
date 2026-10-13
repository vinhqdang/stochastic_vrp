# QUASAR — rare-event certification (direction A, 2026-10-13)

## Question

Safety-critical services (medicine, vaccines in cold chain, hazardous
goods) need route overflow probabilities p of order 10⁻³…10⁻⁶ certified to a
relative error ε_rel. Is quantum amplitude estimation cheaper than the best
classical method in that regime, on real routes of the region plans?

## Setting

Route j with customers i₁…i_L, demands from the BATON model: gamma
marginals (cv 0.3), one-factor Gaussian copula per demand type (ρ = 0.6),
deliveries and pickups with independent factors. Written as a simulator
driven by standard normal inputs
x = (W_d, W_p, E_d,1..L, E_p,1..L) ∈ ℝ^{2L+2}:
d_i = G_i⁻¹(Φ(√ρ W_d + √(1−ρ) E_d,i)), likewise p_i; failure
X = 1[F(x) > c] with F the peak load and c the threshold. The threshold c is
set per route so that p = P(F > c) hits a target level γ ∈ {1e-3, 1e-4,
1e-5, 1e-6} (c is then the capacity giving service level 1 − γ).

This Gaussian-input form is what a black-box digital twin exposes (its
random numbers), so the classical competitor may use any method that
changes the input law: importance sampling (IS) by a mean shift of the
Gaussian inputs, tuned by the cross-entropy method. No method may use the
exact DP of the BATON model (that is the testbed's ground truth only).

## Estimators and costs (95% confidence, relative error ε_rel)

- Classical crude MC: N = z² (1−p)/(p ε_rel²).
- Classical IS: N = z² RV/ε_rel², RV = Var_q(w X)/p² (relative variance per
  sample under the shifted law q, w the likelihood ratio); measured.
- Quantum crude: canonical amplitude estimation to relative error,
  explicit constants as in THEORY_CERT (Fact 1, median of r runs): the
  smallest M with 2π√(p(1−p))/M + π²/M² ≤ ε_rel·p, cost r(2M+1); reported
  ideal (M known) and with the doubling search for M.
- Quantum with IS: amplitude/mean estimation of w X under q. Idealised
  cost: the quantum crude cost with (1−p)/p replaced by RV (optimistic for
  the quantum side: it ignores the cost of reversible likelihood-ratio
  arithmetic and the heavier-tailed weights; Montanaro 2015 and
  Kothari–O'Donnell 2023 give O(σ/ε) up to logs).

Ratios to report per route and level: Q_crude/C_crude, Q_IS/C_IS and
Q_IS/min(C_crude, C_IS). Back-of-envelope: Q_IS/C_IS ≈ 29 ε_rel/√RV with
the M3 constants, so at ε_rel = 0.1 quantum wins against IS only when RV
exceeds ≈ 9, at ε_rel = 0.01 almost always. The experiment decides where
real routes sit.

## Ground truth

p at each threshold from IS with 10⁶ samples (relative standard error
reported); for routes with L ≤ 8 and γ ≥ 1e-4, cross-checked against the
exact peak DP with a fine Gauss–Hermite quadrature over the two factors
(M1/M3 machinery).
