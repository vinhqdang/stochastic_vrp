# RARECERT R2 — route-day simulator without an exact DP

Purpose: answer "why not the exact DP?" The demand-only model of R1 has an
exact peak DP; this simulator does not, yet keeps a Gaussian-input form so
that the dominating-point machinery applies.

## Route day
- Demands: as in R1 (gamma marginals, cv 0.3, one-factor Gaussian copula
  ρ = 0.6 per demand type).
- Travel times: lognormal per leg, T_l = t̄_l exp(σ_T(√ρ_T V + √(1−ρ_T)H_l)
  − σ_T²/2), t̄_l from the OSRM road distance at 40 km/h, σ_T = 0.25,
  shared congestion factor V with ρ_T = 0.5.
- Service time S_i = 10 min + 6 min per tonne handled (couples time to
  demand).
- Failure = overflow at any stage, OR late arrival at any stop beyond its
  nominal arrival + slack, OR return beyond nominal + 2·slack. Thresholds
  set so that the load part and the time part each carry about γ/2.
- No exact DP: the event couples a maximum of partial sums of gamma-copula
  demands with sums of lognormals with a shared factor and load-dependent
  service times.

## Estimators
- MDIS (mode-decomposed IS): failure modes g_m(x) > 0 exposed by the
  simulator (stage loads, stop lateness, return); dominating point of each
  mode by constrained minimisation from several structured starts; mixture
  over the kept points with weights exp(−(E − E_min)) and a defensive
  component; no pilot simulation. SMIS of R1 is the special case with load
  modes only.
- Baselines: cross-entropy Gaussian mixture over the same modes (with pilot
  cost), single CE shift, crude Monte Carlo (reference at γ = 1e-3 with
  10⁷ draws).
- Same 120 routes as R1, γ ∈ {1e-3, 1e-4, 1e-5}.
