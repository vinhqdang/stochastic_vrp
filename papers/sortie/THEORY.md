# SORTIE — theory

**S**tochastic **O**rdered **R**oute fairness with **T**our-consistency via
**I**ntegral **E**valuation.

This file holds the model, the new formulas and their proofs. The
manuscript (`sortie.tex`) is built from it.

---

## 1. Model

Complete graph on nodes `V = {0,…,n}` (depot `0`), symmetric costs `c_ij`
that satisfy the triangle inequality, `m` identical vehicles of capacity
`Q`, demands `q_i`. A *plan* is a partition `S_1,…,S_m` of the customers
(every vehicle used, `Σ_{i∈S_k} q_i ≤ Q`) together with one tour `r_k` of
`{0} ∪ S_k` per vehicle.

**Stochastic workload.** The travel time of arc `(i,j)` is

    τ_ij = Y · c_ij · ε_ij ,

where `Y > 0` is a *day factor* common to the whole fleet (weather,
network-wide congestion) with `E[Y] = 1`, `Var Y = s_Y²`, and the
`ε_ij` are arc-level disturbances, independent of each other and of `Y`,
with `E ε = 1`, `Var ε = v²` (`v` is the arc coefficient of variation).
The workload of driver `k` is `Φ_k = Y · X_k` with

    X_k = Σ_{(i,j)∈r_k} c_ij ε_ij ,  E X_k = L_k = Σ c_ij ,
    Var X_k = σ_k² = v² · W_k ,  W_k = Σ_{(i,j)∈r_k} c_ij² .

`L_k` is the deterministic tour length and `W_k` its *quadratic length*.
Conditionally on `Y` the `Φ_k` are independent; unconditionally they are
positively correlated. For `v = 0, s_Y = 0` the model is the deterministic
one of Ljubić, Puerto & Torrejon (2026).

**Ordered fairness.** For `a ∈ R^m` with order statistics
`a_(1) ≤ … ≤ a_(m)` and a weight vector `λ ∈ R^m`,
`ℓ_λ(a) = Σ_j λ_j a_(j)`. Max: `λ = e_m`; range: `λ = e_m − e_1`;
Gini mean difference: `λ_j = 2(2j−m−1)/m²`; mean absolute deviation from
the median (`m` odd, `q = (m+1)/2`): `λ_j = (−1,…,−1,0,1,…,1)/m`.
All four have **nondecreasing** `λ`.

**Three ways to be fair under uncertainty.** For a plan with random
workload vector `Φ`:

| name | quantity | reads as |
|---|---|---|
| ex-ante | `ℓ_λ(E Φ)` | unfairness of expected workloads (what a deterministic model optimises) |
| ex-post | `E ℓ_λ(Φ)` | expected unfairness of the realised day (sorted *per day*) |
| risk-of-unfairness | `CVaR_β ℓ_λ(Φ)` | bad-day unfairness |

SORTIE optimises the ex-post measure. The identity of the most burdened
driver changes from day to day; sorting per day is what makes the
ex-post measure differ from the ex-ante one.

## 2. New results

### Proposition 1 (common shocks are fairness-neutral in expectation)
For every `λ`, `E ℓ_λ(Φ) = E[Y] · E ℓ_λ(X) = E ℓ_λ(X)`; for the
ex-post empirical variance, `E Var_emp(Φ) = E[Y²] · E Var_emp(X)`.

*Proof.* `ℓ_λ` is positively homogeneous of degree 1 (sorting commutes
with multiplication by `y > 0`), so `ℓ_λ(yX) = y ℓ_λ(X)`; take
expectations using the independence of `Y` and `X`. `Var_emp` is
homogeneous of degree 2. ∎

Consequence: for ordered linear measures the day factor can be dropped
from the planning objective and the evaluation reduces to the
conditionally independent vector `X`. It still matters for the variance
measure and for CVaR of unfairness (reported out of sample).

### Theorem 2 (Poisson–binomial order-statistic integral, PBOI)
Let `X_1,…,X_m ≥ 0` be independent with CDFs `F_k`. For `t ≥ 0` let
`N(t) = #{k : X_k ≤ t}`, a Poisson–binomial count with success
probabilities `p_k(t) = F_k(t)`, and `π_r(t) = P(N(t) = r)`. Define the
tail sums `Λ_r = Σ_{j=r+1}^{m} λ_j` (`r = 0,…,m−1`). Then

    E ℓ_λ(X) = ∫_0^∞ Σ_{r=0}^{m−1} Λ_r π_r(t) dt .                (PBOI)

*Proof.* `X_(j) > t` iff fewer than `j` of the `X_k` are `≤ t`, so
`E X_(j) = ∫_0^∞ P(X_(j) > t) dt = ∫_0^∞ P(N(t) ≤ j−1) dt`. Hence
`E ℓ_λ = ∫ Σ_j λ_j Σ_{r≤j−1} π_r dt = ∫ Σ_r π_r Σ_{j≥r+1} λ_j dt`
(Tonelli applies to the positive and negative parts of `λ` separately). ∎

`π(t)` is the coefficient vector of `Π_k (1 − p_k + p_k z)`, computed
by `m` convolution steps (`O(m²)` per abscissa). Below
`t_lo = min_k F_k^{-1}(10^{-12})` the integrand equals `Λ_0` (contributes
`Λ_0 t_lo` analytically) and above `t_hi = max_k F_k^{-1}(1−10^{-12})` it
is `0` (`Λ_m := 0`). The evaluation is `O(G m²)` for `G` abscissae and is
*deterministic*: plans can be compared without sampling noise, unlike
sample-average approximation. The order-statistics identity is classical
in statistics (Poisson–binomial / permanent representation of
non-identical order statistics); its use as an exact planning objective
for ordered fairness of stochastic routes, with the tail-sum weights
`Λ_r` handling negative `λ`, is the contribution.

Marginals: the moment-matched gamma `X_k ~ Γ(α_k = L_k²/σ_k²,
θ_k = σ_k²/L_k)` (exact moments, positive support). The approximation
error against arc-level simulation is measured in experiment E1.

### Proposition 3 (rank prices)
Define the *rank price* of driver `k`

    ϖ_k = Σ_j λ_j P(R_k = j),   P(R_k = j) = ∫ f_k(t) P(N_{−k}(t) = j−1) dt,

where `R_k` is the rank of `X_k` and `N_{−k}` the Poisson–binomial count
of the other drivers. If `X_k` is shifted by `δ`, then
`d/dδ E ℓ_λ(X + δ e_k) |_{δ=0} = ϖ_k`. Moreover `Σ_k ϖ_k = Σ_j λ_j`.

*Proof.* `ℓ_λ` is piecewise linear with gradient `∂ℓ_λ/∂a_k = λ_{R_k(a)}`
off the null set of ties; it is `‖λ‖_1`-Lipschitz, so differentiation and
expectation commute (dominated convergence). `Σ_k P(R_k = j) = 1`. ∎

For zero-sum dispersion measures (range, GMD, MAD) the prices sum to
zero: burdened drivers carry positive prices, light drivers negative
prices. **A negative price is exactly the incentive to lengthen a route**
— the stochastic face of route inconsistency. SORTIE uses the prices as
a first-order screen: a relocation/swap that changes `(L_a, σ_a)` and
`(L_b, σ_b)` is predicted to change the objective by
`ϖ_a ΔL_a + ϖ_b ΔL_b + ψ_a Δσ_a + ψ_b Δσ_b`, where `ψ_k = ∂E ℓ_λ/∂σ_k`
is obtained from one extra PBOI evaluation per route (finite difference).

### Proposition 4 (fairness-illusion gap)
If `λ` is nondecreasing then `ℓ_λ` is convex and

    0 ≤ E ℓ_λ(X) − ℓ_λ(L) ≤ ‖λ‖_1 · E‖X − L‖_∞ ≤ ‖λ‖_1 · v · (Σ_k W_k)^{1/2}.

For the variance, `E Var_emp(X) = Var_emp(L) + ((m−1)/m²) Σ_k σ_k²`
(exact, independence only). For the Gini mean difference with Gaussian
marginals, `E GMD(X) = m^{-2} Σ_{i≠j} [ s_ij √(2/π) e^{−d_ij²/(2 s_ij²)}
+ d_ij (1 − 2Φ(−d_ij/s_ij)) ]` with `d_ij = L_i − L_j`,
`s_ij² = σ_i² + σ_j²`.

*Proof.* Nondecreasing `λ` gives `ℓ_λ(a) = max_σ Σ_j λ_j a_{σ(j)}`
(rearrangement inequality), a maximum of linear functions; Jensen gives
the left inequality. `|a_(j) − b_(j)| ≤ ‖a − b‖_∞` for every `j` gives the
Lipschitz bound, and `E max_k |Z_k| ≤ (E Σ_k Z_k²)^{1/2}`. The variance
identity expands `E X_k²` and `E X̄²`; the GMD formula is the folded-normal
mean. ∎

Reading: a deterministic (ex-ante) fair plan systematically *overstates*
the fairness drivers will experience, by an amount that grows with the
arc noise `v` and the quadratic lengths `W_k` — long arcs are what make a
plan unfair in practice even when the expected lengths are equal.

### Proposition 5 (monotonicity no longer protects drivers)
Under deterministic workloads a monotone measure (`λ ≥ 0`) always admits
an optimal plan with TSP-optimal routes (Matl et al., 2018). Under
stochastic workloads this fails even for `E max`.

*Example.* Two drivers. `X_1 ≡ 100`. Driver 2 can run tour A
(`L = 100, σ = 10`) or tour B (`L = 101, σ ≈ 0`). With Gaussian `X_2`,
`E max = 100 + 10·φ(0) ≈ 103.99` under A and `≈ 101` under B; the
optimiser picks the TSP-suboptimal tour B. ∎

So under uncertainty route consistency is not implied by monotonicity and
must be imposed for every measure, and "the best tour for the driver"
needs a risk attitude. SORTIE uses

    ρ_κ(r) = E[X | Y] / Y + κ · SD[X | Y] / Y = L(r) + κ v √W(r),

the driver's own mean–standard-deviation of the tour, which ranks tours
the same way on every day (it is invariant to `Y`). `κ = 0` is
TSP-optimality (ties broken by smaller `W`).

### Theorem 6 (risk-consistent tours are supported tours)
For `κ ≥ 0`, some minimiser of `ρ_κ` over the tours of a fixed customer
set minimises `L(r) + θ W(r)` for some `θ ∈ [0, ∞]` (`θ = ∞`: minimise
`W`, ties by `L`). All such tours are found by dichotomic (Aneja–Nair)
search, each step one Held–Karp run with arc weights `c_ij + θ c_ij²`.

*Proof.* `f(L, W) = L + κ v √W` is concave and nondecreasing on `R²_+`.
A concave function attains its minimum over the convex hull of the finite
point set `P = {(L(r), W(r))}` at an extreme point; because `f` is
nondecreasing it is attained at an extreme point of `conv(P) + R²_+`,
and those are exactly the minimisers of `w_1 L + w_2 W` over `P` for
some `(w_1, w_2) ≥ 0, ≠ 0`, i.e. `θ = w_2/w_1 ∈ [0, ∞]`. Dichotomic
search enumerates every extreme point of `conv(P) + R²_+` (Aneja &
Nair, 1979). ∎

### Bilevel model (SORTIE problem)

    min  E ℓ_λ(X(r_1,…,r_m))
    s.t. (S_1,…,S_m) capacity-feasible partition, all S_k ≠ ∅,
         Σ_k L(r_k) ≤ B · C*                       (expected-cost budget),
         r_k ∈ argmin { ρ_κ(r) : r a tour of {0} ∪ S_k }   ∀k.

`C*` is the minimum expected total length (best known). With `v = 0` and
`κ = 0` this is exactly the X-CVRP-TSP of Ljubić et al. (2026).

## 3. Algorithm (SORTIE)

1. **Lower level** `tour(S, κ)`: exact Held–Karp (numba) for `|S| ≤ 15`,
   parametric dichotomic search over `θ` for `κ > 0` (Theorem 6);
   2-opt + Or-opt from several starts for larger sets; memoised on
   `(frozenset S, κ)`.
2. **Evaluator**: PBOI (Theorem 2) on the gamma marginals; exact closed
   form for the variance (Proposition 4).
3. **Upper level**: start from the cost-efficient plan (feasible for any
   `B ≥ 1`). Each iteration computes the rank and dispersion prices
   (Proposition 3), scores *all* relocate and swap moves by the
   first-order prediction using cheap insertion/removal deltas, evaluates
   the best `K` exactly (lower level + PBOI), and applies the best
   budget-feasible improving move. With no improving move, a
   ruin-and-recreate perturbation removes customers preferentially from
   the extreme-price routes and reinserts them by regret; simulated
   annealing acceptance; best feasible plan kept.
4. **Out-of-sample evaluation** by arc-level Monte Carlo (lognormal
   day factor, gamma arc disturbances, common random numbers across
   methods).
