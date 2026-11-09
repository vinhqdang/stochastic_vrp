# SORTIE code specification

Self-contained package `papers/sortie/code/sortie/` (no imports from
`svrpspd_wdro/` or other papers). Python 3.11, numpy, scipy, numba.
Tests in `papers/sortie/code/tests/`, run from `papers/sortie/code/` with
`python -m pytest tests -q`. Math is in `../THEORY.md`.

## Module A — `instances.py`

- `load_x641(path) -> (coords: ndarray (641,2) float, demand: ndarray (641,) int)`
  parse CVRPLIB `NODE_COORD_SECTION` / `DEMAND_SECTION` (file order;
  node 1 of the file = index 0).
- `@dataclass Instance: name, coords (n+1,2), demand (n+1,) with demand[0]=0,
  m, Q, C (n+1,n+1) float64 distance matrix`.
- `make_instance(coords, demand, n, b, m=5, cap_div=4.0) -> Instance`:
  nodes `b*(n+1) … b*(n+1)+n` of the file (block `b`, 0-based); the
  first is the depot (its demand set to 0); `Q = floor(sum(customer demands)/cap_div)`;
  `C[i,j] = ceil(euclid)` for i≠j, 0 diagonal. Name `X641-n{n}-b{b}`.
  Number of blocks available = `641 // (n+1)`.

## Module B — `tsp.py` (lower level)

Tours are Python lists of customer node ids (depot implicit at both ends).
- `held_karp(Wmat: ndarray (k+1,k+1)) -> (order: list[int] of 1..k, cost)`:
  exact DP, numba-jitted, node 0 = depot, k ≤ 16. Ties: any.
- `tour_stats(tour, C) -> (L, W)` with `L = Σ c`, `W = Σ c²` over the
  closed tour depot→…→depot.
- `local_search_tour(nodes, Wmat_full, starts=8, rng)`: 2-opt + Or-opt
  (segments 1..3) on weights, best of several starts (nearest neighbour
  + random); for |S| > 15.
- `class TourSolver(C, kappa, v)`: `solve(S: iterable[int]) -> (tour, L, W)`;
  memoised on frozenset(S).
  * `kappa == 0`: minimise `L` with arc weights `c + 1e-9*c²`
    (tie-break by smaller W). Exact HK when |S|≤15 else local search.
  * `kappa > 0`: Theorem 6. Dichotomic search: tour a = argmin L
    (weights c + 1e-9 c²), tour b = argmin W (weights c² + 1e-9 c);
    recursively, for a pair (a,b) with L_a<L_b, W_a>W_b set
    θ=(L_b−L_a)/(W_a−W_b), solve weights c + θc²; if new tour t has
    L_t+θW_t < L_a+θW_a − 1e-9 recurse on (a,t),(t,b). Return the found
    tour minimising `L + kappa*v*sqrt(W)`.
  * `|S| == 0` returns ([], 0, 0). |S| == 1, 2 trivially.
  * counters: `n_calls`, `n_hk`, `n_cache_hits`.
- `is_tsp_optimal(tour, C) -> (bool, L_opt)` via HK (|S|≤15).

## Module C — `pboi.py` (evaluator)

- Measures: `lam_vector(name, m)` for `"max","range","gmd","mad"`
  (THEORY §1; `"mad"` needs odd m, q=(m+1)/2, λ=(−1..−1,0,1..1)/m),
  and `"var"` handled separately.
- `ordered(a, lam)`: ℓ_λ for a vector, or row-wise for an (S,m) matrix
  (sort ascending along axis −1).
- `marginals(L, sigma, family="gamma"|"normal")` → object exposing
  `cdf(t)` and `pdf(t)` vectorised over (m, G) and `ppf(q)` per route.
  gamma: shape = L²/σ², scale = σ²/L (σ==0 ⇒ degenerate point mass:
  implement as gamma with σ=1e-9·L).
- `pboi(L, sigma, lam, G=257, family="gamma") -> float`: Theorem 2.
  t_lo = min_k ppf(1e−12), t_hi = max_k ppf(1−1e−12); abscissae linspace
  on [t_lo,t_hi] (G odd), composite Simpson; π(t) by convolution
  recursion vectorised over t (array shape (G, m+1)); Λ_r tail sums;
  result = Λ_0·t_lo + ∫. Must handle normal marginals (t_lo<0).
- `exante(L, lam) = ordered(L, lam)`.
- `expost_var(L, sigma) = var_emp(L) + (m−1)/m² Σσ²` (population var, /m).
- `rank_prices(L, sigma, lam, G=257) -> ndarray (m,)`: Prop. 3,
  `P(R_k=j) = ∫ f_k(t) P(N_{−k}(t)=j−1) dt` (Simpson on same grid).
  Check: Σ_k ϖ_k ≈ Σ λ.
- `sigma_prices(L, sigma, lam, h=0.05)`: ψ_k by forward finite difference
  of pboi in σ_k (σ_k·(1+h)); for "var": exact `2(m−1)/m² σ_k`.
- `gmd_normal_closed(L, sigma)` (Prop. 4 folded normal) for tests.
- `objective(L, W, v, measure)` convenience: σ = v·sqrt(W); returns pboi or expost_var.

## Module D — `sim.py` (Monte Carlo, ground truth)

- `simulate(inst, routes, v, sY, S, seed, chunk=2000) -> (X: (S,m), Y: (S,))`:
  Y = exp(sY·Z − sY²/2); ε_ij ~ Gamma(shape 1/v², scale v²) i.i.d. per
  traversed arc per scenario (v=0 ⇒ ε≡1); X_k = Σ_{arcs of r_k} c_ij ε_ij.
  Common random numbers: the ε for arc (i,j) in scenario s must depend
  only on (seed, s, i, j), not on the plan — draw a full (chunk, n+1, n+1)
  array per chunk from `np.random.default_rng([seed, chunk_index])`.
- `evaluate_plans(inst, list_of_routes, v, sY, S=20000, seed=…) -> list[dict]`:
  draws each chunk once and evaluates every plan on it (CRN, speed);
  `evaluate_plan` is the single-plan wrapper. Per plan dict: for
  Φ = Y·X: E ℓ for max/range/gmd/mad, E var_emp, CVaR_0.9 of range,
  mean total, per-driver means/SDs; same for X (no day factor) with key
  suffix `_X`.

## Tests (must pass)

1. instances: n=15,b=0 has 16 nodes, Q=floor(Σq/4), C symmetric, ceil.
   `641//16 == 40`, `641//36 == 17`.
2. HK equals brute force over all permutations for k ≤ 7 (random
   symmetric matrices, 20 cases).
3. TourSolver κ>0 equals brute-force argmin of `L+κ v√W` over all
   permutations for k ≤ 7 (20 random Euclidean cases, κ∈{0.5,1,2}, v=0.3).
4. pboi(normal) vs `gmd_normal_closed` (rel err < 1e−4) for m=2..6; m=2
   range vs folded normal closed form; pboi(max) for iid normals vs
   known E[max] of 2 normals (= μ + σ/√π).
5. pboi(gamma) vs Monte Carlo of independent gammas, 4e5 samples, all
   measures, within 4 standard errors.
6. Σ ϖ_k ≈ Σλ (1e−6); ϖ_k vs finite difference of pboi in L_k with σ
   fixed under normal family (rel err < 1e−3).
7. expost_var formula vs MC.
8. sim: CRN — two plans sharing a route give identical X columns for it;
   v=0,sY=0 ⇒ X equals L exactly.

## Module E — `search.py` (upper level; implemented after A–D)

Plan state: `routes: list[list[int]]` (m tours, all nonempty), cached
per-route `L, W, load`. Everything seeded by `rng = np.random.default_rng(seed)`.

### Objective adapters (`make_objective(mode, measure, v, scen=None)`)
Each adapter exposes `value(Ls, Ws, routes) -> float` and
`prices(Ls, Ws, routes) -> (varpi (m,), psi (m,))`.
- `"cost"`: value = Σ L; varpi = 1, psi = 0.
- `"det"` (ex-ante, deterministic fair, = Ljubić et al. objective):
  value = ordered(L, λ) (or var_emp(L) for "var"); varpi_k = λ_{rank of L_k}
  (for var: 2(L_k−mean)/m); psi = 0.
- `"pboi"` (SORTIE): value = pboi(L, v√W, λ) or expost_var; prices from
  `rank_prices` and `sigma_prices` (THEORY Prop. 3).
- `"saa"`: a fixed scenario tensor `eps` of shape (S, n+1, n+1) (S=200,
  seed independent of the out-of-sample seed); X^s_k = Σ_{arcs} c·eps^s;
  value = mean_s ordered(X^s, λ) (var: mean of var_emp); varpi_k =
  mean_s λ_{rank_k^s}; psi = 0. Cache X^s per route (key = tuple(tour)).

### Moves and first-order screening
Each iteration, for the current plan compute `varpi, psi`. Enumerate
*all* relocate moves (i from route a to route b≠a, b capacity-feasible,
a keeps ≥1 customer) and swap moves (i∈a, j∈b, both capacity-feasible).
For each, cheap deltas from the current tours: removal delta of i in a
(`c_{p i}+c_{i s}−c_{p s}` and the same with squares for W), best
insertion position of i in b (min over edges of `c_{x i}+c_{i y}−c_{x y}`,
W delta at that position); for swaps insert j into a-without-i and i into
b-without-j. Predicted Δ = Σ_{k∈{a,b}} varpi_k ΔL_k + psi_k Δσ_k with
σ = v√W. Budget pre-filter: Σ L + ΔL_a + ΔL_b ≤ budget·(1+0.02).
Vectorise with numpy where it matters (n up to 100).
Take the K=8 best predicted moves (mode `screen="random"`: K uniformly
random feasible moves instead — the ablation), evaluate each exactly
(lower level `TourSolver.solve` on the two changed sets + objective
value), apply the best one that is budget-feasible (exact Σ L ≤ budget)
and improves the current value by > 1e-9·max(1,|value|).

### Perturbation (no improving move among the K)
Ruin: remove `q ~ U{2, …, max(3, ⌈0.25 n⌉)}` customers; with prob. 0.5
from routes chosen with probability ∝ |varpi_k| (+tiny), otherwise
uniformly (related removal: a random seed customer and its nearest
neighbours, prob. 0.5 of the time). Recreate: regret-2 insertion of the
removed customers, insertion score of i into b = varpi_b·ΔL + psi_b·Δσ +
1e-3·ΔL (capacity-feasible only; if none feasible, put i in the route
with max residual capacity). Then re-solve changed routes with the
lower level. If the result violates the budget, run cost-only descent
(relocate/swap with exact L, objective "cost") until feasible or no
improvement; if still infeasible, discard. Accept the new plan as
current by simulated annealing on the objective: accept if
Δ ≤ 0 or with prob exp(−Δ/T), T = T0·(Tend/T0)^(elapsed/time_limit),
T0 = 0.02·|best value| + 1e-9, Tend = T0·1e-3. Keep the best feasible.

### Free-tour mode (`consistency="free"`, ablation for route inconsistency)
After the lower level produces tours, the planner may re-sequence any
changed route to improve the *global* objective: intra-route 2-opt and
Or-opt moves accepted iff they reduce the objective (with the other
routes fixed) and keep the budget. Only meaningful for pboi/det/saa.

### Entry points
- `construct_initial(inst, rng)`: random-angle sweep + regret insertion
  into exactly m nonempty capacity-feasible routes (retry with new angles).
- `solve_cost(inst, time_limit, seed, restarts=3) -> (routes, C*)`: mode
  "cost", κ=0, best of restarts (time split evenly).
- `solve_fair(inst, start_routes, mode, measure, v, kappa, budget,
  time_limit, seed, screen="price", consistency="lower", scen=None)
  -> dict(routes, value, history=[(t, best)], iters, n_exact_evals, solver
  counters, time)`. Budget is an absolute bound on Σ L (callers pass
  B·C*). `kappa` is the lower-level risk weight (0 = TSP-optimal).
- Everything respects `time_limit` (check every iteration).

### Tests for E
- solve_cost on n=15,b=0 returns 5 nonempty capacity-feasible routes
  covering every customer once; all tours TSP-optimal.
- solve_fair(det, range) value ≤ value of the start plan and budget holds.
- solve_fair(pboi) with budget = C* (B=1.00) keeps Σ L ≤ C*+1e-6.
- free mode can return non-TSP-optimal tours; lower mode never does.
- Determinism: same seed and iteration cap ⇒ same result (add an
  optional `max_iters` argument for this test).

## Module F — experiments (`code/experiments/`)

All outputs CSV under `papers/sortie/results/`, logs `*.log`, plans JSON
under `results/plans/`. Default stochastic parameters: v = 0.3,
sY = 0.15, out-of-sample S = 20000 (S = 10000 for n ≥ 75), seed 2026.
Parallelise over instances with `multiprocessing` (4 workers).
- `e0_seed.py`: C* per instance via solve_cost (n=15,20,25: 20 s;
  n=30,35: 40 s; n=50: 60 s; n=75,100: 120 s); instances n ∈
  {15,20,25,30,35} blocks 0–4, n ∈ {50,75,100} blocks 0–2; tiny
  n=10, m=3, cap_div=2.4, blocks 0–9. Writes `seed_plans.csv` + JSON.
- `e1_accuracy.py`: PBOI(gamma) vs arc-level MC (X, no day factor) on
  200 plans (random feasible + solved), all measures; and timing of
  PBOI vs SAA(S=200) per evaluation.
- `e2_exact.py`: tiny instances, enumerate all capacity-feasible
  assignments to 3 labelled vehicles (dedupe by symmetry), tours by HK,
  exact optimum of the SORTIE problem (pboi, κ=0, B=1.10) for range and
  gmd; compare with solve_fair (5 s).
- `e3_main.py`: n ∈ {15..35} × 5 blocks × measures {max, range, gmd,
  mad, var} × B ∈ {1.05, 1.10} × methods {det, saa, sortie0 (pboi κ=0),
  sortie1 (pboi κ=1)}; time limit 15 s (n ≤ 25) / 25 s (n ≥ 30); every
  plan evaluated out of sample on the same scenarios; also cost plan.
- `e4_inconsistency.py`: measures {range, gmd, max}, B=1.10, n ∈ {15..35}
  × 5 blocks: pboi free vs pboi lower vs det free vs det lower; record
  number of non-TSP-optimal routes, excess length, in-sample vs true.
- `e5_ablation.py`: screen price vs random, same time, range & gmd, B=1.10.
- `e6_sensitivity.py`: v ∈ {0.1, 0.3, 0.5}, n=25, 5 blocks, range & gmd,
  B=1.10, det vs sortie0.
- `e7_scale.py`: n ∈ {50,75,100} × 3 blocks, range & gmd, B=1.10,
  det/saa/sortie0, 60 s.
