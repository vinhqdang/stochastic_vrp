# SORTIE — specification, extension 2 (library)

Extends the library for experiments E9–E17. Existing behaviour must not change:
every new argument has a default that reproduces the current results exactly,
and the existing 67 tests stay green.

## L1. Generic instances (`sortie/instances.py`)

- `load_vrp(path) -> (name, coords (N,2) float, demand (N,) int, capacity int)`:
  TSPLIB CVRP reader for EUC_2D files. Robust to CRLF line endings, tabs,
  repeated spaces and trailing spaces. Node ids in the file are 1-based; the
  depot is the node listed in DEPOT_SECTION (always 1 in the X set); return
  arrays in file order with the depot first. `load_x641` must give identical
  arrays to before (it may call `load_vrp`).
- `make_instance(coords, demand, n, b, m=5, cap_div=4.0, prefix="X641")`: add
  the `prefix` argument; the name becomes `f"{prefix}-n{n}-b{b}"`. With the
  default the name is unchanged.
- `make_head_instance(coords, demand, prefix, n, m=5, cap_div=4.0)`: the depot
  of the file plus its first n customers (file order), Q = floor(sum q /
  cap_div), ceiled Euclidean distances, name `f"{prefix}-h{n}"`. This keeps the
  file's own depot position (unlike blocks, whose depot is the first node of
  the block).

## L2. Arc-noise families (`sortie/sim.py`)

Add `dist="gamma"` (and its parameters) to `_draw_chunk`, `simulate`,
`evaluate_plans`, `evaluate_plan`. Every family has mean 1 per arc and
overall per-arc coefficient of variation `v * H_ij` (H = 1 when absent), and
draws depend only on (seed, chunk index, scenario, i, j) as now (common
random numbers across plans). Families:

- `"gamma"`: current behaviour, bit-identical.
- `"lognormal"`: eps = exp(s Z - s^2/2) with s^2 = log(1 + (vH)^2).
- `"incident"`: eps = G * J / E[J], J = inc_mult with probability p_inc and 1
  otherwise (independent per arc and scenario), G gamma with mean 1 and cv
  v0 chosen so the product has cv vH: (1 + v0^2)(1 + cvJ^2) = 1 + (vH)^2,
  cvJ^2 = Var J / (E J)^2. Defaults p_inc = 0.02, inc_mult = 2.5. If v0^2 < 0
  for some arc, raise ValueError.
- `"regional"`: spatially correlated noise. The nodes are split into
  `n_regions` (default 4) regions by the quadrant of their coordinates
  relative to the median x and median y of the instance (pass `coords`; the
  depot's region is that of its own position). Each scenario draws a regional
  factor R_g ~ lognormal(mean 1, cv region_cv) per region (default region_cv
  = 0.15); arc (i, j) gets eps = G * R_{g(i)} where G is gamma with mean 1 and
  cv v0, with (1 + v0^2)(1 + region_cv^2) = 1 + (vH)^2 (raise if negative).
  This makes routes in the same region positively correlated, which PBOI
  ignores.

`evaluate_plans(..., dist="gamma", dist_params=None)` with `dist_params` a dict
of the family parameters; for `"regional"` the instance coordinates come from
`inst.coords`.

## L3. Objectives (`sortie/search.py`)

- Mode `"detsig"`: deterministic ordered measure of the risk-adjusted lengths
  L_k + z * sigma_k, sigma_k = v sqrt(W_k), z = `z` argument (default 1.0).
  For measure "var" use the empirical variance of L + z sigma. Prices: varpi =
  deterministic rank prices of L + z sigma (as in DetAdapter), psi = z * varpi
  (so the first-order screen sees the change in sigma). `make_objective(...,
  z=1.0)` and `solve_fair(..., z=1.0)` pass it through. Add "detsig" to MODES.
- `PboiAdapter(measure, v, family="gamma")`; mode `"pboi_normal"` = PBOI with
  normal marginals (family "normal"), for the marginal-family ablation. Add to
  MODES. Mode "pboi" unchanged.
- `make_objective(..., S=SCEN_S)` already takes the SAA sample size; make sure
  `solve_fair` accepts `saa_S=None` and, when mode is "saa" and `scen` is None,
  builds the scenario tensor with that S (default SCEN_S = 200, unchanged).

## L4. Move diagnostics (`sortie/search.py`)

`move_diagnostics(inst, routes, mode, measure, v, kappa, budget, max_moves=None,
seed=0, z=1.0)` for consistency "lower":

1. Solve every route with the exact lower level (TourSolver) to get the plan.
2. Compute prices of the plan (adapter.prices).
3. Enumerate every capacity- and budget-feasible relocate and swap move,
   with the first-order prediction of the screen (`_screen` with K = all
   moves, or an equivalent refactor that shares code with `_screen`).
4. For each move (or a uniform random subset of `max_moves`, rng(seed)),
   compute the exact change in objective (exact lower level for both changed
   routes, adapter.value). Moves that break the budget after the exact
   re-solve get exact = NaN.

Return a pandas DataFrame with columns kind (relocate/swap), a, b, i, j,
predicted, exact, pred_rank (rank of the move by prediction, 0 = best),
exact_rank. Must not change `_screen`'s results for existing callers.

## Tests (`tests/test_ext.py`)

- `load_vrp` on `data/X-n641-k35.vrp` equals the arrays `load_x641` returned
  before (compare with a CRLF-converted temporary copy too), and on
  `data/X/X-n101-k25.vrp` gives 101 nodes, capacity 206, depot first.
- `make_head_instance` on X-n101-k25 with n=50: 51 nodes, depot = file depot.
- For each noise family: sample mean of eps ≈ 1 and sample cv ≈ v (within 3%)
  on a 20-node instance with 20 000 scenarios; "gamma" bit-identical to the
  old `_draw_chunk` for a fixed seed; "regional" gives positive correlation
  between the totals of two routes in the same region.
- detsig with z = 0 equals det values on random plans; pboi_normal value
  equals `pboi(..., family="normal")`.
- `move_diagnostics` on a 15-customer instance: predicted and exact columns
  present, Spearman correlation between predicted and exact > 0 for mode pboi
  range.
- `solve_fair` with mode "detsig" and "pboi_normal" runs (max_iters=20) and
  the returned routes satisfy the lower-level invariant (as in
  tests/test_lazy.py).

# Part E — experiments E9–E17

Conventions as in the existing drivers (`experiments/common.py`, `run_jobs`,
resumable CSVs in RESULTS, plans in RESULTS/plans/<exp>/, Pool(4), one row per
run with ROW_FIELDS plus experiment-specific columns). Every solve starts from
the E0 cost plan of its instance. Out-of-sample S = 20 000 (n ≤ 35) or 10 000
(n ≥ 50), CRN seed `inst_seed`. Default v = 0.3, sY = 0.15, B = 1.10.

- **E9 budget frontier** (`e9_budget.py`): n ∈ {25, 35} × blocks 0–4, B ∈
  {1.02, 1.05, 1.10, 1.15, 1.20, 1.30}, measures {range, gmd, max}, methods
  {det, sortie0}; E3 time limits. Plot: cost ratio vs ex-post value.
- **E10 fleet size** (`e10_fleet.py`): n = 35 × blocks 0–4, m ∈ {3, 5, 7}
  with cap_div = m − 1 (m = 5 reuses the E3 instances: cap_div 4), measures
  {range, gmd, max}, methods {det, saa, sortie0}, 25 s. Needs seed plans per
  (instance, m) — compute them in the driver with solve_cost (40 s, 3
  restarts) and cache as seed_<name>.json, names `X641-n35-b{b}-m{m}`.
- **E11 day factor** (`e11_dayfactor.py`, evaluation only): every E3 plan with
  B = 1.10 re-evaluated with sY ∈ {0, 0.15, 0.3, 0.5} (same arc draws per
  sY via the same seed). Columns: all E_* and CVaR columns per sY.
- **E12 misspecification** (`e12_misspec.py`, evaluation only): every E3 plan
  with B = 1.10 and E8 plan, evaluated under dist ∈ {gamma, lognormal,
  incident, regional}. Columns: per-family E_* (with and without day factor),
  plus the PBOI prediction (gamma) for the plan, so the PBOI error under each
  family and the det-vs-SORTIE gap under each family can be computed.
- **E13 risk attitude** (`e13_kappa.py`): n ∈ {25, 35} × blocks 0–4,
  κ ∈ {0, 0.5, 1, 2, 4}, measures {range, max}, mode pboi, B = 1.10, E3 time
  limits. Extra columns: number of routes whose lower-level tour differs from
  the κ = 0 (TSP) tour of the same customer set, total extra length vs TSP
  tours, mean per-driver risk L + σ.
- **E14 screening diagnostics** (`e14_screen.py`): (a) `move_diagnostics` on
  the E0 cost plan and on the E3 sortie0 plan (B = 1.10) of n ∈ {25, 35} ×
  blocks 0–4, measures {range, gmd, max}, all moves (max_moves = 400):
  CSV `e14_moves.csv` with one row per move. (b) K sweep: K ∈ {1, 2, 4, 8,
  16, 32}, n ∈ {25, 35} × 5 blocks, measures {range, gmd}, sortie0, B = 1.10,
  E3 time limits: CSV `e14_ksweep.csv`.
- **E15 SAA sample size** (`e15_saa.py`): n ∈ {25, 35} × 5 blocks, measures
  {range, gmd, max}, B = 1.10, SAA with S ∈ {50, 200, 1000, 5000} (scenario
  seed fixed), E3 time limits; plus sortie0 and det rows copied by joining
  with E3 at analysis time (not re-run).
- **E16 risk-adjusted deterministic baseline and marginal family**
  (`e16_baselines.py`): E3 grid with B = 1.10 (25 instances × 5 measures),
  methods {detsig (z = 1), pboi_normal}, E3 time limits.
- **E17 second benchmark family** (`e17_family.py`): the 12 files in
  data/X/, `make_head_instance(n = 50, m = 5, cap_div = 4)`; seed plans by
  solve_cost (60 s, 3 restarts); measures {range, gmd, max}, B = 1.10,
  methods {det, saa, sortie0, detsig}, 120 s, solver seeds {7, 11}.
