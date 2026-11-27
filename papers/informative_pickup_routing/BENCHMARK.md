# Benchmark against costly-information acquisition methods

The [batch model](FLEET_BATCHES.md) chooses all observations before seeing
any outcomes. This benchmark compares methods using the same score laws,
acquisition costs, and standby availability. Savings are incremental
relative to the best allocation using prior expected scores, after
deducting all signal costs.

## Closest published baselines

**Pandora's box with nonobligatory inspection** is the closest
information-acquisition model for $B=1$. One can select an uninspected
option at its prior mean, and acquisition can stop after each outcome.
The benchmark implements the committing policies of Beyhaghi and
Kleinberg. It tries each possible uninspected backup, plus the case with
no backup, and uses Weitzman's reservation rule for the other options.
Beyhaghi and Cai prove the tight 0.8 guarantee for this class and give a
PTAS for one-option sequential search. Fu, Li, and Liu independently give
a PTAS and prove NP-hardness. These guarantees concern the published
sequential objective. The table below reports absolute savings above the
prior-only allocation. [Beyhaghi and Kleinberg](https://doi.org/10.1145/3328526.3329626) · [Beyhaghi and Cai, Theorem 1.4 and §2](https://arxiv.org/abs/2212.01524) · [Fu, Li, and Liu](https://doi.org/10.1145/3564246.3585229) · [Weitzman](https://doi.org/10.2307/1910412)

For small fleets, an **exact sequential dynamic program** provides a
reference for any $B$. It compares stopping now with buying any remaining
signal and continuing optimally after its outcome. Its stopping value is
the sum of the $B$ largest positive scores, using realized posterior
scores for inspected routes and prior means for other routes. This gives
an upper bound on any batch policy's value under the same laws and costs.
It also resolves the one-standby sequential objective exactly on these
instances. The committing-policy implementation is cross-checked against
the capped-score expectation formula.

The **same-protocol** comparisons are singleton greedy, two-flip local
search from both the empty and full batches, and exhaustive search over
all observation subsets. A two-flip move can add, remove, or exchange up
to two observations. The local search uses the exact batch value and the
same objective as the proposed matching method. The locality restriction
applies to each move, so the final batch can include any number of signals.

Florio et al. are relevant routing comparators for a later test that
optimizes the actual sequence of service visits. Their Bayesian
restocking model updates beliefs along a fixed route, and their separate
partial reoptimization model permits swapping consecutive customer
visits. This benchmark holds service plans fixed and varies only which
signals are acquired. [Bayesian restocking](https://doi.org/10.1016/j.ejor.2022.10.045) · [Partial reoptimization](https://doi.org/10.1287/trsc.2022.1129)

## Instances

- **Synthetic:** 24 heterogeneous and 24 near-cutoff, eight-route fleets
  with binary signals. Each family contains 12 cases with $B=1$ and 12
  with $B=4$.
- **Dethloff and City with simulated demands:** Saved deterministic plans
  for Dethloff CON3-0 through CON3-7 and the first two 100-customer plans
  for each of Hanoi and Ho Chi Minh City. For each route, draw 8,192
  scenarios using the repository's gamma demand sampler with coefficient
  of variation 0.30 and within-route Gaussian-copula parameter 0.6.
  Routes are independent. Condition on feasible departure and completion
  of the first stop without a breach. A purchased binary signal gives the
  side of the median of the post-first-stop load. Estimate each
  signal-conditioned score as the mean downstream first-breach cost less
  the cost of a planned handoff at the first stop. Use the repository's
  handoff and emergency cost schedules. Each plan is tested at synthetic
  signal fees of 0.05, 0.25, and 1.00 and multiple standby counts.

The route-derived inputs use benchmark geometry and simulated demand
scenarios. Values are evaluated on the fitted finite score laws. The fees
are sensitivity parameters, and route-level acquisition does not change
the service order or require an actual additional detour.

## Results

Mean net savings by group are below. The three columns headed
*two-flip*, *batch optimum*, and *sequential optimum* use exact
evaluations of their respective policies on the benchmark score laws.

| Instance group | $B$ | Cases | Positive batch optimum | Singleton greedy | Matching | Two-flip | Batch optimum | Published committing | Sequential optimum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Heterogeneous | 1 | 12 | 10 | 1.4583 | 1.1366 | 1.4583 | 1.4583 | 1.6555 | 1.6555 |
| Heterogeneous | 4 | 12 | 12 | 3.1459 | 2.9676 | 3.1459 | 3.1459 | — | 3.3915 |
| Near cutoff | 1 | 12 | 11 | 0.5414 | 0.4434 | 0.5414 | 0.5414 | 0.6101 | 0.6101 |
| Near cutoff | 4 | 12 | 11 | 0.6780 | 0.5282 | 0.6797 | 0.6797 | — | 0.7906 |
| Dethloff | 1 | 24 | 6 | 0.1846 | 0.1846 | 0.1846 | 0.1846 | 0.1846 | 0.1846 |
| Dethloff | 2 | 24 | 6 | 0.1846 | 0.1846 | 0.1846 | 0.1846 | — | 0.1846 |
| City | 1, 2, or 4 | 36 | 0 | 0 | 0 | 0 | 0 | 0 for $B=1$ | 0 |

Two-flip reached the exact batch optimum on all 132 cases. Matching tied
two-flip on 102 and lost on 30. The losses were all in the heterogeneous
and near-cutoff synthetic families. The route-derived groups gave equal
values for all evaluated policies. City gave no positive acquisition
value under the tested score laws and fees.

The refined matching method starts two-flip from the matching, empty,
and full batches. It reaches the exact batch optimum on all 132 cases
and keeps the matching upper bound. The script reports its values in a
separate column.

The separate two-group construction in the fleet model also admits an
exact benchmark at larger sizes by symmetry. If $a$ incumbents and $b$
challengers are observed, the gross gain is
$\mathbb E[\min(U_a,V_b)]$ for independent binomial counts with parameters
$(a,1/2)$ and $(b,1/2)$. Summing products of their tail probabilities and
enumerating $(a,b)$ gives the exact best batch in that construction.
Matching attains that optimum at all four tested sizes, including net
savings of 44.8116 with 256 operating routes and 128 standby vehicles.

The benchmark supports treating matching as a way to obtain a candidate
and a certificate. Two-flip local search is the stronger standalone
baseline on the small cases. The next routing benchmark needs signals
that depend on a feasible change to the visit order and pays the actual
incremental travel cost.

## Reproduce

Run from the repository root with the shared engine dependencies installed.

```bash
python3 papers/informative_pickup_routing/compare_fleet_batches.py
```
