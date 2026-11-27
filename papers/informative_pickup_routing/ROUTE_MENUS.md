# Early visits with route-dependent costs and information

An early visit changes the vehicle's position, load, remaining tour, and
the signal available at dispatch. The [fixed-state batch model](FLEET_BATCHES.md)
represents an observation as a paid signal about an unchanged physical
state. This formulation gives every operating route a menu of feasible
early-visit plans, each with its own costs and score distribution.

## Route-menu model

Choose one option $a_i\in A_i$ for each operating route $i$ before seeing
any pickups. The options respect the same dispatch deadline. Option $a$
has expected primary cost $c_{ia}$, inclusive of travel, pickup
recourse, and any acquisition cost. At the deadline it generates
an observed score $Z_{ia}$, the posterior expected saving from giving
that route an identical standby vehicle. The standby score excludes
any acquisition cost already charged in $c_{ia}$. An unused standby
slot has value zero. The route blocks are independent and the score
laws have finite support.

For an assignment $a$, the expected cost is

$$
C(a)=\sum_i c_{ia_i}-\mathbb E[\Phi_B(Z_{1a_1},\ldots,Z_{ma_m})],
$$

where $\Phi_B$ sums the $B$ largest positive scores. The threshold
recurrence from the batch model evaluates any fixed assignment
exactly, up to floating-point arithmetic. The route menus allow
different primary costs and different prior score means across options.

## A lower bound from advance information

At a branch-and-bound node, let $A_i'$ be the options still available
for route $i$. Let $c_i^*=\min_{a\in A_i'}c_{ia}$. Use an independent
uniform variable $U_i$ for each route and couple its option-specific
scores by quantiles, so that $Z_{ia}=Q_{ia}(U_i)$ has the prescribed
marginal law for every $a$. Define

$$
W_i(U_i)=\max_{a\in A_i'}
  \left\{Q_{ia}(U_i)+c_i^*-c_{ia}\right\}.
$$

Granting advance knowledge of all $U_i$ permits selecting each route
option and the standby allocation together. In this relaxation, an
unassigned route uses its minimum-cost option. An assigned route uses
an option minimizing $c_{ia}-Q_{ia}(U_i)$. Compared with the lowest
primary cost, that route has standby value $W_i(U_i)$. The relaxed
expected cost is therefore

$$
\mathrm{LB}(A')=\sum_i c_i^*-\mathbb E[\Phi_B(W_1,\ldots,W_m)]
  \ \le\ \min_{a_i\in A_i'} C(a).
$$

This bound is exact when every menu has one option. The artificial
coupling preserves the objective of every fixed assignment because
that assignment uses one score law per independently sampled route.

**The quantile coupling is strongest within this relaxation.** For any
coupling of the shifted scores within route $i$, the probability that
their maximum is at most $z$ cannot exceed the smallest of their CDFs
at $z$. With a common uniform and quantiles the events are nested, so
the bound on this probability is attained at every $z$. The resulting
$W_i$ is stochastically smallest. If the route blocks remain independent,
the expected sum of the largest positive standby scores is smallest
under these within-route couplings. The relaxed cost is thus a largest
lower bound obtainable by this advance-information construction.

The implementation uses two-choice local search from the original
order and from a plan that optimizes using prior mean scores. It then
branches on a route with multiple available options. At a node limit,
the best feasible plan and the smallest remaining node bound give a
certificate. With no limit it solves the finite menu model to numerical
tolerance. Before searching a routing menu, the improved fixed-state
batch method can also use matching and two-flip together. It starts from
the matching, empty, and full batches and retains the original matching
upper bound.

## Validation

The optimizer matches exhaustive enumeration on 30 random small menu
instances with valid bounds when stopped early. In a separate
eight-route example with four standby slots, each one- or two-choice
move from the prior plan loses money. The original-order and prior-mean
starts coincide. Branch-and-bound improves on their local solution by
0.253125 after expanding 83 nodes and matches
exhaustive enumeration. Refining the original matching method recovers
the exact batch optimum on all 132 cases in the [batch benchmark](BENCHMARK.md).
Two-flip alone also reaches the optimum in those cases.

## Benchmark setup

The routing experiment uses the saved Dethloff and City deterministic
customer assignments and distance matrices. It uses pickup-only loads,
keeping the saved capacities. Each operating vehicle starts empty.
The early stop is either the original first customer or one of up to
four eligible alternatives with the lowest closed-tour travel costs.
The remaining customers keep their original relative order. An early
stop must finish by a common cutoff set to the latest original
first-stop completion time, assuming 25 km/h travel and two minutes
of service. A standby vehicle can take over the residual order from
the depot with an empty load. The primary then returns to the depot.
Both modes unload at the depot if a pickup would exceed capacity.
The model charges travel, activation, and emergency costs. It permits
waiting until the common dispatch time at zero additional cost.

Independent route demand scenarios use gamma pickups with a 0.30
coefficient of variation and a 0.6 within-route copula parameter.
Loads for individual pickups are capped at capacity. A binary signal
indicates whether the first pickup exceeds its training median.
Option costs and posterior standby scores are estimated on 4,096
training scenarios per route. A further 4,096 draws are held out.
The held-out dispatch uses the training posterior scores and is charged
realized travel, activation, and emergency costs. Every method uses
the same held-out draws.

The comparison uses the original first stop and the exact menu plan
whose objective uses only prior mean standby scores. Both plans still
use signal-based dispatch when evaluated. The experiment tests one
and two standby vehicles, the nominal and elevated pickup volumes,
and two activation-cost assumptions. The per-use standby retainer is
20 in the shared cost model and zero in the prepaid-retainer case.
The prepaid case still pays dispatch, transfer, and travel.

All 96 finite training problems closed their optimality gaps to
numerical tolerance. The table reports average held-out cost savings
per fleet day, with each value paired on the same test scenarios.

| Route group | Pickup multiplier | Per-use retainer | Cases with a changed plan | Saving vs original | Saving vs prior-mean plan |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dethloff | 1.0 | 20 | 0/16 | 0.0000 | 0.0000 |
| Dethloff | 1.0 | 0 | 13/16 | 2.6386 | 0.5170 |
| Dethloff | 1.1 | 20 | 4/16 | 0.0392 | 0.0000 |
| Dethloff | 1.1 | 0 | 13/16 | 2.8977 | 0.2576 |
| City | 1.0 | 20 | 0/8 | 0.0000 | 0.0000 |
| City | 1.0 | 0 | 0/8 | 0.0000 | 0.0000 |
| City | 1.8 | 20 | 8/8 | 0.3272 | 0.0000 |
| City | 1.8 | 0 | 8/8 | 6.9557 | 1.4496 |

With the default per-use retainer the prior-mean plan attains the
same held-out value in all four groups. The incremental value of
planning for the signal appears in the prepaid Dethloff and elevated
City groups. These are descriptive comparisons on a small set of
saved plans. Different fees and pickup volumes are sensitivity settings
on the same plans.

## Reproduce

Run from the repository root with the shared engine dependencies installed.

```bash
python3 papers/informative_pickup_routing/run_route_menus.py
```

The route plans and their distance matrices come from the repository.
Pickup scenarios and costs are modeled, so this benchmark measures
performance on those simulated instances. It leaves joint assignment
of customers, adaptive choice of the early visits, and linked demand
between routes to future work.
