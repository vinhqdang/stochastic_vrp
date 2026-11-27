# Informative Pickup Routing

- **Working title** — Informative Pickup Routing
- **State** — Exploratory research, open for edits
- **Venue** — Undecided
- **Identification result** — A synthetic construction shows that identical marginal
  distributions through order three can conceal which early visits are useful
  before a standby dispatch. The pilot computes exact minimax values and
  simulates identification from synchronized pickup records.
- **Policy optimizer** — The two-visit problem is solved from overload-labeled
  pickup records with a dispatch deadline and travel costs. The implementation
  matches exhaustive enumeration on small instances and learns the optimal
  adaptive policy in a synthetic regional example. An independent holdout can
  certify improvement over an immediate dispatch decision.
- **Fleet formulation** — The [mathematical model](FLEET_MODEL.md) covers
  arbitrary operating and standby fleets. The dispatch stage is a matching
  problem. With independent route signals, singleton information can have
  zero value while joint information has value linear in the number of standby
  vehicles.
- **Fleet batch algorithm** — The [batch model and implementation](FLEET_BATCHES.md)
  select observations with additive costs at a fixed service state. Exact
  batch evaluation and bounds on the optimal batch were checked on 36 small
  synthetic fleets. The algorithm also ran on a synthetic fleet with 256 routes.
- **Benchmark** — [Comparisons](BENCHMARK.md) cover 132 small synthetic and
  route-derived cases, including the published one-standby committing
  policy. Two-flip local search matched the batch optimum on all cases.
  Matching also attained the exact batch optimum on the separate symmetric
  construction with 256 routes.
- **Feasible early visits** — The [route-menu solver](ROUTE_MENUS.md)
  certifies choices across routes using an advance-information bound.
  It improves on two-flip in a complementary synthetic fleet and closes
  the gaps on 96 route-derived problems. Held-out gains beyond the
  prior-mean plan depend on pickup volume and the cost of activating
  standby vehicles.
- **Fee sensitivity** — The [continuous fee analysis](FEE_FRONTIER.md)
  proves that the value of reordering can peak at an interior fee and
  finds the profitable intervals of the fitted route-menu problems.
  In the elevated City scenario, the held-out gain over prior-mean
  planning peaks at 2.552 among the five evaluated fees.
- **Open questions** — Compare with published routing policies under
  matched operational constraints and estimate score laws from
  synchronized operational records.

See [the research note](README.md) for the model, proof, and reproduction
command.
