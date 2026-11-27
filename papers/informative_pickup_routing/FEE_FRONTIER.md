# When is an informative early visit worth its cost?

The [route-menu model](ROUTE_MENUS.md) charges the full travel and
recourse cost of the chosen early visits. Here we vary a common per-use
standby fee $f$. The score $Z_{ia}$ for option $a$ on route $i$ already
deducts all other standby costs. The choice of early visits precedes
observation, and the dispatch decision follows it. Adding $f$ changes
each standby score to $Z_{ia}-f$ and leaves the primary cost $c_{ia}$
unchanged.

## A fee can create or destroy the value of reordering

There may be an interior interval of fees at which an informative
early visit pays. Take one standby slot and a route with two feasible
options. The original option costs zero and has a known standby score
of 6. The informative option costs 1 more and has standby score 2 or
10, equally likely. Both options have mean score 6, so planning on
means always selects the original option. On observing the informative
score, the dispatcher can use or idle the standby. Its advantage
before taking the better of the two plans is

$$
\tfrac12(2-f)_++\tfrac12(10-f)_+-(6-f)_+-1.
$$

It is positive exactly for $4<f<8$, with maximum saving 1 at $f=6$.
At very low fees the standby is valuable across both outcomes. At high
fees the saving in the favorable outcome is too small to recover the
extra cost. The example extends to larger fleets by adding
routes with zero standby value. Additional slots can be occupied by
routes with deterministic standby scores above every relevant fee and
above the two options' scores.

## Recovering the whole fee interval

For a fixed choice $a$ and a realization $z$, sort the standby scores
as $z_{(1)}\ge \dots\ge z_{(m)}$. The standby benefit is
$\sum_{j\le\min(B,m)}(z_{(j)}-f)_+$. Taking an expectation gives a
piecewise-linear convex benefit, with knots only at the possible
scores. The cost of a fixed plan is nondecreasing and concave in $f$.
Its slope between knots is the expected number of standby vehicles
used. The optimal cost is also nondecreasing and concave, as the
pointwise minimum of these costs.

Within an interval between consecutive support knots every fixed plan
has an affine cost. Solve the route-menu problem at the two endpoints.
If an endpoint's plan is optimal at both ends, it is optimal throughout.
Otherwise, intersect their affine cost lines inside the interval and
solve once at that fee. If the better endpoint plan is optimal at the
intersection, the two plans give the entire lower envelope. In each
half, the chord of the optimal cost matches that plan's line, so
concavity provides the lower bound and feasibility provides the upper
bound. If a third plan beats both at the intersection, recurse on the
two shorter intervals. The intersection step follows dichotomic search
for bicriteria optimization. The implementation uses the branch-and-bound
lower bounds to certify each segment to numerical tolerance.
[Aneja and Nair](https://doi.org/10.1287/mnsc.25.1.73)

The comparison uses **prior-mean planning with the same signal-based
dispatch**. Let $c_i^0=\min_a c_{ia}$ and let
$d_i=c_i^0+\max_a(\mathbb E[Z_{ia}]-c_{ia})$. The prior planner takes
the $B$ largest positive values of $d_i-f$. For those routes it uses the
option maximizing $\mathbb E[Z_{ia}]-c_{ia}$. Remaining routes take
their cheapest primary option. Thus the prior plan can change only
where $f=d_i$. Splitting at these additional knots gives intervals on
which the difference between the two plans' evaluated costs is affine.
Threshold crossings then follow by interpolation. At isolated fee
knots, ties in the prior objective can yield different prior plans, so
reported fee intervals concern their interiors.

## Benchmark

The implementation recovers the analytic band $(4,8)$ and matches
exhaustive enumeration at five points per segment on 24 small menu
instances. The routing experiment reuses the saved Dethloff and City
plans, simulated gamma pickup demands, two standby pool sizes, and
training and held-out draws from the route-menu experiment. It analyzes
all fees from 0 through 20. The five held-out evaluation fees are
0, 5, 10, 15, and 20. Acquisition decisions and dispatch scores are
learned from training draws and then held fixed for each evaluation.

| Group | Cases with some fitted fee interval saving more than 1 | Mean total width of those intervals out of 20 |
| --- | ---: | ---: |
| Dethloff, nominal | 10/16 | 2.880 |
| Dethloff, 10% higher pickup demand | 9/16 | 2.194 |
| City, nominal | 0/8 | 0.000 |
| City, 80% higher pickup demand | 8/8 | 8.866 |

For example, the fitted benefit on Dethloff CON3-6 with one standby
exceeds 1 for fees between approximately 3.631 and 10.996. The
strongest City span is for HANOI-100-2 with two standby vehicles and
elevated demand. The fitted benefit there exceeds 1 between
approximately 0 and 11.790. Results describe these specific fitted
laws and candidate visit sets.

Held-out mean savings per fleet day against prior-mean route planning
are below. The margins are approximate 95% Monte Carlo intervals
conditional on the fitted policies and evaluated plans. They use the
paired savings averaged across plans and standby pool sizes at each
held-out day index. They measure simulation noise within this benchmark.

| Group | Fee 0 | Fee 5 | Fee 10 | Fee 15 | Fee 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dethloff, nominal | 0.517 ± 0.108 | 1.015 ± 0.159 | 0.159 ± 0.075 | 0 | 0 |
| Dethloff, 10% higher pickup demand | 0.258 ± 0.086 | 0.468 ± 0.131 | 0.271 ± 0.075 | 0 | 0 |
| City, nominal | 0 | 0 | 0 | 0 | 0 |
| City, 80% higher pickup demand | 1.450 ± 0.174 | 2.552 ± 0.289 | 0.990 ± 0.288 | 0 | 0 |

Three groups show greater mean savings at fee 5 than at fee 0.
Across the five evaluation fees the average is highest at fee 5 for
each of those groups. An estimate of the best fee would require
separating financial policy choices from the conditions that determine
the actual standby charge.

Run from the repository root with the shared engine dependencies installed.

```bash
python3 papers/informative_pickup_routing/run_fee_frontier.py
```
