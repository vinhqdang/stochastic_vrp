# Choosing route observations before standby dispatch

This subproblem fixes the physical dispatch state in the
[fleet model](FLEET_MODEL.md). There are $m$ routes and $B$ identical
standby vehicles. Each route offers one optional local observation with
a known cost $c_i\ge0$. The dispatcher chooses a set $S$ of observations
in advance and sees all selected outcomes before allocating the standby
vehicles. Acquisition costs are additive and simultaneous acquisitions
are feasible. On a route that is observed, $Z_i$ is the posterior expected
net saving from assigning a standby vehicle. On an unobserved route,
$Z_i=\mu_i$, its prior expected saving. The observed scores have finite
supports and are independent across routes.

Write $\Phi_B(z)$ for the sum of the $B$ largest positive entries of $z$.
The objective is to maximize

$$
f(S)-c(S)
=\mathbb E[\Phi_B(Z)]-\Phi_B(\mu)-\sum_{i\in S}c_i.
$$

The prototype takes score distributions and observation costs as input.
Travel and the customer service plan are held fixed. The batch problem
can value candidate observations supplied by a routing optimizer.

## Exact evaluation of a batch

For a realized vector $z$, let $N(t)=\#\{i:z_i>t\}$. Then

$$
\Phi_B(z)=\int_0^\infty \min(B,N(t))\,dt.
$$

At a fixed threshold $t$, $N(t)$ is a sum of independent Bernoulli
variables with probabilities $\Pr(Z_i>t)$. A dynamic program maintains
the mass at counts $0,\ldots,B-1$ and an absorbing count $B$ for all
larger values. The expected truncated count is constant between
consecutive positive support values. Summing over those intervals
evaluates the integral exactly, up to floating-point arithmetic. For
$L$ distinct positive support values this takes $O(mBL)$ time after
sorting and indexing the support events. It avoids enumerating a
Cartesian product of route outcomes.

## Choosing a batch and bounding the optimum

Add enough zero-score dummy routes to represent unused standby slots.
Let $I$ be the $B$ routes assigned standby vehicles using prior scores,
including dummy routes as needed. For a realization $z$, the increase
over keeping those assignments is the value of a maximum-weight
bipartite matching from $I$ to its complement with edge weights
$(z_j-z_i)_+$. Each edge represents replacing one incumbent $i$ with one
challenger $j$. Using at most one swap per endpoint yields a feasible
allocation, and an optimal allocation can be obtained by matching the
routes it adds with the incumbents it drops. Thus

$$
f(S)=\mathbb E\left[\max_M \sum_{(i,j)\in M}(Z_j-Z_i)_+\right].
$$

Here $\mathbb E[\sum_{i\in I}Z_i]=\sum_{i\in I}\mu_i$ by iterated
expectation. Let $w_{ij}(S)=\mathbb E[(Z_j-Z_i)_+]$. Define the potential
exchange graph by including $(i,j)$ if the maximum possible score at $j$
exceeds the minimum possible score at $i$. The prior mean lies within
each route's support interval, so this graph includes every swap that
can have positive value under any observation set. Write $\Delta$ for
its maximum degree and $H(S)=\max_M\sum_{(i,j)\in M}w_{ij}(S)$. Then

$$
H(S)\le f(S)\le\Delta H(S).
$$

For the first inequality, a fixed matching can always be used as a lower
bound on the best realized swaps. For the second, the sum of all edge
weights bounds the realized matching objective. The edges of a bipartite
graph can be partitioned into $\Delta$ matchings, each with expected
weight at most $H(S)$.

For $\lambda>0$, define $J_\lambda=\max_S[H(S)-\lambda c(S)]$.
It can be computed by one maximum-weight matching. On each edge,
enumerate the possible observations of its two endpoints and choose
the highest expected swap value less the scaled acquisition cost.
Matched edges are disjoint, so each cost is charged once. Unmatched
observations can be omitted since they cannot improve a fixed matching.
An optimizer of $J_1$ supplies a candidate batch $\widehat S$.

Let $F^*=\max_S[f(S)-c(S)]$. For $\Delta\ge1$ the returned certificate is

$$
f(\widehat S)-c(\widehat S)
\ \le\ F^*\
\le\min\bigl\{\Delta J_{1/\Delta},\, f(\text{all routes})\bigr\}.
$$

The first upper bound follows by subtracting $c(S)$ from $\Delta H(S)$.
The second follows from nonnegative acquisition costs and the value of
having access to all signals. The batch algorithm evaluates both its
chosen batch and the all-observation bound by the threshold recurrence
above. When $\Delta=1$, the certificate closes and the batch is globally
optimal. When $\Delta=0$, observing nothing is optimal. For dense graphs
the upper bound can be loose.

## Reproduce

Run from the repository root with SciPy installed.

```bash
python3 papers/informative_pickup_routing/run_fleet_batches.py
```

The script checks the value of full observation by direct enumeration of
score realizations on 36 small fleets. It checks the batch certificate
against the exact optimum over all subsets on those same fleets. For
the 27 fleets with positive optimal net value, the selected batch
achieves a mean of 93.59% of the exact optimum, with a minimum of 69.85%.
The sample is a seeded synthetic validation set.

The second experiment uses the complementary two-group construction
from [the fleet model](FLEET_MODEL.md). Each observation costs $1/16$.
Greedy selection stops when every single additional observation has
nonpositive marginal net value.

| Operating routes | Standby vehicles | Selected observations | Achieved net saving | Upper bound on optimal net saving | Singleton greedy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 8 | 0.9531 | 1.4531 | 0 |
| 32 | 16 | 32 | 4.8804 | 6.8804 | 0 |
| 128 | 64 | 128 | 21.7476 | 29.7476 | 0 |
| 256 | 128 | 256 | 44.8116 | 60.8116 | 0 |

These results cover a single dispatch epoch with independent route
signals. Learning the score laws from operational records and optimizing
the actual early travel remain open.
