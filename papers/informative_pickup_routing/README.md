# Informative Pickup Routing

The [fleet formulation](FLEET_MODEL.md) defines the joint routing and
standby allocation problem for arbitrary fleets and gives a lower bound on
the value of pooling information from independent routes. The
[fleet batch algorithm](FLEET_BATCHES.md) chooses observations at a fixed
service state, computes their exact expected value, and bounds the best
possible batch. The [benchmark](BENCHMARK.md) compares it with same-protocol
local search, exact batch and sequential optima, and a published Pandora
policy. The pilot below studies two early observations for a single
dispatch decision.

This pilot measures the value of making two customer visits before deciding whether to dispatch a standby vehicle. There are $M$ candidate pairs. Every customer must be served that day, and all choices of early visits have the same travel cost. The vehicle can choose the second customer after observing demand at the first. On the route that may need support, two customers each have a pickup demand of 0 or 1, and the vehicle has one unit of remaining capacity. An overload occurs when the two pickups sum to more than 1.

## Demand model

Encode each pickup as a sign in $\{-1,+1\}$. Let $x=(z_1,\ldots,z_{2M},u,v)$, where $u$ and $v$ are the two pickups on the route that may need support. In model $j$, customers $2j-1$ and $2j$ form the informative pair. The joint distribution is

$$
P_j(x)=2^{-d}\left[1+\alpha\sum_{r<s}x_rx_s
+\eta z_{2j-1}z_{2j}uv\right],\qquad d=2M+2.
$$

For $\alpha>0$, $0<\eta<1$, and $\alpha d/2+\eta<1$, every demand vector has positive probability. The sum of pairwise products equals $((\sum_r x_r)^2-d)/2$ and is bounded below by $-d/2$. The product of four signs averages to zero when taking any marginal over at most three pickups. As a result, all $P_j$ share the same marginals through order three, and every pair of pickups has correlation $\alpha$. The joint distribution of any one observed pickup and the overload indicator $Y=\mathbf 1\{u=v=+1\}$ is also the same across models.

## Optimal and minimax values

Dispatching costs $1+\alpha$. Waiting costs $4Y$. Before any observations, both actions have expected cost $1+\alpha$. Let $s,t$ be the two observed signs and $r$ the interaction coefficient for the pair visited. The saving from dispatching at outcome $(s,t)$, weighted by the probability of that outcome, is

$$
\frac{2\alpha(s+t)+(r-\alpha^2)st}{4}.
$$

This follows from $4P(Y=1,s,t)-(1+\alpha)P(s,t)$. The value of a branch is the sum of the positive parts of this expression over the possible outcomes of the second visit. Optimize the second visit separately for each outcome of the first visit, then choose the best first visit. This yields the optimal adaptive policy.

To compute the minimax value, consider the uniform mixture of the $M$ models. The optimal value under this mixture is an upper bound on the guaranteed value in the worst case. Randomizing an optimal mixture policy by uniformly permuting the candidate pairs gives it the same expected value in every model, equal to the mixture average. This attains the upper bound. The script computes the minimax value over the $M$ models by setting $r=\eta/M$ for every candidate pair.

With $\alpha=\eta/(16M)$, visiting the mate of a first customer in an informative pair is optimal after either first outcome because $r>4\alpha+2\alpha^2$ for both $r=\eta$ and $r=\eta/M$. This gives

$$
G_{\text{known model}}=\frac{\eta-\alpha^2}{2},
\qquad
G_{\text{minimax}}=\frac{\eta/M-\alpha^2}{2}.
$$

Both quantities measure expected savings relative to making the dispatch decision without observations. With correlation fixed at $0.04$, $M=8$, and $\eta=0.6$, the known-model value is $0.2992$ and the minimax value is $0.05875$. If reordering costs an additional $0.10$, it pays to reorder when the model is known. With access only to the shared marginals, keeping the original route gives a better worst-case outcome.

## How many days of synchronized data are needed?

Each observation day provides pickup demands for the candidate customers and the overload indicator $Y$ for that day. Score pair $j$ by the sample mean of $z_{2j-1}z_{2j}Y$. Its expected score is $\alpha/4$ for an ordinary pair and $(\alpha+\eta)/4$ for the informative pair. For independent days, Hoeffding's inequality and a union bound give the following sufficient sample size for identifying the informative pair with probability at least $1-\delta$.

$$
n\ge \frac{128}{\eta^2}\log\frac{2M}{\delta}.
$$

For a lower bound, let $P_0$ be the model with $\eta=0$ and set $c=1-\alpha d/2>0$. The density of $P_0$ relative to independent fair signs is at least $c$, so the chi-squared bound gives $D_{\rm KL}(P_j\|P_0)\le\eta^2/c$. Fano's inequality gives the following necessary sample size for identifying the informative pair with a worst-case error probability of at most $\delta$, even when all individual pickups are observed each day.

$$
n\ge \frac{c}{\eta^2}\left[(1-\delta)\log M-\log 2\right].
$$

For a fixed error probability and $c$ bounded away from zero, the bounds match in order at $\log M/\eta^2$ synchronized observation days. The pilot assumes that the overload indicator is available, for example by reconstructing it from the two realized pickups. If dispatching a standby vehicle prevents both original pickups from being observed, a separate way to obtain the indicator is needed.

## Optimize early visits from synchronized logs

The optimizer chooses whether to take zero, one, or two early visits under a
dispatch deadline. After the first binary pickup signal, it can stop or select
the second customer based on that signal. Each early visit incurs a known
travel cost. At the end of the chosen visits, it dispatches or waits.

Write $Y$ for the downstream overload label, $L$ for the cost of overload,
and $D$ for the dispatch cost. Dispatching instead of waiting saves $LY-D$.
The optimizer needs only the first-visit outcome probabilities and the
following weighted savings for one- and two-visit histories:

$$
R_i(s)=\mathbb E[(LY-D)\mathbf 1\{X_i=s\}],
\qquad
R_{ij}(s,t)=\mathbb E[(LY-D)\mathbf 1\{X_i=s,X_j=t\}].
$$

After outcome $s$ at $i$, stopping is worth $[R_i(s)]_+$. An eligible
second visit to $j$ is worth
$\sum_t [R_{ij}(s,t)]_+ - P(X_i=s)c_{ij}$, where $c_{ij}$ is the cost of
the additional travel. These values already include the probability of the
first outcome. The algorithm picks the better branch action separately for
each $s$, sums the two values, and subtracts the cost of the first visit.
Comparing the result across first visits and the two immediate dispatch
actions gives the exact optimum for a finite weighted distribution. Using
observation counts as weights gives the empirical optimum. The running
time is $O(nm+m^2u)$ for $n$ uncompressed observation rows, $m$ candidates,
and $u$ distinct rows. Policy search takes $O(m^2u)$ after aggregation.

This establishes sufficiency for the two-visit problem. The pickup-pair
tables must retain the overload label. The preceding construction shows
why separate demand marginals and individual pickup–overload tables
cannot generally substitute for them.

### Finite-sample guarantees

Assume IID days, $0\le D\le L$, a deadline $H$, and travel cost $c$ per
minute. Each feasible policy's saving relative to waiting has range width
at most $W=L+cH$. If $d_i$ second visits are feasible from first visit $i$,
the number of adaptive policies is at most
$K=2+\sum_i(2+4d_i)^2$, summing over eligible first visits. Each first
outcome has two immediate actions or four terminal dispatch rules per
second visit. Hoeffding's inequality and a union bound show that the true
regret of the empirical optimum is at most

$$
2W\sqrt{\frac{\log(2K/\delta)}{2n}}
$$

with probability at least $1-\delta$ over $n$ training days.

For deployment, an independently held-out set gives a separate guarantee.
Compute the learned policy's paired mean saving against each immediate
action, take the smaller, and subtract
$W\sqrt{\log(2/\delta)/(2n_{\mathrm{val}})}$. Each paired difference has
range width at most $W$. A positive lower bound certifies improvement
over the best immediate action at confidence $1-\delta$, provided the
policy was fixed before evaluating the holdout.

### Synthetic regional example

The example has five candidate stops, a noisy signal of which region is
active, and noisy region-specific severity signals. Overload occurs when
two downstream pickup demands exceed one unit of total available capacity.
Travel time is Euclidean, with a 5.5-minute deadline and travel cost of 0.05
per minute. The fixed-order baseline can preselect either second customer
and can still stop after seeing the first signal. Values below are savings
per day relative to the best immediate action, net of travel cost.

| Policy | Value under the known synthetic law |
| --- | ---: |
| Optimal fixed order with early stopping | 0.174956 |
| Optimal adaptive policy | 0.399912 |
| Adaptive policy learned from 4,096 days | 0.399912 |

Both adaptive policies visit the region indicator first and then visit the
west or east customer according to the signal. On 16,384 independent
validation days, the learned policy's one-sided 95% lower bound against
the best immediate action was 0.370681. With a 2-minute deadline, the
optimal fixed and adaptive policies both save 0.008923. The regional
example was designed to exercise both branches. It uses synchronized
binary logs and deterministic prefix travel, with the remaining service
plan outside the optimization.

## Reproduce the results

```bash
python3 papers/informative_pickup_routing/pilot.py
python3 papers/informative_pickup_routing/run_demo.py
```

The first script exhaustively checks all 256 demand states for three candidate pairs, computes the optimal adaptive and minimax values, and runs 256 simulated trials with a fixed seed. Correlation decreases with $M$ in the asymptotic construction. A separate check fixes the correlation at $0.04$. The second script checks the optimizer by exhaustive policy enumeration on 12 small instances, recovers the first script's value for a known model, and reproduces the regional example. Evaluation on operational data remains open.

Related work includes robust optimization with known marginals and adaptive feature acquisition when features are useful in combination. The open novelty question is whether prior work establishes the same minimax limit on the value of early visits when overload is driven by the sum of pickup demands, together with matching bounds on the number of synchronized observation days. [Distributionally Robust Linear and Discrete Optimization with Marginals](https://doi.org/10.1287/opre.2021.2243) · [When Does Synergy Help Active Feature Acquisition?](https://arxiv.org/abs/2609.32301)
