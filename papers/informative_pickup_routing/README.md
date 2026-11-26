# Informative Pickup Routing

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

## Reproduce the pilot

```bash
python3 papers/informative_pickup_routing/pilot.py
```

The script exhaustively checks all 256 demand states for three candidate pairs, computes the optimal adaptive and minimax values, and runs 256 simulated trials with a fixed seed. The simulations use synthetic data. Correlation decreases with $M$ in the asymptotic construction. A separate check fixes the correlation at $0.04$. Evaluation on operational data remains open.

Related work includes robust optimization with known marginals and adaptive feature acquisition when features are useful in combination. The open novelty question is whether prior work establishes the same minimax limit on the value of early visits when overload is driven by the sum of pickup demands, together with matching bounds on the number of synchronized observation days. [Distributionally Robust Linear and Discrete Optimization with Marginals](https://doi.org/10.1287/opre.2021.2243) · [When Does Synergy Help Active Feature Acquisition?](https://arxiv.org/abs/2609.32301)
