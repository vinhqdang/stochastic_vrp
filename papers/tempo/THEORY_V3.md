# TEMPO (direction 1) — theory, corrected after the internal review

Replaces the first version of these notes. Corrections made after
`reviews/internal_review_v3.md`:
(i) the regret statement is now against the library policy at the true
factor, through the performance-difference identity, instead of an
invalid telescoping against the one-step "known" policy;
(ii) the margin condition is stated conditionally on the estimation
error;
(iii) the information per pickup is the gamma Fisher information 1/c^2
about log a, the same for every customer (the earlier "effective count"
was wrong);
(iv) the claimed monotonicity in fleet size is dropped: it is not
supported by the data.

## 1. Model and policies

Fleet of K routes. Route r serves stops k = 1..m_r; state (k, W);
prices H_{r,k} (handoff), R_{r,k} (restock; the load resets to 0),
E_{r,k} (breach, W > B_r). Customers lie in zones z = 1..Z (geographic,
independent of the routes). Day factors b_z = log a_z = mu + eta_z,
mu ~ N(0, s_c^2), eta_z ~ N(0, s_z^2). Pickup of customer i:
p_i ~ Gamma(shape kappa = 1/c^2, mean a_{z(i)} pbar_i).

Library: for each grid value a, pi_a is the BATON policy fitted under
pickups a * pbar; Q_a(s, x) is the expected remaining cost under law a of
taking action x in state s and following pi_a afterwards (computed
exactly on a load grid by `ev/vi.py`), and V_a(s) = Q_a(s, pi_a(s)).

Learning policy L: at each decision state s_k of route r, form an
estimate a_hat_k of the route's factor (from the pickups observed by the
fleet so far, under a pooling rule) and take
x_k = argmin_x Q_{a_hat_k}(s_k, x)  (continue / handoff / restock).

## 2. Regret is driven by the squared estimation error

Fix the true factor a of route r (constant within the day).

Assumption A1 (Lipschitz in the factor). For every decision state s
and action x, |Q_a(s, x) - Q_b(s, x)| <= l(s) |a - b|.

Assumption A2 (margin, conditional on the error). Let
Delta(s) = (second smallest) - (smallest) of Q_a(s, .) and
e_k = a_hat_k - a. For every t >= 0,
P(Delta(s_k) <= t | e_k) <= f(s_k) t  almost surely.

Lemma 1 (one decision). Under A1-A2,
E[ Q_a(s_k, x_k) - min_x Q_a(s_k, x) | e_k ] <= 4 l^2 f e_k^2,
with l = l(s_k), f = f(s_k) (or their suprema over the states reached).
Proof. If x_k is not a minimiser of Q_a(s_k, .), then
Q_a(s_k, x_k) <= Q_{a_hat}(s_k, x_k) + l|e| <= Q_{a_hat}(s_k, x*) + l|e|
<= Q_a(s_k, x*) + 2 l|e|, so the loss is at most 2 l |e_k|, and the same
chain shows that a wrong choice requires Delta(s_k) <= 2 l |e_k|. Hence
loss <= 2 l |e_k| 1{Delta(s_k) <= 2 l |e_k|}; take the conditional
expectation given e_k and use A2. QED.

Theorem 1 (route-day regret). Let J(.) be the expected route cost under
the true law. Then
J(L) - J(pi_a) <= sum_k E[ 4 l(s_k)^2 f(s_k) e_k^2 ],
the sum running over the decision epochs reached by L.
Proof. Performance-difference identity for the finite-horizon MDP with
the benchmark pi_a:
J(L) - J(pi_a) = E_L sum_k [ Q_a(s_k, x_k) - V_a(s_k) ].
Split Q_a(s_k, x_k) - V_a(s_k) = [Q_a(s_k, x_k) - min_x Q_a(s_k, x)]
+ [min_x Q_a(s_k, x) - V_a(s_k)]; the second bracket is <= 0. Apply
Lemma 1 to the first. QED.

Remarks.
- The benchmark pi_a is the library policy at the true factor. The
  "known" reference of the experiments (greedy with respect to Q_a at
  every stop) is a rollout of pi_a, so J(known) <= J(pi_a) by the rollout
  improvement property; Theorem 1 does not bound J(L) - J(known), and
  the experiments report both.
- l(s) and f(s) are computable from the library (finite differences of
  Q_a in a; empirical distribution of Delta over simulated states). The
  bound predicts slope 1 of log regret on log MSE; the measured slope
  across instances and pooling rules is 0.73-0.87 (REVISION_PLAN 8.4).
- Library quantisation adds a term bounded by l times half the grid
  step; it is common to all learning rules.

## 3. Estimation error of the pooling rules

Information. With kappa = 1/c^2, the log-likelihood of one pickup in
b = log a is -kappa b - kappa p / (e^b pbar) + const, whose Fisher
information about b is kappa, the SAME for every customer. Write n_z for
the number of pickups observed so far in zone z, n = sum_z n_z, and
n_r for those on route r. To first order (Gaussian approximation of the
log-likelihood around the truth) a zone's data act as an observation of
b_z with variance c^2 / n_z.

Under the Gaussian model b = mu 1 + eta with prior covariance
Sigma = s_c^2 11' + s_z^2 I and observation precisions
Lambda = diag(n_z / c^2):
- Hierarchical (exact Bayes) error: MSE_hier(z) = [(Sigma^{-1} + Lambda)^{-1}]_{zz}.
  Since the posterior mean minimises posterior expected squared error
  given the data, MSE_hier(z) <= MSE of any other rule that uses the same
  data, for every data configuration (route-only, zone-only and
  full-pool rules are functions of subsets of these data).
- Zone pooling (prior sd s, s^2 = s_c^2 + s_z^2):
  MSE_zone(z) = (1/s^2 + n_z/c^2)^{-1}, ignoring the information from
  other zones.
- Route-only: MSE_route = (1/s^2 + n_r/c^2)^{-1} with n_r <= n_z.
- Full pooling treats all zones as one factor: its estimate is a
  precision-weighted average of the zone means, so its error for zone z
  contains the zone deviations eta_{z'} - eta_z of the other zones, which
  do not average out as n grows.

Proposition 2 (when full pooling stops paying; Z zones observed
equally, n_z = n/Z). As n_z grows,
MSE_full(z) -> s_z^2 (1 - 1/Z) > 0 while MSE_zone(z) ~ c^2 / n_z -> 0;
zone pooling becomes better than full pooling once roughly
n_z > n* = c^2 / (s_z^2 (1 - 1/Z)).
Example: c = 0.3, s_z = 0.2, Z = 2 gives n* = 4.5 pickups per zone; with
s_z = 0.1, n* = 18. Before n*, full pooling's variance reduction wins;
after it, its bias dominates. The hierarchical rule tracks the better of
the two at every n (and is strictly better in between).

Combining with Theorem 1: the regret bound of each rule is the sum over
decision epochs of (local constant) x (its MSE at that epoch), so the
ordering of the rules' regret bounds follows the ordering of their MSE at
the stops where decisions are close. Most decisions that matter occur at
the first two or three stops (REVISION_PLAN 8.5), so the relevant n_z is
small: this is where pooling across vehicles matters most and where full
pooling can still beat zone pooling.

## 4. What is not claimed

- No claim that the value of pooling increases with fleet size: the
  measured correlation of K with the saving is about zero.
- The plug-in rule ignores the value of information (vehicles do not act
  to learn); a full Bayesian average of the value tables was tested and
  adds nothing over the posterior mean.
- Theorem 1 is an upper bound; it does not lower-bound the value of
  pooling.
