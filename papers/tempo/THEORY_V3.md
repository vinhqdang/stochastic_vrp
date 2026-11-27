# TEMPO v3 — theory notes (fleet-pooled learning recourse)

Working notes for the theory section. Notation follows the code
(`scripts/tempo3_hier.py`, `ev/fleet.py`, `ev/vi.py`).

## 1. Model

- Fleet of K routes; route r serves stops k = 1..m_r; load W, slack B_r;
  prices H_{r,k} (handoff), R_{r,k} + continuation from an empty vehicle
  (restock), E_{r,k} (breach). Execution as in BATON.
- Day factors. Each route belongs to a zone z(r). On day d,
  b_z = log a_z = mu + eta_z, mu ~ N(0, s_c^2), eta_z ~ N(0, s_z^2),
  independent. Pickup of customer i (zone z): p_i ~ Gamma(mean a_z pbar_i,
  cv c); deliveries follow the planning law.
- Library: for every a on a grid, pi_a is the BATON policy fitted under
  pickups a * pbar, and U_a(k, W) its exact value (pre-observation, stop k).
- Plug-in learning policy: at stop k, route r takes
  argmin { U_{a_hat}(k+1, W) [continue], H_{r,k}, R_{r,k} + U_{a_hat}(k+1, 0) }
  with a_hat the posterior mean of a_{z(r)} given F_k (all pickups observed
  by the fleet up to stop k), under a chosen pooling rule.

## 2. Plug-in regret is second order in the estimation error

Fix a stop and a state s = (r, k, W). Let Q_a(s, x) be the expected
remaining cost under law a of taking action x now and following pi_a
afterwards, and x*(a) = argmin_x Q_a(s, x). Define the action gap
Delta_a(s) = (second smallest) - (smallest) of Q_a(s, .).

Assumptions.
(A1) Lipschitz in the factor: |Q_a(s, x) - Q_b(s, x)| <= l_s |a - b| for
     every action x (l_s depends on the stop's prices and remaining
     route).
(A2) Margin: conditional on the state, the gap of the TRUE law has a
     density bounded by f_s near zero:  P(Delta_a(s) <= t) <= f_s t.

Proposition 1 (one decision). The expected extra cost of acting at
a_hat instead of a satisfies
  E[ Q_a(s, x*(a_hat)) - Q_a(s, x*(a)) ] <= 2 l_s^2 f_s * 4 * E[(a_hat - a)^2]
                                          = 8 l_s^2 f_s MSE_s.
Proof sketch. The loss is zero unless the ranking flips; a flip requires
Delta_a(s) <= 2 l_s |a_hat - a| (A1), and then the loss is at most
2 l_s |a_hat - a|. So loss <= 2 l_s |e| 1{Delta <= 2 l_s |e|}, e = a_hat - a.
Conditioning on e and using (A2): E[loss | e] <= 2 l_s |e| f_s 2 l_s |e|.
(Constant 8 to be tightened; the point is the square.)

Corollary 1 (route-day). Summing over the decision stops reached,
  J(plug-in) - J(oracle) <= sum_k E[ 8 l_{s_k}^2 f_{s_k} MSE_k ] + (library
  quantisation term) + (continuation mismatch: after stop k the plug-in
  keeps re-estimating, so the comparison is with the oracle that knows a
  from the start; telescoping over stops gives the sum above).
Interpretation: the cost of not knowing the day's factor is driven by the
mean squared error of the factor estimate at the stops where decisions
are close; any pooling rule that lowers MSE at those stops lowers cost,
quadratically.

To verify numerically: l_s and the gap distribution from the library
tables (finite differences of U_a in a); the bound against measured
regret per instance.

## 3. MSE of the pooling rules (Gaussian approximation in log space)

Per pickup, y_i = log(p_i / pbar_i) is approximately b_z + e_i with
var(e_i) ~ c^2 (better: weight by pbar; the sufficient statistic of the
gamma likelihood is sum p_i / pbar_i scaled; use effective counts
n = (sum pbar)^2 / sum pbar^2). Let n_r, n_z, n_F be the effective
pickup counts observed so far on the route, in its zone, in the fleet;
tau^2 = c^2.

- Per-route (prior N(0, s^2), s^2 = s_c^2 + s_z^2):
  MSE_route = (1/s^2 + n_r/tau^2)^{-1}.
- Zone pooling: MSE_zone = (1/s^2 + n_z/tau^2)^{-1}.
- Full pooling (one factor, prior s^2): unbiased only if s_z = 0;
  otherwise the estimate of b_z is a weighted average of zone means, and
  MSE_full = posterior variance + bias variance
  ~ (1/s^2 + n_F/tau^2)^{-1} + s_z^2 (1 - n_z/n_F)^2 + ... (exact form
  from the Gaussian model; the bias term does not vanish as n_F grows).
- Hierarchical (BLUP): the posterior mean of b_z under the two-level
  Gaussian model; its MSE is the posterior variance
  V_hier = Var(b_z | all data), which by Bayes optimality is
  <= min(MSE_route, MSE_zone, MSE_full) for every data configuration
  (each rule is a measurable function of the same data).
  Closed form: with zone means ybar_z (precision P_z = n_z/tau^2),
  marginally ybar_z = mu + eta_z + noise; standard random-effects
  algebra gives the weight on the zone's own mean
  w_z = s_z^2 / (s_z^2 + tau^2/n_z) and on the city estimate 1 - w_z.

## 4. Value of pooling

Combining Sections 2 and 3, the saving of the hierarchical rule over
per-route learning at stop k is bounded below by (constants aside)
  G_k ~ c_k [ MSE_route(k) - V_hier(k) ],
with MSE_route(k) = s^2 tau^2 / (tau^2 + n_r(k) s^2). The bracket is
- increasing in the factor variance (zero when s = 0: nothing to learn),
- increasing in fleet size and zone size (more pooled evidence),
- largest at early and middle stops, where n_r(k) s^2 is comparable to
  tau^2, and vanishing late in the route when the vehicle's own data
  suffice,
- for full pooling, eventually negative when s_z > 0 (bias term), which
  is the failure seen in the "zone only" scenario.

Checks against simulation (to do): per-instance measured saving of
hier over route against sum_k c_k [MSE_route - V_hier] with the
simulated n_r(k), n_z(k), n_F(k); rank correlation across the 10-40
instances and across (s_c, s_z).

## 5. Limitations to state

- The plug-in (certainty-equivalent) rule ignores the value of
  information: a vehicle does not act to learn. The prototype shows the
  full Bayesian average adds nothing over the posterior mean here.
- Zones are given (route groups); unknown or overlapping structure is a
  robustness experiment, not covered by the theory.
- The gamma/log-normal approximation in Section 3 is for the formulas;
  the implemented posterior is exact on the grid.
