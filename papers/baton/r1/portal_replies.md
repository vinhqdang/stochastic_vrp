# Replies for the Elsevier review portal (CAOR-D-26-01885, revision 1)

One reply per reviewer comment block, to paste into "Reply to comment".
The full point-by-point response, with equations and cross-references,
is in the uploaded response letter; section, table and proposition
numbers refer to the revised manuscript.

---

## Reply to Reviewer 1

We thank the reviewer for a careful and constructive report. All eight points have been addressed; the detailed response is in the attached letter. In brief:

1. Stochastic assumptions and state sufficiency. Assumption 1 now requires only that the running load be a Markov chain with stochastically monotone kernels. This covers independent increments and a Gaussian day-factor model, in which the load is sufficient for the factor. We state where the benchmark model satisfies it only approximately. A new experiment (Section 4.5, Table 9) varies the demand correlation (rho = 0, 0.3, 0.6, 0.9) and adds a route-level day factor. BATON keeps the highest saving of the implementable policies under every law (one tie within noise, where there is almost nothing to save), and a direct test on 50,000 paths per route finds the continuation value monotone (significant violations in 0.2%, 0.1% and 0.0% of bin pairs at rho = 0, 0.6, 0.9).

2. Shared standby capacity. We now state that the default model is pay-per-use from a pool shared across the operation, and report the pool size each policy would need. A new experiment (Section 4.6, Table 12) reserves a fixed pool per plan, with holding cost, first-come-first-served dispatch across the routes, a fallback menu for refused routes, and a shadow price. A pool reserved for a single plan never pays at our prices (optimal size 0), which supports the pooled pay-per-use model. Given a cap, shadow pricing beats ignoring the cap by 15.8% (Dethloff) and 4.1% (city) with no reserved vehicle.

3. Depot-return transition. Section 3.5 now describes the post-return load, the residual capacity, repeated returns and the extra Bellman term. The reset-to-zero convention is conservative: the exact post-return state is minus the delivered volume. Pricing the exact reset raises the saving (for example from 52.2% to 57.9% on Salhi-Nagy), so our main results for the depot return are conservative.

4. Threshold-policy comparison. We added a position-dependent threshold (one tuned level per stop, which can represent the optimal boundary) and a cost-scaled rollout (a cost-dependent threshold). On the tight Det plans the position-dependent threshold reaches 25.8% against 26.1% for BATON-ho and 27.9% for BATON; on the conservative gates neither closes the gap (Tables 2 and 6).

5. Computational fairness. All data-driven policies use the same 1,000 training days; only the two reference programs use more. The protocol is now documented (no separate validation set, which favours the tuned competitors; fixed seeds). A data-budget experiment from 100 to 20,000 days (Section 4.7) shows BATON ahead of the tuned threshold at every budget. Table 14 separates offline fitting (27 ms per route for BATON) from online decisions (about 1 µs each). The RL baseline, retrained on the same CPU, needs 1-3 minutes.

6. Statistical analysis. The experimental unit is now the instance within a gate (n = 40; routes of a plan share their days). Table 3 gives paired mean differences with 95% bootstrap intervals, win counts, rank-biserial effect sizes and Holm-adjusted tests: against the strongest competitor, +2.1 to +10.5 points, all intervals above zero. Table 4 adds CVaR95 of the daily plan bill and the daily emergency probability. Route-level results are released as supplementary data.

7. Methodological positioning. The introduction now separates the model, the structural results (Propositions 1-4) and the method. The method is explicitly an adaptation of regression-based stopping (Longstaff-Schwartz; Tsitsiklis-Van Roy; Clement et al.), with novelty claimed only for the isotonic step, the valuation of the post-return state and the menu selection. We also toned down several formulations. All four propositions are now machine-checked in the Lean 4 proof assistant (Mathlib), and the formal proofs are released with the code (papers/baton/BatonProofs).

8. Scope of the conclusions. A new paragraph "Scope of the conclusions" (Section 5) states the limits: no time windows, uniform lateness penalties, fixed routes and no integrated planning, shared resources only through the standby pool, and calibrated rather than operational data. The managerial findings are qualified accordingly.

---

## Reply to Reviewer 2

We thank the reviewer for a thorough and helpful report. All major and minor comments have been addressed; details are in the attached letter.

Major 1 (independence vs. rho = 0.6). The monotonicity result (now Proposition 4) needs only a stochastically monotone Markov load (revised Assumption 1), not independence; this covers a Gaussian day-factor model. For the benchmark copula it holds approximately, and we now say so. We ran the suggested sweep rho = 0, 0.3, 0.6, 0.9 plus a day factor (Table 9). Monotonicity violations are at the false-positive level (0.2%, 0.1%, 0.0%), so isotonic regression remains justified. BATON stays best among implementable policies, and all policies save more as rho grows.

Major 2 (fresh-start bias). We identify and quantify three biases against a near-exact value from 50,000 independent paths (Table 11): (i) suboptimal downstream policy (upward, as you note), (ii) in-sample optimism (downward), and (iii) under dependence, the post-return suffix is correlated with the state that triggered the return (downward on exactly those days). On Salhi-Nagy, (i) and (ii) are small and nearly cancel (error +0.50% of the handoff price in sample, +0.09% out of sample; the saving moves from 52.2% to 51.9% with the near-exact value). (iii) matters on city routes, and a state-conditional fresh-start value (BATON-cf) removes it (city: 5.6% to 9.4% without deployment selection).

Major 3 (action set, third lever). The conclusion now states the boundaries of the action set: actions that end the route at a known price, or reset the load to a known level. A partial handoff is outside it, and we describe what it would require. We now state plainly that the depot return pays where depots are central and plans carry slack (Salhi-Nagy: 53.0% vs 42.3% handoff-only), and rarely on real urban networks (city: 10.6% vs 10.6%).

Major 4 (quantifying Proposition 2). A new Proposition 3 (machine-checked in Lean 4, like all propositions of the revised paper) shows that the myopic stopping time never exceeds the optimal one and bounds its excess cost by the cost of interventions on days that would have completed cleanly, plus forgone price declines. Table 8 evaluates it: regret 3.8% of reactive cost against a bound of 7.7%. The tuned threshold is within 1.0% of BATON-ho on tight Det plans and 0.4% on city routes, and 3.0% behind on SAA plans. So a threshold is good enough on tight plans for the handoff decision; the large gaps on conservative gates come mainly from the depot return.

Major 5 (co-optimization). Yes, we consider it the most important direction, because it targets the first-stop risk that no execution policy can reach. It was not implemented because it would remove the common-plan design that isolates the execution contribution, and because it requires the fitted execution cost inside the planner's inner loop. The conclusion now says this.

Major 6 (exchangeability). New experiment with promotion days (Table 10). On promotion days BATON saves 30.9% with day-type-specific fits, 29.4% with a pooled fit and 28.4% with a stale fit. The conclusion gives the practical rule: fit per known day type; otherwise re-fit on a rolling window and monitor drift.

Minor 1 (RL). Section 4.4 now discusses DQN, PPO and attention/graph architectures and where they could help. The RL baseline was retrained on a consistent route bundle, on the same CPU.

Minor 2 (clairvoyant bound). The explanation now appears in the abstract, introduction, Section 3.6, Section 4.2 and the conclusion, and the tables separate reference points from competitors.

Minor 3 (zero pickups). Added deliver-only twins with 25% and 50% zero-pickup customers, re-planned (Table 6). Savings shrink because breaches become rarer (BATON 4.9% and 3.1%); at 50% the position-dependent threshold ties BATON within a few tenths of a point, which we report.

Minor 4 (Figure 2). Split into two figures (Figures 2 and 3) with print-size fonts.

---

## Reply to Reviewer 3

We thank the reviewer for the careful reading. All five points have been addressed; details are in the attached letter.

1. Standby vs. emergency vehicles. Standby vehicles are reserved capacity under a pre-agreed rate, dispatched before a breach, and they keep customers on schedule. Emergency vehicles are hired on the spot market after a breach, at surge prices, with all downstream customers late. A new paragraph in Section 3.1 explains this. In the default model the standby pool is shared and billed per use, and we now report the pool each policy would need (median 0, maximum 7 vehicles per plan for BATON). A new experiment with a reserved pool (Section 4.6) shows that holding standby vehicles for a single plan does not pay at our prices.

2. Holding cost, Assumption 2 and Eq. (7). The standby day rate in the handoff price is the holding cost, charged per vehicle-day used; the reserved-pool experiment models holding costs paid in advance. Assumption 2 now requires only a non-increasing emergency price: no ordering between handoff and emergency prices is needed anywhere, and Eq. (7) is presented as a property of our calibration. A new configuration with the standby rate above the emergency price (Table 15, last row) confirms that the policy simply stops using the dominated handoff: BATON saves 36.8% there vs 1.4% for the tuned threshold.

3. "Lowest cost" vs. oracle. Thank you; the claim now reads "lowest cost of every implementable policy" throughout. The oracle is clairvoyant (it knows the day's demands) and restricted to the handoff lever; it is a bound, not a competitor, and the tables now separate reference points from competitors.

4. BATON in Tables 3 and 5 (now Tables 5 and 7). Both now report BATON. In the synthetic scenarios the depot return is priced flat at half a handoff (collect-then-deliver: 69.1%). In the RL comparison, BATON-ho is the like-for-like policy (32.9%) and full BATON saves 33.7%.

5. City: BATON below DP50k and the oracle. Both are reference points. The oracle knows the future, so its lead cannot be attained. DP50k uses 50 times more data; with the depot return idle on city routes, the gap (10.6% vs 12.1%) is the finite-sample cost of learning from 1,000 days. The new data-budget experiment shows BATON reaching 12.9% with 20,000 days, slightly above DP50k (Section 4.7).
