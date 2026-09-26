# Replies for the Elsevier review portal (CAOR-D-26-01885, revision 1)

One reply per reviewer comment block, to paste into "Reply to comment".
The full point-by-point response, with equations and cross-references,
is in the uploaded response letter; section, table and proposition
numbers refer to the revised manuscript.

---

## Reply to Reviewer 1

We thank the reviewer for a careful and constructive report. All eight points have been addressed; the detailed response is in the attached letter. In brief:

1. Stochastic assumptions and state sufficiency. Assumption 1 now requires only that the running load be a Markov chain with stochastically monotone kernels. This covers independent increments and a Gaussian day-factor model, in which the load is sufficient for the factor. We state where the benchmark model satisfies it only approximately. A new experiment (Section {r:sec:robust}, Table {r:tab:dependence}) varies the demand correlation (rho = 0, 0.3, 0.6, 0.9) and adds a route-level day factor. BATON keeps the highest saving of the implementable policies under every law (one tie within noise, where there is almost nothing to save), and a direct test on 50,000 paths per route finds the continuation value monotone (significant violations in {m:shapeViolZero}, {m:shapeViolSix} and {m:shapeViolNine} of bin pairs at rho = 0, 0.6, 0.9).

2. Shared standby capacity. We now state that the default model is pay-per-use from a pool shared across the operation, and report the pool size each policy would need. A new experiment (Section {r:sec:pool}, Table {r:tab:pool}) reserves a fixed pool per plan, with holding cost, first-come-first-served dispatch across the routes, a fallback menu for refused routes, and a shadow price. A pool reserved for a single plan never pays at our prices (optimal size 0), which supports the pooled pay-per-use model. Given a cap, shadow pricing beats ignoring the cap by {m:poolDethGainZero} (Dethloff) and {m:poolCityGainZero} (city) with no reserved vehicle.

3. Depot-return transition. Section {r:sec:actions} now describes the post-return load, the residual capacity, repeated returns and the extra Bellman term. The reset-to-zero convention is conservative: the exact post-return state is minus the delivered volume. Pricing the exact reset raises the saving (for example from {m:frSalhiThree} to {m:frSalhiExact} on Salhi-Nagy), so our main results for the depot return are conservative.

4. Threshold-policy comparison. We added a position-dependent threshold (one tuned level per stop, which can represent the optimal boundary) and a cost-scaled rollout (a cost-dependent threshold). On the tight Det plans the position-dependent threshold reaches {m:svDetThrK} against {m:svDetHo} for BATON-ho and {m:svDetBaton} for BATON; on the conservative gates neither closes the gap (Tables {r:tab:grand} and {r:tab:large}).

5. Computational fairness. All data-driven policies use the same 1,000 training days; only the two reference programs use more. The protocol is now documented (no separate validation set, which favours the tuned competitors; fixed seeds). A data-budget experiment from 100 to 20,000 days (Section {r:sec:budget}) shows BATON ahead of the tuned threshold at every budget. Table {r:tab:timing} separates offline fitting ({m:timeBatonMs} per route for BATON) from online decisions (about {m:timeOnlineUs} each). The RL baseline, retrained on the same CPU, needs {m:rlTrainMin}-{m:rlTrainMax} minutes.

6. Statistical analysis. The experimental unit is now the instance within a gate (n = 40; routes of a plan share their days). Table {r:tab:stats} gives paired mean differences with 95% bootstrap intervals, win counts, rank-biserial effect sizes and Holm-adjusted tests: against the strongest competitor, +{m:statCompDeltaMin} to +{m:statCompDeltaMax} points, all intervals above zero. Table {r:tab:tail} adds CVaR95 of the daily plan bill and the daily emergency probability. Route-level results are released as supplementary data.

7. Methodological positioning. The introduction now separates the model, the structural results (Propositions 1-4) and the method. The method is explicitly an adaptation of regression-based stopping (Longstaff-Schwartz; Tsitsiklis-Van Roy; Clement et al.), with novelty claimed only for the isotonic step, the valuation of the post-return state and the menu selection. We also toned down several formulations. All four propositions are now machine-checked in the Lean 4 proof assistant (Mathlib), and the formal proofs are released with the code (papers/baton/BatonProofs).

8. Scope of the conclusions. A new paragraph "Scope of the conclusions" (Section {r:sec:conclusion}) states the limits: no time windows, uniform lateness penalties, fixed routes and no integrated planning, shared resources only through the standby pool, and calibrated rather than operational data. The managerial findings are qualified accordingly.

---

## Reply to Reviewer 2

We thank the reviewer for a thorough and helpful report. All major and minor comments have been addressed; details are in the attached letter.

Major 1 (independence vs. rho = 0.6). The monotonicity result (now Proposition {r:prop:monotone}) needs only a stochastically monotone Markov load (revised Assumption 1), not independence; this covers a Gaussian day-factor model. For the benchmark copula it holds approximately, and we now say so. We ran the suggested sweep rho = 0, 0.3, 0.6, 0.9 plus a day factor (Table {r:tab:dependence}). Monotonicity violations are at the false-positive level ({m:shapeViolZero}, {m:shapeViolSix}, {m:shapeViolNine}), so isotonic regression remains justified. BATON stays best among implementable policies, and all policies save more as rho grows.

Major 2 (fresh-start bias). We identify and quantify three biases against a near-exact value from 50,000 independent paths (Table {r:tab:fresh}): (i) suboptimal downstream policy (upward, as you note), (ii) in-sample optimism (downward), and (iii) under dependence, the post-return suffix is correlated with the state that triggered the return (downward on exactly those days). On Salhi-Nagy, (i) and (ii) are small and nearly cancel (error {m:frSalhiBiasIn} of the handoff price in sample, {m:frSalhiBiasOut} out of sample; the saving moves from {m:frSalhiThree} to {m:frSalhiFstar} with the near-exact value). (iii) matters on city routes, and a state-conditional fresh-start value (BATON-cf) removes it (city: {m:frCityThree} to {m:frCityCf} without deployment selection).

Major 3 (action set, third lever). The conclusion now states the boundaries of the action set: actions that end the route at a known price, or reset the load to a known level. A partial handoff is outside it, and we describe what it would require. We now state plainly that the depot return pays where depots are central and plans carry slack (Salhi-Nagy: {m:svSalhiBaton} vs {m:svSalhiHo} handoff-only), and rarely on real urban networks (city: {m:svCityBaton} vs {m:svCityHo}).

Major 4 (quantifying Proposition 2). A new Proposition {r:prop:regret} (machine-checked in Lean 4, like all propositions of the revised paper) shows that the myopic stopping time never exceeds the optimal one and bounds its excess cost by the cost of interventions on days that would have completed cleanly, plus forgone price declines. Table {r:tab:regret} evaluates it: regret {m:regretAll} of reactive cost against a bound of {m:regretBoundAll}. The tuned threshold is within {m:thrGapDet} of BATON-ho on tight Det plans and {m:thrGapCity} on city routes, and {m:thrGapSaa} behind on SAA plans. So a threshold is good enough on tight plans for the handoff decision; the large gaps on conservative gates come mainly from the depot return.

Major 5 (co-optimization). Yes, we consider it the most important direction, because it targets the first-stop risk that no execution policy can reach. It was not implemented because it would remove the common-plan design that isolates the execution contribution, and because it requires the fitted execution cost inside the planner's inner loop. The conclusion now says this.

Major 6 (exchangeability). New experiment with promotion days (Table {r:tab:daytype}). On promotion days BATON saves {m:dtPromoAware} with day-type-specific fits, {m:dtPromoPooled} with a pooled fit and {m:dtPromoStale} with a stale fit. The conclusion gives the practical rule: fit per known day type; otherwise re-fit on a rolling window and monitor drift.

Minor 1 (RL). Section {r:sec:learning} now discusses DQN, PPO and attention/graph architectures and where they could help. The RL baseline was retrained on a consistent route bundle, on the same CPU.

Minor 2 (clairvoyant bound). The explanation now appears in the abstract, introduction, Section {r:sec:algorithms}, Section {r:sec:headline} and the conclusion, and the tables separate reference points from competitors.

Minor 3 (zero pickups). Added deliver-only twins with 25% and 50% zero-pickup customers, re-planned (Table {r:tab:large}). Savings shrink because breaches become rarer (BATON {m:svZpTwentyFiveBaton} and {m:svZpFiftyBaton}); at 50% the position-dependent threshold ties BATON within a few tenths of a point, which we report.

Minor 4 (Figure 2). Split into two figures (Figures {r:fig:explainer} and {r:fig:explainer2}) with print-size fonts.

---

## Reply to Reviewer 3

We thank the reviewer for the careful reading. All five points have been addressed; details are in the attached letter.

1. Standby vs. emergency vehicles. Standby vehicles are reserved capacity under a pre-agreed rate, dispatched before a breach, and they keep customers on schedule. Emergency vehicles are hired on the spot market after a breach, at surge prices, with all downstream customers late. A new paragraph in Section {r:sec:setting} explains this. In the default model the standby pool is shared and billed per use, and we now report the pool each policy would need (median {m:poolBatonMed}, maximum {m:poolBatonMax} vehicles per plan for BATON). A new experiment with a reserved pool (Section {r:sec:pool}) shows that holding standby vehicles for a single plan does not pay at our prices.

2. Holding cost, Assumption 2 and Eq. (7). The standby day rate in the handoff price is the holding cost, charged per vehicle-day used; the reserved-pool experiment models holding costs paid in advance. Assumption 2 now requires only a non-increasing emergency price: no ordering between handoff and emergency prices is needed anywhere, and Eq. (7) is presented as a property of our calibration. A new configuration with the standby rate above the emergency price (Table {r:tab:costsens}, last row) confirms that the policy simply stops using the dominated handoff: BATON saves {m:sbSixtyBaton} there vs {m:sbSixtyThresh} for the tuned threshold.

3. "Lowest cost" vs. oracle. Thank you; the claim now reads "lowest cost of every implementable policy" throughout. The oracle is clairvoyant (it knows the day's demands) and restricted to the handoff lever; it is a bound, not a competitor, and the tables now separate reference points from competitors.

4. BATON in Tables 3 and 5 (now Tables {r:tab:synthetic} and {r:tab:rl}). Both now report BATON. In the synthetic scenarios the depot return is priced flat at half a handoff (collect-then-deliver: {m:ctdBatonFull}). In the RL comparison, BATON-ho is the like-for-like policy ({m:rlBatonHo}) and full BATON saves {m:rlBaton}.

5. City: BATON below DP50k and the oracle. Both are reference points. The oracle knows the future, so its lead cannot be attained. DP50k uses 50 times more data; with the depot return idle on city routes, the gap ({m:svCityBaton} vs {m:svCityDp}) is the finite-sample cost of learning from 1,000 days. The new data-budget experiment shows BATON reaching {m:budCityLarge} with 20,000 days, slightly above DP50k (Section {r:sec:budget}).
