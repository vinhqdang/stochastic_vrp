# Internal referee report: TEMPO v3 (fleet-pooled hierarchical learning recourse)

Reviewer role: TR-B / Transportation Science referee, pre-manuscript. Scope: `scripts/tempo3_hier.py`, `scripts/tempo3_proto.py`, `ev/fleet.py`, `ev/vi.py`, `results/tempo3/*.csv`, `THEORY_V3.md`, `reviews/novelty_bayes_fleet.md`, REVISION_PLAN.md section 8. No repository file was changed except this report. Analysis scripts are in the session scratchpad.

## Verdict

The numerical result is real: pooling the factor evidence across vehicles lowers mean recourse cost relative to per-route learning, by roughly 1 to 4 cost units (about 2 to 6 percent) in most regimes. The experiment is mostly clean on common random numbers. But as it stands it is not a TR-B/TS contribution. The estimator is textbook, the headline comparison is a Bayes-optimality tautology inside a well-specified generator, zones are given and circular, the theory has a gap in its central corollary and a wrong effective-count formula, and the model has no empirical support. Details below, ranked.

## Paired statistics (requested)

Paired differences of daily cost, hier minus comparator (negative favours hier), from `hier_{Dethloff,City,SalhiNagy}.csv`. "Inst CI" is a bootstrap over instances of per-instance mean differences (2000 resamples). "Crossed CI" resamples instances and days jointly (1000 resamples), which is the correct interval because day-level factor draws are shared across instances (finding M2).

Dethloff (10 instances, 150 days, K = 4 and Z = 2 for every instance):

| (s_c, s_z) | comparator | mean diff | inst CI | crossed CI | hier better |
|---|---|---|---|---|---|
| (0, 0.3) | route | -2.77 | [-3.52, -2.00] | [-4.39, -1.40] | 10/10 |
| (0, 0.3) | zonepool | 0.00 | identical by code | | 0/10 |
| (0.1, 0.3) | route | -2.54 | [-3.91, -1.29] | [-4.57, -0.68] | 9/10 |
| (0.1, 0.3) | zonepool | -0.01 | [-0.18, 0.17] | [-0.40, 0.37] | 6/10 |
| (0.2, 0.2) | route | -2.76 | [-3.77, -1.67] | [-4.63, -1.00] | 9/10 |
| (0.2, 0.2) | zonepool | -0.59 | [-0.98, -0.18] | [-1.47, +0.21] | 8/10 |
| (0.3, 0) | route | -3.91 | [-5.48, -2.29] | [-6.29, -1.55] | 10/10 |
| (0.3, 0) | fullpool | 0.00 | identical by code | | 0/10 |
| (0.3, 0) | zonepool | -1.32 | [-2.10, -0.58] | [-2.93, +0.10] | 8/10 |

City (19 instances, 100 days, K from 6 to 23):

| (s_c, s_z) | comparator | mean diff | inst CI | crossed CI | hier better |
|---|---|---|---|---|---|
| (0, 0.3) | route | -1.90 | [-2.88, -1.12] | [-3.60, -0.48] | 17/19 |
| (0.1, 0.3) | route | -2.09 | [-2.97, -1.31] | [-3.53, -0.86] | 18/19 |
| (0.1, 0.3) | zonepool | **+0.11** | [-0.03, +0.25] | [-0.11, +0.38] | 3/19 |
| (0.2, 0.2) | route | -0.95 | [-1.50, -0.45] | [-2.02, +0.03] | 16/19 |
| (0.2, 0.2) | zonepool | -0.29 | [-0.56, -0.05] | [-0.84, +0.17] | 9/19 |
| (0.2, 0.2) | fullpool | -0.98 | [-2.05, -0.17] | [-2.37, +0.05] | 13/19 |
| (0.3, 0) | route | -1.41 | [-2.28, -0.53] | [-2.95, -0.05] | 13/19 |
| (0.3, 0) | zonepool | -0.13 | [-0.66, +0.34] | [-1.16, +0.84] | 8/19 |

Salhi-Nagy (14 instances, 100 days):

| (s_c, s_z) | comparator | mean diff | inst CI | crossed CI | hier better |
|---|---|---|---|---|---|
| (0, 0.3) | route | -1.47 | [-2.48, -0.49] | [-3.00, -0.05] | 12/14 |
| (0.1, 0.3) | route | -1.20 | [-2.08, -0.33] | [-2.62, +0.29] | 12/14 |
| (0.1, 0.3) | zonepool | -0.05 | [-0.22, +0.10] | [-0.44, +0.20] | 4/14 |
| (0.2, 0.2) | route | -2.05 | [-3.32, -0.86] | [-3.93, -0.51] | 11/14 |
| (0.2, 0.2) | zonepool | -0.11 | [-0.42, +0.19] | [-0.75, +0.49] | 6/14 |
| (0.3, 0) | route | -1.96 | [-3.16, -0.84] | [-3.73, -0.31] | 13/14 |
| (0.3, 0) | zonepool | -0.69 | [-1.14, -0.21] | [-1.68, +0.27] | 12/14 |

Reading: pooled versus per-route (the Florio-style baseline) is robustly significant in 11 of 12 family-by-regime cells even under the crossed bootstrap. "Hierarchical versus the best simple pooling rule" is not significant in any mixed regime under the crossed bootstrap, and in several it is a coin flip across instances (City (0.1, 0.3): zonepool is better, 3/19 for hier). The 8.1 and 8.3 claim "best or tied in every regime" is true to within noise, but see F1.

Fraction of the fixed-to-oracle gap closed, from means (all families, regimes): route 0.46 to 0.83, zonepool 0.71 to 0.93, hier 0.71 to 0.97.

## FATAL

**F1. The hierarchical method is not tested against an independent competitor; "best or tied" is partly by construction, and the remaining margin is not significant.**
`tempo3_hier.py:79-84`: when s_z = 0 `hier_means` returns the fullpool posterior (same prior N(0, s_c), same summed likelihood); when s_c = 0 it returns the zonepool posterior. So in the "city only" and "zone only" cells hier and fullpool/zonepool are the same computation (difference exactly 0.00 in the table above). They are tied by code, not by experiment. In the two genuinely mixed regimes the gain over zonepool is 0 to 0.6 cost units (0 to 0.6 percent), CIs cross zero under the crossed bootstrap, and City (0.1, 0.3) shows zonepool ahead. Statistical-significance-wise the only supported claim is "any zone-aware pooling beats per-route learning given correct zones". That is a much weaker paper than "hierarchical partial pooling". Fix: remove the degenerate scenarios from the headline, add a graded grid of s_c/s_z ratios, and show hier beats the better of {zonepool, fullpool} at the oracle-best choice per regime; or reframe so the contribution does not rest on hier versus zonepool.

**F2. Within a well-specified generator, the headline ordering is Bayes optimality of the data-generating model, i.e. a tautology.**
The generator (`run_day:133-136,152`) draws exactly the two-level log-normal model, the gamma noise with known shape, and independence across customers; `hier` uses the same model with the true hyperparameters and the exact likelihood (`loglik`, verified correct: `-SHAPE*BG - p/(e^BG pbar CV^2)`, constants dropped). Any estimator with the same data is dominated in posterior risk (THEORY_V3 section 3 makes this point itself). The experiment therefore verifies arithmetic, not robustness. All three misspecification tests a referee will ask for are missing: (a) wrong noise law (CV unknown, heavier tails, Poisson-like variance proportional to mean; see M3), (b) non-Gaussian or heavy-tailed day factors, regime shifts, drift within the day (see R3), (c) zones that are not exact partitions of routes. `hier_wrong` (permuted route labels, `tempo3_hier.py:212`) is a single adversarial misspecification (destroys all spatial information at fixed zone count) and in (s_c = 0.3, s_z = 0) it is identical to hier by code (s_z = 0 branch ignores zones), which is why the table shows hier_wrong = hier = 51.20 there. Fix: run a misspecification grid and report the loss relative to the best simple rule (route, fullpool), and make "robust to misspecification" a measured claim.

**F3. The theory's central corollary is not valid as stated, and the effective-count formula is wrong for the implemented model.**
See section "Theory check" below: the Corollary 1 telescoping needs performance-difference with the benchmark's own Q-function and the learner's state distribution; the decision rule is a policy-improvement step over a non-optimal fixed policy, so neither "oracle" nor "regret" is what the corollary bounds. Proposition 1's proof sketch needs a margin condition conditional on (state, error) jointly, which is not implied by (A2). In the model as implemented every pickup carries the same Fisher information about log a (k = 1/CV^2 per observation, independent of pbar); THEORY_V3 section 3's "n = (sum pbar)^2 / sum pbar^2" is wrong for the gamma model (it is the effective size for the ratio-of-sums statistic, not for the exact posterior). Without a fix, the "value of pooling" section is not a theorem and its predicted comparative statics (increasing in fleet size) are already contradicted by the data (M5).

## MAJOR

**M1. The "oracle" is not an upper bound, and the label overstates what it is.**
`run_day:177-196`: the oracle uses `lib.get(r, quantize_a(a_true))` and the one-step rule `ho < cont`, `rest < cont` with exact policy values U_a of a *fixed fitted* policy. This is a rollout-style policy improvement, not the optimal policy under a. Evidence: the "oracle" is worse than the a = 1 policy on 15.2 percent of days (and better on 24.1 percent), and in 44 of 172 (instance, regime) mean cells hier is below the oracle mean (e.g. City HANOI-100-3 (0.1, 0.3): hier 22.87 versus oracle 23.86). Library policies for different grid values a are fitted with different seeds (`fleet.py:272`, N = 1000 days), so the oracle also carries fitting noise that hier inherits asymmetrically. Quantisation caps a at [0.5, 2.0] (`A_GRID`), and for (0.1, 0.3) the log-total sd 0.32 puts about 5 percent of zone-days outside the grid (truncated for the oracle as well as for learners, so no unfair advantage, but "true a" is not what is plugged in). Fix: rename to "informed benchmark", add a real bound (clairvoyant on realised pickups of the remaining stops, or an exact Bayes-adaptive DP on a few small routes), and report "gap closed" against that. Also state that routes whose fitted policy is poor are mischaracterised by the gap statistic.

**M2. Statistical reliability is overstated; instances are not independent replicates and days are heavy-tailed.**
`run` line 217: `rng = default_rng(11)` is re-created for every (instance, regime), and `run_day` draws mu then eta first. So day d has the same city shock in every instance and the same scaled shock across regimes with equal s_c. The "10-40 instances" are therefore not independent in the dominant latent variable. Costs are extremely skewed: at (0.2, 0.2) the per-day cost sd is 82-96 against a mean of 70-80, 62 percent of days have fixed = oracle (nothing at stake), and 5 percent of days carry 55 percent of the fixed-to-oracle gap. Intervals must be computed with a crossed (instance x day) bootstrap, as above; with it, 8 of the 12 family-by-regime "hier vs best pooling alternative" comparisons are inconclusive. Hier equals route on 76 percent of days and hier equals oracle on 75 percent: the entire result lives on about one day in four. 60 to 150 days is thin for that tail; use independent seeds per instance and at least 500 days, report CVaR (these are recourse-risk problems) and the per-day share of nonzero-difference days.

**M3. Fairness of baselines: mostly fine, but the strong rival rules are missing and the noise model flatters learning.**
Checks: (i) common random numbers are correct: `base` gammas and (mu, eta) are drawn once per day and every policy sees them; no policy consumes `rng`. (ii) Priors: route/zonepool/fullpool use the true total prior sd (`run_day:135`), hier uses the true (s_c, s_z), eb uses estimated; this gives the baselines the true hyperparameters, which favours them, so the comparison is conservative on information. (iii) Decision rule identical across policies (`run_day:191-200`). (iv) Weak or missing: no shrinkage-ratio rule (a_hat = (sum p + lambda pbar)/(sum pbar + lambda) with tuned lambda), no fleet-wide tuned-threshold recourse from the BATON paper's pi1-pi3 family, no re-optimisation trigger from section 7 (the section 7 trigger result is silently dropped), and the full-pool baseline puts prior sd s_tot on the *shared mu* even when s_z > 0 (a weak choice, which is why fullpool looks bad in zone-only regimes). (v) Per-vehicle information equals the pooled fleet only because routes are processed in stop-index lockstep (`run_day:147`, all routes at stop k simultaneously); real clocks differ by route length and distance, so how much fleet evidence is available at a given decision is exaggerated and must be modelled with timestamps. (vi) Noise: CV = 0.3 fixed and independent of pbar means each pickup has the same information; real pickups are count-like (variance proportional to mean), so small customers are far less informative and pooling is worth more but late-route/own-evidence arguments shift. Fix: add the shrinkage-ratio and tuned-threshold baselines, an asynchronous clock, and a Poisson-gamma noise variant.

**M4. Zones are circular and the fleet is tiny where it was most thoroughly tested.**
`zones_of` (`tempo3_hier.py:45-54`) clusters routes of the *same plan* by mean customer distance with Z = round(K/3); the generator then draws zone factors on exactly those clusters and the estimator knows them. Zones here are an input and an oracle partition. All ten Dethloff instances have K = 4 and Z = 2 (CSV), i.e. two zones of two routes, so "hierarchical" means a two-group random effect with two vehicles per group; City has K from 6 to 23 and Z from 2 to 8, and there the zonepool-hier difference is the smallest. Real zone effects are spatially smooth fields, plans change daily (so "zone = route group" does not persist across days, which is what the method-of-moments history in `history_estimate` assumes: routes with the same pbar vectors observed on 30 to 120 past days), and the number of zones is a modelling decision. Fix: generate factors from a spatial Gaussian field over customer coordinates with a range parameter, define zones as grid cells, run a mismatch grid (true range versus assumed cell size, true Z versus assumed Z), and estimate zones/hyperparameters from the live day plus history of *different* plans.

**M5. The proposed comparative statics are not supported by the data.**
THEORY_V3 section 4 claims the pooling saving is increasing in fleet size. Across instances the correlation between K and the hier-over-route saving is -0.07 at (0.2, 0.2) and +0.18 at (0.3, 0); the K >= 10 group has gain 2.0 (n = 6) versus 1.9 for K <= 6 at (0.2, 0.2). This is weakly informative (K is confounded with route length) but it contradicts the headline "value of pooling in K". Also in the zone-only regime the benefit saturates at zone size, not fleet size. A value-of-pooling statement must be proved for the actual structure (zone count, zone size, route length) and checked per instance, as THEORY_V3 itself proposes; this "to do" check is not done.

**M6. Contribution after Florio et al. 2023 (answer to question 2).**
What a referee will call trivial: "sharing the estimate across vehicles"; hierarchical (random-effects) Gaussian shrinkage is a first-year-statistics tool; QMDP/posterior-mean plug-in is standard; plug-in regret is quadratic in the estimation error is standard (margin conditions, Mammen-Tsybakov; certainty-equivalence in newsvendor/inventory with Bayesian updating, Scarf/Lariviere-Porteus lineage). The novelty-check document already says the update and the pooled gain "may look incremental". Evidence on non-triviality that would help: (a) a sharp, nonobvious *negative* or *phase-transition* result (pooling can hurt: bias under zone effects, which you do show, but prove the threshold in s_z/s_c and zone mix, with the actual decision-loss rather than MSE); (b) a *decision-relevant* statement: because the loss is quadratic only at near-tie states, the value of better estimates is concentrated at a measurable set of states, so one can predict where pooling pays and where it does not (that is already half of THEORY_V3); (c) a fleet-level coupling, e.g. information-aware routing or standby dispatch that exploits early pooled evidence to pre-position vehicles, which route-level learning (Florio) cannot do; (d) a guarantee that the pooled posterior's regret versus the informed benchmark is O(1/n) in observed pickups with explicit constants under the discrete decision structure; (e) a field-data validation of the factor structure (R1). As it stands the paper is "Florio et al. plus fleet", which is exactly the framing the novelty review warns about; for TR-B the realistic bar is a general theory or an empirical factor structure, not a prototype ranking.

## MINOR

**m1. Library noise and quantisation.** `quantize_a` rounds to steps of 0.1, comparable to the posterior sd after several stops at s = 0.15; the decision error from rounding is not separated from the estimation error. Also a posterior mean on the a-scale is quantised, whereas the policy's best argument might be the median or an optimal predictor under the asymmetric loss; try the decision-optimal point estimate.

**m2. Exact-zero hyperparameter branching.** In `hier_eb*`, the clamped method-of-moments estimate is exactly 0 in 47 to 56 percent of runs in single-factor regimes (sz_hat30 == 0 in 47 percent of (0.3, 0), sc_hat30 == 0 in 56 percent of (0, 0.3)), which switches discontinuously to the fullpool/zonepool posterior (`hier_means:79-84`). It happens to be benign here, but it is a hard switch where a floor or a posterior over hyperparameters would be principled; the 30-day sd of the estimates (0.02 to 0.04) is also optimistic because the history is simulated from the true model (`history_estimate` draws days from `s_c, s_z` directly), not estimated from realised observation streams with early-ending routes.

**m3. Method-of-moments estimator.** Correct algebra (within-day spread = s_z^2 + mean v; variance of the daily zone mean = s_c^2 + (s_z^2 + v_bar)/Z; v_z = CV^2 sum pbar^2 / (sum pbar)^2 is the variance of the ratio of sums, first-order for the log). Unweighted across zones, ignores the log-ratio bias (-v/2), and `B.mean(axis=1).var(ddof=1)` is inefficient at Z = 2. Use REML on the live gamma likelihood or a hierarchical Bayes with weak hyperpriors.

**m4. Mean of the factor.** a = exp(mu + eta) with mu, eta centred at 0 gives E[a] = exp((s_c^2 + s_z^2)/2) > 1 (1.05 at s = 0.3, 1.09 at s_tot = 0.42), so the planning model (pbar) is biased on average; fixed/BATON at a = 1 is handicapped by roughly the same amount. Centre the factor so E[a] = 1 or state it.

**m5. Grid edges.** `BG` spans log 0.35 to log 2.8; `conv_gauss` zero-pads at the edges, so likelihood mass outside is silently lost (`np.convolve(..., "full")[n:n+len]`). For s_z = 0.3 and the grid half-width about 1.05 this is a small bias, larger for large s. The library range (0.5 to 2.0) is narrower still than the posterior grid.

**m6. Information from truncated routes.** After a handoff or breach (`alive[r] = False`) no further pickups of that route are observed, though in reality the covering vehicle observes them; with early exits the effective pool shrinks in a way the plan does not model (and `Lr` correctly retains past evidence). Mention or model it.

**m7. Duplicated and stale tables.** `proto.csv` (`tempo3_proto.py`) draws the true a on the library grid (so oracle quantisation is lossless there) while the hier experiments use continuous a; the plan section 8 table mixes the two. The `proto.csv` full-Bayes versus plug-in finding (no gain from averaging) was only run at one dispersion and 10 instances.

## Q4: Bugs in `tempo3_hier.py` (result of the line-by-line read)

No fatal numerical bug. Verified correct: the gamma log-likelihood in `loglik`; the two-level posterior in `hier_means` (Mz[z] is the zone-z likelihood convolved with N(0, s_z), the leave-one-out Mz.sum(0)-Mz[z] and a second convolution give p(b_z | other zones) before multiplying by Lz[z]; the constant offset `Lz[z].max()` cancels); the handling of early-ending routes (log-likelihood of observed pickups is retained in `Lr` and routes with `pbar = 0` add nothing); CRN. Items to fix or document:

1. `history_estimate` simulates the history from the true parameters and complete routes (m3, m2 above); it is not an estimate from data.
2. The s_z = 0 and s_c = 0 shortcuts make hier identical to fullpool/zonepool (F1).
3. `hier_wrong` with `zwrong = permutation(zone)` is deterministic with seed 5 for all instances and regimes; the result is one random mis-partition, not an average over partitions.
4. In `run_day`, the breach check (`W[r] > rt.B`) occurs after the pooled posterior for stop k is formed from every route's stop-k pickup; routes' stop-k pickups of *other* vehicles are therefore available to the decision at the same stop, which presumes simultaneity (M3-v).
5. The posterior mean is of a (exp scale) and then quantised and clipped to [0.5, 2.0]; correct but see m1, m4, m5.
6. The decision rule `ho < cont and ho <= rest; elif rest < cont` is a one-step improvement using the exact value of the library policy (not BATON's fitted `iso` models at stop k); it is a rollout over BATON, not BATON itself. Say so in the paper. Whether a deployed policy would act this way is fine (a vehicle can run the table lookup), but the paper should call it "BATON-rollout at the posterior factor". Handoffs that are strictly worse than the optimal policy at the true a are possible since U_a is not the optimal value.
7. The 150-day/100-day horizons and 10/19/14 instances are the real sample sizes; there are no seeds beyond a single stream (M2).

## Theory check (THEORY_V3)

- Proposition 1 (plug-in regret <= 8 l^2 f MSE): the loss bound `<= 2 l |e| 1{Delta <= 2 l |e|}` is right under (A1). The step "conditioning on e and using (A2)" needs P(Delta <= t | e, state) <= f t, but e = a_hat - a and Delta_a(s) are both functions of the same history that produced W and the posterior, and a_hat is a function of the state's own observations, so independence is false; assume a conditional margin given (state, e) or bound with the sup. Also (A1) is stated for exact U_a but the library U_a come from independently fitted policies (different seeds per a), so Q is only Lipschitz up to a fitting noise term that does not vanish with MSE; the bound then has an additive floor. The result is the standard quadratic plug-in regret under a margin condition; it is not new.
- Corollary 1 (telescoping): the performance-difference lemma gives J(pi_hat) - J(pi*) as the sum over stops of the *advantage under the benchmark's own Q*, with states drawn from the *learner's* distribution, and the benchmark must be the optimal policy (or, for a rollout base, one must work with the improved policy's value, not U_a). Here the benchmark U_a is the value of a non-optimal fitted policy; the learner continues with a different base policy at each stop (a_hat changes), so the sum does not telescope to the stated bound. Needs the correct statement or a weaker "per-decision" claim only.
- MSE formulas (section 3): the Bayesian posterior-variance statements are right for the Gaussian log model; "n = (sum pbar)^2/sum pbar^2" is wrong for the exact gamma posterior (information is k per observation regardless of pbar, so n counts observations; the sufficient statistic is sum p_i/pbar_i). The full-pooling MSE `... + s_z^2 (1 - n_z/n_F)^2 + ...` hides the other zones' terms. "V_hier <= min(MSE_route, MSE_zone, MSE_full) for every data configuration" is correct pointwise (conditional MSE = posterior variance + squared distance to the posterior mean) but only when the model is correct, hence a restatement of Bayes optimality (F2). MSE is on log a in the notes while the decision uses exp-mean.
- Value of pooling (section 4): the claims "largest when n_r s^2 comparable to tau^2" and "vanishing late" are plausible for continuous quadratic loss, but the discrete-action loss weights c_k are the unproved part: the decision regret is only nonzero near ties; with 62 percent of days having fixed = oracle, c_k is tiny at most stops. Expect weak, noisy dependence on K (M5).
- Discrete actions: the margin density bound f_s is the right device but it will not hold uniformly: at the first stops with W near 0 the gap distribution has an atom (no state-dependent choice: handoff/restock dominated for all a), and at breach-adjacent loads the gap is large with a jump in E; you need margin conditions that are checked numerically from the library tables (promised in THEORY_V3, not done).

## Realism (question 3)

- A multiplicative day-level pickup factor shared across zones is plausible as a model of demand intensity (weather, promotions, holiday peaks, return surges), and it is the Florio et al. "external factor" idea. It is not established for per-stop *pickup volumes in SPD operations*: pickup sizes of identified customers are driven by customer-level lumpiness and by arrival of unpredictable requests rather than a common day scale; also deliveries (outbound parcels, known at dispatch) are frequently not random at all in e-commerce, which the model treats as planning-law gamma with no factor. The model is credible for the volume of unplanned pickups but there is no evidence offered.
- Evidence that could be assembled: (a) the Amazon Last Mile challenge data in `data/AmazonLMC` (pilot slice here is 25 routes; the full package-dimension file is 166 MB on S3) has package volumes per stop, station and date, so one can fit a variance decomposition log(volume per stop / planned) = station-day effect + route effect + stop noise and test the key operational prediction: *the early-stop sums of the other routes of the same station-day improve prediction of a route's later stops beyond its own early stops and planned means*, and estimate the zone-level component using geographic zones in `route_data`. This is the only direction that tests the model. Caveat: it is delivery volume, not pickup volume, so it supports the shared-factor mechanism, not the pickup magnitude; (b) public parcel/mail-pickup or bike-share-return datasets, or a carrier's own return data if the author has access, would test pickups proper; (c) a place-based check (does zone-day correlation of ratio-to-plan exceed route-day correlation at distance?).
- If deliveries also carry a factor (likely, since delivery volume is the dominant e-commerce volume driver): routes' loads W = p - d would load on a *difference* of two factors, with the policy's quantised library indexed by a single scale; the library must be two-dimensional (a_p, a_d) and the likelihood-based learner must separate the two; and because deliveries are visible at dispatch (known load), it is the pickup factor that is identifiable only from pickups; the current code never tests this. The claim "pooled learning helps" may weaken or strengthen depending on the sign of correlation between the factors.
- If the factor drifts within the day (diurnal pattern, shocks): the posterior mean becomes stale; the hierarchical model needs a state-space (AR(1) log-factor, Kalman/particle filter) with forgetting. Under drift the benefit of pooling is *larger* (fast tracking needs more data) but the prior-shrink structure and the plug-in library at a fixed a are misspecified; run it as the headline robustness experiment, since this is the original TEMPO setting.
- Information-structure realism: stop-index lockstep (M3-v), perfect observation of each pickup's size at service, a single day cycle with 30 to 120 days of stationary history, and fixed route groups across days are all optimistic.

## What is sound

- Common random numbers across all policies within a day; no policy consumes the rng; paired comparison is the right design.
- The exact grid posterior and its two-level extension are implemented correctly; the likelihood matches the generator; the baselines are given the same true total prior sd (conservative for hier).
- `RouteValue` gives an exact (not simulated) fixed-policy value on a load grid, so the one-step rule is cheap and deterministic; the library is cached across days.
- The negative finding that full pooling fails under zone effects and zone pooling fails under a city factor is real and replicated across three instance families, with significance in most cells (City, Salhi-Nagy, Dethloff).
- Honest reporting in REVISION_PLAN that full Bayes averaging adds nothing over the posterior mean.

## Prioritised fixes before drafting

1. Replace the circular zone design (M4) with a spatial random field and mismatch grid; drop the degenerate single-factor cells from the headline (F1).
2. Add independent seeds, 500+ days, crossed-bootstrap intervals and CVaR (M2); report the share of days with nonzero decision impact.
3. Rename the oracle; add a genuine bound (clairvoyant on remaining pickups or exact Bayes-adaptive DP on small instances) (M1).
4. Add rival estimators and recourse rules (shrinkage ratio, tuned thresholds, BATON pi1-pi3, section 7 cost-based trigger), asynchronous clocks and Poisson-like noise (M3).
5. Fix the theory: conditional margin, performance-difference with the correct benchmark, correct effective counts; check l_s and f_s from the library tables and the value-of-pooling prediction per instance (F3, M5).
6. Field-data test of the shared-factor structure on Amazon LMC (and a within-day drift variant) before committing to the model (R3).
7. Decide the contribution: either a general result on when pooling can hurt/help with decision-relevant loss, or a coupling that Florio et al. cannot do; "Florio plus fleet" alone will not clear TR-B or TS (M6).
