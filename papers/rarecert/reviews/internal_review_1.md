# Referee report (IJOC): "Certifying rare overflow risk of regional pickup-and-delivery plans"

Files read: manuscript/main.tex, THEORY.md, PROJECT.md, results/rarecert/{smis,r2,plan}_summary.md, results/quasar/rare_fair_comparison.md, data/Region/README.md. Nothing in the repository was edited.

Verdict: major revision. The anatomy result and the silent-failure diagnostic are interesting. The evidence is thinner than the abstract says, the central deliverable (a certificate outcome for the plans) is a red placeholder, and the baselines that matter are missing.

---------------------------------------------------------------------

## 1. Numbers that do not match the result files

Most numbers match (checked: 148 = 122+26 instances, 40,099 customers, 178/178, 0.74%, RV 3.8-5.9 vs 3.8-6.0, 224-276x, 3.2-3.8x, all route-day figures, quantum ratios, 27-37%). Mismatches and misleading renderings:

1. SMIS error "at most 0.6% (90% quantile) in every band". smis_summary.md: SMIS0 q90 reaches 0.61% (1e-6, [0,1/3]); SMIS3 reaches 0.71% (1e-4, [0,1/3]) and 0.60-0.61% elsewhere. The manuscript never says which variant (n_arc = 0 or 3) it reports.
2. "built in 0.4-0.9 s". SMIS0 build medians run 0.39-1.38 s (1.36, 1.15, 1.38 in the n=1 band (0.5,0.75]; 0.86 at 1e-6 for ϱ>0.75). SMIS3 is 0.68-3.06 s. The range is only right if the n=1 band is dropped.
3. "every ϱ band". The band (0.5,0.75] has n = 1 route. The statement is vacuous there.
4. Hubs: "all of Northern Vietnam". plan_summary: "66 hubs ... 128 hub-gamma rows ok, 4 failed". VNN-HANOI-2 and VNN-LANGSON failed at both γ ("no SMIS candidate reaches c=45000"). Results cover 24 of 26 VNN hubs (table: VNN 24, GD 40). The failures are not disclosed, and Hanoi, the largest hub, is among them.
5. Hubs: "the 19 smallest hubs". The file header says "20 smallest hubs"; the table lists 19. |z| ≤ 1.67 matches (GD-RUYUAN, -1.67). The file is internally inconsistent; the manuscript silently picks 19.
6. "estimate lies between max_r p_r and sum_r p_r in every case". The file: "64 / 64 ... checked with slack 4 (rse_hub + max rse_r)". It is a tolerance check, not a strict ordering, and it covers 64 hubs, not 66.
7. "overstates hub risk by a factor 1.75-2.2 in median and up to 3.2". File: indep/P_h median 2.2 (1e-3) and 1.75 (1e-5); 3.17 and 2.27 are the q90, not a maximum. "27-37% of failing days" is the median over hubs (0.371, 0.273), not a pooled share.
8. "single shift diverges by up to 28%, 63%, 98%". These are q90 values of the MDIS-vs-single disagreement (28.37, 63.27, 97.59), not maxima. Same for "up to 40%" in Section 4: 39.94% is the q90 at 1e-6, band (0.75,1], not a maximum.
9. "median error 3-6%" for the single shift when ϱ>0.75. File: 2.86, 5.10, 6.00, 6.05%. Also "0.1-0.2%" for ϱ ≤ 0.5 omits values up to 0.23%.
10. "DP agreement 178 of 178" is for the CE mixture, not SMIS. The abstract credits "the estimator" with matching exact DP. smis_summary: "Reference p_ref = p_dp where present (L <= 8, gamma >= 1e-4), else the CE mixture p_is". For γ ≤ 1e-5 (240 of 480 rows) and for every L > 8 route, SMIS is compared with its own competitor. SMIS vs DP alone is never reported. "CE 0.00%" at ≤1e-5 is trivially self-reference (the file notes this).
11. Abstract: "matches ... 10^7-draw Monte Carlo". Crude MC exists only for the route-day simulator at γ = 1e-3 (r2_summary), not for the SMIS route model.
12. Abstract: "relative variance 4-6 at every level" and Conclusion "accurate to a fraction of a percent at 1e-3 to 1e-6". Route-day was run only at 1e-3..1e-5. For ≤1e-5 the "accuracy" is agreement with another IS method (MDIS vs CE 0.20-0.24% median, q90 0.47-0.59%), not with the truth.
13. Remark "energies only 16-35% of asymptotic, γ in [1e-9,1e-3]". THEORY.md §7: this is one 3-stop test route (E·λ/c = 0.16 at P≈9e-4, 0.35 at P≈1.6e-9), not the real routes.
14. Quantum: "2.5-380 times more calls than classical IS" is the pilot-inclusive ratio (Qcrude/CIS_total), the figure most favourable to quantum. Against sampling-only IS the ratio is 27-3550. The abstract says QAE "does not beat" classical IS, while the body concedes it does in the idealised accounting at ε ≤ 3% (QIS_ideal/CIS_sampling 0.14-0.48). Make the abstract match the body.
15. Intro "100/γ draws for 10% error" is a 1-sigma figure; Section 5 costs use 1.96². The confidence level is never stated.
16. Routes "stratified by region and length": README shows VNN has pickup rate set equal to delivery rate (an assumption). The 40 routes with ϱ>0.75 are, by the numbers (17+62+1+40 = 120; VN 'balanced', GD 'pickups 2.7x'), the VN routes. Region, ϱ and the assumed pickup rate are completely confounded. The paper presents ϱ>0.75 as a finding about "balanced deliveries and pickups (Northern Vietnam)", but README says "PICK_RATE ... balanced with delivery (national rate; no regional split published)". The silent-failure regime is created by a modelling assumption.
17. Route length. DP covers L ≤ 8; 178 DP cases of the 240 rows with γ ≥ 1e-4 (≈ 89 of 120 routes at 2 levels) imply about 3/4 of the routes have ≤ 8 stops, and hubs have roughly 5-8 stops per route (e.g. BACKAN 25 customers, 5 routes). The abstract and Section 4 talk about "long routes", which the data barely contain.

Parameters never stated in the manuscript: ρ, cv (only in a Remark), route-length distribution, ALNS gate level of the plans, CE settings, SMIS η, a₀, n_arc, N, route-day simulator parameters.

---------------------------------------------------------------------

## 2. Overreach: label as sketch / conjecture / heuristic

Must be labelled in the main text (not only in the appendix):
- Theorem 1 (large deviations): sketch. Uniformity of the O(log) remainder (Lemma A1 is stated without proof, and THEORY lists its refinement as an open item), and the lower-bound ball argument, are not written out. "c → ∞ with the route fixed" ignores the simulator's clipping of Φ at 1−1e-15 (|z| ≤ 7.94, THEORY §7), which makes demands bounded; the unreachable-c failures in the hub run are probably this. State it.
- Proposition 5 (mode mass) is derived from Theorem 1, so it inherits the sketch status; its γ^{1/ϱ−1} form requires c ≈ λ* log(1/γ), which the paper's own Remark says is off by a factor of 3-6 at practical γ. Call it qualitative. No experiment compares the predicted mass with the measured bias.
- Theorem 2(b) (log-efficiency when ϱ ≤ 1/4): sketch ("handled by Cauchy-Schwarz (sketch)"). 2(a) is a short proof but rests on Theorem 1. It also concerns the dominating-point shift τ*, whereas the experiments use a CE-fitted single shift. The "sharp criterion" in the contributions is not sharp: (a) and (b) have different hypotheses (b needs an extra second-eigenvalue condition), and the threshold is asymptotic.
- Theorem 3 (SMIS log-efficiency): already "sketch" in the appendix, but §4 says it "gives conditions for logarithmic efficiency" and the abstract and conclusion present SMIS as theoretically backed. Also the condition κ ≥ 4 fails for typical short routes: one-factor block with L=6, ρ=0.3, equal θ gives λ1/λ2 = 2.5/0.7 ≈ 3.6 < 4; heterogeneous θ is worse. THEORY's own rule needs ρΣθ ≥ 3(1−ρ)θ_max or so. Report how many of the 148×2 end blocks satisfy it.
- Proposition 4 (within-stage ties) is "COMPUTED, to be written as a proof", and the arc fix is a CONJECTURE (THEORY §4). Neither appears in the manuscript, yet SMIS3 uses arcs and the abstract claims "within-stage combinations are close behind".
- Abstract claim "middle stages and within-stage combinations are close behind on long routes": no theorem, no figure, no number. THEORY says middle stages have strictly smaller scales and that the crossing stage has λ ≈ λ*ϱ/(1+ϱ) ≤ λ*/2, "so ties matter only for heterogeneous routes". That contradicts the abstract's wording.
- "Exact dominating point": the minimum-norm point of a non-convex set (composition with gamma quantile) is found by local constrained minimisation started on the Perron ray. It is a local, not a certified global, minimum. Say "computed".
- "Certificate" (Definition 1: |p̂−p| ≤ εp at confidence 1−δ): the paper provides CLT-based intervals only. No coverage experiment (replications, empirical coverage, ESS, weight diagnostics), no finite-sample or bounded-relative-error guarantee. Either rename to "estimate with confidence interval" or add coverage evidence.
- "needs no tuning" / "pilot-free": η (0.001, hub 1e-4), a₀, n_arc and the mode-retention rule are tuning parameters; the 0.4-1.4 s construction is an optimisation, not free.
- Hub claim "the same machinery": the hub input is much higher-dimensional; validation is only at 6e-5 (see §3).
- The route model's key assumption (W_d independent of W_p) is what makes the two blocks independent in the proof of Theorem 1 (mgfs multiply) and gives the two-ends picture. With correlated pickup and delivery factors the result changes. It is stated as a modelling choice but not flagged as load-bearing.
- "Region risk 1−∏(1−P_h)" rests on independent factors across hubs (even the two Hanoi hubs). The paper lists region-wide factors as open, but the table of P_region is reported as a result.
- "Certify ... for every route and for the plan" (Introduction, contribution (i)): no plan is shown to pass or fail. Calibrated capacity buffers are a red [pending]. Hub results at γ = 1e-3 have P_h/γ median 0.036 (q90 0.065; plan_summary): the capacity proxy ("a proxy, not a bound") is 15-45x off the target. The reported buffers (median +53% to +97%) are therefore not the buffers needed for γ.
- Attribution to routes is promised in (iv) and "Hubs" but no attribution result is reported.

---------------------------------------------------------------------

## 3. Missing baselines and experiments a referee will demand

Priority order.

1. Exact conditional computation as a classical competitor. Given the factors (W_d, W_p), customers (hence routes of a hub) are independent, so P(hub fail | W) = 1 − ∏_r (1 − p_r(W)) and the hub probability is a 2-D integral. Route p_r(W) is a 1-D DP/convolution for L ≤ 8 (already built: "fine quadrature over both factors"). With about 3/4 of the routes at L ≤ 8 and hub routes of ~5-8 stops, deterministic quadrature may beat SMIS for the test set. Report DP cost and accuracy next to SMIS, extend DP to γ = 1e-5, 1e-6 with log-scale/adaptive quadrature so SMIS gets a true reference there, and state where DP stops being feasible (L, dimension). Also conditional Monte Carlo over the 2 factors + exact/DP inner (the standard variance-reduction competitor).
2. Glasserman-Li two-step IS (factor shift + conditional exponential twisting), planned in PROJECT.md C6 but absent. Also subset simulation / multilevel splitting (also planned) and Owen-Maximov-Chertkov ALOE with linearised half-spaces. Currently the only classical rivals are crude MC, a single CE shift and CE mixtures with a big pilot.
3. Fair CE. Cost ratios of 224-276x (routes) and 1750-1830x (route-day) at ε = 0.1 are dominated by the CE pilot (0.8-1.2e5 draws for routes; ≈3e6 per route for route-day, 1.853e8/60) and shrink to 3-4x and 18-19x at ε = 1%. The pilot size is a CE hyperparameter; vary it, add a smaller-pilot CE and a "two end shifts, no optimisation" estimator. SMIS RV equals CE RV (3.8-5.9 vs 3.8-6.0) and is slightly worse than the single shift in easy bands (3.76 vs 3.51); the only advantage is the pilot. Report wall-clock: route SMIS build 0.4-1.4 s is likely slower than a 1e5-draw pilot; route-day gain is 3-4 s vs 43-55 s (about 14x), not 1800x.
4. Independence of reference. At γ ≤ 1e-5 and L > 8, SMIS is only compared with CE through the same simulator, quantile transform and clipping, so a shared bias is invisible. Needs DP (item 1), or crude MC at 1e9 for γ = 1e-4 on a few routes plus a cross-check by an independent implementation.
5. Sensitivity: ρ ∈ {0.1, 0.3, 0.5, 0.8} (κ ≥ 4 conditions depend on it), cv ∈ {0.1, 0.3, 0.6, 1}, demand family (lognormal and truncated normal, heavy tail where Theorem 1 does not apply, empirical), correlated pickup-delivery factors, number of stops (5 to 60), route-length distribution, pickup-to-delivery ratio varied independently of region (to unconfound ϱ, region and assumed rate), capacity utilisation. Report model risk: how much p at 1e-6 moves under ±10% cv or ρ; precision ε = 1% on a quantity that varies by orders of magnitude with cv is not meaningful on its own.
6. Test of the diagnostic itself: ϱ > 1/4 is claimed to mark inefficiency, yet in the data 79 routes with ϱ ∈ (1/4, 1/2] show 0.2% single-shift error and failure appears only for ϱ > 0.75. Plot measured bias versus the predicted mass exp(−(E_m−E_min)) (exact energies) and versus ϱ; say where the practical threshold lies, and restate the diagnostic with exact energies. Also show that the silent failure is silent: single-shift RV is 512 at 1e-3 and 87 at 1e-4 (reported SE ≈ 2.3% and 0.9%, comparable with the bias 2.9% and 5.1%), so only at ≤1e-5 (RV ≈ 5, SE 0.2%, bias 6%) is it silent. The error signs are not reported, so "underestimates" is untested.
7. Route-day: describe the simulator (no description is in the manuscript), validate at 1e-4 and 1e-5 against something independent, and justify the "thresholds that split failure evenly" design (it builds in the case that defeats a single shift; bisection cost excluded; 16-21 "unreachable modes" per route; 1-3 "slack expanded"). Add naturally chosen thresholds.
8. Hub validation: crude MC at 2e6 draws on the 19 smallest hubs at P ≈ 6e-5 (rse 8-10%) detects only ≥ 20% bias, and is at 1/20 of the stated level. Add hubs at γ = 1e-5 with an independent method (the 2-D quadrature of item 1) and larger hubs.
9. Plan-level result that matters: fraction of routes/hubs passing at each γ for the actual SAA plans, capacity buffer after calibration, and what the plan's chance-gate level was.
10. Confidence-interval coverage and ESS over replications; sensitivity to N, η, a₀.

---------------------------------------------------------------------

## 4. Presentation

- There are no tables and no figures. All numbers (≈60) are in prose. IJOC readers expect an algorithm box (SMIS pseudocode with complexity and the mixture weights/likelihood ratio), an experimental design table, and results tables.
- Structure: Problem → Structure → Single shift → Estimators → Experiments is fine, but the Estimators section is a sketch (no formulae for the hub likelihood ratio, no description of CE baseline, no MDIS specification). The appendix proofs are fragments; move the sketch status to the statements.
- Missing: data/code availability statement and reproducibility pack (IJOC requires code and data submission with the paper); instance description (distribution of L, stops per route, ϱ, plan gate), sources caveats (served population 115.5M vs official 127.8M; 2008 VN provinces; "Guangzhou Province" label in README; pickup = delivery rate for VN assumed; 1 kg/parcel and 15/30 t trucks assumed, as README marks ASSUMPTION).
- Abstract is long (≈230 words) and mixes proved, heuristic, and numerical claims; shorten and make each claim match its evidence (see §1, §2).
- The quantum paragraph is a side note whose calls-only accounting ignores qubit and fault-tolerance cost; either move it to a short appendix or tie it to the abstract wording (see §1 item 14).
- Missing references a referee will expect (verify before citing): Glasserman-Heidelberger-Shahabuddin delta-gamma papers (in THEORY.md but dropped from the manuscript), Au-Beck subset simulation, splitting (Cérou-Guyader), Dieker-Mandjes / Adler-Blanchet-Liu on Gaussian maxima, Blanchet-Glynn on maxima of random walks, a general SVRP review (Gendreau-Jabali-Rei 2016).
- Proposed exhibits:
  1. Table 1, instance and plan summary by region: #instances, customers, routes, stops-per-route quartiles, ρ, cv, Q, pickup/delivery ratio, ϱ band counts, plan gate, fraction of routes passing at each γ.
  2. Figure 1, anatomy of a route: exact mode energies E_k (delivery and pickup block) against stage k for one VN and one GD route, with asymptotic λ-prediction overlaid; shows two ends, near-end stages, and the pre-asymptotic gap.
  3. Figure 2, silent failure: single-shift relative bias (signed) against predicted missed-mode mass and ϱ, coloured by region, with the reported ±2 SE band and the ϱ = 1/4 and 0.75 lines.
  4. Table 2, route-level accuracy and cost: by (γ, ϱ band) vs DP only, for SMIS, CE (several pilots), GL two-step, subset simulation, conditional MC, crude MC; error, RV, build wall-clock, total samples and total seconds at ε = 10% and 1%. A second table (or Figure 3) for hubs: P_h vs max p_r / sum p_r envelope, independent-factor overstatement, calibrated buffers per region at 1e-3 and 1e-5, with the exact quadrature as reference.

---------------------------------------------------------------------

## 5. Scope and contribution

IJOC fit: moderate. The paper states "we do not claim new IS theory"; the novelty is the problem formulation (post-optimisation certification), the pickup-and-delivery peak anatomy, the 1/4 criterion, the hub shared-factor union with attribution, and a large new instance set. Weaknesses: (i) the model is small enough (L about 5-8, two shared factors) for exact numerics, so the need for IS is not shown; (ii) the headline efficiency is parity with CE variance plus saved pilot samples; (iii) theory is mostly sketched and asymptotic where practice is pre-asymptotic; (iv) the certificate outcome for the plans is missing. The route-day MDIS and hub parts are the genuinely computational contribution but are the least validated.

What would most raise the chance of acceptance, in order:
1. Add the exact-conditional (2-D quadrature + DP) competitor and GL two-step/subset simulation, and show the regime (longer routes, larger hubs, richer simulators) where SMIS/MDIS wins on wall-clock; include routes with 15-60 stops.
2. Deliver the plan-level certificate: calibrated capacity search (exact or bounded, not a Laplace proxy), pass/fail rates, buffers, attribution, and a repaired treatment of the failed hubs.
3. Reframe the theory honestly: Theorem 1 and the ϱ criterion as asymptotic sketches or proved lemmas, a finite-level diagnostic in exact energies validated against measured bias, and drop or soften "sharp", "no tuning", "certificate".
4. Unconfound ϱ from region and the assumed pickup rate; add ρ, cv, distribution and correlated-factor sensitivity and a model-risk statement.
5. Provide coverage experiments, a reproducible code and data release (IJOC requirement), tables and figures as above, and align the abstract with the evidence (QAE sentence, "148 instances", DP agreement).
