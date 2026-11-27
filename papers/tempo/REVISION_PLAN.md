# TEMPO after the Transportation Science rejection: analysis and plan

Decision: rejected at Transportation Science (TS-2026-0425), November 2026.
AE recommended rejection; R1 reject, R2 major revision, R3 reject-level
concerns (report in `reviews/trsc_referee3_report.pdf`, full text of all
reports in `reviews/trsc_decision.md`).

## 1. What the reviewers actually said

The four AE points, with the referee comments behind each:

| # | Concern | Who | Valid? | Root cause in the manuscript/code |
|---|---|---|---|---|
| A1 | The routing problem and the feasible recourse are never defined (variant, size, fleet, capacities; what can be reassigned once vehicles are loaded; cost of reassignment) | R1, R2 | Yes | §2 starts from "a routing plan"; the rebalancer (`ev/replan.py`) moves delivery customers between vehicles while ignoring that their goods sit on the original vehicle |
| A2 | The alarm is not decision-relevant: a harmless deviation still triggers it; the false-alarm guarantee covers only exact on-model days, not "replans that were not worth it"; the weights are proxies, not plan costs; how would a dispatcher get continuation values and replan costs? | R3, R2 | Yes | H0 is "the day follows P0". The sensitivity is `1-|B-W|/B` and `(m-k+1)/m` (`ev/eprocess.py::_sensitivity`), not derived from any cost. The ablation shows the tilt alone *hurts* demand detection |
| A3 | No evidence that acting on the alarm saves money against what operators do: tuned periodic replanning, replanning every few stops, continuous cheap re-optimisation, a cost-based trigger, all with the same replanner, a nonzero replan cost, whole fleets, all scenarios. Null fitted in-sample (Amazon); no misspecified-null test; single-route experiments behind fleet-level claims | R2, R3 | Yes | Cost appears only in Tables 6-7 (one surge scenario, ad hoc cost 0.1/km + 40/breach, replan cost ~0). Periodic/rolling baselines dismissed as "uncontrolled alarms" |
| A4 | Not positioned against stochastic routing with recourse: VRPSD a-priori + restocking (Yang et al. 2000), reoptimisation (Secomandi & Margot 2009), partial reoptimisation (Florio et al. 2022). "TEMPO's natural contribution could be to detect when the known-distribution assumption fails" (R3). AE: the biggest barrier | R1, R3, AE | Yes | Related work covers dynamic-VRP surveys and statistics only |
| W | Writing reads as machine-produced and unchecked: long sentences, dashes, "not X but Y", commentary on what is/isn't claimed, internal labels (`tempo2`, `cusum_cal`, "milestone-two", "harness"), broken hyphen spacing, appendix numbered .1-.16. One reference does not exist (`gmira2021rollingopt`) | R3, R2 | Yes | The Gmira entry was flagged in `VERIFY_CITATIONS.md` as a placeholder and was submitted anyway |

None of this is a disagreement about correctness of the proofs. It is
about (i) problem definition, (ii) what the test controls, (iii) whether
it pays, (iv) positioning, (v) presentation and reference integrity.

## 2. The redesign (one idea that answers A1-A4 together)

**Make the execution layer explicit and test the plan's economics, not
the model's fit.**

1. *Problem (A1, A4).* The routed problem is the VRP with simultaneous
   pickup and delivery and stochastic demands (SVRPSPD), fixed fleet,
   a-priori routes from the ALNS planner, executed with recourse:
   continue, hand the route suffix to a standby vehicle (goods
   transferred at the meeting stop), return to the depot, or pay an
   emergency vehicle after a breach; lateness priced per late customer.
   This is the execution model of the companion paper (BATON, now
   accepted at C&OR; cite in third person), which is a restocking-type
   VRPSD policy fitted by backward induction (Yang et al. 2000 lineage).
   Feasible re-optimisation actions are stated explicitly: (a) re-fit
   the recourse policy under an updated demand/time law (no physical
   move), (b) re-sequence a vehicle's remaining stops, (c) hand off a
   suffix with goods transfer at a priced meeting stop. Cross-vehicle
   reassignment of delivery customers without a transfer is removed.
   Instance dimensions (customers, vehicles, capacities, route lengths)
   tabulated; every experiment labelled single-route or fleet.

2. *What is monitored (A2, R3's suggestion).* Like every VRPSD policy,
   the recourse policy reacts to realised demand at every stop but
   assumes the demand (and travel-time) law is known. The monitor's job
   becomes: detect, anytime-validly, when the plan is losing money
   relative to the model it was optimised under, and only then trigger
   re-estimation and re-optimisation.

3. *The statistic: a value-innovation e-process.* Let V be the plan's
   expected remaining cost under the planning law P0 and the fitted
   recourse policy, as a function of the execution state (stop, load,
   schedule delay), computed offline to high precision by simulation
   under P0 (P0 is known, so V is not limited by historical data). At
   each event the innovation
   `I_n = V(s_n) + c_n - E_P0[V(s_n) + c_n | F_{n-1}]`
   is a bounded martingale difference under P0. Under any law,
   `E[I_n | F_{n-1}]` is the first-order excess cost the deviation
   causes (≈ cost gradient · mean shift). Test the **composite,
   decision-relevant null**
   `H0(eps): E[I_n | F_{n-1}] <= eps_n for every n`
   ("the day is costing at most eps more per event than planned") with
   nonparametric betting e-values `e_n = 1 + lambda_n (I_n - eps_n)`
   (Waudby-Smith & Ramdas 2024), predictable `lambda_n`, `eps_n`.
   Consequences:
   - The guarantee now covers what R3 asked for: on every day whose
     deviations do not raise expected cost by more than eps — including
     drifted days in harmless directions (lighter demand, heavy demand
     on an empty vehicle, delays where nothing is late) — the
     probability of ever replanning is at most alpha.
   - Weights are no longer proxies: the cost gradient is inside I_n, so
     the "decision-relevant tilt" of the TRSC version becomes a theorem
     (first-order equivalence) instead of a heuristic.
   - Validity needs only bounded innovations and the mean condition, not
     a correct parametric shape, so a misspecified-but-harmless null is
     inside H0 (answers R2's misspecification point by construction;
     verified empirically anyway).
   - All channels share one currency (cost), so the multi-channel
     combination machinery (dual-regime product/average, Bonferroni
     split) disappears. Fleet pooling = one stream of innovations in
     clock order.
   - eps has an economic reading: replanning at event n pays to first
     order if (excess rate) × (remaining events) > replan cost, giving a
     predictable `eps_n = C_rp / N_remaining(n)`; this replaces the
     alpha* corollary with something R3 can check.
   - The per-channel likelihood-ratio monitor of the TRSC version stays
     as a baseline ("model-fit monitor"): it detects any deviation.

4. *Channels.* Keep demand, travel time, dwell. Drop the accident and
   breakdown channels (a breakdown is observed directly and needs no
   test; accident probabilities drew R1's "how is this measured"
   question). Zonal congestion stays as a travel-time drift pattern.

## 3. Experiments (A3)

All on whole fleets (every route of each plan), BATON cost model, the
same replanner for every trigger, nonzero replan cost `C_rp` (swept),
separate tuning / test days, 95% instance-clustered intervals.

Triggers compared (identical replanner on trigger):
1. never (static model)
2. periodic every T minutes, T tuned on tuning days
3. every k stops, k tuned
4. continuous: re-estimate the law every event (EWMA / conjugate update)
   and re-fit; re-sequence when the plan changes (charged C_rp per change)
5. cost-based trigger: replan when predicted saving under the current
   plug-in estimate exceeds C_rp (R3's suggestion)
6. tuned CUSUM on standardised residuals
7. model-fit LR monitor (TRSC TEMPO)
8. value-innovation monitor (new)
9. oracle (knows onset and post-change law)

Scenarios: null; harmful drift (demand up on tight plans, slowdowns that
cause lateness, ramps, transients, zonal congestion, compound); harmless
drift (demand down, demand up on slack plans, slowdowns with ample
slack, variance-only change); misspecified null without cost impact
(heavy tails, correlation rho, wrong sigma) and with cost impact.

Metrics: fleet day cost split (distance, lateness, handoff, depot
return, emergency, replan cost), number of replans, replan rate on
null/harmless days, detection delay on harmful days.

Also: ablation with/without the cost weighting (LR monitor with flat
vs cost weights vs value-innovation), estimated Delta_c, g, C_fr with
the predicted vs empirical best alpha/eps (R3), Amazon routes with the
null fitted on a training split of routes and evaluated on held-out
routes, sample-size study for the fitted null.

Theorem 2 ("waiting beats acting early") is demoted: stated only for
one-shot costly interventions (R2) and tested against the continuous
policy instead of argued.

## 4. Writing and integrity

- New manuscript file `papers/tempo/tempo.tex` (naming rule), TRSC
  version archived in `trsc/` as the record.
- Plain style: short sentences, no "not X but Y", no commentary on what
  is or is not claimed, no internal labels, no em-dash chains.
- Every reference verified against the publisher record before it
  enters `references.bib` (audit: `reviews/reference_audit.md`); no
  unverified entry may be submitted, flagged or not.
- Hyphenation: no line breaks after a hyphen in source; INFORMS
  appendix via the class's appendix environment (or the target
  journal's template).

## 5. Target journal (equivalent standing to TS)

Recommended: **Transportation Research Part B: Methodological**
(methodological stochastic routing and sequential decisions; same tier
as TS). Alternatives: Transportation Research Part C (real-time
operations angle), Operations Research (higher bar). Check the chosen
journal's anonymity policy and length limit before writing.

## 6. Work modules

| Module | Content | Tests |
|---|---|---|
| M1 `ev/value.py` | V(stop, load, delay) and E_P0[V(next)] tables per route under P0 and the BATON policy | mean-zero innovation under P0 by simulation |
| M2 `ev/vimonitor.py` | betting e-process, composite null eps_n, lambda by aGRAPA/ONS, fleet pooling | alarm rate <= alpha under P0 and under harmless drift; power under harmful drift |
| M3 `ev/fleetsim.py` | fleet-day simulator with BATON recourse, lateness, drift/misspecification scenarios, replanner interface, replan cost | cost accounting identities; reproduces BATON costs with no drift |
| M4 `ev/triggers.py` | the nine trigger policies, tuning on separate days | each trigger fires as specified on synthetic streams |
| M5 `scripts/tempo2_eval.py` | experiment grid, CSVs, Amazon train/test split | — |
| M6 manuscript | `tempo.tex`, rewritten from scratch | reference audit, style pass |

Roles (per the author's instructions): planning/design here; modules
implemented in small specified pieces; a separate review pass.
