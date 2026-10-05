# Peer Review Report: Reviewer 3 (Perspective)

## Manuscript Information
- **Title**: Minimum Weighted Hazard-Exposure Dispatch: Complexity, an Exact Algorithm, and an FPTAS (MWHED), revised manuscript
- **Manuscript ID**: not available to this seat
- **Review Date**: 2026-10-05
- **Review Round**: Round 1 (first-time cold read of the revised manuscript; no response letter or other reports consulted)

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective): cross-disciplinary and practical-impact seat

### Reviewer Identity
Operations researcher in emergency management, wildfire and evacuation logistics, and humanitarian transportation, who also follows real-time and stochastic scheduling theory. I am an outsider to the complexity-theoretic core (NP-hardness, FPTAS, matroids) and do not judge those proofs. I judge model realism, what a practitioner could take from the paper, what the paper claims about practical relevance versus what it shows, ethics, and missing connections to adjacent fields.

### Review Focus
Whether MWHED is a faithful abstraction of a hazard-race dispatch decision; whether the 2018 Camp Fire case study can carry the conclusions drawn from it; the normative content of the objective (who is "sacrificed"); and which neighbouring literatures (real-time overload scheduling, stochastic and adaptive knapsack, deadline routing, humanitarian logistics) the authors should connect to.

---

## Overall Recommendation
**Major Revision.** The theory is cleanly presented and honest about being classical. The practical framing is where the manuscript is weakest: the case study's headline result appears to be an artefact of the return-to-depot assumption, and several practical conclusions go beyond what one four-site instance can support.

### Confidence Score
4 (high on the emergency-logistics and practical-modelling points; moderate on how this journal weighs practical realism for a theory paper; I did not assess proofs).

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Practical feasibility and model realism | Reviewer 3 remit (Practical Feasibility) | PARTLY_MEETS | text: §5.6 "twice the real great-circle distance between the depot and site"; figure: Fig. 5 | Single-depot round-trip abstraction is defensible for one scarce specialised asset but contradicts the geometry of the paper's own case study | Based on coordinates read off Fig. 5, not the authors' data | yes: case-study conclusion changes |
| Stakeholder and ethical considerations | Reviewer 3 remit (Stakeholder Voices, Broader Implications) | DOES_NOT_MEET | absence: no equity or ethics discussion anywhere | Objective concedes low-population sites by construction; never discussed | Journal is a theory venue; a paragraph may suffice | yes: required addition, low effort |
| Cross-disciplinary connections | Reviewer 3 remit (Disciplinary Blind Spots) | PARTLY_MEETS | text: §2 "has no analogue in the feasibility-testing literature" | Real-time and wildfire-OR links are present; overload scheduling, stochastic knapsack adaptivity, deadline routing and humanitarian-logistics equity are missing | Literature completeness is Reviewer 2's remit | no (improves positioning) |
| Claims versus evidence on practical relevance | Reviewer 3 remit (Real-world application) | PARTLY_MEETS | text: §7 "whereas protecting the dominant site first is robust" | Case study is labelled an illustration, but conclusion and §6.3 state a practical rule from one instance | Authors do disclose limits in §5.6 | yes |
| Cross-cultural / contextual validity | Reviewer 3 remit (Cross-Cultural Validity) | PARTLY_MEETS | text: §5.6 "a lower bound on road distance" | US wildland-urban-interface event, single-agency depot; transfer to flood, utility or Southeast-Asian contexts (the authors' own setting) is asserted, not shown | Not examined empirically | no |

---

## Summary Assessment

The paper recasts a hazard-race dispatch problem as the classical weighted on-time-jobs problem, presents hardness, a Lawler-Moore dynamic program and an FPTAS in transportation language, and illustrates the model on four communities in the 2018 Camp Fire. I credit the candour: the authors state what is classical, separate real inputs from assumptions, and point out that the arrival and return readings of a deadline give different answers.

From an emergency-operations viewpoint, three things limit practical meaning. First, the model has the crew return to the depot after every site; in the Camp Fire instance the sites lie 4 to 10 km from one another and 22 to 37 km from the depot, so a crew chaining Yankee Hill, Concow, Paradise and Magalia reaches all four well before the stated deadlines (my estimate below), and the reported "Paradise only" optimum looks like an artefact. Second, "serving" a site costs only travel time, weights are census counts, and the outcome is all-or-nothing, which does not describe evacuation, notification or structure protection. Third, there is no discussion of the ethics of an objective that systematically concedes small communities. The algorithmic content is, as the authors say, classical, and at minute granularity the exact dynamic program solves realistic instances in milliseconds, so the practitioner takeaway is mostly the modelling, which is where the paper is weakest.

---

## Strengths

### S1: Candid positioning of what is classical and what is practical
The manuscript says openly that MWHED equals a classical scheduling problem and does not claim a new technique for it; it also labels the Camp Fire study an illustration rather than a recommendation. This calibrates reader expectations well.
**Evidence Anchor**: text: §1 "we do not claim a new algorithmic technique for the general problem"

### S2: Explicit separation of real inputs from modelling assumptions in the case study
Section 5.6 lists real measurements and numbered, disclosed assumptions (speed, extrapolation, zero dispatch delay, reading). This is the right reporting practice for emergency-management applications and makes the instance auditable.
**Evidence Anchor**: text: §5.6 "We separate real measurements from disclosed modeling assumptions"

### S3: The arrival-versus-return remark is a genuinely useful practitioner insight
Remark 1 shows that the same data give different answers under the two readings, and §5.6 demonstrates this numerically (under the return reading, Concow and Magalia become individually infeasible). Many hazard-dispatch papers leave this implicit.
**Evidence Anchor**: text: Remark 1 "a practitioner must state which one the hazard-arrival times refer to"

### S4: The sensitivity analysis surfaces a real operational lesson about brittleness
Perturbing speed, arrival times and delay shows that the nominal optimum (Concow, then Paradise) is on time in only 20.7% of draws. Reporting this against a simple robust alternative is more informative than the nominal result, and it is the part of the case study I would keep.
**Evidence Anchor**: text: §5.6 "The nominal optimum is therefore brittle"

### S5: Honest discussion of where the model breaks
Section 6 states what fails for several vehicles, a moving depot, a hazard that degrades roads, and stochastic parameters (including the small counterexample for expected-deadline ordering), rather than listing them as future work.
**Evidence Anchor**: text: §6.3 "Lemma~\ref{lem:edd} holds scenario by scenario but not simultaneously"

---

## Weaknesses

### W1: The return-to-depot structure makes the Camp Fire "optimum" an artefact of the model, not of the geometry
**Problem**: Every site is served by a round trip from Oroville (`p_i` = twice the straight-line depot distance). That is the premise that puts the problem in the scheduling family, and it is defensible for one scarce specialised asset or a repair crew with a van-sized load. In the Camp Fire instance, however, the four communities form a cluster 22 to 37 km north of the depot, with inter-site distances of roughly 4 to 10 km (Fig. 5). A crew visiting several of them would not drive 30 km back to Oroville between visits. Reading approximate coordinates off Fig. 5 and using the paper's own speeds, a chained route Yankee Hill, Concow, Paradise, Magalia arrives at about 27, 32, 43 and 54 minutes at 50 km/h (about 17, 20, 27 and 34 minutes at 80 km/h). Against the arrival deadlines of the Camp Fire instance table (§5.6) (64, 52, 71, 58 minutes) all four sites are protected at both speeds, with Magalia tight at 50 km/h. The paper's headline results, "Paradise only" at 50 km/h and "Concow then Paradise" at 80 km/h, and the claim that "any detour first, even the shortest, uses time Paradise's deadline cannot absorb", all follow from charging a full round trip to Oroville per site. The 'brittleness' lesson of §5.6 and the Conclusion inherits the same artefact.
**Evidence Anchor**: text: §5.6 "twice the real great-circle distance between the depot and site"
**Why it matters**: The case study is the only evidence offered for practical relevance, and its qualitative outcome (concede three of four communities) is driven by a modelling choice that the instance's own geometry contradicts. A practitioner reading Fig. 5 will see this immediately.
**Suggestion**: (a) Recompute the instance with a chained route and show both models side by side, stating how much the answer changes. (b) Choose a case where the round-trip structure is natural (an incident-command post or a staging area per call, or a single specialised asset that must return to a base to reload or refuel) and say so. (c) Discuss deadline-TSP, orienteering with deadlines and vehicle routing with time windows (see reading list) as the model that applies when sites are chained, and state what MWHED captures that they do not. Note that the authors' own §6.2 acknowledges the order-dependence of `p_i`, but treats it as an extension, whereas for this instance it is the base case.
**Severity**: Major
**Confidence**: 4: core expertise in wildfire and evacuation dispatch; the numbers rely on coordinates read from a figure and should be recomputed by the authors.

### W2: One four-site instance, dominated by a single site, cannot support the practical rule drawn from it
**Problem**: Paradise's weight (26,218) exceeds the other three combined (12,353). Under such weight dominance the optimum is trivially "Paradise plus whatever fits"; it is in the optimum in 100% of draws because of weights, not because of any algorithmic insight. The Conclusion and §6.3 nevertheless state a rule ("when one site dominates the weight, protect it first") and the Conclusion says protecting the dominant site first "is robust". The 98.6% retention of "Paradise only" is again a consequence of dominance (the optimum can add at most 12,353 of 26,928 weight). The case does not test the combinatorial structure the paper studies (trade-offs between comparable sites), so it cannot show the value of the DP, the FPTAS, or even of the weighted greedy, over the dominant-first rule. In addition, the plans are evaluated open-loop: a real incident commander observes delays and re-plans (skips Concow once a delay appears), so the 26.6% "retained" figure measures a policy nobody would follow.
**Evidence Anchor**: text: §7 "whereas protecting the dominant site first is robust"
**Why it matters**: The practical conclusion is stated more strongly than one instance with a dominant weight warrants, and the brittleness is partly an artefact of non-adaptive evaluation.
**Suggestion**: Reword the conclusion to say that the instance illustrates sensitivity, not a rule. Add either a second event with comparable-weight sites (for example a flood or a multi-community wildfire with a flatter population distribution), or a synthetic set of "realistic" instances in which weights come from a calibrated distribution. Evaluate "adaptive" policies in the sensitivity runs (re-optimise at each decision epoch with observed delay) alongside the open-loop plan; this is the honest comparison and connects to §6.3's two-stage discussion.
**Severity**: Major
**Confidence**: 4: practical-modelling judgement; arithmetic on weights checked from the Camp Fire instance table (§5.6).

### W3: Provenance and semantics of the hazard deadlines are thin for a real event
**Problem**: Only two of the four deadlines are documented. The text mixes two different event types in the spread-rate estimate ("fire reaching Concow" at 52 minutes and "spot fires igniting in Paradise" at 71 minutes), and then applies the resulting 13.1 km/h isotropically to Magalia and Yankee Hill. Spot-fire ignition is ember-driven and normally precedes the arrival of the main front, so the implied rate is a spotting rate, not a front rate; if the front reached Paradise later, Paradise's deadline relaxes and the optimum could change. The authors acknowledge isotropy but not the event-type mismatch. Further, a community is represented by one deadline, though a town of 26,000 spread over a large area is reached over hours, not at one instant, and a crew arriving moments before ignition cannot do anything useful. A deadline for a protective action should be the hazard time minus the lead time the action needs (see W4), not the hazard time itself. Finally, the sensitivity analysis perturbs arrival minutes by independent factors; in reality the arrival times share a common driver (wind, fuel), so errors are strongly positively correlated, and independent draws misstate the joint risk.
**Evidence Anchor**: text: §5.6 "this extrapolation is isotropic, whereas the real spread was wind-driven and directional"
**Why it matters**: The deadlines are the one input the paper stresses it takes "as given" from a forecasting step (Intro). Here the given is partly invented, and the result is sensitive to it.
**Suggestion**: State exactly which event each NIST timestamp denotes and use homogeneous event types; if only two are documented, say that the other two are placeholders and run the study with two sites, or add documented times for the other communities (the incident record has more). Add a safety-margin term to the deadline. In the Monte-Carlo, draw a common-factor shock plus site-specific noise, and report how the 100% and 65.3% figures change.
**Severity**: Major
**Confidence**: 3: wildfire-modelling expertise is adjacent to mine; I have not checked the NIST report itself.

### W4: What it means to "serve" a site, and whether census population is the right weight, are not addressed
**Problem**: In the Camp Fire instance `p_i` is pure travel time; no on-site service time exists, yet the crew is presumed able to turn round immediately. What a visit accomplishes is never specified: notifying and evacuating 26,218 residents, defending structures, and opening an egress route are different actions with different durations, and none is a single visit of one fixed duration with an all-or-nothing payoff. Benefit in evacuation is increasing in lead time and in the fraction of residents reached, not a step function at the deadline. Weights are 2010 census counts, which ignores the people who matter most for criticality in wildfire evacuation (those without vehicles, mobility-limited, institutionalised, or elderly), and ignores that service time scales with population. That last point has a consequence the paper does not remark on: if service time is proportional to population, then `p_i` and `w_i` are strongly correlated, which is exactly the hard "w = p" family of Theorem 1 and the "strongly correlated" family of §5.3, and the polynomial cases in Theorem 5 (constant `p` or constant `w`) are then even less representative of practice. Finally, the instance sets Camp Fire's main operational constraints aside: evacuation egress capacity, notification, and local resources already in the communities (the paper's single Oroville depot is an assumption, not the actual response structure).
**Evidence Anchor**: text: §5.6 "used as a proxy criticality weight"
**Why it matters**: The model's objective is only as meaningful as `w_i` and `p_i`; as built, "protected population" is not an operational quantity that an emergency manager would recognise.
**Suggestion**: Define the action being modelled (for instance "protect structures with one strike team" or "run a notification and contraflow sweep"), add a service time `s_i` (the framework can absorb it: `d_i` = hazard time minus `s_i` plus the unused return leg, a one-line extension of Remark 1), and use a vulnerability-weighted exposure (for example population without vehicles, or structures) with a sensitivity run. Add a sentence on the `p`-`w` correlation and its effect on which tractable cases are relevant. Include a short limitations paragraph on what MWHED does not capture in a real incident (egress capacity, comms).
**Severity**: Major
**Confidence**: 4: core expertise in wildfire evacuation and humanitarian logistics.

### W5: No discussion of the ethics of "sacrificing" low-weight sites
**Problem**: The objective is purely utilitarian and the manuscript's own vocabulary is "sacrifice" and "concede" (Intro, Example 1, Fig. 1 caption, §2). Under any weights proportional to population the optimum systematically gives up small, often remote and poorer communities in favour of large ones; in the Camp Fire instance three of four named real communities are not served at 50 km/h. The paper does not discuss equity, duty-of-care obligations of public agencies, fairness-constrained variants, or the risk that a retrospective optimisation of a real disaster reads as a judgement on what should have been done. The Camp Fire killed 85 people; presenting named communities as "not dispatched" in a figure deserves a framing note even in a theory paper.
**Evidence Anchor**: absence: manuscript as a whole — expected discussion of equity or ethical implications of conceding low-weight sites; checked §1, §3, Fig. 1 caption, §5.6, §6, §7
**Why it matters**: A reader in emergency management will ask immediately who is left out and why a fairness requirement is not modelled. The framing statement is also needed for responsible use of the model.
**Suggestion**: Add a short subsection (or paragraph in §6) that (i) states the utilitarian assumption, (ii) notes that minimum-coverage or max-min variants change the problem (a fixed set of required sites is a trivial extension, a max-min objective is not), (iii) cites the humanitarian-logistics equity and deprivation-cost literature, and (iv) adds a sentence that the case study is a retrospective stylised illustration, not an assessment of the actual response.
**Severity**: Major (repairable by rewriting, no new analysis needed beyond a discussion of which extensions stay tractable)
**Confidence**: 3: ethical framing is partly a matter of venue norms; the absence itself is certain.

### W6: The practical value of the algorithmic results is not stated, and the evidence suggests it is small
**Problem**: With integer minute data the exact dynamic program is `O(nP)`; the authors' own runtime study solves `n` = 40 to 400 instances in milliseconds, and the FPTAS is slower than the exact DP at small `p_max` (§5.4). Realistic dispatch instances (tens of sites, horizons of hours at minute resolution, so `P` of order 10^3 to 10^4) are solved exactly in negligible time, and the weighted greedy repair is within 1% of optimum on average (§5.2). The abstract presents the FPTAS and the NP-hardness classification as contributions; a practitioner would conclude "use the DP" and nothing in the paper says when anything else is needed. The "which heterogeneity causes hardness" classification is described as an account of "which network structure makes dispatch easy"; but a statement about worst-case classes does not carry to practice, in particular because `p` and `w` co-vary (W4). The paper also settles the case "completely" only for the deterministic, single-vehicle, open-loop model, which is the model least like operations (W1, W2).
**Evidence Anchor**: text: §6 "settle the single-depot, single-vehicle, deterministic-deadline case completely"
**Why it matters**: The paper's practical relevance claims (Introduction's wildfire, utility and flood settings; abstract's "transportation audience") are not matched by a statement of what a planner gains.
**Suggestion**: Add a "what to use when" paragraph: for realistic sizes use the exact DP; the FPTAS matters only for very large weight or time magnitudes; greedy repair is a fast baseline whose failures have a known form. Say plainly that the hardest practical difficulties (stochasticity, routing, several heterogeneous resources) are outside the settled case. Reframe the abstract's practical language accordingly.
**Severity**: Major
**Confidence**: 3: partly a judgement about what a theory journal expects; the runtime facts are from the authors' tables.

### W7: Connections to overload scheduling, stochastic and adaptive knapsack, and deadline routing are missing, and one novelty statement is too strong
**Problem**: §2 says the question of which weighted subset to sacrifice "has no analogue in the feasibility-testing literature". Real-time systems has exactly this question: overload and value-based scheduling (maximising the value of on-time jobs under overload, firm deadlines, skip-over models), and non-preemptive EDF (a vehicle round trip is non-preemptive, whereas Liu-Layland is preemptive). Stochastic scheduling and the stochastic/adaptive knapsack literature treat the uncertain-parameter case of §6.3, including the gain of adaptive over fixed-order policies, which is the issue raised by the paper's own brittleness finding. Orienteering and deadline-TSP address the chained-route alternative (W1). Humanitarian logistics has equity and deprivation-cost objectives (W5).
**Evidence Anchor**: text: §2 "has no analogue in the feasibility-testing literature"
**Why it matters**: The cross-field links are where this paper could say something to practitioners; missing them also leaves the stochastic discussion without the literature that says which policy class to use.
**Suggestion**: Soften the novelty sentence and add a paragraph linking to the overload-scheduling and adaptive-knapsack literature (see reading list); for the stochastic section, say whether the §6.3 example shows an adaptivity gap and which policy class a practitioner should prefer.
**Severity**: Minor
**Confidence**: 4: familiar with these fields; specific references below are from memory and flagged for verification.

### W8: Speed, distance and delay assumptions are optimistic and not tied to evacuation conditions
**Problem**: Straight-line distance is a lower bound on road distance (acknowledged), yet 50 km/h is called "conservative" and 80 km/h "optimistic"; on winding Sierra-foothill roads, and in an active evacuation with inbound-outbound conflict, an average of 80 km/h seems implausible and 50 km/h may not be conservative. The sampled speed range extends to 90 km/h. No road-circuity factor, congestion or smoke effect is modelled, and the dispatch delay is limited to 0 to 20 minutes with the crew assumed ready at timeline zero. §6.2 notes the hazard may degrade the road network but the case study does not use it.
**Evidence Anchor**: text: §5.6 "conservative, winding mountain roads"
**Why it matters**: Feasibility in this problem is decided by a few minutes; systematic optimism in `p_i` biases the results toward serving more sites.
**Suggestion**: Use an actual road network (the authors' co-located instances elsewhere use OSM routing) or a documented circuity factor, narrow the speed range to what the incident record supports, and add a congestion scenario.
**Severity**: Minor
**Confidence**: 4: core expertise in evacuation travel times.

---

## Detailed Comments

### Assumption Audit
- **Explicit assumptions**: Single depot, single vehicle, round-trip structure, known deadlines, integer data, no release dates. The authors state them and examine several relaxations in §6. The round trip is stated as the model's defining feature; the point at which it ceases to be plausible (when sites are closer to each other than to the depot) is not discussed.
- **Implicit assumptions**: (i) A visit has zero or fixed duration and a binary payoff. (ii) Population is the criticality. (iii) Hazard arrival is a cliff, not a gradient, and is independent of the response. (iv) The commander commits to an order at time zero. (v) Utilitarian aggregation is acceptable. (vi) The ordering problem can be separated from forecasting and routing without loss (Intro); in the Camp Fire instance, separating routing from ordering removes the dominant effect (W1).
- **Paradigmatic assumptions**: The "optimise a static plan given a forecast" paradigm of OR, versus the observe-and-adapt paradigm of incident command and of online real-time scheduling. The paper's own sensitivity analysis shows that the static plan is the fragile element.

### Cross-Disciplinary Connections
- **Parallel research**: Real-time overload scheduling (online weighted on-time maximisation, competitive analysis); stochastic and adaptive knapsack (adaptivity gaps); deadline-TSP, orienteering and vehicle routing with deadlines; humanitarian-logistics equity; evacuation planning with clearance times.
- **Borrowing opportunities**: Time-utility functions from real-time systems replace the 0/1 deadline cliff with a decaying benefit, which suits evacuation lead time. Mixed-criticality terminology offers a vocabulary for "criticality weight". Deprivation-cost functions from humanitarian logistics offer an equity-aware objective.
- **Methodological borrowing**: Evaluate against an adaptive (re-planning) baseline, as in online scheduling; use common-factor scenario generation (ensemble fire-spread output) rather than independent perturbations.

### Practical Impact
- **Real-world application**: A single scarce specialised asset (one air tanker, heavy dozer, strike team, line-repair crew, mobile pump) that must return to a base between tasks is a legitimate use, and I would build the motivation around it. For a ground crew in a cluster of communities, chained routing applies. For many resources, the paper only offers the equal-cost matroid case and hardness.
- **Implementation feasibility**: The exact DP is trivial to implement and instantaneous at realistic sizes; the barriers are inputs (reliable per-site arrival times with uncertainty, service times, meaningful weights), not computation. A practitioner would also need online re-optimisation as forecasts update.
- **Stakeholders**: Residents of conceded sites; incident commanders and agencies bearing duty-of-care obligations; emergency managers who would own the weights. None is represented, and no practitioner feedback or validation is reported.

### Broader Implications
- **Ethical dimensions**: See W5. Weight choice is a normative decision with distributional consequences; census population privileges large towns over remote communities and ignores vulnerability.
- **Social impact**: Retrospective optimisation of a real disaster on named communities can be misread; a framing statement is needed.
- **Future directions**: (i) Adaptive MWHED with observed-delay re-optimisation and a proven or empirical adaptivity gap. (ii) MWHED with chained routing (deadline orienteering) and a comparison showing when the round-trip simplification is harmless. (iii) Service times and a time-utility objective. (iv) Fairness-constrained variants and their complexity. (v) A flatter-weight multi-event benchmark.

---

## Cross-Disciplinary Reading Recommendations
None of these were verified against a database in this session; author, year and topic are from memory and should be checked before use. All are tagged [UNVERIFIED] as search leads.
- [UNVERIFIED] Baruah, Koren, Mishra, Raghunathan, Rosier, Shasha (1991), on-line scheduling in the presence of overload; and Koren and Shasha (1995), the D-over algorithm for overloaded uniprocessors, plus their skip-over model. Relevance: the real-time version of "which weighted subset to sacrifice", with competitive ratios for the online case.
- [UNVERIFIED] Jeffay, Stanat, Martel (1991), non-preemptive scheduling of periodic and sporadic tasks. Relevance: a vehicle round trip is non-preemptive, unlike the Liu-Layland setting cited.
- [UNVERIFIED] Jensen, Locke, Tokuda (1985), time-utility functions in real-time scheduling. Relevance: replaces the 0/1 deadline cliff with a lead-time-dependent benefit.
- [UNVERIFIED] Dean, Goemans, Vondrak (2008), approximating the stochastic knapsack problem and the benefit of adaptivity. Relevance: the adaptivity gap for the uncertain-parameter version in §6.3.
- [UNVERIFIED] Bansal, Blum, Chawla, Meyerson (2004), approximation algorithms for deadline-TSP and vehicle routing with time windows; Campbell, Gendreau, Thomas (2011), orienteering with stochastic travel and service times. Relevance: the chained-route model for W1.
- [UNVERIFIED] Huang, Smilowitz, Balcik (2012), relief routing models of equity, efficiency and efficacy; Holguin-Veras et al. (2013), unique features of post-disaster humanitarian logistics (deprivation costs). Relevance: equity-aware objectives for W5.
- [UNVERIFIED] Altay and Green (2006), and Galindo and Batta (2013), reviews of OR/MS in disaster operations management; Bayram (2016), review of network evacuation optimisation. Relevance: positioning against evacuation planning, which the Introduction invokes.
- [UNVERIFIED] The after-action and fire-progression reporting on the 2018 Camp Fire beyond the NIST case study (for evacuation notification, egress and local resources). Relevance: tests the single-depot premise.

---

## Questions for Authors
1. In your Camp Fire instance, if the crew may travel directly between communities (at your own speeds, using the Fig. 5 geometry), which sites are protected and what is the optimal route? How much of the headline result survives?
2. What does a "visit" accomplish, and how long does it take on site? Would the conclusions change if service time grew with population, which makes `p` and `w` strongly correlated?
3. Which event do the 52 and 71 minute timestamps denote in the NIST record (front arrival, first spot fire, first structure ignition), and would using one event type consistently change the Paradise deadline?
4. How would the sensitivity conclusions change if the crew re-planned adaptively as delays were observed, and if arrival-time errors were positively correlated across sites?
5. What is the minimal fairness requirement under which the DP still applies (forced inclusion of designated sites), and which natural equity objectives make the problem harder?

---

## Minor Issues

### Language / Grammar
- §5.6: deadlines appear to round half to even (83.5 to 84, 102.5 to 102); state the rounding rule and note that rounding `p` up and `d` down is the conservative direction for feasibility.
- Abstract and §3: "criticality weight" is used for census population without qualification; consider "exposure weight" or define explicitly.
- Intro, fourth paragraph: the claim that the separation of forecasting from ordering "mirrors how the wildfire operations-research literature ... is typically organized" is contradicted by the following related-work paragraph, which says that literature couples spread and routing tightly.

### Figures and Tables
- Fig. 5 (Camp Fire map): the title says "real geography" but the layout is flat distance from the depot; add road network or at least a road-distance annotation, and add the chained-route alternative.
- Fig. 1 caption uses "sacrificed" for the excluded site; see W5.
- Table of Camp Fire results: add the chained-route and adaptive-policy rows once computed (W1, W2).

### Layout
- A short "what is classical, what is new, what a practitioner should use" box in §1 would help the transportation reader the paper targets.
- Data Availability: code and the Lean formalisation are "available on request"; a public repository would help practitioners and reviewers reuse the instance.
