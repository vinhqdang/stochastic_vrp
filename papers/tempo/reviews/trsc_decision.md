# Transportation Science decision, TS-2026-0425 (November 2026)

**Decision: reject** (Editor Ivana Ljubić, on the AE's recommendation).
Two referees recommended rejection, one a major revision. Referee 3's
report is the PDF next to this file (`trsc_referee3_report.pdf`).

## Associate Editor

I have received three referee reports that offer complementary perspectives on your manuscript. Two reviewers recommend rejection, while one recommends a major revision. Having carefully read the paper alongside these constructive reports, I concur with their assessment and recommend rejecting the manuscript. While the paper introduces an novel idea of leveraging statistical testing to trigger reoptimization decisions, it does not yet establish the necessary connections to the broader vehicle routing literature. It also does not fully substantiate the operational value of recourse using this framework. Below is a summary of the key concerns raised by the referees:

1. Definition of the routing and feasible recourse. Reviewers 1 and 2 request a more precise formulation of the underlying routing problem and its operational constraints. In particular, Reviewer 1 emphasizes that the practical value of replanning depends critically on which deliveries can still be reassigned once vehicles are loaded, as well as the associated cost of reassignment.

2. Operations value of TEMPO alarm. Reviewer 3 observes that TEMPO can accurately detect a deviation even in situations where abiding with the current plan remains the optimal decision ("A harmless deviation can therefore still trigger the alarm"). This observation is valid for whether outing weights help distinguish such cases. Relatedly, Reviewer 2 asks how a dispatcher would estimate the continuation value and replanning costs. I agree that these concerns are critical for the implementation of the proposed method in practice.

3. Fleet performance and benchmarks. Reviewer 2 questions drawing fleet-level conclusions primarily from single-route experiments. Maybe comparisons against tuned periodic replanning and frequent, low-cost replanning policies could address this. Reviewer 3 similarly asks for direct comparisons of realized fleet costs using a consistent replanner across policies. e.g., a threshold policy that intervenes whenever expected savings exceed replanning costs. R2 also recommends evaluating the monitor on held-out routes and testing its robustness when the planning model is misspecified.

4. Positioning in stochastic routing. Reviewers 1 and 3 ask for a clearer positioning of how TEMPO advances the existing literature on vehicle routing with recourse. Specifically, Reviewer 3 highlights restocking and partial reoptimization as particularly relevant benchmarks and notes some citations. In fact, this is the biggest barrier to improving this paper, and I concur with reviewers on this comment.

## Referee 1 (reject)

The authors consider a vehicle routing problem with srveral stochastic components: travel time, demand, dwell time, accidents, and breakdowns. In this context it may be advisable to re-optimize the routes at some point. In the words of the authors, the question asked is "when" to reoptimize as opposed to "how". The authors develop a monitor called TEMPO to answer this question.

There are several important problems with this paper. First, the precise nature of the routing problem under study is never specified. What are the constraints of this problem? What is stochastic and how is stochasticity measured? For example, how does one measure the probablities of accidents? Second, there is indeed a cost to re-optimizing too soon or too late, but this cost has to be compared with the cost of the recourse action. Unfortunately, the recourse actions and the reoptimization cost are not cleary explicited in the paper. In fact, the "how" is also important in the context of the research question and cannot be ignored, but it is not dealt with. For example, if the vehicles already on the road are loaded with the demands of their customers, how does the re-optimization work if these vehicles are sent to different customers? These basic questions and other important ones are never answered in the paper but their answers are essential for its understanding.

In my view, the contribution of this paper to the field of stochastic routing cannot be ascertained from the information provided. The emphasis of this work is very much on the developments of theoretical results. The paper is highly unclear and not sufficiently well embedded within the classical literature on stochastic routing, and it has many loose ends. Clearly this is not a good fit for TS.

## Referee 2 (major revision)

Summary, contributions and strengths/weaknesses as in the portal. Major points:

- The routing problem is never explicitly defined (variant, customers, vehicles, capacities, route lengths); the grid uses only "the longest route" of each plan, so most of the study is single-vehicle while the abstract and Section 6 make fleet-level claims; the fleet-pooled monitor appears only in Figure 3 and in the twelve instances of Table 5.
- Unclear how much must be anticipated: the monitor tests only the five modelled channels against a correctly specified conditional null; the construction requires a planning layer that already fits a continuation cost, which a fleet running periodic re-solves does not have. Discuss what a fleet must possess before deployment and what happens when parts are missing or wrong.
- The economic quantities (Delta_c, g, C_fr, kappa_0) are asserted to be available, not shown to be; no estimation procedure; are they available for any method and VRP variant; what is the cost of estimating them?
- Missing natural competitors: a periodic policy with a period tuned on separate days; tuned-periodic + rebalance row in Table 5; continuous cheap re-optimisation (local moves at every event, or re-optimising one route) in realised cost.
- Theorem 2 ("acting on a statistical alarm can beat acting sooner") must be qualified when re-optimisation is flexible: add a replan-continuously policy or state that the dominance is specific to one-shot costly interventions.
- Every evaluation day, including the Amazon pilot, is generated from the paper's own model; the Amazon null is fitted on the same 50 routes it is evaluated on; fit the null on separate training data, study how much data is needed, and test a deliberately misfitted null.

Minor: broken spacing around hyphens/dashes ("deep-reinforcement- learning", "graph-neural- network", "single- rebalancing- decision", "Miller–Tucker– Zemlin", "cheapest- insertion"); Appendix sections numbered .1–.16.

Recommendation: Major Revision.

## Referee 3 (see PDF)

Main text only. (1) The coupling to routing economics is not supported: the test still asks whether the day follows the model, so harmless deviations trigger the alarm; the false-alarm guarantee says nothing about replanning on days whose deviations do not justify it; weights are simple proxies; the ablation shows the weights alone reduce demand detection; no with/without-weights comparison; no harmless-change scenarios. (2) The evaluation does not measure routing performance: compare routing cost (distance, lateness, breaches, number of replans) against fixed-schedule, every-few-stops and cost-based-trigger policies, all scenarios, whole fleets, same replanner, realistic nonzero replan cost; estimate the economic quantities and check the predicted best significance level. (3) Not positioned against VRPSD with restocking (Yang, Mathur, Ballou 2000), reoptimisation (Secomandi & Margot 2009), partial reoptimisation (Florio, Feillet, Poggi, Vidal 2022); TEMPO's natural contribution could be to detect when the known-distribution assumption fails. Also: writing hard to read, signs of unchecked AI-produced text, undefined internal labels, and a non-existent reference (Gmira, Glize, Hoenig, Rousseau 2021; the real paper is Gmira, Gendreau, Lodi, Potvin 2021, TR-C 132:103379).
