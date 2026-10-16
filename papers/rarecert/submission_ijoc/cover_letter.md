Dear Editors,

Please consider the manuscript "Certifying rare overflow risk of regional pickup-and-delivery plans" for the INFORMS Journal on Computing.

**What the paper does.** Distribution of medical, cold-chain and dangerous goods often requires that a routing plan overflow its vehicles with probability at most a service level between 10^-3 and 10^-6. We study how to verify such a level for a given plan of the vehicle routing problem with simultaneous pickup and delivery, under gamma demands with a Gaussian factor copula. We show that a route overflows through one of two end modes, that a single importance-sampling shift misses the weaker one and underestimates by about 0.8 times its computed mass, and that a mixture at the two computed dominating points removes the bias without pilot simulation. The method extends to route days with stochastic travel times and to hubs whose routes share demand factors. It is checked on 148 instances (6,242 routes) from Northern Vietnam and Guangdong against a quadrature reference, crude Monte Carlo, and budget-fair classical rivals (cross-entropy mixtures, Glasserman-Li, subset simulation), including coverage studies at 10^-5 and 10^-6. The plans we certify, built for a per-route risk of 0.10, do not meet any level at their planned capacity and need 35-76% more capacity. A short section tests quantum amplitude estimation on the same problem and reports that it does not beat the classical estimator here.

**Why IJOC.** The contribution is a computational method with a theory of when it works, a reproducible implementation, and a large experimental study; this is the combination the journal asks for. The paper sits between simulation methodology (rare-event estimation, variance reduction, confidence intervals) and its use inside operations research (routing plans, chance constraints). IJOC has published work in this line, for example Homem-de-Mello, "A study on the cross-entropy method for rare-event probability estimation", INFORMS Journal on Computing 19(3):381-394 (2007). [AUTHOR TO ADD: two or three recent IJOC papers on rare-event or stochastic-routing computation, checked on pubsonline.informs.org.] All code, instances, plans and the scripts behind every table and figure are public, and we will release them under the IJOC software policy.

**Area.** We submit to the area "Simulation, Stochastic Models, & Stochastic Optimization": the core of the paper is importance-sampling estimation with certified intervals for a stochastic-model event. [AUTHOR TO CONFIRM against the current editorial statement page.]

**Related manuscripts and overlap.** The submission uses the instances and plan files that also appear in two of our other manuscripts, one under review at Computers & Operations Research (online mid-route recourse) and one under review at Transportation Science (e-process monitoring). Their results are different: this paper contains no recourse policy and no e-values, and reports no number taken from them. A separate manuscript on quantum amplitude estimation for stochastic routing is in preparation; the quantum section here is a short companion check and its conclusion is a negative one. This manuscript has not been submitted elsewhere.

**Funding and conflicts.** [AUTHOR TO COMPLETE: funding statement, or "none".] We declare no conflict of interest.

Sincerely,
Quang-Vinh Dang
British University Vietnam, Hung Yen, Vietnam
vinh.dq4@buv.edu.vn, ORCID 0000-0002-3877-8024
