# RARECERT — citations to verify

Snippet-level checks from R0 (2026-10-14). Nothing here is cited until read in full.

## Importance sampling for unions of rare events

# Literature check: IS for unions of rare events, CE with mixtures, efficiency theory

Verification key: VERIFIED = bibliographic data seen on a publisher, arXiv or index page in this pass. SNIPPET = seen only in search snippets. UNVERIFIED = recalled or not found; check before citing.

## 1. Union of rare events (ALOE)

- Owen, A.B., Maximov, Y., Chertkov, M. "Importance sampling the union of rare events with an application to power systems analysis." Electronic Journal of Statistics 13(1), 231-254, 2019. DOI 10.1214/18-EJS1527. arXiv:1710.06965 (v4, Dec 2018). VERIFIED.
  - Method: pick one of the J events uniformly at random, sample x conditionally on that event, form an unbiased estimate of the union probability (the ALOE sampler; the name comes from the paper's own usage, which I saw only in a snippet).
  - Guarantee (from arXiv abstract, VERIFIED): CV of the estimator is at most sqrt((J + 1/J - 2)/(4n)) for n samples, regardless of overlap among the J events. Relative variance is therefore at most (J + 1/J - 2)/(4n), with no dependence on the event probabilities. This is a bound that is uniform in rarity, not a bounded-relative-error statement in the usual sense.
  - NOT VERIFIED: the exact theorem wording, its hypotheses, and whether the bound is stated for half-spaces under Gaussian inputs. The abstract motivates the problem with joint Gaussian phase differences but does not state Gaussianity as a hypothesis. Read the full PDF (projecteuclid 18-EJS1527) before quoting any hypothesis.
- Karp, R.M., Luby, M. (1983), the classical union-of-events estimator, cited in the Owen et al. abstract as the origin of this sampler family. SNIPPET.

## 2. Cross-entropy and mixtures

- Rubinstein, R.Y., Kroese, D.P. (2004). The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation and Machine Learning. Springer, Information Science and Statistics. DOI 10.1007/978-1-4757-4321-0. ISBN 978-0-387-21240-1. VERIFIED (publisher page). A 2016 softcover reprint exists (ISBN 978-1-4419-1940-3); the date 2016 is from memory.
- de Boer, P.-T., Kroese, D.P., Mannor, S., Rubinstein, R.Y. (2005). "A Tutorial on the Cross-Entropy Method." Annals of Operations Research 134(1), 19-67. DOI 10.1007/s10479-005-5724-z. VERIFIED (RePEc). Some bibliographic records give 2004; use 2005.
- Kurtz, N., Song, J. (2013). "Cross-entropy-based adaptive importance sampling using Gaussian mixture." Structural Safety 42, 35-44. SNIPPET (several sources). DOI UNVERIFIED. I did not see the abstract.
- Geyer, S., Papaioannou, I., Straub, D. (2019). "Cross entropy-based importance sampling using Gaussian densities revisited." Structural Safety 76, 15-27. DOI 10.1016/j.strusafe.2018.07.001. VERIFIED (TUM portal, search). Shows that single Gaussian and Gaussian mixture CE proposals are tested; proposes a modified weighted EM and DBSCAN-based component selection.
- Papaioannou, I., Geyer, S., Straub, D. (2019). "Improved cross entropy-based importance sampling with a flexible mixture model." Reliability Engineering & System Safety 191, 106564. DOI 10.1016/j.ress.2019.106564. VERIFIED (IDEAS/RePEc). Flexible mixture (von Mises-Fisher-Nakagami mixture) for multimodal failure domains. Author order differs between sources; the journal order is Papaioannou first.

## 3. Efficiency theory and portfolio/copula IS

- Glasserman, P., Li, J. (2005). "Importance Sampling for Portfolio Credit Risk." Management Science 51(11), 1643-1656. DOI 10.1287/mnsc.1050.0415. VERIFIED (INFORMS URL; pages from search). Normal copula, two-step IS: shift the common factors, then apply conditional exponential twisting to the defaults given the factors. The 2003 working-paper version is the one I read.
  - Log-efficiency or asymptotic optimality claims: UNVERIFIED for the journal version. Check Section on asymptotics before claiming or citing them as a result.
- Glasserman, P., Kang, W., Shahabuddin, P. (2008). Multifactor extension of the same two-step scheme (mean and covariance changes on factors). SNIPPET only via citing papers; title, venue and year UNVERIFIED.
- Bassamboo, A., Juneja, S., Zeevi, A. (2008). "Portfolio credit risk with extremal dependence: asymptotic analysis and efficient simulation." Operations Research 56(3), 593-606. DOI 10.1287/opre.1080.0513. VERIFIED (INFORMS URL). Sharp asymptotics for t-copula-type (extremal dependence) models and asymptotically optimal IS. Earlier version: Bassamboo, Juneja, Zeevi, "Expected shortfall in credit portfolios with extremal dependence," WSC 2005. SNIPPET.
- Chan, J.C.C., Kroese, D.P. (2010). "Efficient estimation of large portfolio loss probabilities in t-copula models." European Journal of Operational Research 205(2), 361-367. VERIFIED (RePEc). DOI UNVERIFIED. Two conditional MC estimators for the t-copula; the less efficient one is shown to have bounded relative error (from abstract snippet). Also Scott & Metzler (2015) compared an IS method with this one.
- Dupuis, P., Wang, H. (2007). "Subsolutions of an Isaacs equation and efficient schemes for importance sampling." Mathematics of Operations Research 32(3), 723-757. DOI 10.1287/moor.1070.0266. VERIFIED (INFORMS URL). Dynamic IS from classical subsolutions; asymptotic optimality framework. The logarithmic-efficiency wording was not confirmed in the snippet.
- Asmussen, S., Glynn, P.W. (2007). Stochastic Simulation: Algorithms and Analysis. Springer, SMAP 57. DOI 10.1007/978-0-387-69033-9. VERIFIED (Springer/search). Standard reference for IS and efficiency definitions (chapter numbers not checked).
- Bucklew, J.A. (2004). Introduction to Rare Event Simulation. Springer. UNVERIFIED (search returned nothing; publisher and year from memory).
- Juneja, S., Shahabuddin, P. (2006). "Rare-event simulation techniques: an introduction and recent advances." In Henderson & Nelson (eds), Handbooks in OR and MS, vol. 13: Simulation, Elsevier. Chapter title SNIPPET. Authorship of this chapter UNVERIFIED (the search did not show the chapter authors). Check the ScienceDirect TOC.

## 4. Sums and maxima

- Siegmund, D. (1976). "Importance sampling in the Monte Carlo study of sequential tests." Annals of Statistics 4(4), 673-684. UNVERIFIED (not found; the page and year are from memory). A 2008 arXiv review attributes the likelihood-ratio tilting identity to Siegmund (citing his book), which supports the attribution.
- Asmussen, S., Kroese, D.P. (2006). "Improved algorithms for rare event simulation with heavy tails." Advances in Applied Probability 38(2), 545-558. VERIFIED (Cambridge page). DOI UNVERIFIED. Conditional MC on the max-decomposition P(S_n > u) = n P(S_n > u, M_n = Y_n); the first heavy-tailed bounded-relative-error estimator. Hartinger and Kortschak later showed vanishing relative error; Asmussen and Kortschak (2015) computed error rates. SNIPPET.
- Blanchet, J., Glynn, P.W. and related work on heavy-tailed random walks and maxima: not searched in this pass. UNVERIFIED.

## 5. Alternatives

- Au, S.-K., Beck, J.L. (2001). "Estimation of small failure probabilities in high dimensions by subset simulation." Probabilistic Engineering Mechanics 16(4), 263-277. VERIFIED (Caltech record, Oct 2001). DOI 10.1016/S0266-8920(01)00019-4 from memory, UNVERIFIED.
- Multilevel splitting / sequential MC for rare events (Cerou, Del Moral, Furon, Guyader 2012, Statistics and Computing): not searched. UNVERIFIED.

## Established status (what a new paper must cite and cannot claim as new)

(a) Mixture IS over failure modes of a union: ESTABLISHED. The Karp-Luby-type proposal (pick event, sample conditionally, unbiased) and its CV bound (Owen-Maximov-Chertkov 2019). Mixture and multimodal CE proposals for multiple failure regions (Kurtz-Song 2013; Geyer-Papaioannou-Straub 2019; Papaioannou et al. 2019). A new paper can only claim a specific new mixture weighting, a different bound, or a new application.

(b) Factor-shift plus conditional twisting for Gaussian copula models: ESTABLISHED. Glasserman-Li (2005) is the direct prior art; Glasserman-Kang-Shahabuddin (2008) for multifactor versions. Bassamboo-Juneja-Zeevi (2008) and Chan-Kroese (2010) for t-copula. The two-step scheme itself cannot be claimed.

(c) Bounded relative error / log-efficiency proofs in such models: PARTLY ESTABLISHED. Bounded relative error for conditional MC in the t-copula (Chan-Kroese 2010, from abstract). Asymptotic optimality of IS in t-copula portfolios (Bassamboo-Juneja-Zeevi 2008). Bounded relative error for heavy-tailed sums and maxima (Asmussen-Kroese 2006). ALOE's uniform CV bound (Owen et al. 2019) for unions. Not established by this pass, so treat as open for the specific model: log-efficiency for a one-factor Gaussian copula with gamma marginals and a max-over-stages peak-load predicate. Check Glasserman-Li's asymptotic section and Bassamboo-Juneja-Zeevi before making any claim.

## Open items before citing

1. Read the full Owen et al. (2019) PDF to get the exact theorem, hypotheses and whether Gaussian half-spaces are assumed.
2. Check the Glasserman-Li (2005) journal version for asymptotic-efficiency statements.
3. Obtain DOIs for Chan-Kroese 2010, Kurtz-Song 2013, Asmussen-Kroese 2006, Au-Beck 2001, Siegmund 1976, Bucklew 2004.
4. Confirm Juneja-Shahabuddin chapter authorship.

## Rare risk in routing

# Literature check: small route-failure probabilities in VRP (2018-2026 focus)

Status legend: SNIPPET = verified from search snippet/bibliographic record only, full text not opened. UNVERIFIED = recalled or cited second-hand, not confirmed in this session.

## 1. Chance-constrained VRP

- **Laporte, Louveaux & Mercure (1989), "Models and exact solutions for a class of stochastic location-routing problems", EJOR 39(1):71-78.** SNIPPET. Note: this is a location-routing paper in EJOR, not Operations Research as the brief states. The plain VRP work by the same authors is Laporte, Louveaux & Mercure (1992), "The vehicle routing problem with stochastic travel times", Transportation Science 26(3):161-170, which has chance-constrained, 2-index and 3-index recourse models. Evaluation: chance constraint on route/time feasibility; the snippet gives no formula. Correct the citation before use.
- **Dinh, Fukasawa & Luedtke (2018), "Exact algorithms for the chance-constrained vehicle routing problem", Math. Programming 172(1-2):105-138, DOI 10.1007/s10107-017-1151-6.** SNIPPET (DOI from a search snippet, not checked at doi.org). Earlier IPCO 2016 version, LNCS 9682. Evaluation: route-level chance constraint on capacity overflow with correlated demands, branch-and-cut-and-price; pricing is strongly NP-hard, so the route relaxation is solved by DP. The chance constraint is handled exactly for the tested distributions, not by a rare-event estimator. Extends to a distributionally robust version.
- **Ghosal & Wiesemann (2020), "The Distributionally Robust Chance-Constrained Vehicle Routing Problem", Operations Research 68(3):716-732 (per one listing; volume pages given inconsistently).** DOI 10.1287/opre.2019.1924 (SNIPPET, from the INFORMS URL). Evaluation: distributionally robust chance constraint, feasibility with probability at least 1-epsilon over a moment ambiguity set with support plus first and second moment bounds. Shows the constraint is tractable with branch-and-cut when the set is subadditive, and gives cut generation. Not Monte Carlo and not rare-event; the bound is worst-case over the set.
- **Gounaris, Wiesemann & Floudas (2013), "The Robust Capacitated Vehicle Routing Problem Under Demand Uncertainty", Operations Research 61(3):677-693.** SNIPPET. Evaluation: robust (worst case over a box/ellipsoid-type support) counterparts of CVRP formulations with robust rounded capacity inequalities. Links to chance constraints via a risk-of-shortfall interpretation; no probability estimate per route.
- **Ghosal, Ho & Wiesemann (2024), "A Unifying Framework for the Capacitated Vehicle Routing Problem Under Risk and Ambiguity", Operations Research 72(2).** SNIPPET (DBLP listing). Generalises to risk measures and disutility functions. Closest 2024 item found on the DRO side.
- **Noorizadegan & Chen (2018), "Vehicle routing with probabilistic capacity constraints", EJOR 270(2):544-555.** SNIPPET (Northumbria repository). Set-partitioning formulation solved by branch-and-price; sensitivity analysis on the allowed route-failure probability. Route failure evaluated exactly under the stated demand distribution; full text not opened.
- **Chen, Kuhn & Wiesemann (2023), Operations Research Letters** (data-driven chance constraints over Wasserstein balls: CVaR and Bonferroni approximations can be poor for limited data). SNIPPET. Not a VRP paper; it is the general tool a 2024-26 Wasserstein CVRP would use.
- **Sluijk, Florio, Kinable, Dellaert & Van Woensel (2023), Transportation Science** (two-echelon routing, chance constraints on second-echelon capacity with data-driven demand, statistical tests). SNIPPET. Certification-style check of chance constraints, not importance sampling.
- **Chen et al. (2018), "The probabilistic vehicle routing problem with service guarantees", Transportation Research Part E.** SNIPPET. Nonlinear chance-constrained CVRP solved with Monte Carlo simulation for the probability of shortage.
- 2022-2026 DRO/chance VRP: searches surfaced no 2024-2026 EJOR or Transportation Science paper combining Wasserstein ambiguity with route-level chance constraints. Gap, not a proof of absence.

## 2. Stochastic VRPSPD with a mid-route load-peak constraint

- **Dethloff (2001), "Vehicle routing and reverse logistics: the vehicle routing problem with simultaneous delivery and pick-up", OR Spektrum 23.** UNVERIFIED (recalled, not fetched). Deterministic.
- **Wang & Qiu (2012), cross-entropy for simultaneous delivery and pickup with stochastic demands.** SNIPPET. Cross-entropy used as a search heuristic; violation probability not clearly estimated by rare-event methods.
- **Zhu & Sheu (2018), EJOR, SVRPSPD with stochastic demands and failure-specific cooperative recourse.** SNIPPET (reference listing only). Recourse, not a chance constraint.
- **Gazi University Journal of Science (2025), multi-depot SVRPSPD with stochastic pickups under a chance constraint.** SNIPPET. Normal pickup demands; nonlinear constraints linearised; reliability level varied across 135 instances. The violation probability is enforced through the normal quantile, not estimated by simulation.
- **Juan et al. (2011), Transportation Research Part C.** SNIPPET. Monte Carlo estimate of the probability a vehicle runs out of load before finishing its route, plus expected recourse cost; the closest method match to "load-peak evaluated by simulation" found, but for plain stochastic-demand VRPSD (delivery only).
- No paper found that treats the mid-route peak load of simultaneous pickup and delivery as a chance constraint and evaluates it by exact formula, DP, IS or DRO bounds. This is the gap the brief asks about; it is not established as a settled gap.

## 3. Importance sampling / rare-event simulation in logistics and routing

- **Glasserman & Liu (1996), "Rare-event simulation for multistage production-inventory systems", Management Science 42(9):1292-1307.** SNIPPET. Tilted demand so stockouts are not rare; bounded relative-error IS estimators. Closest IS-in-logistics classic.
- **Ridder (2004/2005), cross-entropy for reliability.** SNIPPET. Reliability, not routing.
- **2023 ICPDI/EAI paper on IS for supply-chain reliability and severe disruptions.** SNIPPET only (eudl.eu DOI 10.4108/eai.1-9-2023.2338823). Full text not checked.
- **Information-Collecting VRP for storm response (arXiv 2301.06497, 2023).** SNIPPET. "Importance sampling" here selects among sampled outcomes with a uniform sampling distribution; it is not a rare-event tilt.
- No IS-based certification of a routing plan's overflow or failure probability found in 2018-2026 searches.

## 4. HAZMAT and service-level targets

- **Erkut & Verter (1998), "Modeling of transport risk for hazardous materials", Operations Research 46(5):625-642, DOI 10.1287/opre.46.5.625.** SNIPPET. Risk = sum over segments of accident probability times consequence; compares risk models that pick different paths. The snippet does NOT show a per-km accident probability. The "about 1e-6 per km" figure in the brief is UNVERIFIED. Check the full text before quoting any rate.
- **Erkut, Tjandra & Verter (2007), "Hazardous Materials Transportation", Chapter 9 in Barnhart & Laporte (eds), Handbooks in OR/MS vol. 14, Elsevier, pp. 539-621.** SNIPPET (bibliographic records; book listing dated Dec 2006). Full text not seen.
- **Verter & Kara (2001), Risk Analysis 21(6):1109-1120** (GIS route evaluation; distance, exposure, accident probability). SNIPPET; no numeric standard values in abstract.
- Logistics service-level or failure-probability targets (GDP, pharma cold chain, dangerous goods): no numeric target found. EU GDP Guidelines 2013/C 343/01 and WHO TRS 961 Annex 5 were identified as the places to look for temperature-excursion rules, but were not opened. Result: no source found quantifying an acceptable 1e-3 to 1e-6 failure probability for routing.

## 5. Certification of a given routing plan

- Noorizadegan & Chen (2018) evaluates route reliability for a set of chosen routes, but as part of optimisation.
- Juan et al. (2011) and Chen et al. (2018) estimate route shortage/failure probabilities by Monte Carlo for given or optimised plans.
- Sluijk et al. (2023) runs statistical tests on chance constraints for data-driven demand.
- No paper found that certifies a given (already optimised) plan with importance sampling or sequential e-values.

## Conclusion

Searches found no paper that uses importance sampling or rare-event simulation to certify route overflow or peak-load failure probabilities of VRP/VRPSPD plans in 2018-2026. Closest:
1. Juan et al. (2011, TR-C): Monte Carlo route-failure estimate for a given plan (not IS; not pickup-and-delivery).
2. Chen et al. (2018, TR-E): Monte Carlo inside a chance-constrained CVRP.
3. Noorizadegan & Chen (2018, EJOR): exact route-failure probability with sensitivity to the risk level.
4. Glasserman & Liu (1996, Management Science): IS for stockout rarity in production-inventory, the IS-in-logistics precedent.
5. Dinh, Fukasawa & Luedtke (2018) and Ghosal & Wiesemann (2020): exact and DRO treatments of route-level chance constraints.

Caveats: most items are checked only from search snippets. Confirm DOIs at doi.org and full texts before citing. The 1989 Laporte et al. citation needs correcting (EJOR location-routing; the VRP paper is TS 1992). The 1e-6 HAZMAT rate and any GDP-type target remain unverified.
