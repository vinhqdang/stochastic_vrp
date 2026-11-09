# SORTIE literature base: fair stochastic vehicle routing

Compiled 2026-11-09. Companion file: `references.bib` (96 entries, all checked against
Crossref / DOI resolver / arXiv API; keys match the headings below).

Verification standard: title, all authors, venue, year, volume/pages and DOI or arXiv id were
taken from Crossref metadata or the arXiv API. Summaries are based on abstracts (Crossref, arXiv,
publisher snippets) and, for the starting preprint and its follow-up, the full text. Where I saw
only title and metadata, the summary says so; do not cite a claimed content point beyond that
without reading the paper. Items marked **(core)** are the ones I would cite first; the rest are
supporting or background.

Legend: Unc = handles uncertainty; Fair = handles fairness/equity; Ord = ordered/OWA-type measure;
TSPopt = TSP-optimality / route consistency in the sense of Ljubić et al.

---

## Starting point (read in full)

**ljubic2026fair (core)**. Ljubić, Puerto, Torrejón, *Fair Vehicle Routing via Bilevel Optimization*,
Optimization Online, submitted 5 March 2026. Fairness through objectives on the sorted vector of
route lengths (min-max, range, Gini mean difference, variance, absolute deviation; monotone and
non-monotone), with a bilevel model that forces every route to be a shortest tour of its customer
set (per-vehicle TSP-optimality). Unc: none. Its conclusion lists "uncertain demands" as future work
and never mentions uncertain travel times; the only stochastic-routing reference is Laporte and
Louveaux (1998). Fair, Ord, TSPopt: yes (deterministic). Full text read (version at
`optimization-online.org/wp-content/uploads/2026/03/FairVRP_new.pdf`). Its reference list:
Agius 2026, Bruinink 2025, Bektaş and Letchford 2020, Karsu and Morton 2015, Hooker and Williams
2012, Tsang and Shehadeh 2025/26, van Rossum 2026a/b, Blanco et al. 2024 and 2026, Lozano 2016,
López Sánchez 2024, Ljubić et al. 2026, and others used below.

---

## Cluster 1. Workload/route balance and fairness in deterministic VRP

**matl2018workload (core)**. Matl, Hartl, Vidal (Transportation Science 52(2):239-260). The
standard survey and computational analysis of equity measures for route workload (range, min-max,
Gini, mean absolute deviation, and so on), showing that measures differ in monotonicity and in how
they behave on benchmark instances. Fair, Ord (measures over the vector of route workloads); no
uncertainty; it flags the inconsistency of non-monotone measures that Ljubić et al. then formalise.

**matl2019impact (core)**. Matl, Hartl, Vidal (Computers & OR 110:116-129). Studies how the
choice of workload resource (distance, duration, load, number of customers) changes the optimal
balanced plan and the cost of equity. Fair, deterministic. Relevant because under random durations the
"workload resource" itself becomes a random variable, which this paper does not consider.

**matl2019leveraging**. Matl, Hartl, Vidal (Networks 73(4):382-400). Heuristic box splitting: uses
a single-objective heuristic repeatedly to approximate the Pareto front of a bi-objective problem,
applied to routing with cost versus balance. Fair, bi-criteria; deterministic. Method precedent for
a cost-versus-fairness Pareto sweep.

**vanrossum2024branching**. van Rossum, Chen, Lodi (IPCO 2024, LNCS, pp. 433-445). Branching rule
for range minimisation problems (conference version of the next entry). Fair (range), exact method;
deterministic.

**vanrossum2026efficient (core)**. van Rossum, Chen, Lodi (Math. Programming 216(1-2):539-572).
Branching rules for range and order-based objectives, giving exact solution of the range
objective in routing. Fair, Ord; deterministic. Cited by Ljubić et al. as 2026a.

**vanrossum2026enforcing (core)**. van Rossum, Chen, Lodi, *Enforcing TSP-optimality in fair vehicle
routing by cutting planes* (Operations Research Letters 70:107539; arXiv 2604.23748). Branch-price-and-cut
for the range objective with cuts that forbid arc sequences dominated by a shorter tour of the same
customers; instances up to 25 customers. Fair, TSPopt; deterministic (abstract and reference list read).
Cited by Ljubić et al. as 2026b. Crossref dates the volume January 2027; the preprint cites 2026.

**ljubic2026ordered**. Ljubić, Marín, Puerto, Temprano (Networks 88(1):59-84). The ordered median
TSP: a single tour whose edge costs are weighted by rank. Ord; deterministic; single vehicle, so no
inter-driver fairness or TSP-consistency issue. Source of the ordered-measure machinery.

**ljubic2024benders**. Ljubić, Pozo, Puerto, Torrejón (EJOR 317(3):858-874). Benders decomposition
for the discrete ordered median problem (location, not routing). Ord; deterministic. Background for
sorting-based formulations.

**blanco2024intra**. Blanco, Marín, Puerto (Computers & OR 162:106487). Equity measures inside
facility location, including ordered-measure formulations. Fair, Ord; deterministic; location.

**blanco2026optimization (core)**. Blanco, Pozo, Puerto, Torrejón (Annals of OR 364(3):2799-2834).
Optimisation-based computation and integration of ordered measures (linear and quadratic ordered
measures; Ljubić et al. use it to write variance and absolute deviation as ordered measures). Ord;
deterministic as far as the title and reference list indicate (abstract not retrieved).

**lozano2016statistical**. Lozano, González-Gurrola, Rodríguez-Tello, Lacomme (MICAI 2016,
pp. 130-135). Statistical comparison of balance objective functions for the VRP with route balancing.
Fair; deterministic; heuristic study. This is "Lozano et al. 2016" in the preprint.

**lopezsanchez2024fair (core)**. López Sánchez, Lujak, Semet, Billhardt (AI Communications
37(2):223-245). Fair and efficient routing of cooperative fleets: mTSP variants maximising egalitarian
and elitist social welfare, plus a procedure to balance fleet efficiency and fairness. Fair; deterministic.
This is "López Sánchez et al. 2024" in the preprint.

**agius2026solution (core)**. Agius, Absi, Feillet, Garaix (Annals of OR 359(3):2309-2349). Exact
algorithms for a VRP with route-cost equity constraints. Fair; deterministic. Closest exact-method
neighbour of the range objective.

**bruinink2025beyond**. Bruinink, Berghman, Dollevoet (Operations Research Letters 63:107344). Cost
effects of prioritising driver satisfaction in routing. Fair; deterministic (title/metadata only).

**bhadoriya2026equitable (core)**. Bhadoriya, Deka, Sundar (J. Operational Research Society, online
2026; arXiv 2404.08157). Parametric fairness variants of the mTSP (epsilon-Fair via a
second-order cone program and Delta-Fair, which bounds the Gini coefficient of tour lengths with a
linear constraint). Fair, Ord-like (Gini); deterministic. In the mTSP each tour is optimised per assigned
set only implicitly, so the TSP-consistency issue is not treated as such.

**nekooghadirli2026workload**. Nekooghadirli, Gendreau, Potvin, Vidal (ITOR 33(1):245-267; arXiv
2206.14596). Workload equity over several periods: produce cheapest plans per period, then allocate
routes to drivers to equalise cumulative workload. Fair; deterministic. Relevant as the ex-ante
analogue in time: equity of cumulative workload rather than of one day.

**mahmoodian2025equitable**. Mahmoodian, Charkhgard, Dayarian (Production and Operations Management
34(7):1875-1900). Nash social welfare for workload allocation with heterogeneous drivers under a cost
cap. Fair; abstract does not mention uncertainty.

**sobhanan2026equity**. Sobhanan, Charkhgard, Dayarian (POM 35(8):3180-3203). Equity-oriented
workload and pay framework for crowdsourced last-mile delivery. Fair; the retrieved part of the
abstract does not mention uncertainty (not checked beyond that).

**applegate2002solution**. Applegate, Cook, Dash, Rohé (INFORMS J. Computing 14(2):132-143). Solves
a large min-max VRP exactly. Fair (min-max); deterministic. Classic benchmark for the max-route objective.

**bertazzi2015minmax**. Bertazzi, Golden, Wang (EJOR 240(2):372-381). Worst-case analysis of min-max
versus min-sum routing: how much longer the longest route can be under min-sum. Fair; deterministic.

**halvorsen2016biobjective**. Halvorsen-Weare, Savelsbergh (EJOR 251(2):451-465). Bi-objective mixed
capacitated general routing with several route-balance criteria. Fair, bi-objective; deterministic.

**bektas2020lp**. Bektaş, Letchford (Computers & OR 120:104975). Using l_p-norms as fairness
measures in combinatorial optimisation, with routing examples. Fair; deterministic.

**ogryczak2003minimizing**. Ogryczak, Tamir (Information Processing Letters 85(3):117-122).
Minimising the sum of the k largest functions in linear time (top-k sums as LP-representable
building blocks of ordered measures). Ord; no uncertainty. Underlies the LP reformulation of OWA.

**nickel2005location**. Nickel, Puerto, *Location Theory: A Unified Approach* (Springer). The ordered
median framework. Ord; deterministic. Hardcover 2005; eBook 2006 (the preprint cites 2006).

---

## Cluster 2. Fairness, equity or consistency under uncertainty

Only three papers here combine routing, fairness and uncertainty at all, and none treats
travel-time or duration uncertainty with an ordered measure (see GAP).

**kinable2026fair (core)**. Kinable, Sluijk, Gendreau, Rei, Van Woensel (Transportation Science
60(2):264-283). Fair stochastic VRP with partial deliveries: uncertain demand, and fairness as a
minimum expected fill rate per customer, with Rawlsian criteria over customers. Unc, Fair; fairness
is among customers (service), not among drivers (workload); uncertainty is demand. Closest routing paper on the stochastic side.

**gao2026sequential (core)**. Gao, Iravani, Sinclair (arXiv 2606.19278, June 2026). Sequential fair
allocation with routing under demand uncertainty; max-min objectives defined ex-post (per realisation)
and "forward"; the search summary reported ex-ante fairness as disparity of expected fill rates.
Unc, Fair, routing; customers, not drivers; max-min, not ordered measures. Useful for the ex-ante/ex-post vocabulary
in routing (abstract read; ex-ante definition from a search snippet, check the paper).

**gutjahr2021inequity (core)**. Gutjahr (EJOR 288(1):258-270). Inequity-averse stochastic decision
processes: ex-ante versus ex-post inequity-averse objectives, time-consistency (global optimisation
is time-consistent only in the ex-post case), humanitarian application. Unc, Fair; Gini-type
measures; not routing (snippet-level summary, check before quoting details). Best
theory reference for the ex-ante/ex-post distinction in stochastic optimisation.

**mostajabdaveh2019inequity (core)**. Mostajabdaveh, Gutjahr, Salman (IISE Transactions 51(8):809-829).
Shelter location with a weighted mix of ex-ante and ex-post Gini-based inequity; reported by the
paper as the first use of a Gini-derived measure inside a stochastic program. Unc, Fair, Ord (Gini);
location, not routing (snippet-level).

**mostajabdaveh2025branch**. Mostajabdaveh, Salman, Gutjahr (EJOR 324(2):522-537; arXiv 2512.19882).
Branch-and-price for bi-objective routing and allocation in relief distribution: Gini index of unsatisfied
demand versus travel time. Fair, Ord (Gini), routing; deterministic data (abstract read), fairness is among shelters.

**alem2022revisiting**. Alem, Caunhye, Moreno (Socio-Economic Planning Sciences 82:101312). Gini-based
equity in two-stage stochastic humanitarian logistics (location-allocation). Unc, Fair, Ord (Gini);
not routing (snippet-level).

**shehadeh2023reducing**. Shehadeh (Transportation Research Part C 153:104199). Reducing disparities
in transport distance in stochastic facility location. Unc, Fair; location (title/metadata only).

**tsang2026unified (core)**. Tsang, Shehadeh (Operations Research 74(2):1087-1103; arXiv 2211.13427).
A class of convex fairness measures, with stability results on how the choice of measure changes the
solution; cited in the Ljubić preprint. Fair, Ord-like; the framework is general, used in deterministic and stochastic
settings by the authors elsewhere (not verified here). Candidate source for convexity/stability arguments.

**ye2024distributionally**. Ye, Hanasusanto, Xie (arXiv 2402.01872). Distributionally fair stochastic
optimisation using Wasserstein distance. Unc, Fair; about fairness of decisions across groups in a learning
and stochastic-optimisation setting, not routing. Preprint only.

**anh2022workload (core)**. Nguyen Duy Anh (Asia-Pacific J. OR 39(5):2150045; arXiv 2103.04166).
Distributionally robust task assignment that reduces the difference in total working time among
workers under uncertain service times, with a joint chance constraint approximated by worst-case CVaR.
Unc, Fair (balance up to a threshold with high probability); assignment, not routing. Nearest
chance-constrained workload balance paper found.

**lin2022robust (core)**. Lin, Bakker, Leung (Scientific Reports 12:21546). Two-echelon routing for elderly
meal delivery with efficiency and workload fairness among workers; robust optimisation for service-time
uncertainty. Unc, Fair, routing. Fairness is modelled with robust constraints on a deterministic-style balance measure, not ordered
statistics of random durations (abstract read).

**wu2025exact**. Wu, Jin, Bo (Computers & Industrial Engineering 202:110940). Workload-consistent
VRP with priority distribution and demand uncertainty (budget-of-uncertainty robust model; abstract not
retrieved, summary from a search snippet). Unc, Fair-ish/consistency; demand uncertainty.

**alvarez2024consistent (core)**. Alvarez, Cordeau, Jans (Transportation Research Part B 186:102968).
Consistent VRP with stochastic customers and demands: two-stage stochastic program, first-stage
driver-customer assignments, penalties for consistency violations, SAA with branch-and-cut and Benders. Unc,
consistency in the driver-customer sense; not travel-time uncertainty and not workload equity.

**groer2009consistent (core)**. Groër, Golden, Wasil (Manufacturing & Service Operations Management
11(4):630-643). Introduces the consistent VRP (same driver, similar arrival times). Consistency of
service, deterministic. Note: this "consistency" differs from the TSP-optimality consistency of
Ljubić et al. (see GAP).

**kovacs2015generalized**. Kovacs, Golden, Hartl, Parragh (Transportation Science 49(4):796-816).
Generalised consistent VRP (ConVRP with several consistency features). Deterministic.

**yu2024consistent**. Yu, Hu, Wu (Computers & Industrial Engineering 187:109803). Consistent VRP
with driver equity and flexible route consistency. Fair + consistency; the abstract excerpt I found does not
mention uncertainty (not confirmed either way).

**singh2023fairassign (core)**. Singh, Das, Chakraborty (FAccT 2023, pp. 753-763). FairAssign: dynamic
assignment of gig delivery drivers to zones through probability distributions so that similar drivers
get statistically similar distributions (stochastic fairness via dependent rounding). Fair, randomised
rounding, ex-ante flavour on assignment probabilities; not routing durations.

**basik2021fair**. Basik, Gedik, Ferhatosmanoglu, Wu (IEEE Trans. Services Computing 14(4):1040-1053;
arXiv 1807.02987). Fair task allocation in crowdsourced delivery. Fair; deterministic (title/metadata only).

**hosseini2025algorithmic**. Hosseini, Schierreich (arXiv 2503.16002). Algorithmic landscape of fair and
efficient distribution of delivery orders in the gig economy (fair-division view; abstract not read,
title/metadata only). Preprint only.

**nanda2020balancing**. Nanda, Xu, Sankararaman, Dickerson, Srinivasan (AAAI 2020, 34(2):2210-2217; arXiv
1912.08388). Profit versus fairness for riders and drivers in rideshare during high-demand hours,
with stochastic arrivals. Fair, Unc; matching, not routing.

**sinclair2023sequential**. Sinclair, Jain, Banerjee, Yu (Operations Research 71(5):1689-1705; arXiv
2011.14382). Sequential fair allocation under stochastic demand: optimal envy-efficiency trade-off.
Unc, Fair; allocation, no routing. Foundation for the ex-ante/ex-post allocation line that Gao et al. extend to routing.

**karsu2015inequity (core)**. Karsu, Morton (EJOR 245(2):343-359). Survey of inequity-averse
optimisation in OR (Lorenz dominance, OWA, Gini, and so on). Fair, Ord; deterministic.

**fleurbaey2012exante**. Fleurbaey, Peragine (Economica 80(317):118-130). Ex ante versus ex post equality
of opportunity (welfare economics). Conceptual source for the two fairness notions.

**barak2026exante**. Barak, Talgam-Cohen (arXiv 2608.14499, Aug 2026). Facility-location mechanism design;
ex-ante cost is the maximum of expected agent costs, ex-post the expected maximum. Mechanism design,
not OR routing; included only because it fixes the same two definitions in the form used in this project (ex-ante: measure of expected loads; ex-post: expected measure).

---

## Cluster 3. Stochastic travel and service times in VRP; reliable routing

**laporte1992vehicle (core)**. Laporte, Louveaux, Mercure (Transportation Science 26(3):161-170).
VRP with stochastic travel times: chance-constrained and recourse models with a-priori routes. Unc;
no fairness.

**kenyon2003stochastic (core)**. Kenyon, Morton (Transportation Science 37(1):69-82). Stochastic VRP
with random travel times, including a min-max type objective on expected/quantile makespan. Unc; per
Bakach et al. it is the only min-max routing problem with stochastic travel times they found (claim from that paper's literature review).

**gomez2016modeling**. Gómez, Mariño, Akhavan-Tabatabaei, Medaglia, Mendoza (Transportation Science
50(2):627-641). Modelling stochastic travel and service times (distribution fitting, correlation) in
routing. Unc.

**adulyasak2016models (core)**. Adulyasak, Jaillet (Transportation Science 50(2):608-626). Stochastic
and robust VRP with deadlines. Unc; compares stochastic and robust formulations; no fairness.

**errico2016apriori**. Errico, Desaulniers, Gendreau, Rei, Rousseau (EJOR 249(1):55-66). A-priori
optimisation with recourse for VRP with hard time windows and stochastic service times. Unc.

**errico2018vehicle**. Same authors (EURO J. Transportation and Logistics 7(3):223-251). Journal companion on
the stochastic-service-time VRPTW. Unc.

**rostami2021branch (core)**. Rostami, Desaulniers, Errico, Lodi (Operations Research 69(2):436-455).
Branch-price-and-cut for VRP with stochastic and correlated travel times (correlation between arcs).
Unc, correlation. Key reference for correlated durations across routes.

**bakach2021solving (core)**. Bakach, Campbell, Ehmke, Urban (EURO J. Transportation and Logistics
10:100029). VRP with stochastic and correlated travel times and makespan objectives. Unc, correlation,
min-max style objective; no fairness measure beyond makespan.

**rajabibahaabadi2021reliable**. Rajabi-Bahaabadi, Shariat-Mohaymany, Babaei, Vigo (Operational Research
21(1):299-330). Reliable VRP in stochastic networks with correlated travel times. Unc, reliability.

**bomboi2022stochastic**. Bomboi, Buchheim, Pruente (4OR 20(2):217-239). Approximate route feasibility
test for stochastic VRPTW with correlated, time-dependent travel times (jointly normal; chance
constraints). Unc, correlation.

**zhang2013stochastic**. Zhang, Lam, Chen (Networks and Spatial Economics 13(4):471-496). Stochastic VRP
with travel-time uncertainty: cost versus customer service trade-off. Unc.

**zhang2021distribution**. Zhang, Li, Sun, Hou (EJOR 292(1):311-326). VRP with distribution uncertainty
in deadlines (distributionally robust). Unc, DRO.

**han2014robust**. Han, Lee, Park (Transportation Science 48(3):373-390). Robust scenario approach for VRP
with uncertain travel times. Unc, robust.

**ghosal2020distributionally**. Ghosal, Wiesemann (Operations Research 68(3):716-732). Distributionally
robust chance-constrained CVRP (uncertainty is customer demand in this paper, per its abstract summary
in the search results). Unc, DRO.

**jaillet2016routing**. Jaillet, Qi, Sim (Operations Research 64(1):186-200). Routing optimisation under
uncertainty (robust, satisficing-type, no distribution needed). Unc.

**ehmke2015ensuring**. Ehmke, Campbell, Urban (EJOR 240(2):539-550). Service-level guarantees with time
windows and stochastic travel times. Unc.

**osoriomora2025riskaverse**. Osorio-Mora, Saldanha-da-Gama, Toth (EJOR 321(3):837-850). Risk-averse
latency location-routing with stochastic travel times (CVaR-type risk). Unc, tail risk; risk, not fairness.

**oyola2018part1**. Oyola, Arntzen, Woodruff (EURO J. Transportation and Logistics 7(3):193-221). Literature
review of stochastic VRP, part I (models). Survey.

**gendreau2016future**. Gendreau, Jabali, Rei (Transportation Science 50(4):1163-1173). Future research
directions in stochastic VRP. Survey.

**verweij2003sample**. Verweij, Ahmed, Kleywegt, Nemhauser, Shapiro (Computational Optimization and
Applications 24(2-3):289-333). SAA applied to stochastic routing; the standard scenario-based method.

**nikolova2014meanrisk (core)**. Nikolova, Stier-Moses (Operations Research 62(2):366-382). Mean-risk
(mean plus standard-deviation type) model for traffic assignment with stochastic travel times; reliable-path
formulation. Unc, reliability; Ord: none.

**sen2001meanvariance**. Sen, Pillai, Joshi, Rathi (Transportation Science 35(1):37-49). Mean-variance model
for route guidance with correlated link times. Unc.

---

## Cluster 4. Fairness and risk theory

**yager1988owa (core)**. Yager (IEEE Trans. Systems, Man, Cybernetics 18(1):183-190). Defines ordered
weighted averaging. Ord. Foundation of ordered measures (min-max, range, Gini are special weight vectors).

**ogryczak2003solving (core)**. Ogryczak, Śliwiński (EJOR 148(1):80-91). Linear programming
formulations for OWA objectives with sorted weights (monotone weights). Ord.

**ogryczak2000inequality**. Ogryczak (EJOR 122(2):374-391). Inequality measures and equitable approaches
to location. Fair, Ord; deterministic.

**ogryczak2014fair**. Ogryczak, Luss, Pióro, Nace, Tomaszewski (J. Applied Mathematics 2014:612018). Survey
of fair optimisation for networks (max-min fairness, OWA, alpha-fairness). Fair.

**rockafellar2000optimization (core)**. Rockafellar, Uryasev (Journal of Risk 2(3):21-41). CVaR optimisation by
linear programming. Risk; basis for CVaR of stochastic loads.

**rockafellar2002conditional (core)**. Rockafellar, Uryasev (J. Banking & Finance 26(7):1443-1471). CVaR for
general loss distributions (atoms included, relevant for discrete scenario sets).

**bertsimas2011price (core)**. Bertsimas, Farias, Trichakis (Operations Research 59(1):17-31). Price of fairness
for proportional fairness and max-min in resource allocation.

**mo2000fair**. Mo, Walrand (IEEE/ACM Trans. Networking 8(5):556-567). Alpha-fairness family.

---

## Cluster 5. Order statistics of independent non-identically distributed variables

**bapat1989order (core)**. Bapat, Beg (Sankhyā A 51(1):79-93). Distribution function of one or several order
statistics of independent non-identical variables as sums of permanents (Bapat-Beg theorem). No DOI;
bibliographic data confirmed from the Indian Academy of Sciences repository listing (JSTOR id given in
the note field); the full text was not available to me. Per a secondary source it is computationally heavy
because it involves exponentially many permanents (Glueck et al.). Directly relevant to expected ordered
measures of non-identical route durations.

**balakrishnan2007permanents (core)**. Balakrishnan (Rev. Mat. Complut. 20(1):7-107). Survey linking
permanents, order statistics of INID variables, outliers and robustness (pages confirmed from the journal
page).

**david2003order**. David, Nagaraja, *Order Statistics*, 3rd ed. (Wiley). Standard text; chapter on non-identically distributed variables.

**vaughan1972permanent**. Vaughan, Venables (J. Royal Statistical Society B 34(2):308-310). Permanent
expressions for order-statistic densities.

**sen1970note**. Sen (Annals of Mathematical Statistics 41(6):2137-2139). Order statistics for heterogeneous
distributions (bounds relative to the identical case).

**glueck2008fast**. Glueck, Karimpour-Fard, Mandel, Hunter, Muller (Communications in Statistics, Theory
and Methods 37(18):2815-2824). Block-permanent computation of order-statistic CDFs from several populations.

**hoeffding1956distribution**. Hoeffding (Annals of Mathematical Statistics 27(3):713-721). Number of successes
in independent non-identical trials (Poisson-binomial; extremal results).

**hong2013computing**. Hong (Computational Statistics & Data Analysis 59:41-51). Exact and approximate
computation of the Poisson-binomial CDF.

**fernandez2010closed**. Fernandez, Williams (IEEE Trans. Aerospace and Electronic Systems 46(2):803-817).
Closed-form Poisson-binomial pdf.

**biscarri2018simple**. Biscarri, Zhao, Brunner (Computational Statistics & Data Analysis 122:92-100).
Fast algorithm for the Poisson-binomial distribution function.

---

## Cluster 6. Parametric and bi-criteria optimisation, DP for TSP

**eisner1976mathematical (core)**. Eisner, Severance (J. ACM 23(4):619-635). Record segmentation in shared
databases; contains parametric analysis. I could not find a source stating that this paper treats
"parametric combinatorial optimisation" in general; the DOI, authors and pages are confirmed, but the
framing in the brief is not (treat the attribution as the user's, and read the paper before citing it
for supported efficient solutions).

**ehrgott2005multicriteria (core)**. Ehrgott, *Multicriteria Optimization*, 2nd ed. (Springer). DOI resolves;
the DOI metadata lacks the author field (the first edition is 2000, DOI 10.1007/978-3-662-22199-0).

**ehrgott2000survey**. Ehrgott, Gandibleux (OR Spectrum 22(4):425-460). Survey of multiobjective combinatorial optimisation.

**carstensen1983complexity**. Carstensen (Math. Programming 26(1):64-75). Complexity of parametric integer and network programming problems (number of breakpoints).

**ralphs2006improved**. Ralphs, Saltzman, Wiecek (Annals of OR 147(1):43-70). Biobjective integer programs.

**heldkarp1962dynamic (core)**. Held, Karp (J. SIAM 10(1):196-210). The subset DP for TSP; there is also a 1961 conference version.

**bellman1962dynamic**. Bellman (J. ACM 9(1):61-63). Independent DP treatment of the TSP.

---

## GAP

### G1. Ordered/Gini/range measures of RANDOM route durations: ex-ante versus ex-post

Finding: I found no paper that optimises an ordered measure (min-max, range, Gini mean difference,
variance, MAD or OWA) of random route durations in a routing problem, and none that separates
(a) ex-ante fairness, the measure applied to the vector of expected workloads, from (b) ex-post expected
fairness, the expectation of the measure applied to the realised sorted durations (E[range],
E[Gini], E over order statistics).

Evidence:
- The Ljubić et al. preprint (full text, reference list) is deterministic; "uncertain demands" is listed only
  as future work, and no stochastic-travel-time reference appears.
- Closest on the routing side: Kinable et al. 2026 (stochastic demand, fairness among customers, expected fill
  rate threshold and Rawlsian criterion) and Gao-Iravani-Sinclair 2026 (arXiv, max-min allocation plus routing, ex-post
  and forward objectives). Both treat fairness among customers or locations, not drivers' durations, and neither uses
  ordered measures.
- Closest on the driver side: Anh 2022 (assignment, joint chance constraint on workload difference, no routing) and Lin et al. 2022
  (robust two-echelon routing with workload fairness; robust constraints, not order statistics).
- The ex-ante/ex-post distinction with Gini exists in stochastic location (Mostajabdaveh et al. 2019; Gutjahr 2021;
  Alem et al. 2022; Shehadeh and Snyder), not in routing.
- Searches that returned nothing relevant (WebSearch standard/extended, Crossref bibliographic queries, arXiv
  API, run 2026-11-09): "fair vehicle routing stochastic travel times workload equity drivers";
  "equitable routing stochastic demand ex-ante ex-post expected Gini vehicle routing";
  "stochastic vehicle routing Gini range ordered median random route durations";
  "min-max vehicle routing stochastic travel times expected maximum route length" (only Kenyon-Morton 2003 and
  Bakach et al. 2021, both makespan/min-max, no fairness distribution);
  "workload balancing vehicle routing uncertain service times chance constraint" (Anh 2022 and Lin 2022 only);
  Crossref queries "Gini coefficient stochastic vehicle routing" and "chance-constrained vehicle routing equitable" (no hit).
  arXiv title/abstract keyword queries (fair/equitable + routing + stochastic/uncertainty) returned unrelated items.
- Caveat: the search tools are not exhaustive (US-only web search; no Scopus/WoS access), and
  conference papers or very recent preprints may be missed. Absence of evidence, not proof.

### G2. Route consistency / TSP-optimality under stochastic travel times

Finding: I found no paper that defines TSP-optimality (or workload consistency in the Ljubić-Puerto-Torrejón
or van Rossum-Chen-Lodi sense) when travel times are random. The consistent-VRP literature uses "consistency" in a different sense
(same driver, similar arrival times: Groër et al. 2009, Kovacs et al. 2015, Alvarez et al. 2024 with stochastic
customers and demands, Wu et al. 2025 with demand uncertainty). Under random times the notion is not unique
(each driver's realised-optimal tour, the a-priori tour optimal in expectation, or the optimal-in-expectation tour with recourse),
and I found nothing addressing this; Ljubić et al. only remark that the bilevel structure lets "the routing part change
without touching the fairness part".

Evidence: search queries "TSP-optimality stochastic travel times", "route inconsistency fair routing uncertainty",
"consistent routing equity stochastic travel times" returned only the deterministic fair-VRP papers and the consistent-VRP papers above.
Related but different: the a-priori TSP/probabilistic TSP line (random customer subset; not retrieved into the bib)
and the mean-risk shortest-path literature (Nikolova; Sen et al.), which give candidate definitions of a "reliable" tour.

### G3. Other gaps noted

- Correlated durations (Rostami et al. 2021; Bakach et al. 2021; Bomboi et al. 2022) are modelled for feasibility or makespan,
  never for fairness; correlation changes the order-statistic structure (the Bapat-Beg permanent representation assumes independence).
- Order statistics of independent non-identical variables are well developed (Cluster 5), but I found no OR paper using the Poisson-binomial
  representation for expected ordered fairness measures in routing.
- No paper found on price of fairness (Bertsimas et al.) for stochastic routing.

---

## UNVERIFIED (not in references.bib)

- **Yitzhaki (1982), "Stochastic Dominance, Mean Variance, and Gini's Mean Difference", American Economic Review 72(1?):178-185.**
  Seen only as a reference-list citation in other papers and a Hebrew University working-paper record; no DOI found, no
  publisher page retrieved, issue number unconfirmed.
- **Yager and Kacprzyk (eds.), The Ordered Weighted Averaging Operators (Springer, 1997), DOI 10.1007/978-1-4615-6123-1.**
  DOI exists (Crossref); not used because no verified role.
- **Chen and Liu (1997), "Statistical applications of the Poisson-binomial and conditional Bernoulli distributions", Statistica Sinica 7:875-892.**
  Crossref query found nothing; not confirmed.
- **Hande (1994), simple proof of the Bapat-Beg theorem.** Only seen as a mention in a search summary; venue not confirmed.
- **Rhea and Ohlmann, "Mobile Facility Routing for Equitable Service of (Spatially) Covered Stochastic Demand".** Only SSRN working-paper
  records (DOI 10.2139/ssrn.4961208 and 10.2139/ssrn.5576249, 2024-2025) found; no journal version located. Potentially relevant (equity, stochastic demand, routing).
- **Ljubić, Puerto, Torrejón update date.** One search summary said the preprint was updated 12 October 2026; the Optimization Online
  page I fetched confirms submission on 5 March 2026 only.
- **Blanco et al. (2024)**: the preprint's reference list shows the 2024 Blanco item is "Intra-facility equity" (Blanco, Marín, Puerto), verified. The "Blanco et al. (2026)" is the ordered-measures paper, verified.
- **Lesmana, Zhang, Bei (2019), NeurIPS, "Balancing efficiency and fairness in on-demand ridesourcing."** Not found by Crossref; unverified.
- **Sources mentioned only through search snippets (content claims unverified, bibliographic data verified):** Gutjahr 2021, Mostajabdaveh 2019, Alem 2022,
  Wu 2025, Yu 2024, Hosseini and Schierreich 2025, Basik 2021, Bruinink 2025, Sobhanan 2026.
- **Eisner and Severance framing** (see Cluster 6): bibliographic data verified, "parametric combinatorial optimisation" framing not.
- **Ehrgott 2005** author attribution rests on the known ISBN/title; the DOI metadata has no author field.
