# Verified references for RARECERT (new VRP literature, 2019-2026)

Verification rule applied: every entry below was checked against its Crossref
record (`https://api.crossref.org/works/<doi>`, fetched 2026-10-16). Title,
authors, journal, year, volume, issue and pages are taken from that record.
Entries not confirmed in Crossref are listed at the end as UNVERIFIED and are
not in the BibTeX.

Note on years: where Crossref gives an online year and a print year that
differ, the BibTeX uses the year Crossref reports as `issued`.

One BibTeX entry is pre-2019 (Bertsimas et al. 1995, the a-priori versus
a-posteriori reference); it is marked PRE-2019. Other pre-2019 verified works
are listed at the end without BibTeX.

---

## Topic 1: Surveys and reviews

```bibtex
@article{akhmetbek2024stochastic,
  author  = {Akhmetbek, Yernar},
  title   = {Stochastic dynamic vehicle routing problem survey},
  journal = {Suleyman Demirel University Bulletin: Natural and Technical Sciences},
  year    = {2024},
  volume  = {62},
  number  = {1},
  pages   = {75--87},
  doi     = {10.47344/sdubnts.v62i1.959}
}
```
USE: Recent survey of stochastic and dynamic VRP solution approaches; cite in the related-work paragraph that places certification of a fixed plan alongside planning-side stochastic VRP methods.

```bibtex
@article{karimi2026stochastic,
  author  = {Karimi, Amin and Zhang, Lele and Fackrell, Mark},
  title   = {Stochastic Vehicle Routing Problem: classification and review},
  journal = {Transportation Letters},
  year    = {2026},
  pages   = {1--21},
  doi     = {10.1080/19427867.2026.2726413}
}
```
USE: Most recent classification of SVRP variants (stochastic demands, travel times, customers); cite to position the demand-uncertainty setting that RARECERT evaluates, and to state that route-level overflow certification is not covered by planning surveys.

```bibtex
@incollection{ma2021stochastic,
  author    = {Ma, Jun and Zhang, Jihui and Guo, Yiyun},
  title     = {The Stochastic and Dynamic Vehicle Routing Problem: A Literature Review},
  booktitle = {Lecture Notes in Electrical Engineering},
  year      = {2021},
  pages     = {344--351},
  doi       = {10.1007/978-981-16-6372-7_39}
}
```
USE: Secondary survey of stochastic and dynamic VRP; cite only alongside the primary surveys above for the breadth of the literature on demand uncertainty.

---

## Topic 2: Distributionally robust and chance-constrained VRP

```bibtex
@article{ghosal2020distributionally,
  author  = {Ghosal, Shubhechyya and Wiesemann, Wolfram},
  title   = {The Distributionally Robust Chance-Constrained Vehicle Routing Problem},
  journal = {Operations Research},
  year    = {2020},
  volume  = {68},
  number  = {3},
  pages   = {716--732},
  doi     = {10.1287/opre.2019.1924}
}
```
USE: Canonical chance-constrained CVRP with stochastic demand under a moment-based ambiguity set; cite as the closest DRO-CVRP model and as the reference for "capacity chance constraint on the route" that RARECERT certifies at a per-route level.

```bibtex
@article{ghosal2024unifying,
  author  = {Ghosal, Shubhechyya and Ho, Chin Pang and Wiesemann, Wolfram},
  title   = {A Unifying Framework for the Capacitated Vehicle Routing Problem Under Risk and Ambiguity},
  journal = {Operations Research},
  year    = {2024},
  volume  = {72},
  number  = {2},
  pages   = {425--443},
  doi     = {10.1287/opre.2021.0669}
}
```
USE: Unifies risk measures and ambiguity sets for CVRP with stochastic demand; cite when stating that CVaR and chance-constraint formulations of capacity risk are already established, so the contribution is in the certification of rare overflow, not in the planning model.

```bibtex
@article{thiebaut2022approximating,
  author  = {Thiebaut, Karina and Pessoa, Artur},
  title   = {Approximating the chance-constrained capacitated vehicle routing problem with robust optimization},
  journal = {4OR},
  year    = {2022},
  volume  = {21},
  number  = {3},
  pages   = {513--531},
  doi     = {10.1007/s10288-022-00523-3}
}
```
USE: Robust-optimization approximation of the chance-constrained CVRP; cite as an alternative to exact chance constraints for capacity, and to support the claim that the chance-constrained capacity model is an active research line.

```bibtex
@article{wang2023distributionally,
  author  = {Wang, Weiqiao and Yang, Kai and Yang, Lixing and Gao, Ziyou},
  title   = {Distributionally robust chance-constrained programming for multi-period emergency resource allocation and vehicle routing in disaster response operations},
  journal = {Omega},
  year    = {2023},
  volume  = {120},
  pages   = {102915},
  doi     = {10.1016/j.omega.2023.102915}
}
```
USE: Distributionally robust chance constraints combined with vehicle routing for disaster response; cite as a 2023 application of DR chance constraints on routing capacity.

```bibtex
@article{yin2022distributionally,
  author  = {Yin, Fanghao and Zhao, Yi},
  title   = {Distributionally robust equilibrious hybrid vehicle routing problem under twofold uncertainty},
  journal = {Information Sciences},
  year    = {2022},
  volume  = {609},
  pages   = {1239--1255},
  doi     = {10.1016/j.ins.2022.07.140}
}
```
USE: Distributionally robust VRP under two sources of uncertainty; cite for the claim that ambiguity-set models with multiple uncertain factors are now standard in routing.

```bibtex
@article{yin2026goal,
  author  = {Yin, Fanghao and Yang, Meng and Wang, Xiaobin},
  title   = {Goal programming-based Wasserstein distributionally robust equilibrium vehicle routing design with carbon cost intensity},
  journal = {Information Sciences},
  year    = {2026},
  volume  = {741},
  pages   = {123316},
  doi     = {10.1016/j.ins.2026.123316}
}
```
USE: Wasserstein ambiguity applied to vehicle routing design (2026); cite as the most recent evidence that Wasserstein balls are being used in routing, which is the ambiguity class RARECERT's certification must be robust to.

```bibtex
@article{wang2025distributionally,
  author  = {Wang, Xiaopeng and Zhao, Jinling},
  title   = {Distributionally robust optimization of the vehicle routing problem with uncertain customers},
  journal = {Journal of Industrial and Management Optimization},
  year    = {2025},
  volume  = {21},
  number  = {3},
  pages   = {1983--2006},
  doi     = {10.3934/jimo.2024159}
}
```
USE: Distributionally robust VRP with uncertain customer sets (2025); cite with the other DRO-VRP entries to show the recent growth of DRO for routing.

```bibtex
@article{keyantuo2023distributionally,
  author  = {Keyantuo, Patrick and Wang, Ruiting and Zeng, Teng and Vishwanath, Aashrith and Borhan, Hoseinali and Moura, Scott J.},
  title   = {Distributionally Robust and Data-Driven Solutions to Commercial Vehicle Routing Problems},
  journal = {IFAC-PapersOnLine},
  year    = {2023},
  volume  = {56},
  number  = {2},
  pages   = {10497--10502},
  doi     = {10.1016/j.ifacol.2023.10.1069}
}
```
USE: Data-driven distributionally robust commercial VRP; cite for the data-driven ambiguity-set construction that RARECERT's distributional assumptions must be tested against.

```bibtex
@article{ota2025hardness,
  author  = {Ota, Matheus J. and Fukasawa, Ricardo},
  title   = {Hardness of Pricing Routes for Two-Stage Stochastic Vehicle Routing Problems with Scenarios},
  journal = {Operations Research},
  year    = {2025},
  volume  = {73},
  number  = {4},
  pages   = {2177--2187},
  doi     = {10.1287/opre.2023.0569}
}
```
USE: Complexity of pricing routes in two-stage scenario-based SVRP (2025); cite to explain why exact per-route scenario evaluation is hard in general and why a certification method must work on fixed plans.

---

## Topic 3: VRPSPD with stochastic demand, recourse, and metaheuristics

```bibtex
@article{salavati2019rulebased,
  author  = {Salavati-Khoshghalb, Majid and Gendreau, Michel and Jabali, Ola and Rei, Walter},
  title   = {A Rule-Based Recourse for the Vehicle Routing Problem with Stochastic Demands},
  journal = {Transportation Science},
  year    = {2019},
  volume  = {53},
  number  = {5},
  pages   = {1334--1353},
  doi     = {10.1287/trsc.2018.0876}
}
```
USE: Rule-based recourse for route failure under stochastic demand (2019); cite as the recourse policy whose load-dependent failure events RARECERT would certify, and as the standard against which overflow is defined.

```bibtex
@article{salavati2019hybrid,
  author  = {Salavati-Khoshghalb, Majid and Gendreau, Michel and Jabali, Ola and Rei, Walter},
  title   = {A hybrid recourse policy for the vehicle routing problem with stochastic demands},
  journal = {EURO Journal on Transportation and Logistics},
  year    = {2019},
  volume  = {8},
  number  = {3},
  pages   = {269--298},
  doi     = {10.1007/s13676-018-0126-y}
}
```
USE: Hybrid recourse for VRP with stochastic demands (2019); cite with the rule-based recourse paper to show that recourse policies differ and that the failure probability depends on the policy.

```bibtex
@article{hu2025vehicle,
  author  = {Hu, Hanru and Tao, Fengming},
  title   = {Vehicle routing problem for simultaneous pickup and delivery in rural areas: considering uncertain demand and government subsidy},
  journal = {Annals of Operations Research},
  year    = {2025},
  volume  = {358},
  number  = {3},
  pages   = {1285--1327},
  doi     = {10.1007/s10479-025-06877-1}
}
```
USE: VRPSPD with uncertain demand (2025); the closest recent paper to the RARECERT setting of simultaneous pickup and delivery under demand uncertainty; cite to show that uncertain-demand VRPSPD is still an open planning topic.

```bibtex
@article{wang2023hyper,
  author  = {Wang, Zheng and Liu, Jinlong and Zhang, Jingling},
  title   = {Hyper-heuristic algorithm for traffic flow-based vehicle routing problem with simultaneous delivery and pickup},
  journal = {Journal of Computational Design and Engineering},
  year    = {2023},
  volume  = {10},
  number  = {6},
  pages   = {2271--2287},
  doi     = {10.1093/jcde/qwad097}
}
```
USE: Recent metaheuristic (hyper-heuristic) for VRPSPD with time-varying traffic flow; cite as recent VRPSPD metaheuristic work. Note: the model is deterministic in demand; cite only for the VRPSPD heuristic literature, not for demand uncertainty.

```bibtex
@article{liu2022variable,
  author  = {Liu, Ran and Jiang, Shan},
  title   = {A variable neighborhood search algorithm with constraint relaxation for the two-echelon vehicle routing problem with simultaneous delivery and pickup demands},
  journal = {Soft Computing},
  year    = {2022},
  volume  = {26},
  number  = {17},
  pages   = {8879--8896},
  doi     = {10.1007/s00500-021-06692-3}
}
```
USE: Recent VNS for two-echelon VRPSPD; cite for the VRPSPD metaheuristic literature. Note: deterministic demand; cite only for the planning side.

```bibtex
@inproceedings{gu2022express,
  author    = {Gu, Zhenyu and Chen, Lin and Zheng, Jiajia},
  title     = {Express Delivery Vehicle Routing Scheduling with Simultaneous Pickup and Delivery Oriented to Dynamic Customer Demand},
  booktitle = {CICTP 2022},
  year      = {2022},
  pages     = {1240--1252},
  doi       = {10.1061/9780784484265.117}
}
```
USE: Express-delivery VRPSPD under dynamic customer demand (2022); cite as an applied link between simultaneous pickup and delivery and parcel operations, which RARECERT's regional parcel setting relates to.

---

## Topic 4: Evaluation and validation of routing solutions under uncertainty

```bibtex
@article{wang2025simulation,
  author  = {Wang, N. N. and Jian, M. and Fang, X.},
  title   = {Simulation-Based Robustness Evaluation of Coordination under Stochastic Demand},
  journal = {International Journal of Simulation Modelling},
  year    = {2025},
  volume  = {24},
  number  = {4},
  pages   = {730--741},
  doi     = {10.2507/ijsimm24-4-co20}
}
```
USE: Simulation-based robustness evaluation of a plan under stochastic demand (2025); cite as recent a-posteriori simulation evaluation of a fixed plan, the same validation step RARECERT formalises with importance sampling for rare tails.

```bibtex
@article{bertsimas1995computational,
  author  = {Bertsimas, Dimitris and Chervi, Philippe and Peterson, Michael},
  title   = {Computational Approaches to Stochastic Vehicle Routing Problems},
  journal = {Transportation Science},
  year    = {1995},
  volume  = {29},
  number  = {4},
  pages   = {342--352},
  doi     = {10.1287/trsc.29.4.342}
}
```
PRE-2019 (background only). USE: Classic a-priori versus a-posteriori comparison for SVRP; cite as the origin of evaluating a fixed plan on sampled demand, to position the certification question against the long-standing evaluation practice.

---

## Topic 5: Service-level and risk requirements in specialised distribution

```bibtex
@article{campelo2019consistent,
  author  = {Campelo, Pedro and Neves-Moreira, F{\'a}bio and Amorim, Pedro and Almada-Lobo, Bernardo},
  title   = {Consistent vehicle routing problem with service level agreements: A case study in the pharmaceutical distribution sector},
  journal = {European Journal of Operational Research},
  year    = {2019},
  volume  = {273},
  number  = {1},
  pages   = {131--145},
  doi     = {10.1016/j.ejor.2018.07.030}
}
```
USE: Service-level agreements in pharmaceutical distribution (2019); cite to motivate that routing plans in regulated distribution are judged against explicit service-level targets, not expected cost alone. It does not model stochastic demand overflow.

```bibtex
@article{holeczek2021analysis,
  author  = {Holeczek, Nikolai},
  title   = {Analysis of different risk models for the hazardous materials vehicle routing problem in urban areas},
  journal = {Cleaner Environmental Systems},
  year    = {2021},
  volume  = {2},
  pages   = {100022},
  doi     = {10.1016/j.cesys.2021.100022}
}
```
USE: Comparison of risk models for hazmat routing (2021); cite for the claim that route risk functions in hazardous-materials routing are defined by modelling choice, supporting the need to certify the chosen risk criterion rather than assume it.

```bibtex
@article{zhao2020fuzzy,
  author  = {Zhao, Liying and Cao, Ningbo},
  title   = {Fuzzy Random Chance-Constrained Programming Model for the Vehicle Routing Problem of Hazardous Materials Transportation},
  journal = {Symmetry},
  year    = {2020},
  volume  = {12},
  number  = {8},
  pages   = {1208},
  doi     = {10.3390/sym12081208}
}
```
USE: Chance-constrained hazmat routing with random elements (2020); cite as an example of chance constraints used as the risk requirement in hazardous-materials distribution.

```bibtex
@article{yang2024multistage,
  author  = {Yang, Yuze and Luo, Zunhao and Yang, Yongjian and Wang, Dujuan},
  title   = {Multistage adaptive distributionally robust optimization for the medical supplies distribution problem with uncertain demand in humanitarian aid},
  journal = {Computers \& Operations Research},
  year    = {2024},
  volume  = {167},
  pages   = {106631},
  doi     = {10.1016/j.cor.2024.106631}
}
```
USE: Distributionally robust medical-supply routing under uncertain demand (2024); cite as the medical-distribution case in which a demand-uncertainty risk requirement is set and certified.

---

## Topic 6: Regional parcel and last-mile planning

```bibtex
@article{thango2026performance,
  author  = {Thango, Bonginkosi A. and Omoruyi, Osayuwamen},
  title   = {Performance Comparison Among Classical Metaheuristic Algorithms for Stochastic Last-Mile Delivery Routing Problem},
  journal = {Algorithms},
  year    = {2026},
  volume  = {19},
  number  = {9},
  pages   = {785},
  doi     = {10.3390/a19090785}
}
```
USE: Comparison of metaheuristics for stochastic last-mile routing (2026); cite as the most recent stochastic last-mile routing benchmark, showing that planners compare plans on performance under uncertainty.

---

## UNVERIFIED leads (NOT in the BibTeX above; do not cite until checked)

- Carlsson, Behroozi & Mihic (2018), *Operations Research*, distributionally robust Euclidean TSP with Wasserstein ambiguity. Seen only in a search snippet. A Crossref query did not return a matching record, so no DOI is given.
- Angel et al. (2011), Monte Carlo reliability evaluation for VRPSD via safety stocks. Seen only in a search snippet. No record checked.
- arXiv 2310.17368, "Leveraging Contextual Information for Robustness in Vehicle Routing Problems". Seen only as a search result; arXiv page not fetched.
- Abdullahi et al. (2025), *Journal of Heuristics*, reliability-extended simheuristic for the sustainable VRP with stochastic travel times and demands. Seen only in a search snippet; the DOI (springer link s10732-025-09555-4 in a result URL) was not confirmed in Crossref.

## Verified, pre-2019 (not in the BibTeX; optional background)

- Oyola, Arntzen & Woodruff (2018), "The stochastic vehicle routing problem, a literature review, part I: models", *EURO Journal on Transportation and Logistics* 7(3):193-221, DOI 10.1007/s13676-016-0100-5. Pre-2019 survey.
- Gendreau, Laporte & Séguin (1996), "Stochastic vehicle routing", *European Journal of Operational Research* 88(1):3-12, DOI 10.1016/0377-2217(95)00050-x. Pre-2019 survey.
- Wang & Qiu (2011), "Vehicle Routing Problem with Stochastic Demands and Simultaneous Delivery and Pickup Based on the Cross-Entropy Method", *Lecture Notes in Electrical Engineering*, DOI 10.1007/978-3-642-25646-2_7. Pre-2019, closest VRPSPD + stochastic-demand precedent.
