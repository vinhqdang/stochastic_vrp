# Verified references for the RARECERT demand, travel-time and rare-event models

Scope: candidate citations for the gamma-marginal / one-factor Gaussian-copula demand model, lognormal route-day travel times, rare overflow certification, and the verification of IS-based certificates.

Verification rule applied: an entry is VERIFIED only if its title, authors, venue, year, volume/pages and DOI were seen in the Crossref record (api.crossref.org/works/<doi> or a bibliographic query), or, for preprints, on the arXiv abstract page (export.arxiv.org/abs/<id>). Entries are split into (A) in-window works (2019 onward), (B) pre-window precedents kept only because the method descends from them, and (C) items seen but not verified. A USE line marked "snippet" means I read the search-result summary, not the full text; check the wording against the paper before citing a specific claim.

Counts: 24 verified in-window entries (section A), 3 verified pre-window precedents (section B), 0 entries invented.

---

## A. Verified, 2019 onward

### Topic 1: correlated and heavy-tailed demand in routing

```bibtex
@article{Florio2023Recent,
  author  = {Florio, Alexandre M. and Gendreau, Michel and Hartl, Richard F. and Minner, Stefan and Vidal, Thibaut},
  title   = {Recent advances in vehicle routing with stochastic demands: {B}ayesian learning for correlated demands and elementary branch-price-and-cut},
  journal = {European Journal of Operational Research},
  year    = {2023},
  volume  = {306},
  number  = {3},
  pages   = {1081--1093},
  doi     = {10.1016/j.ejor.2022.10.045}
}
```
USE: Related work (correlated demand in VRPSD). Supports the statement that demand correlation is modelled in routing mainly through a single latent factor, which is the structural analogue of our one-factor copula; supports the claim that accounting for correlation changes restocking decisions (the abstract reports savings above 10%). Checked from the abstract and bibliographic record only.

```bibtex
@article{Ota2025Generic,
  author  = {Ota, Matheus J. and Fukasawa, Ricardo},
  title   = {On vehicle routing problems with stochastic demands -- Generic disaggregated integer {L}-shaped formulations},
  year    = {2025},
  month   = oct,
  note    = {arXiv preprint arXiv:2510.04043},
  url     = {https://arxiv.org/abs/2510.04043}
}
```
USE: Introduction, one sentence. Cite for the statement that the independence assumption in VRPSD is convenient but often unrealistic, and that correlated formulations remain few. Snippet only; verify the wording. Preprint, no DOI in the arXiv record.

### Topic 2: travel-time uncertainty (lognormal, congestion, correlated, time-dependent)

```bibtex
@article{Heakl2025SVRPBench,
  author  = {Heakl, Ahmed and Shaaban, Yahia Salaheldin and Takac, Martin and Lahlou, Salem and Iklassov, Zangir},
  title   = {{SVRPBench}: A Realistic Benchmark for Stochastic Vehicle Routing Problem},
  year    = {2025},
  month   = may,
  note    = {arXiv preprint arXiv:2505.21887},
  url     = {https://arxiv.org/abs/2505.21887}
}
```
USE: Related work (benchmarks with stochastic travel times) and Experiments (instance design). Its abstract states that it simulates time-dependent congestion, log-normal delays and probabilistic accidents at urban scale, which is the closest published precedent for lognormal route-day travel times in a benchmark. Preprint; no DOI in the arXiv record.

```bibtex
@article{Guo2019Vehicle,
  author  = {Guo, Zhaoxia and Wallace, Stein W. and Kaut, Michal},
  title   = {Vehicle Routing with Space- and Time-Correlated Stochastic Travel Times: Evaluating the Objective Function},
  journal = {INFORMS Journal on Computing},
  year    = {2019},
  volume  = {31},
  pages   = {654--670},
  doi     = {10.1287/ijoc.2019.0906}
}
```
USE: Related work (correlated travel times) and Problem section (why correlated travel times make the route-day objective hard to evaluate). Supports a claim that spatial and temporal correlation of link times is handled in the objective evaluation, not only in heuristics.

```bibtex
@article{Bomboi2021Stochastic,
  author  = {Bomboi, Federica and Buchheim, Christoph and Pruente, Jonas},
  title   = {On the stochastic vehicle routing problem with time windows, correlated travel times, and time dependency},
  journal = {4OR},
  year    = {2021},
  volume  = {20},
  pages   = {217--239},
  doi     = {10.1007/s10288-021-00476-z}
}
```
USE: Related work (chance-constrained routing with correlated, time-dependent travel times). The model assumes jointly normal travel times, so cite it as the normal-case contrast to our lognormal choice.

```bibtex
@article{Rostami2021Branch,
  author  = {Rostami, Borzou and Desaulniers, Guy and Errico, Fabio and Lodi, Andrea},
  title   = {Branch-Price-and-Cut Algorithms for the Vehicle Routing Problem with Stochastic and Correlated Travel Times},
  journal = {Operations Research},
  year    = {2021},
  volume  = {69},
  pages   = {436--455},
  doi     = {10.1287/opre.2020.2037}
}
```
USE: Related work (exact methods under stochastic and correlated travel times). Supports the statement that exact methods exist for correlated travel-time VRPs, so any comparison with a plug-in or sampled exact baseline should be framed against them.

```bibtex
@article{Blauth2024Vehicle,
  author  = {Blauth, Jannis and Held, Stephan and M{\"u}ller, Dirk and Schlomberg, Niklas and Traub, Vera and Tr{\"o}bst, Thorben},
  title   = {Vehicle routing with time-dependent travel times: Theory, practice, and benchmarks},
  journal = {Discrete Optimization},
  year    = {2024},
  volume  = {53},
  pages   = {100848},
  doi     = {10.1016/j.disopt.2024.100848}
}
```
USE: Experiments (instance and travel-time model design, and benchmark comparison). Its search summary says the benchmarks use OpenStreetMap maps and Uber speed data; I have not read the full text, so check that claim before citing it for data sourcing.

```bibtex
@article{Liu2019Stochastic,
  author  = {Liu, Yuqing and Blandin, Sebastien and Samaranayake, Samitha},
  title   = {Stochastic on-time arrival problem in transit networks},
  journal = {Transportation Research Part B: Methodological},
  year    = {2019},
  volume  = {119},
  pages   = {122--138},
  doi     = {10.1016/j.trb.2018.11.013}
}
```
USE: Related work (on-time arrival and probabilistic service in networks with random link travel times). Transit rather than road networks, so use it for the reliability framing only.

### Topic 3: probabilistic guarantees and risk measures for routing plans

```bibtex
@article{Ghosal2020Distributionally,
  author  = {Ghosal, Soumyadeep and Wiesemann, Wolfram},
  title   = {The Distributionally Robust Chance-Constrained Vehicle Routing Problem},
  journal = {Operations Research},
  year    = {2020},
  volume  = {68},
  pages   = {716--732},
  doi     = {10.1287/opre.2019.1924}
}
```
USE: Related work and Planning-stage discussion (distributionally robust chance constraints in VRP). Supports the claim that moment-based ambiguity sets can be handled by branch-and-cut under a subadditivity condition, which is the setting of the Wasserstein and moment gates we compare against.

```bibtex
@article{Ghosal2024Unifying,
  author  = {Ghosal, Soumyadeep and Ho, Chin Pang and Wiesemann, Wolfram},
  title   = {A Unifying Framework for the Capacitated Vehicle Routing Problem Under Risk and Ambiguity},
  journal = {Operations Research},
  year    = {2024},
  volume  = {72},
  number  = {2},
  pages   = {425--443},
  doi     = {10.1287/opre.2021.0669}
}
```
USE: Related work and Estimators section (risk measures and disutilities combined with full or partial demand distributions, including Wasserstein ambiguity). Supports placing CVaR and Wasserstein gates in one framework. Check the paper's treatment of CVaR before citing it for that specific point; I read the abstract summary only.

```bibtex
@article{Zhang2021Vehicle,
  author  = {Zhang, Dezhi and Li, Dan and Sun, Huijun and Hou, Lu},
  title   = {A vehicle routing problem with distribution uncertainty in deadlines},
  journal = {European Journal of Operational Research},
  year    = {2021},
  volume  = {292},
  pages   = {311--326},
  doi     = {10.1016/j.ejor.2020.10.026}
}
```
USE: Related work (CVaR approximations of probabilistic routing constraints). Supports the statement that the standard CVaR approximation of a chance constraint is conservative and motivates a tighter gate; relevant to the claim that the SAA-CVaR gate is an approximation to the chance constraint.

```bibtex
@misc{Deo2021Blackbox,
  author        = {Deo, Anand and Murthy, Karthyek},
  title         = {Efficient Black-Box Importance Sampling for {VaR} and {CVaR} Estimation},
  year          = {2021},
  month         = jun,
  eprint        = {2106.10236},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2106.10236}
}
```
USE: Estimators section (importance sampling for VaR and CVaR estimation, with asymptotically optimal variance reduction on the log scale). Supports the estimator choice for tail risk measures. arXiv preprint; check whether a journal version exists before submission.

### Topic 4: copula and factor-copula rare-event estimation (in-window)

```bibtex
@misc{Cui2024Portfolio,
  author        = {Cui, Hengxin and Tan, Ken Seng and Yang, Fan},
  title         = {Portfolio credit risk with Archimedean copulas: asymptotic analysis and efficient simulation},
  year          = {2024},
  month         = nov,
  eprint        = {2411.06640},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2411.06640}
}
```
USE: Related work (copula-dependent rare-event estimation beyond the Gaussian case, extremal dependence). Supports the statement that efficient simulation for copula-dependent losses is an active area. Archimedean rather than factor copula; cite as a neighbour, not a direct precedent.

```bibtex
@misc{Fang2026ISCOS,
  author        = {Fang, Fang and Shen, Xiaoyu and Wang, Qinling},
  title         = {Importance Sampling Enhanced with the {COS} Method for the Portfolio Risk Allocation},
  year          = {2026},
  month         = aug,
  eprint        = {2608.30749},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.30749}
}
```
USE: Related work and Estimators section (importance-sampling proposals for Gaussian and Student-t copula portfolios, calibrated through a COS-based approximation). Supports the claim that factor-copula IS with a shifted proposal is still the working method. Preprint dated 2026; flag it as such in the paper and check the full text.

### Topic 5: validation of simulation-based certificates and coverage

```bibtex
@article{Owen2019Importance,
  author  = {Owen, Art B. and Maximov, Yury and Chertkov, Michael},
  title   = {Importance sampling the union of rare events with an application to power systems analysis},
  journal = {Electronic Journal of Statistics},
  year    = {2019},
  volume  = {13},
  pages   = {231--254},
  doi     = {10.1214/18-ejs1527}
}
```
USE: Estimators section and Sensitivity and coverage. The sampler picks one rare event at random and samples conditionally on it, with a variance bound in terms of the union bound; the paper reports coefficients of variation at probabilities below 10^-22. Supports the claim that a bounded relative error is a checkable property of an IS certificate, and that union-type route overflows can be estimated by conditional sampling.

```bibtex
@article{Bai2024Uncertainty,
  author  = {Bai, Yuanlu and Lam, Henry},
  title   = {Uncertainty quantification and confidence intervals for naive rare-event estimators},
  journal = {Journal of Applied Probability},
  year    = {2024},
  volume  = {62},
  pages   = {84--110},
  doi     = {10.1017/jpr.2024.43}
}
```
USE: Sensitivity and coverage (naive Monte Carlo confidence intervals for rare probabilities at 10^-3 to 10^-6). The paper finds the normal and Wilson intervals are not always valid, while an exact interval is conservative. Supports the coverage check we report for the naive baseline. Crossref gives 2024 as the year of record; the arXiv v1 is 2023.

```bibtex
@incollection{Budde2021Replicating,
  author    = {Budde, Carina and Hartmanns, Arnd},
  title     = {Replicating {RESTART} with Prolonged Retrials: An Experimental Report},
  booktitle = {Lecture Notes in Computer Science},
  year      = {2021},
  pages     = {373--380},
  doi       = {10.1007/978-3-030-72013-1_21},
  % NOTE: the Crossref record gives the chapter title and pages; the proceedings title is not in the record. Verify before citing as a proceedings paper.
}
```
USE: Experiments (reproducibility of rare-event estimators). The authors report that replicating a rare-event method required resolving ambiguities in the original setup, supporting our claim that the plan cache and the simulator settings are documented in full.

```bibtex
@article{Jiang2020Accurate,
  author  = {Jiang, Tony and Cao, Bo and Shan, Guogen},
  title   = {Accurate confidence intervals for risk difference in meta-analysis with rare events},
  journal = {BMC Medical Research Methodology},
  year    = {2020},
  volume  = {20},
  doi     = {10.1186/s12874-020-00954-8}
}
```
USE: Sensitivity and coverage (coverage studies for rare-event confidence intervals built from importance sampling). It is a medical meta-analysis, not routing; cite only as a methodological precedent for reporting coverage of rare-event intervals. The search summary says its intervals are close to nominal coverage, which I did not check in the full text.

### Topic 6: open geospatial data and synthetic instances

```bibtex
@article{Arnold2019Efficiently,
  author  = {Arnold, Florian and Gendreau, Michel and S{\"o}rensen, Kenneth},
  title   = {Efficiently solving very large-scale routing problems},
  journal = {Computers \& Operations Research},
  year    = {2019},
  volume  = {107},
  pages   = {32--42},
  doi     = {10.1016/j.cor.2019.03.006}
}
```
USE: Experiments (large regional instance sizes, benchmark construction from a real-world problem). Cite when justifying scale and the choice of benchmark sets.

```bibtex
@article{Sartori2020Study,
  author  = {Sartori, Claudio and Buriol, Luciana},
  title   = {A study on the pickup and delivery problem with time windows: Matheuristics and new instances},
  journal = {Computers \& Operations Research},
  year    = {2020},
  volume  = {124},
  pages   = {105065},
  doi     = {10.1016/j.cor.2020.105065}
}
```
USE: Experiments (instance generation from OpenStreetMap). The OVIG generator that goes with this paper is described in its public repository as using OSM maps and OSRM travel times; I did not verify that description in the paper's text, so check it before citing for the generation method.

```bibtex
@article{Giraud2022osrm,
  author  = {Giraud, Timothée},
  title   = {{osrm}: Interface Between {R} and the {OpenStreetMap}-Based Routing Service {OSRM}},
  journal = {Journal of Open Source Software},
  year    = {2022},
  volume  = {7},
  pages   = {4574},
  doi     = {10.21105/joss.04574}
}
```
USE: Experiments (OSRM distance and duration matrices). Software citation for how the road-network matrices are built.

```bibtex
@article{Calka2020GHS,
  author  = {Calka, Beata and Bielecka, Elzbieta},
  title   = {{GHS-POP} Accuracy Assessment: Poland and Portugal Case Study},
  journal = {Remote Sensing},
  year    = {2020},
  volume  = {12},
  number  = {7},
  pages   = {1105},
  doi     = {10.3390/rs12071105}
}
```
USE: Experiments (population grid as a demand proxy; known bias). The assessment reports that dense regions are underestimated and thin regions overestimated, which supports a limitation statement about grid-based demand weights.

```bibtex
@article{Pesaresi2024Advances,
  author  = {Pesaresi, Martino and Schiavina, Marcello and Politis, Panagiotis and Freire, Sergio and Krasnod{\k{e}}bska, Krystyna and Uhl, Johannes},
  title   = {Advances on the Global Human Settlement Layer by joint assessment of Earth Observation and population survey data},
  journal = {International Journal of Digital Earth},
  year    = {2024},
  volume  = {17},
  doi     = {10.1080/17538947.2024.2390454}
}
```
USE: Data section (methodological reference for GHS-POP as cited by the JRC product documentation; I did not read the full text). The record gives volume 17 but no page range.

```bibtex
@misc{Son2025RRNCO,
  author        = {Son, Jiwoo and Zhao, Zhikai and Berto, Federico and Hua, Chuanbo and Cao, Zhiguang and Kwon, Changhyun and Park, Jinkyoo},
  title         = {{RRNCO}: Towards Real-World Routing with Neural Combinatorial Optimization},
  year          = {2025},
  month         = mar,
  eprint        = {2503.16159},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2503.16159}
}
```
USE: Experiments (city selection and OSRM-based realistic routing datasets). Snippet only: the search summary says city maps are built with OSRM for realistic VRP data. Preprint; cite as a neighbour for regional instance construction.

---

## B. Verified pre-2019 precedents (outside the window; cite only for the method lineage)

```bibtex
@article{Glasserman2005Importance,
  author  = {Glasserman, Paul and Li, Jingyi},
  title   = {Importance Sampling for Portfolio Credit Risk},
  journal = {Management Science},
  year    = {2005},
  volume  = {51},
  pages   = {1643--1656},
  doi     = {10.1287/mnsc.1050.0415}
}
```
USE: Estimators section (origin of the factor-shift IS for the Gaussian-copula model, the lineage of our factor shift).

```bibtex
@article{Glasserman2008Fast,
  author  = {Glasserman, Paul and Kang, Wanmo and Shahabuddin, Perwez},
  title   = {Fast Simulation of Multifactor Portfolio Credit Risk},
  journal = {Operations Research},
  year    = {2008},
  volume  = {56},
  number  = {5},
  pages   = {1200--1217},
  doi     = {10.1287/opre.1080.0558}
}
```
USE: Estimators section (multifactor Gaussian-copula IS with a mixture of factor-shifted proposals, with asymptotic optimality). Closest method precedent for the one-factor case with a stage mixture.

```bibtex
@article{Noorizadegan2018Vehicle,
  author  = {Noorizadegan, Mahdi and Chen, Bintong},
  title   = {Vehicle routing with probabilistic capacity constraints},
  journal = {European Journal of Operational Research},
  year    = {2018},
  volume  = {270},
  pages   = {544--555},
  doi     = {10.1016/j.ejor.2018.04.010}
}
```
USE: Related work (probabilistic capacity constraints and route failure in routing). Pre-window, so cite as background for the route-failure framing only.

---

## C. Seen but not verified (do not cite until checked)

- Kontur Population dataset (Kontur, 400 m H3 hexagons, fusing GHSL, HRSL and Microsoft Building Footprints; Humanitarian Data Exchange lists a 2020-03-11 issue date, CC BY). No DOI, no peer-reviewed description found. UNVERIFIED as a citable source; cite the dataset page with its access date if used.
- WorldPop methodology paper (Tatem 2017, Scientific Data, and Lloyd, Sorichetta and Tatem 2017). Seen in search results only, not checked in Crossref; both are pre-window in any case. No 2019 or 2021 WorldPop methodology paper was found.
- GHS-POP R2019 methodology (Schiavina et al.). Not located as a verified record.
- Nagaraj, Xu, Pasupathy and Ghosh, "Efficient Estimation in the Tails of Gaussian Copulas" (arXiv:1607.01375, revised 2022). arXiv record confirmed the title and 2016 date. Journal version not verified; pre-window.
- Dinh, Fukasawa and Luedtke (correlated demand, chance-constrained VRP, Mathematical Programming). Described in search summaries as 2018; not checked in Crossref. Likely pre-window.
- Hoogendoorn (2024) thesis on correlated multivariate-normal demand. Named in a 2025 preprint summary; not checked.
- Gap in topic 1 for 2019 onward: only Florio et al. (2023) and the Ota and Fukasawa (2025) preprint were verified for correlated demand in routing. Fewer than the three requested.
- Gap in topic 4 for 2018 onward: copula-specific verified work is limited to Cui et al. (2024, Archimedean) and Fang et al. (2026, preprint). Owen et al. (2019) and Deo and Murthy (2021) are rare-event IS but not copula models.
