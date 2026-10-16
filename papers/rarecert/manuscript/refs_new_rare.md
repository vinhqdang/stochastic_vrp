# New references: rare-event simulation (for RARECERT)

Scope: rare-event simulation, 2019-2026 preferred. Two pre-2019 anchors are included and flagged. Total: 25 verified entries.

**Verification.** Every entry was checked against its Crossref record (`api.crossref.org/works/<doi>`). The exception is Vehtari et al. (2024), which has no DOI in Crossref. For that entry the fields were checked on the arXiv abstract page (arXiv:1507.02646), which gives the JMLR reference. No field was taken from a search snippet alone. Nothing here is UNVERIFIED.

**Anchors outside the 2019-2026 window** (cited for lineage only): Glasserman & Li (2005); Glasserman, Kang & Shahabuddin (2008).

Entries are grouped by topic. Keys are natbib-friendly. Each `USE:` line says where the entry would be cited in a paper on certifying rare overflow of routing plans.

---

## 1. Importance sampling for unions, multiple failure modes, mixtures

```bibtex
@article{Owen2019Union,
  author  = {Owen, Art B. and Maximov, Yury and Chertkov, Michael},
  title   = {Importance sampling the union of rare events with an application to power systems analysis},
  journal = {Electronic Journal of Statistics},
  year    = {2019},
  volume  = {13},
  number  = {1},
  pages   = {231--254},
  doi     = {10.1214/18-EJS1527}
}
```
USE: Core citation for union-of-events IS (Karp-Luby style one-event-at-random with union-bound scaling). Supports the claim that a mixture proposal over per-route or per-stage overflow events gives a bounded coefficient of variation for the union. Use it to motivate the stage-mixture proposal in the RARECERT method and to contrast with the factor-shift (Glasserman-Li type) route.

```bibtex
@article{GlassermanLi2005,
  author  = {Glasserman, Paul and Li, Jingyi},
  title   = {Importance Sampling for Portfolio Credit Risk},
  journal = {Management Science},
  year    = {2005},
  volume  = {51},
  number  = {11},
  pages   = {1643--1656},
  doi     = {10.1287/mnsc.1050.0415}
}
```
USE: Pre-2019 anchor (outside the window). Cite for the factor-shift IS on a Gaussian-factor model, i.e. the single-shift baseline that the certification paper extends from a sum to a maximum over stages.

```bibtex
@article{GlassermanKangShahabuddin2008,
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
USE: Pre-2019 anchor (outside the window). Cite as the multifactor successor of Glasserman-Li, where the IS distribution is a combination of several factor-shift components. Supports the mixture-of-twists idea for the stage mixture.

```bibtex
@article{Papaioannou2019Mixture,
  author  = {Papaioannou, Iason and Geyer, Sebastian and Straub, Daniel},
  title   = {Improved cross entropy-based importance sampling with a flexible mixture model},
  journal = {Reliability Engineering \& System Safety},
  year    = {2019},
  volume  = {191},
  pages   = {106564},
  doi     = {10.1016/j.ress.2019.106564}
}
```
USE: Gaussian/flexible mixture proposals fitted by cross-entropy for reliability. Cite for the mixture-proposal construction and for the claim that single-component proposals fail on multimodal failure sets. Supports the stage-mixture design of RARECERT.

```bibtex
@article{Geyer2019Gaussian,
  author  = {Geyer, Sebastian and Papaioannou, Iason and Straub, Daniel},
  title   = {Cross entropy-based importance sampling using Gaussian densities revisited},
  journal = {Structural Safety},
  year    = {2019},
  volume  = {76},
  pages   = {15--27},
  doi     = {10.1016/j.strusafe.2018.07.001}
}
```
USE: Cross-entropy IS with Gaussian densities, revisited. Cite as the adaptive-CE baseline for the proposal we compare against, and for the known weakness that Gaussian CE misses far tails at 1e-6.

```bibtex
@article{Ling2021SVM,
  author  = {Ling, Chunyan and Lu, Zhenzhou},
  title   = {Support vector machine-based importance sampling for rare event estimation},
  journal = {Structural and Multidisciplinary Optimization},
  year    = {2021},
  volume  = {63},
  number  = {4},
  pages   = {1609--1631},
  doi     = {10.1007/s00158-020-02809-8}
}
```
USE: Surrogate-assisted IS for rare events. Cite as a surrogate-based alternative to exact IS, and to note that surrogate error enters the certificate and must be separated from IS sampling error.

```bibtex
@article{Ling2022Compound,
  author  = {Ling, Chunyan and Lu, Zhenzhou},
  title   = {Compound kriging-based importance sampling for reliability analysis of systems with multiple failure modes},
  journal = {Engineering Optimization},
  year    = {2022},
  volume  = {54},
  number  = {5},
  pages   = {805--829},
  doi     = {10.1080/0305215x.2021.1900837}
}
```
USE: IS for systems with multiple failure modes via compound kriging. Cite beside Owen et al. (2019) when stating that a route-level overflow event is a union over stages and that a union-aware proposal is needed.

```bibtex
@article{Helal2026Multimodal,
  author  = {Helal, Sara and Elvira, V{\'i}ctor},
  title   = {Efficient rare event estimation for multimodal and high-dimensional system reliability via subset adaptive importance sampling},
  journal = {Reliability Engineering \& System Safety},
  year    = {2026},
  volume  = {272},
  pages   = {112473},
  doi     = {10.1016/j.ress.2026.112473}
}
```
USE: Recent (2026) adaptive IS for multimodal failure sets, combining subset ideas with adaptive IS. Cite as the state of the art on multimodal rare-event IS, and to position the stage mixture against subset-adaptive hybrids.

```bibtex
@article{Zhang2026GMM,
  author  = {Zhang, Yuming and Ma, Juan},
  title   = {Nonparametric Gaussian mixture models enhanced by sequential importance sampling for high-dimensional rare event analysis},
  journal = {Computers \& Structures},
  year    = {2026},
  volume  = {325},
  pages   = {108198},
  doi     = {10.1016/j.compstruc.2026.108198}
}
```
USE: Gaussian-mixture proposals with sequential IS in high dimension. Cite with the mixture-proposal line of work when stating that mixture proposals are now common in reliability but rarely certified.

```bibtex
@article{Lukashevich2024Chance,
  author  = {Lukashevich, Aleksander and Gorchakov, Vyacheslav and Vorobev, Petr and Deka, Deepjyoti and Maximov, Yury},
  title   = {Importance Sampling Approach to Chance-Constrained DC Optimal Power Flow},
  journal = {IEEE Transactions on Control of Network Systems},
  year    = {2024},
  volume  = {11},
  number  = {2},
  pages   = {928--937},
  doi     = {10.1109/tcns.2023.3330232}
}
```
USE: IS for chance constraints in a networked operations problem, with Maximov as co-author of the union-IS paper. Cite as the closest application of IS to a chance-constrained planning problem, showing that IS is now used inside an optimisation loop, not only for post hoc estimation.

## 2. Adaptive, cross-entropy, normalizing-flow and neural IS (2020+)

```bibtex
@article{Dasgupta2024REIN,
  author  = {Dasgupta, Agnimitra and Johnson, Erik A.},
  title   = {REIN: Reliability Estimation via Importance sampling with Normalizing flows},
  journal = {Reliability Engineering \& System Safety},
  year    = {2024},
  volume  = {242},
  pages   = {109729},
  doi     = {10.1016/j.ress.2023.109729}
}
```
USE: Normalizing-flow proposal for reliability IS. Cite as the flow-based adaptive IS baseline, and note that the flow's fit error is not in the reported CI, which motivates a separate certification step.

```bibtex
@article{Asghar2024Flows,
  author  = {Asghar, Solomon and Pei, Qing-Xiang and Volpe, Giorgio and Ni, Ran},
  title   = {Efficient rare event sampling with unsupervised normalizing flows},
  journal = {Nature Machine Intelligence},
  year    = {2024},
  volume  = {6},
  number  = {11},
  pages   = {1370--1381},
  doi     = {10.1038/s42256-024-00918-3}
}
```
USE: Unsupervised normalizing flows for rare-event sampling in physics. Cite in the neural-IS paragraph as evidence that flows reach rare regimes but give no finite-sample guarantee on the tail probability.

```bibtex
@article{Zuniga2021Surrogate,
  author  = {Zuniga, M. Munoz and Murangira, A. and Perdrizet, T.},
  title   = {Structural reliability assessment through surrogate based importance sampling with dimension reduction},
  journal = {Reliability Engineering \& System Safety},
  year    = {2021},
  volume  = {207},
  pages   = {107289},
  doi     = {10.1016/j.ress.2020.107289}
}
```
USE: Surrogate-based IS with dimension reduction. Cite as an example of how dimension reduction changes the proposal, and of the bias that surrogate-plus-IS pipelines can carry.

## 3. Subset simulation, multilevel splitting, SMC (2019+)

```bibtex
@article{Cerou2019Adaptive,
  author  = {C{\'e}rou, Fr{\'e}d{\'e}ric and Guyader, Arnaud and Rousset, Mathias},
  title   = {Adaptive multilevel splitting: Historical perspective and recent results},
  journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  year    = {2019},
  volume  = {29},
  number  = {4},
  doi     = {10.1063/1.5082247}
}
```
USE: Review of adaptive multilevel splitting (AMS). Cite in the comparison paragraph: AMS handles path-dependent rare events well but gives a different variance structure from IS, which matters for the certified CI.

```bibtex
@article{Wagner2020Multilevel,
  author  = {Wagner, F. and Latz, J. and Papaioannou, I. and Ullmann, E.},
  title   = {Multilevel Sequential Importance Sampling for Rare Event Estimation},
  journal = {SIAM Journal on Scientific Computing},
  year    = {2020},
  volume  = {42},
  number  = {4},
  pages   = {A2062--A2087},
  doi     = {10.1137/19m1289601}
}
```
USE: Multilevel SMC/IS for rare events. Cite as the SMC-based rival that sits between subset simulation and IS, and to state that multilevel schemes need a level-wise error budget that a single-shift IS does not.

```bibtex
@article{Du2021Variance,
  author  = {Du, Qiming and Guyader, Arnaud},
  title   = {Variance estimation in adaptive sequential Monte Carlo},
  journal = {The Annals of Applied Probability},
  year    = {2021},
  volume  = {31},
  number  = {3},
  pages   = {1021--1060},
  doi     = {10.1214/20-aap1611}
}
```
USE: Asymptotically valid variance estimators for adaptive SMC. Cite when the paper says a rare-event CI from an adaptive scheme needs its own variance theory, and that a plug-in variance from the weights is not enough.

```bibtex
@article{Abdollahi2020Refined,
  author  = {Abdollahi, Azam and Azhdary Moghaddam, Mehdi and Hashemi Monfared, Seyed Arman and Rashki, Mohsen and Li, Yong},
  title   = {A refined subset simulation for the reliability analysis using the subset control variate},
  journal = {Structural Safety},
  year    = {2020},
  volume  = {87},
  pages   = {102002},
  doi     = {10.1016/j.strusafe.2020.102002}
}
```
USE: Control variates to reduce subset-simulation variance. Cite as a recent subset-simulation variance-reduction method, in the paragraph comparing subset simulation with IS on cost per digit of accuracy.

## 4. Large deviations, maxima and copula-driven sums (2018+)

```bibtex
@article{Cossette2019Archimedean,
  author  = {Cossette, H{\'e}l{\`e}ne and Marceau, Etienne and Nguyen, Quang Huy and Robert, Christian Y.},
  title   = {Tail Approximations for Sums of Dependent Regularly Varying Random Variables Under Archimedean Copula Models},
  journal = {Methodology and Computing in Applied Probability},
  year    = {2019},
  volume  = {21},
  number  = {2},
  pages   = {461--490},
  doi     = {10.1007/s11009-017-9614-z}
}
```
USE: Conditional MC for sums of dependent heavy-tailed variables under copulas, with bounded relative error. Cite as the copula-dependence analogue of the conditional-MC step, and to note it covers Archimedean rather than Gaussian copulas, so a Gaussian-copula case remains open in the window.

```bibtex
@article{Karim2025Hawkes,
  author  = {Karim, Raviar S. and Laeven, Roger J. A. and Mandjes, Michel},
  title   = {Compound multivariate Hawkes processes: Large deviations and rare event simulation},
  journal = {Bernoulli},
  year    = {2025},
  volume  = {31},
  number  = {4},
  pages   = {3113--3138},
  doi     = {10.3150/24-bej1840}
}
```
USE: Large deviations and rare-event simulation for a dependent (self-exciting) process. Cite as a recent LD-plus-IS treatment of dependence, to set up the claim that asymptotic efficiency for dependent maxima is still an active area.

```bibtex
@inproceedings{Conroy2022Maxima,
  author    = {Conroy, Michael and Olvera-Cravioto, Mariana},
  title     = {Efficient Rare Event Estimation for Maxima of Branching Random Walks},
  booktitle = {2022 Winter Simulation Conference (WSC)},
  year      = {2022},
  pages     = {145--155},
  doi       = {10.1109/WSC57314.2022.10015264}
}
```
USE: Efficient rare-event estimation for the maximum of a dependent random process. Cite beside the stage-wise maximum in the paper: the overflow event for a plan is a maximum over stages, and this is the nearest recent method for estimating a maximum's tail.

```bibtex
@article{Moka2026RGG,
  author  = {Moka, Sarat and Hirsch, Christian and Schmidt, Volker and Kroese, Dirk},
  title   = {Efficient Rare-Event Simulation for Random Geometric Graphs via Importance Sampling},
  journal = {Methodology and Computing in Applied Probability},
  year    = {2026},
  volume  = {28},
  number  = {2},
  doi     = {10.1007/s11009-026-10273-y}
}
```
USE: IS for rare events on a dependent random structure (random geometric graphs). Cite as a 2026 IS-with-dependence example, supporting the claim that single-shift IS on dependent structures needs a graph- or copula-aware shift.

## 5. Diagnostics: heavy-tailed weights, ESS, confidence intervals, silent failure

```bibtex
@article{Chatterjee2018Sample,
  author  = {Chatterjee, Sourav and Diaconis, Persi},
  title   = {The sample size required in importance sampling},
  journal = {The Annals of Applied Probability},
  year    = {2018},
  volume  = {28},
  number  = {2},
  pages   = {1099--1135},
  doi     = {10.1214/17-aap1326}
}
```
USE: Sample size required for IS to be reliable; the number of samples scales with the exponential of the KL divergence between target and proposal. Cite as the theoretical reason that a single-shift IS can silently under-report an overflow probability at 1e-6, i.e. the basis of a certification bound on ESS-like diagnostics.

```bibtex
@article{Vehtari2024PSIS,
  author  = {Vehtari, Aki and Simpson, Daniel and Gelman, Andrew and Yao, Yuling and Gabry, Jonah},
  title   = {Pareto Smoothed Importance Sampling},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {72},
  pages   = {1--58},
  eprint  = {1507.02646},
  archivePrefix = {arXiv}
}
% No DOI: none found in Crossref. Journal reference taken from the arXiv abstract page (arXiv:1507.02646).
```
USE: Pareto-tail diagnostic for IS weights (k-hat), with a finite-sample convergence rate diagnostic (abstract). Cite as the heavy-tail weight diagnostic to report beside each certified estimate; the k-hat threshold value must be taken from the paper body, which was not checked here.

```bibtex
@incollection{Chan2020Infinite,
  author    = {Chan, Joshua C. C. and Hou, Chenghan and Yang, Thomas Tao},
  title     = {Robust Estimation and Inference for Importance Sampling Estimators with Infinite Variance},
  booktitle = {Essays in Honor of Cheng Hsiao},
  year      = {2020},
  pages     = {255--285}
}
% Editors and DOI not seen in the Crossref record; the record was checked via Crossref query and the DOI 10.1108/s0731-905320200000041008 was seen in the Crossref record.
```
USE: Robust inference for IS estimators with infinite-variance weights. Cite when a certified CI must remain valid when the weight variance is infinite, i.e. where a naive Gaussian CI is not justified.

```bibtex
@inproceedings{LEcuyer2023RQMC,
  author    = {L'Ecuyer, Pierre and Nakayama, Marvin K. and Owen, Art B. and Tuffin, Bruno},
  title     = {Confidence Intervals for Randomized Quasi-Monte Carlo Estimators},
  booktitle = {2023 Winter Simulation Conference (WSC)},
  year      = {2023},
  pages     = {445--456},
  doi       = {10.1109/WSC60868.2023.10408613}
}
```
USE: Valid confidence intervals for randomized QMC, and the method that the QMC part of the certificate would use. Cite in the QMC paragraph, and to note that RQMC CIs need the randomization to be reported with the estimate, which the certificate must do.

---

## Gaps and leads not included

These are not in the file because they were not verified in Crossref or on an arXiv abstract page during this search.

- **Conditional MC, 2018+:** the search found only Cossette et al. (2019) inside the window. Leads needing a Crossref check: Kortschak & Hashorva (log-elliptical sums), Blanchet & Rojas-Nandayapa (log-elliptical sums, classic), Asmussen & Rojas-Nandayapa (Gaussian-copula asymptotics, 2008). The older Chan & Kroese (2009) conditional-MC paper was found in Crossref but is outside the window.
- **Eshra, Papakonstantinou & Nikbakht (2025),** direct IS in non-Gaussian spaces, Reliability Engineering & System Safety 264, 111200, DOI 10.1016/j.ress.2025.111200. Verified in Crossref; dropped only because of the 25-entry cap.
- **Gaussian-copula maxima, 2018+:** no verified in-window work. Topic 4 is thin; the cited items are about dependent sums or maxima but not Gaussian-copula maxima specifically.
- **Subset simulation, pre-2019 anchor:** Papaioannou et al. (2016) was not verified in this session.
- **Naderi & Jia (2026),** multi-fidelity subset simulation, Reliability Engineering & System Safety 266, 111739, DOI 10.1016/j.ress.2025.111739, appeared in Crossref. Not included because the 25-entry cap was reached. Strong candidate for topic 3 if the cap is relaxed.
- **Zheng, Yuan, Bao & Valdebenito & Faes (2026),** active Bayesian SVR with IS, DOI 10.1016/j.ress.2026.112896, appeared in Crossref. Not included, same reason.
- **Rached, Haji-Ali, Subbiah Pillai & Tempone (2025),** multilevel IS for McKean-Vlasov, DOI 10.1007/s11222-024-10508-3, Statistics and Computing 35(1). Verified in Crossref but outside the core topics.
- **Zhao & Chen (2025),** "When Machine Learning Meets Importance Sampling", WSC 2025, DOI 10.1109/wsc68292.2025.11338953. Only the Crossref record was seen; the content was not checked.
- **Nagy (2026),** "Spectral Importance Sampling", SSRN 7000958. Appeared in search; content not checked. Not used.
