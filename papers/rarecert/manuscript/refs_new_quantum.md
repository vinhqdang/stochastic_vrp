# Verified quantum references for the RARECERT quantum note

Every entry below was checked against the Crossref API (`api.crossref.org/works/<doi>` or a bibliographic query) or the arXiv abstract page (`arxiv.org/abs/<id>`). Fields not visible there are omitted rather than guessed. Entries marked "arXiv only" have no journal record in the checked sources. Author lists follow Crossref or arXiv, not the names in the original request, which were wrong for several works.

Content claims in the USE lines are based on the abstract metadata and on search summaries. Read the abstract before quoting any numerical claim.

---

## Topic 1: QAE without phase estimation, practical and low-depth variants

```bibtex
@article{Grinko2021Iterative,
  author  = {Grinko, Dmitry and Gacon, Julien and Zoufal, Christa and Woerner, Stefan},
  title   = {Iterative quantum amplitude estimation},
  journal = {npj Quantum Information},
  volume  = {7},
  number  = {1},
  pages   = {52},
  year    = {2021},
  doi     = {10.1038/s41534-021-00379-1},
  eprint  = {1912.05559},
  archivePrefix = {arXiv}
}
```
USE: Introduction and QAE background. Cite as the phase-estimation-free QAE (Grover iterations only) that a practical rare-event comparison would actually run on near-term hardware.

```bibtex
@article{Suzuki2020Amplitude,
  author  = {Suzuki, Yohichi and Uno, Shumpei and Raymond, Rudy and Tanaka, Tomoki and Onodera, Tamiya and Yamamoto, Naoki},
  title   = {Amplitude estimation without phase estimation},
  journal = {Quantum Information Processing},
  volume  = {19},
  number  = {2},
  pages   = {75},
  year    = {2020},
  doi     = {10.1007/s11128-019-2565-2}
}
```
USE: QAE variants section. Cite as the maximum-likelihood approach that combines measurements from circuits with different numbers of amplification steps and avoids controlled operations.

```bibtex
@article{Tanaka2021Amplitude,
  author  = {Tanaka, Tomoki and Suzuki, Yohichi and Uno, Shumpei and Raymond, Rudy and Onodera, Tamiya and Yamamoto, Naoki},
  title   = {Amplitude estimation via maximum likelihood on noisy quantum computer},
  journal = {Quantum Information Processing},
  volume  = {20},
  number  = {9},
  pages   = {293},
  year    = {2021},
  doi     = {10.1007/s11128-021-03215-9},
  eprint  = {2006.16223},
  archivePrefix = {arXiv}
}
```
USE: Noise section. Cite as the noisy-hardware extension of maximum-likelihood QAE, which the note should mention when discussing depolarizing noise in a comparison.

```bibtex
@misc{Callison2022Improved,
  author  = {Callison, Adam and Browne, Dan E.},
  title   = {Improved maximum-likelihood quantum amplitude estimation},
  year    = {2022},
  eprint  = {2209.03321},
  archivePrefix = {arXiv},
  note    = {arXiv only; no journal record checked}
}
```
USE: QAE variants section. Cite as a later improvement to maximum-likelihood QAE, to show the ML line is still developing and to support the claim that the variant comparison is not frozen.

```bibtex
@article{Giurgica2022Low,
  author  = {Giurgica-Tiron, Tudor and Kerenidis, Iordanis and Labib, Farrokh and Prakash, Anupam and Zeng, William},
  title   = {Low depth algorithms for quantum amplitude estimation},
  journal = {Quantum},
  volume  = {6},
  pages   = {745},
  year    = {2022},
  doi     = {10.22331/q-2022-06-27-745},
  eprint  = {2012.03348},
  archivePrefix = {arXiv}
}
```
USE: Depth discussion. Cite as the depth-versus-oracle-calls tradeoff (power-law schedules, interpolating between classical and standard quantum query counts), relevant to any claim about depth-limited QAE for rare events.

```bibtex
@article{Nakaji2020Faster,
  author  = {Nakaji, Kouhei},
  title   = {Faster amplitude estimation},
  journal = {Quantum Information and Computation},
  volume  = {20},
  number  = {13\&14},
  pages   = {1109--1123},
  year    = {2020},
  doi     = {10.26421/QIC20.13-14-2},
  eprint  = {2003.02417},
  archivePrefix = {arXiv}
}
```
USE: QAE variants section. Cite as an earlier phase-estimation-free or reduced-overhead QAE variant, to show the variant space that the note compares.

```bibtex
@misc{Brassard2002Amplitude,
  author  = {Brassard, Gilles and H{\o}yer, Peter and Mosca, Michele and Tapp, Alain},
  title   = {Quantum amplitude amplification and estimation},
  year    = {2002},
  note    = {Contemporary Mathematics 305, pp. 53--74. Preprint arXiv:quant-ph/0005055 (2000). Foundational work, outside the 2019--2026 window.},
  doi     = {10.1090/conm/305/05215},
  eprint  = {quant-ph/0005055},
  archivePrefix = {arXiv}
}
```
USE: Background section on amplitude estimation and its phase-estimation core. Cite as the original QAE algorithm (the baseline that all variants above modify).

---

## Topic 2: Quantum Monte Carlo for risk and finance

```bibtex
@article{Woerner2019Quantum,
  author  = {Woerner, Stefan and Egger, Daniel J.},
  title   = {Quantum risk analysis},
  journal = {npj Quantum Information},
  volume  = {5},
  number  = {1},
  pages   = {15},
  year    = {2019},
  doi     = {10.1038/s41534-019-0130-6},
  eprint  = {1806.06893},
  archivePrefix = {arXiv}
}
```
USE: Rare-event and risk-measure section. Cite as the first QAE estimator of VaR and CVaR, with the O(M^-2/3) convergence claim for the shortest-depth circuit that the note should quote with its caveats.

```bibtex
@article{Stamatopoulos2020Option,
  author  = {Stamatopoulos, Nikitas and Egger, Daniel J. and Sun, Yue and Zoufal, Christa and Iten, Raban and Shen, Ning and Woerner, Stefan},
  title   = {Option pricing using quantum computers},
  journal = {Quantum},
  volume  = {4},
  pages   = {291},
  year    = {2020},
  doi     = {10.22331/q-2020-07-06-291},
  eprint  = {1905.02666},
  archivePrefix = {arXiv}
}
```
USE: Introduction and finance background. Cite as the canonical QAE-for-expectation application that the Monte Carlo speedup claim rests on, and as the source of the circuit-construction details for payoff-based estimators.

```bibtex
@article{Egger2021Credit,
  author  = {Egger, Daniel J. and Garc{\'i}a Guti{\'e}rrez, Ricardo and Mestre, Jordi Cahu{\'e} and Woerner, Stefan},
  title   = {Credit risk analysis using quantum computers},
  journal = {IEEE Transactions on Computers},
  volume  = {70},
  number  = {12},
  pages   = {2136--2145},
  year    = {2021},
  doi     = {10.1109/tc.2020.3038063},
  eprint  = {1907.03044},
  archivePrefix = {arXiv}
}
```
USE: Rare-event section. Cite as the economic-capital (VaR minus expected loss, a tail quantity) QAE estimator on a credit portfolio, the closest application to a tail-probability comparison in the financial literature.

```bibtex
@article{Chakrabarti2021Threshold,
  author  = {Chakrabarti, Shouvanik and Krishnakumar, Rajiv and Mazzola, Guglielmo and Stamatopoulos, Nikitas and Woerner, Stefan and Zeng, William J.},
  title   = {A threshold for quantum advantage in derivative pricing},
  journal = {Quantum},
  volume  = {5},
  pages   = {463},
  year    = {2021},
  doi     = {10.22331/q-2021-06-01-463},
  eprint  = {2012.03819},
  archivePrefix = {arXiv}
}
```
USE: Discussion of fault-tolerant resources. Cite as the resource estimate showing that a QAE advantage for pricing needs very large logical qubit counts and T-depth, the benchmark any QAE-versus-importance-sampling claim should be checked against.

```bibtex
@misc{Laudage2022Beyond,
  author  = {Laudag{\'e}, Christian and Turkalj, Ivica},
  title   = {Quantum risk analysis: Beyond (conditional) value-at-risk},
  year    = {2022},
  eprint  = {2211.04456},
  archivePrefix = {arXiv},
  note    = {arXiv only; no journal record checked}
}
```
USE: VaR and CVaR section. Cite as QAE estimators for expectile and range VaR, with the reported result that CVaR and RVaR do not survive device noise while VaR and EVaR do.

```bibtex
@article{Herbert2022Quantum,
  author  = {Herbert, Steven},
  title   = {Quantum Monte Carlo integration: The full advantage in minimal circuit depth},
  journal = {Quantum},
  volume  = {6},
  pages   = {823},
  year    = {2022},
  doi     = {10.22331/q-2022-09-29-823}
}
```
USE: Fault-tolerance and depth section. Cite as a minimal-depth quantum Monte Carlo integration result that narrows the gap between the QAE and classical estimates when depth is counted.

---

## Topic 3: Quantum speedups for Monte Carlo, dequantization caveats

```bibtex
@inproceedings{Tang2019Quantum,
  author    = {Tang, Ewin},
  title     = {A quantum-inspired classical algorithm for recommendation systems},
  booktitle = {Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing},
  pages     = {217--228},
  year      = {2019},
  doi       = {10.1145/3313276.3316310},
  eprint    = {1807.04271},
  archivePrefix = {arXiv}
}
```
USE: Caveats section. Cite as the first dequantization result, to show that exponential quantum speedups can disappear when the classical side gets comparable sampling access to the data.

```bibtex
@article{Chia2022Sampling,
  author  = {Chia, Nai-Hui and Gily{\'e}n, Andr{\'a}s Pal and Li, Tongyang and Lin, Han-Hsuan and Tang, Ewin and Wang, Chunhao},
  title   = {Sampling-based sublinear low-rank matrix arithmetic framework for dequantizing quantum machine learning},
  journal = {Journal of the ACM},
  volume  = {69},
  number  = {5},
  pages   = {1--72},
  year    = {2022},
  doi     = {10.1145/3549524}
}
```
USE: Caveats section. Cite as the general dequantization framework (quantum SVT in the QRAM input model), supporting the statement that the claimed advantage must be checked under matched input assumptions.

```bibtex
@inproceedings{Blanchet2025Quantum,
  author    = {Blanchet, Jose and Hamoudi, Yassine and Szegedy, Mario and Wang, Guanyang},
  title     = {Quantum speedup of non-linear Monte Carlo problems},
  booktitle = {Advances in Neural Information Processing Systems 38},
  pages     = {21289--21313},
  year      = {2025},
  doi       = {10.52202/085713-0631}
}
```
USE: Speedup section. Cite as the quadratic speedup for non-linear and infinite-variance Monte Carlo quantities, the case that matters for heavy-tailed overflow loads. The arXiv preprint is arXiv:2401.07497 (Blanchet, Szegedy, Wang, 2024).

```bibtex
@misc{Intallura2023Survey,
  author  = {Intallura, Philip and Korpas, Georgios and Chakraborty, Sudeepto and Lawrence, Rufus and Wodecki, Ales and Kungurtsev, Vyacheslav and Mare{\v{c}}ek, Jakub},
  title   = {A survey of quantum alternatives to randomized algorithms: Monte Carlo integration and beyond},
  year    = {2023},
  eprint  = {2303.04945},
  archivePrefix = {arXiv},
  note    = {arXiv only; no journal record checked}
}
```
USE: Introduction and related work. Cite as a survey of quantum Monte Carlo alternatives, with the query-complexity framing that the note should use when comparing QAE with classical sampling.

---

## Topic 4: Quantum algorithms for vehicle routing

```bibtex
@article{Feld2019Hybrid,
  author  = {Feld, Sebastian and Roch, Christoph and Gabor, Thomas and Seidel, Christian and Neukart, Florian and Galter, Isabella and Mauerer, Wolfgang and Linnhoff-Popien, Claudia},
  title   = {A hybrid solution method for the capacitated vehicle routing problem using a quantum annealer},
  journal = {Frontiers in ICT},
  volume  = {6},
  pages   = {13},
  year    = {2019},
  doi     = {10.3389/fict.2019.00013},
  eprint  = {1811.07403},
  archivePrefix = {arXiv}
}
```
USE: Related-work section on quantum routing. Cite as an early annealing approach to CVRP, to show that the routing side of quantum work is deterministic and that stochastic demand has not been handled in this line.

```bibtex
@article{Fitzek2024Applying,
  author  = {Fitzek, David and Ghandriz, Toheed and Laine, Leo and Granath, Mats and Kockum, Anton Frisk},
  title   = {Applying quantum approximate optimization to the heterogeneous vehicle routing problem},
  journal = {Scientific Reports},
  volume  = {14},
  number  = {1},
  pages   = {25415},
  year    = {2024},
  doi     = {10.1038/s41598-024-76967-w},
  eprint  = {2110.06799},
  archivePrefix = {arXiv}
}
```
USE: Related-work section on QAOA for routing. Cite as QAOA on heterogeneous-fleet VRP with reported qubit scaling in the number of customers, which supports the note's point that routing qubit counts grow quickly.

---

## Topic 5: Reliability, power-system and structural failure probabilities

```bibtex
@misc{Jong2023Quantum,
  author  = {Jong, Emilie and S{\ae}varsson, Brynjar and J{\'o}hannsson, Hj{\"o}rtur and Chatzivasileiadis, Spyros},
  title   = {Quantum amplitude estimation for probabilistic methods in power systems},
  year    = {2023},
  eprint  = {2309.17299},
  archivePrefix = {arXiv},
  note    = {arXiv only; no journal record checked}
}
```
USE: Applications section on rare failure probabilities. Cite as a QAE-versus-Monte-Carlo comparison in power-system reliability (per search summary, comparing IQAE, MLAE and FAE across distribution types; confirm on the full text before quoting).

```bibtex
@misc{Tabarraei2026Bayesian,
  author  = {Tabarraei, Alireza},
  title   = {Bayesian sequential quantum amplitude estimation for rare-event structural failure probability},
  year    = {2026},
  eprint  = {2607.18996},
  archivePrefix = {arXiv},
  note    = {arXiv only; submitted 2026-07-21}
}
```
USE: Rare-event section. Cite as the most direct structural rare-event QAE application (failure probability near 1.5e-5 per search summary). Treat as a preprint.

---

## Topic 6: Surveys of quantum logistics, and quantum advantage for Monte Carlo

```bibtex
@misc{Dalzell2023Survey,
  author  = {Dalzell, Alexander M. and McArdle, Sam and Berta, Mario and Bienias, Przemyslaw and Chen, Chi-Fang and Gily{\'e}n, Andr{\'a}s and Hann, Connor T. and Kastoryano, Michael J. and Khabiboulline, Emil T. and Kubica, Aleksander and Salton, Grant and Wang, Samson and Brand{\~a}o, Fernando G. S. L.},
  title   = {Quantum algorithms: A survey of applications and end-to-end complexities},
  year    = {2023},
  eprint  = {2310.03011},
  archivePrefix = {arXiv},
  note    = {Crossref lists the monograph DOI 10.1017/9781009639651 (2025), titled Quantum Algorithms; the publisher was not checked}
}
```
USE: Discussion of end-to-end cost. Cite as the reference that counts end-to-end complexity (including state preparation and oracle costs) for QAE-based Monte Carlo, the caveat that the quadratic speedup alone does not decide the comparison.

```bibtex
@misc{Phillipson2024Quantum,
  author  = {Phillipson, Frank},
  title   = {Quantum computing in logistics and supply chain management an overview},
  year    = {2024},
  eprint  = {2402.17520},
  archivePrefix = {arXiv},
  note    = {arXiv only; no journal record checked}
}
```
USE: Introduction on quantum logistics. Cite as an overview of quantum routing and logistics, with the note that most reviewed work is hybrid or annealing-based.

---

## Recent QAE-versus-classical tail-probability studies (preprints)

```bibtex
@misc{Kirke2026Quantum,
  author  = {Kirke, Alexis},
  title   = {Quantum amplitude estimation for catastrophe insurance tail-risk pricing: Empirical convergence and NISQ noise analysis},
  year    = {2026},
  eprint  = {2603.15664},
  archivePrefix = {arXiv},
  note    = {arXiv only; submitted 2026-03-10}
}
```
USE: Core comparison section. Cite as a direct QAE-versus-importance-sampling tail-quantile comparison. Per search summary, classical conditional tail Monte Carlo, importance sampling and QMC outperformed QAE, and NISQ noise removed the advantage. Read the full paper before quoting these results.

```bibtex
@misc{Srikant2026Quantum,
  author  = {Srikant, R.},
  title   = {Quantum estimation of delay tail probabilities in scheduling and load balancing},
  year    = {2026},
  eprint  = {2602.09059},
  archivePrefix = {arXiv},
  note    = {arXiv only; submitted 2026-02-08}
}
```
USE: Core comparison section. Cite as the theoretical argument that QAE avoids problem-specific change-of-measure design that importance sampling needs, with a sample-count estimate at probabilities 1e-6 to 1e-9 (a theoretical claim, not a benchmark).

---

## Entries considered but not included

- Montanaro, "Quantum speedup of Monte Carlo methods" (arXiv:1504.06987; Proc. R. Soc. A 471, 20150301). Verified, but published in 2015, outside the requested window.
- Gómez et al., "A survey on quantum computational finance for derivatives pricing and VaR" (DOI 10.1007/s11831-022-09732-9; Arch. Comput. Methods Eng. 29, 4137-4163). Verified in Crossref. Dropped only to stay within 25 entries.
- Gachnang et al., "Quantum computing in supply chain management state of the art and research directions" (DOI 10.14710/ajlm.2022.14325; Asian J. Logistics Manag. 1(1), 57-73). Verified in Crossref. Dropped for length.
- Tabarraei, "Stabilized maximum-likelihood iterative quantum amplitude estimation for structural CVaR" (arXiv:2602.09847, 2026). Verified on arXiv. Dropped for length.
- Leonidas et al., "Qubit efficient quantum algorithms for the vehicle routing problem on NISQ processors" (arXiv:2306.08507, 2023). Verified on arXiv. Dropped for length.
- No quantum paper on stochastic-demand or chance-constrained routing was found in two searches. Topic 4 coverage is deterministic only.
- No 2021-2026 quantum importance-sampling paper was found for rare-event probabilities. The closest match is the 2026 Kirke preprint above.

## Author corrections against the original request

The names in the original request did not match the records for these works: Grinko et al. (Burgholzer and Marina are not authors), Chakrabarti et al. (Gribling, Modi, Minssen and Egger are not authors), Brassard et al. (Grover is not an author), and Egger et al. IEEE TC (the author list is Egger, García Gutiérrez, Mestre, Woerner). Entries above use the Crossref or arXiv author lists.
