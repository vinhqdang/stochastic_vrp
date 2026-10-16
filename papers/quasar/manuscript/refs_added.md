# refs_added.md — verification of added/open citations (2026-10-16)

Method: Crossref API (DOI records), DataCite API (LIPIcs DOIs, which Crossref does not hold; each also resolves at doi.org), arXiv API (title, full author list, journal-ref, comments, abstract). Only abstract/record-level facts were checked; no theorem numbers were verified here.

## Verified, new entries

### 1. Huggins2022 — VERIFIED (Crossref + arXiv v3)
```bibtex
@article{Huggins2022,
  author  = {Huggins, William J. and Wan, Kianna and McClean, Jarrod and O'Brien, Thomas E. and Wiebe, Nathan and Babbush, Ryan},
  title   = {Nearly Optimal Quantum Algorithm for Estimating Multiple Expectation Values},
  journal = {Physical Review Letters},
  volume  = {129},
  number  = {24},
  pages   = {240501},
  year    = {2022},
  doi     = {10.1103/PhysRevLett.129.240501},
  eprint  = {2111.09283},
  archivePrefix = {arXiv},
}
```
USE: quantum baseline for simultaneous estimation of M expectation values (O~(sqrt(M)/eps)-type cost with matching lower bound), against which per-route shared-sample certification is compared.

### 2. Tang2025 — VERIFIED as arXiv preprint; NO journal version found
Current arXiv title (v3, 2025-05-29) is "More-efficient Quantum Multivariate Mean Value Estimator from Generalized Grover Operator" (v1/v2 titled "...Generalized Grover Gate"). Single author Letian Tang. Comment: "39 pages". No journal_ref or DOI in the arXiv record; web search found none. Cite as preprint.
```bibtex
@misc{Tang2025,
  author = {Tang, Letian},
  title  = {More-efficient Quantum Multivariate Mean Value Estimator from Generalized Grover Operator},
  year   = {2025},
  eprint = {2504.06940},
  archivePrefix = {arXiv},
  note   = {Version 3; preprint, no journal version located},
}
```
USE: best known quantum mean estimation in l_infinity with O(n log(d/delta)) samples for error sqrt(tr Sigma)/n; the quantum comparator for simultaneous multi-route estimation. Abstract states a log(d/delta) gap to the lower bound.

### 3. vanApeldoorn2021 — VERIFIED (DataCite)
Single author; LIPIcs vol. 197, article 9, pages 9:1-9:11.
```bibtex
@inproceedings{vanApeldoorn2021,
  author    = {van Apeldoorn, Joran},
  title     = {Quantum Probability Oracles \& Multidimensional Amplitude Estimation},
  booktitle = {16th Conference on the Theory of Quantum Computation, Communication and Cryptography (TQC 2021)},
  series    = {LIPIcs},
  volume    = {197},
  pages     = {9:1--9:11},
  year      = {2021},
  doi       = {10.4230/LIPIcs.TQC.2021.9},
}
```
(The TQC 2021 edition number "16th" is from memory; drop that word if unchecked. DOI, volume, pages, title, author are verified.)
USE: multidimensional amplitude estimation via probability oracles; cite as the prior vector-valued quantum estimator.

### 4. Cornelissen2022 — ALREADY in references.bib; SKIP
Re-checked against Crossref (10.1145/3519935.3520045): authors, STOC 2022 pages 33-43, year 2022 all match. arXiv id 2111.09787 consistent with DataCite citation lists. Note Crossref title capitalisation "Near-optimal Quantum algorithms..."; the existing entry is fine.

### 5. Audibert2009 — VERIFIED (Crossref)
```bibtex
@article{Audibert2009,
  author  = {Audibert, Jean-Yves and Munos, R{\'e}mi and Szepesv{\'a}ri, Csaba},
  title   = {Exploration--exploitation tradeoff using variance estimates in multi-armed bandits},
  journal = {Theoretical Computer Science},
  volume  = {410},
  number  = {19},
  pages   = {1876--1902},
  year    = {2009},
  doi     = {10.1016/j.tcs.2009.01.016},
}
```
USE: variance-adaptive (empirical-Bernstein) classical sampling for bandit-style allocation; classical variance-sensitive comparator.

### 6. Maurer2009 — VERIFIED (arXiv record); no DOI
arXiv 0907.3740v1, authors Maurer, Pontil; comment "Proc. Computational Learning Theory Conference (COLT 2009)". Page numbers not verified; omitted.
```bibtex
@inproceedings{Maurer2009,
  author    = {Maurer, Andreas and Pontil, Massimiliano},
  title     = {Empirical {Bernstein} Bounds and Sample Variance Penalization},
  booktitle = {Proceedings of the 22nd Annual Conference on Learning Theory (COLT)},
  year      = {2009},
  eprint    = {0907.3740},
  archivePrefix = {arXiv},
}
```
(COLT edition "22nd" from memory; arXiv record only states COLT 2009.)
USE: data-dependent empirical-Bernstein confidence bound for the classical sequential/certification baseline.

### 7. Barenco1995 — VERIFIED (Crossref)
```bibtex
@article{Barenco1995,
  author  = {Barenco, Adriano and Bennett, Charles H. and Cleve, Richard and DiVincenzo, David P. and Margolus, Norman and Shor, Peter and Sleator, Tycho and Smolin, John A. and Weinfurter, Harald},
  title   = {Elementary gates for quantum computation},
  journal = {Physical Review A},
  volume  = {52},
  number  = {5},
  pages   = {3457--3467},
  year    = {1995},
  doi     = {10.1103/PhysRevA.52.3457},
}
```
USE: multi-controlled gate decompositions underlying the gate counts of the reversible peak-load oracle.

### 8. Hoeffding1963 — VERIFIED (Crossref)
```bibtex
@article{Hoeffding1963,
  author  = {Hoeffding, Wassily},
  title   = {Probability Inequalities for Sums of Bounded Random Variables},
  journal = {Journal of the American Statistical Association},
  volume  = {58},
  number  = {301},
  pages   = {13--30},
  year    = {1963},
  doi     = {10.1080/01621459.1963.10500830},
}
```
(Issue number 301 is from memory; Crossref confirmed volume 58, pages 13-30, March 1963. Drop `number` if unchecked.)
USE: classical sample-size bound O(log(1/delta)/eps^2) for a bounded indicator.

### 9. deBoer2005 — VERIFIED (Crossref)
```bibtex
@article{deBoer2005,
  author  = {de Boer, Pieter-Tjerk and Kroese, Dirk P. and Mannor, Shie and Rubinstein, Reuven Y.},
  title   = {A Tutorial on the Cross-Entropy Method},
  journal = {Annals of Operations Research},
  volume  = {134},
  number  = {1},
  pages   = {19--67},
  year    = {2005},
  doi     = {10.1007/s10479-005-5724-z},
}
```
USE: cross-entropy tilting as a classical rare-event IS baseline.

### 10. Glasserman2005 — VERIFIED (Crossref)
```bibtex
@article{Glasserman2005,
  author  = {Glasserman, Paul and Li, Jingyi},
  title   = {Importance Sampling for Portfolio Credit Risk},
  journal = {Management Science},
  volume  = {51},
  number  = {11},
  pages   = {1643--1656},
  year    = {2005},
  doi     = {10.1287/mnsc.1050.0415},
}
```
USE: factor-shift plus conditional-twist IS that RARECERT extends from a sum to a maximum over stages.

### 11. Owen2019 — VERIFIED (Crossref; DOI is 10.1214/18-EJS1527)
```bibtex
@article{Owen2019,
  author  = {Owen, Art B. and Maximov, Yury and Chertkov, Michael},
  title   = {Importance sampling the union of rare events with an application to power systems analysis},
  journal = {Electronic Journal of Statistics},
  volume  = {13},
  number  = {1},
  pages   = {231--254},
  year    = {2019},
  doi     = {10.1214/18-EJS1527},
}
```
USE: mixture IS for a union of rare events; the baseline a RARECERT estimator is compared against (not claimable as ours).

### 12. Wainwright2019 — textbook only; NO specific shared-sample l_infinity lower bound found
Searched; no citable theorem that is a classical Omega(log d / eps^2) lower bound for simultaneous estimation of d Bernoulli means from shared samples was identified. Verified book record (Crossref): Wainwright, High-Dimensional Statistics: A Non-Asymptotic Viewpoint, Cambridge University Press, 2019, DOI 10.1017/9781108627771. Cite only for Fano / minimax machinery (the chapter on minimax lower bounds); the specific bound would be our own derivation via Fano.
```bibtex
@book{Wainwright2019,
  author    = {Wainwright, Martin J.},
  title     = {High-Dimensional Statistics: A Non-Asymptotic Viewpoint},
  publisher = {Cambridge University Press},
  year      = {2019},
  doi       = {10.1017/9781108627771},
}
```
USE: citation for Fano's method when proving our classical lower bound; do not attribute the final bound to the book.

### 13. Goto and Ohzeki, arXiv 2601.20887 — READ (abstract, arXiv API)
Title: "Micro-mobility dispatch optimization via quantum annealing incorporating historical data". Takeru Goto, Masayuki Ohzeki; v1 2026-01-28; 15 pages, 7 figures, 1 table; no journal ref. What it does: formulates micro-mobility vehicle dispatch (allocating vacant vehicles to stations for waiting/charging) as a QUBO solved on a quantum annealer; customer arrival frequencies and destination distributions are modelled via a Bayesian approach and folded into the QUBO coefficients; simulations compare with conventional formulations (including VRP), classical solvers, and reverse annealing.
Answer: it does NOT use quantum amplitude estimation, gate-model algorithms, or Monte Carlo sampling of random demand, and does not treat stochastic routing with capacity-feasibility. Demand uncertainty enters only as historical-frequency parameters in a deterministic QUBO. It does not undercut a novelty claim about amplitude estimation for stochastic VRP (SVRPSPD), but a statement like "no prior quantum work incorporates demand statistics" would be false; phrase it as "no prior work estimates route-feasibility probabilities with amplitude estimation". Based on abstract only (full text not read).
Bib entry (arXiv record verified):
```bibtex
@misc{GotoOhzeki2026,
  author = {Goto, Takeru and Ohzeki, Masayuki},
  title  = {Micro-mobility dispatch optimization via quantum annealing incorporating historical data},
  year   = {2026},
  eprint = {2601.20887},
  archivePrefix = {arXiv},
}
```

## 14. Re-verification of OPEN keys in references.bib

- **Hamoudi2021** — VERIFIED. DataCite: "Quantum Sub-Gaussian Mean Estimator", Yassine Hamoudi, ESA 2021, LIPIcs 204, article 50, pages 50:1-50:17, DOI 10.4230/LIPIcs.ESA.2021.50 resolves at doi.org (not in Crossref, as expected). Existing entry is correct; arXiv id 2108.12172 not re-checked in this pass (not in my arXiv query), so leave as is.
- **HamoudiMagniez2019** — VERIFIED. DataCite: "Quantum Chebyshev's Inequality and Applications", Hamoudi and Magniez, ICALP 2019, LIPIcs 132, article 69, pages 69:1-69:16; IsVersionOf arXiv:1807.06456. Existing entry correct; page range resolved.
- **Doriguello2022** — VERIFIED. DOI 10.4230/LIPIcs.TQC.2022.2 resolves at doi.org and DataCite: LIPIcs 232, TQC 2022, article 2, pages 2:1-2:24, five authors Doriguello, Luongo, Bao, Rebentrost, Santha, IsVersionOf arXiv:2111.15332. Existing entry correct (Crossref 404 was only because it is a DataCite DOI).
- **chakrabarti2022bnb** — PARTLY VERIFIED. arXiv 2210.03210v1: title and four authors (Chakrabarti, Minssen, Yalovetzky, Pistoia) confirmed; 25 pages. No journal version found. The result (about sqrt(Q)*d for a classical B&B of cost Q at solution depth d) was seen via search summary of the abstract; entry stays valid as @misc. Keep the claim at abstract level.
- **sanders2020compilation** — VERIFIED, complete author list. arXiv record: PRX Quantum 1, 020312 (2020), DOI 10.1103/PRXQuantum.1.020312.
```bibtex
@article{sanders2020compilation,
  author  = {Sanders, Yuval R. and Berry, Dominic W. and Costa, Pedro C. S. and Tessler, Louis W. and Wiebe, Nathan and Gidney, Craig and Neven, Hartmut and Babbush, Ryan},
  title   = {Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization},
  journal = {PRX Quantum},
  volume  = {1},
  pages   = {020312},
  year    = {2020},
  doi     = {10.1103/PRXQuantum.1.020312},
  eprint  = {2007.07391},
  archivePrefix = {arXiv},
}
```
- **omanakuttan2025threshold** — VERIFIED as preprint (no journal version found in arXiv record; Crossref search found none). Full author list (arXiv v1):
```bibtex
@misc{omanakuttan2025threshold,
  author = {Omanakuttan, Sivaprasad and He, Zichang and Zhang, Zhiwei and Hao, Tianyi and Babakhani, Arman and Boulebnane, Sami and Chakrabarti, Shouvanik and Herman, Dylan and Sullivan, Joseph and Perlin, Michael A. and Shaydulin, Ruslan and Pistoia, Marco},
  title  = {Threshold for Fault-tolerant Quantum Advantage with the Quantum Approximate Optimization Algorithm},
  year   = {2025},
  eprint = {2504.01897},
  archivePrefix = {arXiv},
}
```
- **beverland2022requirements** — VERIFIED as preprint (no journal_ref; Crossref search found none). Full author list (arXiv v1):
```bibtex
@misc{beverland2022requirements,
  author = {Beverland, Michael E. and Murali, Prakash and Troyer, Matthias and Svore, Krysta M. and Hoefler, Torsten and Kliuchnikov, Vadym and Low, Guang Hao and Soeken, Mathias and Sundaram, Aarthi and Vaschillo, Alexander},
  title  = {Assessing requirements to scale to practical quantum advantage},
  year   = {2022},
  eprint = {2211.07629},
  archivePrefix = {arXiv},
}
```
- **acharya2024below** — VERIFIED. arXiv journal-ref: Nature 638 (2025) 920-926, DOI 10.1038/s41586-024-08449-y (resolves). Author list is the Google Quantum AI collaboration (about 190 names); use the corporate form. Year should be 2025, not 2024.
```bibtex
@article{acharya2024below,
  author  = {{Google Quantum AI and Collaborators}},
  title   = {Quantum error correction below the surface code threshold},
  journal = {Nature},
  volume  = {638},
  pages   = {920--926},
  year    = {2025},
  doi     = {10.1038/s41586-024-08449-y},
  eprint  = {2408.13687},
  archivePrefix = {arXiv},
}
```
(First listed arXiv author is Rajeev Acharya; "Google Quantum AI and Collaborators" is the Nature byline convention, from memory. If unsure, use `Acharya, Rajeev and others`.)
- **Feld2019** — VERIFIED, full eight-author list (Crossref + doi.org): Feld, Roch, Gabor, Seidel, Neukart, Galter, Mauerer, Linnhoff-Popien; Frontiers in ICT 6, article 13, 25 June 2019. Existing entry already has all eight authors and is correct.

## 15. Fault-tolerant gate-time / surface-code cycle assumptions — VERIFIED
Gidney and Ekera, Crossref: "How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits", Quantum 5, 433 (2021-04-15). (Assumptions of 1 microsecond surface-code cycle, 10 microsecond reaction time, 0.1% physical error are from memory of the paper; confirm in its text before quoting numbers.)
```bibtex
@article{Gidney2021RSA,
  author  = {Gidney, Craig and Eker{\aa}, Martin},
  title   = {How to factor 2048 bit {RSA} integers in 8 hours using 20 million noisy qubits},
  journal = {Quantum},
  volume  = {5},
  pages   = {433},
  year    = {2021},
  doi     = {10.22331/q-2021-04-15-433},
}
```
USE: standard hardware model (surface-code cycle time, reaction time, physical error rate) for the fault-tolerant crossover analysis.

## Summary
- Verified: Huggins2022, Tang2025 (preprint only), vanApeldoorn2021, Cornelissen2022 (already present), Audibert2009, Maurer2009 (arXiv/COLT, no DOI), Barenco1995, Hoeffding1963, deBoer2005, Glasserman2005, Owen2019, Wainwright2019 (book only), GotoOhzeki2026, Gidney2021RSA, plus open keys Hamoudi2021, HamoudiMagniez2019, Doriguello2022, sanders2020compilation, omanakuttan2025threshold, beverland2022requirements, acharya2024below, Feld2019, chakrabarti2022bnb (preprint, abstract level).
- Unverified: none outright; minor from-memory details flagged inline (TQC/COLT edition numbers, Hoeffding issue number, Nature byline, Gidney numeric assumptions).
- Item 12: no citable shared-sample classical lower bound found.
