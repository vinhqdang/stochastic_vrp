# Reference audit 2 (2026-11-27)

Method: Crossref REST API (`/works/<DOI>` and `query.bibliographic`), arXiv API, OpenAlex (secondary aggregator, flagged where used). DBLP and doi.org were blocked/unavailable from this environment (bot challenge / proxy), so DBLP was not checked.

## Summary

12 of 13 items VERIFIED against Crossref; 1 PARTIAL (Kakade-Langford: no DOI, pages come from OpenAlex only). Two caller-supplied DOIs were wrong or absent: item 1 (suspected DOI 10.1016/j.ejor.2022.11.040 belongs to a different paper; correct DOI is 10.1016/j.ejor.2022.10.045, and it is vol. 306 issue 3, not 2023 vol. 308) and item 7 (the Secomandi 2001 DOI is 10.1287/opre.49.5.796.10608). Item 6 DOI found: 10.1080/03081069508717555. Item 8 DOI found: 10.1023/A:1009635226865. Notes: Efron-Morris end page 130 is not in Crossref (it lists only "117"); end page 130 confirmed by OpenAlex only. Gelman's Crossref title carries the suffix "(comment on article by Browne and Draper)" (the paper's own title; the short form is what you cite). arXiv 2302.02538 v1 title/authors checked; the arXiv API record shows no journal-ref/DOI, so the DOI comes from Crossref only.

---

## 1. [florio2023bayesian] VERIFIED (DOI corrected)
- Authors (in order): Alexandre M. Florio, Michel Gendreau, Richard F. Hartl, Stefan Minner, Thibaut Vidal
- Title: Recent advances in vehicle routing with stochastic demands: Bayesian learning for correlated demands and elementary branch-price-and-cut
- European Journal of Operational Research, 2023 (issued May 2023), vol. 306, issue 3, pp. 1081-1093
- DOI: 10.1016/j.ejor.2022.10.045
- Source: Crossref query.bibliographic hit + /works/10.1016/j.ejor.2022.10.045 (publisher Elsevier); arXiv 2302.02538 title matches.
- WARNING: the suspected DOI 10.1016/j.ejor.2022.11.040 resolves in Crossref to De La Vega, Gendreau, Morabito, Munari, Ordonez, "An integer L-shaped algorithm for the vehicle routing problem with time windows and stochastic demands", EJOR 308(2):676-695. Do not use it.

```bibtex
@article{florio2023bayesian,
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

## 2. [ak2007paired] VERIFIED
- Authors: Arda Ak, Alan L. Erera (Crossref family names Ak, Erera; given names not checked beyond family)
- Title: A Paired-Vehicle Recourse Strategy for the Vehicle-Routing Problem with Stochastic Demands
- Transportation Science, 2007 (May), vol. 41, issue 2, pp. 222-237
- DOI: 10.1287/trsc.1060.0180
- Source: Crossref /works/10.1287/trsc.1060.0180

```bibtex
@article{ak2007paired,
  author  = {Ak, Arda and Erera, Alan L.},
  title   = {A Paired-Vehicle Recourse Strategy for the Vehicle-Routing Problem with Stochastic Demands},
  journal = {Transportation Science},
  year    = {2007},
  volume  = {41},
  number  = {2},
  pages   = {222--237},
  doi     = {10.1287/trsc.1060.0180}
}
```
(Given names "Arda", "Alan L." are from general knowledge; Crossref check covered family names only. Verify given names if they will be printed in full; use initials otherwise.)

## 3. [zhu2014paired] VERIFIED
- Authors: Zhu, Rousseau, Rei, Li (family names per Crossref, in order; given names not retrieved)
- Title: Paired cooperative reoptimization strategy for the vehicle routing problem with stochastic demands
- Computers & Operations Research, 2014 (Oct), vol. 50, pp. 1-13 (no issue number in Crossref)
- DOI: 10.1016/j.cor.2014.03.027
- Source: Crossref /works/10.1016/j.cor.2014.03.027

```bibtex
@article{zhu2014paired,
  author  = {Zhu, Lei and Rousseau, Louis-Martin and Rei, Walter and Li, Bo},
  title   = {Paired cooperative reoptimization strategy for the vehicle routing problem with stochastic demands},
  journal = {Computers \& Operations Research},
  year    = {2014},
  volume  = {50},
  pages   = {1--13},
  doi     = {10.1016/j.cor.2014.03.027}
}
```
(Given names Lei/Louis-Martin/Walter/Bo not verified; family names and order verified. Safe to use initials only.)

## 4. [goodson2013rollout] VERIFIED
- Authors: Justin C. Goodson, Jeffrey W. Ohlmann, Barrett W. Thomas (family names verified in order; given names from general knowledge)
- Title: Rollout Policies for Dynamic Solutions to the Multivehicle Routing Problem with Stochastic Demand and Duration Limits
- Operations Research, 2013 (Feb), vol. 61, issue 1, pp. 138-154
- DOI: 10.1287/opre.1120.1127
- Source: Crossref

```bibtex
@article{goodson2013rollout,
  author  = {Goodson, Justin C. and Ohlmann, Jeffrey W. and Thomas, Barrett W.},
  title   = {Rollout Policies for Dynamic Solutions to the Multivehicle Routing Problem with Stochastic Demand and Duration Limits},
  journal = {Operations Research},
  year    = {2013},
  volume  = {61},
  number  = {1},
  pages   = {138--154},
  doi     = {10.1287/opre.1120.1127}
}
```

## 5. [ghosal2020dro] VERIFIED
- Authors: Ghosal, Wiesemann (family names verified; given names Sarita/Soumyadip? NOT verified here, see note)
- Title: The Distributionally Robust Chance-Constrained Vehicle Routing Problem
- Operations Research, 2020 (May), vol. 68, issue 3, pp. 716-732
- DOI: 10.1287/opre.2019.1924
- Source: Crossref

NOTE: given names of Ghosal were not retrieved (Crossref output was family-only in this audit). The first author is, from memory, Shubhechyya Ghosal; Wiesemann is Wolfram Wiesemann. Treat given names as unverified; initials-only is safest.

```bibtex
@article{ghosal2020dro,
  author  = {Ghosal, S. and Wiesemann, W.},
  title   = {The Distributionally Robust Chance-Constrained Vehicle Routing Problem},
  journal = {Operations Research},
  year    = {2020},
  volume  = {68},
  number  = {3},
  pages   = {716--732},
  doi     = {10.1287/opre.2019.1924}
}
```

## 6. [popovic1995bayesian] VERIFIED
- Author: Jovan Popovic
- Title: Vehicle routing in the case of uncertain demand: a Bayesian approach
- Transportation Planning and Technology, 1995 (Jan), vol. 19, issue 1, pp. 19-29
- DOI: 10.1080/03081069508717555
- Source: Crossref query.bibliographic (exact title match, single author)

```bibtex
@article{popovic1995bayesian,
  author  = {Popovic, Jovan},
  title   = {Vehicle routing in the case of uncertain demand: a {B}ayesian approach},
  journal = {Transportation Planning and Technology},
  year    = {1995},
  volume  = {19},
  number  = {1},
  pages   = {19--29},
  doi     = {10.1080/03081069508717555}
}
```

## 7. [secomandi2001rollout] VERIFIED
- Author: Nicola Secomandi
- Title: A Rollout Policy for the Vehicle Routing Problem with Stochastic Demands
- Operations Research, 2001 (Oct), vol. 49, issue 5, pp. 796-802
- DOI: 10.1287/opre.49.5.796.10608 (note: 10.1287/opre.49.5.796.10497 returns "Resource not found" in Crossref; do not use)
- Source: Crossref query.bibliographic

```bibtex
@article{secomandi2001rollout,
  author  = {Secomandi, Nicola},
  title   = {A Rollout Policy for the Vehicle Routing Problem with Stochastic Demands},
  journal = {Operations Research},
  year    = {2001},
  volume  = {49},
  number  = {5},
  pages   = {796--802},
  doi     = {10.1287/opre.49.5.796.10608}
}
```

## 8. [bertsekas1997rollout] VERIFIED
- Authors: Dimitri P. Bertsekas, John N. Tsitsiklis, Cynara Wu
- Title: Rollout Algorithms for Combinatorial Optimization
- Journal of Heuristics, 1997 (Dec), vol. 3, issue 3, pp. 245-262
- DOI: 10.1023/A:1009635226865 (Crossref lowercases it)
- Source: Crossref query.bibliographic

```bibtex
@article{bertsekas1997rollout,
  author  = {Bertsekas, Dimitri P. and Tsitsiklis, John N. and Wu, Cynara},
  title   = {Rollout Algorithms for Combinatorial Optimization},
  journal = {Journal of Heuristics},
  year    = {1997},
  volume  = {3},
  number  = {3},
  pages   = {245--262},
  doi     = {10.1023/A:1009635226865}
}
```
(Crossref metadata does not state the "rollout improvement property" content; that claim is a statement about the paper's content, not checked here.)

## 9. [kakade2002approx] PARTIAL
- Authors: Sham Kakade, John Langford (author names seen in OpenAlex; order Kakade first)
- Title: Approximately Optimal Approximate Reinforcement Learning
- Venue: Proceedings of the 19th International Conference on Machine Learning (ICML 2002), 2002
- Pages: 267-274 (OpenAlex biblio only; NOT confirmed by a primary source)
- DOI: none found. Crossref has no record for this ICML 2002 paper (query returned unrelated items). DBLP was unreachable (bot challenge), so editor/publisher (Morgan Kaufmann, Sydney) and month are UNVERIFIED.
- Source: OpenAlex (secondary).

```bibtex
@inproceedings{kakade2002approx,
  author    = {Kakade, Sham and Langford, John},
  title     = {Approximately Optimal Approximate Reinforcement Learning},
  booktitle = {Proceedings of the 19th International Conference on Machine Learning (ICML)},
  year      = {2002},
  pages     = {267--274}, % UNVERIFIED vs primary source (OpenAlex only)
  % publisher, address: UNVERIFIED, omitted
}
```

## 10. [efron1973stein] VERIFIED
- Authors: Bradley Efron, Carl Morris
- Title: Stein's Estimation Rule and Its Competitors -- An Empirical Bayes Approach
- Journal of the American Statistical Association, 1973 (March), vol. 68, issue 341, pp. 117-130
- DOI: 10.2307/2284155
- Source: Crossref (volume, issue, authors, start page 117; Crossref's page field is just "117"); end page 130 from OpenAlex only.

```bibtex
@article{efron1973stein,
  author  = {Efron, Bradley and Morris, Carl},
  title   = {Stein's Estimation Rule and Its Competitors---An Empirical {B}ayes Approach},
  journal = {Journal of the American Statistical Association},
  year    = {1973},
  volume  = {68},
  number  = {341},
  pages   = {117--130},
  doi     = {10.2307/2284155}
}
```

## 11. [gelman2006prior] VERIFIED
- Author: Andrew Gelman (single author per Crossref)
- Title: Prior distributions for variance parameters in hierarchical models (comment on article by Browne and Draper)
- Bayesian Analysis, 2006 (Sep), vol. 1, issue 3, pp. 515-534
- DOI: 10.1214/06-BA117A
- Source: Crossref

```bibtex
@article{gelman2006prior,
  author  = {Gelman, Andrew},
  title   = {Prior distributions for variance parameters in hierarchical models (comment on article by {B}rowne and {D}raper)},
  journal = {Bayesian Analysis},
  year    = {2006},
  volume  = {1},
  number  = {3},
  pages   = {515--534},
  doi     = {10.1214/06-BA117A}
}
```

## 12. [audibert2007fast] VERIFIED
- Authors: Jean-Yves Audibert, Alexandre B. Tsybakov (family names verified; given names from general knowledge)
- Title: Fast learning rates for plug-in classifiers
- The Annals of Statistics, 2007 (Apr), vol. 35, issue 2, pp. 608-633
- DOI: 10.1214/009053606000001217
- Source: Crossref

```bibtex
@article{audibert2007fast,
  author  = {Audibert, Jean-Yves and Tsybakov, Alexandre B.},
  title   = {Fast learning rates for plug-in classifiers},
  journal = {The Annals of Statistics},
  year    = {2007},
  volume  = {35},
  number  = {2},
  pages   = {608--633},
  doi     = {10.1214/009053606000001217}
}
```

## 13. [secomandi2009reopt] VERIFIED (DOI resolves in Crossref)
- Authors: Nicola Secomandi, Francois Margot (family names verified; "Francois" given name from general knowledge)
- Title: Reoptimization Approaches for the Vehicle-Routing Problem with Stochastic Demands
- Operations Research, 2009 (Feb), vol. 57, issue 1, pp. 214-230
- DOI: 10.1287/opre.1080.0520
- Source: Crossref /works/10.1287/opre.1080.0520

```bibtex
@article{secomandi2009reopt,
  author  = {Secomandi, Nicola and Margot, Fran{\c{c}}ois},
  title   = {Reoptimization Approaches for the Vehicle-Routing Problem with Stochastic Demands},
  journal = {Operations Research},
  year    = {2009},
  volume  = {57},
  number  = {1},
  pages   = {214--230},
  doi     = {10.1287/opre.1080.0520}
}
```
