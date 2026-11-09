# SORTIE — citations to verify before submission

Every entry in `lit/references.bib` has bibliographic data (authors, title,
venue, year, DOI or stable URL) checked against Crossref, the publisher page
or the arXiv/Optimization Online record. The items below need a further check.

## Content claims resting on abstracts or search snippets only

The bibliographic record is verified; the sentence the manuscript attributes
to the paper was checked only against its abstract or a search snippet. Read
the full text before submission and confirm the attribution in `sortie.tex`.

- Gutjahr (2021)
- Mostajabdaveh et al. (2019)
- Alem et al. (2022)
- Wu et al. (2025)
- Yu et al. (2024)
- Hosseini and Schierreich (2025)
- Basik et al. (2021)
- Bruinink et al. (2025)
- Sobhanan et al. (2026)

## Framing claims

- **Eisner and Severance (1976).** Bibliographic data verified; in
  `references.bib` but not currently cited in `sortie.tex`. If it is cited for
  parametric (weighted-sum) combinatorial optimisation, confirm that framing
  against the full text first.
- **Ehrgott (2005), Multicriteria Optimization.** The author attribution rests
  on the ISBN and title; the DOI metadata carries no author field.
- **Ljubić, Puerto and Torrejón (2026).** Cited as the Optimization Online
  preprint submitted 5 March 2026. One search summary mentioned an update on
  12 October 2026 that I could not confirm. Check for a journal version before
  submission.

## Not cited (could not verify; kept out of `references.bib`)

- Yitzhaki (1982), American Economic Review — no DOI or publisher page found.
- Chen and Liu (1997), Statistica Sinica 7:875–892 — not found by Crossref.
  The Poisson-binomial recursion in the paper is standard and is cited through
  verified sources instead.
- Hande (1994), proof of the Bapat–Beg theorem — venue not confirmed.
- Rhea and Ohlmann, equitable mobile facility routing — SSRN working papers
  only (10.2139/ssrn.4961208, 10.2139/ssrn.5576249). Cite if a journal
  version appears.
- Lesmana, Zhang and Bei (2019), NeurIPS — not found by Crossref.
- Yager and Kacprzyk (1997), OWA operators book — DOI exists but no role in
  the paper.
