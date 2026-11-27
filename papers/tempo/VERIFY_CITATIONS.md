# Citations to verify before submission

Per `CLAUDE.md`'s citation policy, entries below are cited in
`main.tex`/`references.bib` from memory or from a secondary source and
should be checked against the primary source before submission.

- **Gmira, Glize, Hoenig & Rousseau (2021), rolling-horizon dynamic
  routing heuristic** (`gmira2021rollingopt`). We only have the
  author/year attribution as it appears in Chen, Imdahl, Lai & Van
  Woensel (2025)'s own reference list (the DGTA-RL paper's rolling-opt
  baseline). The title, venue, and DOI in `references.bib` are
  placeholders (title paraphrased from the codebase docstring) — locate
  the primary source and replace before submission, or drop the direct
  citation and cite only via Chen et al. (2025) if it cannot be found.
- **Ramdas, Grünwald, Vovk & Shafer (2023), "Game-Theoretic Statistics
  and Safe Anytime-Valid Inference," Statistical Science 38(4):576–601,
  DOI 10.1214/23-STS894** (`ramdas2023safi`). Volume/issue/page range
  and DOI recalled from training knowledge, not verified against the
  journal record — confirm against the published version.
- **Waudby-Smith & Ramdas (2024), "Estimating means of bounded random
  variables by betting," Journal of the Royal Statistical Society
  Series B 86(1):1–27, DOI 10.1093/jrsssb/qkad009**
  (`waudbysmithramdas2024`). Same caveat: volume/page range recalled
  from training knowledge, not verified against the journal record.

All other citations in `references.bib` (Ville 1939, Page 1954, Gama et
al. 2014, Adams & MacKay 2007, Shafer 2021, Vovk & Wang 2021, Shin,
Ramdas & Rinaldo 2024, He et al. 2026, Chen et al. 2025, Merchán et al.
2024, Dethloff 2001, Salhi & Nagy 1999) carry DOIs already used
elsewhere in this repository or independently cross-checked during
this session and are not flagged here.

## TEMPO (direction 1) bibliography, `tempo.bib` (November 2026)

Built only from entries verified in `reviews/reference_audit.md` and
`reviews/reference_audit2.md`. Open points:
- kakade2002approx: no DOI or Crossref record; pages 267-274 from
  OpenAlex only; publisher/address omitted. Check against the ICML 2002
  proceedings before submission.
- efron1973stein: end page 130 from OpenAlex only (Crossref gives the
  start page).
- Given names for ak2007paired, zhu2014paired, goodson2013rollout,
  ghosal2020dro, audibert2007fast, secomandi2009reopt partly from memory
  (Crossref returned family names only); confirm on the publisher pages.
- dang2026baton: accepted at Computers & OR; add volume/DOI when issued.
The TRSC bibliography (`references.bib`) had one non-existent reference
and wrong author names in two others; it must not be reused.
