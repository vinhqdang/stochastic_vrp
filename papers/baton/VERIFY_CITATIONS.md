# Citation verification status

**Current state (2026-09-26, revision 1).** `references.bib` holds 48
entries, 43 of them cited in `main.tex` (the other five —
`min1989multiple`, `ropke2006adaptive`, `ai2009particle`,
`subramanian2010parallel`, `hoogendoorn2025evaluation` — are not cited
and do not appear in the reference list); 43 entries carry a DOI. Every DOI was checked
against its registry record (Crossref, or DataCite for the two arXiv
DOIs) and the registered title compared with the BibTeX title:

- 41 DOIs: registered title matches the BibTeX title.
- `puterman1994markov`: matches; Crossref stores the subtitle separately.
- `legault2025superadditivity` (arXiv:2508.05877) and `ota2026scenario`
  (arXiv:2604.02496): DataCite title and authors match; preprints, check
  journal status at submission.
- No DOI (books and one chapter): `barlow1972statistical`,
  `chow1971great`, `christofides1979vehicle`, `muller2002comparison`;
  `iklassov2024rl` cites the PMLR proceedings page instead of the arXiv
  DOI.

Corrections made in this pass (after an external reference audit):

- `novoa2009approximate`: DOI was `10.1016/j.ejor.2008.02.038`, which
  resolves to a different paper (Castillo, Joro and Li, *Workforce
  scheduling with multiple objectives*). Corrected to
  `10.1016/j.ejor.2008.03.023` (Novoa and Storer, EJOR 196(2):509-515).
  Resolving is not the same as being the right paper; titles are now
  compared for every DOI.
- `hu2025vehicle`: final version is Annals of OR 358:1285-1327 (2026);
  volume, pages and year updated (online 2025).
- Duplicates removed: `salhi1999cluster2` (same work as
  `salhi1999cluster`) and `iklassov2024reinforcement` (same work as
  `iklassov2024rl`); neither duplicate was cited.
- `christofides1979vehicle`: CVRPLIB URL removed (it is the data set's
  page, not a record of the chapter).

The rest of this file is the earlier log, kept for the record; its
counts (for example "20 DOIs") refer to the bibliography of July 2026.

## Earlier log (2026-07-05)

Every entry in `references.bib` was checked automatically on 2026-07-05:
each DOI was resolved through `https://doi.org/<doi>`. **None returned
404** — all 20 DOIs exist and redirect to a publisher landing page.

HTTP codes: `200` = landing page fetched; `403`/`202` = the DOI resolved
and redirected correctly but the publisher blocks automated clients
(INFORMS, OUP, Palgrave, IEEE), so the landing-page *metadata* could not
be machine-compared against the BibTeX fields.

## Please eyeball these (DOI valid, metadata not machine-checked)

| Key | DOI | What to confirm |
|---|---|---|
| longstaff2001valuing | 10.1093/rfs/14.1.113 | RFS 14(1):113–147, 2001 |
| ropke2006adaptive | 10.1287/trsc.1050.0135 | Transp. Sci. 40(4):455–472, 2006 |
| bertsimas1992vehicle | 10.1287/opre.40.3.574 | Oper. Res. 40(3):574–585, 1992 |
| dror1989vehicle | 10.1287/trsc.23.3.166 | Transp. Sci. 23(3):166–176, 1989 |
| secomandi2001rollout | 10.1287/opre.49.5.796.10608 | Oper. Res. 49(5):796–802, 2001 |
| ulmer2019offline | 10.1287/trsc.2017.0767 | Transp. Sci. 53(1):185–202, 2019; author list Ulmer/Goodson/Mattfeld/Hennig |
| salhi1999cluster | 10.1057/palgrave.jors.2600808 | JORS 50(10):1034–1042, 1999 |
| tsitsiklis2001regression | 10.1109/72.935083 | IEEE TNN 12(4):694–703, 2001 |

## Books without DOIs (stable URLs supplied instead)

| Key | Note |
|---|---|
| barlow1972statistical | Barlow–Bartholomew–Bremner–Brunk 1972, Wiley. HathiTrust record URL in the entry. |
| chow1971great | Chow–Robbins–Siegmund 1971, Houghton Mifflin. HathiTrust record URL in the entry. |

## Metadata machine-verified OK (200 + known venues)

clement2002analysis, dethloff2001vehicle, min1989multiple,
gendreau1996stochastic, novoa2009approximate, esfahani2018data,
ai2009particle, subramanian2010parallel, peskir2006optimal,
oyola2018stochastic, robbins1955empirical (the Ayer et al. PAVA paper),
koc2020review.

All author lists in the .bib are complete (no "et al.").

- [ ] `yang2000stochastic` — Yang, Mathur, Ballou (2000), "Stochastic Vehicle
  Routing Problem with Restocking", Transportation Science 34(1):99-112,
  DOI 10.1287/trsc.34.1.99.12278. Verify volume/issue/pages and DOI resolve.

- [ ] `salhi1999cluster2` — Salhi & Nagy (1999), JORS 50(10):1034-1042,
  DOI 10.1057/palgrave.jors.2600808. NOTE: check this is the correct paper
  for the X/Y VRPSPD instance construction (some papers credit the split
  rule to Nagy & Salhi 2005, EJOR 162(1):126-141, DOI
  10.1016/j.ejor.2003.08.041 — verify which to cite for the benchmark,
  and dedupe against the existing `salhi1999cluster` entry).
- [ ] `christofides1979vehicle` — book chapter, no DOI; CVRPLIB given as
  stable URL. Verify editor/page details.
- [ ] `montane2006tabu` — DOI 10.1016/j.cor.2004.07.009, verify pages.
- [ ] `boeing2017osmnx` — DOI 10.1016/j.compenvurbsys.2017.05.004 (city
  instance generation uses OSMnx; cite in the data section).

## SOTA comparison sweep (agent-verified via Crossref/arXiv, re-verify before submission)
- [ ] `ghosal2024unifying` — OR 72(2):425-443, DOI 10.1287/opre.2021.0669 (agent Crossref-verified)
- [ ] `gounaris2013robust` — OR 61(3):677-693, DOI 10.1287/opre.1120.1136 — verify pages
- [ ] `bertsimas2004price` — OR 52(1):35-53, DOI 10.1287/opre.1030.0065
- [ ] `legault2025superadditivity` — arXiv:2508.05877, preprint (check journal status before submission)
- [ ] `iklassov2024reinforcement` — PMLR v222:502-517; DOI given is the arXiv one (PMLR has no DOI)
- [ ] `hoogendoorn2025evaluation` — EJOR 321(1):107-122, DOI 10.1016/j.ejor.2024.09.007 (CC BY)
- [ ] `hu2025vehicle` — AOR online Dec 2025, DOI 10.1007/s10479-025-06877-1 (paywalled; volume/pages TBD)
- [ ] IMPORTANT: repo's older "Cui et al. (2025)" baseline (algorithms/cui baseline.py) cites an
  inventory paper (Data Science and Management) — the method used is Bertsimas-Sim (2004) budget
  uncertainty. In the manuscript, attribute the BSIM gate to Bertsimas & Sim 2004, NOT Cui.
- [ ] `salavati2019rule` — TS 53(5):1334-1353, DOI 10.1287/trsc.2018.0876 (open CIRRELT-2017-36 PDF exists)
- [ ] `florio2020new` — TS 54(4):1073-1090, DOI 10.1287/trsc.2020.0976 (arXiv:1806.08549; code github.com/amflorio/vrpsd-optimal-restocking)
- [ ] `secomandi2009reoptimization` — OR 57(1):214-230, DOI 10.1287/opre.1080.0520
- [ ] `pessoa2021branch` — OR 69(3), DOI 10.1287/opre.2020.2035 — VERIFY pages 739-754 (agent did not confirm)
- [ ] `gounaris2016adaptive` — TS 50(4):1239-1260, DOI 10.1287/trsc.2014.0559
- [ ] `ota2026scenario` — arXiv:2604.02496 (2026 preprint); check journal status at submission
- [ ] `wang2026drone` — Transportation Letters 18(2):270-288; agent reported online 2025 / print 2026 — verify the volume year

## Revision 1, round 2 (Crossref-verified 2026-09-26)
- [x] `egloff2005monte` — AAP 15(2):1396-1432, DOI 10.1214/105051605000000043 (Crossref title/author/volume confirmed; pages from the journal site)
- [x] `zanger2013quantitative` — Finance Stoch 17(3):503-534, DOI 10.1007/s00780-013-0204-9
- [x] `serfozo1976monotone` — Math. Programming Studies 6:202-215, DOI 10.1007/BFb0120752
- [x] `puterman1994markov` — Wiley, DOI 10.1002/9780470316887
- [ ] `muller2002comparison` — Wiley 2002, ISBN 978-0-471-49446-1; no DOI located, verify publisher city
- [x] `minis2011stochastic` — EJOR 213(1):37-51, DOI 10.1016/j.ejor.2011.03.011
- [x] `tatarakis2009stochastic` — EJOR 197(2):557-571, DOI 10.1016/j.ejor.2008.07.006
