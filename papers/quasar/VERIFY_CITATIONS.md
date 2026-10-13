# QUASAR — citations still to verify

Nothing listed here may be cited until it is checked against the source.

## Records with open issues (in references.bib)
- `Hamoudi2021` (quantum sub-Gaussian mean estimator, ESA 2021): confirmed
  by one review, not located by another; re-check the DataCite DOI.
- `HamoudiMagniez2019` (ICALP 2019): page range uncertain ("69:1–69:16").
- `Feld2019`: author list beyond the sixth name to be completed from the
  publisher page.
- `Doriguello2022`: the DOI given on arXiv (10.4230/LIPIcs.TQC.2022.2) did not
  resolve at Crossref (DataCite DOI); confirm on the Dagstuhl page.
- `chakrabarti2022bnb` (arXiv 2210.03210): result statement seen only in a
  summary; confirm authors and claim.
- Entries whose author lists end in "and others": Borowski2020,
  sanders2020compilation, acharya2024below, beverland2022requirements,
  omanakuttan2025threshold, koch2026qoblib, abbas2024challenges,
  erdmann2026strategic.
- Crossref author strings incomplete: Mahroo2026benders, RibesMarza2026jcc,
  Zheng2026gas, AmaniBani2026ride, Towada2026kernel.

## Not in references.bib (snippet or abstract-page only)
- Muralitharan, Squillante, Wang, Wang, Wu, "Quantum speedup of multi-armed
  bandits at scale by tackling memory decoherence", NeurIPS 2026 poster.
- Goto & Ohzeki, arXiv 2601.20887 (micro-mobility dispatch with historical
  data; may encode demand uncertainty) — read before claiming no prior
  quantum work on random demand.
- Ribes Marzá & González Grandón, arXiv 2512.03925 (preprint of the J. Math.
  Industry paper).
- Palanivel et al., QIC 2026 (QCVaR); Eder et al. 2026 (dial-a-ride with
  annealing); Fedouaki et al., Logistics 2026; Weinberg 2026 and Ghosh 2026
  (SSRN).
- A quantum local-search upper bound O(N^{1/3} δ^{1/6}) seen in a snippet.
- arXiv 2309.17299 and 2602.09847 (snippets only).

## Must be read in full before the positioning is final
WangFilipovska2026, HuHuZhou2026, HuHu2026, Gacon2020, An2021mlmc,
Moosavi2026RLQALNS, RibesMarza2026jcc.

## M4 (search and lower bounds) — checked 2026-10-13
Verified against the arXiv full texts; added to `references.bib`:
boyer1998tight (Thm 3, exact oracle), hoyer2003bounded (model: total
Boolean f_i with two-sided error; promise inputs not covered),
ambainis2006variable (now the TOCS 2010 version; Thm 3 for ideal
subroutines), NayakWu1998Median (Thm 1.11), Kothari2014OracleID,
DurrHeiligmanHoyerMhalla2006, ReichardtReflections2011,
LeeMittalReichardtSpalekSzegedy2011, Wang2021bandits (Thm 5 only for biases
in [p, 1−p], no variance parameter), MannorTsitsiklis2004,
KanoHondaSakamakiMatsuuraNakamuraSugiyama2019 (KL-type bound),
LocatelliGutzeitCarpentier2016 (fixed budget, unit-variance Gaussian).
Open points:
- Find-first in O(√p) expected queries: stated in Kothari2014OracleID §2.2
  as a remark, not a lemma; the attribution to DHHM06 was not found in the
  DHHM arXiv text. Cite as a remark only.
- NayakWu1998Median: the STOC 1999 venue is not verified; cite the arXiv
  version. The specialisation Δ = εn, t = pn ⇒ Ω(1/√ε + √(p(1−p))/ε) is our
  derivation.
- No composition theorem found for OR_k with a promise (partial) inner
  function; Reichardt and LMRSS state composition for total functions only.
  Proposition 3(c) stays a CONJECTURE.
- The variance-sensitive classical bound Ω(k α(1−α)/ε²) is not stated in
  this form in the bandit papers above; it is now proved directly in
  THEORY_M4 Prop. 3(a) from the transportation lemma.
- Added from memory, metadata to check against the publisher pages:
  KaufmannCappeGarivier2016 (JMLR 17, Lemma 1 = transportation lemma) and
  Bennett1997BBBV (SIAM J. Comput. 26(5)). Prop. 3(b) no longer relies on
  NayakWu1998Median (membership oracle); it uses a direct hybrid argument.
- ReichardtReflections2011: SODA venue from the arXiv record, DOI unchecked.
  LocatelliGutzeitCarpentier2016: pages unchecked.
