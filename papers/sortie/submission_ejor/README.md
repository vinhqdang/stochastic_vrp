# SORTIE — submission package for the European Journal of Operational Research

Prepared 10 November 2026. Article type: **Theory and Methodology**, area
*Production, Manufacturing, Transportation and Logistics*. First keyword /
"Section/Category" in the submission system: **Routing** (editor code O).

## Files to upload (`upload/`)

| File | Submission item |
|---|---|
| `manuscript.pdf` | Manuscript (31 pages, ≤ 35 allowed) |
| `manuscript_source.zip` | LaTeX source: `sortie.tex`, `sortie.bbl`, `references.bib`, figures, tables |
| `Figure_1.pdf`, `Figure_2.pdf`, `Figure_3.pdf` | Figures as separate vector files, in order of appearance |
| `online_supplement.pdf` | Supplementary material (27 pages) |
| `highlights.docx` | Highlights (5 bullets, each ≤ 85 characters) |
| `graphical_abstract.png` / `.pdf` | Graphical abstract (1975 px wide) |
| `cover_letter.pdf` | Cover letter (one page) |

## Sources in this folder

| File | Built by |
|---|---|
| `sortie.tex` | `pdflatex sortie && bibtex sortie && pdflatex sortie && pdflatex sortie` |
| `sortie_supplement.tex` | Same commands |
| `cover_letter.tex` | `pdflatex cover_letter` |
| `highlights.txt`, `highlights.docx` | Written directly |
| `graphical_abstract.*` | `cd ../code && python experiments/graphical_abstract.py` |
| `references.bib` | Copy of `../lit/references.bib` with accents escaped and internal notes removed |
| `figures/`, `tables/` | Copies of `../figures/` and `../results/tables/paper/` |

The submission manuscript is a condensed version of `../sortie.tex`.
- **What moved to the Online Supplement:** experiments E4–E8, E10, E11 and E13–E15, Algorithm S1 and the parameter table. These are cited in the main text by their labels E1–E17.
- **What was not changed:** the theory and the algorithm.

## Format compliance (EJOR checklist)

- [x] Single column, 11pt, 1.5 line spacing, no line numbers
- [x] ≤ 35 pages including tables, figures, references (31)
- [x] Figures and tables embedded next to the text, and also uploaded separately
- [x] APA citations (apacite), alphabetical reference list, DOIs as https://doi.org/
- [x] Title page: title, all authors, affiliations, corresponding author with e-mail
- [x] No formulae or acronyms in the title
- [x] Abstract ≤ 250 words (250), no formulae or unqualified abbreviations
- [x] Keywords: Routing (EJOR list, first); Fairness; Stochastic travel times; Order statistics; Bilevel optimization
- [x] Abbreviations defined at first use (TSP, X-CVRP-TSP, SAA, GMD, MAD, PBOI, CVaR, SORTIE)
- [x] Proofs in the main text
- [x] Declarations sections before the references: competing interests, funding, CRediT, data availability

## To confirm before submitting

1. **Affiliations and postal addresses.** EJOR asks for the full postal
   address of each affiliation, including the country. The title page now
   gives only city and country:
   - British University Vietnam, Hung Yen
   - University of Economics Ho Chi Minh City
   Add the street addresses if wanted.
2. **Third author's name.** It is given as *Phuc-Son Nguyen*. The author
   list also used *Son P. Nguyen*. Use the same form as in the submission
   system.
3. **CRediT roles.** The statement in the manuscript is a draft. Every
   author should confirm their roles.
4. **Funding.** The manuscript states that there was no specific funding.
   Change this if there was a grant.
5. **Data availability.** The manuscript says the material will be released
   in a repository upon acceptance. Name the repository (Zenodo/GitHub) if
   one is to be cited at submission.
6. **Declaration of generative AI use.** EJOR requires a statement in a
   section before the references whenever AI tools were used in preparing
   the manuscript. Decide whether this applies and add it if so; nothing has
   been added.
7. **Competing interests.** Complete the Elsevier declarations tool and
   upload the generated .docx.
8. **Co-authors.** Enter ORCID iDs for the co-authors if available.
9. **Preprint.** None has been posted. Posting one is allowed.
10. **Suggested or opposed reviewers.** These are requested separately in
    the system, not in the cover letter.
