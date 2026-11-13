# RARECERT — PEIS submission package

Target: *Probability in the Engineering and Informational Sciences* (Cambridge
University Press). Initial submission is one PDF (no template required).
History: rejected without review at INFORMS J. Computing on 2026-11-13; the editor
suggested PEIS (see `../reviews/ijoc_decision_2026-11-13.md`).

| file | content |
|---|---|
| `rarecert_peis.tex` / `.pdf` | manuscript (17 pages) |
| `cover_letter.tex` / `.pdf` | cover letter (mentions the IJOC history) |
| `references.bib` | bibliography (apalike) |
| `make_tables.py` | writes `tab_conditions.tex`, `tab_bias.tex`, `bias_sentence.tex` from the result CSVs — never edit those three by hand |
| `figures/` | copied from `../manuscript/figures/` (made by `svrpspd_wdro/scripts/make_rarecert_figures*.py`) |

## Where every number comes from

- Table 1 (theorem conditions on 5 920 routes):
  `svrpspd_wdro/scripts/run_rarecert_peis_conditions.py` →
  `results/rarecert/peis/conditions.csv`; covering constant in
  `svrpspd_wdro/rarecert/covering.py` (tests: `tests/test_rarecert_covering.py`).
- Table 3 (silent underestimation): `svrpspd_wdro/scripts/run_rarecert_peis_bias.py` →
  `results/rarecert/peis/bias_check.csv` (single-block end probabilities by SMIS,
  measured errors from `results/rarecert/baselines_routes.csv`).
- Tables 2 and 4 and the remaining figures of Section 7: the IJOC-era runs
  (`results/rarecert/smis_routes.csv`, `baselines_routes.csv`, `r3_*`); Table 2 merges the
  two lower ϱ bands of the IJOC table, recomputed from `smis_routes.csv`.

## Build

```
python make_tables.py
pdflatex rarecert_peis && bibtex rarecert_peis && pdflatex rarecert_peis && pdflatex rarecert_peis
pdflatex cover_letter
```
