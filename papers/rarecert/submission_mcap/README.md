# RARECERT — MCAP submission package

Target: *Methodology and Computing in Applied Probability* (MCAP, Springer;
subscription/hybrid — publishing is free when the open-access option is not chosen).
History: rejected without review at INFORMS J. Computing on 2026-11-13; the editor
suggested a probability outlet such as PEIS, but PEIS has been gold open access since
2024 (article processing charge), so the author chose MCAP (2026-11-13). See
`../reviews/ijoc_decision_2026-11-13.md`. The scripts and result folder keep the
`peis` prefix (`run_rarecert_peis_*.py`, `results/rarecert/peis/`) from the first
retargeting; the name is historical.

| file | content |
|---|---|
| `rarecert_mcap.tex` / `.pdf` | manuscript, Springer Nature template (19 pages) |
| `cover_letter.tex` / `.pdf` | cover letter (mentions the IJOC history) |
| `references.bib` | bibliography |
| `sn-jnl.cls`, `sn-mathphys-ay.bst` | Springer Nature LaTeX template (author-year), from the author's template zip |
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
pdflatex rarecert_mcap && bibtex rarecert_mcap && pdflatex rarecert_mcap && pdflatex rarecert_mcap
pdflatex cover_letter
```

Note: the figures are matplotlib PDFs with Type 3 fonts; if Springer production asks
for Type 1/TrueType fonts, regenerate them with `matplotlib.rcParams['pdf.fonttype'] = 42`.
