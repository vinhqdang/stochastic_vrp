# CSoNet 2026 Journal Track submission package

**Target:** Journal of Combinatorial Optimization (Springer), via
CSoNet 2026's Journal Track. Track D ("Transportation and
infrastructure networks") is the suggested conference-abstract track.

**Status:** submitted to JOCO; **major revision requested** (reviews
received 2026-10-05); the revised manuscript and the response to
reviewers are prepared in this directory — see `STATUS.md` for the
submission record and the revision log.

## Files

- `main.tex` / `main.pdf` — the manuscript (Springer `sn-jnl` class,
  `sn-mathphys-ay` style: author-year citations, as JOCO requires).
  Revised version: 38 pages, Theorems 2-5 and 9, Propositions 6-8,
  full pseudocode for all algorithms, a running example, 5 figures,
  a "Discussion and extensions" section, two appendices, a redesigned
  numerical study (Section 5) and a revised Camp Fire case study
  (Section 5.6). Builds clean: `pdflatex main && bibtex main && pdflatex
  main && pdflatex main`. For the journal's revision upload send only
  the editable sources (`main.tex`, `references.bib`, `figures/`, the
  class files), not the PDF.
- `response_to_reviewers.tex` — point-by-point response to the two
  reviewers, including a list of errors of our own that the revision
  corrected; compile with `pdflatex` twice (+ `bibtex`).
- `verify_small.py` / `verify_small_results.txt` — exhaustive-search
  verification of every algorithm and of Theorems 5-6 claims
  (`python3 verify_small.py`, ~10 s).
- `references.bib` — 26 citations, all verified against live
  publisher/DOI/arXiv records before use, spanning 1968–2026 (four
  2026 papers included).
- `experiment.py` — the algorithms (exact DP, FPTAS with optional
  completion step, equal-cost matroid greedy for m vehicles, greedy
  repair, EDD baselines) and the numerical study: E1 accuracy vs n
  (weights <= 100; the FPTAS scaling is vacuous there, and the share is
  recorded), E2 FPTAS with active scaling (large weights, strongly
  correlated, adversarial family), E3 runtime, E4 robustness.
  Regenerate every table with `python3 experiment.py` (~15 s; needs
  numpy).
- `results_illustration.json` — the synthetic experiments' raw output.
- `case_study_campfire.py` / `case_study_campfire_results.txt` — a
  real-world MWHED instance built from public records of the 2018
  Camp Fire (revised: arrival-reading deadlines, preprocessing, a
  Monte-Carlo sensitivity analysis) (NIST fire-progression timeline, real community
  coordinates and 2010 Census populations); every disclosed modeling
  assumption (response speed, deadline extrapolation for two sites) is
  documented in the script's docstring and in main.tex
  Section~5.5. Regenerate with `python3 case_study_campfire.py`.
- `figures/` — the source for all 5 figures: `make_running_example_figure.py`
  (Fig. 1, the running-example schematic), `make_result_figures.py`
  (Figs. 2–4, plotting `results_illustration.json`'s already-verified
  numbers -- computes nothing new), and `make_campfire_map.py` (Fig. 5,
  a real-geography map plotted from the same coordinates used in
  `case_study_campfire.py`). Regenerate all of them from within
  `figures/` with `python3 make_running_example_figure.py && python3
  make_result_figures.py && python3 make_campfire_map.py`.
- `cover_letter.md` — cover letter for the JOCO submission; explicitly
  states "This manuscript is submitted to the journal track of CSoNet
  2026" per the Journal Track's requirement.
- `conference_abstract_submission.md` — the abstract text and metadata
  to paste into CSoNet's own conference submission system (required
  separately from the journal submission, to secure a presentation
  slot).
- `STATUS.md` — review-status tracker, same convention as
  `papers/baton/STATUS.md`.

## What the paper is (and is not)

A genuinely new combinatorial-scheduling contribution — Minimum
Weighted Hazard-Exposure Dispatch (MWHED) — motivated by dispatch
decisions under a spreading hazard (wildfire, flood, contamination,
congestion), but **not a repackaging of BATON or TEMPO**. It shares no
code, no evaluation instances, and no results with either paper; the
theory (NP-hardness via a knapsack equivalence, an exact
pseudo-polynomial DP, an FPTAS, and a matroid-greedy tractable special
case) is new and self-contained. This separation was deliberate: JOCO
and CSoNet's own submission guidelines explicitly prohibit
simultaneous submission and salami-slicing of one study into multiple
papers, and BATON is under review at *Computers & OR* while TEMPO is
under review at *Transportation Science* — reusing either paper's
results here would jeopardize both.

## Before submitting

- [ ] Confirm affiliation/ORCID details are current.
- [ ] Decide on a Data Availability Statement wording if reviewers
  request the experiment code hosted externally (currently: "included
  with this submission").
- [ ] Read the compiled PDF once, end to end, before uploading — JOCO's
  own guidance: "the Meteor submission system does not support
  post-submission edits."
- [ ] Submit to editorialmanager.com/joco, then submit the abstract to
  meteor.springer.com/CSoNet2026 (both required).
