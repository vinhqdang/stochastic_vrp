# CSoNet 2026 Journal Track submission package

**Target:** Journal of Combinatorial Optimization (Springer), via
CSoNet 2026's Journal Track. Track D ("Transportation and
infrastructure networks") is the suggested conference-abstract track.

**Status:** prepared, NOT YET SUBMITTED. Submission is a manual step
the author takes at https://www.editorialmanager.com/joco (journal)
and https://meteor.springer.com/CSoNet2026 (conference abstract, per
the Journal Track's dual-submission rule) — see `STATUS.md`.

## Files

- `main.tex` / `main.pdf` — the manuscript (Springer `sn-jnl` class,
  `sn-mathphys-ay` style: author-year citations, as JOCO requires).
  30 pages: 4 theorems + 1 proposition, full pseudocode for all four
  algorithms, a running numerical example threaded through every
  result, a "Discussion and extensions" section, two appendices
  (exhaustive hand-verification of the running example; a fully
  spelled-out FPTAS approximation-ratio derivation), and a real-world
  case study (Section 5.5). Builds clean: `pdflatex main && bibtex main
  && pdflatex main && pdflatex main`.
- `references.bib` — 26 citations, all verified against live
  publisher/DOI/arXiv records before use, spanning 1968–2026 (four
  2026 papers included).
- `experiment.py` — self-contained numerical experiments on synthetic
  instances: the exact DP, the FPTAS (implemented as the value-indexed
  dual DP from the proof, not a shortcut), a weighted
  Moore–Hodgson-style greedy repair heuristic, and the naive EDD
  baseline, across four experiments (accuracy vs. instance size,
  runtime scaling, epsilon sensitivity, robustness to alternative
  instance-generation regimes). Regenerate every synthetic table with
  `python3 experiment.py`.
- `results_illustration.json` — the synthetic experiments' raw output.
- `case_study_campfire.py` / `case_study_campfire_results.txt` — a
  real-world MWHED instance built from public records of the 2018
  Camp Fire (NIST fire-progression timeline, real community
  coordinates and 2010 Census populations); every disclosed modeling
  assumption (response speed, deadline extrapolation for two sites) is
  documented in the script's docstring and in main.tex
  Section~5.5. Regenerate with `python3 case_study_campfire.py`.
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
