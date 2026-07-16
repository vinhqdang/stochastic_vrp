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
  Builds clean: `pdflatex main && bibtex main && pdflatex main &&
  pdflatex main`.
- `references.bib` — all citations verified against live
  publisher/DOI records before use (see the verification notes in the
  session that produced this).
- `experiment.py` — self-contained numerical illustration (the exact
  DP, the FPTAS, and a naive baseline on random instances); regenerate
  Table 1's numbers with `python3 experiment.py`.
- `results_illustration.json` — the experiment's raw output.
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
