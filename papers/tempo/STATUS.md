# TEMPO paper — status

- **Title:** When to Replan: A Decision-Relevant E-Process for
  Anytime-Valid Re-Optimization in Stochastic Vehicle Routing
- **Venue:** Transportation Science (INFORMS), TS-2026-0425 —
  **REJECTED (November 2026)**: AE and two referees reject, one major
  revision. Reports in `reviews/` (`trsc_decision.md`,
  `trsc_referee3_report.pdf`). Analysis and redesign plan:
  `REVISION_PLAN.md`. Next target: a journal of equivalent standing
  (recommended Transportation Research Part B), after the redesign.
- **Policy:** OPEN for the redesign. The TRSC manuscript (`main.tex`)
  is kept unchanged as the record of that submission; the new paper is
  written in `tempo.tex`. Every reference must be verified before
  submission (one non-existent reference reached the TRSC version).
- Submission set: `main.pdf` (35-page main body + Appendix, INFORMS
  `informs4` template), title/abstract/area/keywords/running
  head/cover letter text (prepared in-session, not stored as separate
  files — see conversation history if needed again).
- Author: Quang-Vinh Dang (British University Vietnam) — kept
  anonymous in the manuscript itself pending the double-blind review
  decision.
- Relationship to the other two papers in this repo: `papers/baton/`
  is the frozen, separately-under-review sibling (planning layer,
  this paper is the execution/monitoring layer); `papers/csonet2026/`
  is an independent, unrelated combinatorial-scheduling paper — no
  shared results, code, or evaluation instances with either.

## 27 November 2026 — new manuscript drafted

- `tempo.tex` (fleet-pooled learning for recourse, "direction 1") is
  complete: theory (Lemma 1, Theorem 1, Proposition 2), computational
  study on 73 instances x 8 scenarios x 150 days, conclusion.
- Results: `svrpspd_wdro/results/tempo_pool/` (CSVs, `tables.md`);
  LaTeX tables regenerate with `python scripts/tempo_pool_latex.py`
  from `svrpspd_wdro/`. Never hand-edit `tables/*.tex`.
- An internal referee pass was addressed: the theory is restricted to
  the in-model setting with uniform constants, the baselines' oracle
  variances are disclosed, a wrong-zone scenario was added, and
  overclaims were removed.
- Bibliography author names were checked against OpenAlex. The only
  open item is the BATON volume/DOI.
- Journal: not chosen yet; the draft carries `\journal{Computers & OR}`.
