# SORTIE — status

- **Title (working):** SORTIE: Stochastic Ordered Route fairness with
  Tour-consistency via Integral Evaluation — fair vehicle routing when
  route durations are random.
- **State:** SUBMITTED to the European Journal of Operational Research
  (Theory and Methodology, Routing) on 10 November 2026; **UNDER REVIEW**.
  EJOR manuscript number: _to be added when the confirmation e-mail arrives_.
- **Submitted version:** `submission_ejor/upload/` (31-page manuscript,
  27-page Online Supplement, cover letter, highlights, graphical abstract,
  separate figures); git tag `sortie-ejor-v1`.
- **Authors:** Quang-Vinh Dang (corresponding), Hoang-Viet Vu, Phuc-Son Nguyen.
- **Freeze policy:** do not modify `submission_ejor/`, `sortie.tex`, `code/`,
  `results/` or `figures/` while the paper is under review, except for
  corrections to the editor's or reviewers' requests. Revisions only when the
  decision arrives; new work goes in a new directory.
- **Known caveat in the submitted text:** the last abstract sentence says the
  method "outperforms sample average approximation". This holds with up to
  1000 scenarios; with 5000 scenarios the gain is not significant at 5%
  (p = 0.0505, E15). Fix it in the revision if the reviewers ask.
- **Declarations in the submission:** no funding; competing interests: none;
  generative-AI use declared (Claude, to support writing); data and code to be
  released upon acceptance (repository to be named).
- **Venues considered:** Transportation Science and EJOR (chosen). Computers &
  Operations Research was avoided because it overlaps BATON's venue.
- **Separation:** self-contained. Shares no code, instances, results or
  scenario generators with BATON, TEMPO, QUASAR or RARECERT (all under
  review). Instances come from the public CVRPLIB X-n641-k35 file using
  the van Rossum et al. / Ljubić et al. slicing rule, which none of the
  other papers uses.
