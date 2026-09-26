# Disposition of the internal review (round 1)

How each item of `internal_review_round1.md` was handled in the revised
manuscript, letter and code. Numbers live in `../tables/macros.tex`.

## Must fix

- **M1 (Proposition 2 overclaim).** Abstract, introduction, conclusion and
  §4.4 now attribute over-triggering to the myopic break-even rule only;
  the tuned threshold is stated to be outside the bound, and its gap to
  BATON-ho is no longer called "the same pattern".
- **M2 (headline range).** Dethloff-only range (86–96%) labelled as such;
  full range over benchmarks and laws with non-negligible reactive cost
  (`\ratioAllLo`–`\ratioAllHi`) and the exact-DP shares (Det, SAA) added.
  "Six planning regimes" attached only to the Dethloff results.
- **M3 (ties).** Paired bootstrap intervals for the key contrast in
  Tables 5 (large), 7 (dependence), 10 (pool) and in the day-type text;
  every tie and the one loss (ρ = 0.9, Det, vs DP³_N) reported with its
  interval in the text, abstract, introduction and letter.
- **M4 (thr.-k optimiser).** Coordinate descent now starts from the best
  of reactive, the cuts of the tuned global threshold and BATON-ho's
  boundaries (`cuts_from_models`, `fit_threshold_k(starts=...)`); unit
  test added; all tables re-run; §4.2 text rewritten.
- **M5 (letter consistency).** Reference points labelled and separated;
  RL test now instance-level; exact-reset recommendation added to the
  appendix discussion; BATON-cf acknowledged wherever it is highest;
  regret wording corrected (smallest on City Det; pooled ratio vs route
  average both reported); aggregation stated in captions; the single
  bound failure explained as Monte Carlo error; clean-day share caveat.
- **M6 (theory text).** Bellman3 exact under independence and
  misspecified under factor dependence; sufficiency qualified to the
  handoff problem; Clément citation replaced by Egloff (2005) and Zanger
  (2013) without a convergence claim, with the fitted-policy caveat;
  classical antecedents added (Serfozo 1976, Puterman 1994, Müller &
  Stoyan 2002, Yang et al. 2000, Minis & Tatarakis 2011, Tatarakis &
  Minis 2009; Crossref-verified in `../VERIFY_CITATIONS.md`); Ass. 2 added
  to the boundary claim of Prop. 2; Lean abstraction sentence added.
- **M7 (geography).** City instances re-planned under the SAA gate; the
  plans are breach-free (reactive cost 0.04 per plan-day), so the claim is
  reworded as plan slack and demand profile; "congested networks" and
  "nearly free downtown" removed.

## Should fix

- **S1** two-lever rule (`tune_two_lever`) and DP³_N in every main table.
- **S2** three-action clairvoyant (`oracle3_costs`) in the main tables;
  BATON as a share of it reported.
- **S3** pool redesigned: Det and SAA plans, metropolitan pools,
  holding-cost sweep, break-even h1, λ = F_sb and λ = ∞ columns, decision
  at the refusal stop.
- **S4** day-type fits on the typed subsets (≈800/200) of one history;
  result changed (split fits no better than pooling) and text rewritten.
- **S5** shape test extended to ρ = 0.3 and the day factor, with
  injected-dip power at 10% and 25%.
- **S6** fixed depot-return fee sweep (5–30) in Table 11 and Fig. 8;
  break-even fee reported.
- **S7** references relabelled "high-data plug-in references"; DP³ uses
  the state-conditional F (documented); exact grid DP at ρ = 0 added
  (Table 8, `run_baton_r1.py exact`).

## Consider

Tables 5, 7, 11, 13, 14 (old numbering), the fleet figure and the proof
of Proposition 1 moved to an appendix; BATON-cf in the main tables with
the post-hoc disclosure; "no tuning" qualified; small fixes (Remark 1
wording, currency units in text and figures, l. 901 phrase, "orders of
magnitude", [0,1], σ restricted to k < T, DP_N at N = 100, anecdotal
field observations, budget-table caption, Det tail claim, reset wording,
Alg. 1 comment, piecewise-linear lookup, timing wording, per-gate pool
sizes, Fig. 2a trace cut at the breach, deployment-selection caveat in
the Fig. 3 caption); letter tone (R3.2, R3.5, opening paragraph).
