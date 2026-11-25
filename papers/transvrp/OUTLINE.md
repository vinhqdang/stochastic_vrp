# TRANSVRP — manuscript plan (2026-11-25)

Decisions (author, 2026-11-25): **venue Transportation Science**; **single author**
Quang-Vinh Dang; **story: the state matters, the learner does not.** BATON is cited
(accepted at C&OR, 24 Nov 2026, in press). Manuscript file: `transvrp.tex`
(never `main.tex`). Tables are generated from the committed results by a script
(`make_tables.py`), never typed by hand.

To check before drafting the front matter: TS submission format (INFORMS template,
page limit, single- or double-blind review). If double-blind, BATON is cited in the
third person.

## Working title

*What the driver knows: learned continuation values for in-route handoff in vehicle
routing with stochastic pickups and deliveries*

(Alternatives: "State, not learner: …"; "Does the learner matter? …")

## One-paragraph story

A vehicle on a fixed route with simultaneous pickups and deliveries must decide at
each stop whether to continue or hand the rest of the route to a standby vehicle
before the load breaches capacity. Optimal-stopping recourse (BATON, Longstaff–Schwartz
on the scalar state (stop, load)) is the incumbent. The 2023–2026 literature offers
many learners for the same decision: threshold DP, approximate LP, randomized neural
regression, deep RL (DQN, PPO), transformers. We ask which ingredient matters. Two
pre-registered experiments on 62 routes of 20 instances give one answer: **the
information in the state**. Given the online split of pickups and deliveries, every
competent learner (neural regression, gradient-boosted trees, randomized networks,
DQN) lands within about 0.3% of the others and 0.5–0.85% below BATON; architecture does not matter
(MLP = set encoder = trees). Methods restricted to the load (threshold DP, ALP) lose
11–13%, and the loss disappears in a world where stops are independent, so it is the
dependence between stops that the scalar state cannot see. Policy gradient (PPO) is
the one learner that falls short at our budget.

## Contributions

1. A clean decision-level comparison of seven published learner families for in-route
   handoff, on identical simulated paths, with pre-registered decision rules, a frozen
   test stream opened once per experiment, and family-wise error control.
2. Attribution: the gain over the scalar optimal-stopping policy is information
   (FULL vs SCALAR), not architecture (MLP, set encoder, boosted trees, randomized
   networks equivalent within ±0.7%), not the learning paradigm (regression-MC = DQN).
3. A mechanism test: an independence-resampled world removes the gain and the loss of
   load-only methods, which identifies dependence between stops as the source.
4. A data-size curve: the richer state needs about 500 paths per route to pay; below
   that the scalar policy wins.
5. Reproducible code and a protocol (gates, manifests, hashed family line) that other
   learned-recourse comparisons can reuse.

## Sections

1. **Introduction** (story above; the question "which ingredient matters"; contributions;
   honest scope: recourse cost only, one generator family).
2. **Related work**
   - Recourse for VRPSD: Bertsimas 1992; Yang–Mathur–Ballou 2000; Secomandi–Margot 2009;
     Louveaux–Salazar-González 2009, 2018; Goodson et al. 2016; Venkateshan–Mathur 2026.
   - Learned value functions / ADP: Zhu–Adulyasak–Rousseau 2026 (ALP); Dastpak–Errico–
     Jabali 2023 (offline ADP, DQN), 2026 preprint; Akkerman et al. 2025.
   - Optimal stopping by regression: Longstaff–Schwartz 2001; Tsitsiklis–Van Roy 2001;
     Herrera et al. 2024 (RLSM); Becker et al. 2019 (DOS); Andersson et al. (preprint).
   - Neural routing and transformers: Kool et al.; Nazari et al.; POMO; SVRPBench 2025;
     Kadyrov et al. 2025 (PPO); OR-Transformer (Liu et al.) as the seed.
   - BATON (in press, C&OR) as the incumbent.
3. **Problem**: SVRPSPD route, three-class fleet cost, two-action stopping problem,
   recourse cost (H, E, 0); demand generator with two copulas (ρ = 0.6); what the
   driver observes online.
4. **Methods compared** (one table: state, learner, training signal, source)
   - BATON (scalar LS + isotonic), DP on (k, W) at 10^6 paths;
   - FULL (rich state, neural regressor, cross-fitted LS), SCALAR, MLP, BOOST;
   - THR, ALP-Z, RLSM, DQN, PPO (adaptations to the two-action problem; deviations from
     the source papers listed; DOS cut for compute).
5. **Experimental protocol**: instances and the 62 eligible routes; train/dev/test
   streams; unit = instance; δ = 0.7% from refitting noise; decision rules (WIN, LOSE,
   TIE_STRONG, TIE_WEAK); Holm; dev gates; freeze and manifest procedure; compute.
6. **Results**
   - 6.1 Primary A: FULL vs BATON (+0.82% [0.65, 0.99], 19/20).
   - 6.2 Attribution A: information (H2), competence (H3), architecture (ARCH, BOOST).
   - 6.3 Data-size curve (crossover N = 500; FULL worse at N = 100).
   - 6.4 Published baselines (B): FULL vs ALP-Z, PPO (WIN), RLSM, DQN (TIE_STRONG);
     THR descriptive; every arm vs BATON.
   - 6.5 Mechanism: independence world.
   - 6.6 Strata, replicate variance, compute.
7. **Discussion**: what transfers (state design beats architecture search); when the
   rich state pays (enough paths; dependence between stops); policy gradient's
   sample cost; limits.
8. **Limitations** (explicit): recourse cost only; one generator family; DQN/PPO budgets
   below their source papers; DOS not run; 2 replicates in B; THR fails its own-model
   check (5 of 7 failures are zero-variance routes); one test stream per experiment.
9. **Conclusion.**
Appendices: feature definitions; hyperparameters and tuning record; gate definitions and
dev results; protocol deviations (from RESEARCH_LOG); reproduction commands.

## Tables and figures (all generated)

| id | content | source |
|---|---|---|
| T1 | methods compared: state, learner, signal, citation | text |
| T2 | instances and routes (m, strata, breach rate) | `results/expA/eligibility_dev.csv` |
| T3 | primary and attribution contrasts of A | `results/expA/analysis_test/*.csv` (+ M11) |
| T4 | family contrasts of B and D_a | `results/expB/analysis_test_b/family.csv`, `da.csv`, `outside.csv` |
| T5 | independence world | `results/expB/analysis_test_b/independence.csv` |
| T6 | compute | expA/expB merged CSVs (fit_seconds) |
| F1 | route and decision timeline (schematic) | TikZ |
| F2 | data-size curve with CI | expA analysis |
| F3 | forest plot: every learner vs BATON, coloured by state class | expA + expB |
| F4 | main vs independence world, by method | expB independence + family |

## Work plan

1. Check TS author guidelines (format, length, blind review); fetch the INFORMS template.
2. `make_tables.py` + `make_figures.py` from committed CSVs.
3. Draft sections 3–6 first (facts), then 1, 2, 7–9.
4. Citations: every key verified (DOI) in `VERIFY_CITATIONS.md`; BATON as in press.
5. Compile with pdflatex/bibtex; review by independent agents (method, statistics,
   writing, citation check); revise; then author read.
