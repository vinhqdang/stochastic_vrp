# QUASAR — literature review (state of 2026-10-12)

Built from five topic reviews and one verification pass. Every key below is
in `references.bib` (139 verified records, 45 of them dated 2026). Records
seen only in search snippets are in `VERIFY_CITATIONS.md` and are not used.
The arXiv API was rate-limited during the searches, so "nothing found"
statements are evidence, not proof; they are re-checked before submission.

## 1. Quantum computing for vehicle routing

The field is large and almost entirely deterministic. Surveys and anchors:
Osaba2022, Harwood2021, Feld2019, Irie2019, Borowski2020, Azad2023. The
2025–2026 work is dominated by QUBO/QAOA/annealing formulations and their
feasibility problems: Lei2027RuleTRC (constraint-aware QAOA mixing, TR-C),
Azfar2026SciRepFeasibility, Onah2026ColoredPerm (qubit-efficient encodings),
Sharma2026LagrangianCVRP, Gurevich2026ArcQUBO (connectivity of arc-based
QUBOs), Rezk2026GNNCoarsening (D-Wave, Solomon instances),
Matwiejew2026QWOA (quantum walks), Okada2026EVSciRep, Holliday2026TabuQMI,
Maciejunes2025QCE, Rosendo2025IEEESoftware, Osaba2025CEC; learning hybrids
Ritt2026HybridQNN, Murthy2026QAOAQRL, Giang2025gat, Sofi2026APGEM.
Two are close to our setting in method rather than in model:
- **Moosavi2026RLQALNS**: quantum samplers inside the repair step of ALNS on
  IBM Heron; the quantum repair is admissible in about 16% of states and not
  better on average. It is the only quantum-ALNS paper and it is
  deterministic.
- **Kim2026FeasibilityQS**, **MeghaziBourreau2026**: Grover-based routing with
  feasibility-preserving search; deterministic.
Go/no-go analyses: Onah2025RequirementsCVRP, xu2026urban (no stable,
reproducible advantage in urban transport yet).

**Stochastic routing.** The only quantum paper on a stochastic VRP is
**WangFilipovska2026** (arXiv, August 2026): amplitude estimation of
path-level travel-time quantities, small Qiskit experiments. It treats
travel-time uncertainty only, with no random demand, chance constraint,
CVaR, recourse, route comparison or resource count. Dixit2023 formulates a
scenario-based stochastic shortest path for annealing.

## 2. Quantum stochastic optimization

- Amplitude estimation over discrete decisions: Gacon2020 (simulation-based
  optimization; random-demand newsvendor and VaR portfolio; a sum register
  built with controlled adders and an integer comparator, the same primitive
  as our load-versus-capacity predicate; variational outer loop, no
  coupling between candidates, no resource estimates), Sharma2024
  (newsvendor), Rotello2024 (expected value function of a two-stage
  program), Skarlatos2025 (CVaR subgradients), WoernerEgger2019 (VaR/CVaR).
- Two-stage and Benders: Kuroiwa2025qgan, Mahroo2026benders,
  LopezBanos2026benders, Zheng2026gas, AmaniBani2026ride.
- Chance constraints: RibesMarza2026jcc (annealing for joint
  chance-constrained unit commitment) is the only quantum chance-constraint
  paper found, and it is not routing.
- Sequential decisions: Doriguello2022 (quantum Longstaff–Schwartz),
  Luo2025mdp, Ganguly2025regret, Xu2025qnpg, Cunha2025pomdp, Eker2026qantis.

## 3. Amplitude estimation and quantum Monte Carlo

Foundations: Brassard2002, Montanaro2015 (Õ(σ/ε) given a bound on σ),
KothariODonnell2023 (error σ/n without prior knowledge of σ), Hamoudi2021,
HamoudiMagniez2019, Cornelissen2022, AaronsonRall2020. Practical variants:
Suzuki2020 (maximum likelihood), Grinko2021 (iterative), nakaji2020,
giurgica2022, plekhanov2022, rall2021, herbert2022. 2025–2026 advances:
ramoa2025 and li2026 (Bayesian), labib2026 (non-power-of-two schedules),
erle2026 (near-optimal at any depth), tabarraei2026 (rare events),
borras2026 and lal2026 (hardware: MLAE on IonQ has an error floor near
2–5·10⁻² and no surviving scaling advantage). State preparation:
GroverRudolph2002, Mottonen2004, Mottonen2005, zoufal2019. Applications:
stamatopoulos2020, egger2021, Chakrabarti2021, kirke2026catastrophe.

Toolchain: qiskit-algorithms 0.4.0 still provides AmplitudeEstimation,
IterativeAmplitudeEstimation, MaximumLikelihoodAmplitudeEstimation and
FasterAmplitudeEstimation on Qiskit 2.x primitives (V2); IBM no longer
officially supports the package, so the project pins versions.

## 4. Search, local search and selection

Grover and minimum finding: grover1996fast, boyer1998tight, DurrHoyer1996;
bounded-error and variable-time search: hoyer2003bounded,
ambainis2006variable, ambainis2023variable. Error models decide whether a
speedup survives: regev2012faulty (independent per-call failure: no
speedup), shenvi2003noisy, quek2020robust. Our feasibility predicate is a
coherent bounded-error algorithm, the setting of hoyer2003bounded, not the
faulty-oracle model; the paper must say so. Local search: Aaronson2006
(no general speedup in the number of steps to a local optimum), so only the
per-iteration neighbourhood scan can be accelerated. Branch and bound:
Montanaro2020bb, chakrabarti2022bnb. Grover adaptive search:
gilliam2021grover, hess2026gas.
Bandits and simulation optimization: Wang2021bandits (best-arm
identification Õ(√Σ Δᵢ⁻²), matching lower bound), wan2022logregret,
buchholz2025bandits (restricted access to the reward randomness removes the
speedup), yamagami2025spatial, chen2026noise, liu2026qmab,
**HuHuZhou2026** (Grover adaptive search for discrete simulation
optimization) and **HuHu2026** (rate-optimal quantum selection, October
2026): both give each candidate an independent noise register; correlated
sampling is listed as future work.

## 5. Coupled differences (relevant to coherent common random numbers)

An2021mlmc (quantum multilevel Monte Carlo estimates E[P_l − P_{l−1}] on one
path, with cost governed by Var(P_l − P_{l−1})) and Blanchet2025 are the
precedents for estimating a difference on shared randomness. A shared-register
difference estimator follows from Montanaro2015 / KothariODonnell2023.

## 6. Resources and realism

Babbush2021 and sanders2020compilation: quadratic speedups do not pay on
early surface-code machines. omanakuttan2025threshold: the only rigorous
optimization crossover found (QAOA with amplitude amplification: 179
variables, about 15 h, about 7.4·10⁷ physical qubits at p = 10⁻³, 1 μs
cycle). Arithmetic and compilation: Cuccaro2004, Draper2000, Gidney2018,
remaud2024adders, litinski2019distillation, fowler2018lattice,
gidney2024cultivation, gidney2021rsa, gidney2025rsa, beverland2022requirements,
vandam2023azure, leblond2026surface; measured logical performance
acharya2024below. Benchmarks and critiques: abbas2024challenges,
koch2026qoblib, erdmann2026strategic. Typical assumptions used in the
literature: physical error 10⁻³, 1 μs cycle, 10 μs reaction time, code
distance about 15 for cultivated T states.

## 7. Gap and positioning (what QUASAR may claim)

| Item | Status in the literature | What QUASAR adds |
|---|---|---|
| QAE for a stochastic VRP | exists for travel times (WangFilipovska2026) | random demands, pickup and delivery, peak-load chance and CVaR gates |
| QAE for VaR/CVaR over discrete decisions | known (WoernerEgger2019, Gacon2020, Rotello2024, Skarlatos2025) | a feasibility oracle inside a local-search loop, with explicit circuits and counts |
| Estimating a difference on shared randomness | corollary of Montanaro2015 / KothariODonnell2023; precedent An2021mlmc | route-comparison construction; the insertion lemma (an insertion moves the peak by at most one demand, so the difference is an indicator of a window below capacity); measured window masses on real instances |
| Quantum selection with independent noise | HuHuZhou2026, HuHu2026, Wang2021bandits | shared-register comparison of moves, which these papers leave as future work |
| Grover over VRP/ALNS neighbourhoods with estimated feasibility | not found; generic bounded-error search exists | cheapest-feasible search on a sorted move list with an estimated predicate; honest crossover analysis, given Aaronson2006 and Babbush2021 |
| Toffoli counts for a logistics workload | not found | resource formulas and a fault-tolerant crossover estimate |

Claims to avoid: "first quantum algorithm for stochastic VRP" (false since
WangFilipovska2026); "new quantum variance reduction technique" (the
estimator itself is a corollary); any practical advantage without the
crossover analysis.
