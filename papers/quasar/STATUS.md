# QUASAR — status

- **Working title:** QUASAR: quantum amplitude estimation for stochastic
  vehicle routing — coherent feasibility oracles, common random numbers in
  superposition, and nested search for chance-constrained SVRPSPD.
- **Venue:** not fixed. Candidates: INFORMS Journal on Computing; IEEE
  Transactions on Quantum Engineering. Decide at M3 from the strength of the
  results.
- **State:** **ACTIVE — M0 (scaffold and literature review).** Open for
  edits.
- **Sharing:** uses the shared engine's instances, scenario generators and
  route conventions (`svrpspd_wdro/core`); quantum code is separate in
  `svrpspd_wdro/quantum/`. The research question (quantum estimation and
  search) is disjoint from BATON's (planning gates and online recourse), so
  there is no overlap of claims with papers 1–2.
- **Toolchain:** Qiskit 2.x, qiskit-aer, qiskit-algorithms (Python 3.11).
  GPU only if needed, via Colab.

## Milestones
- [ ] M0 scaffold + literature review (2025–2026 included)
- [ ] M1 classical references (exact peak DP, discretisation, insertion lemma)
- [ ] M2 Qiskit oracle + resource counts + correctness tests
- [ ] M3 amplitude estimation experiments, coherent CRN
- [ ] M4 cheapest-feasible search, lower bounds
- [ ] M5 resource estimation, crossover, manuscript

## Freeze policy
Editable until submission; frozen while under review, like the other papers.
