# QUASAR — status

- **Working title:** QUASAR: quantum amplitude estimation for stochastic
  vehicle routing — coherent feasibility oracles, common random numbers in
  superposition, and nested search for chance-constrained SVRPSPD.
- **Venue:** not fixed. Candidates: INFORMS Journal on Computing; IEEE
  Transactions on Quantum Engineering. Decide at M3 from the strength of the
  results.
- **State:** **ACTIVE — M3 next (amplitude estimation experiments, coherent CRN on real instances).** Open for
  edits.
- **Sharing:** uses the shared engine's instances, scenario generators and
  route conventions (`svrpspd_wdro/core`); quantum code is separate in
  `svrpspd_wdro/quantum/`. The research question (quantum estimation and
  search) is disjoint from BATON's (planning gates and online recourse), so
  there is no overlap of claims with papers 1–2.
- **Toolchain:** Qiskit 2.x, qiskit-aer, qiskit-algorithms (Python 3.11).
  GPU only if needed, via Colab.

## Milestones
- [x] M0 scaffold + literature review (2025–2026 included; LITERATURE.md, 139 verified records, 45 from 2026)
- [x] M1 classical references (exact peak DP, discretisation, insertion lemma; svrpspd_wdro/quantum/classical.py, 112 tests)
- [x] M2 Qiskit oracle + resource counts + correctness tests (quantum/oracle.py, DESIGN_M2.md, exact Toffoli formula)
- [ ] M3 amplitude estimation experiments, coherent CRN
- [ ] M4 cheapest-feasible search, lower bounds
- [ ] M5 resource estimation, crossover, manuscript

## Freeze policy
Editable until submission; frozen while under review, like the other papers.
