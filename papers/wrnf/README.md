# Paper 4 — WRNF (Wasserstein-Robust Network Flow)

Target venue: **Computational Optimization and Applications** (Springer).
State: **spec + verified prototype, no manuscript yet.** See `STATUS.md`.

## What the paper is

Two-stage distributionally robust optimization is usually motivated as a way
to get *better decisions* than sample-average approximation under limited
data. For network flow with ambiguous demand, that is sharply conditional,
and the paper draws the line:

- Over an **unbounded** support, the type-1 Wasserstein robust problem equals
  the sample-average problem plus the constant $\varepsilon p$, so the robust
  decision is *identical* to the SAA decision. Robustness moves the objective
  and changes nothing you do.
- Over a **bounded** support the collapse fails, decisions genuinely differ,
  and the worst case has no closed form — this is where the algorithmic
  content is, and where the problem is known to be hard in general.
- The Lipschitz modulus driving all of this is **combinatorial** in both
  regimes, which is what makes a polynomial separation oracle possible under
  a precisely stated restriction.

Full problem statement, the theorems to prove, the algorithm design, the
experiment plan, and the prior art to engage: **`PROJECT.md`**.

## File manifest

| File | What it is |
|---|---|
| `PROJECT.md` | The spec: model, findings so far, theorems to prove, algorithm and experiment plan, prior art, venue facts |
| `STATUS.md` | Venue, state, freeze policy, and the recorded idea-selection rationale (why three other candidates were rejected) |
| `prototype_check.py` | Numerical falsification tests for the three central claims; computes the worst-case distribution by brute-force LP over transport plans, so nothing depends on a reformulation being right |
| `prototype_results.txt` | Raw output of the above — the evidence of record |

## Reproducing the prototype

```bash
export GRB_LICENSE_FILE=/root/gurobi.lic
cd papers/wrnf
python3 prototype_check.py
```

Runs in about a minute on 4 cores. All three claims currently pass, exact to
machine precision:

```
Claim 1 objective identity        : HOLDS     (gap <= 3.6e-15)
Claim 1 decision irrelevance      : HOLDS     (argmin SAA == argmin DRO)
Claim 2 bounded-support relevance : HOLDS     ([4,4] vs [5.5,6.5])
Claim 3 combinatorial modulus     : HOLDS     (both regimes, exact)
==> paper spine SURVIVES
```

Note that the prototype earned its keep: it **falsified** the first version of
the theory. The Lipschitz modulus was initially predicted as the cheapest
incoming arc cost, which is wrong — once installed capacity binds, the
marginal action is refusing demand at the penalty rate, so the modulus is $p$.
The corrected two-regime statement is what now passes, and the collapse result
follows from the modulus being independent of $x$.

## Independence from the rest of the repo — deliberate

This paper's code shares **nothing** with `svrpspd_wdro/`; it does not import
from it and nothing imports from it. That is not incidental. BATON
(`papers/baton/`) is under review at *Computers & OR* and also uses
Wasserstein-DRO machinery — but as one of six capacity-feasibility *gates*
inside an ALNS route constructor, filtering candidate routes. Here the
Wasserstein ball defines the *objective* of a two-stage capacity-installation
problem, the decisions are installed capacities rather than routes, the method
is an exact cutting plane rather than a heuristic, and the instances are
newly generated flow networks rather than Dethloff / Salhi–Nagy / OSM city
sets. `PROJECT.md` §6 gives the full comparison table and the rules adopted to
keep the separation verifiable.

The same reasoning was applied to `papers/csonet2026/` (paper 3), for the same
reason: papers under review elsewhere should not be entangled with new
submissions.
