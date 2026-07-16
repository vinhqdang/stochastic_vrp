# Cover Letter

To the Editors of the Journal of Combinatorial Optimization,

This manuscript is submitted to the journal track of CSoNet 2026.

We are pleased to submit our manuscript, "Minimum Weighted
Hazard-Exposure Dispatch: Complexity, an Exact Algorithm, and an
FPTAS," for consideration in the Journal of Combinatorial Optimization
via CSoNet 2026's Journal Track.

The paper studies a scheduling problem that arises whenever a single
depot must choose the order in which to dispatch a vehicle or crew to
a set of sites, each racing against a spreading hazard — a wildfire,
flood, contamination plume, or growing congestion zone — that
compromises a site once it arrives. We formalize this as Minimum
Weighted Hazard-Exposure Dispatch (MWHED) and give what we believe is
a complete complexity and algorithmic picture of it: we show the
problem is exactly the classical single-machine scheduling problem of
maximizing the weighted number of on-time jobs in transportation
language; we prove it is weakly NP-hard via an elementary,
self-contained reduction showing its equal-deadline special case is
exactly the 0/1 knapsack problem; we give an exact pseudo-polynomial
dynamic program derived from an earliest-deadline-first exchange
argument; we adapt the classical value-scaling technique to obtain a
fully polynomial-time approximation scheme; and we identify a second,
structurally distinct tractable special case — equal dispatch cost,
solvable in near-linear time via a matroid greedy algorithm — showing
precisely that it is the *combination* of heterogeneous dispatch cost
and heterogeneous criticality, not either alone, that drives the
problem's hardness; and we prove that the natural fallback of
dispatching sites in deadline order with no reconsideration at all has
unbounded worst-case loss relative to the optimum, so an exact or
near-exact algorithm is necessary for any guarantee, not merely
convenient.

We believe this work fits the Journal of Combinatorial Optimization's
scope directly: it is a new combinatorial scheduling problem, with a
complete hardness/tractability/approximation characterization, that is
motivated by and connects transparently to transportation and
infrastructure-network applications (CSoNet 2026's Track D). To our
knowledge, no prior work establishes the knapsack-equivalence hardness
result, the exact dynamic program, the FPTAS, or the equal-cost
tractable case we give here, though the underlying scheduling problem
it recovers is classical.

This manuscript is original, has not been published previously, and is
not under consideration at any other journal. The author has approved
the submission and has no competing interests to disclose. Per the
journal's editorial policy, the use of an AI coding assistant during
manuscript preparation (LaTeX formatting and running the reported
numerical illustration) is disclosed in the manuscript's Statements and
Declarations section; all theorem statements, proofs, and text are
authored and taken responsibility for by the author.

As required, the paper's abstract will also be submitted separately to
the CSoNet 2026 conference submission system to secure a presentation
slot.

Thank you for your consideration. We look forward to your response.

Sincerely,

Quang-Vinh Dang
British University Vietnam, Hung Yen, Vietnam
vinh.dq4@buv.edu.vn
