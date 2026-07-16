# Title Page

**Title:** Minimum Weighted Hazard-Exposure Dispatch: Complexity, an
Exact Algorithm, and an FPTAS

**Short title / running head:** Minimum Weighted Hazard-Exposure
Dispatch

**Author:** Quang-Vinh Dang (sole author)

**Affiliation:** British University Vietnam, Hung Yen, Vietnam

**Corresponding author:** Quang-Vinh Dang, vinh.dq4@buv.edu.vn

**ORCID:** (add if you have one — not currently recorded in this
package)

**Abstract:**

We study a dispatch-scheduling problem arising when a single depot
must send a vehicle or crew to a set of sites before a spreading
hazard—a flood, wildfire, contamination plume, or growing congestion
zone—reaches them. Each site has a fixed round-trip dispatch time, a
deterministic hazard-arrival deadline, and a criticality weight; the
objective is to choose a dispatch order maximizing the total weight of
sites reached before the hazard arrives. We call this problem Minimum
Weighted Hazard-Exposure Dispatch (MWHED) and show it is exactly the
classical single-machine problem of maximizing weighted on-time jobs,
reparametrized in transportation terms. We prove MWHED is weakly
NP-hard, by an elementary reduction showing that its equal-deadline
special case is exactly the 0/1 knapsack problem; we give a
pseudo-polynomial exact dynamic program, derived directly from an
earliest-deadline-first exchange argument; we adapt the classical
value-scaling technique to obtain a fully polynomial-time
approximation scheme; and we identify a second, structurally distinct
special case—equal round-trip dispatch times, corresponding to sites
equidistant from the depot—that is solvable exactly in $O(n\log n)$
time by a matroid greedy algorithm, showing that hardness requires the
joint heterogeneity of dispatch cost and criticality, not either
alone. A small numerical illustration confirms the approximation
scheme tracks the exact optimum closely as instance size grows, while
a naive earliest-deadline dispatch order without optimization
degrades. Implications: the results give infrastructure and
emergency-response planners exact and near-exact tools for a provably
hard scheduling problem, and delineate precisely which structural
features of a dispatch network make hazard-exposure minimization
tractable.

**Keywords:** combinatorial optimization; scheduling under deadlines;
computational complexity; approximation algorithms; transportation
networks; disaster response

**Declarations:** see the manuscript's "Statements and Declarations"
section (Funding, Competing Interests, Author Contributions, Data
Availability, Use of Large Language Models).
