# CSoNet 2026 conference-submission-system abstract

(Per CSoNet's Journal Track rules: "To ensure presentation scheduling,
authors must also submit their paper's abstract to the CSoNet
organizers via the conference submission link"
https://meteor.springer.com/CSoNet2026 — separate from the JOCO
submission itself. Suggested track: Track D, "Transportation and
infrastructure networks." Paste the fields below into that system.)

**Title:** Minimum Weighted Hazard-Exposure Dispatch: Complexity, an
Exact Algorithm, and an FPTAS

**Author:** Quang-Vinh Dang, British University Vietnam, Hung Yen,
Vietnam (vinh.dq4@buv.edu.vn)

**Submission type:** Journal Track (Journal of Combinatorial
Optimization)

**Track:** Track D — Transportation and infrastructure networks

**Abstract:**

We study a dispatch-scheduling problem arising when a single depot
must send a vehicle or crew to a set of sites before a spreading
hazard — a flood, wildfire, contamination plume, or growing congestion
zone — reaches them. Each site has a fixed round-trip dispatch time, a
deterministic hazard-arrival deadline, and a criticality weight; the
goal is to choose a dispatch order maximizing the total weight of
sites reached before the hazard arrives. We call this problem Minimum
Weighted Hazard-Exposure Dispatch (MWHED) and give a complete
complexity and algorithmic picture of it. We show MWHED is exactly the
classical single-machine problem of maximizing weighted on-time jobs,
reparametrized in transportation terms, and prove it is weakly
NP-hard by an elementary reduction showing its equal-deadline special
case is exactly the 0/1 knapsack problem. We give a pseudo-polynomial
exact dynamic program derived from an earliest-deadline-first exchange
argument, and adapt the classical value-scaling technique to obtain a
fully polynomial-time approximation scheme. We further identify a
second, structurally distinct special case — equal round-trip dispatch
times, corresponding to sites equidistant from the depot — solvable
exactly in near-linear time by a matroid greedy algorithm, showing
that hardness requires the joint heterogeneity of dispatch cost and
criticality, not either alone. A numerical illustration confirms the
approximation scheme tracks the exact optimum closely, while a naive
dispatch order without optimization degrades substantially. The
results give infrastructure and emergency-response planners exact and
near-exact tools for a provably hard scheduling problem central to
transportation and infrastructure-network resilience, and delineate
precisely which structural features of a dispatch network make
hazard-exposure minimization tractable.

**Keywords:** combinatorial optimization, scheduling under deadlines,
computational complexity, approximation algorithms, transportation
networks
