# Title Page

**Title:** Minimum Weighted Hazard-Exposure Dispatch: Complexity, an
Exact Algorithm, and an FPTAS

**Short title / running head:** Minimum Weighted Hazard-Exposure
Dispatch

**Authors:**
1. Quang-Vinh Dang — British University Vietnam, Hung Yen, Vietnam — vinh.dq4@buv.edu.vn (corresponding author)
2. Minh Ngoc Dinh — Millennia Education, Ho Chi Minh City, Vietnam — minh.dinh@maeducation.com
3. Hoang-Viet Vu — British University Vietnam, Hung Yen, Vietnam — 30066555@st.buv.edu.vn
4. Phuc-Son Nguyen — UEH University, Ho Chi Minh City, Vietnam — son.np33@ueh.edu.vn

**Corresponding author:** Quang-Vinh Dang, vinh.dq4@buv.edu.vn

**Abstract:**

We study a dispatch-scheduling problem arising when a single depot
must send a vehicle or crew to a set of sites before a spreading
hazard—a flood, wildfire, contamination plume, or growing congestion
zone—reaches them. Each site has a fixed round-trip dispatch time, a
deadline at which the hazard arrives, and a criticality weight; the
objective is a dispatch order maximizing the total weight of sites
reached before the hazard arrives. We call this problem Minimum
Weighted Hazard-Exposure Dispatch (MWHED) and show it is exactly the
classical single-machine problem of maximizing weighted on-time jobs,
reparametrized in transportation terms. We prove MWHED is weakly
NP-hard via a reduction showing its equal-deadline special case is
exactly the 0/1 knapsack problem; give a pseudo-polynomial exact
dynamic program from an earliest-deadline-first exchange argument;
adapt value-scaling to obtain a fully polynomial-time approximation
scheme; and identify a second tractable special case—equal dispatch
times—solvable in $O(n\log n)$ time by a matroid greedy algorithm,
showing hardness requires heterogeneity in both dispatch cost and
criticality, not either alone. We further show the naive fallback of
dispatching in deadline order with no reconsideration has unbounded
worst-case loss, so an exact or near-exact algorithm is necessary, not
merely convenient. Numerical experiments, robust across alternative
weight and deadline distributions, confirm the approximation scheme
tracks the true optimum as instance size grows, while the naive policy
degrades. The results give planners exact and near-exact tools for a
provably hard scheduling problem, and delineate which structural
features of a dispatch network make hazard-exposure minimization
tractable.

(250 words, within JOCO's 150–250 word limit.)

**Keywords:** combinatorial optimization; scheduling under deadlines;
computational complexity; approximation algorithms; transportation
networks; disaster response

**Declarations:** see the manuscript's "Statements and Declarations"
section (Funding, Competing Interests, Author Contributions, Data
Availability, Use of Large Language Models).
