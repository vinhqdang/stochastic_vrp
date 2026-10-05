# Title Page

**Title:** Minimum Weighted Hazard-Exposure Dispatch: Complexity, an
Exact Algorithm, and an FPTAS

**Short title / running head:** Minimum Weighted Hazard-Exposure
Dispatch

**Authors:**
1. Quang-Vinh Dang — British University Vietnam, Hung Yen, Vietnam — vinh.dq4@buv.edu.vn (corresponding author)
2. Minh Ngoc Dinh — Millennia Education, Ho Chi Minh City, Vietnam — minh.dinh@maeducation.com
3. Hoang-Viet Vu — British University Vietnam, Hung Yen, Vietnam — 30066555@st.buv.edu.vn
4. Phuc-Son Nguyen — UEH University, Ho Chi Minh City, Vietnam — son.np3393@ueh.edu.vn

**Corresponding author:** Quang-Vinh Dang, vinh.dq4@buv.edu.vn

**Abstract:**

We study a dispatch-scheduling problem arising when a single
depot must send a vehicle or crew to sites before a spreading hazard
(flood, wildfire, contamination plume) reaches them. Each site has a
round-trip dispatch time, a hazard-arrival deadline and a criticality
weight; the objective is a dispatch order maximizing the total weight of
sites served in time. We call this Minimum Weighted Hazard-Exposure
Dispatch (MWHED) and show it is exactly the classical problem
$1‖\sum w_jU_j$, so its basic picture is inherited from scheduling
theory: weak NP-hardness (the equal-deadline case is 0/1 knapsack), a
Lawler–Moore-type pseudo-polynomial dynamic program, and a value-scaling
FPTAS. We present these in a self-contained form for the transportation
audience and say precisely which are classical. The framing adds a
classification: restricted to the instances in which some of the
dispatch times, weights and deadlines are constant, the problem is
polynomial exactly when dispatch times or weights are constant and
weakly NP-hard otherwise, so heterogeneous deadlines alone never cause
hardness; and a matroid-greedy algorithm for equal dispatch times that
extends to any number of identical vehicles. Natural heuristics (deadline
order with or without skipping, a weighted greedy repair) have unbounded
worst-case ratio, and the FPTAS analysis is tight up to a factor
$1-\epsilon^2$. A numerical study with active FPTAS rounding, verified
against exhaustive search, and a Camp Fire case study with a sensitivity
analysis complete the paper.

(Revised abstract, 229 words, within JOCO's 150–250 word limit.)

**Keywords:** combinatorial optimization; scheduling under deadlines;
computational complexity; approximation algorithms; transportation
networks; disaster response

**Declarations:** see the manuscript's "Statements and Declarations"
section (Funding, Competing Interests, Author Contributions, Data
Availability, Use of Large Language Models).
