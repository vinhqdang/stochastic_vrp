Thank you. New experiments ([a,b] = 95% cluster CI, Δ in accuracy points):

**Q1 / dependent agents.** We ran 120 MuSiQue chains of 2–4 agents. Each agent's question uses the upstream agent's answer, so errors propagate; the baseline forwards the whole accumulating conversation (gemini-3.5-flash-lite, API token counts). The findings transfer:
- Oracle vs. full history: +20.8 [+11.7,+29.2] at 13.6× fewer tokens.
- Per-receiver vs. one shared set: +20.8 (both p<.001).
- Cross-encoder per-receiver: −0.8 [−10.8,+8.3] at 8.3× fewer tokens, +11.7 over lexical (p=.013), but still 21.7 below the oracle (UNKNOWN answers cascade).

Cost grows roughly linearly in agents, not quadratically; we will correct this. On additivity: chains are not additive, but that hurts the shared set more. Under a conjunctive utility, Proposition 1(a)'s instance gives the shared set 0 and per-receiver 1 (unbounded ratio). Proposition 4 does need additivity; we will state this scope.

**Q2 / estimation vs. allocation.** Tested two ways.
(i) Same scores to Algorithm 1 and uniform top-k, interpolated from practical to perfect labels (250 instances, heterogeneous receivers, recall at matched tokens). Algorithm 1 gains +0.9 (lexical), +1.8 (cross-encoder), but **+20.0 [+19.1,+21.0] with perfect scores**; with homogeneous receivers it gains ≤0.
(ii) Same allocation (per-receiver top-3) on 120 new held-out instances: a cross-encoder instead of lexical moves Δ vs. broadcast from −13.8 to −1.3 [−5.0,+2.5].
So the allocator has headroom that only near-perfect estimates unlock. (i) is a recall-level result.

**Novelty vs. RCR-Router.** RCR-Router selects per-agent memory under a budget without guarantees. We add: (1) a proof that one shared set loses a factor n even on monotone submodular instances, measured at +19 to +24 points across Gemini, Llama-3.3-70B, held-out data and the pipeline; (2) an exact decomposition, so per-receiver allocation inherits single-receiver approximability; (3) evidence that a strong per-agent scorer recovers most of the gap and smarter allocation adds little.

**Evidence.** With q=0.3 frozen, the held-out loss at 2.6× is −4.6 [−8.7,−0.4]. The oracle needs labels (+5.0, p=.049), but the cross-encoder does not: −1.3 at 6.7× (k fixed in advance), −0.4 [−4.6,+3.7] at 23.6× (top-1, post hoc).
