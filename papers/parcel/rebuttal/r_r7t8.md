Thank you. New experiments ([a,b] = 95% cluster CI, Δ in accuracy points):

**Theory.** Accepted as elementary; Proposition 3 becomes a remark and the IM framing is dropped (one shared set vs. per-receiver sets). The comparison now replicates: +19.2 held-out, +24.2 on Llama-3.3-70B, +20.8 in a dependent pipeline, +7.5 with a cross-encoder on both sides.

**Multi-agent system.** Added: 120 MuSiQue chains of 2–4 agents, each fed upstream answers, against a baseline forwarding the whole accumulating conversation. Oracle +20.8 [+11.7,+29.2] at 13.6× fewer tokens; per-receiver vs. one shared set +20.8 (both p<.001); cross-encoder per-receiver −0.8 [−10.8,+8.3] at 8.3× fewer. Cost grows ~linearly in agents, not quadratically; we will fix that sentence.

**Baselines.** With a cross-encoder, per-receiver top-3 (k fixed in advance) is −1.3 [−5.0,+2.5] vs. broadcast at 6.7× fewer tokens; top-1 (post hoc) is −0.4 at 23.6×. It beats the cross-encoder seed set by +7.5 (p=.001) at 32% fewer tokens. We did not run RCR-Router, AgentPrune or KVComm. Our arm is an RCR-style per-agent router, not a reimplementation. Pruning messages (keep only referenced answers) saves just 2.3% of tokens in the pipeline because documents dominate. KVComm changes the medium and is complementary.

**Evidence.**
- 120 new held-out instances (240 receivers), k=5 and q=0.3 frozen: −7.9 and −4.6 (submitted −8.3, −4.2).
- Oracle +5.0 [+0.4,+9.6], p=.049 (borderline, same sign in every run); +20.8 in the pipeline.
- Llama-3.3-70B: per-receiver vs. seed set +24.2, p<.001.

**Bottleneck.** Cross-encoders rank the gold paragraph first for 91%/80% of receivers (lexical 58%), and the same allocation improves from −13.8 to −1.3. Given identical scores interpolated to perfect labels, Algorithm 1 beats uniform by +20.0 recall points only with perfect scores (+1–2 with real ones). An LLM judge was blocked by API quota; we will add it.

**Inconsistencies.** The TL;DR/abstract came from an earlier draft; sorry, we will fix it. Table 4: before submission, an earlier draft's table could not be reproduced, so we rebuilt it with `value_curves_check.py` (in the supplement; all submitted numbers regenerate). One cell changed (captured-mass, plain: tie→loss). The caption disclosed this; we will add a clearer provenance note.

**Writing.** Agreed; Section 4 will be cut and de-hedged.
