# PARCEL — citations pending verification

Repo rule: `references.bib` holds **only** entries checked against live
publisher / DOI / arXiv records. Everything below was surfaced by web
search during planning on 2026-09-06 and is recorded here with the
claim it is meant to support. **Nothing moves to `references.bib`
until the record is opened and the metadata confirmed.** Search-result
snippets are not verification.

## Foundational — influence maximization

| Key | Work | Supports | Status |
|---|---|---|---|
| kempe2003 | Kempe, Kleinberg, Tardos — Maximizing the spread of influence through a social network, KDD 2003 | the monotone+submodular ⇒ `(1−1/e)` greedy result the paper argues does not transfer | well known, still verify DOI/pages |
| borgs2014 | Borgs, Brautbar, Chayes, Lucier — reverse influence sampling, SODA 2014 | first near-linear-time IM; head of the scaling lineage | verify |
| tang2015 | Tang, Xiao, Shi — IMM / influence maximization via martingales, SIGMOD 2015 | the sketch-based state of the art that the GPU work accelerates | verify exact venue/year |

## Recent IM — the "2003 is old" update

| Key | Work | Supports | Status |
|---|---|---|---|
| eim2025 | eIM: GPU-Accelerated Efficient Influence Maximization, SC'25 Workshops, doi 10.1145/3731599.3767442 | that current IM scaling work still assumes the monotone-submodular structure | verify |
| imsurvey2026 | Exploring Influence Maximization: State-of-the-Art Methods, Taxonomies, and Trends, ACM TKDD, doi 10.1145/3779058 | taxonomy/currency of the field; supports the "the whole lineage assumes X" claim | verify |
| temporalreview2023 | Influence maximization on temporal networks: a review, arXiv:2307.00181 | temporal IM exists and is active | verify, check for journal version |
| aamas2023temporal | Being an Influencer is Hard: The Complexity of Influence Maximization in Temporal Graphs with a Fixed Source, AAMAS 2023 | AAMAS precedent for this exact style of complexity paper — venue-fit evidence | verify |
| dtinf2026 | DTInf: Dynamic Topic-Aware Influence Maximization with Incremental Embedding Updates, ACM WebSci 2026, doi 10.1145/3795766.3799740 | current dynamic IM treats evolution as exogenous | verify |
| temprl2026 | TempRL-IM — temporal IM via continuous-time GNNs + deep RL, Scientific Reports 2026 | same | verify |
| seaiu2026 | SEAIU — incremental updating for dynamic IM, Intl. J. Machine Learning & Cybernetics 2026, doi 10.1007/s13042-026-03077-6 | same | verify |
| hotlink | HoTLink — forecast-driven temporal IM in streaming settings (SSRN preprint) | exogenous-forecast framing, explicitly | verify; preprint only — check for peer-reviewed version |

⚠️ The claim "no existing temporal-IM work models *endogenous*
topology" is a **novelty/absence claim**. It cannot be supported by a
citation. It needs a documented search (queries, databases, dates) plus
the named nearest prior work, or it must be softened. Do not let it
into the manuscript as a bare assertion.

## Non-monotone / non-submodular optimization

| Key | Work | Supports | Status |
|---|---|---|---|
| feige2011 | Feige, Mirrokni, Vondrák — Maximizing non-monotone submodular functions, FOCS 2007 / SICOMP 2011 | the approximation landscape once monotonicity goes | verify which version to cite |
| buchbinder2012 | Buchbinder, Feldman, Naor, Schwartz — double greedy, tight 1/2 for unconstrained non-monotone submodular | the best available if we could recover submodularity — and we cannot, which is the point | verify |
| iyerbilmes | Iyer, Bilmes — algorithms for difference-of-submodular (DS) optimization | the `rel − ρ` decomposition | verify |

## LLM context degradation — calibrates the saturation penalty

| Key | Work | Supports | Status |
|---|---|---|---|
| distracted2025 | How Is LLM Reasoning Distracted by Irrelevant Context? An Analysis Using a Controlled Benchmark, EMNLP 2025 (2025.emnlp-main.674, arXiv:2505.18761) | **the 43%→19% figure and the −55.89% average regression** — load-bearing for F1 | verify the exact numbers in the paper, not the snippet |
| contextrot | Context Rot: How Increasing Input Tokens Impacts LLM Performance (Chroma technical report) | degradation with input length on simple tasks | industry tech report, not peer reviewed — cite with that caveat or find a peer-reviewed equivalent |
| shi2023 | Shi et al. — Large Language Models Can Be Easily Distracted by Irrelevant Context, ICML 2023 | the original distraction result | verify |
| liu2024 | Liu et al. — Lost in the Middle, TACL 2024 | non-uniform use of the context window | verify |

⚠️ F1 is the paper's load-bearing empirical claim. If the 43%→19%
figure does not survive checking, or is model-specific in a way that
does not generalize, the saturation argument needs a different anchor.
**Check this one first.**
