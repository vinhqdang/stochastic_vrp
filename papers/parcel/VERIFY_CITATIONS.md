# PARCEL — citation verification log

Verification pass run 2026-09-06. Most entries are now **verified and
moved to `references.bib`**. This file retains (a) the corrections
found, (b) what is still unverified, and (c) the claims that cannot be
fixed by citation at all.

---

## ⚠️ CORRECTIONS — errors found in the planning draft

### 1. The "−55.89% average regression" does not exist

The planning draft attributed an *"average regression of −55.89%"* to
`distracted2025`. **That number is not in the paper.** The string
`55.89` appears zero times in both arXiv v1 and v2, and the word
"regression" appears zero times. It came from a search-result snippet,
not the source.

**Action: never use it.** It has been removed from `PROJECT.md`. If a
similar figure is wanted, find its actual source first.

### 2. The 43% → 19% figure is real but far narrower than assumed

Verbatim from the paper: *"at a fixed reasoning depth of rs=5,
Grok-3-Beta's step accuracy drops from 43% with one irrelevant context
to just 19% under fifteen irrelevant context."*

That is **one model, one metric, one reasoning depth, on a synthetic
grade-school-math benchmark (GSM-DC) with injected distractors.** Using
it as a general effect would be a misrepresentation a referee catches
immediately.

**What to cite instead** — the paper's own cross-model claim, which is
weaker in magnitude but far stronger as evidence: *"all six models
exhibit a clear degradation in reasoning accuracy as the number of
irrelevant context increases"* (Grok-3-Beta, GPT-4.1, GPT-4o-mini,
LLaMA-3.3-70B, LLaMA-3.1-8B, LLaMA-3.2-1B), with GPT-4.1 declining more
steeply still, 26% → 2%.

### 3. IMM author order was wrong

`tang2015` is **Tang, Shi, Xiao** (SIGMOD 2015). "Tang, **Xiao**, Shi"
is the *different* SIGMOD 2014 TIM/TIM+ paper. Corrected in the .bib.

### 4. A Crossref metadata bug to not copy

Crossref's SICOMP record for `buchbinder2012` renders the third author
as "Seffi, Joseph". That is a bug — he is Joseph (Seffi) **Naor**. The
.bib carries the correct form with a warning comment.

### 5. Feige & Izsak DOI

The correct DOI is `10.1145/2422436.2422466`.
`10.1145/2422436.2422500` is a *different* ITCS'13 paper.

---

## 🔴 Counter-evidence — log it, do not hide it

At least one 2026 preprint reports large dense models holding
**97.5–98.5% accuracy under 15,000 words of distractors** — i.e. very
little degradation. Context degradation is **task- and
model-dependent**, and a referee may well raise exactly this against F1.

This does not sink the argument (PARCEL needs degradation to exist in
*some* regime, not universally), but the manuscript must acknowledge
heterogeneity rather than assert a universal law. **Find and verify
this preprint before drafting §2.**

---

## 🟡 Upgrade available — use it

`nolima2025` (NoLiMa, ICML 2025, PMLR 267:44554–44570) is a **stronger,
peer-reviewed** anchor than the Chroma `contextrot` tech report: 13
models claiming ≥128K context, **11 of 13 drop below 50% of their
short-context baseline at 32K**; GPT-4o falls 99.3% → 69.7%.

Re-anchor F1 on `nolima2025` + the `distracted2025` cross-model
statement; keep `contextrot` only as a labelled industry supplement.
No *2026* peer-reviewed result was found that beats NoLiMa (the 2026
hits are preprints/OpenReview submissions, not archival).

---

## 🔴 SCOOPING RISK — assess before drafting

**Shi & Lai, "Approximation algorithm of maximizing non-monotone
non-submodular functions under knapsack constraint," Theoretical
Computer Science 990:114409, 2024** (`shilai2024`) addresses
*non-monotone* AND *non-submodular* maximization under a *knapsack* —
structurally PARCEL's exact optimization setting, with ratios
parameterized by weak-submodularity/weak-monotonicity.

**This must be read in full before drafting.** PARCEL's novelty has to
live in the *model* (agent context saturation, the endogenous graph,
the derived admission price) and not in the abstract optimization
result, which may already be covered. If the paper claims a general
non-monotone-non-submodular-knapsack theorem, it risks being scooped by
a 2024 TCS paper.

Related finding: **the clean "weakly submodular under knapsack" theorem
assumed in the planning draft does not exist.** Chen, Feldman &
Karbasi (`chen2017weakly`) generalize beyond cardinality to
**matroids** — and a knapsack is not a matroid. So PARCEL must prove
its own knapsack result and cite `shilai2024` as the nearest prior
guarantee, rather than importing an off-the-shelf theorem.

---

## 🔴 Full-text verification pass, 2026-09-06 — SIX DISCREPANCIES

Every item below was opened and read in full text. Six errors were
found; one is serious.

1. ⚠️⚠️ **Levy et al. (ACL 2024) points the OTHER WAY.** Its
   padding-*type* arm compares *Similar* (resampled from the same task)
   against *Different* (Books Corpus) — and **dissimilar padding hurt
   MORE**. Verbatim: *"the drop for the different setup is mostly larger
   than for the similar one."* The earlier note asserted the opposite.
   **Never cite it for "similar distractors hurt more."** It is still
   good for the 0.92→0.68 length sweep and for its
   duplicated-relevant-padding arm. This is the most serious error found
   in the project so far and it was in an *archival, peer-reviewed*
   source — the kind a referee is most likely to know.
2. **arXiv:2606.06203 overstated.** "Near-perfect to <60% by varying
   only density" compares **three different benchmarks**. The clean
   within-benchmark sweep (Table 6) gives a **24-point swing**, and
   WordChecker is non-monotonic. Use the 24 points.
3. **arXiv:2603.15723's venue is unconfirmed.** It claims "Math AI 2026";
   no independent record exists, and the NeurIPS MATH-AI 2026 workshop
   deadline postdates the posting. Cite as preprint. Non-academic
   affiliations, n=200, and a suspiciously low SQuAD ceiling (0.635 for
   gpt-4.1 at the *shortest* setting) — do not let it carry a claim alone.
4. **arXiv:2608.29477's PDF header falsely reads "Published as a
   conference paper at COLM 2026."** It is a **non-archival workshop**
   paper (CBW @ COLM 2026), per its own arXiv comment.
5. **arXiv:2601.11564** — Mixtral-8x7B is **MoE, not dense**; the paper's
   97.5–98.5% sentence names only Llama and Qwen. Its 719.64% latency
   figure is a serving-infrastructure measurement over **5 sampled
   queries** with prefix caching disabled, not a model property.
6. **arXiv:2608.22140 and arXiv:2605.12366 support different mechanisms
   than the section they were filed under.** The first's "fragmentation"
   is subword-tokenization damage from typos; the second is a pure
   *length* result against a near-zero-context baseline. Neither is
   evidence for topical confusability.

**Verified-correct as attributed:** the MUDDLE hard-negative gaps
(0.030 / 0.041, p=0.016), the distractor-aware truncation sign flip,
the 2×–30× classifier figure, and the Dhara & Sheth 0.270 vs 0.065
multi-hop/single-span contrast.

## ⚪ The calibration-sweep citations — status after verification

These came from an assisted extraction pass, not from my own reading of
the records. **None may be cited until checked.** Grouped by the claim
they support (`PROJECT.md` §11).

| Key | Work | Supports |
|---|---|---|
| — | arXiv:2601.11564 — *Context Discipline and Performance Correlation* | the counter-evidence: dense models hold 97.5–98.5% under 15,000 words of generic filler; real cost is **latency** (~720% spike), not accuracy |
| — | arXiv:2606.06203 — *Dense Contexts Are Hard Contexts* | at **fixed ~12K tokens**, lexical density alone drives retrieval from near-perfect to <60% — the strongest single argument for confusability over length |
| — | arXiv:2608.29477 — MUDDLE (CBW workshop @ COLM 2026, non-archival) | length-matched hard negatives cost more than random docs of identical length |
| — | arXiv:2608.22140 — *Lexical Perturbations Disrupt LLM Reasoning* (EMNLP 2026) | "filler insertion has little effect… fragmentation, not prompt length, drives the loss" |
| — | arXiv:2608.03297 — *Distractor-Aware Truncation* | the sign flip: distractor-aware truncation makes frontier models flat-at-ceiling where naive truncation shows the textbook decay |
| — | arXiv:2605.12366 — *Classifier Context Rot* | frontier models miss dangerous actions 2×–30× more often after 800K tokens — rebuts "it's been fixed in 2026 models" |
| — | arXiv:2603.15723 — Dhara & Sheth, *Context-Length Robustness in QA* | the cleanest controlled design; the multi-hop vs single-span contrast (0.270 vs 0.065) |
| — | arXiv:2402.14848 — Levy, Jacoby & Goldberg, FLenQA (ACL 2024) | best-designed length sweep with a padding-*type* arm; **dataset and code released**, so points can be regenerated |

⚠️ **Peer-review status varies sharply here** — one EMNLP 2026 paper,
one ACL 2024 paper, one *non-archival workshop* paper, and several bare
preprints. Label each accordingly at point of use; do not let a preprint
carry a load-bearing claim without saying what it is.

## 🔴 Reconstructed data — never quote as published

The GSM-DC per-point values used in the §11 shape analysis were
**extracted from the source SVG of the paper's Figure 4**, not from a
published table. The extraction was validated against the only two
values the authors state in prose (Grok rs=5: 43→19 ✔; GPT-4.1 rs=5:
26→2 ✔), with roughly ±1 point reading error elsewhere.

**Usable for shape analysis. Never quotable as published values.** If
the manuscript needs those numbers, either request them from the
authors or regenerate them from the released benchmark — and if a
figure-derived number does appear anywhere, say plainly that it was
digitized from the figure.

## ⚪ Still unverified

| Key | Work | Note |
|---|---|---|
| temporalreview2023 | Influence maximization on temporal networks: a review, arXiv:2307.00181 | not checked; also look for a journal version |
| temprl2026 | TempRL-IM, Scientific Reports 2026 | not checked |
| seaiu2026 | SEAIU, Intl. J. ML & Cybernetics 2026, doi 10.1007/s13042-026-03077-6 | not checked |
| hotlink | HoTLink, SSRN preprint | not checked; preprint only — look for peer-reviewed version |
| — | the 2026 counter-evidence preprint above | must be found and verified |

---

## ⚠️ Not fixable by citation

The claim **"no existing temporal-IM work models *endogenous*
topology"** is a novelty/absence claim. No citation can support an
absence. It needs a **documented search** — queries, databases, dates —
plus the named nearest prior work, or it must be softened to a
search-bounded statement. **This search has not been performed.**

---

## ✅ Verification pass, 2026-09-07 — LLM-agent context reduction

Source of truth: the **arXiv API** (`export.arxiv.org/api/query`), read
directly, and for the ICML record the **PMLR v97 landing page's own
`citation_*` metadata**. Titles, author lists, first-posting dates and
arXiv `comment` fields were taken from those responses, not from search
snippets.

| Key | Record | Status |
|---|---|---|
| `harshaw2019` | Harshaw, Feldman, Ward, Karbasi, ICML 2019, PMLR 97:2634–2643 | ✅ verified; PMLR spells the title "…Maximization **beyond** Non-negativity" (lowercase), now matched |
| `bps2026` | Chen, Chen, Wang, Li, Huang — *Optimal Skill Selection for LLM Agents with Provable Bicriteria Guarantees*, arXiv:2608.19993v1, 2026-08-20 | ✅ verified; **preprint**, no venue in the comment field |
| `pacms2026` | Ghulyani, Singh, Bharadwaj, Nath, Goswami — *PACMS*, arXiv:2606.20047v2 | ✅ verified; **preprint** |
| `rcrrouter2025` | Liu et al. (15 authors) — *RCR-Router*, arXiv:2508.04903v3, first posted **2025**-08-06 | ✅ verified; **preprint**. Note the year: earlier notes filed it as 2026 |
| `phasetransition2026` | Liu, Kong, Pei — *Phase Transition for Budgeted Multi-Agent Synergy*, arXiv:2601.17311v2 | ✅ verified; **preprint**, 55 pages |
| `agentprune2024` | Zhang et al. — *Cut the Crap*, arXiv:2410.02506v1, 2024-10-03 | ✅ verified; **preprint**. The paper's title and its method name (AgentPrune) differ — cite the title, name the method in the note |
| `gtd2026` | Jiang et al. — *Guided Topology Diffusion*, arXiv:2510.07799v2 | ✅ verified; arXiv comment states **ACL 2026 Main** |
| `kvcomm2026` | Shi, Chiesa, Maguire, Kostic — *KVComm*, arXiv:2510.03346v3 | ✅ verified; arXiv comment states **ICLR 2026** |

### Two claims confirmed against the primary text

- **BPS's concession is real and quotable.** Its abstract says, verbatim,
  that redundant or poorly chosen skills "waste scarce context tokens
  and can even degrade performance", and its objective is stated as "a
  monotone submodular benefit minus context penalty" under "a hard
  token budget", with a bicriteria $(1-1/e,1)$ ratio. The §10.2
  positioning rests on exactly these, and they check out.
- **RCR-Router's "up to 30%" and its benchmarks check out** — HotpotQA,
  MuSiQue and 2WikiMultihop. That MuSiQue overlap is worth stating in
  the paper: our evaluation set is the same benchmark.

### BPS read in full, 2026-09-07

Model, Theorem 1, §4.3 proof and Appendix A read from the arXiv HTML of
v1. Two attributions in this project's notes were **wrong** and are
corrected in `PROJECT.md` §10.2: BPS's *objective* is non-monotone and
they say so (it is the *benefit* that is monotone submodular), and
their penalty coefficient is described by them as a **first-order**
per-token sensitivity, so "linearization" is the fair word, not
"error". The pre-emption that survives, and is now in `main.tex`: their
benefit is a concave-coverage form, additive across capability
dimensions, in which a separately-inert jointly-decisive pair has no
representation at all.

### Still outstanding
- `Cut the Crap` (2410.02506) is listed here at v1 only; check whether
  a peer-reviewed version now exists before camera-ready.
- The 8 calibration citations above remain unverified. None is
  load-bearing in `main.tex` as it stands.
