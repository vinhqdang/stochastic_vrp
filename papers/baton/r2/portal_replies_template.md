# Replies for the Elsevier review portal (CAOR-D-26-01885, revision 2)

One reply per reviewer comment block of the second round, to paste into
"Reply to comment". The full point-by-point response is in the uploaded
response letter; section, table and proposition numbers refer to the
revised manuscript.

---

## Reply to Reviewer 2

We thank the reviewer for a second careful reading and for checking the code. The reviewer's reading of the implementation is correct on every point; we have aligned the claims with the evidence, corrected two conventions in the evaluation code and re-run the affected experiments. Details are in the attached letter.

1. BATON-cf. Section {r:sec:actions} now distinguishes the cumulative increment S_k from the physical load X_k, notes that unloading destroys no information (S_k is still observed), and explains that under factor dependence equal (k, X_k) can hide different predictions. As the reviewer found, BATON-cf fits its models on unreset sums and applies them to the reset load after a return; it is now described as an approximation that reduces, rather than removes, the dependence bias for the first return, and an exact treatment is identified as one on (k, X_k, S_k). Proposition {r:prop:monotone} does not cover F_k(.) under dependence: we give a sufficient condition, call the isotonic form a modelling choice, and check it with the binned shape test (significant decreases in {m:shapeFZero} of bin pairs under independence, the false-positive level, and {m:shapeFSix} and {m:shapeFNine} at rho = 0.6 and 0.9). The text also states that constancy under independence is a population property, so BATON-cf and BATON need not agree in finite samples.

2. Fresh-start bias. We agree that "nearly cancel" was not established and have withdrawn it. Section {r:sec:actions} defines the optimal F_k, the population value F^pop of the fitted downstream policy, and the in-sample estimate; F^pop >= F_k is stated as a population property. Table {r:tab:fresh} reports the two measurable differences with instance-level bootstrap intervals: the in-sample effect is {m:frSalhiInEff} of the handoff price on Salhi-Nagy ({m:frSalhiInEffCI}) and {m:frDethInEff} on Dethloff ({m:frDethInEffCI}); F^pop - F^ref is {m:frSalhiDsEff} ({m:frSalhiDsEffCI}) and {m:frDethDsEff} ({m:frDethDsEffCI}), which combines downstream suboptimality with reference error; on the city routes it is negative ({m:frCityDsEff}, {m:frCityDsEffCI}), a direct sign that the reference is only approximate. The conclusion is limited to the small change in savings when F is replaced ({m:frSalhiDeltaF} points on Salhi-Nagy, {m:frSalhiDeltaFCI}). The constant-F reference used for the bias columns and the state-conditional DP3_50k of the last column are now labelled separately, both as approximate references.

3. Day types and intervals. The abstract, introduction, Section {r:sec:robust} and conclusion now speak of known day types; the text states that days drawn from a fixed mixture remain exchangeable and that the experiment focuses on known day types, with drift and serial dependence left for future work. The interval now resamples instances with both plans kept together ({m:dtAwarePooledDelta} points, {m:dtAwarePooledCI}), and each gate is reported separately (Det {m:dtAwarePooledDetDelta}, {m:dtAwarePooledDetCI}; SAA {m:dtAwarePooledSaaDelta}, {m:dtAwarePooledSaaCI}). Wherever an interval contains zero, including the deliver-only comparison, we now write "no statistically detectable difference" instead of "tie".

4. Conventions and code. The simulator uses nominal loading; Section {r:sec:setting} now states this, reports that the day's deliveries exceed the loaded cargo on {m:stockoutAll} of route-days, explains that such shortfalls are unpriced and policy-independent, and contrasts the realised-loading convention. The physical post-return state is now capped at an empty vehicle, -min(D_<=k, L0), and the physical-reset column was re-computed. The B <= 0 branch was activated for 16 routes that depart exactly full (1 Salhi-Nagy route, 15 city routes; no Dethloff, RL or city SAA routes). All evaluation code now keeps the physical slack B = 0, and every affected experiment was re-run: implementable-policy savings moved by at most 0.7 points, reference points by at most 1.7, and no ranking or conclusion changed. The repository states of the submitted versions are tagged baton-r1 and baton-r2.

(a) The over-triggering bound is now stated as a sufficient condition, not a necessary one, and Table {r:tab:regret} is interpreted with the qualification that the optimal rule is estimated from 50,000 paths.

(b) We added a grid-refinement and truncation check of the exact program for all {m:gridRoutes} routes of Table {r:tab:exact}. Single-route values move by up to {m:gridCoarseThreeMax} between 1,500 and 6,000 cells (median {m:gridCoarseThreeMedian}), but the error affects the reactive, handoff-only and three-action values alike, so the pooled exact savings, and the percentages in the abstract, change by at most {m:gridPooledMaxDiff} points across all grids and truncations.

(c) Algorithm {r:alg:offline} now spells out the three action-specific updates (handoff price, estimated R_k + F_k on return, realised path cost on continuation), and the protocol notes that DP3_N splits its training sample while BATON reuses it, so the comparison reflects sample allocation as well as estimators.

---

## Reply to Reviewer 1

We thank the reviewer for the careful second reading and the encouraging assessment. The changes in this round, made in response to Reviewers 2 and 3, narrow several claims further and correct two conventions in the evaluation code; none of them changes a conclusion that the reviewer's comments concerned.

---

## Reply to Reviewer 3

We thank the reviewer for the four questions; each led to a clarification in the manuscript. Details are in the attached letter.

1. Bound vs reference point. The two words are not equivalent, and we now use them consistently. A clairvoyant policy is a bound for a class of policies when no policy in that class can do better on any day. Oracle-ho may only hand off, so it is a bound for the handoff-only policies but only a reference point for policies that can also return to the depot, which can prevent a breach more cheaply and exceed it (BATON does so on four of the six gates). Oracle3 is a bound for every policy. Table {r:tab:competitors} now states for which policies each clairvoyant is a bound, Section {r:sec:algorithms} defines the terms, and the table captions use them in this sense.

2. SAA vs WDRO. The savings are percentages of each gate's own reactive cost, which differs by a factor of about five. The tighter WDRO threshold gives much safer plans: the reactive cost is {m:reactWDRO} per plan-day on WDRO plans against {m:reactSAA} on SAA plans, and a breach occurs on {m:pemWDRO} of days against {m:pemSAA}. In absolute terms BATON is far cheaper on WDRO plans ({m:costWDROBaton} against {m:costSAABaton}), but there are fewer breaches to prevent and few in a thousand training days, the regime where BATON falls furthest short of more data ({m:svWDROBaton} against {m:svWDRODpThree} for the high-data program, versus {m:svSAABaton} against {m:svSAADpThree} on SAA). The handoff-only clairvoyant saves a similar share on both gates ({m:svWDROOrc} and {m:svSAAOrc}), hence the different margins. A new paragraph in Section {r:sec:headline} explains this.

3. Cheap vs dear prices. In absolute terms both cheaper prices are better: BATON's daily cost per plan is {m:csBaseBaton} at the defaults, {m:csSbLowBaton} with cheap standby and {m:csEmgLowBaton} with cheap emergencies, against {m:csSbHighBaton} and {m:csEmgHighBaton} with dear standby and dear emergencies. The savings in Table {r:tab:costsens} are relative to the reactive cost of the same configuration, which itself moves with the emergency price ({m:csEmgLowReact} with cheap and {m:csEmgHighReact} with dear emergencies, against {m:csBaseReact}); cheap emergencies leave less to save, while cheap standby makes interventions cheaper at an unchanged reactive cost. Table {r:tab:costsens} now has two columns with the absolute daily costs, and Section {r:sec:sensitivity} explains this.

4. F_ho. Yes, F_ho is the dispatch fee, the fixed charge for calling out a standby vehicle. The symbol is now named in the text of Section {r:sec:setting}.
