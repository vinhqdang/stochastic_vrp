# Replies for the Elsevier review portal (CAOR-D-26-01885, revision 2)

One reply per reviewer comment block of the second round, to paste into
"Reply to comment". The full point-by-point response is in the uploaded
response letter; section, table and proposition numbers refer to the
revised manuscript.

---

## Reply to Reviewer 2

We thank the reviewer for a second careful reading and for checking the code. The reviewer's reading of the implementation is correct on every point; we have aligned the claims with the evidence, corrected two conventions in the evaluation code and re-run the affected experiments. Details are in the attached letter.

1. BATON-cf. Section 3.5 now distinguishes the cumulative increment S_k from the physical load X_k, notes that unloading destroys no information (S_k is still observed), and explains that under factor dependence equal (k, X_k) can hide different predictions. As the reviewer found, BATON-cf fits its models on unreset sums and applies them to the reset load after a return; it is now described as an approximation that reduces, rather than removes, the dependence bias for the first return, and an exact treatment is identified as one on (k, X_k, S_k). Proposition 4 does not cover F_k(.) under dependence: we give a sufficient condition, call the isotonic form a modelling choice, and check it with the binned shape test (significant decreases in 2.2% of bin pairs under independence, the false-positive level, and 0.2% and 0.2% at rho = 0.6 and 0.9). The text also states that constancy under independence is a population property, so BATON-cf and BATON need not agree in finite samples.

2. Fresh-start bias. We agree that "nearly cancel" was not established and have withdrawn it. Section 3.5 defines the optimal F_k, the population value F^pop of the fitted downstream policy, and the in-sample estimate; F^pop >= F_k is stated as a population property. Table A.3 reports the two measurable differences with instance-level bootstrap intervals: the in-sample effect is +0.40% of the handoff price on Salhi-Nagy ([+0.2, +0.6]) and -0.01% on Dethloff ([-0.05, +0.04]); F^pop - F^ref is +0.09% ([-0.5, +0.6]) and +0.01% ([+0.00, +0.02]), which combines downstream suboptimality with reference error; on the city routes it is negative (-1.09%, [-1.7, -0.5]), a direct sign that the reference is only approximate. The conclusion is limited to the small change in savings when F is replaced (-0.3 points on Salhi-Nagy, [-0.77, -0.02]). The constant-F reference used for the bias columns and the state-conditional DP3_50k of the last column are now labelled separately, both as approximate references.

3. Day types and intervals. The abstract, introduction, Section 4.5 and conclusion now speak of known day types; the text states that days drawn from a fixed mixture remain exchangeable and that the experiment does not establish robustness to drift or serial dependence. The interval now resamples instances with both plans kept together (-0.5 points, [-0.7, -0.3]), and each gate is reported separately (Det +0.1, [-0.1, +0.3]; SAA -1.1, [-1.5, -0.8]). Wherever an interval contains zero, including the deliver-only comparison, we now write "no statistically detectable difference" instead of "tie".

4. Conventions and code. The simulator uses nominal loading; Section 3.1 now states this, reports that the day's deliveries exceed the loaded cargo on 47% of route-days, explains that such shortfalls are unpriced and policy-independent, and contrasts the realised-loading convention. The physical post-return state is now capped at an empty vehicle, -min(D_<=k, L0), and the physical-reset column was re-computed. The B <= 0 branch was activated for 16 routes that depart exactly full (1 Salhi-Nagy route, 15 city routes; no Dethloff, RL or city SAA routes). All evaluation code now keeps the physical slack B = 0, and every affected experiment was re-run: implementable-policy savings moved by at most 0.7 points, reference points by at most 1.7, and no ranking or conclusion changed. The repository states of the submitted versions are tagged baton-r1 and baton-r2.

(a) The over-triggering bound is now stated as a sufficient condition, not a necessary one, and Table 6 is interpreted with the qualification that the optimal rule is estimated from 50,000 paths.

(b) We added a grid-refinement and truncation check of the exact program for all 627 routes of Table 8. Single-route values move by up to 2.91% between 1,500 and 6,000 cells (median 0.4%), but the error affects the reactive, handoff-only and three-action values alike, so the pooled exact savings, and the percentages in the abstract, change by at most 0.01 points across all grids and truncations.

(c) Algorithm 1 now spells out the three action-specific updates (handoff price, estimated R_k + F_k on return, realised path cost on continuation), and the protocol notes that DP3_N splits its training sample while BATON reuses it, so the comparison reflects sample allocation as well as estimators.

---

## Reply to Reviewer 1

We thank the reviewer for the careful second reading and the encouraging assessment. The changes in this round, made in response to Reviewers 2 and 3, narrow several claims further and correct two conventions in the evaluation code; none of them changes a conclusion that the reviewer's comments concerned.

---

## Reply to Reviewer 3

We thank the reviewer for the four questions; each led to a clarification in the manuscript. Details are in the attached letter.

1. Bound vs reference point. The two words are not equivalent, and we now use them consistently. A clairvoyant policy is a bound for a class of policies when no policy in that class can do better on any day. Oracle-ho may only hand off, so it is a bound for the handoff-only policies but only a reference point for policies that can also return to the depot, which can prevent a breach more cheaply and exceed it (BATON does so on four of the six gates). Oracle3 is a bound for every policy. Table 1 now states for which policies each clairvoyant is a bound, Section 3.6 defines the terms, and the table captions use them in this sense.

2. SAA vs WDRO. The savings are percentages of each gate's own reactive cost, which differs by a factor of about five. The tighter WDRO threshold gives much safer plans: the reactive cost is 2.9 per plan-day on WDRO plans against 16.5 on SAA plans, and a breach occurs on 2.3% of days against 10.0%. In absolute terms BATON is far cheaper on WDRO plans (1.9 against 8.3), but there are fewer breaches to prevent and few in a thousand training days, the regime where BATON falls furthest short of more data (41.2% against 47.8% for the high-data program, versus 53.7% against 56.4% on SAA). The handoff-only clairvoyant saves a similar share on both gates (42.0% and 43.5%), hence the different margins. A new paragraph in Section 4.2 explains this.

3. Cheap vs dear prices. In absolute terms both cheaper prices are better: BATON's daily cost per plan is 52.7 at the defaults, 45.8 with cheap standby and 48.3 with cheap emergencies, against 60.9 and 57.7 with dear standby and dear emergencies. The savings in Table 11 are relative to the reactive cost of the same configuration, which itself moves with the emergency price (64.2 with cheap and 97.3 with dear emergencies, against 78.4); cheap emergencies leave less to save, while cheap standby makes interventions cheaper at an unchanged reactive cost. Table 11 now has two columns with the absolute daily costs, and Section 4.8 explains this.

4. F_ho. Yes, F_ho is the dispatch fee, the fixed charge for calling out a standby vehicle. The symbol is now named in the text of Section 3.1.
