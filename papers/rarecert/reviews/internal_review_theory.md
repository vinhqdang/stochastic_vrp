# Referee report: RARECERT THEORY.md (R1, draft 1)

Checks done by hand plus two numerical checks (scripts chk.py, chk2.py in this directory).
Severity: FATAL / NEEDS FIX / MINOR.

## Summary

Prop 1, Lemma 2, Theorem 1, Prop 3 (modulo wording) and Lemma 4 are correct. The rate is
lambda_max(Theta^{1/2} C Theta^{1/2}) with the 1/2 convention right. The **single-shift threshold
in Theorem 2 is wrong**: the true threshold is lambda_other <= lambda*/4 (not /3), and the proof step
"the inner product is 0 for the disjoint end mode" is false. The hypothesis of Theorem 2(a)
"every other stage has lambda_k <= lambda*/3" is essentially never satisfiable (stage k=1 has
lambda_1 ~ lambda*). Prop 4's constant 0.26 does not reproduce. Theorem 3's conditions need
re-deriving (factor 4 not 3; retention/defensive logic).

## (1) Proposition 1 - correct
L_0 = D_0 (P_0 is the empty sum), L_L = P_L; D_k <= D_0, P_k <= P_L since demands >= 0. Fine.
MINOR: state d_i, p_i >= 0 (true for gamma) and k ranges 0..L.

## (2) Lemma 2 - correct, state more precisely
Upper-tail of gamma: -log(1-G(x)) = y - (kappa-1) log y + log Gamma(kappa) + o(1), y=x/theta.
Mills: -log PhiBar(z) = z^2/2 + log z + log sqrt(2 pi) + o(1). Equate:
x(z) = theta z^2/2 + theta (2 kappa - 1) log z + O(1) for z -> inf. So O(log) is right and the
coefficient is explicit (positive iff kappa > 1/2, i.e. cv < sqrt 2).
z <= 0: Phi(z) <= 1/2 so 0 <= x(z) <= G^{-1}(1/2) = x(0) = median (not the mean). Correct as stated.
For the upper bound state it as: x(z) <= theta z_+^2/2 + C(1+log(1+z_+)) for ALL real z (the z<0 branch
is absorbed into the constant). MINOR: for z in [0, small] the "O(log(2+z))" must be read as
two-sided bounded, and the two-sided form is only used inside the ball z>0 for the lower bound.
MINOR (practical, not asymptotic): for cv ~ 0.3 (kappa ~ 11) and Z <= 5 (the 1e-3..1e-6 range) the
quantile is in the Wilson-Hilferty (cubic-in-z, moderate-deviation) regime, far from theta z^2/2.
The asymptotic scales lambda are rates at c -> inf, and may mispredict at finite gamma. Say so.

## (3) Theorem 1 - correct; proof needs the cleanups below
- Conventions verified. Z ~ N(0,C), C = B B' with B = [sqrt(rho) 1 | sqrt(1-rho) I]. Q = (1/2) Z' Theta Z has
  eigenvalues mu_j/2 where mu_j = eig(Theta^{1/2} C Theta^{1/2}) (similarity AB vs BA). P(Q > c) = exp(-c/mu_max (1+o(1))).
  Check n=1: D = theta Z^2/2 gives P ~ exp(-c/theta) and lambda = theta. OK.
  E exp(tQ) = det(I - t Theta^{1/2} C Theta^{1/2})^{-1/2}, finite iff t < 1/lambda. OK.
- Secular equation: Theta^{1/2} C Theta^{1/2} = (1-rho) Theta + rho s s', s_i = sqrt(theta_i), so
  1 = rho sum theta_i/(lambda - (1-rho) theta_i). Correct. Useful bounds: lambda >= rho sum theta_i + (1-rho) theta_min
  and lambda <= rho sum theta_i + (1-rho) theta_max.
- Remainders: handled correctly. P(R > eta c) with R <= C sum log(2+|Z_i|) is doubly exponentially small.
  Needed also: second-moment version (see Thm 2) via Cauchy-Schwarz.
- Independence of delivery/pickup blocks: correct; abscissa of the product MGF is 1/max(lambda^d, lambda^p). The text's
  "rate is the smaller of the two" means the smaller of 1/lambda's: fine but write it as lambda_k = max(...).
- Lower bound direction: the correct INPUT-space direction is u* = B' Theta^{1/2} v / sqrt(lambda) (|v|=1),
  NOT left undefined. Check: (B' Theta B)(B' Theta^{1/2} v) = B' Theta^{1/2}(Theta^{1/2} B B' Theta^{1/2} v) = lambda B'Theta^{1/2}v;
  |B'Theta^{1/2}v|^2 = lambda. Components: W-coordinate sqrt(rho) sum sqrt(theta_i) v_i/sqrt(lambda) > 0,
  E_i-coordinates sqrt(1-rho) sqrt(theta_i) v_i/sqrt(lambda) > 0. Latent point Z = s C Theta^{1/2} v/sqrt(lambda) > 0 entrywise, and
  Q(s u*) = lambda s^2/2. So s = (1+eta) sqrt(2c/lambda) gives Q = (1+eta)^2 c. Ball of radius O(1): Q changes by
  O(sqrt c), all Z_i stay > 0, so D > c in the ball for large c; Gaussian mass exp(-s^2/2 - O(s)). OK (NEEDS FIX only in
  that u* must be written down; the text's "(1+eta) sqrt(2c/lambda)" is fine).
  Edge case rho = 0: Perron vector is nonnegative not positive; ball argument needs one line.
- Cleaner proof (suggest): contraction principle. u/sqrt(c) satisfies a Gaussian LDP (speed c, rate |u|^2/2); the map
  u -> (1/2)|Theta^{1/2}(Bu)_+|^2 is continuous and positively 2-homogeneous; rate = inf{|u|^2/2: h_+(u) >= 1} = 1/lambda
  by the Perron argument; the remainder R is exponentially negligible. Gives upper and lower bound simultaneously and removes Chernoff/ball.
- Interlacing/strictness: lambda^d_k non-increasing, lambda^p_k non-decreasing by Cauchy interlacing; strictness for
  proper principal submatrices holds because Theta^{1/2} C Theta^{1/2} has all entries > 0 for rho > 0 (irreducible; Perron-Frobenius).
  So lambda_k < lambda* for 0<k<L. Correct (state rho > 0 and theta_i > 0). BUT the sentence "for long routes lambda_k is close
  to lambda* for k near 0 or L" is exactly what kills Theorem 2(a)'s hypothesis (see 6).
- Severity: Theorem 1 NEEDS FIX only in presentation; statement true.

## (4) Proposition 3 - true on the log scale; wording NEEDS FIX
log P(L_k>c) = -c/lambda_k(1+o(1)) and log P(F>c) = -c/lambda*(1+o(1)) give a ratio exp(-c(1/lambda_k - 1/lambda*) + o(c)).
Writing "(1+o(1))" inside the exponent is legitimate only for a FIXED gap lambda_k < lambda* as c->inf. As the ratio
varrho -> 1 the gap vanishes and prefactors (powers of c, multiplicity, constants) decide the share; asymptotics say nothing.
Restate: for fixed varrho<1, share = gamma^{1/varrho - 1 + o(1)}; for varrho = 1 the share is Theta(1) times a ratio of
prefactors. The algebra c = lambda* log(1/gamma) -> gamma^{lambda*/lambda_other - 1} is right. Also note that interior stages k=1,2,...
are not separate "modes" (their dominating points nearly coincide with stage 0 or L); lambda_other is the other END block.
The empirical statement (varrho <= 1/2 negligible bias, bias <= ~gamma) is consistent with this; but see (6): VARIANCE inflation
starts at varrho > 1/4, bias only matters for varrho close to 1.

## (5) Lemma 4 - correct as proved; two remarks
Case a >= 0: c <= h_+(u) <= h(u) = (1/2)(lambda1 a^2 + u_perp' M u_perp) (cross term vanishes, v1 eigenvector)
<= (1/2)(lambda1 a^2 + lambda2 b^2) => b^2 >= kappa (r^2 - a^2). |u+theta|^2 = (a+r)^2 + b^2 >= f(a) = (1-kappa) a^2 + 2ar + (1+kappa) r^2,
concave for kappa >= 1; f(0) = (1+kappa) r^2, f(r) = 4 r^2; min on [0,r] is 4r^2 iff kappa >= 3; a >= r gives (a+r)^2 >= 4 r^2. Verified.
Case a < 0: A >= 0 and v1 >= 0 => A v1 >= 0 => A u <= A u_perp entrywise => (Au)_+ <= (Au_perp)_+ => h_+(u) <= h_+(u_perp)
<= h(u_perp) <= (1/2) lambda2 b^2 => b^2 >= kappa r^2. Then |u+theta|^2 = (a+r)^2 + b^2 >= b^2 >= kappa r^2 >= 4 r^2 iff kappa >= 4. Verified.
So the lemma's overall condition is kappa >= 4 (kappa >= 3 only suffices for a >= 0). Correct as stated.
This a<0 case is NOT a technicality: it corresponds to points with negative inner product with the shift (u = -theta + ...), which is
what breaks Theorem 2 (see 6). It is attained for independent blocks (necessity), so 4 is the sharp constant there.
Needs fix: (i) theta is used both for the shift and for the gamma scale: rename shift to m or tau. (ii) Perron v1 >= 0 exists for
M = A'Theta A with A >= 0 (A v1 >= 0 uses v1 >= 0); (iii) the statement is for h_+(u) >= c; the real event is h_+(u) + R(u) >= c, so apply with
c(1-eta) and handle {R > eta c} by Cauchy-Schwarz: E_p[w 1_B] <= (E_p w^2)^{1/2} P(B)^{1/2} = e^{3|tau|^2/2} P(B)^{1/2}, and P(B) is doubly exponentially small.
One-factor block claim: lambda2 <= (1-rho) max theta_i is correct (Weyl: lambda_2(D + rho s s') <= lambda_1(D)); M and M~=(1-rho)Theta+rho ss' share
nonzero eigenvalues. But the sufficient condition "kappa >= 4 as soon as rho sum theta >~ 3 (1-rho) max theta" is WRONG in general: with
lambda1 >= rho sum theta + (1-rho) theta_min, kappa >= 4 is guaranteed when rho sum theta_i + (1-rho) theta_min >= 4 (1-rho) theta_max
(the "3" is exact only for equal theta: kappa = 1 + rho n/(1-rho); the example n=2, rho=0.6 gives kappa = 1.6/0.4 = 4 exactly, borderline, OK).
Severity: NEEDS FIX (constant 3 -> condition above).

## (6) Theorem 2

(b) FATAL (statement wrong; correct statement is stronger).
The proof evaluates the integrand at a ball around theta_o with u_d = 0. But the event {P_L > c} does not constrain u_d, so
m2(theta*) >= E_p[w 1{P_L>c}] = E_p[w(u_d)] * P(P_L > c) = e^{|theta*|^2} P(P_L>c)  (E_p[w] = e^{|theta|^2}, NOT 1).
Hence log m2 >= 2c/lambda* - c/lambda_o (1+o(1)), and
   log m2 - 2 log P(A) >= c (4/lambda* - 1/lambda_o)(1+o(1)).
The single shift is NOT log-efficient as soon as lambda_o > lambda*/4. The paper's c(3/lambda* - 1/lambda_o) and threshold /3 are wrong.
(The displayed line "log m2 >= -c/lambda_o + 2c/lambda*" contains an inconsistency too: the pointwise value at u=(0,theta_o) is
exp(c/lambda* - c/lambda_o); the true mass sits at u_d ~ -theta*.) The exact one-line proof needs no ball and no Theorem 1 lower bound beyond P(P_L>c).
Numerical confirmation (2-D toy: D = x_+^2/2, P = th_p y_+^2/2, shift (sqrt(2c),0), exact quadrature): th_p/th_d = 0.3 (between 1/4 and 1/3):
excess log m2 - 2 log P grows with slope 0.67 per unit c (= 4 - 1/0.3 = 0.667); paper predicts efficient (negative). th_p/th_d = 0.45: slope 1.78 (= 4 - 2.22). th_p/th_d = 0.2: excess ~ O(log c) (efficient).
Also (b)'s "ball of radius O(1) around theta_o lies in A": true only if the centre is inflated to (1+eta) theta_o (boundary otherwise); moot now.
Consequence for the narrative: single-shift variance blows up from varrho > 1/4, but its bias is the neglected mass gamma^{1/varrho - 1}, which is tiny
unless varrho is near 1. Both statements should appear.

(a) NEEDS FIX (hypotheses essentially unsatisfiable; proof step false).
Missing, exactly:
 1. The inner-product step. The text says <u,theta*> >= 0 "holds for the disjoint end mode, where it is 0". False: <u,theta*> = <u_d,theta*> ranges over R on the pickup-mode region
    (u_d free). The worst points are u = (-theta*, theta_o): |u+theta*|^2 = |theta_o|^2 = 2c/lambda_o must be >= 4|theta*|^2 = 8c/lambda*, i.e. lambda_o <= lambda*/4. This is the a<0 branch of Lemma 4.
 2. The hypothesis "every other stage block has lambda_k <= lambda*/3" fails for stage k=1 (and k=2,...): lambda_1^d >= rho sum_{i>1} theta_i vs
    lambda_0^d <= rho sum theta_i + (1-rho) theta_max, so lambda_1/lambda* -> 1 as n grows. The condition is vacuous. Mixed (delivery+pickup) stage points are also not treated.
Cleanest sufficient condition (proof is a corollary of Lemma 4, no stage analysis): since D_k <= D_0 and P_k <= P_L,
   A = {F>c} subset {D_0 + P_L > c} subset {h_+(u) + R(u) >= c}   with  h_+(u) = (1/2)|Theta^{1/2}(Au)_+|^2,  A = blockdiag(B_d, B_p) >= 0.
Apply Lemma 4 to the block-diagonal form M = diag(M_d, M_p): lambda1 = lambda* (dominant end block), lambda2 = max(lambda_2(dominant block), lambda_other end block).
Statement: if lambda* is attained by one end block only and
   lambda_other <= lambda*/4  and  lambda_2(dominant block) <= lambda*/4  (e.g. 4(1-rho) theta_max <= rho sum theta_i + (1-rho) theta_min for the dominant block),
then the shift tau* = sqrt(2c/lambda*) u* is log-efficient; if lambda_other > lambda*/4, it is not (part (b)). So the sharp end-mode criterion is lambda_other <= lambda*/4.
Middle stages and mixed points are all inside the joint-form bound; no hypothesis on them. Remainders via Cauchy-Schwarz as in (5)(iii). Second moment identity m2 = e^{|tau|^2}(2pi)^{-n/2} int_A e^{-|u+tau|^2/2} du is right.
Equivalent general criterion (Glasserman-Wang style): log-efficient iff inf_A (|u|^2/2 + <u,tau>) >= (3/2)|tau|^2.

## (7) Theorem 3 and Proposition 4

Theorem 3: outline right (w <= w_m/a_m on the region assigned to m; with a_m proportional to e^{-c/lambda_m}, term_m ~ e^{-c/lambda*-c/lambda_m} <= e^{-2c/lambda*}, correct). Needs fix:
- (ii) must be factor 4, not 3: Lemma 4 on the joint stage block needs kappa = lambda_dominant/max(lambda_2(dom), lambda_nondom) >= 4.
  Note however that the points can be ASSIGNED freely to components (pickup-end points to the pickup component), so the factor-4 condition is only needed where mixed points are assigned to one component.
- The defensive component must carry a weight bounded below (constant or polynomial), not a_def proportional to exp(-c/lambda_def); then w <= 1/a_def and a stage with lambda_k <= lambda*/2 contributes <= e^{-c/lambda_k}/a_def <= e^{-2c/lambda*}.
  So the retention cut in (iii) is lambda*/2, not /3 (/3 is harmless but wasteful; it retains ~L near-duplicate components with nearly identical directions).
- Interior stages near the ends have directions almost parallel to the end mode; fine, but the proof should go through the end-block reduction of (6) rather than per-stage Lemma 4.
- In the linear regime (equal theta) the crossing stage has lambda_x ~ lambda* varrho/(1+varrho) <= lambda*/2, so tie arcs there are covered by the defensive component. Genuine arc trouble only arises for stages with lambda_k > lambda*/2, i.e. heterogeneous routes. Worth stating: it explains why simple two-end SMIS works empirically.
- Status: keep PROVED-SKETCH; remove "PROVED" language elsewhere.

Proposition 4: conclusion qualitatively right (a finite mixture of fixed block directions cannot be efficient when a stage has a tie whose scale is large enough), but the constant is NOT reproducible and "PROVED" is premature ("computation in the notes").
My computation (isotropic tie, event |u|^2 > 2c/lambda_k, components at the two block directions, mixing at angle phi: |u-tau|^2 = 2R^2(1-cos phi), worst phi = 45 degrees, second-moment exponent -R_k^2 max(cos,sin) + log(1/a_m)); with x = lambda_k/lambda*:
  * equal weights, no defensive: fails iff x > 1/sqrt2 = 0.707 (eta < 0.293);
  * weights proportional to e^{-c/lambda_m}: fails iff x > sqrt2 - 1 = 0.414 (eta < 0.586);
  * same plus a defensive N(0,I) with constant weight: fails iff x > 1/2 (eta < 0.5).
Numerics (chk2.py) agree with all three. 0.26 (x = 0.74) matches none; at x=1 the exponent is -1/sqrt2 times the needed. Fix: state the mixture precisely (weights, defensive) and give the threshold above; failure for x > 1/2 with defensive weight.
The "arc mixture restores efficiency up to polynomial factor" is plausible (continuum of dominating points); cite Dupuis-Wang rather than claim.

## (8) Relation to the literature (what is new)
Not verified in full text; only the Glasserman-Wang abstract was retrieved (Ann. Appl. Probab. 7(3):731-746; examples with more than one way for the event to occur where large-deviations twisting performs poorly, with alternative estimators that weight the lower-probability paths). Statements below are my recollection of those papers; mark VERIFY.
- Sadowsky & Bucklew (1990, IEEE Trans. Inf. Theory 36(3)): exponential twisting is asymptotically efficient under a unique dominating point plus a geometric condition on the event. Single block case of Lemma 4 / Thm 2(a) is that, in the Gaussian setting; here the event is nonconvex (h_+ >= c), so Lemma 4 is the needed self-contained geometric check.
- Glasserman & Wang (1997): single twist fails when the event has several ways to occur; threshold phenomena of Theorem 2(b) (two disjoint-input modes, shift to one inflates the second moment by e^{|tau|^2} P(other mode)) is a concrete instance. Not new in principle; new: closed-form threshold lambda_o <= lambda*/4 for this model, and its link to Perron/secular scales.
- Glasserman & Li (2005, Mgmt Sci 51(11)): two-step (factor shift + conditional exponential twist) for credit portfolios, large-portfolio asymptotics (n -> inf, losses ~ n), no maximum over stages, no quadratic-in-latent-normal tail. Their theorem is different asymptotics (c ~ n), so RARECERT's C2 is an adaptation, not a corollary. The structure here (gamma marginal through Gaussian copula => quadratic form in the Gaussian) is the delta-gamma structure of Glasserman-Heidelberger-Shahabuddin (2000, 2002) on VaR; cite and check which parts of Theorem 1 are special cases (rate of Gaussian quadratic form = largest eigenvalue).
- Dupuis & Wang (2004, Stoch. Stoch. Rep. 76; 2007, Math. Oper. Res. 32): large-deviation analysis of IS as a game; mixtures / minima of affine subsolutions give asymptotically optimal schemes when there are several minimizers; continuum of minimizers (tie arcs) needs state-dependent or continuum mixtures. SMIS and the arc mixture of Prop 4 are instances; "arc restores efficiency" is likely a known consequence.
- Owen, Maximov & Chertkov (2019, EJS 13): ALOE gives bounded relative error uniformly in rarity (relative variance <= (J-1)/N) when P(E_j) and sampling from P(.|E_j) are exact; here neither is exact, so only log-efficiency is claimable, which is a weaker, as the draft says. Do not claim bounded relative error.
Genuinely new (modest, theory-note level): (i) explicit scales lambda(S) via the secular equation, hence the stage/mode structure (two end modes, strict interlacing) for this routing peak; (ii) the sharp single-shift threshold lambda_other <= lambda*/4 and the relative-mass formula gamma^{lambda*/lambda_other - 1}, giving a closed-form diagnostic; (iii) the plan-level extension via shared factors; (iv) a closed-form (no pilot) mixture. The asymptotic theory otherwise reduces to standard Gaussian LDP/contraction and the known second-moment criterion. Position the paper as application plus diagnostic, not as new IS theory.

## Corrected statements to adopt
- Thm 1: lim_{c->inf} c^{-1} log P(F>c) = -1/lambda*, lambda* = max(lambda_0^d, lambda_L^p), lambda(S) = lambda_max((1-rho)Theta_S + rho ss'), s = Theta_S^{1/2} 1; input direction u* = B'Theta^{1/2}v/sqrt(lambda).
- Thm 2: with lambda_other the other end block, the single shift tau* is log-efficient if lambda_other <= lambda*/4 and lambda_2(dominant block) <= lambda*/4; it is not log-efficient if lambda_other > lambda*/4, with log m2 - 2 log P >= c(4/lambda* - 1/lambda_other)(1+o(1)).
- Lemma 4 unchanged; one-factor sufficient condition kappa >= 4 if rho sum theta + (1-rho) theta_min >= 4(1-rho) theta_max.
- Prop 3: for fixed varrho < 1; share = gamma^{1/varrho - 1 + o(1)}.
- Prop 4: replace 0.26 by the explicit threshold for the stated mixture (0.707 equal weights; 0.414 e^{-c/lambda} weights; 0.5 with defensive).
