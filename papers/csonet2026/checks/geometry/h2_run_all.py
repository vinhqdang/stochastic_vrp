"""Run every H2 verification with fixed seeds and print the verification table
(copied into H2_geometry.md).  Any failed assertion aborts with a traceback."""
import time
import h2_gap as G
import h2_line as L
import h2_affine as AF
import h2_extra as E

t0 = time.time()
rows = []


def add(claim, n, detail):
    rows.append((claim, n, detail))


# ---- gap theorems
tot = dict(T1=0, agree=0, star=0, T4=0, lem=0, T5=0, strict=0)
rmax = 0
for s in (11, 101, 102, 103, 104, 105):
    c = G.T1_T2_T4_T5(seed=s, trials=1000)
    tot["T1"] += c["T1"]; tot["agree"] += c["solvers_agree"]; tot["star"] += c["star_eq"]
    tot["T4"] += c["T4"]; tot["lem"] += c["T4_lemma"]; tot["T5"] += c["T5"]
    tot["strict"] += c["strict_gap"]; rmax = max(rmax, c["ratio_max"])
add("3 chain solvers (permutations / ordered-subset DFS / Held-Karp Pareto) agree", tot["agree"],
    "random Euclid/graph/line/star metrics, hazards uniform/front/tight, n<=6")
add("W_spoke <= W_chain <= |F| w_max <= n W_spoke (Thm 1)", tot["T1"],
    f"strict gap in {tot['strict']}, max ratio seen {rmax}")
add("star metric => W_chain = W_spoke (Thm 2, =>)", tot["star"], "all equal")
add("non-star pair => explicit 2-site gap instance (Thm 2, <=)", G.T3_converse(seed=3, trials=800)
    + G.T3_converse(seed=301, trials=800), "all W_spoke=1, W_chain=2")
add("lemma A_j <= B_j/kappa on random orders (j>=2) and equality at j=1", tot["lem"], "")
add("speed augmentation W_chain(1) <= W_spoke(1/kappa) (Thm 3)", tot["T4"], "")
add("bounds ceil((3rho-1)/2kappa), (floor(log2 rho)+1) ceil(5/2kappa) (Thm 4)", tot["T5"], "")
cls = sum(G.T5_classes(seed=s, trials=3000) for s in (2, 201, 202))
add("residue-class construction J_r spoke-feasible for h=B (proof of Thm 4)", cls,
    "random orders on Euclid/graph/line metrics, n<=12")
fam = G.T6()
add("extremal families (cluster ratio n, ray ratio n, uniform ratio n/(floor((n-1)k)+1))",
    len(fam), "exact brute force, n=2..7")
add("random search over tight-hazard instances: largest ratio found (equals n; bound far larger)",
    3000, str(G.adversarial_search()))
e1 = E.E1()
add("ray: minimal speed factor to cover all n -> 1/kappa from below (Thm 3 sharp)", len(e1),
    "r in {2,3,5}, n<=20; brute-force confirmation n<=6")
# ---- line
ok = sum(L.validate_dp(seed=s, trials=1000)[0] for s in (21, 401, 402))
add("line interval DP (weighted) and O(n^3) count DP == exhaustive", ok, "n<=7, int data")
add("one ray: optimum = sum of individually feasible weights", L.validate_one_ray(), "")
y = nn = 0
for s in (23, 501, 502):
    a, b = L.validate_reduction(seed=s, trials=80)
    y += a; nn += b
add("PARTITION -> weighted line (Thm 5): optimum == K*M + max{b(C)<=A/2}", y + nn,
    f"{y} yes / {nn} no; interval DP and independent Held-Karp agree for <=11 sites")
yf = nf = 0
for s_ in (24, 801, 802):
    a, b = L.validate_reduction_front(seed=s_, trials=80)
    yf += a; nf += b
add("same reduction with front-induced deadlines h=(3/2)|x|-(A+2) (radial front from depot, v=2/3, delay)", yf + nf,
    f"{yf} yes / {nf} no; DP and Held-Karp agree")
tr, worst = E.E2()
add("line FPTAS (value-indexed interval DP)", tr, f"worst ratio {worst:.4f} >= 1-eps")
# ---- affine
a0, a1 = AF.A0_A1()
add("subset/EDD solver == permutation brute force", a0, "")
add("alpha<=1: feasible <=> sum p - alpha p_max <= beta", a1 * 1, "all nonempty subsets")
st = {"yes": 0, "no": 0}
for s in (32, 601, 602, 603, 604):
    r = AF.A2(seed=s, trials=90)
    st["yes"] += r["yes"]; st["no"] += r["no"]
add("PARTITION -> MWHED, alpha in {1/10..1}, w=p (Thm C1)", st["yes"] + st["no"],
    f"{st['yes']} yes / {st['no']} no, alpha in 9 values")
st = {"yes": 0, "no": 0}
for s in (33, 701, 702):
    r = AF.A3(seed=s, trials=50, nmax=5)
    st["yes"] += r["yes"]; st["no"] += r["no"]
add("PARTITION -> MWHED, alpha>1, beta=0 (Thm C2)", st["yes"] + st["no"],
    f"{st['yes']} yes / {st['no']} no, alpha in {{6/5,3/2,2,5/2,3,9/2,11/2,7/5,13/10,4}}")
add("beta=0, alpha>1: geometric growth law T_k >= lambda T_{k-1}, |S| <= 1+log_lambda", AF.A4(), "")

print(f"{'claim':100s} | {'#checks':>8s} | detail")
print("-" * 140)
for c, n, d in rows:
    print(f"{c:100s} | {n:8d} | {d}")
print("\nCamp Fire (depot dist km, pair dist km, kappa_min, rho):", E.E3())
print(f"total runtime {time.time()-t0:.1f}s")
