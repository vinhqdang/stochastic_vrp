"""Exhaustive check that the nearest general framework's parameters degenerate.

Shi & Lai (Theoretical Computer Science 990:114409, 2024) give approximation
guarantees for maximizing non-monotone, non-submodular functions under a
knapsack constraint. Their Theorem 4 -- the general case, allowing negative
objective values -- requires the objective to be BOTH:

  Def 1  gamma1-weak submodular   (gamma1 >= 1):
         for U1 (strict subset of) U2 and x not in U2,
             F(U2 + x) - F(U2)  <=  gamma1 * [F(U1 + x) - F(U1)]

  Def 3  gamma2-weak supermodular (gamma2 >= 1):
         for U1 (strict subset of) U2 and x not in U2,
             gamma2 * (F(U2 + x) - F(U2))  >=  F(U1 + x) - F(U1)

A finite parameter fails to exist when a pair (U1, U2, x) forces an
inequality no finite multiplier can satisfy:

  Def 1 is unsatisfiable when marg(U1) <= 0 < marg(U2)
        -- no gamma1 makes a positive quantity <= gamma1 * (non-positive one).
  Def 3 is unsatisfiable when marg(U2) <= 0 < marg(U1)
        -- gamma2 * (non-positive) can never dominate a positive quantity.

This script enumerates every (U1, U2, x) triple on small ground sets and
reports whether each parameter can exist at all.

Result: the two phenomena PARCEL is about break one parameter each.
  - complementarity breaks gamma1, leaves gamma2 intact
  - saturation       breaks gamma2, leaves gamma1 intact
Theorem 4 needs both, so it applies to neither.

Run: python3 degeneracy_check.py
"""

from itertools import combinations


def subsets(ground):
    ground = sorted(ground)
    return [frozenset(c) for r in range(len(ground) + 1)
            for c in combinations(ground, r)]


def parameters_exist(F, ground):
    """Return (def1_ok, def1_witness, def3_ok, def3_witness)."""
    def1_ok = def3_ok = True
    w1 = w3 = None
    for U2 in subsets(ground):
        for U1 in subsets(U2):
            if U1 == U2:
                continue
            for x in ground:
                if x in U2:
                    continue
                m2 = F(U2 | {x}) - F(U2)
                m1 = F(U1 | {x}) - F(U1)
                if m1 <= 0 < m2:
                    def1_ok, w1 = False, (set(U1), set(U2), x, m1, m2)
                if m2 <= 0 < m1:
                    def3_ok, w3 = False, (set(U1), set(U2), x, m1, m2)
    return def1_ok, w1, def3_ok, w3


def report(name, F, ground):
    d1, w1, d3, w3 = parameters_exist(F, ground)
    print(f"\n{name}")
    print(f"  gamma1-weak submodular   (Def 1): {'exists' if d1 else 'DOES NOT EXIST'}")
    if w1:
        U1, U2, x, m1, m2 = w1
        print(f"      witness: U1={U1 or '{}'}, U2={U2}, x={x} -> "
              f"marg(U1)={m1}, marg(U2)={m2}")
    print(f"  gamma2-weak supermodular (Def 3): {'exists' if d3 else 'DOES NOT EXIST'}")
    if w3:
        U1, U2, x, m1, m2 = w3
        print(f"      witness: U1={U1 or '{}'}, U2={U2}, x={x} -> "
              f"marg(U1)={m1}, marg(U2)={m2}")


# (A) Pure complementarity: a hard AND-pair, no saturation anywhere.
#     Two facts worthless alone, jointly decisive.
def and_pair(S):
    return 1.0 if S >= {"f1", "f2"} else 0.0


# (B) Pure saturation: modular relevance, convex load penalty, no
#     complementarity anywhere. Each fact is individually useful; the
#     receiver degrades once loaded. Matches main.tex's Proposition 3
#     witness (a) exactly: F={a,b}, rel(S)=conf(S)=|S|, rho(0)=0,
#     rho(1)=0.5, rho(2)=3 -- giving marginals +0.5 then -1.5.
RHO = {0: 0.0, 1: 0.5, 2: 3.0}


def saturating(S):
    return len(S) - RHO[len(S)]


# (C) Combined instance -- direct sum of (A) and (B) over disjoint
#     ground sets, one per receiver. Added after a review caught that
#     "no single instance has both" is FALSE in general: nothing stops
#     one instance containing an (A)-style receiver and a (B)-style
#     receiver side by side, and the two violating triples embed
#     unchanged since neither touches the other receiver's items.
def combined(S):
    return and_pair(S & {"f1", "f2"}) + saturating(S & {"a", "b"})


if __name__ == "__main__":
    print(__doc__.split("Run:")[0].rstrip())
    report("(A) Pure complementarity -- hard AND-pair",
           and_pair, {"f1", "f2"})
    report("(B) Pure saturation -- modular relevance, convex penalty",
           saturating, {"a", "b"})
    print("\nEach phenomenon kills exactly one of the two parameters that")
    print("Theorem 4 requires simultaneously.")

    print()
    report("(C) Combined -- (A) and (B) on disjoint receivers, same instance",
           combined, {"f1", "f2", "a", "b"})
    d1, _, d3, _ = parameters_exist(combined, {"f1", "f2", "a", "b"})
    assert not d1 and not d3, "expected BOTH parameters to fail on the combined instance"
    print("\n(C) confirms: gamma1 and gamma2 can both fail on ONE instance,")
    print("not just on two separate ones -- a single instance is enough to")
    print("place it entirely outside Shi & Lai's Theorem 4.")
