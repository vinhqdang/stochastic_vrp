"""Where the objective is submodular, and where it stops being.

The measured saturation penalty rho is S-shaped: convex below a model's
effective length c*, concave above it (PROJECT.md §11.1). This script
checks what that shape does to the OPTIMIZATION structure, by exhaustive
enumeration over small ground sets.

Setting: one receiver holding S, with

    u(S) = rel(S) - rho( conf(S) ),     conf(S) = sum of per-item weights

Claims under test:

  C1  rho CONVEX, conf modular  =>  rho o conf is SUPERMODULAR,
      so -rho o conf is submodular, so u is SUBMODULAR (though
      non-monotone: rho can overwhelm rel).

  C2  rho CONCAVE, conf modular =>  rho o conf is SUBMODULAR,
      so -rho o conf is SUPERmodular, and u is generally NOT submodular.

  C3  complementarity in rel breaks submodularity regardless of rho.

  C4  conf NON-modular (distractors interacting) breaks C1 even with
      rho convex.

If C1 and C2 hold, the empirically measured inflection c* is not merely
where the admission bar stops rising -- it is exactly the boundary of
submodularity of the objective. That ties the measurement to the theory
rather than leaving them as separate observations.

Run: python3 structure_check.py
"""

from itertools import combinations


def subsets(ground):
    g = sorted(ground)
    return [frozenset(c) for r in range(len(g) + 1)
            for c in combinations(g, r)]


def is_submodular(F, ground, tol=1e-9):
    """Diminishing marginals: for S subset T, x outside T,
       F(S+x) - F(S) >= F(T+x) - F(T)."""
    worst = None
    for T in subsets(ground):
        for S in subsets(T):
            if S == T:
                continue
            for x in ground:
                if x in T:
                    continue
                mS = F(S | {x}) - F(S)
                mT = F(T | {x}) - F(T)
                gap = mT - mS
                if gap > tol and (worst is None or gap > worst[0]):
                    worst = (gap, set(S), set(T), x, mS, mT)
    return worst is None, worst


def is_monotone(F, ground, tol=1e-9):
    for T in subsets(ground):
        for S in subsets(T):
            if F(T) < F(S) - tol:
                return False
    return True


GROUND = {"a", "b", "c", "d"}
W = {"a": 1.0, "b": 1.4, "c": 0.8, "d": 1.2}          # confusability weights
VAL = {"a": 2.0, "b": 2.2, "c": 1.6, "d": 1.9}        # modular relevance


def conf(S):
    return sum(W[x] for x in S)


def rel_modular(S):
    return sum(VAL[x] for x in S)


def rel_submodular(S):
    """Coverage-like: concave in the modular total. Submodular, monotone."""
    return 3.0 * (1.0 - 2.718281828 ** (-0.8 * rel_modular(S)))


def rel_complementary(S):
    """Same, plus a hard AND-bonus on {a,b} -- an irreducible bundle."""
    return rel_submodular(S) + (1.5 if {"a", "b"} <= S else 0.0)


def rho_convex(x):
    return 0.35 * x ** 2


def rho_concave(x):
    return 3.0 * (x ** 0.5)


def conf_interacting(S):
    """Confusable load with interaction: similar distractors confuse
    superadditively, so conf is itself supermodular rather than modular."""
    return conf(S) + 0.5 * conf(S) ** 2 / (1 + len(S))


def report(name, F, expect):
    sub, w = is_submodular(F, GROUND)
    mono = is_monotone(F, GROUND)
    verdict = "SUBMODULAR" if sub else "NOT submodular"
    ok = "OK " if (sub == expect) else "!! "
    print(f"{ok}{name}")
    print(f"      {verdict}; {'monotone' if mono else 'NON-monotone'}"
          f"   (expected {'submodular' if expect else 'not submodular'})")
    if w:
        gap, S, T, x, mS, mT = w
        print(f"      worst violation: S={S or '{}'} T={T} x={x} -> "
              f"marg(S)={mS:.3f} < marg(T)={mT:.3f}")


if __name__ == "__main__":
    print(__doc__.split("Run:")[0].rstrip())
    print()

    print("C1  rho convex, conf modular, rel submodular")
    report("    u = rel_sub - rho_convex(conf)",
           lambda S: rel_submodular(S) - rho_convex(conf(S)), True)

    print("\nC2  rho concave (above c*), conf modular, rel submodular")
    report("    u = rel_sub - rho_concave(conf)",
           lambda S: rel_submodular(S) - rho_concave(conf(S)), False)

    print("\nC3  complementarity in rel, rho convex")
    report("    u = rel_compl - rho_convex(conf)",
           lambda S: rel_complementary(S) - rho_convex(conf(S)), False)

    # C4 did NOT reproduce with this particular interaction form: the
    # composite stayed submodular. Recorded as INCONCLUSIVE rather than
    # quietly dropped -- a stronger interaction may still break C1, but
    # this instance does not demonstrate it, so the paper must not claim
    # non-modular confusability breaks submodularity on this evidence.
    print("\nC4  rho convex but conf NON-modular (interacting distractors)"
          "   [INCONCLUSIVE]")
    report("    u = rel_sub - rho_convex(conf_interacting)",
           lambda S: rel_submodular(S) - rho_convex(conf_interacting(S)),
           True)

    print("\nControl: the penalty terms alone")
    sub_cx, _ = is_submodular(lambda S: rho_convex(conf(S)), GROUND)
    sub_cc, _ = is_submodular(lambda S: rho_concave(conf(S)), GROUND)
    print(f"    rho_convex  o conf : "
          f"{'submodular' if sub_cx else 'supermodular/neither'}"
          f"   (expect supermodular)")
    print(f"    rho_concave o conf : "
          f"{'submodular' if sub_cc else 'supermodular/neither'}"
          f"   (expect submodular)")
