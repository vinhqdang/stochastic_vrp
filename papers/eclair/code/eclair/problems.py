"""Problem suite: CSPLib-style problem families with (a) a reference
CPMpy model builder, (b) an INDEPENDENT pure-Python spec checker used
by the Tier-A brute-force oracle, (c) instance generators at "micro"
(exhaustively enumerable) and "mid" (CP-SAT-only) scale, and (d) the
family's provably valid metamorphic relations (PROJECT.md §2 Tier B).

A spec's `build(params)` consults the reserved key `_flip` (set by the
op_flip mutation in mutations.py) to flip one constraint family's
comparison operator — the builder-level mutation that breaks
parameter-monotonicity MRs. All other mutations act on params alone.

MR convention: each MR is (name, fn(params, rng) -> (params2, cmp))
where cmp in {"ge", "le", "eq"} asserts opt(params2) CMP opt(params)
on raw objective values, a property of the SPEC (hence of any faithful
model). Infeasible optima are handled by the caller (probes.py) as
-inf for max-sense and +inf for min-sense problems.
"""

from __future__ import annotations

import copy
import itertools

import cpmpy as cp


def _aux_max(arr):
    """Return (extra_constraints, aux_var) with aux >= every arr entry."""
    z = cp.intvar(0, 10**6, name="auxmax")
    return [z >= a for a in arr], z


class Spec:
    def __init__(self, name, sense, gen, build, checker, objective, enum,
                 mrs, mut):
        self.name = name
        self.sense = sense              # 'max' | 'min'
        self.gen = gen                  # gen(rng, size) -> params
        self.build = build              # build(params) -> cp.Model
        self.checker = checker          # checker(params, assign) -> bool
        self.objective = objective      # objective(params, assign) -> int
        self.enum = enum                # enum(params) -> iter of assigns
        self.mrs = mrs                  # [(name, fn)]
        self.mut = mut                  # mutation targets, see mutations.py


# ── knapsack with conflicts (max) ────────────────────────────────────────────

def _kc_gen(rng, size):
    n = 12 if size == "micro" else 22
    ncf = 3 if size == "micro" else 8
    w = [rng.randint(1, 10) for _ in range(n)]
    v = [rng.randint(1, 10) for _ in range(n)]
    cap = max(1, int(0.4 * sum(w)))
    pairs = list(itertools.combinations(range(n), 2))
    rng.shuffle(pairs)
    return {"n": n, "w": w, "v": v, "cap": cap, "conflicts": pairs[:ncf]}


def _kc_build(p):
    x = cp.boolvar(shape=p["n"], name="x")
    load = cp.sum([x[i] * p["w"][i] for i in range(p["n"])])
    cons = [load >= p["cap"]] if p.get("_flip") == "cap" else [load <= p["cap"]]
    cons += [x[i] + x[j] <= 1 for (i, j) in p["conflicts"]]
    m = cp.Model(cons)
    m.maximize(cp.sum([x[i] * p["v"][i] for i in range(p["n"])]))
    return m


def _kc_check(p, a):
    if sum(a[i] * p["w"][i] for i in range(p["n"])) > p["cap"]:
        return False
    return all(a[i] + a[j] <= 1 for (i, j) in p["conflicts"])


def _kc_obj(p, a):
    return sum(a[i] * p["v"][i] for i in range(p["n"]))


def _kc_mr_cap_up(p, rng):
    q = copy.deepcopy(p)
    q["cap"] += rng.randint(1, 5)
    return q, "ge"


def _kc_mr_conflict_drop(p, rng):
    q = copy.deepcopy(p)
    if q["conflicts"]:
        q["conflicts"].pop(rng.randrange(len(q["conflicts"])))
    return q, "ge"


def _kc_mr_permute(p, rng):
    q = copy.deepcopy(p)
    perm = list(range(p["n"]))
    rng.shuffle(perm)
    q["w"] = [p["w"][perm[i]] for i in range(p["n"])]
    q["v"] = [p["v"][perm[i]] for i in range(p["n"])]
    inv = {perm[i]: i for i in range(p["n"])}
    q["conflicts"] = [tuple(sorted((inv[i], inv[j])))
                      for (i, j) in p["conflicts"]]
    return q, "eq"


KNAPSACK = Spec(
    "knapsack_conflicts", "max", _kc_gen, _kc_build, _kc_check, _kc_obj,
    enum=lambda p: itertools.product((0, 1), repeat=p["n"]),
    mrs=[("cap_up", _kc_mr_cap_up), ("conflict_drop", _kc_mr_conflict_drop),
         ("permute", _kc_mr_permute)],
    mut={"scalars": ["cap"], "vectors": ["w", "v"], "rows": ["conflicts"],
         "flips": ["cap"]},
)


# ── set cover (min) ──────────────────────────────────────────────────────────

def _sc_gen(rng, size):
    nelem = 8 if size == "micro" else 14
    nsets = 12 if size == "micro" else 20
    sets = [sorted(rng.sample(range(nelem), rng.randint(1, max(2, nelem // 3))))
            for _ in range(nsets)]
    for e in range(nelem):                       # every element covered >= 2x
        holders = [s for s in sets if e in s]
        while len(holders) < 2:
            s = sets[rng.randrange(nsets)]
            if e not in s:
                s.append(e)
                s.sort()
                holders.append(s)
    cost = [rng.randint(1, 10) for _ in range(nsets)]
    return {"nelem": nelem, "sets": sets, "cost": cost}


def _sc_build(p):
    y = cp.boolvar(shape=len(p["sets"]), name="y")
    cons = []
    for e in range(p["nelem"]):
        covers = cp.sum([y[s] for s in range(len(p["sets"]))
                         if e in p["sets"][s]])
        cons.append(covers <= 1 if p.get("_flip") == "cover" else covers >= 1)
    m = cp.Model(cons)
    m.minimize(cp.sum([y[s] * p["cost"][s] for s in range(len(p["sets"]))]))
    return m


def _sc_check(p, a):
    for e in range(p["nelem"]):
        if not any(a[s] and e in p["sets"][s] for s in range(len(p["sets"]))):
            return False
    return True


def _sc_obj(p, a):
    return sum(a[s] * p["cost"][s] for s in range(len(p["sets"])))


def _sc_mr_elem_drop(p, rng):
    q = copy.deepcopy(p)
    e = rng.randrange(q["nelem"])
    q["sets"] = [[x if x < e else x - 1 for x in s if x != e]
                 for s in q["sets"]]
    q["nelem"] -= 1
    return q, "le"


def _sc_mr_cost_up(p, rng):
    q = copy.deepcopy(p)
    q["cost"][rng.randrange(len(q["cost"]))] += rng.randint(1, 4)
    return q, "ge"


def _sc_mr_permute(p, rng):
    q = copy.deepcopy(p)
    perm = list(range(len(p["sets"])))
    rng.shuffle(perm)
    q["sets"] = [p["sets"][perm[s]] for s in range(len(p["sets"]))]
    q["cost"] = [p["cost"][perm[s]] for s in range(len(p["sets"]))]
    return q, "eq"


SETCOVER = Spec(
    "set_cover", "min", _sc_gen, _sc_build, _sc_check, _sc_obj,
    enum=lambda p: itertools.product((0, 1), repeat=len(p["sets"])),
    mrs=[("elem_drop", _sc_mr_elem_drop), ("cost_up", _sc_mr_cost_up),
         ("permute", _sc_mr_permute)],
    mut={"scalars": [], "vectors": ["cost"], "rows": ["sets"],
         "flips": ["cover"]},
)


# ── graph coloring, minimize number of colors (min) ─────────────────────────

def _gc_gen(rng, size):
    n = 5 if size == "micro" else 13
    dens = 0.45 if size == "micro" else 0.25
    edges = [e for e in itertools.combinations(range(n), 2)
             if rng.random() < dens]
    if not edges:
        edges = [(0, 1)]
    return {"n": n, "edges": edges}


def _gc_build(p):
    c = cp.intvar(0, p["n"] - 1, shape=p["n"], name="c")
    if p.get("_flip") == "edges":
        cons = [c[u] == c[v] for (u, v) in p["edges"]]
    else:
        cons = [c[u] != c[v] for (u, v) in p["edges"]]
    extra, z = _aux_max([c[i] for i in range(p["n"])])
    m = cp.Model(cons + extra)
    m.minimize(z + 1)
    return m


def _gc_check(p, a):
    return all(a[u] != a[v] for (u, v) in p["edges"])


def _gc_obj(p, a):
    return max(a) + 1


def _gc_mr_edge_drop(p, rng):
    q = copy.deepcopy(p)
    if q["edges"]:
        q["edges"].pop(rng.randrange(len(q["edges"])))
    return q, "le"


def _gc_mr_edge_add(p, rng):
    q = copy.deepcopy(p)
    non = [e for e in itertools.combinations(range(p["n"]), 2)
           if e not in set(map(tuple, p["edges"]))]
    if non:
        q["edges"].append(non[rng.randrange(len(non))])
    return q, "ge"


def _gc_mr_relabel(p, rng):
    q = copy.deepcopy(p)
    perm = list(range(p["n"]))
    rng.shuffle(perm)
    q["edges"] = [tuple(sorted((perm[u], perm[v]))) for (u, v) in p["edges"]]
    return q, "eq"


COLORING = Spec(
    "coloring", "min", _gc_gen, _gc_build, _gc_check, _gc_obj,
    enum=lambda p: itertools.product(range(p["n"]), repeat=p["n"]),
    mrs=[("edge_drop", _gc_mr_edge_drop), ("edge_add", _gc_mr_edge_add),
         ("relabel", _gc_mr_relabel)],
    mut={"scalars": [], "vectors": [], "rows": ["edges"],
         "flips": ["edges"]},
)


# ── generalized assignment (min) ─────────────────────────────────────────────

def _ga_gen(rng, size):
    T = 7 if size == "micro" else 12
    M = 3 if size == "micro" else 4
    p_ = [rng.randint(2, 8) for _ in range(T)]
    cap = [max(3, int(sum(p_) / M * 1.6)) for _ in range(M)]
    cost = [[rng.randint(1, 12) for _ in range(M)] for _ in range(T)]
    return {"T": T, "M": M, "p": p_, "cap": cap, "cost": cost}


def _ga_build(pr):
    x = cp.boolvar(shape=(pr["T"], pr["M"]), name="x")
    cons = [cp.sum([x[t, m] for m in range(pr["M"])]) == 1
            for t in range(pr["T"])]
    for m in range(pr["M"]):
        load = cp.sum([x[t, m] * pr["p"][t] for t in range(pr["T"])])
        cons.append(load >= pr["cap"][m] if pr.get("_flip") == "cap"
                    else load <= pr["cap"][m])
    mdl = cp.Model(cons)
    mdl.minimize(cp.sum([x[t, m] * pr["cost"][t][m]
                         for t in range(pr["T"]) for m in range(pr["M"])]))
    return mdl


def _ga_check(pr, a):
    for m in range(pr["M"]):
        if sum(pr["p"][t] for t in range(pr["T"]) if a[t] == m) > pr["cap"][m]:
            return False
    return True


def _ga_obj(pr, a):
    return sum(pr["cost"][t][a[t]] for t in range(pr["T"]))


def _ga_mr_cap_up(pr, rng):
    q = copy.deepcopy(pr)
    q["cap"][rng.randrange(q["M"])] += rng.randint(1, 5)
    return q, "le"


def _ga_mr_cost_up(pr, rng):
    q = copy.deepcopy(pr)
    q["cost"][rng.randrange(q["T"])][rng.randrange(q["M"])] += rng.randint(1, 4)
    return q, "ge"


def _ga_mr_permute(pr, rng):
    q = copy.deepcopy(pr)
    perm = list(range(pr["T"]))
    rng.shuffle(perm)
    q["p"] = [pr["p"][perm[t]] for t in range(pr["T"])]
    q["cost"] = [pr["cost"][perm[t]] for t in range(pr["T"])]
    return q, "eq"


GAP = Spec(
    "gen_assignment", "min", _ga_gen, _ga_build, _ga_check, _ga_obj,
    enum=lambda pr: itertools.product(range(pr["M"]), repeat=pr["T"]),
    mrs=[("cap_up", _ga_mr_cap_up), ("cost_up", _ga_mr_cost_up),
         ("permute", _ga_mr_permute)],
    mut={"scalars": [], "vectors": ["p", "cap"], "rows": [],
         "matrices": ["cost"], "flips": ["cap"]},
)


# ── weighted on-time job selection, single machine (max) ────────────────────

def _sj_gen(rng, size):
    n = 11 if size == "micro" else 14
    p_ = [rng.randint(1, 6) for _ in range(n)]
    d = [rng.randint(3, max(4, int(sum(p_) * 0.7))) for _ in range(n)]
    wt = [rng.randint(1, 10) for _ in range(n)]
    return {"n": n, "p": p_, "d": d, "wt": wt}


def _sj_build(pr):
    s = cp.boolvar(shape=pr["n"], name="s")
    cons = []
    for j in range(pr["n"]):
        earlier = [i for i in range(pr["n"]) if pr["d"][i] <= pr["d"][j]]
        load = cp.sum([s[i] * pr["p"][i] for i in earlier])
        body = (load >= pr["d"][j]) if pr.get("_flip") == "deadline" \
            else (load <= pr["d"][j])
        cons.append(s[j].implies(body))
    m = cp.Model(cons)
    m.maximize(cp.sum([s[j] * pr["wt"][j] for j in range(pr["n"])]))
    return m


def _sj_check(pr, a):
    for j in range(pr["n"]):
        if a[j]:
            load = sum(pr["p"][i] for i in range(pr["n"])
                       if a[i] and pr["d"][i] <= pr["d"][j])
            if load > pr["d"][j]:
                return False
    return True


def _sj_obj(pr, a):
    return sum(a[j] * pr["wt"][j] for j in range(pr["n"]))


def _sj_mr_deadline_up(pr, rng):
    q = copy.deepcopy(pr)
    q["d"][rng.randrange(q["n"])] += rng.randint(1, 4)
    return q, "ge"


def _sj_mr_weight_up(pr, rng):
    q = copy.deepcopy(pr)
    q["wt"][rng.randrange(q["n"])] += rng.randint(1, 4)
    return q, "ge"


def _sj_mr_permute(pr, rng):
    q = copy.deepcopy(pr)
    perm = list(range(pr["n"]))
    rng.shuffle(perm)
    for key in ("p", "d", "wt"):
        q[key] = [pr[key][perm[i]] for i in range(pr["n"])]
    return q, "eq"


SCHED = Spec(
    "ontime_selection", "max", _sj_gen, _sj_build, _sj_check, _sj_obj,
    enum=lambda pr: itertools.product((0, 1), repeat=pr["n"]),
    mrs=[("deadline_up", _sj_mr_deadline_up), ("weight_up", _sj_mr_weight_up),
         ("permute", _sj_mr_permute)],
    mut={"scalars": [], "vectors": ["p", "d", "wt"], "rows": [],
         "flips": ["deadline"]},
)


ALL_SPECS = [KNAPSACK, SETCOVER, COLORING, GAP, SCHED]
CALIB_SPECS = [KNAPSACK, COLORING]          # held-out calibration families
EVAL_SPECS = [SETCOVER, GAP, SCHED]         # evaluation families (disjoint)
