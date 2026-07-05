"""B&C-exact with Fractional Separation (MIPNODE) + Strong interior cuts."""
import numpy as np, math, time
import gurobipy as gp
from gurobipy import GRB
from matheuristic_core import make_instance, alns, cvar, route_cvar_peak, valley_order
from rci_bound import rci_of_S

def min_veh_lb(S, inst):
    r = rci_of_S(tuple(S), inst)
    if r >= 2: return r
    vseq = valley_order(list(S), inst)
    if route_cvar_peak(vseq, inst) > inst.Qeff + 1e-9:
        return 2 
    return 1

def matheur_arcs(best):
    arcs=set()
    for r in best.routes:
        nodes=[0]+[c+1 for c in r]+[0]
        for i in range(len(nodes)-1): arcs.add((nodes[i],nodes[i+1]))
    return arcs

def solve(inst, omega_V, warm=None, timelimit=90):
    n=inst.n; V=range(n+1)
    m=gp.Model(); m.Params.OutputFlag=0; m.Params.LazyConstraints=1; m.Params.TimeLimit=timelimit
    m.Params.PreCrush=1  # REQUIREMENT FOR USER CUTS (MIPNODE)
    m.Params.MIPGap=1e-4
    x=m.addVars([(i,j) for i in V for j in V if i!=j], vtype=GRB.BINARY, ub=1)
    m.setObjective(gp.quicksum(inst.C[i,j]*x[i,j] for i in V for j in V if i!=j)
                   + omega_V*gp.quicksum(x[0,j] for j in range(1,n+1)), GRB.MINIMIZE)
    
    for j in range(1,n+1):
        m.addConstr(gp.quicksum(x[i,j] for i in V if i!=j)==1)
        m.addConstr(gp.quicksum(x[j,i] for i in V if i!=j)==1)
    m.addConstr(gp.quicksum(x[0,j] for j in range(1,n+1))==gp.quicksum(x[j,0] for j in range(1,n+1)))
    
    if warm is not None:
        for k in x: x[k].Start = 1.0 if k in warm else 0.0
        
    m._x=x; m._n=n; m._inst=inst; m._setcuts=0; m._seqcuts=0; m._fraccuts=0
    
    def cb(model, where):
        xx=model._x; nn=model._n; ins=model._inst; VV=range(nn+1)
        
        # -------------------------------------------------------------
        # 1. FRACTIONAL SEPARATION (USER CUTS)
        # -------------------------------------------------------------
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                rel = model.cbGetNodeRel(xx)
                
                # Heuristic: Fast Connected Components on heavy fractional edges
                adj = {i: [] for i in VV}
                for (i, j), val in rel.items():
                    if val > 0.2 and i != 0 and j != 0:
                        adj[i].append(j)
                        adj[j].append(i)
                        
                visited = set()
                for i in range(1, nn + 1):
                    if i not in visited:
                        comp = set()
                        q = [i]
                        visited.add(i)
                        while q:
                            u = q.pop(0)
                            comp.add(u)
                            for v in adj[u]:
                                if v not in visited:
                                    visited.add(v)
                                    q.append(v)
                        
                        if len(comp) > 1:
                            lb = min_veh_lb(comp, ins)
                            if lb >= 2:
                                flow_out = sum(rel[u, v] for u in comp for v in VV if v not in comp and (u, v) in rel)
                                if flow_out < lb - 0.05:  # Violated fractional cut
                                    model.cbCut(gp.quicksum(xx[u, v] for u in comp for v in VV if v not in comp and (u, v) in xx) >= lb)
                                    model._fraccuts += 1

        # -------------------------------------------------------------
        # 2. INTEGER SEPARATION (LAZY CUTS)
        # -------------------------------------------------------------
        elif where == GRB.Callback.MIPSOL:
            xv=model.cbGetSolution(xx)
            out={i:[j for (a,j) in xx if a==i and xv[a,j]>0.5] for i in VV}
            routes=[]; visited=set()
            for st in out[0]:
                seq=[]; cur=st; g=0
                while cur!=0 and g<=nn+1: seq.append(cur-1); visited.add(cur-1); cur=out[cur][0] if out[cur] else 0; g+=1
                if seq: routes.append(seq)
            rem=set(range(nn))-visited; subs=[]
            while rem:
                s=next(iter(rem)); comp=[]; cur=s+1; g=0
                while g<=nn+1:
                    comp.append(cur-1); rem.discard(cur-1)
                    if not out[cur]: break
                    cur=out[cur][0]
                    if cur==s+1 or cur==0: break
                    g+=1
                if comp: subs.append(comp)
            
            def enter(Sset): 
                Sn={c+1 for c in Sset}
                return gp.quicksum(xx[i,j] for i in VV for j in Sn if i not in Sn and (i,j) in xx)
                
            for S in subs:
                model.cbLazy(enter(S)>=min_veh_lb(S,ins))
                
            for seq in routes:
                S=set(seq)
                if route_cvar_peak(seq, ins) > ins.Qeff+1e-6:
                    lb=min_veh_lb(S,ins)
                    if lb>=2:
                        model.cbLazy(enter(S)>=lb); model._setcuts+=1
                    else:
                        arcs=[(0,seq[0]+1)]+[(seq[k]+1,seq[k+1]+1) for k in range(len(seq)-1)]+[(seq[-1]+1,0)]
                        model.cbLazy(gp.quicksum(xx[a,b] for (a,b) in arcs)<=len(arcs)-1); model._seqcuts+=1

    m.optimize(cb)
    return m.ObjVal, m.ObjBound, m.Status, m._setcuts, m._seqcuts, m._fraccuts

print("B&C-exact with FRACTIONAL SEPARATION + STRONG interior cuts:")
print(f"{'n':>3} {'UB':>8} {'opt':>8} {'bound':>8} {'gap%':>6} {'solved?':>8} {'setCut':>7} {'seqCut':>7} {'fracCut':>7} {'t':>6}")
for n in [30, 40, 50]:
    inst=make_instance(n=n, r=4, N=500, seed=1, eps=1.5, alpha=0.9, cap_frac=0.5)
    omega_V=inst.C[0,1:].mean()
    best,_=alns(inst, iters=1500, seed=1, omega_V=omega_V, verbose=False); ub=best.cost()
    warm=matheur_arcs(best)
    
    t0=time.perf_counter()
    opt,bound,status,sc,qc,fc=solve(inst, omega_V, warm=warm, timelimit=120)
    dt=time.perf_counter()-t0
    
    solved=status==GRB.OPTIMAL
    gap=100*(ub-bound)/ub if ub > 0 else 0
    print(f"{n:>3} {ub:>8.1f} {opt:>8.1f} {bound:>8.1f} {gap:>6.2f} {('YES' if solved else 'TIMEOUT'):>8} {sc:>7} {qc:>7} {fc:>7} {dt:>6.1f}")