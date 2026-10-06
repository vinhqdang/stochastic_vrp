import random, math, itertools, sys
INF=1e18
def metric(kind,n,rng):
    pts=[(0.0,0.0)]
    for i in range(n):
        if kind=='euclid': pts.append((rng.uniform(-10,10),rng.uniform(-10,10)))
        elif kind=='geo': 
            r=1.6**rng.uniform(0,8); th=rng.uniform(0,6.28); pts.append((r*math.cos(th),r*math.sin(th)))
        elif kind=='ray': pts.append((1.5**i*rng.uniform(.9,1.1),rng.uniform(-.3,.3)))
        elif kind=='cluster':
            c=rng.choice([(8,0),(0,12),(-9,-9)]); pts.append((c[0]+rng.gauss(0,.5),c[1]+rng.gauss(0,.5)))
    N=n+1
    d=[[math.dist(pts[i],pts[j]) for j in range(N)] for i in range(N)]
    return d
def chain_dp(d,w,h,sites,budget=None):
    # best[S][last]=min arrival; return per (S,last) feasible; opt weight
    n=len(sites); best={}
    for i,s in enumerate(sites):
        if d[0][s]<=h[s]+1e-12: best[(1<<i,i)]=d[0][s]
    res=0; layer=dict(best)
    allb=dict(best)
    frontier=list(best.items())
    for (S,l),B in frontier: pass
    # iterate by popcount
    cur=best
    for size in range(1,n+1):
        nxt={}
        for (S,l),B in cur.items():
            for j in range(n):
                if S>>j&1: continue
                Bn=B+d[sites[l]][sites[j]]
                if Bn<=h[sites[j]]+1e-12:
                    k=(S|1<<j,j)
                    if Bn<nxt.get(k,INF): nxt[k]=Bn
        allb.update(nxt); cur=nxt
        if not cur: break
    return allb
def wt(S,w,sites): return sum(w[sites[i]] for i in range(len(sites)) if S>>i&1)
def opt(d,w,h,n):
    sites=list(range(1,n+1)); a=chain_dp(d,w,h,sites)
    return max([wt(S,w,sites) for (S,l) in a]+[0])
def alg(d,w,lam,beta,n,theta,exact=True,rng=None):
    rho=lam-1; a=[d[0][i] for i in range(n+1)]
    P=math.ceil(2/theta)
    m=1
    while not (2*theta+4/rho <= (1-theta)*(2**m-1)): m+=1
    a0=beta/rho if beta>0 else 1.0
    cls={}
    for i in range(1,n+1):
        if beta>0 and a[i]<a0: k=-1
        else: k=math.floor(math.log2(a[i]/a0))
        cls.setdefault(k,[]).append(i)
    R={}
    for k,C in cls.items():
        alk=0 if k==-1 else a0*2**k
        fk=beta+rho*alk; Phi=theta*fk
        # exact oracle: max-weight route over class sites with excess<=Phi at every site (last suffices)
        h={i:a[i]+Phi for i in C}   # excess<=Phi  <=> B<=a+Phi
        sites=C; allb=chain_dp(d,w,{**{i:INF for i in range(n+1)},**h},sites)
        bestS=0;bestroute=None
        # reconstruct best by recomputing with parent tracking: simple brute reconstruct
        bw=0;bk=None
        for (S,l),B in allb.items():
            if wt(S,w,sites)>bw: bw=wt(S,w,sites); bk=(S,l)
        if bk is None: R[k]=(0,[]); continue
        # reconstruct route by backward search
        S,l=bk; route=[sites[l]]; B=allb[bk]
        while S&(S-1):
            S2=S&~(1<<l); found=False
            for j in range(len(sites)):
                if S2>>j&1 and (S2,j) in allb and abs(allb[(S2,j)]+d[sites[j]][sites[l]]-B)<1e-9:
                    route.append(sites[j]); S,l,B=S2,j,allb[(S2,j)]; found=True;break
            assert found
        route.reverse(); R[k]=(bw,route)
    ks=sorted(R); 
    # DP weighted independent set gap>=m
    best={}; 
    def f(idx):
        if idx<0: return (0,[])
        if idx in best: return best[idx]
        k=ks[idx]; j=idx-1
        while j>=0 and k-ks[j]<m: j-=1
        w1,s1=f(j); take=(w1+R[k][0],s1+[k]); skip=f(idx-1)
        best[idx]=take if take[0]>=skip[0] else skip; return best[idx]
    val,K=f(len(ks)-1)
    route=[]
    for k in K: route+=R[k][1]
    return route,P,m
def simulate(d,w,route,lam,beta):
    B=0;prev=0;tot=0;ok=True
    for s in route:
        B+=d[prev][s]; prev=s
        if B>beta+lam*d[0][s]+1e-9: ok=False
        else: tot+=w[s]
    return ok,tot
if __name__=="__main__":
    rng=random.Random(int(sys.argv[1]) if len(sys.argv)>1 else 1)
    viol=0;worst=9;cnt=0
    for it in range(int(sys.argv[2]) if len(sys.argv)>2 else 200):
        kind=rng.choice(['euclid','geo','ray','cluster']); n=rng.randint(4,9)
        d=metric(kind,n,rng); lam=rng.choice([1.05,1.2,1.5,2,3,5,12,40]); beta=rng.choice([0,0,0.5,3.0])
        w=[0]+[rng.choice([1,1,2,5,rng.randint(1,30)]) for _ in range(n)]
        theta=rng.choice([0.5,0.3,2/3,0.25])
        # keep only instances where each site individually protected? not required, but opt>0
        h={i:beta+lam*d[0][i] for i in range(n+1)}
        o=opt(d,w,h,n)
        route,P,m=alg(d,w,lam,beta,n,theta)
        ok,tot=simulate(d,w,route,lam,beta)
        cnt+=1
        if not ok: viol+=1; print("INFEASIBLE",kind,n,lam,beta,theta)
        if o>0:
            r=tot/o; worst=min(worst,r)
            if r < 1/(P*m)-1e-9: viol+=1; print("RATIO VIOLATION",r,1/(P*m))
        if tot>o+1e-9: viol+=1; print("ALG>OPT?!")
    print("instances",cnt,"violations",viol,"worst ratio",worst)
