import numpy as np, math
from common import *
from alg import *
# (a) spoke bound on unit ray:  W_spoke(c) <= 1 + ln(n(n+1)/2)/ln(1+1/q), q=(c*lam-1)/2   (c*lam>1)
viol=0; cnt=0
for n in range(2,15):
    D=euclid(np.arange(n+1,dtype=float)[:,None]); w=np.ones(n+1); w[0]=0
    for lam in [1.0,1.2,1.5,2,3,5]:
        for c in [1,1.5,2,4]:
            if c*lam<=1: continue
            q=(c*lam-1)/2
            ws=spoke_exact(D,w,lam,0,speed=c)
            bound=1+math.log(n*(n+1)/2)/math.log(1+1/q)
            cnt+=1
            if ws>bound+1e-9: viol+=1; print('VIOL',n,lam,c,ws,bound)
print('ray spoke bound checks',cnt,'violations',viol)
# (b) star with legs 2^k (one per class): OPT vs sum of standalone class optima
for lam in [1.1,1.25,1.5]:
    rho=lam-1
    for K in [10,12]:
        N=K; W=np.full((N+1,N+1),np.inf); np.fill_diagonal(W,0)
        for i in range(N): W[0,i+1]=W[i+1,0]=2.0**i
        D=graph_metric(W); w=np.ones(N+1); w[0]=0
        opt=chain_exact(D,w,lam)
        print(f'star legs 2^k, K={K}, lam={lam}: OPT={opt:.0f}, sum_k standalone = {K}, ratio {K/opt:.2f}, m(theta=.5)={gap_m(rho,.5)}, log2(1/rho)={math.log2(1/rho):.2f}')
