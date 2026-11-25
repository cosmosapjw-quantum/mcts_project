"""Numerical observables adapted from RESTORE38 and audited against original MQFT targets.
All scores are dimensionless reward units, not measured energies.
"""
from collections import Counter,defaultdict
import itertools
import math
import numpy as np

def entropy(p):
    x=np.asarray(p,float)
    if np.any(x<0) or not np.isfinite(x).all() or abs(x.sum()-1)>2e-6:raise ValueError("probability vector")
    x=x[x>0]
    return float(-x@np.log(x))

def kl(p,q):
    p,q=np.asarray(p,float),np.asarray(q,float)
    i=p>0
    if np.any(q[i]<=0):return math.inf
    return float(np.dot(p[i],np.log(p[i]/q[i])))

def fit_beta(pi,score):
    """Minimum KL(pi||softmax(beta*score)), beta>=0; exact boundary states."""
    pi=np.asarray(pi,float);s=np.asarray(score,float)
    if len(s)!=len(pi) or not np.isfinite(s).all():raise ValueError("score")
    h=entropy(pi)
    scale=float(np.ptp(s))
    if scale<=1e-13*max(1.,float(np.max(abs(s)))):
        return dict(beta=None,status="flat",kl=math.log(len(s))-h)
    y=(s-s.max())/scale;target=float(pi@y)
    if target<=float(y.mean())+1e-14:
        return dict(beta=0.,status="zero",kl=max(0.,math.log(len(s))-h))
    if target>=0. and np.all(pi[y<0]==0):
        return dict(beta=None,status="infinite",kl=max(0.,math.log(int(np.sum(y==0)))-h))
    if target>=0.:
        return dict(beta=None,status="unresolved_roundoff",kl=None)
    def moment(b):
        x=np.exp(b*y);return float((x@y)/x.sum())
    lo=0.;hi=1.
    while moment(hi)<target:
        hi*=2
        if hi>1e16:return dict(beta=None,status="unresolved_precision",kl=None)
    for _ in range(100):
        mid=(lo+hi)/2
        if moment(mid)<target:lo=mid
        else:hi=mid
    b=(lo+hi)/2;logp=b*y-math.log(float(np.exp(b*y).sum()))
    return dict(beta=float(b/scale),status="finite",kl=max(0.,float(-h-pi@logp)))

def frozen_action(q,p,k,c,pi):
    """Finite shifted PUCT objective, exact KKT optimum and fixed-active tangent curvature.
    Laplace covariance is conditional on an explicitly supplied scale theta, not native."""
    q,p,pi=np.asarray(q,float),np.asarray(p,float),np.asarray(pi,float)
    if k<=0 or c<=0 or np.any(p<=0):raise ValueError("strict positive prior and k,c required")
    lam=c/math.sqrt(k);shift=1/k
    lo=float(q.max());hi=lo+lam*float(p.sum())+1.
    def x(mu):return np.maximum(lam*p/(mu-q)-shift,0.)
    for _ in range(90):
        mid=(lo+hi)/2
        if mid<=lo or mid>=hi:break
        if x(mid).sum()>1:lo=mid
        else:hi=mid
    mu=(lo+hi)/2
    opt=x(mu);score=q+lam*p/(opt+shift);active=opt>1e-10
    if not np.any(active):raise ArithmeticError("empty optimum")
    def obj(a):return float(q@a+lam*(p@np.log(a+shift)))
    dual=float((q+lam*p/(pi+shift)).max()-(q+lam*p/(pi+shift))@pi)
    gap=obj(opt)-obj(pi)
    residual=max(abs(opt.sum()-1),float(np.max(abs(score[active]-mu))),float(np.max(np.maximum(score[~active]-mu,0),initial=0)),max(-gap,0.),max(gap-dual,0.))
    if residual>2e-8:raise ArithmeticError(f"KKT residual {residual}")
    d=lam*p[active]/(opt[active]+shift)**2
    # Orthonormal simplex tangent determinant det(T^T diag(d) T)
    # = (prod d) * sum(1/d)/m. This avoids O(m^3) per ply.
    m=len(d);logdet=float(np.log(d).sum()+math.log(float((1/d).sum()/m))) if m>1 else 0.
    # Fixed-active response d x / d h for h added to q: diag(1/d)-vv^T/sum(v).
    # Here report trace and eigenvalue-free logdet, not a physical loop coefficient.
    v=1/d;trace=float(v.sum()-v@v/v.sum()) if m>1 else 0.
    return dict(gap=max(0.,gap),dual_bound=dual,tv=float(.5*np.sum(abs(opt-pi))),kkt_residual=residual,
                active_count=int(m),tangent_logdet=logdet,tangent_inverse_trace=trace,
                optimum=opt.tolist(),active_mask=active.tolist(),curvature=d.tolist(),
                convention="negative Hessian of shifted frozen objective; no calibrated loop scale")

def tangent_response(q,p,k,c,pi,field):
    f=frozen_action(q,p,k,c,pi)
    active=np.array(f["active_mask"]);d=np.array(f["curvature"]);v=1/d
    field=np.asarray(field,float);out=np.zeros_like(field)
    out[active]=v*(field[active]-float(v@field[active]/v.sum()))
    return out,f

def marks(events):
    """Exact per-prefix N,W,S2 in the root frame, no fabricated W² substitution."""
    out=defaultdict(lambda:[0,0.,0.])
    for e in events:
        word=tuple(e["actions"]);v=float(e["root_value"])
        for d in range(1,len(word)+1):
            row=out[word[:d]];row[0]+=1;row[1]+=v;row[2]+=v*v
    return out

def path_entropy(events):
    if not events:return None
    full=Counter(tuple(e["actions"]) for e in events)
    root=Counter()
    for w,n in full.items():root[w[0]]+=n
    k=len(events);h=entropy(np.array(list(full.values()),float)/k);hr=entropy(np.array(list(root.values()),float)/k)
    return dict(path_H=h,root_H=hr,conditional_H=max(0.,h-hr),distinct_paths=len(full))

def graph_and_coarse(events,max_pairs=10000,seed=0):
    """Root-frame prefix-tree graph. TT is off; replicas must be kept separate.
    Covariance uses uniform visited prefix-node measure, n>=2, with pair sampling explicitly marked."""
    m=marks(events);items=[(w,n,wv/n) for w,(n,wv,s2) in sorted(m.items()) if n>=2]
    if not items:return dict(graph=[],coarse=[],eligible_nodes=0,total_pairs=0,sampled_pairs=0)
    vals=np.array([x[2] for x in items]);weights=np.array([x[1] for x in items],float)
    centered=vals-vals.mean();total=len(items)*(len(items)-1)//2
    def unrank(t):
        # inverse triangular row offsets, binary search without materializing all pairs
        n=len(items);lo=0;hi=n-1
        while hi-lo>1:
            z=(lo+hi)//2
            if z*(2*n-z-1)//2<=t:lo=z
            else:hi=z
        i=lo;j=i+1+t-i*(2*n-i-1)//2
        return i,j
    if total<=max_pairs:pairs=itertools.combinations(range(len(items)),2);sampled=total
    else:
        import random
        pairs=(unrank(t) for t in sorted(random.Random(seed).sample(range(total),max_pairs)));sampled=max_pairs
    agg=defaultdict(lambda:[0,0.])
    for i,j in pairs:
        a,b=items[i][0],items[j][0];d=0
        for x,y in zip(a,b):
            if x!=y:break
            d+=1
        dist=len(a)+len(b)-2*d;row=agg[dist];row[0]+=1;row[1]+=float(centered[i]*centered[j])
    coarse=[];mean=float(weights@vals/weights.sum());total_var=float(weights@((vals-mean)**2)/weights.sum())
    for width in (1,2,4,8,16):
        blocks=defaultdict(list)
        for i,(w,n,v) in enumerate(items):
            anchor=((len(w)-1)//width)*width+1
            blocks[w[:anchor]].append(i)
        between=within=0.
        for ids in blocks.values():
            ww=weights[ids];vv=vals[ids];z=float(ww@vv/ww.sum())
            within+=float(ww@((vv-z)**2));between+=float(ww.sum()*(z-mean)**2)
        within/=weights.sum();between/=weights.sum()
        if abs(total_var-within-between)>1e-10:raise ArithmeticError("variance decomposition")
        coarse.append(dict(width=width,blocks=len(blocks),within=within,between=between,total_variance=total_var,unresolved_fraction=within/total_var if total_var>0 else None))
    return dict(graph=[dict(distance=d,pairs=n,cov=v/n) for d,(n,v) in sorted(agg.items())],
                coarse=coarse,eligible_nodes=len(items),total_pairs=total,sampled_pairs=sampled,
                graph_measure="uniform eligible nodes, within-tree pairs; pair subsampling without replacement" if sampled<total else "all within-tree pairs, uniform eligible nodes")

def root_observables(edges,k,c):
    p=np.array([x["prior"] for x in edges],float)
    n=np.array([x["n"] for x in edges],float);w=np.array([x["w"] for x in edges],float);s2=np.array([x["s2"] for x in edges],float)
    if k<=0 or n.sum()!=k or np.any(p<=0):raise ValueError("root support or count")
    pi=n/k;q=np.divide(w,n,out=np.zeros_like(w),where=n>0);qroot=float(w.sum()/k)
    # Native Quartz uses sqrt(N) (not sqrt(max(N,1))); only positive quiescent cuts.
    score=q+c*math.sqrt(k)*p/(1+n);h=entropy(pi);fit=fit_beta(pi,score)
    act=frozen_action(q,p,k,c,pi)
    var=float(s2.sum()/k-qroot*qroot);within=float((s2.sum()-n@(q*q))/k);between=float(pi@((q-qroot)**2))
    if min(var,within,between)<-1e-10 or abs(var-within-between)>1e-10:raise ArithmeticError("mark variance")
    return dict(H=h,H_ceiling=math.log(min(k,len(edges))),H_fraction=h/math.log(min(k,len(edges))) if min(k,len(edges))>1 else None,
        order_parameter=float(np.sort(pi)[-1]-(np.sort(pi)[-2] if len(pi)>1 else 0.)),support=int((n>0).sum()),effective_actions=float(1/(pi@pi)),q=qroot,variance=max(0.,var),within=max(0.,within),between=max(0.,between),
        prior_KL=kl(pi,p/p.sum()),score_mean=float(pi@score),beta=fit["beta"],beta_status=fit["status"],fit_KL=fit["kl"],beta_MF=math.sqrt(k)/c,
        kkt_gap=act["gap"],kkt_tv=act["tv"],kkt_residual=act["kkt_residual"],active_count=act["active_count"],
        tangent_logdet=act["tangent_logdet"],tangent_inverse_trace=act["tangent_inverse_trace"],
        actions=[x["action"] for x in edges],pi=pi.tolist(),prior=p.tolist(),action_q=q.tolist(),action_variance=np.maximum(np.divide(s2,n,out=np.zeros_like(s2),where=n>0)-q*q,0.).tolist(),score=score.tolist(),optimum=act["optimum"])
def delta(a,b):
    """Exact same-root score expectation product rule and dyadic drift."""
    if a["actions"]!=b["actions"]:raise ValueError("support changed")
    p,q=np.asarray(a["pi"]),np.asarray(b["pi"]);e,f=-np.asarray(a["score"]),-np.asarray(b["score"])
    de=float(q@f-p@e);field=float(.5*(p+q)@(f-e));red=float(.5*(e+f)@(q-p))
    if abs(de-field-red)>1e-10:raise ArithmeticError("score product rule")
    return dict(dE=de,field=field,redistribution=red,TV=float(.5*sum(abs(p-q))),dH=b["H"]-a["H"],dQ=b["q"]-a["q"],
        plurality_overlap=bool(set(np.flatnonzero(p==p.max()))&set(np.flatnonzero(q==q.max()))))
def empirical_mi(xs,ys):
    if len(xs)!=len(ys) or not xs:raise ValueError("paired labels")
    n=len(xs);a=Counter(xs);b=Counter(ys);ab=Counter(zip(xs,ys))
    h=entropy(np.array(list(a.values()),float)/n)
    mi=sum(v/n*math.log(v*n/(a[x]*b[y])) for (x,y),v in ab.items())
    return dict(H=h,I=max(0.,mi),fraction=max(0.,mi)/h if h>1e-14 else None,n=n,unique_pairs=len(ab))
