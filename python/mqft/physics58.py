"""RESTORE58: actual occupancy correlations, first detection, landscape
filtration and explicit backup checksum recovery. All native scores
dimensionless; no physical tunneling, quantum code or critical exponent."""
import math,heapq,itertools
import numpy as np
from collections import defaultdict
from .physics49 import root_observed
from .observables import root_observables,entropy

def neighbors(i,L):
    y,x=divmod(int(i),L)
    return [v*L+w for v,w in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if 0<=v<L and 0<=w<L]

def topology(values,L):
    """Lower-star vertex cubical complex on explicitly present vertices.
    H0 barcode by elder rule; Betti1 at sampled thresholds by Euler.
    No unobserved vertex filled or toroidal wrap."""
    v={int(k):float(x) for k,x in values.items()};assert v
    parent={};birth={};bars=[];E=F=0;cc=0;active=set()
    def root(x):
        while x!=parent[x]:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(i,j,t):
        nonlocal cc
        a,b=root(i),root(j)
        if a==b:return
        if (birth[a],a)>(birth[b],b):a,b=b,a
        bars.append((birth[b],t));parent[b]=a;cc-=1
    levels=sorted(set(v.values()));snap=[]
    by=defaultdict(list)
    for i,t in v.items():by[t].append(i)
    for t in levels:
        new=sorted(by[t])
        for i in new:parent[i]=i;birth[i]=t;active.add(i);cc+=1
        for i in new:
            for j in neighbors(i,L):
                if j not in active or (v[j]==t and j>i):continue
                E+=1;join(i,j,t)
        for i in new:
            y,x=divmod(i,L)
            for yy in (y-1,y):
                for xx in (x-1,x):
                    if not(0<=yy<L-1 and 0<=xx<L-1):continue
                    q=[yy*L+xx,yy*L+xx+1,(yy+1)*L+xx,(yy+1)*L+xx+1]
                    if all(j in active for j in q) and i==max(q,key=lambda j:(v[j],j)):
                        F+=1
        b1=cc-len(active)+E-F;assert b1>=0
        snap.append(dict(level=t,V=len(active),E=E,F=F,b0=cc,b1=b1))
    # Independent DFS and counts at predetermined rank quantiles.
    tests=[]
    for k in sorted({0,len(snap)-1,*[int(q*(len(snap)-1)) for q in (.1,.25,.5,.75,.9)]}):
        s=snap[k];a={i for i,t in v.items() if t<=s['level']};left=set(a);n=0
        while left:
            n+=1;stack=[left.pop()]
            while stack:
                for j in neighbors(stack.pop(),L):
                    if j in left:left.remove(j);stack.append(j)
        ed=sum(j in a for i in a for j in neighbors(i,L))//2
        faces=sum(all(i in a for i in (y*L+x,y*L+x+1,(y+1)*L+x,(y+1)*L+x+1)) for y in range(L-1) for x in range(L-1))
        assert (n,ed,faces)==(s['b0'],s['E'],s['F'])
        # finite planar cubical 2-complex has H2=0.
        tests.append(s)
    essential=[birth[i] for i in parent if root(i)==i]
    positive=[(a,b) for a,b in bars if b>a+1e-14]
    assert len(essential)==snap[-1]['b0'] and len(bars)+len(essential)==len(v)
    return dict(vertices=len(v),levels=len(levels),final=snap[-1],max_b0=max(x['b0'] for x in snap),max_b1=max(x['b1'] for x in snap),
                positive_H0=len(positive),max_persistence=max((b-a for a,b in positive),default=0.),mean_persistence=float(np.mean([b-a for a,b in positive])) if positive else 0.,
                H0_bars=positive,essential_births=essential,filtration=snap,independent_thresholds=len(tests))

def minimax(values,L,start,target):
    """Geometric graph reference, not a restriction on native root draws."""
    v={int(k):float(x) for k,x in values.items()}
    assert start in v and target in v
    d={start:v[start]};q=[(v[start],start)];prev={}
    while q:
        w,i=heapq.heappop(q)
        if w!=d[i]:continue
        if i==target:
            path=[i]
            while path[-1]!=start:path.append(prev[path[-1]])
            path=path[::-1];assert max(v[x] for x in path)==w
            # independent threshold connectivity: connected at w,
            # disconnected immediately below w (unless start==target).
            def connected(t):
                if v[start]>t or v[target]>t:return False
                st=[start];seen={start}
                while st:
                    z=st.pop()
                    if z==target:return True
                    for j in neighbors(z,L):
                        if j in v and v[j]<=t and j not in seen:seen.add(j);st.append(j)
                return False
            assert connected(w) and not connected(np.nextafter(w,-np.inf))
            return dict(connected=True,bottleneck=w,barrier=w-v[start],endpoint_gap=v[target]-v[start],path=path,
                        excess=w-max(v[start],v[target]),length=len(path)-1)
        for j in neighbors(i,L):
            if j not in v:continue
            a=max(w,v[j])
            if a<d.get(j,math.inf):d[j]=a;prev[j]=i;heapq.heappush(q,(a,j))
    return dict(connected=False,bottleneck=None,barrier=None,endpoint_gap=v[target]-v[start],path=[],excess=None,length=None)

def parity(x):
    # Independent stored moment checks, indices 1..d. Float64.
    a=np.arange(1,len(x)+1,dtype=float);return np.array([math.fsum(x),math.fsum(x*a),math.fsum(x*a*a)])
def repair(y,checks):
    y=np.asarray(y,float);d=parity(y)-checks
    # Single-error candidate and independent quadratic parity.
    if np.max(abs(d))<1e-10:return 'clean',y.copy(),None
    if abs(d[0])<1e-10:return 'detected_uncorrectable',y.copy(),None
    j=int(round(d[1]/d[0]))-1
    if not 0<=j<len(y):return 'detected_uncorrectable',y.copy(),None
    z=y.copy();z[j]-=d[0];err=abs(parity(z)-checks)
    # Floating residual bound, scaling with vector length and sum.
    if max(err)>1e-7:return 'detected_uncorrectable',y.copy(),None
    return 'corrected',z,j
def faults(edges):
    """Inject actual numerical corruption in copied root W vector only.
    Assumes independent pristine checks, isolated single-error model."""
    x=np.array([e['w'] for e in edges]);c=parity(x);n=len(x);assert n>=2
    a=int(np.argmax(np.abs(x)));b=(a+n//2)%n;rows=[]
    for j in (a,b):
        for d in (.125,-.5):
            y=x.copy();y[j]+=d;status,z,k=repair(y,c)
            assert status=='corrected' and k==j and max(abs(z-x))<1e-10
            rows.append(dict(kind='single',index=j,delta=d,status=status,error=float(max(abs(z-x)))))
    for da,db in ((.125,.25),(.125,-.125),(-.5,.125)):
        y=x.copy();y[a]+=da;y[b]+=db;s,z,k=repair(y,c)
        assert s=='detected_uncorrectable'
        rows.append(dict(kind='double',indices=[a,b],deltas=[da,db],status=s,error=None))
    assert repair(x,c)[0]=='clean'
    return dict(checks=c.tolist(),results=rows,visited=int(sum(e['n']>0 for e in edges)),legal=n)

def trace_extra(t,L,source):
    r=root_observed(t,L,source);ev=sorted(t['replicas'][0]['events'],key=lambda e:e['root_selection']['clock']);rep=t['replicas'][0];B=r['B']
    first=ev[8]['root_selection']['law'];v={int(a):-float(s) for a,s in zip(first['actions'],first['scores'])}
    # Frozen initial score known for all legal actions (not unvisited Q measurement).
    start=min(v,key=lambda a:(v[a],a));bar=minimax(v,L,start,source)
    e=rep['cuts'][-1]['edges'];q={x['action']:-x['w']/x['n'] for x in e if x['n']>0}
    obs=root_observables(e,B,t['spec']['c_puct'])
    p=np.asarray(r['p']);I=np.asarray(r['I']);at=np.r_[True,np.cumsum(I[:-1])==0];det=int(I.any());firsthit=int(np.argmax(I)+1) if det else None
    A=float(p@at);var=float((p*(1-p))@at);z=det-A
    assert abs(z-float((I-p)@at))<1e-10
    assert np.max(abs(p[4:8]-p[4]))<1e-12
    # First full width-four wave, conditional on earlier survival.
    firstprob=float(not I[:4].any())*(-math.expm1(4*math.log1p(-float(p[4]))))
    # Actual paths form a rooted trie, not a board-space complex.
    prefixes={()}
    for e0 in ev:
        word=tuple(e0['actions']);prefixes.update(word[:k] for k in range(1,len(word)+1))
    assert all(w[:-1] in prefixes for w in prefixes if w)
    depths=[len(w) for w in prefixes]
    r.update(order=obs['order_parameter'],mark_variance=obs['variance'],H_fraction=obs['H_fraction'],
             first_hit=firsthit,detected=det,at_risk=at.astype(int).tolist(),compensator=A,compensator_var=var,compensator_residual=z,
             firstwave_prob=firstprob,firstwave_hit=int(not I[:4].any() and I[4:8].any()),barrier=bar,root_k8_score=[v.get(a,None) for a in range(L*L)],
             visited_Q=[q.get(a,None) for a in range(L*L)],topology_k8=topology(v,L),topology_visited=topology(q,L),
             trie=dict(V=len(prefixes),E=len(prefixes)-1,b0=1,b1=0,max_depth=max(depths)),fault=faults(e))
    return r

def spatial(rr):
    """Open-boundary legal-pair correlation; fixed-sum connected
    field, no translational invariance assumption. Separate B and L."""
    L=rr[0]['L'];B=rr[0]['B'];n=len(rr);a=np.array([r['counts'] for r in rr],float)/B;d=a-a.mean(0)
    C=d.T@d/(n-1);trace=float(np.trace(C));assert abs(C.sum())<1e-12 and np.max(abs(C.sum(0)))<1e-12
    m=np.zeros(L*L,bool);m[rr[0]['legal']]=True;assert all(r['legal']==rr[0]['legal'] for r in rr)
    # radial actual open distance pairs, ordered pair sum; zero-lag
    # and negative offdiagonal integral due to fixed total.
    yy,xx=np.indices((L,L));dy=yy.ravel()[:,None]-yy.ravel();dx=xx.ravel()[:,None]-xx.ravel()
    rad=np.sqrt(dx*dx+dy*dy);bins=np.floor(rad+1e-10).astype(int);ix=m[:,None]&m[None,:]
    mass=np.bincount(bins[ix],weights=C[ix]);num=np.bincount(bins[ix])
    assert abs(mass.sum())<1e-12 and abs(mass[0]-trace)<1e-12
    f=np.fft.fft2(d.reshape(n,L,L),axes=(-2,-1));sf=(abs(f)**2).sum(0)/(n-1)
    assert abs(sf[0,0])<1e-12 and abs(sf.sum()/L**2-trace)<1e-12
    # No S0-based second moment xi; define nonzero-mode spectral ratio
    # only, may be <1 or >1, not correlation length.
    low=float((sf[0,1]+sf[0,-1]+sf[1,0]+sf[-1,0])/4);diag=float((sf[1,1]+sf[1,-1]+sf[-1,1]+sf[-1,-1])/4)
    source=rr[0]['source'];g=np.array([sum(r['G']) for r in rr]);y=a[:,source]
    v=float(np.var(y,ddof=1));resp=float(np.cov(y,g,ddof=1)[0,1]);central=m & (abs(dx[:,L*L//2])<=2)&(abs(dy[:,L*L//2])<=2)
    cen=a[:,central].sum(1);dm=cen-cen.mean();b2=np.mean(dm**2)
    binder=None if b2<1e-15 else float(1-np.mean(dm**4)/(3*b2*b2))
    return dict(L=L,B=B,n=n,trace=trace,S_low=low,S_diag=diag,S0=float(sf[0,0]),spectral_ratio=low/diag if diag>1e-15 else None,
        radial=[dict(r=i,pairs=int(k),C=float(mass[i]/k),mass=float(mass[i])) for i,k in enumerate(num) if k],
        rank=int(np.linalg.matrix_rank(d,tol=1e-12)),H_fraction=float(np.mean([r['H_fraction'] for r in rr])),
        order=float(np.mean([r['order'] for r in rr])),source_variance=v,source_response=resp,central_mass=float(cen.mean()),central_binder=binder,
        first_hit_rate=float(np.mean([r['detected'] for r in rr])),firstwave_observed=float(np.mean([r['firstwave_hit'] for r in rr])),firstwave_exact=float(np.mean([r['firstwave_prob'] for r in rr])),
        compensator_mean=float(np.mean([r['compensator_residual'] for r in rr])),compensator_SE=float(np.std([r['compensator_residual'] for r in rr],ddof=1)/np.sqrt(n)))
def late_response(arms):
    """Last-wave exact IID conditional source occupancy and detection.
    Prefix no-hit determines the exact conditional first detection."""
    z=sorted(arms[0.],key=lambda r:r['seed']);n=len(z);p=np.array([r['p'][-4] for r in z])
    surv=np.array([not any(r['I'][:-4]) for r in z],float)
    a,b=[sorted(arms[h],key=lambda r:r['seed']) for h in (.2,-.2)]
    assert [r['seed'] for r in a]==[r['seed'] for r in b]==[r['seed'] for r in z]
    q=lambda h:p*np.exp(h)/(1+p*np.expm1(h))
    for h,x in ((0.,z),(.2,a),(-.2,b)):
        assert np.max(abs(np.array([r['p'][-4:] for r in x])-q(h)[:,None]))<1e-10
        assert np.array_equal(np.array([r['I'][:-4] for r in x]),np.array([r['I'][:-4] for r in z]))
    v=(q(.2)-q(-.2))/.4
    d=(np.array([np.mean(r['I'][-4:]) for r in a])-np.array([np.mean(r['I'][-4:]) for r in b]))/.4
    fd=surv*((1-q(-.2))**4-(1-q(.2))**4)/.4
    hit=(np.array([r['detected'] for r in a],float)-np.array([r['detected'] for r in b],float))/.4
    def stats(a,b):
        return dict(direct=float(a.mean()),conditional=float(b.mean()),difference=float((a-b).mean()),direct_SE=float(a.std(ddof=1)/np.sqrt(n)),conditional_SE=float(b.std(ddof=1)/np.sqrt(n)),difference_SE=float((a-b).std(ddof=1)/np.sqrt(n)))
    return dict(n=n,occupancy=stats(d,v),detection=stats(hit,fd),analytic_occ_derivative=float(np.mean(p*(1-p))),analytic_detection_derivative=float(np.mean(surv*4*p*(1-p)**4)),surviving_prefixes=int(surv.sum()))
