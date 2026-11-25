"""RESTORE48: finite-response precision, causal response kernels, finite-shot
conditional process tomography and duration-dependent thermal cycles."""
import math
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from .physics47 import P,choi_from_ptm
from .observables import entropy

def cov(A,G):
    A=np.asarray(A,float);G=np.asarray(G,float)
    if A.ndim==1:A=A[:,None]
    if G.ndim==1:G=G[:,None]
    assert len(A)==len(G) and len(A)>=2
    return (A-A.mean(0)).T@(G-G.mean(0))/(len(A)-1)

def jackknife(f,n):
    """Delete entire paired histories, not individual plies or score terms."""
    allidx=np.arange(n);v=np.asarray(f(allidx),float)
    if n<3:return v,np.full_like(v,np.nan)
    a=np.stack([f(allidx[allidx!=i]) for i in allidx]);se=np.sqrt((n-1)/n*np.sum((a-a.mean(0))**2,axis=0))
    return v,se

def response_precision(arms):
    """First/second finite contrast and score; paired seeds inside B.
    Output errors from deleting a whole seed group. No finite-h equality
    assumption for the derivative."""
    z=sorted(arms[0.],key=lambda r:r["seed"]);n=len(z)
    A=np.array([[r["occupancy"],r["mean_mark"]] for r in z])
    g=np.array([r["G"] for r in z]);F=np.array([r["F"] for r in z]);q=g*g-F
    def lw(h):
        return np.array([np.exp(h*sum(r["target_ind"])-np.log1p(np.asarray(r["conditional_p"])*np.expm1(h)).sum()) for r in z])
    def estimate(v,i):return cov(A[i],v[i]).ravel()
    out=[]
    contrasts={}
    for h in sorted(x for x in arms if x>0):
        plus,minus=[sorted(arms[x],key=lambda r:r["seed"]) for x in (h,-h)]
        assert [r["seed"] for r in plus]==[r["seed"] for r in z]==[r["seed"] for r in minus]
        p=np.array([[r["occupancy"],r["mean_mark"]] for r in plus]);m=np.array([[r["occupancy"],r["mean_mark"]] for r in minus])
        d=(p-m)/(2*h);d2=(p+m-2*A)/h**2
        lp,lm=lw(h),lw(-h);v=(lp-lm)/(2*h);v2=(lp+lm-2)/h**2
        def estim(i):
            a=estimate(v,i);b=estimate(v2,i);c=d[i].mean(0);e=d2[i].mean(0)
            return np.stack([estimate(g,i),estimate(q,i),a,b,c,e,a-c,b-e])
        val,se=jackknife(estim,n)
        names=("score","score2","LR","LR2","direct","direct2","LR_minus_direct","LR2_minus_direct2")
        out.append(dict(h=h,n=n,values=dict(zip(names,val.tolist())),se=dict(zip(names,se.tolist())),
            Lplus=float(lp.mean()),Lminus=float(lm.mean()),ESSplus=float(lp.sum()**2/(lp@lp)),ESSminus=float(lm.sum()**2/(lm@lm))))
        contrasts[h]=(d,d2,v,v2)
    rich=None
    if .1 in contrasts and .2 in contrasts:
        a,b=contrasts[.1],contrasts[.2]
        # Fourth order for smooth expectation; finite seed samples remain discontinuous.
        v=[(4*x-y)/3 for x,y in zip(a,b)]
        def est(i):
            return np.stack([v[0][i].mean(0),v[1][i].mean(0),estimate(v[2],i),estimate(v[3],i),
                v[0][i].mean(0)-estimate(g,i),v[1][i].mean(0)-estimate(q,i)])
        val,se=jackknife(est,n)
        rich=dict(names=["direct_Richardson","direct2_Richardson","LR_Richardson","LR2_Richardson","direct_minus_score","direct2_minus_score2"],values=val.tolist(),se=se.tolist())
    return out,rich

def causal_kernel(records):
    """Partition into four-completed-backup blocks, including initial short
    phases grouped into first block. Block score and mark are measured in
    actual root frame. Causal truncation is unbiased under ideal law if
    full backups at each block boundary precede later selection."""
    n=len(records);B=records[0]["k"];assert B%4==0 and all(r["k"]==B for r in records)
    I=np.array([r["target_ind"] for r in records]).reshape(n,-1,4)
    M=np.array([r["marks"] for r in records]).reshape(n,-1,4)
    p=np.array([r["conditional_p"] for r in records]).reshape(n,-1,4)
    G=(I-p).sum(2);F=(p*(1-p)).sum(2);prefix=G.cumsum(1);fp=F.cumsum(1)
    X=np.stack([I.mean(2),M.mean(2)],axis=-1);d=X.shape[1]
    lower=np.tril(np.ones((d,d)));upper=1-lower
    K=np.stack([cov(X[:,:,i],G) for i in range(2)])
    assert np.max(abs(K.sum(2).mean(1)-cov(X.mean(1),G.sum(1)).ravel()))<1e-10
    def estimator(ids):
        x=X[ids];g=G[ids];pf=prefix[ids];v=fp[ids]
        k=np.stack([cov(x[:,:,i],g) for i in range(2)])
        full=k.sum((1,2))/d;causal=(k*lower).sum((1,2))/d
        score2=cov(x.mean(1),g.sum(1)**2-F[ids].sum(1)).ravel()
        causal2=np.array([np.mean(np.diag(cov(x[:,:,i],pf**2-v))) for i in range(2)])
        future=(k*upper).sum((1,2))/d
        return np.stack((full,causal,score2,causal2,future))
    val,se=jackknife(estimator,n)
    # block coarse-graining preserves sums. Compare fluctuation amplitude
    # at block scales; correlation between blocks is retained.
    scales=[]
    for ell in (1,2,4,8,16):
        if d%ell:continue
        a=X.reshape(n,d//ell,ell,2).mean(2)
        vari=np.mean([np.diag(cov(a[:,j],a[:,j])) for j in range(d//ell)],axis=0)
        scales.append(dict(block_backups=4*ell,mean_within_position_variance=vari.tolist()))
    xx=X.mean(1);cc=xx[:,0]-xx[:,0].mean();v=np.mean(cc**2)
    binder=float(1-np.mean(cc**4)/(3*v*v)) if v>1e-14 else None
    return dict(n=n,B=B,kernel=K.tolist(),names=["full_score","causal_score","full_score2","causal_score2","future_score"],estimates=val.tolist(),se=se.tolist(),
        future_frobenius=[float(np.linalg.norm(k*upper)) for k in K],variance_scales=scales,binder_centered=binder,mean_F=float(F.sum(1).mean()))

def distance(a,b):
    return float(np.abs(np.linalg.eigvalsh((a-b+a.conj().T-b.conj().T)/2)).sum()/2)

def shot_channel(T,shots,rng):
    """Simulated independent binomial outcomes of six known preparations
    and three Pauli measurements; linear inversion. Depolarizing mixture
    repair preserves TP; not a maximum-likelihood reconstruction."""
    T=np.asarray(T,float)
    expectation=np.array([[[T[i,0]+s*T[i,j] for i in range(1,4)] for s in (-1,1)] for j in range(1,4)])
    prob=np.clip((1+expectation)/2,0,1);counts=rng.binomial(shots,prob);obs=2*counts/shots-1
    A=np.zeros((4,4));A[0,0]=1;A[1:,0]=obs.mean((0,1));A[1:,1:]=((obs[:,1]-obs[:,0])/2).T
    J=choi_from_ptm(A);m=float(np.linalg.eigvalsh(J).min())
    # (1-r)J+r I/4 has min eig >=0 and same partial trace.
    r=max(0.,-m/(.25-m));R=(1-r)*J+r*np.eye(4)/4
    assert np.linalg.eigvalsh(R).min()>=-1e-12
    assert np.max(abs(np.trace(R.reshape(2,2,2,2),axis1=1,axis2=3)-np.eye(2)/2))<1e-10
    return J,R,m,r

def tomography_trials(probes,shots=(128,512,2048),replicates=32,seed=48004):
    """One experiment per distinct drive-count pair, not 48 independent
    physical states. Same reference model on null channels."""
    rng=np.random.default_rng(seed);by={(x["n1"],x["n2"]):x for x in probes};out=[]
    for (a,b),x in sorted(by.items()):
        for N in shots:
            for i in range(replicates):
                tmp={}
                for name in ("static","reset_static","dephased"):
                    T=x["channels"][name]["ptm"];exact=choi_from_ptm(np.array(T))
                    J,R,m,r=shot_channel(T,N,rng);tmp[name]=(J,R)
                    out.append(dict(n1=a,n2=b,shots=N,replicate=i,kind=name,min_eig=m,repair=r,raw_error=distance(J,exact),repaired_error=distance(R,exact)))
                # Independent sampling of identical persistent channel as null.
                _,R0,_,_=shot_channel(x["channels"]["static"]["ptm"],N,rng)
                a0=tmp["static"][1];b0=tmp["reset_static"][1]
                out[-1]["contrast"]={"true":x["choi_static_reset"],"observed":distance(a0,b0),"same_channel_null":distance(a0,R0)}
    return out

def duration_cycle(ps,duration,subdiv):
    """Conditional two-level cyclic Hamiltonian, beta E*=1 and gamma=1.
    Interpolate log odds along three segments, quench each dt then
    relax exactly under reset generator. DP computes work MGF in both
    directions; no exponential path enumeration."""
    ps=np.asarray(ps,float);assert ps.shape==(3,) and np.all((ps>0)&(ps<1))
    z=np.log(ps/(1-ps));z=np.r_[z,z[0]]
    field=np.concatenate([np.linspace(z[i],z[i+1],subdiv,endpoint=False) for i in range(3)]+[z[-1:]])
    p=1/(1+np.exp(-field));pi=np.column_stack([1-p,p]);E=-np.log(pi)
    eta=-math.expm1(-duration/(3*subdiv));T=(1-eta)*np.eye(2)[None,:,:]+eta*np.repeat(pi[1:,None,:],2,axis=1)
    mu=pi[0].copy();W=Q=0.
    for j,M in enumerate(T):
        W+=mu@(E[j+1]-E[j]);v=mu@M;Q+=(v-mu)@E[j+1];mu=v
    U=float(mu@E[-1]-pi[0]@E[0]);ds=entropy(mu)-entropy(pi[0]);sig=ds-Q;lag=float(mu@np.log(mu/pi[0]))
    def mgf(l,rev=False):
        a=pi[-1 if rev else 0].copy()
        if rev:
            for j in range(len(T)-1,-1,-1):a=(a@T[j])*np.exp(-l*(E[j]-E[j+1]))
        else:
            for j,M in enumerate(T):a=(a*np.exp(-l*(E[j+1]-E[j])))@M
        return float(a.sum())
    es=max(abs(mgf(t)-mgf(1-t,True)) for t in (0.,.25,.5,.75,1.))
    res=max(es,abs(mgf(1)-1),abs(U-W-Q),abs(W-sig-lag))
    assert res<1e-9 and sig>=-1e-10 and lag>=-1e-10
    return dict(duration=duration,subdiv=subdiv,steps=3*subdiv,eta=eta,work=float(W),heat=float(Q),delta_U=U,delta_S=ds,sigma=float(sig),lag=lag,JE=mgf(1),crooks_mgf_residual=es,balance_residual=res)


def prefix_finite(arms):
    """Prefix likelihood control: each block observable is measurable at
    its completed-backup boundary; suffix likelihood integrates to one.
    Centered covariance estimates; jackknife complete paired histories."""
    z=sorted(arms[0.],key=lambda r:r["seed"]);n=len(z);B=z[0]["k"];d=B//4
    p=np.array([r["conditional_p"] for r in z]);I=np.array([r["target_ind"] for r in z])
    def X(a):
        a=sorted(a,key=lambda r:r["seed"])
        return np.stack([np.array([r["target_ind"] for r in a]).reshape(n,d,4).mean(2),
                         np.array([r["marks"] for r in a]).reshape(n,d,4).mean(2)],axis=-1)
    A=X(z);out=[]
    for h in sorted(k for k in arms if k>0):
        assert all([r["seed"] for r in sorted(arms[t],key=lambda r:r["seed"])]==[r["seed"] for r in z] for t in (-h,h))
        lp=np.exp(np.cumsum(h*I-np.log1p(p*np.expm1(h)),axis=1)[:,3::4]);lm=np.exp(np.cumsum(-h*I-np.log1p(p*np.expm1(-h)),axis=1)[:,3::4])
        D=(lp-lm)/(2*h);D2=(lp+lm-2)/h**2
        plus,minus=X(arms[h]),X(arms[-h]);direct=(plus-minus).mean(1)/(2*h);direct2=(plus+minus-2*A).mean(1)/h**2
        def f(ids):
            a=A[ids];d1,d2=D[ids],D2[ids]
            full=cov(a.mean(1),d1[:,-1]).ravel();full2=cov(a.mean(1),d2[:,-1]).ravel()
            pre=np.array([np.diag(cov(a[:,:,i],d1)).mean() for i in range(2)])
            pre2=np.array([np.diag(cov(a[:,:,i],d2)).mean() for i in range(2)])
            v,w=direct[ids].mean(0),direct2[ids].mean(0)
            return np.stack((full,pre,full2,pre2,v,w,pre-v,pre2-w,full-v,full2-w))
        v,se=jackknife(f,n)
        names=("full","prefix","full2","prefix2","direct","direct2","prefix_diff","prefix2_diff","full_diff","full2_diff")
        out.append(dict(h=h,n=n,values=dict(zip(names,v.tolist())),se=dict(zip(names,se.tolist()))))
    return out
