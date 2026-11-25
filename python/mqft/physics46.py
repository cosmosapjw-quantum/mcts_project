"""RESTORE46: actual marked-path martingale diagnostics, finite count
memory closure, operational causal-break on a declared probe.
These are distinct evidence classes, not native quantum dynamics."""
import math,itertools
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from .conditional import density
from .framework import instrument
from .observables import entropy

def log_ratio(p,I,delta):
    p=np.asarray(p,float);I=np.asarray(I,float)
    assert p.shape==I.shape and np.all((0<=p)&(p<=1)) and set(I.ravel())<={0.,1.}
    # log P_(h+delta)/P_h on the same path, fixed downstream law.
    den=np.log1p(p*np.expm1(delta))
    return float(np.sum(delta*I-den)),float(np.sum(den-delta*p))

def score_evidence(records,lambdas=(-1.,-.5,-.25,.25,.5,1.)):
    """Fixed symmetric mixture of test martingales under ideal independent
    conditional uniform draws. No fitted lambda, no multiple-arm reuse."""
    p=np.concatenate([np.asarray(r["conditional_p"],float) for r in records])
    I=np.concatenate([np.asarray(r["target_ind"],float) for r in records])
    g=I-p;G=float(g.sum());F=float(np.sum(p*(1-p)))
    vals=[float(t*G-np.sum(np.log1p(p*np.expm1(t))-t*p)) for t in lambdas]
    loge=float(logsumexp(vals)-math.log(len(vals)))
    return dict(n_paths=len(records),n_draws=len(p),G=G,F=F,score_over_sqrt_F=G/math.sqrt(F) if F else None,
                log_e_mixture=loge,e_mixture=math.exp(loge),fixed_lambdas=list(lambdas),
                fisher_gap=float(sum(r["G"]**2-r["F"] for r in records)))

def actual_protocol(r,target):
    """Source-protocol comparison, NOT temporal reversal. Sigma=log P_h/P_target.
    Sum conditional KL is predictable compensator, not observed heat."""
    delta=target-r["h"]
    ll,kl=log_ratio(r["conditional_p"],r["target_ind"],delta)
    return dict(sigma=-ll,weight=math.exp(ll),conditional_KL=kl,compensated=-ll-kl)

def fit_count_memory(train,test,kind,shrink=4.0):
    """Declared homogeneous pair-state continuation fitted to the last
    three complete width4 waves only. Predict last count on held-out roots.
    No native time homogeneity assumption. Train counts smoothed with
    Jeffreys 1/2 and hierarchical backoff to first order."""
    d=2 if kind=="pl" else 5
    A=np.zeros((d,d));B=np.zeros((d,d,d))
    for r in train:
        a,b,c=r["full_wave_counts"][-3:]
        A[b,c]+=1;B[a,b,c]+=1
    M=(A+.5)/(A.sum(1)[:,None]+.5*d)
    N=(B+shrink*M[None,:,:])/(B.sum(2)[:,:,None]+shrink)
    pairs=list(itertools.product(range(d),repeat=2));K=np.zeros((d*d,d*d));E=np.zeros((d*d,d))
    for i,(a,b) in enumerate(pairs):
        E[i,b]=1
        for c in range(d):K[i,b*d+c]=N[a,b,c]
    assert np.max(abs(K.sum(1)-1))<1e-12
    # Stationary conditional embedding in this declared fitted reference.
    q=np.ones(d*d)/(d*d)
    for _ in range(10000):
        z=q@K
        if np.max(abs(z-q))<1e-14:q=z;break
        q=z
    assert min(q)>0 and np.max(abs(q@K-q))<1e-12
    R=(E.T*q)/(q@E)[:,None];P=E@R;Q=np.eye(d*d)-P
    assert np.max(abs(R@E-np.eye(d)))<1e-12
    assert np.max(abs(Q@Q-Q))<1e-12
    # Block Dyson/Mori-Zwanzig recurrence, acting on lifted functions.
    G=R@K@E;L=R@K@Q;C=Q@K@E;D=Q@K@Q
    horizon=8;T=[np.eye(d)];kernels=[]
    for j in range(1,horizon):
        kernels.append(L@np.linalg.matrix_power(D,j-1)@C)
    exact=[R@np.linalg.matrix_power(K,n)@E for n in range(horizon+1)]
    for n in range(horizon):
        x=G@T[n]
        for i in range(n):x+=kernels[n-i-1]@T[i]
        T.append(x)
    res=max(float(np.max(abs(x-y))) for x,y in zip(T,exact));assert res<1e-10
    groups=defaultdict(list)
    for r in test:
        a,b,c=r["full_wave_counts"][-3:]
        groups[r["position"]].append((-math.log(M[b,c]),-math.log(N[a,b,c])))
    errors=[dict(position=k,n=len(x),markov=float(np.mean(np.array(x)[:,0])),memory=float(np.mean(np.array(x)[:,1]))) for k,x in sorted(groups.items())]
    return dict(kind=kind,train_n=len(train),test_n=len(test),train_table=A.tolist(),memory_table=B.tolist(),
                markov=float(np.mean([x["markov"] for x in errors])),memory=float(np.mean([x["memory"] for x in errors])),
                test_groups=errors,projected_generator=G.tolist(),kernel_norms=[float(np.linalg.norm(x)) for x in kernels],
                markov_defect=[float(np.linalg.norm(x-np.linalg.matrix_power(G,i))) for i,x in enumerate(exact)],
                mz_residual=res,model="empirical homogeneous pair-state continuation, stationarity/projection declared; not fitted native generator")

def causal_break(word,alpha=.5):
    """Condition on the SAME prescribed classical word. Two nonoverlapping
    complete waves drive a qubit; Y readout, reset to +X, second Y readout.
    Same marginal one-step channels, different hidden sign memory.
    CP measurement-and-reprepare at the break. Exact probabilities."""
    m=np.asarray(word,int);assert m.shape==(8,) and set(m)<={0,1}
    psi=np.array([1.,1.])/math.sqrt(2);r0=np.outer(psi,psi)
    yp=np.array([1.,1j])/math.sqrt(2);ym=np.array([1.,-1j])/math.sqrt(2)
    # K_y=|+X><y|; Y measurement and deterministic preparation independent of y.
    K=[np.outer(psi,y.conj()) for y in (yp,ym)];P=[np.outer(y,y.conj()) for y in (yp,ym)]
    def evolve(r,a,signs):
        for z,s in zip(a,signs):
            u=np.diag(np.exp(-1j*alpha*z*s*np.array([1.,-1.])))
            r=u@r@u.conj().T
        return density(r)
    # Given a fixed sign, first and second branch probabilities.
    past=np.asarray(m[:4]);future=np.asarray(m[4:])
    stat=np.zeros((2,2))
    for s in (-1,1):
        r=evolve(r0,past,[s]*4)
        b=instrument(r,[[K[0]],[K[1]]])
        for y,(prob,state) in enumerate(b):
            if prob==0:continue
            assert np.max(abs(state-r0))<1e-10
            z=evolve(state,future,[s]*4)
            stat[y]+=prob*np.array([np.trace(x@z).real for x in P])/2
    # Independent hidden signs across steps and the break.
    def indep(a):
        z=sum(evolve(r0,a,s) for s in itertools.product((-1,1),repeat=4))/16
        return np.array([np.trace(x@z).real for x in P])
    independent=np.outer(indep(past),indep(future))
    # Analytic conditional correlation; same fixed classical drive cancels.
    c=math.sin(2*alpha*past.sum())*math.sin(2*alpha*future.sum())
    ref=(1+c*np.array([[1,-1],[-1,1]]))/4
    residual=float(max(np.max(abs(stat-ref)),np.max(abs(independent-.25))))
    assert residual<1e-10 and abs(stat.sum()-1)<1e-10 and min(stat.ravel())>=-1e-12
    mi=lambda a:entropy(a.sum(0))+entropy(a.sum(1))-entropy(a.ravel())
    return dict(correlation=c,static_MI=max(0.,mi(stat)),iid_MI=max(0.,mi(independent)),
                static_joint=stat.tolist(),iid_joint=independent.tolist(),residual=residual,
                future_Y_plus_gap=float(abs(stat[0,0]/stat[0].sum()-stat[1,0]/stat[1].sum())),
                past_hits=int(past.sum()),future_hits=int(future.sum()))
