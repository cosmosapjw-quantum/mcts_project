"""Post-pruning conditional mathematical primitives (RESTORE28--37).
No model or quantum phase is inferred by invoking native MCTS. Each function
takes its explicit measure/state/operator or graph. Small finite models only.
"""
import math
from itertools import product
import numpy as np
from .conditional import density,reduce,von_neumann
from .observables import entropy

def probability(p):
    p=np.asarray(p,float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>1e-10:raise ValueError("probability")
    return p

def hermitian(a,tol=1e-10):
    a=np.asarray(a,complex)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not len(a) or not np.isfinite(a).all() or np.max(abs(a-a.conj().T))>tol:raise ValueError("Hermitian operator")
    return (a+a.conj().T)/2

def gram_density(p,phase,gram):
    """rho=D_psi F D_psi†; diagonal fixed, F must be a correlation Gram."""
    p=probability(p);ph=np.asarray(phase,float);g=hermitian(gram)
    if ph.shape!=p.shape or not np.isfinite(ph).all() or g.shape!=(len(p),len(p)) or np.max(abs(np.diag(g)-1))>1e-10 or np.linalg.eigvalsh(g).min()<-1e-9:raise ValueError("phase or Gram contract")
    psi=np.sqrt(p)*np.exp(1j*ph);r=density(psi[:,None]*g*psi.conj()[None,:])
    if np.max(abs(r.diagonal().real-p))>1e-10:raise ArithmeticError("diagonal identity")
    return r

def gram_from_environment(operators,rho_b):
    """F_ij=Tr(U_i rho_B U_j†), same declared environment and histories."""
    r=density(rho_b);u=[np.asarray(x,complex) for x in operators];n=len(r)
    if not u or any(x.shape!=(n,n) or not np.isfinite(x).all() or np.max(abs(x.conj().T@x-np.eye(n)))>1e-9 for x in u):raise ValueError("unitary histories")
    f=np.array([[np.trace(a@r@b.conj().T) for b in u] for a in u])
    if np.linalg.eigvalsh(hermitian(f)).min()<-1e-8 or np.max(abs(f.diagonal()-1))>1e-8:raise ArithmeticError("influence Gram")
    return f

def purify(r,tol=1e-12):
    """Schmidt factor X of rho; discard only declared numerical rank tail."""
    r=density(r);v,u=np.linalg.eigh(r)
    keep=v>tol;discard=float(v[~keep].sum())
    if not keep.any() or discard<-1e-8:raise ValueError("rank tolerance")
    x=u[:,keep]*np.sqrt(v[keep]);err=float(np.linalg.norm(r-x@x.conj().T))
    return x,dict(numerical_rank=int(keep.sum()),tolerance=tol,discarded_weight=discard,reconstruction_error=err)

def instrument(r,branches):
    """Kraus instrument; branches is list of lists of rectangular Kraus matrices."""
    r=density(r);d=len(r);groups=[[np.asarray(k,complex) for k in b] for b in branches]
    if not groups or any(not b for b in groups):raise ValueError("empty instrument branch")
    k=[a for b in groups for a in b]
    if any(a.ndim!=2 or a.shape[1]!=d or not np.isfinite(a).all() for a in k):raise ValueError("Kraus dimensions")
    if np.max(abs(sum(a.conj().T@a for a in k)-np.eye(d)))>1e-9:raise ValueError("not trace preserving")
    out=[]
    for b in groups:
        if any(a.shape!=b[0].shape for a in b):raise ValueError("branch output dimension")
        x=sum(a@r@a.conj().T for a in b);p=float(np.trace(x).real)
        if p<-1e-10:raise ArithmeticError("negative branch")
        out.append((max(0.,p),density(x/p) if p>1e-14 else None))
    if abs(sum(p for p,_ in out)-1)>1e-9:raise ArithmeticError("instrument mass")
    return out

def coarse_density(r,labels,tags):
    """Explicit isometry |i> -> |label_i>|tag_i>, then trace tag."""
    r=density(r);labels=list(labels);tags=list(tags)
    if len(labels)!=len(r) or len(tags)!=len(r) or len(set(zip(labels,tags)))!=len(r) or any(type(x)!=int or x<0 for x in labels+tags):raise ValueError("noninjective or invalid label/tag map")
    d=max(labels)+1;t=max(tags)+1
    if d*t>4096:raise ValueError("finite model dimension cap")
    out=np.zeros((d,d),complex)
    for a in range(len(r)):
        for b in range(len(r)):
            if tags[a]==tags[b]:out[labels[a],labels[b]]+=r[a,b]
    return density(out)

def qfi(r,dr,tol=1e-12):
    r=density(r);dr=hermitian(dr)
    if dr.shape!=r.shape or abs(np.trace(dr))>1e-9:raise ValueError("density tangent")
    v,u=np.linalg.eigh(r);x=u.conj().T@dr@u;den=v[:,None]+v[None,:];keep=den>tol
    if np.max(abs(x[~keep]),initial=0)>1e-8:raise ValueError("unresolved rank-changing tangent")
    return float(np.sum(2*abs(x[keep])**2/den[keep]))

def phase_probe(r,operator):
    r=density(r);o=hermitian(operator)
    if r.shape!=o.shape:raise ValueError("probe dimension")
    return float(np.trace(r@o).real)

def negativity(r,ds,df):
    r=density(r)
    if len(r)!=ds*df:raise ValueError("factor")
    pt=r.reshape(ds,df,ds,df).transpose(0,3,2,1).reshape(ds*df,ds*df)
    return float(-np.minimum(np.linalg.eigvalsh(pt),0).sum())

def gibbs(H,beta):
    """Declared H energy units, beta inverse energy; no native thermometry."""
    H=hermitian(H)
    if not math.isfinite(beta) or beta<0:raise ValueError("beta")
    e,u=np.linalg.eigh(H);x=-beta*(e-e.min());p=np.exp(x);p/=p.sum()
    r=density((u*p)@u.conj().T);mean=float(p@e);var=float(p@((e-mean)**2))
    return r,dict(energy=mean,entropy=entropy(p),variance=var,C_over_kB=beta*beta*var)

def unitary(H,t,hbar):
    H=hermitian(H)
    if not math.isfinite(t) or not math.isfinite(hbar) or hbar<=0:raise ValueError("explicit positive action scale")
    e,u=np.linalg.eigh(H);return (u*np.exp(-1j*e*t/hbar))@u.conj().T

def pushforward(p,labels,score):
    """Score conditioning, exact Fisher loss E Cov(G|Y), fixed support."""
    p=probability(p);g=np.asarray(score,float)
    if g.ndim==1:g=g[:,None]
    if len(labels)!=len(p) or len(g)!=len(p) or not np.isfinite(g).all():raise ValueError("pushforward contract")
    if np.max(abs(p@g))>1e-8:raise ValueError("score is not centered")
    names=list(dict.fromkeys(labels));pi=[];gc=[];loss=np.zeros((g.shape[1],g.shape[1]))
    for key in names:
        mask=np.array([x==key for x in labels]);mass=float(p[mask].sum())
        if mass<=0:continue
        a=(p[mask]@g[mask])/mass;z=g[mask]-a
        pi.append(mass);gc.append(a);loss+=(z.T*p[mask])@z
    pi=np.array(pi);gc=np.array(gc);F=(g.T*p)@g;Fc=(gc.T*pi)@gc
    if np.max(abs(F-Fc-loss))>1e-8 or np.linalg.eigvalsh(loss).min()<-1e-9:raise ArithmeticError("Fisher contraction")
    return dict(probabilities=pi,conditional_score=gc,fine_Fisher=F,coarse_Fisher=Fc,missing_Fisher=loss)

def graph_reduction(adjacency,mass,labels,mu,t):
    """Weighted finite graph reference, Galerkin, Schur and projected-memory defect."""
    A=np.asarray(adjacency,float);m=np.asarray(mass,float)
    if A.shape!=(len(m),len(m)) or len(labels)!=len(m) or len(m)>512 or np.any(m<=0) or np.any(A<0) or not np.isfinite(A).all() or not np.isfinite(m).all() or not np.allclose(A,A.T,atol=1e-12) or not math.isfinite(mu) or mu<=0 or not math.isfinite(t) or t<0:raise ValueError("graph contract")
    H=(np.diag(A.sum(1))-A)/np.sqrt(m[:,None]*m[None,:]);keys=list(dict.fromkeys(labels));U=np.zeros((len(m),len(keys)))
    for j,key in enumerate(keys):
        a=np.array([x==key for x in labels]);U[a,j]=np.sqrt(m[a]/m[a].sum())
    G=U.T@H@U;K=H+mu*np.eye(len(m));eff=np.linalg.inv(U.T@np.linalg.solve(K,U))
    e,u=np.linalg.eigh(H);P=(u*np.exp(-t*e))@u.T;T=U.T@P@U;defect=U.T@P@P@U-T@T
    if np.linalg.eigvalsh(defect).min()<-1e-8 or np.linalg.eigvalsh(U.T@K@U-eff).min()<-1e-8:raise ArithmeticError("graph contraction identity")
    return dict(galerkin=G,static_effective=eff,projected=T,memory_defect=defect,isometry=U)

def exact_wave(n,w,prior,c,beta,width,kind,source=None,h=0.):
    """Restricted finite wave enumeration on a declared fixed root state.
    P/Q path law for actual adaptive traces is in path_law.py; this is a reference.
    """
    from .path_law import scores,softmax
    n=np.asarray(n,int);w=np.asarray(w,float);prior=probability(prior);d=len(n)
    if len(w)!=d or len(prior)!=d or np.any(n<0) or np.any(abs(w)>n+1e-9) or type(width)!=int or width<1 or d**width>100000 or kind not in ("iid","pl","reserve") or (kind=="pl" and width>d):raise ValueError("finite wave contract")
    rows=[]
    for word in product(range(d),repeat=width):
        v=np.zeros(d,int);lp=0.;g=0.;fi=0.;ok=True
        for a in word:
            active=v==0 if kind=="pl" else np.ones(d,bool)
            if not active[a]:ok=False;break
            s=scores(n,w,v,v,prior,c,kind=="reserve");p=softmax(s,beta,active,list(range(d)),source,h)
            lp+=math.log(p[a])
            if source is not None:g+=float(a==source)-p[source];fi+=p[source]*(1-p[source])
            v[a]+=1
        if ok:rows.append((word,math.exp(lp),g,fi,tuple(v.tolist())))
    p=np.array([r[1] for r in rows]);g=np.array([r[2] for r in rows]);fi=np.array([r[3] for r in rows])
    if abs(p.sum()-1)>1e-10 or abs(p@g)>1e-10 or abs(p@(g*g)-p@fi)>1e-9:raise ArithmeticError("wave normalization or score identity")
    return dict(words=[r[0] for r in rows],p=p,score=g,conditional_fisher=fi,counts=[r[4] for r in rows])
