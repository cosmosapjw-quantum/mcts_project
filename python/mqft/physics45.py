"""RESTORE45: multivariate lattice action, nonthermal fluctuation law and
operational probe on actual-law reference. Explicitly scoped model inputs."""
import math,itertools
import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import logsumexp
from .physics44 import scores,softmax
from .conditional import density,von_neumann
from .framework import instrument
from .observables import entropy

def choose_target(mu,M,width,kind):
    """Integer composition in relative interior of the actual full support."""
    d=len(mu);cap=M-1 if kind=="pl" else M*width-1
    k=np.clip(np.floor(M*mu).astype(int),1,cap)
    target=M*width
    while k.sum()<target:
        i=min((i for i in range(d) if k[i]<cap),key=lambda i:2*(k[i]-M*mu[i])+1)
        k[i]+=1
    while k.sum()>target:
        i=min((i for i in range(d) if k[i]>1),key=lambda i:-2*(k[i]-M*mu[i])+1)
        k[i]-=1
    assert k.sum()==target and min(k)>0 and max(k)<=cap
    return k

def exact_coefficient(x,p,k,M):
    """Nonnegative coefficient DP on bounded partial sums, no cancellation.
    x nonnegative lattice coordinates; any path exceeding k cannot return."""
    x=np.asarray(x,int);k=np.asarray(k,int)
    shape=tuple(k+1);a=np.zeros(shape);a[(0,)*len(k)]=1.
    valid=[(v,w) for v,w in zip(x,p) if np.all(v<=k) and w>0]
    for _ in range(M):
        b=np.zeros(shape)
        for v,w in valid:
            dest=tuple(slice(int(i),int(j+1)) for i,j in zip(v,k))
            src=tuple(slice(0,int(j-i+1)) for i,j in zip(v,k))
            b[dest]+=w*a[src]
        a=b
    return float(a[tuple(k)])

def lattice_action(z,replicas=(8,16),coarse=False):
    """Full 5-action fixed-count tangent or three-sector coarse projection.
    Unit lattice coordinates delete last count. Include covolume sqrt(d)
    if expressed in orthonormal tangent coordinates. Explicit support only."""
    J=z["J"].astype(int);p=np.asarray(z["p"],float);width=z["width"];kind=z["kind"]
    if coarse:
        J=np.column_stack((J[:,0],J[:,1],J[:,2:].sum(1)))
    d=J.shape[1];X,inv=np.unique(J,axis=0,return_inverse=True)
    p=np.bincount(inv,weights=p);p/=p.sum()
    # For coarse PL the third group has sum 2..3. Choose target by taking
    # first two from the full 5-action interior target, aggregate remaining.
    A=np.vstack((np.eye(d-1),-np.ones(d-1)));U=helmert(d,full=False).T
    if abs(np.linalg.det(A.T@A)-d)>1e-10:raise ArithmeticError("lattice covolume")
    x=X[:,:-1];lp=np.log(p);mu=p@X;rows=[]
    for M in replicas:
        full=choose_target(np.asarray(z["p"])@z["J"],M,width,kind)
        k=np.r_[full[:2],sum(full[2:])] if coarse else full
        dest=k[:-1]/M
        def stats(s):
            w=lp+x@s;K=float(logsumexp(w));q=np.exp(w-K);m=q@x;y=x-m;C=(y.T*q)@y
            return K,m,C,q
        def f(s):
            K,m,_,_=stats(s);return K-s@dest,m-dest
        a=minimize(f,np.zeros(d-1),jac=True,method="BFGS",options={"gtol":1e-11,"maxiter":300})
        # BFGS often exits on precision loss; use exact residual criterion.
        s=a.x;K,m,C,q=stats(s)
        for _ in range(12):
            if np.max(abs(m-dest))<1e-10:break
            direction=np.linalg.solve(C,m-dest);old=K-s@dest
            for t in (.5**j for j in range(16)):
                ns=s-t*direction;nk,nm,nc,nq=stats(ns)
                if nk-ns@dest<=old+1e-13 and np.max(abs(nm-dest))<np.max(abs(m-dest)):
                    s,K,m,C,q=ns,nk,nm,nc,nq;break
            else:break
        res=float(np.max(abs(m-dest)));assert res<2e-8,(kind,d,M,res)
        eig=np.linalg.eigvalsh(C);assert eig.min()>0
        Ct=U.T@A@C@A.T@U
        ld=float(np.linalg.slogdet(C)[1]);ldt=float(np.linalg.slogdet(Ct)[1])
        geom=abs(ldt-ld-math.log(d));assert geom<1e-8
        rate=float(s@dest-K);leading=-M*rate
        a0=exact_coefficient(x,p,k[:-1],M)
        a1=exact_coefficient(x,q,k[:-1],M)
        assert a0>0 and a1>0
        exact=math.log(a0);dpres=abs(exact-leading-math.log(a1));assert dpres<2e-6
        loop=leading-(d-1)/2*math.log(2*math.pi*M)-ld/2
        loop_tan=leading-(d-1)/2*math.log(2*math.pi*M)-ldt/2+.5*math.log(d)
        assert abs(loop-loop_tan)<1e-8
        rows.append(dict(M=M,d=d,dimension=d-1,target=k.tolist(),mean=mu.tolist(),
          exact_log_mass=exact,leading_log_mass=leading,one_loop_log_mass=loop,
          leading_error=abs(exact-leading),one_loop_error=abs(exact-loop),determinant=float(np.exp(ld)),
          tangent_eigenvalues=np.linalg.eigvalsh(Ct).tolist(),geometry_residual=geom,dp_residual=dpres,saddle_residual=res,
          rate=rate,source=s.tolist(),lattice_covolume=math.sqrt(d)))
    return rows

def word_law(edges,chosen,kind,source,schedule,beta=1/.3):
    """Explicit frozen wave protocol with per-draw logit field; all words
    on registered common support, reservation clock as in RESTORE44."""
    actions=[x["action"] for x in edges];idx=[actions.index(a) for a in chosen];d=len(chosen);width=len(schedule)
    n=np.array([x["n"] for x in edges]);w=np.array([x["w"] for x in edges]);p=np.array([x["prior"] for x in edges])
    out={}
    for word in itertools.product(range(d),repeat=width):
        if kind=="pl" and len(set(word))<width:continue
        v=np.zeros(len(n),int);lp=0.
        for a,h in zip(word,schedule):
            s=scores(n,w,v,v,p,1.4,kind=="reserve")[idx]
            active=v[idx]==0 if kind=="pl" else np.ones(d,bool)
            z=softmax(s,beta,active,chosen,source,h)
            lp+=math.log(z[a]);v[idx[a]]+=1
        out[word]=math.exp(lp)
    assert abs(sum(out.values())-1)<1e-10
    return out

def protocol_entropy(edges,chosen,kind,source,amplitude=.6):
    """Nonthermal path-ratio reference. Reverse is a separately normalized
    reversed field schedule and reversed ordered word, same initial
    snapshot, not microscopic physical time reversal of native MCTS."""
    h=np.linspace(-amplitude,amplitude,4)
    f=word_law(edges,chosen,kind,source,h)
    r=word_law(edges,chosen,kind,source,h[::-1])
    assert set(f)=={tuple(reversed(k)) for k in r}
    p=np.array(list(f.values()));q=np.array([r[w[::-1]] for w in f]);sigma=np.log(p/q)
    # Data-processing under unordered count; ratio decomposition by partition.
    groups={}
    for word,a,b in zip(f,p,q):
        key=tuple(np.bincount(word,minlength=len(chosen)));u,v=groups.get(key,(0.,0.));groups[key]=(u+a,v+b)
    pp,qq=np.array(list(groups.values())).T
    fine=float(p@sigma);coarse=float(pp@np.log(pp/qq))
    ift=float(p@np.exp(-sigma));reverse_ift=float(q@np.exp(sigma))
    assert max(abs(ift-1),abs(reverse_ift-1))<1e-10 and fine+1e-10>=coarse>=-1e-10
    assert np.max(abs(p-q*np.exp(sigma)))<1e-12
    return dict(amplitude=amplitude,IFT=ift,reverse_IFT=reverse_ift,KL=fine,coarse_KL=coarse,
        order_loss=fine-coarse,sigma_mean=fine,sigma_var=float(p@((sigma-fine)**2)),
        min_sigma=float(sigma.min()),max_sigma=float(sigma.max()),forward_mass=float(p.sum()),reverse_mass=float(q.sum()))

def probe_from_words(words,p,source_idx,alpha=.5,echo=False):
    """Actual/declared classical records drive an explicit qubit probe.
    Two noise models have identical one-step random-unitary channels.
    Mid-sequence X pulses implement a toggling-frame echo, followed by
    undoing known controls. Measure X and Y, no native quantum inference."""
    p=np.asarray(p,float);p/=p.sum();m=np.asarray(words)==source_idx;T=m.shape[1]
    sign=np.ones(T);sign[T//2:]= -1 if echo else 1
    # Noncommuting readout, with fixed declared phase pi/4.
    theta=np.pi/4
    fi=np.prod(np.cos(2*alpha*m*sign),axis=1)
    fs=np.cos(2*alpha*(m@sign))
    out={}
    for name,c in (("iid",fi),("static",fs)):
        val=float(p@c)
        rho=density(np.array([[.5,.5*val*np.exp(-1j*theta)],[.5*val*np.exp(1j*theta),.5]]))
        plus=np.array([1.,1.])/math.sqrt(2);py=np.array([1.,1j])/math.sqrt(2)
        P=np.outer(plus,plus);Q=np.outer(py,py.conj())
        px=instrument(rho,[[P],[np.eye(2)-P]])[0][0];yy=instrument(rho,[[Q],[np.eye(2)-Q]])[0][0]
        assert abs(px-(1+val*math.cos(theta))/2)<1e-10 and abs(yy-(1+val*math.sin(theta))/2)<1e-10
        out[name]=dict(X_plus=px,Y_plus=yy,S=von_neumann(rho),coherence=val,diagonal=rho.diagonal().real.tolist())
    out["trace_distance"]=abs(out["iid"]["coherence"]-out["static"]["coherence"])/2
    out["dephased_X_plus"]=.5
    return out

def centered_estimates(A,G,D):
    """Unbiased leave-one-out baseline for E[A G] / E[A D] if E[G]=E[D]=0
    and complete histories independent. Direct empirical checks stay separate."""
    A=np.asarray(A,float);G=np.asarray(G,float);D=np.asarray(D,float);n=len(A)
    assert n>=2 and A.shape==G.shape==D.shape
    b=(A.sum()-A)/(n-1)
    return dict(raw=A*G,centered=(A-b)*G,raw_fd=A*D,centered_fd=(A-b)*D)


def cq_record_information(words,labels,alpha=.5,kind="static"):
    """Empirical classical label -> declared qubit record ensemble. Labels
    derived from disjoint future window; no claim of population MI or SBS."""
    x=np.asarray(words,int);y=np.asarray(labels,int)
    assert x.ndim==2 and len(x)==len(y) and set(y)<= {0,1} and kind in ("iid","static")
    p=np.bincount(y,minlength=2)/len(y);rho=[];ps=[]
    for k in range(2):
        if p[k]==0:rho.append(np.eye(2)/2);ps.append(.5);continue
        z=x[y==k];r=probe_from_words(z,np.ones(len(z))/len(z),1,alpha,False)[kind]
        v=r["coherence"];theta=np.pi/4
        rho.append(density(np.array([[.5,.5*v*np.exp(-1j*theta)],[.5*v*np.exp(1j*theta),.5]])))
        ps.append(r["X_plus"])
    avg=p[0]*rho[0]+p[1]*rho[1];chi=von_neumann(avg)-sum(p[i]*von_neumann(rho[i]) for i in range(2))
    q=np.array(ps);mi=entropy([p@q,1-p@q])-sum(p[i]*entropy([q[i],1-q[i]]) for i in range(2))
    assert -1e-10<=mi<=chi+1e-10 and chi<=entropy(p)+1e-10
    return dict(n=len(y),label_counts=np.bincount(y,minlength=2).tolist(),pointer_H=entropy(p),Holevo=max(0.,chi),X_readout_MI=max(0.,mi),dephased_MI=0.,scope="conditional empirical CQ ensemble; disjoint windows; no population redundancy")
