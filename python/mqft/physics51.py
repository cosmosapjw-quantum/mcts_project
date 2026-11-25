"""RESTORE51: informative-source diagnostics, geometry-block validation,
and explicit coherent environment intervention."""
import math
import numpy as np
from scipy.linalg import helmert
from .physics48 import cov,jackknife
from .physics50 import sector_blocks,sectors
from .conditional import density,von_neumann

def selection_diagnostics(records):
    """Pilot-selected action, independent complete-history estimates."""
    I=np.array([r["I"] for r in records],float);p=np.array([r["p"] for r in records],float)
    assert I.shape==p.shape and len(I)>=2
    G=(I-p).sum(1);F=(p*(1-p)).sum(1);H=I.sum(1);E=p.sum(1)
    def stat(x):
        return dict(mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))),total=float(x.sum()))
    return dict(n=len(I),B=I.shape[1],hits=stat(H),expected_hits=stat(E),Fisher=stat(F),score=stat(G),
                score2_minus_F=stat(G*G-F),no_hit_paths=int(np.sum(H==0)))

def exact_fisher_projection(prob):
    """Categorical conditional Fisher on action space, projected to
    four sector logits; use actual root law and no stationarity."""
    P=np.asarray(prob,float)
    assert P.ndim==2 and np.allclose(P.sum(1),1)
    F=sum(np.diag(p)-np.outer(p,p) for p in P)
    assert np.max(abs(F.sum(0)))<1e-10
    return F

def causal_env(n1,n2,theta,alpha=.35):
    """Declared coherent two-qubit memory. Actual sector counts set angles.
    S starts +X, E starts |0>, U1=exp(-i a Z_S X_E). Measure Y_S
    and reset S to +X, retain E. Rz(theta) on E then U2. Compare
    dephasing E and reset E. Joint probabilities from explicit complex
    matrices vs analytic conditional expression.
    This model is not identified by native classical data."""
    I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
    Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.])
    plus=np.array([1.,1.],complex)/np.sqrt(2);zero=np.array([1.,0.],complex)
    a,b=alpha*n1,alpha*n2
    zx=np.kron(Z,X);u=lambda t:np.cos(t)*np.eye(4)-1j*np.sin(t)*zx
    y=[np.array([1.,s*1j])/np.sqrt(2) for s in (1,-1)]
    r=np.outer(np.kron(plus,zero),np.kron(plus,zero).conj())
    v=u(a)@r@u(a).conj().T
    # trace S/E, partial transpose E
    tensor=v.reshape(2,2,2,2)
    env=np.trace(tensor,axis1=0,axis2=2);sys=np.trace(tensor,axis1=1,axis2=3)
    neg=float((np.abs(np.linalg.eigvalsh(tensor.transpose(0,3,2,1).reshape(4,4))).sum()-1)/2)
    assert abs(neg-abs(np.sin(2*a))/2)<1e-10
    R=np.diag(np.exp(-1j*theta*np.array([1.,-1.])/2))
    ps=np.outer(plus,plus.conj());p0=np.outer(zero,zero.conj())
    out={k:np.zeros((2,2)) for k in ("coherent","dephased","reset")}
    envs=[]
    for j,yy in enumerate(y):
        # nonnormalized E state after Y measurement; reset S separately.
        E=np.einsum("i,iajb,j->ab",yy.conj(),tensor,yy)
        q=float(np.trace(E).real);assert abs(q-.5)<1e-10
        envs.append(E/q)
        for name,e in (("coherent",E),("dephased",np.diag(np.diag(E))),("reset",q*p0)):
            e=R@e@R.conj().T
            a0=np.kron(ps,e);rr=u(b)@a0@u(b).conj().T
            ss=np.trace(rr.reshape(2,2,2,2),axis1=1,axis2=3)
            for k,zz in enumerate(y):out[name][j,k]=float((zz.conj()@ss@zz).real)
    corr=np.sin(2*a)*np.sin(2*b)*np.cos(theta)
    ref=(1+corr*np.array([[1.,-1.],[-1.,1.]]))/4
    err=max(float(np.max(abs(out["coherent"]-ref))),float(np.max(abs(out["dephased"]-.25))),float(np.max(abs(out["reset"]-.25))))
    assert err<1e-10
    def mi(q):
        p=q.sum(1)[:,None]*q.sum(0)[None,:];m=q>1e-15
        return float(np.sum(q[m]*np.log(q[m]/p[m])))
    return dict(n1=n1,n2=n2,theta=theta,alpha=alpha,correlation=float(corr),MI=mi(out["coherent"]),
                dephased_MI=mi(out["dephased"]),reset_MI=mi(out["reset"]),negativity=neg,
                system_entropy=von_neumann(density(sys)),env_entropy=von_neumann(density(env)),
                future_gap=float(abs(out["coherent"][0,0]/.5-out["coherent"][1,0]/.5)),
                probabilities={k:q.tolist() for k,q in out.items()},residual=err,
                scope="conditional explicit coherent E memory; compatible classical hidden-variable explanations not ruled out"),np.stack(envs)

def geometry_block_memory(records):
    """Leave-one-central-template-out, explicit size/variant dependence.
    Coarse target chosen from pilot; separate return and sector errors.
    All histories are h0 confirm only. Compare fixed ridge .05."""
    from collections import Counter
    U=helmert(4,full=False).T;out=[]
    for law in ("iid","pl","reserve"):
        rows=[]
        for r in records:
            if r["law"]!=law:continue
            a=sector_blocks(r);a=np.column_stack((a[:,:4]@U,a[:,4:]))
            geom=int(r["position"].split("_g")[1][0])
            for j in (1,2):
                rows.append(dict(position=r["position"],geom=geom,L=r["L"],rho=r["density"],state=a[j],lag=a[j-1],target=a[j+1],clock=(j+1)/4))
        for g in range(3):
            tr=[r for r in rows if r["geom"]!=g];te=[r for r in rows if r["geom"]==g]
            c=Counter(r["position"] for r in tr);w=np.array([1/(len(c)*c[r["position"]]) for r in tr])
            Y=np.array([r["target"] for r in tr]);Z=np.array([r["target"] for r in te])
            assert len(c)==12 and len({r["position"] for r in te})==6
            mods={}
            for mode in ("state","clock","memory","clock_memory"):
                def features(r):
                    x=r["state"];v=[1.,(r["L"]-17)/4,r["rho"]/.06,*x]
                    if "clock" in mode:v.extend([r["clock"],*(r["clock"]*x)])
                    if "memory" in mode:v.extend(r["lag"])
                    return v
                X=np.array([features(r) for r in tr]);T=np.array([features(r) for r in te]);pen=.05*np.eye(X.shape[1]);pen[0,0]=0
                K=np.linalg.solve(X.T@(w[:,None]*X)+pen,X.T@(w[:,None]*Y));v=T@K
                for j in range(len(v)):
                    z=.25+U@v[j,:3];u=np.sort(z)[::-1];s=np.cumsum(u)-1;k=np.flatnonzero(u-s/np.arange(1,5)>0)[-1]
                    q=np.maximum(z-s[k]/(k+1),0);v[j,:3]=q@U
                v[:,3]=np.clip(v[:,3],0,1);v[:,4]=np.clip(v[:,4],-1,1);e=(v-Z)**2
                groups=[]
                for pos in sorted({r["position"] for r in te}):
                    i=[j for j,r in enumerate(te) if r["position"]==pos];m=e[i].mean(0)
                    groups.append(dict(position=pos,sector=float(m[:3].sum()),target=float(m[3]),mark=float(m[4])))
                mods[mode]={k:float(np.mean([r[k] for r in groups])) for k in ("sector","target","mark")}
                mods[mode]["groups"]=groups
            out.append(dict(law=law,heldout_geometry=g,train_transitions=len(tr),test_transitions=len(te),models=mods))
    return out


def score_bound(records):
    """Finite-sample Cauchy-Schwarz bound and ideal-law bracket comparison.
    Y=(3 sector tangent coordinates,target mass,return mark).
    Full path G; sample variance vs predictive bracket kept distinct."""
    U=helmert(4,full=False).T
    X=np.array([sector_blocks(r).mean(0) for r in records])
    Y=np.column_stack((X[:,:4]@U,X[:,4:]))
    G=np.array([sum(r["G"]) for r in records])
    F=np.array([np.sum(np.asarray(r["p"])*(1-np.asarray(r["p"]))) for r in records])
    C=cov(Y,Y);R=cov(Y,G).ravel();V=float(np.var(G,ddof=1))
    M=V*C-np.outer(R,R);err=float(np.linalg.eigvalsh(M).min())
    assert err>=-1e-10
    den=V*np.diag(C);frac=np.divide(R**2,den,out=np.zeros(5),where=den>1e-15)
    assert np.max(frac)<=1+1e-10
    return dict(n=len(G),score_variance=V,mean_bracket=float(F.mean()),response=R.tolist(),observable_variance=np.diag(C).tolist(),
                response_fraction=frac.tolist(),min_schur_eigenvalue=err,
                scope="sample CS uses empirical score variance; ideal population bound replaces it by expected predictable Fisher, not finite-sample identity")

def continuous_cycle(probs,tau,rtol=1e-10,atol=1e-12,method="DOP853"):
    """Declared four-state Gibbs reset model, continuous log-weight
    interpolation and relaxation. Independent ODE, no discrete quenches.
    tau=gamma t_total, beta E*=1; friction integral on normalized clock."""
    from scipy.integrate import solve_ivp,quad
    from scipy.special import softmax
    from .observables import entropy
    p=np.asarray(probs,float);assert p.shape==(3,4) and np.min(p)>0
    p=p/p.sum(1)[:,None];z=np.log(np.vstack((p,p[0])))
    w=0.;fric=0.;mu=p[0].copy();nfev=0;maxmass=0.
    for j in range(3):
        d=z[j+1]-z[j]
        def eq(u):
            q=softmax(z[j]+u*d);v=d-q@d
            return q,v
        def rhs(u,y):
            q,v=eq(u)
            return np.r_[(tau/3)*(q-y[:4]),-y[:4]@v]
        sol=solve_ivp(rhs,(0,1),np.r_[mu,0.],method=method,rtol=rtol,atol=atol)
        assert sol.success
        mu=sol.y[:4,-1];w+=sol.y[4,-1];nfev+=sol.nfev
        maxmass=max(maxmass,float(np.max(abs(sol.y[:4].sum(0)-1))))
        fric+=3*quad(lambda u: float(eq(u)[0]@(eq(u)[1]**2)),0,1,epsabs=1e-12,epsrel=1e-12)[0]
    E=-np.log(p[0]);du=float((mu-p[0])@E);q=du-w;lag=float(mu@np.log(mu/p[0]))
    sig=entropy(mu)-entropy(p[0])-q
    res=max(abs(w-sig-lag),maxmass)
    assert res<1e-8 and min(mu)>0 and sig>=-1e-8
    return dict(tau=tau,work=float(w),heat=float(q),sigma=float(sig),lag=lag,friction=float(fric),tau_work=float(tau*w),relative_gap=float((tau*w-fric)/fric) if fric>1e-12 else None,
                residual=res,nfev=nfev,method=method)
