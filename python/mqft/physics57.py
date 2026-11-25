"""Dedicated original outputs: native score-energy, noncyclic work,
signed entropy balance, and explicit classical heat-engine references.
All E,w,q in E*, beta E*=1 unless specified. Q positive into system."""
import math
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.integrate import solve_ivp
from .observables import root_observables,delta,entropy,kl

def score_energy(t,ply):
    """Actual root snapshots, no cross-root energy conservation claim."""
    c=t["spec"]["c_puct"];k0=None;prev=None;out=[]
    for cut in t["replicas"][0]["cuts"]:
        k=cut["local_k"]
        if not k:continue
        a=root_observables(cut["edges"],k,c)
        p=np.asarray(a["pi"]);Q=np.asarray(a["action_q"]);score=np.asarray(a["score"])
        eq=-float(p@Q);es=-float(p@score);ex=-float(p@(score-Q))
        assert abs(eq+ex-es)<1e-10 and abs(eq+a["q"])<1e-10
        z=dict(k=k,E_Q=eq,E_PUCT=es,E_bonus=ex,H=a["H"],side=t["root"]["side"],fit_beta=a["beta"],fit_status=a["beta_status"])
        if prev is not None:
            v=delta(prev,a);z.update(v);assert abs(es-out[-1]["E_PUCT"]-v["field"]-v["redistribution"])<1e-10
        out.append(z);prev=a
    return dict(ply=ply,rows=out,scope="native dimensionless root-player score proxies, no thermal energy")

def levels(prob):
    p=np.asarray(prob,float);assert p.shape[1]==4 and p.min()>0
    p=p/p.sum(1)[:,None]
    # Gauge declared by anchoring state 3 at zero energy across protocol.
    E=-np.log(p/p[:,3,None]);assert np.max(abs(E[:,3]))<1e-12
    return E

def protocol(E,tau,N,seed,initial=None):
    """Piecewise-linear energies, homogeneous exact Poisson refresh.
    Forward mean moments/complex tilt by ODE, work and heat by path.
    Time points align to all knots (4 substeps per segment)."""
    E=np.asarray(E,float);m=len(E)-1;assert E.shape==(m+1,4) and tau>0
    pi0=softmax(-E[0]);pi1=softmax(-E[-1]);mu0=pi0 if initial is None else np.asarray(initial,float)
    assert abs(mu0.sum()-1)<1e-10 and mu0.min()>0
    grid=np.linspace(0,1,4*m+1);k=np.array([0.,1.,4.,16.])
    # columns: moments a0..a4, work-tilted lambda=1-ik, ordinary CF,
    # cumulative mean heat and entropy production.
    a=np.zeros((5,4));a[0]=mu0
    g=np.tile(mu0,(len(k),1)).astype(complex);cf=g.copy()
    y=np.r_[a.ravel(),g.ravel(),cf.ravel(),0.,0.]
    rows=[]
    def evalrow(t,y,e,der):
        A=y[:20].real.reshape(5,4);mu=A[0];w=A.sum(1);G=y[20:20+4*len(k)].reshape(len(k),4)
        C=y[20+4*len(k):20+8*len(k)].reshape(len(k),4)
        p=softmax(-e);F=-logsumexp(-e);F0=-logsumexp(-E[0])
        qdot=tau*(p-mu);heat=float(y[-2].real);sigma=float(y[-1].real);S=entropy(mu)
        dS=-float(qdot@np.log(mu));Qdot=float(qdot@e)
        J=tau*(p[:,None]*mu[None,:]-p[None,:]*mu[:,None])
        logratio=np.log(p[:,None]*mu[None,:]/(p[None,:]*mu[:,None]))
        flux=.5*float(np.sum(J*logratio));assert flux>=-1e-10
        assert abs(flux-(dS-Qdot))<1e-10
        first=float(mu@e-mu0@E[0]-w[1]-heat)
        balance=float(S-entropy(mu0)-heat-sigma)
        rel=kl(mu,p);dF=F-F0;diss=float(w[1]-dF)
        free_balance=diss-(sigma+rel-kl(mu0,pi0))
        if initial is None:
            assert np.max(abs(G[0]-np.exp(-dF)*p))<1e-8
        assert max(abs(first),abs(balance),abs(free_balance))<1e-8
        return dict(t=float(t),mu=mu.tolist(),pi=p.tolist(),E=e.tolist(),F=float(F),deltaF=float(dF),W=float(w[1]),Q=heat,U=float(mu@e),S=S,sigma=sigma,rel_entropy=rel,
                    entropy_rate=dS,heat_rate=Qdot,flux_sigma=flux,first_residual=first,entropy_residual=balance,free_residual=float(free_balance),
                    moments=w.tolist(),cf_re=C.sum(1).real.tolist(),cf_im=C.sum(1).imag.tolist(),tilt_re=G.sum(1).real.tolist(),tilt_im=G.sum(1).imag.tolist())
    for j in range(m):
        d=E[j+1]-E[j]
        def rhs(u,y):
            e=E[j]+u*d;p=softmax(-e);A=y[:20].real.reshape(5,4)
            G=y[20:20+4*len(k)].reshape(len(k),4);C=y[20+4*len(k):20+8*len(k)].reshape(len(k),4)
            rr=tau/m;da=rr*(A.sum(1)[:,None]*p-A);da[1:]+=np.arange(1,5)[:,None]*d*A[:-1]
            dg=rr*(G.sum(1)[:,None]*p-G)+(-1+1j*k[:,None])*d*G
            dc=rr*(C.sum(1)[:,None]*p-C)+1j*k[:,None]*d*C
            dp=rr*(p-A[0]);heat=float(dp@e);sig=-float(dp@np.log(A[0]))-heat
            return np.r_[da.ravel(),dg.ravel(),dc.ravel(),heat,sig]
        sol=solve_ivp(rhs,(0,1),y,method="DOP853",rtol=2e-11,atol=2e-13,t_eval=np.linspace(0,1,5))
        assert sol.success
        for u,z in zip(sol.t,sol.y.T):
            if j and u==0:continue
            rows.append(evalrow((j+u)/m,z,E[j]+u*d,d*m))
        y=sol.y[:,-1]
    assert len(rows)==len(grid)
    # Conditional MC: exact Poisson refresh, not a discretized jump chain.
    # At each observation boundary exploit exponential memorylessness.
    rng=np.random.default_rng(seed);state=rng.choice(4,N,p=mu0);initial_state=state.copy()
    W=np.zeros((N,len(grid)));Q=np.zeros_like(W);X=np.zeros_like(W,dtype=np.uint8);X[:,0]=state
    def energy(t):
        t=np.asarray(t);j=np.minimum((m*t).astype(int),m-1);u=m*t-j
        return E[j]+u[:,None]*(E[j+1]-E[j])
    for l in range(1,len(grid)):
        a,b=grid[l-1:l+1];now=np.full(N,a);active=np.arange(N);w=W[:,l-1].copy();q=Q[:,l-1].copy()
        while len(active):
            nxt=np.minimum(now[active]+rng.exponential(1/tau,len(active)),b)
            e0=energy(now[active]);e1=energy(nxt);i=state[active]
            w[active]+=(e1-e0)[np.arange(len(active)),i];now[active]=nxt
            keep=nxt<b;active=active[keep]
            if len(active):
                e=e1[keep];p=softmax(-e,axis=1);new=np.minimum(np.sum(rng.random(len(active))[:,None]>np.cumsum(p,axis=1),1),3)
                q[active]+=e[np.arange(len(active)),new]-e[np.arange(len(active)),state[active]];state[active]=new
        W[:,l]=w;Q[:,l]=q;X[:,l]=state
    en=np.array([energy(np.array([t]))[0] for t in grid]);DU=en[np.arange(len(grid))[None,:],X]-E[0,initial_state,None]
    path_error=float(np.max(abs(DU-W-Q),initial=0));assert path_error<1e-10
    mu=np.asarray(rows[-1]["mu"]);sig=np.log(mu0[initial_state])-np.log(mu[state])-Q[:,-1]
    w=W[:,-1];dF=rows[-1]["deltaF"];exact=rows[-1]["W"];se=float(w.std(ddof=1)/np.sqrt(N))
    e=np.exp(-(w-dF));es=float(e.std(ddof=1)/np.sqrt(N));mc=dict(W=float(w.mean()),W_se=se,W_z=float(abs(w.mean()-exact)/max(se,1e-15)),
       Q=float(Q[:,-1].mean()),sigma=float(sig.mean()),IFT=float(np.exp(-sig).mean()),JE=float(e.mean()),JE_se=es,JE_z=float(abs(e.mean()-1)/max(es,1e-15)),path_residual=path_error)
    return dict(tau=tau,N=N,k=k.tolist(),initial=mu0.tolist(),levels=E.tolist(),rows=rows,mc=mc),dict(W=W,Q=Q,X=X,sigma=sig)

def compare_noncyclic(E,tau,N,seed):
    f,F=protocol(E,tau,N,seed);r,R=protocol(E[::-1],tau,N,seed+1)
    dF=f["rows"][-1]["deltaF"];assert abs(dF+r["rows"][-1]["deltaF"])<1e-12
    a=f["rows"][-1];b=r["rows"][-1];z=(np.asarray(a["tilt_re"])+1j*np.asarray(a["tilt_im"]))*np.exp(dF)
    # Crooks shifted work w-dF: characteristic function of reflected reverse shifted work.
    k=np.asarray(f["k"]);lhs=z*np.exp(-1j*k*dF);rhs=(np.array(b["cf_re"])-1j*np.array(b["cf_im"]))*np.exp(-1j*k*dF)
    err=float(np.max(abs(lhs-rhs)));assert err<1e-8
    wf=F["W"][:,-1]-dF;wr=R["W"][:,-1]+dF;tests=[]
    for k0 in k:
        x=np.exp((-1+1j*k0)*wf);y=np.exp(-1j*k0*wr);d=x.mean()-y.mean()
        se=math.sqrt((x.real.var(ddof=1)+x.imag.var(ddof=1)+y.real.var(ddof=1)+y.imag.var(ddof=1))/N)
        tests.append(dict(k=float(k0),re=float(d.real),im=float(d.imag),se=se,z=float(abs(d)/max(se,1e-15))))
    return dict(deltaF=dF,forward=f,reverse=r,crooks_error=err,weighted_checks=tests),F,R

def cramer(E,tau):
    """Independent repeats of declared protocol: SCGF & parametric
    Cramer rate, not a rate inferred from a single finite histogram."""
    E=np.asarray(E,float);m=len(E)-1;lam=np.linspace(-8,8,65);p=softmax(-E[0]);y=np.r_[np.tile(p,(len(lam),1)).ravel(),np.zeros(4*len(lam))]
    for j in range(m):
        d=E[j+1]-E[j]
        def rhs(u,y):
            p=softmax(-E[j]-u*d);g,b=y.reshape(2,len(lam),4);L=lambda x:tau/m*(x.sum(1)[:,None]*p-x)+lam[:,None]*d*x
            return np.r_[L(g).ravel(),(L(b)+g*d).ravel()]
        sol=solve_ivp(rhs,(0,1),y,method="DOP853",rtol=2e-10,atol=2e-12);assert sol.success;y=sol.y[:,-1]
    g,b=y.reshape(2,len(lam),4);psi=np.log(g.sum(1));x=b.sum(1)/g.sum(1);I=lam*x-psi
    assert np.min(np.diff(x))>=-1e-10 and I.min()>-1e-9 and abs(psi[32])<1e-9
    return dict(lambda_grid=lam.tolist(),psi=psi.tolist(),x=x.tolist(),I=I.tolist())

def heat_engines(p,duration):
    """Declared diagonal 4-level Otto engine, finite bath contacts,
    ideal reversible Carnot reference. No native P,V,or bath claim."""
    e=levels(np.asarray([p]))[0];e=e-e.min();r=1.5;bc=1.;bh=.5;u=np.exp(-duration)
    pc=softmax(-bc*e);ph=softmax(-bh*r*e)
    # after cold/hot contacts in the periodic state
    c=(pc+u*ph)/(1+u);h=(ph+u*pc)/(1+u)
    assert np.max(abs(h-(u*c+(1-u)*ph)))<1e-12 and np.max(abs(c-(u*h+(1-u)*pc)))<1e-12
    qh=float(r*e@(h-c));qc=float(e@(c-h));win=float((r-1)*e@(c-h));sig=-bh*qh-bc*qc
    assert abs(win+qh+qc)<1e-12 and qh>=-1e-12 and sig>=-1e-12
    eta=-win/qh if qh>1e-14 else None
    if eta is not None:assert abs(eta-(1-1/r))<1e-10 and eta<=1-bh/bc
    # Quasistatic hot isotherm scale 1.8->1.2, scaled adiabats to cold.
    sA=entropy(softmax(-bh*1.8*e));sB=entropy(softmax(-bh*1.2*e));dS=sB-sA
    cqH=dS/bh;cqC=-dS/bc;cw=-cqH-cqC
    return dict(duration=duration,levels=e.tolist(),cold=c.tolist(),hot=h.tolist(),Qh=qh,Qc=qc,W_in=win,sigma=sig,eta=eta,
                carnot=dict(S=[sA,sB,sB,sA,sA],T=[1/bh,1/bh,1/bc,1/bc,1/bh],Qh=cqH,Qc=cqC,W_in=cw,eta=1-bh/bc),
                otto=dict(scale=[1,r,r,1,1],force=[-float(e@c),-float(e@c),-float(e@h),-float(e@h),-float(e@c)]))

def equilibrium_susceptibility(p):
    """Static FDT for declared conjugate energy perturbation -h f."""
    p=np.asarray(p,float);f=np.array([1.,-1.,0.,0.])
    C=float(p@(f-p@f)**2)
    m=lambda h:float(softmax(np.log(p)+h*f)@f)
    raw=[(m(h)-m(-h))/(2*h) for h in (.001,.0005)]
    extrap=(4*raw[1]-raw[0])/3
    assert abs(extrap-C)<1e-9
    return dict(variance=C,derivative=extrap,error=abs(extrap-C),raw=raw)
