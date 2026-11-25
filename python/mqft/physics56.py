"""R56 predictable mark control, longitudinal cut diagnostics,
coherent environmental erasure and nonstationary response.
Native histories and declared conditional models are kept separate."""
import math
import numpy as np
from scipy.special import softmax
from scipy.integrate import solve_ivp,quad_vec
from .physics50 import sectors
from .physics53 import U
from .physics55 import longitudinal
from .observables import root_observables,fit_beta,entropy

def long56(t,ply):
    r=longitudinal(t,ply);s=t["spec"];rep=t["replicas"][0]
    rows=[]
    for c in rep["cuts"]:
        k=c["local_k"]
        if k<=0:continue
        v=root_observables(c["edges"],k,s["c_puct"]);p=np.asarray(v["pi"]);q=np.asarray(v["action_q"])
        m=p>0
        rows.append(dict(k=k,H=v["H"],H_fraction=v["H_fraction"],support=v["support"],beta_MF=v["beta_MF"],
                         Q_all=fit_beta(p,q),Q_visited=fit_beta(p[m],q[m]),PUCT_all=fit_beta(p,v["score"])))
    r["cuts"]=rows
    return r

def action_features(t):
    """Pre-selection categorical action features. No child/evaluation
    result or future backup used in feature construction."""
    rep=t["replicas"][0];sp=t["spec"];B=sp["budget"];L=t["root"]["board_size"];a=np.asarray(rep["root_order"])
    by={e["action"]:e for e in rep["cuts"][-1]["edges"]}
    prior=np.array([by[i]["prior"] for i in a]);sec=sectors(L,a);d=len(a)
    n=np.zeros(d);w=np.zeros(d);pending=np.zeros(d);rows=[];timeline=[]
    for e in rep["events"]:
        timeline.extend([(e["root_selection"]["clock"],0,e),(e["root_backup_clock"],1,e)])
    timeline.sort(key=lambda x:x[0]);assert [x[0] for x in timeline]==list(range(2*B))
    lookup={int(i):j for j,i in enumerate(a)}
    for clock,kind,e in timeline:
        j=lookup[e["actions"][0]]
        if kind==0:
            law=e["root_selection"]["law"];assert law["picked"]==j and law["actions"]==a.tolist()
            p=np.array(law["p"]);q=np.divide(w,n,out=np.zeros_like(w),where=n>0)
            s=np.asarray(law["scores"]);v=np.log(np.maximum(prior*d,1e-15))
            logp=np.log(np.maximum(p*d,1e-15))
            X=np.column_stack([np.ones(d),q,n/B,(n>0).astype(float),pending/4,s,v,logp,
                               (a//L-L//2)/L,(a%L-L//2)/L,U[sec],q*(n>0)])
            # Covariates clipped deterministically only in prediction;
            # do not alter the actual categorical p.
            X=np.clip(X,-20,20)
            rows.append(dict(X=X,p=p,j=j,sector=int(sec[j]),y=float(e["root_value"]),block=law["t"]//4))
            pending[j]+=e["vl_visit"][0]
        else:
            pending[j]-=e["vl_visit"][0];n[j]+=1;w[j]+=e["root_value"]
    assert len(rows)==B and not pending.any()
    return rows

def fit_mark(rows,ridge=.1):
    X=np.array([r["X"][r["j"]] for r in rows]);y=np.array([r["y"] for r in rows])
    mean=X.mean(0);scale=np.maximum(X.std(0),1e-6);mean[0]=0.;scale[0]=1.
    A=(X-mean)/scale;reg=ridge*len(y)*np.eye(A.shape[1]);reg[0,0]=0.
    b=np.linalg.solve(A.T@A+reg,A.T@y)
    sm=np.zeros((4,4));ct=np.ones((4,4))
    for r in rows:sm[r["block"],r["sector"]]+=r["y"];ct[r["block"],r["sector"]]+=1
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=b.tolist(),sector=(sm/ct).tolist(),ridge=ridge,n=len(rows))

def evaluate_mark(rows,model,L,B):
    """Exact one-step categorical expectation of frozen predictor."""
    mean=np.asarray(model["mean"]);scale=np.asarray(model["scale"]);b=np.asarray(model["coef"]);sm=np.asarray(model["sector"])
    vals=[];pred=[];y=[]
    for r in rows:
        p=r["p"];x=r["X"];j=r["j"];g=np.clip((x-mean)/scale@b,-1,1)
        # reconstruct sector features for every action from orthonormal basis
        ix=np.argmax(x[:,10:13]@U.T,axis=1);g0=sm[r["block"],ix]
        assert ix[j]==r["sector"]
        vals.append([g0[j]-p@g0,g[j]-p@g]);pred.append([0.,g0[j],g[j]]);y.append(r["y"])
    v=np.asarray(vals).reshape(B//4,4,2).mean(1)
    return dict(control=v.tolist(),pred=np.asarray(pred).tolist(),marks=y)

def finite_control_samples(arms,evals):
    base=sorted(arms[-1,0.],key=lambda r:r["seed"]);n=len(base)
    def get(key):
        rr=sorted(arms[key],key=lambda r:r["seed"]);assert [x["seed"] for x in rr]==[x["seed"] for x in base]
        X=np.array([x["X"] for x in rr]);C=np.array([evals[x["job"]]["control"] for x in rr])
        return X,C
    A,C=get((0,.25));B,E=get((0,-.25));D=(A-B)/.5;V=(C-E)/.5
    I=np.asarray([r["I"][:4] for r in base]);P=np.asarray([r["P"][:4] for r in base]);f=np.array([1.,-1.,0.,0.])
    w=[np.exp(np.sum((I@f)*h-np.log(P@np.exp(h*f)),axis=1)) for h in (.25,-.25)]
    W=(w[0]-w[1])/.5;X,C=get((-1,0.));LR=X*W[:,None,None]
    # Future only: W is measurable before future choices.
    return D[:,:,3],V,LR[:,:,3],W[:,None,None]*C

def coefficients(train):
    """Separate response coefficient for each control, objective, future
    block. No test labels. Fixed clipping and positive ridge."""
    out=[]
    for k in (0,2):
        D=np.concatenate([x[k] for x in train],axis=0);C=np.concatenate([x[k+1] for x in train],axis=0)
        v=np.zeros((4,2))
        for j in range(1,4):
            for i in range(2):
                x,y=D[:,j],C[:,j,i];v[j,i]=np.clip(np.mean((x-x.mean())*(y-y.mean()))/(np.var(y)+1e-4),-2,2)
        out.append(v)
    return np.array(out)

def score_controls(sample,coef):
    out=[]
    for k in (0,1):
        Y=sample[k*2];C=sample[k*2+1]
        for j in range(1,4):
            vals=np.column_stack((Y[:,j],Y[:,j]-C[:,j,0],Y[:,j]-coef[k,j,0]*C[:,j,0],
                                  Y[:,j]-C[:,j,1],Y[:,j]-coef[k,j,1]*C[:,j,1]))
            out.append(dict(method="direct" if k==0 else "LR",block=j,n=len(Y),mean=vals.mean(0).tolist(),
                            variance=vals.var(0,ddof=1).tolist(),names=["raw","sector_unit","sector_scaled","action_unit","action_scaled"],
                            coef=coef[k,j].tolist()))
    return out

def erasure(p,g=.5,L=8):
    """Conditional record channel with physical Kraus flag/Fourier
    measurement and feed-forward. PGM uses weighted Gram matrix.
    No native quantum phase inferred."""
    from .conditional import density,von_neumann
    p=np.asarray(p,float);assert len(p)==4 and np.min(p)>0 and abs(p.sum()-1)<1e-10
    a=np.sqrt(p)*np.exp(1j*np.arange(4)*.37);rho0=np.outer(a,a.conj());D=np.diag(p)
    M=[np.sqrt(1-g)*np.eye(4)]+[np.sqrt(g/4)*np.diag(np.exp(-2j*np.pi*j*np.arange(4)/4)) for j in range(4)]
    V=[np.eye(4)]+[np.diag(np.exp(2j*np.pi*j*np.arange(4)/4)) for j in range(4)]
    assert np.max(abs(sum(m.conj().T@m for m in M)-np.eye(4)))<1e-10
    states={};rows=[];H=entropy(p)
    for m in range(L+1):
        rho=D+(1-g)**(L-m)*(rho0-D)
        # Erasure of m accessible fragments followed by tracing L-m.
        unc=D+(1-g)**L*(rho0-D);r=rho.copy()
        if m:
            # Direct one-fragment corrected channel is identity on
            # arbitrary input, tested on the actual partially mixed state.
            for _ in range(m):r=sum(v@k@r@k.conj().T@v.conj().T for v,k in zip(V,M))
        assert np.max(abs(r-rho))<1e-10
        deph=sum(k@rho0@k.conj().T for k in M);assert np.max(abs(deph-(D+(1-g)*(rho0-D))))<1e-10
        r=density(r);states[str(m)]=r
        # PGM joint outcome probabilities from sqrt of weighted Gram.
        G=np.sqrt(p[:,None]*p[None,:])*((1-g)**m*np.ones((4,4))+(1-(1-g)**m)*np.eye(4))
        e,b=np.linalg.eigh(G);A=(b*np.sqrt(np.maximum(e,0)))@b.conj().T;joint=abs(A)**2
        assert np.max(abs(joint.sum(1)-p))<1e-10
        q=joint.sum(0);prod=p[:,None]*q[None,:];mask=joint>1e-15
        mi=float(np.sum(joint[mask]*np.log(joint[mask]/prod[mask])))
        chi=von_neumann(density(G));assert mi<=chi+1e-10
        flag=(1-(1-g)**m)*H
        rows.append(dict(m=m,PGM_MI=mi,flag_MI=flag,Holevo=chi,H=H,coherence=float(np.sum(abs(r-D))),
                         recovered_fraction=float(np.sum(abs(r-D))/np.sum(abs(rho0-D))),
                         purity=float(np.trace(r@r).real),entropy=von_neumann(r),distance_uncorrected=float(np.linalg.norm(r-unc))))
    return dict(g=g,L=L,H=H,rows=rows),states

def driven_response(probs):
    """Time-dependent Gibbs reset, instantaneous detailed balance.
    Energy and kinetic perturbations independently specified.
    Variational ODE vs exact convolution and finite differences."""
    p=np.asarray(probs,float);assert p.shape==(3,4) and p.min()>0
    p=p/p.sum(1)[:,None];z=np.log(p);f=np.array([1.,-1.,0.,0.])
    def pi(t):
        j=min(int(2*t),1);return softmax(z[j]+(2*t-j)*(z[j+1]-z[j]))
    def gamma(t):return 1+.5*t
    def u(t):return np.sin(np.pi*t)
    def v(t):return np.cos(np.pi*t)
    def rhs(t,y):
        x,ra,rb=y.reshape(3,4);q=pi(t);g=gamma(t)
        return np.r_[g*(q-x),g*(u(t)*q*(f-q@f)-ra),g*(v(t)*(q-x)-rb)]
    # Piecewise smooth protocol; integrate at the knot explicitly.
    def integrate(fun,y):
        r=[]
        for a,b in ((0.,.5),(.5,1.)):
            sol=solve_ivp(fun,(a,b),y,method="DOP853",rtol=2e-13,atol=2e-15,dense_output=True)
            assert sol.success;y=sol.y[:,-1];r.append(sol.sol)
        return lambda t:r[int(t>.5)](t)
    sol=integrate(rhs,np.r_[p[0],np.zeros(8)])
    def direct(h,kind):
        def fun(t,x):
            q=pi(t);g=gamma(t)
            if kind=="energy":q=softmax(np.log(q)+h*u(t)*f)
            else:g*=np.exp(h*v(t))
            return g*(q-x)
        return integrate(fun,p[0])
    d={k:{h:[direct(h,k),direct(-h,k)] for h in (1e-3,5e-4)} for k in ("energy","rate")}
    A=lambda t:t+.25*t*t
    out=[]
    for t in np.linspace(0,1,9):
        x,ra,rb=sol(t).reshape(3,4)
        assert abs(x.sum()-1)<1e-10 and max(abs(ra.sum()),abs(rb.sum()))<1e-10 and x.min()>-1e-12
        def src(s):
            q=pi(s);return np.stack([gamma(s)*u(s)*q*(f-q@f),gamma(s)*v(s)*(q-sol(s)[:4])])
        q=quad_vec(lambda s:np.exp(A(s)-A(t))*src(s),0,t,points=[.5] if t>.5 else None,epsabs=1e-12,epsrel=1e-10)[0] if t else np.zeros((2,4))
        fd0,fd1=[np.array([(d[k][h][0](t)-d[k][h][1](t))/(2*h) for k in ("energy","rate")]) for h in (1e-3,5e-4)]
        fd=(4*fd1-fd0)/3
        err=float(np.max(abs(q-np.array([ra,rb]))));fe=float(np.max(abs(fd-np.array([ra,rb]))))
        assert err<1e-8 and fe<1e-8,(t,err,fe)
        out.append(dict(t=float(t),mu=x.tolist(),pi=pi(t).tolist(),energy=(ra@U).tolist(),rate=(rb@U).tolist(),quadrature_error=err,finite_error=fe,raw_finite_errors=[float(np.max(abs(v-np.array([ra,rb])))) for v in (fd0,fd1)]))
    return dict(rows=out,scope="declared energy -h sin(pi t) f and independent rate exp(h cos(pi t)), nonstationary reset reference")
