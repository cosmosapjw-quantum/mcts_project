"""RESTORE47: native finite-budget response, empirical joint action,
clock-dependent count/mark closure, declared quantum and thermal protocols.
Never identifies native physical phase, heat or continuum criticality."""
import math,itertools
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from .conditional import density,von_neumann
from .framework import qfi
from .observables import entropy

def moments(X,w=None):
    X=np.asarray(X,float);n=len(X)
    if X.ndim!=2 or n<2 or not np.isfinite(X).all():raise ValueError("observables")
    w=np.ones(n)/n if w is None else np.asarray(w,float)
    if len(w)!=n or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:raise ValueError("weights")
    w=w/w.sum();mu=w@X;d=X-mu;C=(d.T*w)@d
    return mu,C

def cg(X,J,w=None):
    X=np.asarray(X,float);J=np.asarray(J,float);w=np.ones(len(X))/len(X) if w is None else np.asarray(w,float)
    w=w/w.sum();z=np.log(w)+X@J;K=float(logsumexp(z));p=np.exp(z-K)
    mu,C=moments(X,p)
    return K,mu,C

def joint_action(train,test):
    """Empirical Legendre local curvature with explicit finite-rank gate.
    J is a source on final output observables, not native logit h.
    Fit only train; test independent seeded histories at same fixed root."""
    X=np.array([[r["occupancy"],r["mean_mark"]] for r in train])
    Y=np.array([[r["occupancy"],r["mean_mark"]] for r in test])
    G=np.array([r["G"] for r in train]);p=np.array([r["F"] for r in train])
    mu,C=moments(X);eig=np.linalg.eigvalsh(C);rank=int(np.sum(eig>1e-10))
    out=dict(n_train=len(X),n_test=len(Y),mean=mu.tolist(),covariance=C.tolist(),eigenvalues=eig.tolist(),rank=rank)
    q=G-G.mean();R=(X-mu).T@q/len(X)
    out["mixed_response"]=R.tolist()
    if rank<2:out.update(status="rank_deficient",predictions=[]);return out
    H=np.linalg.inv(C);d=-H@R
    # derivative of fitted (normalized) empirical source tilt at zero.
    # Hold observable fixed and solve the stationarity equation for J.
    def source_shift(h):
        a=np.array([h*sum(r["target_ind"])-np.log1p(np.asarray(r["conditional_p"])*np.expm1(h)).sum() for r in train])
        w=np.exp(a-logsumexp(a));J=np.zeros(2)
        for _ in range(20):
            _,m,K=cg(X,J,w);err=m-mu
            if np.linalg.norm(err)<1e-12:break
            J-=np.linalg.solve(K,err)
        if np.linalg.norm(cg(X,J,w)[1]-mu)>1e-9:raise ArithmeticError("dual stationarity")
        return J
    eps=1e-5;err=float(np.max(abs((source_shift(eps)-source_shift(-eps))/(2*eps)-d)))
    if err>2e-6:raise ArithmeticError("mixed Legendre curvature")
    rows=[]
    for J in (np.array([2.,1.]),np.array([-2.,-1.]),np.array([0.,2.]),np.array([2.,0.])):
        K,a,B=cg(X,J);Kt,b,Bt=cg(Y,J);gauss=mu+C@J
        rows.append(dict(J=J.tolist(),K=K,test_K=Kt,empirical=a.tolist(),quadratic=gauss.tolist(),test=b.tolist(),zero=mu.tolist(),
            empirical_error=float(np.linalg.norm(a-b)),quadratic_error=float(np.linalg.norm(gauss-b)),zero_error=float(np.linalg.norm(mu-b))))
    return dict(**out,status="full_rank",inverse_hessian=H.tolist(),mixed_dual=d.tolist(),duality_error=err,predictions=rows)

def scale_response(groups):
    """One record per fixed root/law/B. Paired arms within budget only."""
    out=[]
    for key,arms in sorted(groups.items()):
        pos,law,B=key;z=sorted(arms[0.],key=lambda r:r["seed"])
        n=len(z);assert n>=2
        A=np.array([[r["occupancy"],r["mean_mark"]] for r in z]);G=np.array([r["G"] for r in z]);F=np.array([r["F"] for r in z])
        mu,C=moments(A);R=(A-mu).T@(G-G.mean())/(n-1);H=G*G-F
        curv=(A-mu).T@(H-H.mean())/(n-1)
        for h in sorted(x for x in arms if x>0):
            p,m=[sorted(arms[x],key=lambda r:r["seed"]) for x in (h,-h)]
            assert [r["seed"] for r in p]==[r["seed"] for r in z]==[r["seed"] for r in m]
            x=np.array([[r["occupancy"],r["mean_mark"]] for r in p]);y=np.array([[r["occupancy"],r["mean_mark"]] for r in m])
            d=(x-y)/(2*h);direct=d.mean(0);second=((x+y-2*A)/h**2).mean(0)
            def lw(t):
                return np.array([np.exp(t*sum(r["target_ind"])-np.log1p(np.asarray(r["conditional_p"])*np.expm1(t)).sum()) for r in z])
            Lp,Lm=lw(h),lw(-h);D=(Lp-Lm)/(2*h);D2=(Lp+Lm-2)/h**2
            lr=(A-mu).T@(D-D.mean())/(n-1);lr2=(A-mu).T@(D2-D2.mean())/(n-1)
            out.append(dict(position=pos,law=law,budget=B,n=n,h=h,mean=mu.tolist(),covariance=C.tolist(),score=R.tolist(),score_curvature=curv.tolist(),finite_lr=lr.tolist(),finite_lr_curvature=lr2.tolist(),direct=direct.tolist(),direct_curvature=second.tolist(),
              G_mean=float(G.mean()),F_mean=float(F.mean()),mean_Lplus=float(Lp.mean()),mean_Lminus=float(Lm.mean()),ESSplus=float(Lp.sum()**2/(Lp@Lp)),ESSminus=float(Lm.sum()**2/(Lm@Lm)),
              direct_se=(d.std(0,ddof=1)/math.sqrt(n)).tolist()))
    return out

def wave_rows(records):
    """Complete width-4 waves only; retain actual backup clock (not wave index).
    A history is sampling unit; never use future wave in features."""
    rows=[]
    for r in records:
        B=r["k"];waves=r["complete_waves"]
        for j in range(1,len(waves)-1):
            a,b,c=waves[j-1:j+2]
            rows.append(dict(position=r["position"],seed=r["seed"],budget=B,clock=b["k_end"],law=r["law"],prev=[a["count"]/4,a["mark"]],state=[b["count"]/4,b["mark"]],target=[c["count"]/4,c["mark"]]))
    return rows

def clock_closure(train,test,law,ridge=.05):
    """Fixed ridge multivariate continuation. Equal root weights, fit train
    only. Compare explicit clock and lag additions on new D4 roots."""
    a=[r for r in wave_rows(train) if r["law"]==law];b=[r for r in wave_rows(test) if r["law"]==law]
    assert a and b and not {r["position"] for r in a}&{r["position"] for r in b}
    def features(r,mode):
        x=np.asarray(r["state"]);y=np.asarray(r["prev"]);t=r["clock"]/r["budget"];B=math.log2(r["budget"]/16)
        f=[1.,*x]
        if "clock" in mode:f.extend([t,B,*(t*x)])
        if "memory" in mode:f.extend(y)
        if mode=="clock_memory":f.extend(t*y)
        return f
    from collections import Counter
    c=Counter(r["position"] for r in a);w=np.array([1/(len(c)*c[r["position"]]) for r in a]);Y=np.array([r["target"] for r in a]);Z=np.array([r["target"] for r in b])
    out={}
    for mode in ("state","clock","memory","clock_memory"):
        X=np.array([features(r,mode) for r in a]);T=np.array([features(r,mode) for r in b]);pen=np.eye(X.shape[1])*ridge;pen[0,0]=0
        coef=np.linalg.solve(X.T@(w[:,None]*X)+pen,X.T@(w[:,None]*Y))
        pred=T@coef;pred[:,0]=np.clip(pred[:,0],0,1);pred[:,1]=np.clip(pred[:,1],-1,1)
        e=(pred-Z)**2
        groups=[]
        for pos in sorted({r["position"] for r in b}):
            m=np.array([r["position"]==pos for r in b]);groups.append(dict(position=pos,n=int(m.sum()),mse=e[m].mean(0).tolist()))
        out[mode]=dict(mse=np.mean([g["mse"] for g in groups],0).tolist(),groups=groups,coef=coef.tolist())
    return dict(law=law,n_train=len(a),n_test=len(b),train_histories=len([r for r in train if r["law"]==law]),test_histories=len([r for r in test if r["law"]==law]),models=out,scope="finite multivariate ridge clock/lag closure; chosen roots, not exact native generator")

P=np.array([np.eye(2),[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
I,X,Y,Z=P
def choi_from_ptm(T):
    # normalized Choi input first, output second.
    return sum(T[i,j]*np.kron(P[j].T,P[i]) for i in range(4) for j in range(4))/4

def process_probe(n1,n2,alpha=.5):
    """Exact six-input Pauli process tomography of declared driven qubit.
    Independent per-hit sign vs quasistatic sign; fixed Rx(pi/2) between.
    Same drive, explicit noncommuting control. Direct unitary mixture check."""
    assert 0<=n1<=4 and 0<=n2<=4
    R=(I-1j*X)/math.sqrt(2)
    def U(s,n):return np.diag(np.exp(-1j*alpha*s*n*np.array([1.,-1.])))
    def dec(r,n):
        z=r.copy();z[0,1]*=math.cos(2*alpha)**n;z[1,0]*=math.cos(2*alpha)**n
        return z
    def channel(r,kind):
        if kind=="iid":
            a=dec(r,n1);return dec(R@a@R.conj().T,n2)
        if kind=="reset_static":
            return sum((u:=U(t,n2)@R@U(s,n1))@r@u.conj().T for s,t in itertools.product((-1,1),repeat=2))/4
        if kind=="dephased":
            z=channel(np.diag(np.diag(r)),"static");return np.diag(np.diag(z))
        return sum((u:=U(s,n2)@R@U(s,n1))@r@u.conj().T for s in (-1,1))/2
    out={};mat={}
    for kind in ("iid","static","reset_static","dephased"):
        # Reconstruct from probabilities of six physical input preparations.
        obs=np.array([[[float(np.trace(P[i]@channel((I+sg*P[j])/2,kind)).real) for i in range(1,4)] for sg in (-1,1)] for j in range(1,4)])
        T=np.zeros((4,4));T[0,0]=1;T[1:,0]=obs.mean((0,1));T[1:,1:]=((obs[:,1]-obs[:,0])/2).T
        direct=np.array([[np.trace(a@channel(b,kind)).real/2 for b in P] for a in P])
        err=float(np.max(abs(direct-T)));assert err<1e-10
        J=density(choi_from_ptm(T));tp=np.trace(J.reshape(2,2,2,2),axis1=1,axis2=3)
        assert np.max(abs(tp-I/2))<1e-10
        # Independent explicit 2^(n1+n2) random unitary path sum.
        if kind=="iid":
            ref=np.zeros((2,2),complex);rho=(I+Y)/2
            for s in itertools.product((-1,1),repeat=n1+n2):
                u=U(sum(s[n1:]),1)@R@U(sum(s[:n1]),1)
                ref+=u@rho@u.conj().T/(2**(n1+n2))
            assert np.max(abs(ref-channel(rho,kind)))<1e-10
        # phase tangent for |+X>, theta=0; d rho/d theta=Y/2.
        r=channel((I+X)/2,kind);dr=channel(Y/2,kind)
        val=qfi(r,dr)
        out[kind]=dict(ptm=T.tolist(),min_choi_eig=float(np.linalg.eigvalsh(J).min()),tp_error=float(np.max(abs(tp-I/2))),
            tomography_error=err,phase_QFI=val,readout_Z_plus=float((1+np.trace(Z@r).real)/2))
        mat[kind]=J
    dist=lambda a,b:float(np.abs(np.linalg.eigvalsh(a-b)).sum()/2)
    return dict(n1=n1,n2=n2,alpha=alpha,channels=out,choi_static_iid=dist(mat["static"],mat["iid"]),choi_static_dephased=dist(mat["static"],mat["dephased"]),choi_static_reset=dist(mat["static"],mat["reset_static"])),mat

def thermal_cycle(probs,eta=.5):
    """Conditional externally driven two-state cyclic Hamiltonian.
    Energy unit E*, beta E*=1; p_t are actual recorded propensity
    averages, NOT observed energies. Quench then partial Gibbs reset.
    Reverse: partial reset then inverse quench. Exact path sum."""
    ps=np.asarray(probs,float)
    assert ps.shape==(3,) and np.all((ps>0)&(ps<1)) and 0<eta<=1
    pi=np.stack([1-ps,ps],axis=1);pi=np.vstack((pi,pi[0]))
    E=-np.log(pi);T=np.array([(1-eta)*np.eye(2)+eta*np.tile(p,(2,1)) for p in pi[1:]])
    pfin=pi[0].copy();avg=np.zeros(5);mass=0.;je=0.;maxratio=0.;rev_mass=0.
    for path in itertools.product(range(2),repeat=4):
        pf=pi[0,path[0]];pr=pi[-1,path[-1]];W=Q=0.
        for j,(a,b) in enumerate(zip(path,path[1:])):
            pf*=T[j,a,b];pr*=T[j,b,a]
            W+=E[j+1,a]-E[j,a];Q+=E[j+1,b]-E[j+1,a]
        du=E[-1,path[-1]]-E[0,path[0]]
        assert abs(du-W-Q)<1e-10
        maxratio=max(maxratio,abs(math.log(pf/pr)-W));mass+=pf;rev_mass+=pr;je+=pf*math.exp(-W)
        avg+=pf*np.array([W,Q,du,math.log(pf/pr),W*W])
    for t in T:pfin=pfin@t
    ds=entropy(pfin)-entropy(pi[0]);sigma=ds-avg[1];lag=float(pfin@np.log(pfin/pi[0]))
    res=max(abs(mass-1),abs(rev_mass-1),abs(je-1),maxratio,abs(avg[2]-avg[0]-avg[1]),abs(avg[0]-sigma-lag))
    assert res<1e-9 and sigma>=-1e-10 and lag>=-1e-10
    return dict(probabilities=ps.tolist(),eta=eta,work=float(avg[0]),heat=float(avg[1]),delta_U=float(avg[2]),delta_S=ds,entropy_production=float(sigma),final_lag_KL=lag,JE=je,crooks_residual=maxratio,balance_residual=res)

def nonstationary_projection(operators,E,R):
    """Column-state operators L_t, fixed coarse lift/marginal. Exact
    chronological discrete block elimination (no stationarity)."""
    E,R=np.asarray(E,float),np.asarray(R,float);L=[np.asarray(a,float) for a in operators]
    assert np.max(abs(R@E-np.eye(E.shape[1])))<1e-12
    Q=np.eye(len(E))-E@R
    T=[np.eye(E.shape[1])];V=E.copy();exact=[R@V];kernels=[]
    for n,ln in enumerate(L):
        A=R@ln@E;B=R@ln@Q
        pred=A@T[-1];row=[]
        for j in range(n):
            W=Q@L[j]@E
            for k in range(j+1,n):W=Q@L[k]@Q@W
            M=B@W;pred+=M@T[j];row.append(float(np.linalg.norm(M)))
        kernels.append(row);T.append(pred);V=ln@V;exact.append(R@V)
    residual=max(float(np.max(abs(x-y))) for x,y in zip(T,exact))
    assert residual<1e-10
    M=np.eye(E.shape[1]);defect=[0.]
    for n,ln in enumerate(L):
        M=(R@ln@E)@M;defect.append(float(np.linalg.norm(exact[n+1]-M)))
    return dict(residual=residual,kernel_norms=kernels,markov_defect=defect,
                projected=[x.tolist() for x in exact])

def categorical_clock_memory(train,test,law):
    """Four count/mark categories; pair-state closure fitted on actual
    previous waves. Two clock bins, fixed Dirichlet/backoff smoothing.
    Chronological nonstationary projection in a declared finite model."""
    def category(x):
        return 2*int(x[0]>0)+int(x[1]>0)
    def convert(r):
        return category(r["prev"]),category(r["state"]),category(r["target"]),int(r["clock"]/r["budget"]>=.5)
    a=[r for r in wave_rows(train) if r["law"]==law];b=[r for r in wave_rows(test) if r["law"]==law]
    assert a and b and not {r["position"] for r in a}&{r["position"] for r in b}
    d=4;C=np.zeros((2,d,d,d))
    for r in a:
        i,j,k,t=convert(r);C[t,i,j,k]+=1
    A=C.sum((0,1));M=(A+.5)/(A.sum(-1,keepdims=True)+.5*d)
    At=C.sum(1);Mt=(At+4*M[None,:,:])/(At.sum(-1,keepdims=True)+4)
    B=C.sum(0);N=(B+4*M[None,:,:])/(B.sum(-1,keepdims=True)+4)
    Nt=(C+4*Mt[:,None,:,:])/(C.sum(-1,keepdims=True)+4)
    L=[]
    for t in (0,1):
        K=np.zeros((d*d,d*d))
        for i,j in itertools.product(range(d),repeat=2):
            K[i*d+j,j*d:j*d+d]=Nt[t,i,j]
        assert np.max(abs(K.sum(1)-1))<1e-12
        L.append(K.T)
    R=np.zeros((d,d*d))
    for i,j in itertools.product(range(d),repeat=2):R[j,i*d+j]=1
    E=R.T/d
    # Illustrative continuation with specified clock; initial hidden
    # distribution uniform given coarse state, not identified from test.
    proj=nonstationary_projection([L[0]]*4+[L[1]]*4,E,R)
    v=defaultdict(list)
    for r in b:
        i,j,k,t=convert(r)
        v[r["position"]].append([-math.log(x) for x in (M[j,k],Mt[t,j,k],N[i,j,k],Nt[t,i,j,k])])
    groups=[dict(position=pos,n=len(x),nll=np.mean(x,0).tolist()) for pos,x in sorted(v.items())]
    return dict(law=law,train=len(a),test=len(b),models=["state","clock","memory","clock_memory"],
                nll=np.mean([x["nll"] for x in groups],0).tolist(),groups=groups,
                early_late_counts=C.tolist(),projection=proj,scope="fitted two-clock four-category pair chain with declared uniform conditional lift")
