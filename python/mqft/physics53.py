"""Vector clock-field response on real paths; explicit conditional Q/N models."""
import math
import numpy as np
from scipy.linalg import helmert
from scipy.integrate import solve_ivp,quad
from scipy.special import softmax,logsumexp
from .physics48 import cov,jackknife
from .physics50 import sectors
from .observables import entropy
U=helmert(4,full=False).T

def vector_record(t,L):
    sp=t["spec"];B=sp["budget"];rep=t["replicas"][0];law=sp["root_law"]
    assert law["proposal"]=="p" and sp["source_action"] is None and sp["source_h"]==0
    ev=sorted(rep["events"],key=lambda x:x["root_selection"]["clock"])
    assert len(ev)==B and B%4==0 and t["root"]["board_size"]==L
    a=[e["actions"][0] for e in ev];I=np.eye(4)[sectors(L,a)]
    P=[];H=[];maxerr=0.
    for e in ev:
        d=e["root_selection"]["law"];j=d["t"];h=np.array(d["sector_logits"])
        assert np.array_equal(h,law["sector_schedule"][j])
        s=sectors(L,d["actions"]);p=np.asarray(d["p"]);q=np.bincount(s,weights=p,minlength=4)
        assert abs(q.sum()-1)<1e-10
        # Independent categorical sector tilt vs full action tilt.
        v=np.array([.17,-.21,.31,-.12]);Z=np.dot(p,np.exp(v[s]));w=p*np.exp(v[s])/Z
        q1=q*np.exp(v)/np.dot(q,np.exp(v))
        maxerr=max(maxerr,float(np.max(abs(np.bincount(s,weights=w,minlength=4)-q1))),
                   abs(float(np.sum(p[p>0]*np.log(p[p>0]/w[p>0])))-(np.log(Z)-q@v)))
        P.append(q);H.append(h)
    P=np.asarray(P);H=np.asarray(H);assert maxerr<1e-10
    G=(I-P)@U;F=np.array([U.T@(np.diag(q)-np.outer(q,q))@U for q in P])
    assert np.min(np.linalg.eigvalsh(F))>-1e-10
    M=np.asarray([e["root_value"] for e in ev]);X=np.column_stack((I.reshape(-1,4,4).mean(1)@U,M.reshape(-1,4).mean(1)))
    assert all(max(e["root_backup_clock"] for e in ev[i-4:i])<ev[i]["root_selection"]["clock"] for i in range(4,B,4))
    N=np.zeros(L*L)
    for a0,n in t["move_counts"]:N[a0]=n
    assert N.sum()==B
    prior=np.zeros(L*L)
    for a0 in rep["cuts"][0]["edges"]:prior[a0["action"]]=a0["prior"]
    return dict(L=L,B=B,law=law["kind"],seed=sp["seed"],I=I.tolist(),P=P.tolist(),field=H.tolist(),G=G.reshape(-1,4,3).sum(1).tolist(),F=F.reshape(-1,4,3,3).sum(1).tolist(),X=X.tolist(),marks=M.tolist(),counts=N.astype(int).tolist(),prior=prior.tolist(),action_history=a,legal=t["root"]["legal"],H=entropy(N[N>0]/B),max_tilt_error=maxerr)

def arm(r,delta=.3):
    h=np.asarray(r["field"]);v=h@U
    w=np.flatnonzero(np.max(abs(v),axis=1)>1e-10)
    if not len(w):return (-1,0,0,0)
    assert len(w)==4 and w[0]%4==0 and np.array_equal(w,np.arange(w[0],w[0]+4))
    x=np.rint(v[w[0]]/delta).astype(int);assert np.max(abs(v[w]-delta*x))<1e-10
    return (int(w[0]//4),*map(int,x))

def vector_responses(arms,delta=.3):
    """Separate zero-field derivatives and paired finite LR/direct contrasts.
    Complete histories jackknifed as units; mixed derivative for directions 0,1."""
    z=sorted(arms[-1,0,0,0],key=lambda r:r["seed"]);n=len(z);d=z[0]["B"]//4
    A=np.asarray([r["X"] for r in z]);G=np.asarray([r["G"] for r in z]);F=np.asarray([r["F"] for r in z])
    I=np.asarray([r["I"] for r in z]);P=np.asarray([r["P"] for r in z])
    out=[]
    def get(key):
        a=sorted(arms[key],key=lambda r:r["seed"]);assert [r["seed"] for r in a]==[r["seed"] for r in z]
        return np.array([r["X"] for r in a])
    def L(j,v):
        h=delta*U@np.asarray(v);sl=slice(4*j,4*j+4)
        return np.exp(np.sum(I[:,sl]@h-np.log(P[:,sl]@np.exp(h)),axis=1))
    for j,i,k in [(0,0,0),(3,0,0),(3,1,1),(3,2,2),(3,0,1)]:
        if (j,1,0,0) not in arms:continue
        ei=np.eye(3,dtype=int)[i];ej=np.eye(3,dtype=int)[k]
        if i==k:
            plus,minus=get((j,*ei)),get((j,*-ei))
            lp,lm=L(j,ei),L(j,-ei);w=(lp-lm)/(2*delta);w2=(lp+lm-2)/delta**2
            D=(plus-minus)/(2*delta);D2=(plus+minus-2*A)/delta**2
        else:
            w2=np.zeros(n);D2=np.zeros_like(A)
            for s in (-1,1):
                for t in (-1,1):
                    v=s*ei+t*ej;w2+=s*t*L(j,v)/(4*delta**2);D2+=s*t*get((j,*v))/(4*delta**2)
            w=np.zeros(n);D=np.zeros_like(A)
        assert np.max(abs(D[:,:j]),initial=0)==0 and np.max(abs(D2[:,:j]),initial=0)==0
        def calc(ids):
            x=A[ids];g=G[ids,j];f=F[ids,j]
            def est(v):
                r=cov(x.reshape(len(ids),-1),v).reshape(d,4);r[:j]=0;return r
            first=est(g[:,i]) if i==k else np.zeros((d,4))
            second=est(g[:,i]*g[:,k]-f[:,i,k])
            q1=est(w[ids]);q2=est(w2[ids]);d1=D[ids].mean(0);d2=D2[ids].mean(0)
            return np.stack((first,q1,d1,q1-d1,second,q2,d2,q2-d2))
        v,se=jackknife(calc,n)
        out.append(dict(j=j,i=i,k=k,n=n,values=v.tolist(),se=se.tolist(),names=["score","LR","direct","LR_minus_direct","score2","LR2","direct2","LR2_minus_direct2"]))
    # full block-pair retarded score matrix. Future raw covariance is
    # noise diagnostic; exact causal estimate uses explicit triangular mask.
    K=np.empty((d,d,4,3))
    for i in range(d):
        for j in range(d):K[i,j]=cov(A[:,i,:],G[:,j,:])
    ret=K.copy()
    for i in range(d):ret[i,i+1:]=0
    f=F.sum(1).mean(0);assert np.min(np.linalg.eigvalsh(f))>-1e-10
    fullG=G.sum(1);Y=A.mean(1);R=cov(Y,fullG);V=cov(fullG,fullG);C=cov(Y,Y)
    block=np.block([[C,R],[R.T,V]])
    assert np.linalg.eigvalsh(block).min()>-1e-10
    return dict(n=n,fisher=f.tolist(),fisher_eigen=np.linalg.eigvalsh(f).tolist(),score_cov=V.tolist(),
                full_response=R.tolist(),retarded=ret.tolist(),raw_future_norm=float(np.linalg.norm(K-ret)),
                block_PSD_min=float(np.linalg.eigvalsh(block).min()),responses=out)

def thermal_work(probs,tau,N,seed):
    """Declared cyclic reset generator. Moments and CF by ODE vs exact
    Poisson refresh simulation; no time grid approximation in MC work."""
    p=np.asarray(probs,float);assert p.shape==(3,4) and np.min(p)>0
    p=p/p.sum(1)[:,None];z=np.log(np.vstack((p,p[0])))
    k=np.array([0.,1.,2.,4.,8.,16.,32.])
    a=np.zeros((5,4));a[0]=p[0];g=np.tile(p[0],(len(k),1)).astype(complex)
    for j in range(3):
        d=z[j+1]-z[j]
        def fun(u,y):
            pi=softmax(z[j]+u*d);e=-(d-pi@d)
            A=y[:20].real.reshape(5,4);G=y[20:].reshape(len(k),4)
            da=(tau/3)*(A.sum(1)[:,None]*pi-A);da[1:]+=np.arange(1,5)[:,None]*e*A[:-1]
            dg=(tau/3)*(G.sum(1)[:,None]*pi-G)+1j*k[:,None]*e*G
            return np.r_[da.ravel(),dg.ravel()]
        sol=solve_ivp(fun,(0,1),np.r_[a.ravel(),g.ravel()],method="DOP853",rtol=2e-10,atol=2e-12)
        assert sol.success
        a=sol.y[:20,-1].real.reshape(5,4);g=sol.y[20:,-1].reshape(len(k),4)
    moments=a.sum(1);cf=g.sum(1);assert abs(cf[0]-1)<1e-9 and abs(moments[0]-1)<1e-9
    def energy(s):
        s=np.asarray(s,float);j=np.minimum((3*s).astype(int),2);u=3*s-j
        l=z[j]+u[:,None]*(z[j+1]-z[j]);return -(l-logsumexp(l,axis=1)[:,None])
    rng=np.random.default_rng(seed);state=rng.choice(4,N,p=p[0]);t=np.zeros(N);w=np.zeros(N);changed=np.zeros(N,bool);alive=np.arange(N)
    while len(alive):
        now=t[alive];nxt=np.minimum(now+rng.exponential(1/tau,len(alive)),1.)
        en=energy(nxt);eo=energy(now);w[alive]+=(en-eo)[np.arange(len(alive)),state[alive]]
        t[alive]=nxt;alive=alive[nxt<1.]
        if len(alive):
            pnew=np.exp(-energy(t[alive]));new=np.sum(rng.random(len(alive))[:,None]>np.cumsum(pnew,axis=1),axis=1);new=np.minimum(new,3)
            changed[alive]|=new!=state[alive];state[alive]=new
    # Probability of no actual state transition: exact survival, even
    # if many Poisson refreshes redraw same state. For cyclic W=0 there.
    integral=sum(np.array([quad(lambda u:softmax(z[j]+u*(z[j+1]-z[j]))[i],0,1,epsabs=1e-12)[0] for i in range(4)])/3 for j in range(3))
    atom=float(p[0]@np.exp(-tau*(1-integral)))
    assert np.max(abs(w[~changed]),initial=0)<1e-10
    mc=np.exp(1j*w[:,None]*k);m=mc.mean(0)
    se=np.sqrt(np.var(mc.real,axis=0,ddof=1)/N+np.var(mc.imag,axis=0,ddof=1)/N)
    return dict(tau=tau,N=N,seed=seed,k=k.tolist(),cf_re=cf.real.tolist(),cf_im=cf.imag.tolist(),moments=moments.tolist(),mc_re=m.real.tolist(),mc_im=m.imag.tolist(),mc_se=se.tolist(),mc_mean=float(w.mean()),mc_mean_se=float(w.std(ddof=1)/np.sqrt(N)),no_transition_prob=atom,mc_no_transition=float((~changed).mean()),zero_work_fraction=float(np.mean(abs(w)<1e-10)),max_cf_error=float(np.max(abs(m-cf))),variance=float(moments[2]-moments[1]**2)),w

def ordered_environment(n1,n2,angle):
    """Conditional coherent E state after Y record and reset. Nonorthogonal
    dephasings along X and tilted X-Z axis; second S couples to Z_E."""
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]);a=.35*n1;b=.35*n2
    plus=np.array([1.,1.])/np.sqrt(2);ps=np.outer(plus,plus)
    N=X;M=np.cos(angle)*X+np.sin(angle)*Z;zz=np.kron(Z,Z);V=np.cos(b)*np.eye(4)-1j*np.sin(b)*zz
    D=lambda r,o:(r+o@r@o)/2
    res={}
    for name,seq in (("NM",(N,M)),("MN",(M,N)),("none",()),("Z",(Z,))):
        q=np.zeros((2,2))
        for i,y in enumerate((1,-1)):
            e=np.array([np.cos(a),y*np.sin(a)],complex);r=np.outer(e,e.conj())/2
            for o in seq:r=D(r,o)
            u=V@np.kron(ps,r)@V.conj().T;s=np.trace(u.reshape(2,2,2,2),axis1=1,axis2=3)
            for j,yp in enumerate((1,-1)):
                w=np.array([1,yp*1j])/np.sqrt(2);q[i,j]=float(np.real(w.conj()@s@w))
        assert abs(q.sum()-1)<1e-10
        res[name]=float(q[0,0]+q[1,1]-q[0,1]-q[1,0])
    # Sign depends on Y convention; for |+X> exp(-ib Z) gives +Y.
    ref=float(np.sin(2*a)*np.sin(2*b)*np.cos(angle)*np.sin(angle))
    assert abs(res["NM"]-ref)<1e-10 and abs(res["MN"])<1e-10
    return dict(n1=n1,n2=n2,angle=angle,correlations=res,order_gap=res["NM"]-res["MN"],analytic=ref,residual=abs(res["NM"]-ref))


def prospective_response(arms,delta=.3):
    """Fixed split by seed label, no treatment data in training.
    Compare direct heldout first contrasts with zero, frozen local
    propensity, baseline score and finite likelihood predictions.
    Freeze conditional law on baseline paths; no future frozen effect."""
    z=sorted(arms[-1,0,0,0],key=lambda r:r["seed"]);n=len(z)
    train=z[:n//2];test=z[n//2:];ids=[r["seed"] for r in test]
    A=np.asarray([r["X"] for r in train]);P=np.asarray([r["P"] for r in train]);I=np.asarray([r["I"] for r in train]);G=np.asarray([r["G"] for r in train]);out=[]
    for j,k in [(0,0),(3,0),(3,1),(3,2)]:
        x=np.eye(3,dtype=int)[k];plus,minus=[sorted(arms[j,*v],key=lambda r:r["seed"])[n//2:] for v in (x,-x)]
        assert [r["seed"] for r in plus]==[r["seed"] for r in minus]==ids
        direct=(np.array([r["X"] for r in plus])-np.array([r["X"] for r in minus])).mean(0)/(2*delta)
        h=delta*U@x;sl=slice(4*j,4*j+4);p=P[:,sl];i=I[:,sl]
        lp=np.exp(np.sum(i@h-np.log(p@np.exp(h)),1));lm=np.exp(np.sum(-i@h-np.log(p@np.exp(-h)),1))
        def c(g):
            a=cov(A.reshape(len(train),-1),g).reshape(4,4);a[:j]=0;return a
        score=c(G[:,j,k]);lr=c((lp-lm)/(2*delta))
        fp=p*np.exp(h);fm=p*np.exp(-h);fp/=fp.sum(-1,keepdims=True);fm/=fm.sum(-1,keepdims=True)
        frozen=np.zeros((4,4));frozen[j,:3]=((fp-fm)@U).mean((0,1))/(2*delta)
        pred={"zero":np.zeros_like(frozen),"frozen":frozen,"score":score,"LR":lr}
        # Score each block equally within the post-source window; mark
        # frozen has no model of downstream marks and is not ranked there.
        rows={k0:dict(sector=float(np.mean(np.sum((a[j:,:3]-direct[j:,:3])**2,1))),
                      mark=float(np.mean((a[j:,3]-direct[j:,3])**2)) if k0!="frozen" else None) for k0,a in pred.items()}
        out.append(dict(j=j,k=k,train_n=len(train),test_n=len(test),direct=direct.tolist(),predictions={k0:v.tolist() for k0,v in pred.items()},errors=rows))
    return out
