"""RESTORE50: finite spatial tangent action, paired scheduled source,
conditional coherent record readout. Native histories remain sampling units."""
import math
import numpy as np
from scipy.linalg import helmert
from .physics48 import cov,jackknife
from .observables import entropy

def sectors(L,actions):
    a=np.asarray(actions,int);c=L//2
    return 2*(a//L>=c).astype(int)+(a%L>=c).astype(int)

def sector_blocks(r):
    L=r["L"];B=r["B"]
    z=sectors(L,r["action_history"]).reshape(-1,4)
    a=np.eye(4)[z].mean(1)
    assert np.allclose(a.sum(1),1)
    # 4 spatial sector components, target occupancy and mark.
    return np.column_stack((a,np.asarray(r["X"])))

def projected_action(records):
    """Fit on half the independent histories, test on the other.
    Rank-aware finite Gaussian reference, not native continuum 1PI."""
    a=sorted(records,key=lambda x:x["seed"]);n=len(a);assert n>=4 and n%2==0
    X=np.array([sector_blocks(r)[:,:4].mean(0) for r in a]);U=helmert(4,full=False).T
    assert np.max(abs(U.T@np.ones(4)))<1e-12 and np.max(abs(U.T@U-np.eye(3)))<1e-12
    Y=X@U;mu=Y[:n//2].mean(0);C=cov(Y[:n//2],Y[:n//2])
    Cfull=cov(X,X);Cp=U.T@Cfull@U
    assert np.max(abs(Cp-cov(Y,Y)))<1e-12 and np.max(abs(Cfull.sum(0)))<1e-12
    eig=np.linalg.eigvalsh(C);rank=int(np.sum(eig>1e-12))
    g=np.array([sum(r["G"]) for r in a])
    R=cov(Y[:n//2],g[:n//2]).ravel();d=Y[n//2:]-mu
    out=[]
    for lam in (.01,.001,.0001):
        M=C+lam*np.eye(3);H=np.linalg.inv(M);iso=np.trace(C)/3+lam
        def nll(H,ld):
            return .5*(3*math.log(2*math.pi)+ld+np.einsum("ij,jk,ik->i",d,H,d))
        x=nll(H,np.linalg.slogdet(M)[1]);y=nll(np.eye(3)/iso,3*math.log(iso))
        assert np.max(abs(H@M-np.eye(3)))<1e-9
        assert np.min(np.linalg.eigvalsh(M))>0
        out.append(dict(ridge=lam,heldout_nll=float(x.mean()),isotropic_nll=float(y.mean()),
                        delta=float((x-y).mean()),hessian_eigenvalues=np.linalg.eigvalsh(H).tolist(),
                        mixed_dual=(-H@R).tolist(),condition=float(np.linalg.cond(M))))
    # Test an independent finite output tilt using the same projected measure.
    # No claims of true native h intervention from this empirical tilt.
    v=[]
    for j in np.eye(3):
        for s in (-2.,2.):
            J=j*s
            def tilted(z):
                l=z@J;w=np.exp(l-l.max());return (w@z)/w.sum()
            train=Y[:n//2];test=Y[n//2:];m=tilted(test)
            v.append(dict(J=J.tolist(),empirical=float(np.linalg.norm(tilted(train)-m)),
                          quadratic=float(np.linalg.norm(mu+C*(1-2/n)@J-m)),zero=float(np.linalg.norm(mu-m))))
    return dict(n=n,train_n=n//2,train_rank=rank,full_rank=int(np.linalg.matrix_rank(Cp,tol=1e-12)),
                eig=eig.tolist(),mean=mu.tolist(),U=U.tolist(),C=C.tolist(),fits=out,output_tilts=v)

def pulse_label(h):
    a=np.asarray(h,float)
    if np.all(a==0):return ("base",0.,None)
    v=np.flatnonzero(a);assert len(v)==4 and v[0]%4==0 and np.array_equal(v,np.arange(v[0],v[0]+4)) and np.all(a[v]==a[v[0]])
    return ("pulse",float(a[v[0]]),int(v[0]//4))

def retarded(arms):
    z=sorted(arms[("base",0.,None)],key=lambda r:r["seed"]);n=len(z)
    A=np.array([sector_blocks(r) for r in z]);d=A.shape[1]
    G=np.array([r["G"] for r in z]);I=np.array([r["I"] for r in z]);P=np.array([r["p"] for r in z])
    FF=(P*(1-P)).reshape(n,d,4).sum(2)
    out=[]
    for j in sorted({k[2] for k in arms if k[0]=="pulse"}):
        plus,minus=[sorted(arms["pulse",h,j],key=lambda r:r["seed"]) for h in (.2,-.2)]
        assert [r["seed"] for r in plus]==[r["seed"] for r in minus]==[r["seed"] for r in z]
        Xp=np.array([sector_blocks(r) for r in plus]);Xm=np.array([sector_blocks(r) for r in minus])
        # Direct first and second finite source differences, paired.
        D=(Xp-Xm)/.4;D2=(Xp+Xm-2*A)/.04
        assert np.all(D[:,:j,:]==0) and np.all(D2[:,:j,:]==0)
        sel=slice(4*j,4*j+4);p=P[:,sel];i=I[:,sel]
        def L(h):return np.exp(np.sum(h*i-np.log1p(p*np.expm1(h)),1))
        lp,lm=L(.2),L(-.2);w=(lp-lm)/.4;w2=(lp+lm-2)/.04
        def estimate(idx):
            x=A[idx];g=G[idx,j];f=FF[idx,j]
            # Independent histories, all block/sector outputs retained.
            def projected(v):
                a=cov(x.reshape(len(idx),-1),v).reshape(d,6);a[:j]=0;return a
            score=projected(g);score2=projected(g*g-f)
            pre=projected(w[idx]);pre2=projected(w2[idx]);direct=D[idx].mean(0);direct2=D2[idx].mean(0)
            return np.stack([score,pre,direct,pre-direct,score2,pre2,direct2,pre2-direct2])
        v,se=jackknife(estimate,n)
        assert np.max(abs(v[:,:,:4].sum(-1)))<1e-9
        assert np.max(abs(v[:,:j,:]),initial=0)==0
        out.append(dict(block=j,n=n,names=["score","LR","direct","LR_minus_direct","score2","LR2","direct2","LR2_minus_direct2"],values=v.tolist(),se=se.tolist(),
                        mean_Lplus=float(lp.mean()),mean_Lminus=float(lm.mean()),ESSplus=float(lp.sum()**2/(lp@lp))))
    return out

def coherent_records(records):
    """Explicit spatially separated fragments with complementary readouts.
    |Psi>=sum sqrt(p_a)|a>|e_a>^m. Compare coherent system and
    its dephased mixture, after local heralded erasure on one fragment.
    Exact pure state reductions plus explicit complex operation.
    The p_a come from actual sector occupancy; this is a declared model."""
    from .conditional import density,von_neumann
    from .framework import qfi
    X=np.array([sector_blocks(r)[:,:4].mean(0) for r in records])
    p=X.mean(0);g=.35;a=np.sqrt(p);e=np.zeros((5,4))
    e[0,:]=math.sqrt(1-g);e[1:,:]=np.eye(4)*math.sqrt(g)
    Gram=e.T@e;assert np.allclose(np.diag(Gram),1)
    # Two separated fragments, one readout erased or sector measured.
    # Rank-one heralding |f> = normalized (|0>+sum e^{i phase_a}|a+1>)/sqrt5
    # leaves the other fragment unobserved. Phase varied independently.
    rhos=[];rows=[];G=np.diag(np.arange(4));u=np.ones(4)/2
    rho0=density(np.outer(a,a)*Gram**2);S0=von_neumann(rho0)
    for theta in (0.,.3,.6,1.2):
        f=np.r_[1.,np.exp(1j*theta*np.arange(4))]/math.sqrt(5)
        b=a*(f.conj()@e)
        raw=np.outer(b,b.conj())*Gram
        q=float(np.trace(raw).real);assert q>0
        rho=density(raw/q);diag=density(np.diag(np.diag(rho)))
        # independent explicit 4x5 pure amplitude of system+remaining fragment
        amp=b[:,None]*e.T;assert np.max(abs(amp@amp.conj().T-raw))<1e-12
        dr=-1j*(G@rho-rho@G)
        rows.append(dict(theta=theta,herald_prob=q,coherent_readout=float((u@rho@u).real),
                         dephased_readout=float((u@diag@u).real),QFI=qfi(rho,dr),dephased_QFI=qfi(diag,np.zeros((4,4))),
                         entropy=von_neumann(rho),imag_norm=float(np.linalg.norm(rho.imag))))
        rhos.append(rho)
    # Sum over a full orthonormal measurement of the first fragment,
    # Fourier basis includes f(theta=0). No-signalling if outcome ignored.
    F=np.fft.fft(np.eye(5),axis=0)/math.sqrt(5);avg=np.zeros((4,4),complex)
    for f in F:
        b=a*(f.conj()@e);avg+=np.outer(b,b.conj())*Gram
    err=float(np.max(abs(avg-rho0)));assert err<1e-12
    return dict(p=p.tolist(),unconditioned_entropy=S0,nonselective_residual=err,rows=rows,
                scope="conditional two-fragment coherent erasure, not native phase or quantum hardware"),rhos


def noise_decomposition(observed,propensity):
    """Spatial-sector mass = predictable drift + martingale increment.
    Arrays (histories, draws, sectors). Exact sample covariance identity;
    equality of sample martingale covariance and mean predictable bracket
    is only in expectation under the ideal conditional law."""
    X=np.asarray(observed,float);P=np.asarray(propensity,float)
    assert X.shape==P.shape and X.ndim==3 and X.shape[-1]==4 and len(X)>=2
    assert np.allclose(X.sum(-1),1) and np.allclose(P.sum(-1),1)
    B=X.shape[1];a=X.mean(1);b=P.mean(1);g=a-b
    C=cov(a,a);D=cov(b,b);M=cov(g,g);K=cov(b,g)
    V=np.mean([sum(np.diag(p)-np.outer(p,p) for p in q)/B**2 for q in P],0)
    err=float(np.max(abs(C-D-M-K-K.T)));assert err<1e-12
    assert np.max(abs(C.sum(0)))<1e-12 and np.max(abs(V.sum(0)))<1e-12
    return dict(n=len(X),B=B,total_trace=float(np.trace(C)),drift_trace=float(np.trace(D)),
                martingale_trace=float(np.trace(M)),cross_trace=float(2*np.trace(K)),bracket_trace=float(np.trace(V)),
                bracket_gap=float(np.trace(M-V)),score_mean=g.mean(0).tolist(),residual=err,
                covariance=C.tolist(),predictable_bracket=V.tolist(),
                scope="exact finite-sample decomposition, bracket identity only in ideal-law expectation")
