"""RESTORE54: last-wave conditional response, independent predictive checks,
explicit quantum transfer channel, and reverse-work distribution."""
import math
import numpy as np
from scipy.special import softmax
from .physics53 import U,vector_record,thermal_work
from .physics48 import cov,jackknife

def arm54(r):
    h=np.asarray(r["field"]);x=h@U;d=len(h)//4
    assert np.max(abs(h[:-4]),initial=0)<1e-10 and np.max(abs(x[-4:]-x[-1]))<1e-10
    a=np.round(x[-1],10)
    assert all(v in (-.3,-.15,0.,.15,.3) for v in a)
    return tuple(float(v) for v in a)

def lastwave(arms):
    """Exact conditional sector response for iid frozen completed-root wave.
    For PL/reserve only a first-draw frozen approximation; no exactness claim.
    Baseline likelihood derivative estimates and finite direct contrasts retained."""
    z=sorted(arms[0.,0.,0.],key=lambda r:r["replicate"]);n=len(z);B=z[0]["B"];law=z[0]["law"]
    p=np.array([r["P"][-4] for r in z]);X=np.array([r["X"][-1] for r in z]);G=np.array([r["G"][-1] for r in z]);F=np.array([r["F"][-1] for r in z])
    I=np.array([r["I"][-4:] for r in z]);P=np.array([r["P"][-4:] for r in z])
    seed=[r["seed"] for r in z]
    def paths(v):
        a=sorted(arms[v],key=lambda r:r["replicate"]);assert [r["seed"] for r in a]==seed
        if law=="iid":
            pp=softmax(np.log(p)+U@np.array(v),axis=1)
            assert np.max(abs(np.array([r["P"][-4:] for r in a])-pp[:,None,:]))<1e-10
        return np.array([r["X"][-1] for r in a])
    if law=="iid":
        assert np.max(abs(P-p[:,None,:]))<1e-10
    def weight(v):
        h=U@np.array(v);return np.exp(np.sum(I@h-np.log(P@np.exp(h)),axis=1))
    def exact(v):
        return softmax(np.log(p)+U@np.array(v),axis=1)@U
    out=[]
    # General central first/diagonal second and mixed second contrasts.
    def axis(i,h):
        a=np.zeros(3);a[i]=h;return tuple(a)
    for i in range(3):
        for h in (.15,.3):
            a,b=axis(i,h),axis(i,-h)
            if a not in arms or b not in arms:continue
            v,w=paths(a),paths(b);lp,lm=weight(a),weight(b);xp,xm=exact(a),exact(b);x0=exact((0.,0.,0.))
            out.extend([(f"d{i}",h,1,i,i,(v-w)/(2*h),(lp-lm)/(2*h),(xp-xm)/(2*h),G[:,i]),
                        (f"d{i}{i}",h,2,i,i,(v+w-2*X)/h**2,(lp+lm-2)/h**2,(xp+xm-2*x0)/h**2,G[:,i]**2-F[:,i,i])])
    for i,j in ((0,1),(0,2),(1,2)):
        h=.3;keys=[]
        for s in (-1,1):
            for t in (-1,1):
                v=np.zeros(3);v[i]=s*h;v[j]=t*h;keys.append((s*t,tuple(v)))
        if not all(v in arms for s,v in keys):continue
        D=sum(s*paths(v) for s,v in keys)/(4*h*h);W=sum(s*weight(v) for s,v in keys)/(4*h*h);A=sum(s*exact(v) for s,v in keys)/(4*h*h)
        out.append((f"m{i}{j}",h,2,i,j,D,W,A,G[:,i]*G[:,j]-F[:,i,j]))
    rows=[]
    for label,h,order,i,j,D,W,A,S in out:
        def fit(ids):
            d=D[ids].mean(0);l=cov(X[ids],W[ids]).ravel();g=cov(X[ids],S[ids]).ravel();a=A[ids].mean(0)
            return np.r_[d,l,g,a,d[:3]-a,l[:3]-a,l-d]
        vals,se=jackknife(fit,n)
        # Exact derivative of the conditional softmax, evaluated per history.
        m=p@U;c=U[None,:,:]-m[:,None,:];vf=np.einsum('ns,nsi,nsj->nij',p,c,c)
        if order==1:truth=np.einsum('ns,ns,sk->nk',p,c[:,:,i],U)
        else:truth=np.einsum('ns,ns,sk->nk',p,c[:,:,i]*c[:,:,j]-vf[:,i,j,None],U)
        # First output to compare (i); all vector outputs persisted.
        direct_var=np.var(D[:,:3],axis=0,ddof=1);rb_var=np.var(A,axis=0,ddof=1)
        precision=[]
        for k in (8,16,32):
            if n>=k:
                ids=np.arange(k);v0,s0=jackknife(fit,k);precision.append(dict(n=k,value=v0.tolist(),se=s0.tolist()))
        rows.append(dict(label=label,h=h,order=order,i=i,j=j,n=n,law=law,B=B,
            direct=vals[:4].tolist(),LR=vals[4:8].tolist(),score=vals[8:12].tolist(),conditional=vals[12:15].tolist(),
            conditional_derivative=truth.mean(0).tolist(),conditional_truncation=(A-truth).mean(0).tolist(),
            direct_se=se[:4].tolist(),LR_se=se[4:8].tolist(),conditional_se=se[12:15].tolist(),
            direct_minus_conditional=vals[15:18].tolist(),direct_minus_conditional_se=se[15:18].tolist(),
            LR_minus_conditional=vals[18:21].tolist(),LR_minus_direct=vals[21:25].tolist(),LR_minus_direct_se=se[21:25].tolist(),
            direct_variance=direct_var.tolist(),conditional_variance=rb_var.tolist(),prefix_precision=precision,
            exact_iid=law=="iid",scope="last-block occupancy only; mark not conditionally integrated"))
    f=np.mean([np.sum(r["F"],axis=0) for r in z],0)
    return dict(n=n,B=B,law=law,fisher_eigen=np.linalg.eigvalsh(f).tolist(),rows=rows)

def transfer_channel(n1,n2):
    """Declared qubit-memory transfer after complete S reset, explicit
    partial iSWAP unitaries and complex Choi; no native quantum claim."""
    from .conditional import density,von_neumann
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]);I=np.eye(2)
    a,b=.35*n1,.35*n2;eta=np.sin(a)*np.sin(b)
    def U0(t):
        q=np.eye(4,dtype=complex);q[1,1]=q[2,2]=np.cos(t);q[1,2]=q[2,1]=-1j*np.sin(t);return q
    A,B=U0(a),U0(b);r0=np.diag([1.,0.])
    def channel(r,mode,theta=0.):
        e=A@np.kron(r,r0)@A.conj().T;e=np.trace(e.reshape(2,2,2,2),axis1=0,axis2=2)
        if mode=="dephase":e=np.diag(np.diag(e))
        if mode=="reset":e=np.trace(e)*r0
        R=np.diag([1.,np.exp(1j*theta)]);e=R@e@R.conj().T
        q=B@np.kron(r0,e)@B.conj().T;q=np.trace(q.reshape(2,2,2,2),axis1=1,axis2=3)
        return Z@q@Z
    # Reconstruct Choi from six actual specified input density matrices
    # (exact reference tomography); independent matrix-unit calculation.
    inp=[(I+s*q)/2 for q in (X,Y,Z) for s in (1,-1)]
    out={};states={}
    for mode in ("coherent","dephase","reset"):
        rr=[density(channel(r,mode)) for r in inp];ei=rr[0]+rr[1];ex=rr[0]-rr[1];ey=rr[2]-rr[3];ez=rr[4]-rr[5]
        J=np.block([[(ei+ez)/2,(ex+1j*ey)/2],[(ex-1j*ey)/2,(ei-ez)/2]])/2
        J0=np.block([[channel(np.array([[1,0],[0,0]],complex),mode),channel(np.array([[0,1],[0,0]],complex),mode)],[channel(np.array([[0,0],[1,0]],complex),mode),channel(np.array([[0,0],[0,1]],complex),mode)]])/2
        err=float(np.max(abs(J-J0)));assert err<1e-10
        J=density(J);tr=np.trace(J.reshape(2,2,2,2),axis1=1,axis2=3);assert np.max(abs(tr-I/2))<1e-10
        pt=J.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4);neg=float((np.abs(np.linalg.eigvalsh(pt)).sum()-1)/2)
        phi=np.array([1.,0.,0.,1.])/np.sqrt(2);Fe=float((phi@J@phi).real);Favg=float(np.mean([np.trace(x@y).real for x,y in zip(inp,rr)]))
        ref=(3+2*eta+eta**2)/6 if mode=="coherent" else (3+eta**2)/6 if mode=="dephase" else .5
        assert abs(Favg-ref)<1e-10 and abs(Favg-(2*Fe+1)/3)<1e-10
        assert abs(neg-(eta**2/2 if mode=="coherent" else 0.))<1e-10
        out[mode]=dict(Favg=Favg,Fe=Fe,negativity=neg,choi_entropy=von_neumann(J),residual=err)
        states[mode]=J
    theta=np.pi/2
    mats=[np.eye(2,dtype=complex)[i:i+1].T@np.eye(2,dtype=complex)[j:j+1] for i in range(2) for j in range(2)]
    J=density(np.block([[channel(mats[0],"coherent",theta),channel(mats[1],"coherent",theta)],[channel(mats[2],"coherent",theta),channel(mats[3],"coherent",theta)]])/2)
    assert np.max(abs(np.trace(J.reshape(2,2,2,2),axis1=1,axis2=3)-I/2))<1e-10
    probe=(I+X)/2;y=float(np.trace(Y@channel(probe,"coherent",theta)).real);yd=float(np.trace(Y@channel(probe,"dephase",theta)).real)
    assert abs(y-eta)<1e-10 and abs(yd)<1e-10
    states["phase_pi2"]=J
    return dict(n1=n1,n2=n2,transfer=float(eta),models=out,phase_probe_Y=y,dephased_probe_Y=yd,phase_imag_norm=float(np.linalg.norm(J.imag))),states

def crooks_distribution(p,tau,N,seed):
    """Independent exact-clock forward/reverse paths. Test integrated
    Crooks equality with smooth bounded functions, not histogram ratios.
    E_F[e^-w f(w)] = E_R[f(-w)]."""
    p=np.asarray(p);r=p[[0,2,1]]
    f,w=thermal_work(p,tau,N,seed);b,v=thermal_work(r,tau,N,seed+1)
    # At k=0 this is JE; use complex test functions and independent
    # MC sample variance of real/imag, no simultaneous coverage claim.
    rows=[]
    for k in (0.,1.,4.,16.,32.):
        x=np.exp((-1+1j*k)*w);y=np.exp(-1j*k*v)
        d=x.mean()-y.mean();se=math.sqrt((np.var(x.real,ddof=1)+np.var(x.imag,ddof=1)+np.var(y.real,ddof=1)+np.var(y.imag,ddof=1))/N)
        rows.append(dict(k=k,diff_re=float(d.real),diff_im=float(d.imag),se=se,z=float(abs(d)/max(se,1e-15))))
    # Unweighted first four moment checks from R53 ODE; exact
    # forward/reverse real tilt symmetry independently in caller.
    return dict(tau=tau,N=N,forward=f,reverse=b,weighted_checks=rows,max_z=max(x["z"] for x in rows)),w,v


def exact_count_reference(q,observed):
    """Actual IID last-wave q, full 35 count probabilities and exact
    binary pushforward. Independent frozen-law repetition only for
    controlled Stirling/lattice one-loop calculation."""
    from scipy.special import gammaln
    q=np.asarray(q,float);assert q.shape==(4,) and min(q)>0 and abs(q.sum()-1)<1e-10
    counts=np.array([(a,b,c,4-a-b-c) for a in range(5) for b in range(5-a) for c in range(5-a-b)])
    assert len(counts)==35
    lp=gammaln(5)-gammaln(counts+1).sum(1)+counts@np.log(q);p=np.exp(lp)
    assert abs(p.sum()-1)<1e-10
    score=(counts-4*q)@U
    mean=p@score;F=(score.T*p)@score
    ref=4*U.T@(np.diag(q)-np.outer(q,q))@U
    assert np.max(abs(mean))<1e-10 and np.max(abs(F-ref))<1e-10
    # Merge quadrants 0/1 and 2/3; exact score pushforward.
    b=counts[:,:2].sum(1);pB=np.bincount(b,weights=p,minlength=5);m=np.array([(p[b==k]@score[b==k])/pB[k] for k in range(5)])
    Fb=(m.T*pB)@m
    assert np.linalg.eigvalsh(F-Fb).min()>-1e-10
    f0=q[:2]@U[:2]/q[:2].sum();f1=q[2:]@U[2:]/q[2:].sum()
    check=4*q[:2].sum()*q[2:].sum()*np.outer(f0-f1,f0-f1)
    assert np.max(abs(Fb-check))<1e-10
    n0=np.asarray(observed,int);i=np.flatnonzero(np.all(counts==n0,axis=1));assert len(i)==1
    rows=[]
    for N in (16,32,64,128):
        n=np.floor(N*q).astype(int);r=N-n.sum();order=np.argsort(-(N*q-n),kind="stable");n[order[:r]]+=1
        assert n.sum()==N
        if min(n)==0:
            rows.append(dict(N=N,interior=False,counts=n.tolist()));continue
        x=n/N;actual=float(-gammaln(N+1)+gammaln(n+1).sum()-n@np.log(q))
        lead=float(N*np.sum(x*np.log(x/q)));loop=lead+1.5*np.log(2*np.pi*N)+.5*np.log(x).sum()
        corr=(np.sum(1/n)-1/N)/12
        # Robbins remainder for -log mass: sum r_n-r_N.
        lo=float(np.sum(1/(12*n+1))-1/(12*N));hi=float(np.sum(1/(12*n))-1/(12*N+1))
        assert lo-1e-10<=actual-loop<=hi+1e-10
        rows.append(dict(N=N,interior=True,counts=n.tolist(),exact=actual,leading=lead,one_loop=float(loop),stirling_corrected=float(loop+corr),
                         leading_error=abs(actual-lead),one_loop_error=abs(actual-loop),corrected_error=abs(actual-loop-corr),robbins=[lo,hi]))
    return dict(q=q.tolist(),counts=counts.tolist(),probabilities=p.tolist(),observed=n0.tolist(),observed_log_mass=float(lp[i[0]]),
                exact_mean=(p@counts).tolist(),F=F.tolist(),F_binary=Fb.tolist(),F_retained_trace=float(np.trace(Fb)/np.trace(F)),
                binary_prob=pB.tolist(),rows=rows,scope="exact conditional IID final-wave count law and independent frozen repetitions, no native adaptive continuum action")

def complex_work_tilt(p,tau):
    """Forward E[exp((-1+ik)w)] by complex Feynman-Kac ODE."""
    from scipy.integrate import solve_ivp
    p=np.asarray(p,float);p=p/p.sum(1)[:,None];z=np.log(np.vstack((p,p[0])))
    k=np.array([0.,1.,4.,16.,32.]);g=np.tile(p[0],(len(k),1)).astype(complex)
    for j in range(3):
        d=z[j+1]-z[j]
        def rhs(u,y):
            q=softmax(z[j]+u*d);e=-(d-q@d);a=y.reshape(len(k),4)
            return ((tau/3)*(a.sum(1)[:,None]*q-a)+(-1+1j*k[:,None])*e*a).ravel()
        sol=solve_ivp(rhs,(0,1),g.ravel(),method="DOP853",rtol=2e-10,atol=2e-12)
        assert sol.success;g=sol.y[:,-1].reshape(len(k),4)
    return g.sum(1)
