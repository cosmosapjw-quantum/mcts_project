"""RESTORE55: longitudinal native physics; sequential conditional controls.
All native marks use the root player frame unless explicitly converted.
Conditional quantum and Gibbs models have declared external dynamics."""
import itertools
import math
import numpy as np
from scipy.special import softmax
from .physics50 import sectors
from .physics53 import U
from .physics48 import cov
from .observables import entropy,fit_beta,root_observables,marks,path_entropy,graph_and_coarse,delta

def longitudinal(t,ply):
    """Direct original Q-score thermometry and observed tree depth flow.
    No physical temperature assigned to native dimensionless scores."""
    L=t["root"]["board_size"];sp=t["spec"];rep=t["replicas"][0];B=sp["budget"]
    ev=sorted(rep["events"],key=lambda e:e["root_selection"]["clock"])
    e=rep["cuts"][-1]["edges"];v=root_observables(e,B,sp["c_puct"])
    pi=np.array(v["pi"]);q=np.array(v["action_q"]);mask=pi>0
    fits={"Q_all":fit_beta(pi,q),"Q_visited":fit_beta(pi[mask],q[mask]),
          "PUCT_all":fit_beta(pi,v["score"])}
    # Return marks from actual prefix tree. Weighted node-mean variance
    # at depth d only among trajectories surviving to depth d.
    m=marks(ev);depth=[]
    for d in sorted({len(w) for w in m}):
        a=[(n,w,s) for k,(n,w,s) in m.items() if len(k)==d]
        n,w,s=np.asarray(a).T;N=n.sum();q=w/n;mu=w.sum()/N
        within=(s.sum()-n@(q*q))/N;between=n@((q-mu)**2)/N
        total=s.sum()/N-mu**2
        assert min(within,between,total)>-1e-10 and abs(total-within-between)<1e-10
        depth.append(dict(depth=d,survivors=int(N),nodes=len(n),mean=float(mu),within=max(0.,float(within)),
                          between=max(0.,float(between)),total=max(0.,float(total))))
    for x,y in zip(depth,depth[1:]):assert y["survivors"]<=x["survivors"]
    # Matched-survival depth change removes composition from path censoring.
    flow=[]
    for d in range(1,len(depth)):
        a=[e for e in ev if len(e["actions"])>=d+1];a0=marks(a)
        def var(k):
            b=[(n,w) for p,(n,w,s) in a0.items() if len(p)==k]
            n,w=np.array(b).T;mu=w.sum()/n.sum()
            return float(n@((w/n-mu)**2)/n.sum())
        x,y=var(d),var(d+1);assert y>=x-1e-10
        flow.append(dict(depth=d,survivors=len(a),coarse=x,fine=y,beta_depth=y-x))
    # Same-root completed cut score product-rule increments, not cross-ply heat.
    cuts=[(c["local_k"],root_observables(c["edges"],c["local_k"],sp["c_puct"])) for c in rep["cuts"] if c["local_k"]>0]
    inc=[dict(k0=k,k1=l,**delta(a,b)) for (k,a),(l,b) in zip(cuts,cuts[1:])]
    p=np.array([np.bincount(sectors(L,x["root_selection"]["law"]["actions"]),weights=x["root_selection"]["law"]["p"],minlength=4) for x in ev])
    assert p.shape==(B,4) and np.allclose(p.sum(1),1)
    F=sum(U.T@(np.diag(x)-np.outer(x,x))@U for x in p)
    obs=dict(H=v["H"],H_fraction=v["H_fraction"],order=v["order_parameter"],q=v["q"],variance=v["variance"],
             within=v["within"],between=v["between"],kkt_gap=v["kkt_gap"],tangent_logdet=v["tangent_logdet"],
             support=v["support"],prior_KL=v["prior_KL"])
    return dict(L=L,B=B,ply=ply,side=t["root"]["side"],root_hash=t["root"]["state_hash"],
                black_q=t["root"]["side"]*v["q"],fits=fits,**obs,**path_entropy(ev),depth=depth,flow=flow,
                increments=inc,sector_probability=p.mean(0).tolist(),fisher_eigen=np.linalg.eigvalsh(F).tolist(),
                graph=graph_and_coarse(ev),visited_q_coverage=float(pi[mask].sum()),legal=len(e))

def contrast_record(t):
    from .physics53 import vector_record
    r=vector_record(t,t["root"]["board_size"]);h=np.asarray(r["field"]);a=h[:,0]
    assert np.max(abs(h[:,1]+a))<1e-12 and np.max(abs(h[:,2:]))==0
    w=np.flatnonzero(abs(a)>1e-12)
    if len(w):
        assert len(w)==4 and w[0]%4==0 and np.all(a[w]==a[w[0]])
        arm=(int(w[0]//4),float(a[w[0]]))
    else:arm=(-1,0.)
    r["arm"]=list(arm)
    return r

def paired_controls(arms,mark_model):
    """Early pulse weights known before future draws. Each control is a
    martingale difference using pre-selection conditional law. Train
    mark_model on a different geometric core; no test-response fit.
    Return mark control is a proxy, not an exact conditional mark."""
    base=sorted(arms[-1,0.],key=lambda r:r["seed"]);n=len(base);assert n>=2
    def arrays(a):
        a=sorted(a,key=lambda r:r["seed"]);assert [r["seed"] for r in a]==[r["seed"] for r in base]
        I=np.asarray([r["I"] for r in a]).reshape(n,4,4,4)
        P=np.asarray([r["P"] for r in a]).reshape(n,4,4,4)
        X=np.asarray([r["X"] for r in a]);M=(I-P).mean(2)
        mark=np.einsum("nbs,bs->nb",M,mark_model)
        control=np.concatenate((M@U,mark[...,None]),axis=2)
        return I,P,X,control
    I,P,X,C=arrays(base);f=np.array([1.,-1.,0.,0.]);out=[]
    for j in (0,3):
        ap,am=[arrays(arms[j,h]) for h in (.25,-.25)]
        D=(ap[2]-am[2])/.5;DC=D-(ap[3]-am[3])/.5
        # Direct controls are valid also within the perturbed block.
        # LR controls only after the source block; current W is not
        # predictable with respect to draws within source block.
        lp,lm=[np.exp(np.sum((I[:,j]@f)*h-np.log(P[:,j]@np.exp(h*f)),axis=1)) for h in (.25,-.25)]
        W=(lp-lm)/.5
        LR=X*W[:,None,None];LC=LR.copy();LC[:,j+1:]-=C[:,j+1:]*W[:,None,None]
        assert np.max(abs(D[:,:j]),initial=0)==0 and np.max(abs(DC[:,:j]),initial=0)==0
        LR[:,:j]=0;LC[:,:j]=0
        for k in range(j,4):
            a=np.stack([D[:,k],DC[:,k],LR[:,k],LC[:,k]],axis=1)
            means=a.mean(0);V=a.var(0,ddof=1)
            out.append(dict(source=j,output=k,n=n,mean=means.tolist(),se=np.sqrt(V/n).tolist(),
                            variance=V.tolist(),methods=["direct","direct_control","LR","LR_future_control"],
                            direct_diff_SE=(np.std(a[:,0]-a[:,1],axis=0,ddof=1)/np.sqrt(n)).tolist(),
                            LR_diff_SE=(np.std(a[:,2]-a[:,3],axis=0,ddof=1)/np.sqrt(n)).tolist()))
    return out

def learn_mark(records):
    """Fixed per-sector/block prior mean, shrink toward zero by one
    pseudo-observation; no feature/parameter search."""
    total=np.zeros((4,4));count=np.ones((4,4))
    for r in records:
        I=np.array(r["I"]).reshape(4,4,4);m=np.array(r["marks"]).reshape(4,4)
        total+=np.einsum("bts,bt->bs",I,m);count+=I.sum(1)
    return total/count

def restricted_vector(edges,chosen,L,law,eta,width=4):
    """Full parent state but explicitly restricted selection support.
    Exact 5^4 enumeration, independent categorical conditional update."""
    from .path_law import scores
    from scipy.special import logsumexp
    actions=[x["action"] for x in edges];idx=np.array([actions.index(x) for x in chosen])
    n=np.array([x["n"] for x in edges]);w=np.array([x["w"] for x in edges]);p=np.array([x["prior"] for x in edges])
    feat=U[sectors(L,chosen)];words=[];prob=[];G=[];F=[];counts=[]
    for word in itertools.product(range(len(idx)),repeat=width):
        if law=="pl" and len(set(word))<width:continue
        v=np.zeros(len(n),int);g=np.zeros(3);fi=np.zeros((3,3));lp=0.
        for a in word:
            s=np.asarray(scores(n,w,v,v,p,1.4,law=="reserve"),float)[idx]* (10/3)+feat@eta
            if law=="pl":s=np.where(v[idx]==0,s,-np.inf)
            q=np.exp(s-logsumexp(s));mu=q@feat;z=feat-mu
            lp+=np.log(q[a]);g+=z[a];fi+=(z.T*q)@z;v[idx[a]]+=1
        words.append(word);prob.append(np.exp(lp));G.append(g);F.append(fi);counts.append(v[idx].copy())
    q=np.array(prob);G=np.asarray(G);F=np.asarray(F);J=np.asarray(counts);Y=J@feat/width
    assert abs(q.sum()-1)<1e-10 and np.max(abs(q@G))<1e-10
    fisher=(G.T*q)@G;assert np.max(abs(fisher-np.einsum("n,nij->ij",q,F)))<1e-10
    R=((Y-q@Y).T*q)@G
    return dict(words=words,p=q,G=G,F=fisher,J=J,Y=Y,mean=q@Y,R=R,chosen=chosen)

def quantum_records(p,g,mmax=8):
    """Declared controlled-record isometry, coherent system phase.
    Pure fragments |e_s>=sqrt(1-g)|0>+sqrt(g)|s+1>; Gram exact.
    Orthogonal records at g=1; g=0 no information."""
    from .conditional import density,von_neumann
    p=np.asarray(p,float);assert np.min(p)>0 and abs(p.sum()-1)<1e-10
    phase=np.exp(1j*np.arange(4)*.37);a=np.sqrt(p)*phase
    def rho(m):
        F=np.full((4,4),(1-g)**m);np.fill_diagonal(F,1.)
        return density(np.outer(a,a.conj())*F)
    H=entropy(p);S=von_neumann(rho(mmax));rows=[]
    for m in range(mmax+1):
        chi=von_neumann(rho(m));qmi=S+chi-von_neumann(rho(mmax-m));mi=(1-(1-g)**m)*H
        assert mi<=chi+1e-10 and chi<=H+1e-10 and qmi>=-1e-10
        # Independent direct environment Gram for fragment m (small m via
        # inner-product powers, no enormous Hilbert-space construction).
        rows.append(dict(m=m,Holevo=chi,QMI=qmi,basis_MI=mi,coherence=float(np.sum(abs(rho(m)-np.diag(p)))),purity=float(np.trace(rho(m)@rho(m)).real)))
    eligible=[x["m"] for x in rows if x["basis_MI"]>=.9*H and x["m"]>0]
    # Fixed local flag measurement; disjoint fragment count, not exact SBS.
    return dict(g=g,H=H,mmax=mmax,rows=rows,redundancy_lower=mmax//min(eligible) if eligible else 0,record_threshold=.9),rho(mmax)

def gibbs_fdt(p,gamma=1.):
    """Conditional stationary reset generator L=gamma(p 1^T-I).
    Energy E/E*=-log p; field -h f, beta E*=1; rate fixed.
    Step response vs equilibrium correlation and finite +/- h."""
    from scipy.linalg import expm
    p=np.array(p,float);f=np.array([1.,-1.,0.,0.]);a=U
    m=p@a;C=(a-m).T@(p*(f-p@f))
    out=[]
    for t in (0.,.1,.5,1.,2.,4.):
        T=expm(gamma*(p[:,None]*np.ones((1,4))-np.eye(4))*t)
        corr=a.T@T@np.diag(p)@f-m*(p@f)
        assert np.max(abs(corr-np.exp(-gamma*t)*C))<1e-10
        def step(h):
            q=softmax(np.log(p)+h*f);M=expm(gamma*(q[:,None]*np.ones((1,4))-np.eye(4))*t)
            return (M@p)@a
        finite=[((step(h)-step(-h))/(2*h)).tolist() for h in (.1,.025,.0001)]
        exact=(1-np.exp(-gamma*t))*C
        assert np.max(abs(np.asarray(finite[-1])-exact))<1e-8
        out.append(dict(t=t,correlation=corr.tolist(),response=exact.tolist(),C0_minus_Ct=(C-corr).tolist(),finite=finite))
    return dict(p=p.tolist(),gamma=gamma,C0=C.tolist(),rows=out)

def pointer_dynamics(p,gamma=1.):
    from .conditional import density,von_neumann
    p=np.asarray(p,float);a=np.sqrt(p)*np.exp(1j*np.arange(4)*.37)
    R=np.outer(a,a.conj());D=np.diag(p)
    P=[np.diag(np.eye(4)[i]) for i in range(4)]
    lind=lambda x:gamma*sum(M@x@M-.5*(M@x+x@M) for M in P)
    rows=[];states=[]
    for t in (0.,.25,.5,1.,2.,4.):
        r=density(D+np.exp(-gamma*t)*(R-D))
        assert np.max(abs(lind(r)+gamma*(r-D)))<1e-10
        assert np.max(abs(np.diag(r)-p))<1e-10
        assert all(np.max(abs(lind(M)))<1e-12 for M in P)
        rows.append(dict(t=t,coherence=float(np.sum(abs(r-D))),purity=float(np.trace(r@r).real),entropy=von_neumann(r)))
        states.append(r)
    # Orthogonal environment labels and compensating swap in the
    # phase-absorbed Schmidt basis; envariance only for equal weights.
    psi=np.sqrt(p);en=[]
    for i,j in itertools.combinations(range(4),2):
        q=psi.copy();q[i],q[j]=q[j],q[i]
        err=float(np.linalg.norm(q-psi));assert abs(err-np.sqrt(2)*abs(psi[i]-psi[j]))<1e-12
        en.append(dict(i=i,j=j,swap_residual=err))
    return dict(rows=rows,pointer_stationary=True,envariance=en),states

def noneq_fdt_control(p):
    p=np.asarray(p,float);f=np.array([1.,-1.,0.,0.]);mu=softmax(np.log(p)+.7*f)
    def cc(q):
        return (U-q@U).T@(q*(f-q@f))
    C=cc(p);C0=cc(mu);out=[]
    for t in (.1,.5,1.,2.,4.):
        e=np.exp(-t);resp=(1-e)*C;naive=(1-e)*C0
        # At initial mu, connected covariance Cov(f_0,a_t)=e*C0.
        out.append(dict(t=t,response=resp.tolist(),naive_initial_FDT=naive.tolist(),gap=float(np.linalg.norm(resp-naive))))
    return dict(initial=mu.tolist(),reference=p.tolist(),rows=out)
