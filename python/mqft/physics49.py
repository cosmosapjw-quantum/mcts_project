"""Size-family and scheduled-source physics. Actual histories only for
native statistics; conditional models explicitly separated."""
import math
import numpy as np
from collections import defaultdict
from .observables import entropy
from .physics48 import cov,jackknife

def root_observed(trace,L,source):
    rep=trace["replicas"][0];sp=trace["spec"];B=sp["budget"]
    ev=sorted(rep["events"],key=lambda x:x["root_selection"]["clock"])
    assert len(ev)==B and trace["root"]["board_size"]==L
    assert sp["source_action"]==source and len(trace["root"]["board"])==L*L
    if B%4:raise ValueError("complete four-draw blocks")
    draws=[e["root_selection"]["law"] for e in ev];I=np.array([e["actions"][0]==source for e in ev],int)
    p=np.array([a["p"][a["actions"].index(source)] for a in draws])
    m=np.array([e["root_value"] for e in ev],float)
    counts=np.zeros(L*L)
    for a,n in trace["move_counts"]:counts[a]=n
    assert counts.sum()==B and counts[source]==sum(I)
    h=np.array([d["logit_h"] for d in draws])
    spec=sp["root_law"];expected=spec.get("logit_schedule",[spec["logit_h"]]*B)
    assert np.array_equal(h,np.asarray(expected))
    X=np.stack((I.reshape(-1,4).mean(1),m.reshape(-1,4).mean(1)),axis=-1)
    G=(I-p).reshape(-1,4).sum(1)
    # Conditional spatial root-action scores, under actual root law.
    ga=np.zeros(L*L)
    for e,d in zip(ev,draws):
        ga[d["actions"]]-=np.asarray(d["p"])
        ga[e["actions"][0]]+=1
    assert abs(ga.sum())<1e-10 and abs(ga[source]-G.sum())<1e-10
    assert all(max(e["root_backup_clock"] for e in ev[i-4:i])<ev[i]["root_selection"]["clock"] for i in range(4,B,4))
    n=counts/B;z=np.flatnonzero(n);H=entropy(n[z])
    pri=np.zeros(L*L);prior=rep["cuts"][0]["edges"]
    for x in prior:pri[x["action"]]=x["prior"]
    assert abs(pri.sum()-1)<2e-6
    return dict(L=L,B=B,law=spec["kind"],source=source,seed=sp["seed"],h=h.tolist(),I=I.tolist(),p=p.tolist(),marks=m.tolist(),G=G.tolist(),X=X.tolist(),
                counts=counts.astype(int).tolist(),prior=pri.tolist(),action_score=ga.tolist(),legal=trace["root"]["legal"],H=H,participation=float(1/(n@n)),prior_H=entropy(pri[pri>0]),
                max_count=float(n.max()),occupancy=float(n[source]),mark=float(m.mean()),
                action_history=[e["actions"][0] for e in ev],path_words=[e["actions"] for e in ev],root_hash=trace["root"]["state_hash"],
                parent_queries=len(rep["evaluations"]),child_queries=sum(not a["terminal"] for e in rep["evaluations"] for a in e["samples"]),
                sample_count=sum(len(e["samples"]) for e in rep["evaluations"]),root_log_prob=float(sum(x["log_p"] for x in draws)))

def spatial_structure(records):
    """Fixed root, fixed B: connected occupancy fluctuations across whole
    independent histories, not equilibrium spin correlation.
    Fourier normalizes by number of legal actions only when explicitly
    requested; k=0 must vanish because occupancy sums to one."""
    L=records[0]["L"];B=records[0]["B"];n=len(records);assert n>=2
    a=np.array([x["counts"] for x in records],float)/B
    u=a.mean(0);d=a-u
    g=np.array([x["action_score"] for x in records]);src=records[0]["source"]
    assert all(r["legal"]==records[0]["legal"] and r["source"]==src for r in records)
    C=(d.T@d)/(n-1);tr=float(np.trace(C))
    assert np.max(abs(C.sum(0)))<1e-12
    f=np.fft.fft2(d.reshape(n,L,L),axes=(-2,-1))
    S=(abs(f)**2).sum(0)/(n-1)
    assert abs(S[0,0])<1e-12 and abs(float(S.sum())/L**2-tr)<1e-12
    # Dimensionless low-mode spectrum; no correlation length from S(0)=0.
    low=float((S[1,0]+S[-1,0]+S[0,1]+S[0,-1])/4)
    second=float((S[1,1]+S[-1,1]+S[1,-1]+S[-1,-1])/4)
    # Fourier-space identity against explicit coordinate basis.
    yy,xx=np.indices((L,L));v=np.exp(-2j*np.pi*xx.ravel()/L)
    direct=float((v.conj()@C@v).real)
    assert abs(direct-S[0,1])<1e-10
    rr=cov(a,g[:,src]).ravel();assert abs(rr.sum())<1e-10
    # Projection to central 5x5 region plus complement. Exact covariance
    # aggregation, not a Wilsonian RG fixed point.
    c=L//2;mask=(np.abs(yy-c)<=2)&(np.abs(xx-c)<=2)
    Q=np.column_stack((mask.ravel(),~mask.ravel())).astype(float)
    y=a@Q;Cc=Q.T@C@Q
    assert np.max(abs(Cc-cov(y,y)))<1e-12
    v=np.array([r["occupancy"] for r in records]);d0=v-v.mean();v2=np.mean(d0*d0)
    binder=None if v2<1e-14 else float(1-np.mean(d0**4)/(3*v2*v2))
    return dict(L=L,B=B,n=n,law=records[0]["law"],cov_trace=tr,S_low=low,S_diag=second,S0=float(S[0,0]),rank=int(np.linalg.matrix_rank(C,tol=1e-10)),
        H=float(np.mean([r["H"] for r in records])),prior_H=float(np.mean([r["prior_H"] for r in records])),
        H_over_log_min=float(np.mean([r["H"] for r in records]))/math.log(min(B,len(records[0]["legal"]))),
        central_mass=float(y[:,0].mean()),central_variance=float(Cc[0,0]),source_cov_response=float(rr[src]),
        source_binder=binder,mean_occupancy=u.tolist(),spatial_response=rr.tolist(),structure_factor=S.tolist(),
        scope="connected complete-history occupancy; finite B, selected common geometry, n=4; no critical exponent")

def schedule_label(h):
    a=np.asarray(h,float)
    if np.all(a==a[0]):return ('constant',float(a[0]))
    v=np.flatnonzero(a)
    assert len(v)==4 and np.all(a[v]==a[v[0]]) and v.tolist() in [list(range(4)),list(range(len(a)-4,len(a)))]
    return ('early' if v[0]==0 else 'late',float(a[v[0]]))

def scheduled_response(arms):
    """Paired finite pulse contrast vs independent baseline-score prediction.
    For pulse field h on block j, prefix likelihood ratio is exactly one
    prior to j and constant after j. No future direct causal effect."""
    z=sorted(arms[('constant',0.)],key=lambda r:r["seed"]);n=len(z);d=z[0]["B"]//4
    A=np.asarray([r["X"] for r in z]);G=np.asarray([r["G"] for r in z])
    out=[]
    for kind,j in (("constant",None),("early",0),("late",d-1)):
        if (kind,.2) not in arms:continue
        plus,minus=[sorted(arms[kind,h],key=lambda r:r["seed"]) for h in (.2,-.2)]
        assert [r["seed"] for r in plus]==[r["seed"] for r in minus]==[r["seed"] for r in z]
        D=(np.asarray([r["X"] for r in plus])-np.asarray([r["X"] for r in minus]))/.4
        sel=np.arange(d*4) if j is None else np.arange(j*4,(j+1)*4)
        def lr(h):
            ans=[]
            for r in z:
                p=np.asarray(r["p"])[sel];I=np.asarray(r["I"])[sel]
                q=h*I-np.log1p(p*np.expm1(h))
                x=np.zeros(d*4);x[sel]=q
                ans.append(np.exp(np.cumsum(x)[3::4]))
            return np.asarray(ans)
        # Full-path counterfactual weight at all blocks, prefix otherwise.
        lp,lm=lr(.2),lr(-.2);v=(lp-lm)/.4
        def estimate(ids):
            x=A[ids];g=G[ids];u=v[ids]
            if j is not None:
                score=np.stack([cov(x[:,:,k],g[:,j]).ravel() for k in (0,1)],axis=-1)
                score[:j]=0
            else:
                score=np.stack([np.diag(cov(x[:,:,k],g.cumsum(1))) for k in (0,1)],axis=-1)
            pred=np.stack([np.diag(cov(x[:,:,k],u)) for k in (0,1)],axis=-1)
            obs=D[ids].mean(0)
            return np.stack([score,pred,obs,pred-obs])
        v0,se=jackknife(estimate,n)
        if j is not None:
            assert np.all(v0[:2,:j,:]==0)
        out.append(dict(kind=kind,j=j,n=n,values=v0.tolist(),se=se.tolist(),names=["score_derivative","finite_prefix_LR","direct_finite","difference"],
                        Lplus=float(lp[:,-1].mean()),Lminus=float(lm[:,-1].mean()),
                        ESSplus=float(lp[:,-1].sum()**2/(lp[:,-1]@lp[:,-1]))))
    return out

def actual_schedule_ratio(record,target_h):
    """Actual path-space protocol change (same downstream kernel, no time
    reversal). Exact log-ratio relative to actual draw probabilities."""
    p=np.array(record["p"]);I=np.array(record["I"]);dh=np.array(target_h)-np.array(record["h"])
    q=np.log1p(p*np.expm1(dh));logL=float(np.sum(dh*I-q));kl=float(np.sum(q-dh*p))
    assert kl>=-1e-10
    return dict(sigma=-logL,conditional_KL=kl,weight=math.exp(logL),balance=float(-logL-kl+np.sum(dh*(I-p))))

def spatial_record_model(records,fragments=(1,2,4,8)):
    """Conditional quantum record isometry driven by actual first root
    action sectors. Coherent pointer state over 4 spatial quadrants,
    fragment orthogonality model with explicit coupling g=.35.
    No native phase or observed environment inferred."""
    L=records[0]["L"];c=L//2
    # geometric sectors by signs of center-relative coordinates, deterministic
    # tie assignment at axes; not D4 invariant, explicitly a partition.
    actions=[a for r in records for a in r["action_history"]]
    sec=[2*int(a//L>=c)+int(a%L>=c) for a in actions]
    p=np.bincount(sec,minlength=4).astype(float);p/=p.sum()
    g=.35;total=8;rows=[]
    from .conditional import von_neumann,density
    def gram(k):
        F=np.full((4,4),(1-g)**k);np.fill_diagonal(F,1)
        return density(np.sqrt(p[:,None]*p[None,:])*F)
    H=entropy(p);S=von_neumann(gram(total))
    for m in fragments:
        chi=von_neumann(gram(m));I=S+chi-von_neumann(gram(total-m));basis=(1-(1-g)**m)*H
        assert -1e-10<=basis<=chi+1e-9 and chi<=H+1e-9 and chi<=I+1e-9
        rows.append(dict(m=m,H=H,QMI=I,Holevo=chi,basis_MI=basis))
    return dict(L=L,law=records[0]["law"],p=p.tolist(),values=rows,scope="declared controlled record state from empirical spatial action sectors, not native quantum records")
