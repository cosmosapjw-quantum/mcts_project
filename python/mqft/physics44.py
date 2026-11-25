"""RESTORE44 actual-path response and finite conditional physical reference.
Native full-support histories vs restricted exact wave and explicit Q/thermal
models are disjoint evidence classes. No inferred native phase or bath.
"""
import math
import itertools
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from .path_law import scores,softmax
from .framework import pushforward,gram_density,negativity
from .conditional import density,reduce,von_neumann
from .observables import entropy

def native_record(trace,source):
    """Direct conditional score and counterfactual LR on same augmented path."""
    rep=trace["replicas"][0];ev=sorted(rep["events"],key=lambda x:x["root_selection"]["clock"])
    draws=[x["root_selection"]["law"] for x in ev];k=len(ev)
    assert all(a["proposal"]=="p" and a["beta_p"]==a["beta_q"] and a["source"]==source for a in draws)
    prob=[];ind=[];logp=0.
    for x,a in zip(ev,draws):
        j=a["actions"].index(source);p=a["p"][j];I=int(x["actions"][0]==source)
        assert a["picked"]==a["actions"].index(x["actions"][0])
        prob.append(p);ind.append(I);logp+=math.log(a["p"][a["picked"]])
    p=np.array(prob);I=np.array(ind);G=float(np.sum(I-p));F=float(np.sum(p*(1-p)))
    downstream=sum(s["log_path_prob"] for a in rep["evaluations"] for s in a["samples"])
    h=draws[0]["logit_h"]
    w={str(x):float(np.exp(np.sum(x*I-np.log1p(p*np.expm1(x))))) for x in (-.3,.3)} if h==0 else {}
    marks=np.array([e["root_value"] for e in ev])
    # One repeat is a complete adaptive history. Fragments are time windows, not independent samples.
    count=int(I.sum())
    return dict(k=k,h=h,count=count,occupancy=count/k,mean_mark=float(marks.mean()),G=G,F=F,
                log_root_prob=logp,log_downstream_prob=downstream,log_augmented=logp+downstream,
                weights=w,score_steps=(I-p).tolist(),conditional_p=p.tolist(),target_ind=I.tolist(),
                marks=marks.tolist(),first_phase=ev[0]["phase"])

def mean_se(x):
    a=np.asarray(x,float)
    return dict(mean=float(a.mean()),se=float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else None,n=len(a))

def paired_summary(rows):
    """Finite pilot; per root independent seed repeats, opening-level summaries separate."""
    groups=defaultdict(dict)
    for r in rows:
        key=(r["position"],r["law"],r["replicate"])
        if r["h"] in groups[key]:raise ValueError("duplicate h")
        groups[key][r["h"]]=r
    by=defaultdict(list)
    for (pos,law,rep),arms in groups.items():
        if set(arms)!={-.3,0.,.3}:raise ValueError("missing h")
        a,b,c=arms[-.3],arms[0.],arms[.3]
        assert len({x["seed"] for x in arms.values()})==1
        z=dict(rep=rep,G=b["G"],F=b["F"],fisher_gap=b["G"]**2-b["F"],Lminus=b["weights"]["-0.3"],Lplus=b["weights"]["0.3"])
        for A in ("occupancy","mean_mark"):
            z[A]=b[A];z[A+"_LR"]=b[A]*b["G"]
            z[A+"_FD"]=(c[A]-a[A])/.6
            z[A+"_ISFD"]=b[A]*(z["Lplus"]-z["Lminus"])/.6
        by[pos,law].append(z)
    out=[]
    for (pos,law),rr in sorted(by.items()):
        a={k:mean_se([r[k] for r in rr]) for k in rr[0] if k!="rep"}
        for name in ("occupancy","mean_mark"):
            v=np.array([r[name] for r in rr]);g=np.array([r["G"] for r in rr])
            a[name+"_cov"]=float(np.cov(v,g,ddof=1)[0,1])
        for name in ("Lminus","Lplus"):
            w=np.array([r[name] for r in rr]);a[name+"_ESS"]=float(w.sum()**2/(w@w))
        out.append(dict(position=pos,law=law,stats=a,paired=rr))
    return out

def exact_wave_full_state(edges,chosen,width,kind,source,h=0.,beta=1/.3):
    """Restriction only in selection normalization; keep full parent visits,
    priors and reservation arithmetic. Source field is uniform per draw."""
    actions=[int(x["action"]) for x in edges];indices=[actions.index(a) for a in chosen]
    if source not in chosen or len(set(chosen))!=len(chosen) or width<1 or width>len(chosen) or kind not in ("iid","pl","reserve"):raise ValueError("reference support")
    n=np.array([x["n"] for x in edges],int);w=np.array([x["w"] for x in edges]);p=np.array([x["prior"] for x in edges])
    d=len(chosen);words=[];probs=[];G=[];F=[];cnt=[]
    for word in itertools.product(range(d),repeat=width):
        if kind=="pl" and len(set(word))!=width:continue
        v=np.zeros(len(n),int);g=np.zeros(d);fi=np.zeros((d,d));lp=0.
        for a in word:
            score=scores(n,w,v,v,p,1.4,kind=="reserve")[indices]
            active=(v[indices]==0) if kind=="pl" else np.ones(d,bool)
            u=softmax(score,beta,active,chosen,source,h)
            lp+=math.log(u[a]);g-=u;g[a]+=1;fi+=np.diag(u)-np.outer(u,u);v[indices[a]]+=1
        words.append(word);probs.append(math.exp(lp));G.append(g);F.append(fi);cnt.append(v[indices])
    p=np.array(probs);g=np.array(G);f=np.array(F);j=np.array(cnt,float);mu=p@j
    C=((j-mu).T*p)@(j-mu);R=((j-mu).T*p)@g;I=(g.T*p)@g;Fbar=np.einsum("i,ijk->jk",p,f)
    errors=dict(norm=abs(float(p.sum())-1),score=float(np.max(abs(p@g))),fisher=float(np.max(abs(I-Fbar))),
                tangent=float(max(np.max(abs(x@np.ones(d))) for x in (C,R,I))))
    assert max(errors.values())<1e-10
    return dict(words=words,p=p,G=g,F=Fbar,J=j,mean=mu,C=C,R=R,I=I,errors=errors,chosen=chosen,source=source,width=width,kind=kind)

def projected_information(z):
    """Different coarse maps; probability pushforward and score conditioning."""
    p=z["p"];G=z["G"];d=len(z["chosen"]);src=z["chosen"].index(z["source"])
    maps={"ordered":z["words"],"prefix1":[w[:1] for w in z["words"]],
          "unordered":list(map(tuple,z["J"].astype(int))),"target_count":list(z["J"][:,src].astype(int)),"constant":[0]*len(p)}
    out={}
    for name,labels in maps.items():
        a=pushforward(p,labels,G);I0=float(np.trace(a["fine_Fisher"]));Ic=float(np.trace(a["coarse_Fisher"]))
        out[name]=dict(fine_trace=I0,coarse_trace=Ic,lost_trace=I0-Ic,
            relative=Ic/I0 if I0>1e-14 else None,entropy=entropy(a["probabilities"]),
            psd_residual=float(min(0,np.linalg.eigvalsh(a["missing_Fisher"]).min())))
    return out

def saddlepoint(z,replicas=(4,8,16,32,64)):
    """Exact lattice convolution vs leading large-deviation and one-loop prefactor.
    Independent replicated frozen waves, NOT native growing tree."""
    i=z["chosen"].index(z["source"]);j=z["J"][:,i].astype(int)
    a=np.bincount(j,weights=z["p"]);lo,hi=np.flatnonzero(a>0)[[0,-1]]
    assert np.all(a[lo:hi+1]>0) and hi>lo
    js=np.arange(lo,hi+1);logp=np.log(a[lo:hi+1]);mu=float(js@a[lo:hi+1])
    out=[]
    for m in replicas:
        conv=np.array([1.])
        for _ in range(m):conv=np.convolve(conv,a)
        assert abs(conv.sum()-1)<1e-10
        # interior point nearest mean with at least one spacing from endpoints.
        k=max(m*lo+1,min(m*hi-1,round(m*mu)));x=k/m
        def tilted(s):
            l=logp+s*js;K=float(logsumexp(l));q=np.exp(l-K);mean=float(q@js);var=float(q@((js-mean)**2))
            return K,mean,var
        s=brentq(lambda y:tilted(y)[1]-x,-100,100,xtol=1e-13)
        K,_,var=tilted(s);rate=s*x-K
        exact=math.log(conv[k]);tree=-m*rate;loop=tree-.5*math.log(2*math.pi*m*var)
        out.append(dict(M=m,k=k,x=x,source=s,exact_log_mass=exact,leading_log_mass=tree,one_loop_log_mass=loop,
                         leading_error=abs(exact-tree),one_loop_error=abs(exact-loop),variance=var))
    return out

def quantum_memory(z,alphas=(0.,.25,.5,.75,1.)):
    """Declared nonthermal two-draw quantum lift on full ordered-pair basis."""
    assert z["width"]==2
    d=len(z["chosen"]);p=np.zeros(d*d);src=z["chosen"].index(z["source"])
    for word,x in zip(z["words"],z["p"]):p[word[0]*d+word[1]]=x
    words=np.array(list(itertools.product(range(d),repeat=2)));m=(words==src).astype(float);theta=.37*(m[:,0]+2*m[:,1])
    # Extend to zero-probability words; Gram positive on full basis.
    delta=m[:,None,:]-m[None,:,:];out=[];states={}
    for a in alphas:
        fi=np.prod(np.cos(a*delta),axis=-1);fs=np.cos(a*delta.sum(-1))
        ri=gram_density(p,theta,fi);rs=gram_density(p,theta,fs)
        # Independent random-unitary representation, CP; static retains sign across both steps.
        psi=np.sqrt(p)*np.exp(1j*theta)
        ii=sum(np.outer(v,v.conj()) for s in itertools.product((-1,1),repeat=2) for v in [psi*np.exp(1j*a*(m@np.array(s)))])/4
        ss=sum(np.outer(v,v.conj()) for s in (-1,1) for v in [psi*np.exp(1j*a*s*m.sum(1))])/2
        assert max(np.max(abs(ii-ri)),np.max(abs(ss-rs)))<1e-10
        redi=reduce(ri,[d,d],[0]);reds=reduce(rs,[d,d],[0]);u=np.ones(d*d)/d
        distance=lambda x,y:float(np.sum(abs(np.linalg.eigvalsh(x-y)))/2)
        out.append(dict(alpha=a,S_iid=von_neumann(ri),S_static=von_neumann(rs),reduced_S_iid=von_neumann(redi),reduced_S_static=von_neumann(reds),
          trace_distance=distance(ri,rs),reduced_trace_distance=distance(redi,reds),probe_iid=float((u@ri@u).real),probe_static=float((u@rs@u).real),
          negativity_iid=negativity(ri,d,d),negativity_static=negativity(rs,d,d),diag_error=float(max(np.max(abs(ri.diagonal()-p)),np.max(abs(rs.diagonal()-p)))),
          imag_norm=float(np.linalg.norm(rs.imag)),purity_iid=float(np.trace(ri@ri).real),purity_static=float(np.trace(rs@rs).real)))
        states[f"iid_{a}"]=ri;states[f"static_{a}"]=rs
    return out,states

def record_fragments(z,g=.35,L=6):
    """Pure controlled isometry, exact reduced spectra via Gram.
    Same first-draw distribution as exact wave, declared coupling."""
    p=np.bincount([w[0] for w in z["words"]],weights=z["p"],minlength=len(z["chosen"]));d=len(p)
    def gram(m):
        F=np.full((d,d),(1-g)**m);np.fill_diagonal(F,1.)
        return np.sqrt(p[:,None]*p[None,:])*F
    s=lambda m:von_neumann(gram(m));S=s(L);Hp=entropy(p)
    out=[]
    for m in range(1,L+1):
        chi=s(m);qmi=S+chi-s(L-m);basis=(1-(1-g)**m)*Hp
        assert -1e-10<=basis<=chi+1e-9 and chi<=Hp+1e-9 and chi<=qmi+1e-9
        out.append(dict(fragment=m,QMI=qmi,Holevo=chi,basis_MI=basis,H_pointer=Hp,S_system=S,
             coherence_l1=float(np.sum(abs(gram(L)-np.diag(p)))),env_overlap=(1-g)**m))
    return out

def thermal_reference(z,drive=.3):
    """Explicit finite conditional equilibrium model; dimensionless E/E_unit.
    H1=H0-drive*J, beta E_unit=1. Quench then Gibbs reset; reverse order.
    Exact JE and path-resolved Crooks, nontrivial DeltaF. Not native heat/work."""
    p=z["p"];a=z["chosen"].index(z["source"]);J=z["J"][:,a];E0=-np.log(p);E1=E0-drive*J
    lz=float(logsumexp(-E1));q=np.exp(-E1-lz);dF=-lz;W=E1-E0
    je=float(p@np.exp(-W));F=p[:,None]*q[None,:];R=q[None,:]*q[:,None] # reverse start j~q, reset i~q
    crooks=np.log(F)-np.log(R)-(W[:,None]-dF)
    Q=E1[None,:]-E1[:,None];dU=E1[None,:]-E0[:,None]
    dS=entropy(q)-entropy(p);heat=float(np.sum(F*Q));work=float(p@W);sigma=dS-heat
    kl=float(p@np.log(p/q))
    assert max(abs(je-math.exp(-dF)),float(np.max(abs(crooks))),float(np.max(abs(dU-W[:,None]-Q))),abs(sigma-kl))<1e-10
    temp=[]
    for beta in (.25,.5,1.,2.,4.):
        b=np.exp(-beta*E0-logsumexp(-beta*E0));U=float(b@E0);C=float(beta**2*(b@((E0-U)**2)))
        temp.append(dict(beta=beta,U=U,S=entropy(b),C_over_kB=C))
    return dict(drive=drive,delta_F=dF,JE=je,JE_expected=math.exp(-dF),crooks_residual=float(np.max(abs(crooks))),heat=heat,work=work,delta_U=float(q@E1-p@E0),delta_S=dS,sigma=sigma,KL=kl,temp=temp)

def cumulants(z):
    i=z["chosen"].index(z["source"]);x=z["J"][:,i];p=z["p"];d=x-p@x;v=float(p@(d*d));s=float(p@(d**3))/v**1.5;ex=float(p@(d**4))/v**2-3
    return dict(variance=v,skew=s,excess=ex,replicas=[dict(M=M,skew=s/math.sqrt(M),excess=ex/M) for M in (1,2,4,8,16,32,64)])
