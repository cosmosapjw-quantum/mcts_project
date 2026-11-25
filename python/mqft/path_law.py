"""Independent root chronology and causal law audit for trace.v2.

This is a replay of the actual root lock sequence. Descendant asynchronous
schedule is not linearized. Stochastic root policies are opt-in and restricted
to serial/leaf/wave. No physical probability is assigned to a greedy score.
"""
import math
import numpy as np

def scores(n,w,v,vv,p,c,use_pending):
    n=np.asarray(n,dtype=np.int64)
    ne=n+v if use_pending else n
    wf=np.asarray(w,dtype=np.float32)
    if use_pending:wf=wf-np.asarray(vv,dtype=np.float32)
    q=np.divide(wf,np.maximum(ne,1).astype(np.float32))
    return q+np.float32(c)*np.asarray(p,dtype=np.float32)*np.sqrt(np.float32(ne.sum()))/(1+ne).astype(np.float32)

def softmax(s,beta,active,actions,source,h):
    z=beta*np.asarray(s,float)+h*(np.asarray(actions)==source)
    active=np.asarray(active,bool)
    if not active.any():raise ValueError("empty law support")
    p=np.zeros(len(z));x=z[active];v=np.exp(x-x.max());p[active]=v/v.sum()
    if np.any(p[active]<=0):raise ValueError("numerical loss of supported probability")
    return p

def replay(rep,spec):
    order=rep["root_order"]
    es=rep["cuts"][-1]["edges"]
    if len(order)!=len(es) or set(order)!={x["action"] for x in es}:raise ValueError("root materialization order")
    by={x["action"]:x for x in es};idx={a:i for i,a in enumerate(order)}
    prior=np.array([by[a]["prior"] for a in order]);n=np.zeros(len(order),np.int64);w=np.zeros(len(order));s2=w.copy();v=n.copy();vv=w.copy()
    timeline=[];ev=rep["events"];law=spec.get("root_law");lp=lq=lb=gp=gq=fp=fq=0.;t=0;pending=set();max_score_error=0.
    for e in ev:
        a=e["root_selection"]
        if a is None:raise ValueError("missing root selection")
        timeline.extend([(a["clock"],0,e),(e["root_backup_clock"],1,e)])
    timeline.sort(key=lambda x:x[0])
    if [x[0] for x in timeline]!=list(range(2*len(ev))):raise ValueError("root chronology not contiguous")
    # For cut barriers, exactly 2*k chronological events have finished.
    cuts={2*x["local_k"]:x for x in rep["cuts"]}
    def check_cut(k):
        if k in cuts:
            cut=cuts[k];b={x["action"]:x for x in cut["edges"]}
            if v.any() or np.max(abs(vv),initial=0)>1e-10 or pending:raise ValueError("cut not quiescent")
            for j,a in enumerate(order):
                x=b[a]
                if x["n"]!=n[j] or abs(x["w"]-w[j])>1e-8*(1+n[j]) or abs(x["s2"]-s2[j])>1e-8*(1+n[j]):raise ValueError("root chronology cut mismatch")
    check_cut(0)
    features=[];recent=[]
    for clock,kind,e in timeline:
        j=idx[e["actions"][0]];rs=e["root_selection"];key=rs["clock"];dv=e["vl_visit"][0];dw=e["vl_value"][0]
        if kind==0:
            if rs["edge_index"]!=j or key in pending or dv not in (0.,1.) or dw not in (0.,1.):raise ValueError("root edge or VL")
            if law is None:
                if rs["law"] is not None:raise ValueError("unexpected stochastic draw")
                sc=scores(n,w,v,vv,prior,spec["c_puct"],True)
                err=abs(float(sc[j])-rs["score"]);max_score_error=max(max_score_error,err)
                if err>2e-5 or float(sc.max())-rs["score"]>2e-5:raise ValueError("native greedy score mismatch")
                # Stable lower slab-index tie when exactly representable.
                if sc[j]==sc.max() and j!=int(np.argmax(sc)):raise ValueError("native greedy tie mismatch")
            else:
                a=rs["law"]
                if not a or a["t"]!=t or a["actions"]!=order or a["picked"]!=j or a["kind"]!=law["kind"] or a["proposal"]!=law["proposal"]:raise ValueError("law identity")
                sc=scores(n,w,v,vv,prior,spec["c_puct"],law["kind"]=="reserve")
                active=v==0 if law["kind"]=="pl" else np.ones(len(n),bool)
                if a["active"]!=active.tolist():raise ValueError("conditional support")
                err=float(np.max(abs(sc-np.array(a["scores"]))))
                max_score_error=max(max_score_error,err)
                if err>2e-5 or abs(rs["score"]-a["scores"][j])>1e-7:raise ValueError("law score replay")
                pp=law["beta_p"][0 if len(law["beta_p"])==1 else t];qq=law["beta_q"][0 if len(law["beta_q"])==1 else t]
                h=law.get("logit_h",0.);source=spec.get("source_action")
                if a["beta_p"]!=pp or a["beta_q"]!=qq or a["logit_h"]!=h or a["source"]!=source:raise ValueError("law parameter identity")
                # Use independently verified actual FP32 score, not a rounded reconstruction.
                p=softmax(a["scores"],pp,active,order,source,h);q=softmax(a["scores"],qq,active,order,source,h)
                norm=float(np.logaddexp(lp,lq));alpha=math.exp(lp-norm);alpha_q=math.exp(lq-norm)
                prop=law["proposal"];b=p if prop=="p" else q if prop=="q" else .5*(p+q) if prop=="local" else alpha*p+alpha_q*q
                if abs(a["alpha"]-alpha)>1e-9:raise ValueError("path mixture posterior")
                for x,y in ((p,a["p"]),(q,a["q"]),(b,a["b"])):
                    if not np.allclose(x,y,atol=1e-14,rtol=1e-9) or abs(x.sum()-1)>1e-9:raise ValueError("conditional law normalization")
                u=a["u"]
                if not 0<=u<1 or not active[j] or not (sum(b[:j])-1e-12<=u<=sum(b[:j+1])+1e-12):raise ValueError("root RNG selection")
                logs=[math.log(x[j]) for x in (p,q,b)]
                if max(abs(x-a[y]) for x,y in zip(logs,("log_p","log_q","log_b")))>1e-8:raise ValueError("conditional log probability")
                lp+=logs[0];lq+=logs[1];lb+=logs[2]
                if max(abs(x-a[y]) for x,y in zip((lp,lq,lb),("log_p_path","log_q_path","log_b_path")))>1e-7:raise ValueError("path likelihood replay")
                if source is not None:
                    z=idx[source];gp+=float(j==z)-p[z];gq+=float(j==z)-q[z];fp+=p[z]*(1-p[z]);fq+=q[z]*(1-q[z])
                if prop=="path":
                    if abs(lb-(float(np.logaddexp(lp,lq))-math.log(2)))>1e-7:raise ValueError("path mixture identity")
                # Conditional on the selected action; features only from pre-reservation state
                # and previously completed backups. No future reward is used as input.
                history=np.array([x[1] for x in recent[-8:]],float)
                pp0=n/max(1,int(n.sum()))
                nz=pp0>0
                features.append(dict(clock=clock,action=int(order[j]),label=1+int(np.sign(e["root_value"])),
                    state=[float(n.sum()/spec["budget"]),float(n[j]/max(1,n.sum())),float(w[j]/max(1,n[j])),
                           float(prior[j]),float(v[j]),float(-np.sum(pp0[nz]*np.log(pp0[nz])))],
                    memory=[float(len(history)/8),float(history.mean()) if len(history) else 0.,
                            float(history.std()) if len(history) else 0.,
                            sum(x[0]==j for x in recent[-8:])/8],
                    geometry=[float(-sum(x*math.log(x) for x in b if x>0)),
                              float(np.dot(b,(np.asarray(a["scores"])-np.dot(b,a["scores"]))**2))]))
                t+=1
            pending.add(key);v[j]+=int(dv);vv[j]+=dw
        else:
            if key not in pending:raise ValueError("backup without live reservation")
            pending.remove(key);v[j]-=int(dv);vv[j]-=dw
            if v[j]<0 or vv[j]<-1e-10:raise ValueError("reservation underflow")
            n[j]+=1;r=e["root_value"];w[j]+=r;s2[j]+=r*r;recent.append((j,r))
        check_cut(clock+1)
    if pending or v.any() or np.max(abs(vv),initial=0)>1e-10:raise ValueError("unconsumed root reservations")
    out=dict(root_score_max_error=max_score_error,root_operations=len(timeline))
    if law is not None:
        out.update(root_law=law["kind"],proposal=law["proposal"],log_p=lp,log_q=lq,log_b=lb,
                   log_weight_p=lp-lb,log_weight_q=lq-lb,score_p=gp,score_q=gq,conditional_fisher_p=fp,conditional_fisher_q=fq,
                   mean_mark=float(sum(e["root_value"] for e in ev)/len(ev)),history_features=features)
        if law["proposal"]=="path":
            a,b=math.exp(lp-lb),math.exp(lq-lb)
            if max(a,b)>2+1e-7 or abs(a+b-2)>1e-7:raise ValueError("path weight bound")
            out.update(weight_p=a,weight_q=b)
    return out

def categorical_characteristic(p,marks,u):
    """Exact one-step joint characteristic and connected first/second cumulants."""
    p=np.asarray(p,float);m=np.asarray(marks,float);u=np.asarray(u,float)
    if p.ndim!=1 or m.ndim!=2 or len(p)!=len(m) or m.shape[1]!=len(u) or np.any(p<0) or abs(p.sum()-1)>1e-10:raise ValueError("marked law")
    mean=p@m;d=m-mean
    return complex(p@np.exp(1j*(m@u))),mean,(d.T*p)@d


def path_weight_statistics(rows):
    """Replicated paths from one fixed root/protocol, only bounded path mixture.
    Unnormalized estimates of expectations under P or Q; SNIS separately.
    Score response uses the actual logit source (not prior tilt).
    """
    if not rows or any(r.get("proposal")!="path" for r in rows):raise ValueError("bounded path mixture rows required")
    out={}
    for target in ("p","q"):
        w=np.array([r["weight_"+target] for r in rows],float)
        z=np.array([r["mean_mark"] for r in rows],float)
        g=np.array([r["score_"+target] for r in rows],float)
        f=np.array([r["conditional_fisher_"+target] for r in rows],float)
        if np.any(w<0) or np.any(w>2+1e-7) or not np.isfinite(np.r_[w,z,g,f]).all():raise ValueError("path statistics")
        mass=float(w.sum())
        out[target]=dict(weight_mean=float(w.mean()),ESS=float(mass**2/(w@w)) if mass else 0.,
                         mark_IS=float(np.mean(w*z)),mark_SNIS=float(w@z/mass) if mass else None,
                         score_mean_IS=float(np.mean(w*g)),fisher_IS=float(np.mean(w*g*g)),
                         conditional_fisher_IS=float(np.mean(w*f)),response_IS=float(np.mean(w*z*g)))
        if all(r.get("source_action") is None for r in rows):
            for k in ("score_mean_IS","fisher_IS","conditional_fisher_IS","response_IS"):out[target][k]=None
    return out
