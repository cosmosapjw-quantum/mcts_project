"""Held-out causal-history ablation, no automatic model or selector admission."""
import json
from pathlib import Path
import numpy as np
from .client import sha256
from .campaign import atomic_json

def evaluate(rows,train_groups,test_groups,ridge=0.01):
    from scipy.optimize import minimize
    tr,te=set(train_groups),set(test_groups)
    if not tr or not te or tr&te or ridge<=0 or not np.isfinite(ridge):raise ValueError("disjoint group split and positive ridge required")
    groups=[(r["kind"],r["position_id"]) for r in rows]
    # Whole position/opening held out, including all replicas, interventions and games.
    if set(groups)!=tr|te:raise ValueError("split must cover all source groups")
    train=np.array([g in tr for g in groups]);test=~train
    y=np.array([r["label"] for r in rows],int)
    if not train.any() or not test.any() or np.any((y<0)|(y>2)):raise ValueError("classification support")
    # Each group contributes equal weight to the fit and evaluation, not each correlated event.
    def weights(mask):
        g=[groups[i] for i in np.flatnonzero(mask)]
        from collections import Counter
        c=Counter(g);return np.array([1/(len(c)*c[k]) for k in g])
    wt,ws=weights(train),weights(test);one=np.eye(3)[y]
    result={}
    for name,keys in (("state",("state",)),("history",("state","memory")),
                      ("geometry",("state","geometry")),("history_geometry",("state","memory","geometry"))):
        x=np.asarray([sum([r[k] for k in keys],[]) for r in rows],float)
        if not np.isfinite(x).all():raise ValueError("feature value")
        mu=wt@x[train];sd=np.sqrt(wt@((x[train]-mu)**2));sd=np.where(sd>1e-12,sd,1)
        x=np.c_[np.ones(len(x)),(x-mu)/sd];xt=x[train];yt=one[train]
        def prob(a):
            z=a-a.max(axis=1,keepdims=True);ex=np.exp(z);return ex/ex.sum(axis=1,keepdims=True)
        def objective(v):
            b=v.reshape(x.shape[1],3);z=xt@b;p=prob(z)
            logp=z-np.logaddexp.reduce(z,axis=1)[:,None]
            loss=-float(wt@np.sum(yt*logp,axis=1))+.5*ridge*np.sum(b*b)
            grad=xt.T@((p-yt)*wt[:,None])+ridge*b
            return loss,grad.ravel()
        fit=minimize(objective,np.zeros(x.shape[1]*3),jac=True,method="L-BFGS-B",options={"maxiter":1000,"gtol":1e-8})
        if not fit.success or not np.isfinite(fit.fun):raise RuntimeError("closure optimization: "+str(fit.message))
        z=x[test]@fit.x.reshape(x.shape[1],3);p=prob(z);logp=z-np.logaddexp.reduce(z,axis=1)[:,None]
        nll=-np.sum(one[test]*logp,axis=1);brier=np.sum((p-one[test])**2,axis=1)
        result[name]=dict(nll=float(ws@nll),brier=float(ws@brier),train_objective=float(fit.fun),iterations=int(fit.nit),
            test_group_NLL=[dict(kind=k[0],position_id=k[1],nll=float(np.mean([nll[i] for i,j in enumerate(np.flatnonzero(test)) if groups[j]==k]))) for k in sorted(te)])
    return dict(scope="selected-action return-sign prediction; fixed group split; not strength/quantum identification",
                train_groups=[list(g) for g in sorted(tr)],test_groups=[list(g) for g in sorted(te)],
                train_events=int(train.sum()),test_events=int(test.sum()),ridge=ridge,models=result)

def run(analysis,split,output):
    p=Path(analysis);m=json.loads((p/"manifest.json").read_text());name="history_features.jsonl"
    if m.get("schema")!="mqft.analysis.v2" or name not in m["outputs"] or sha256(p/name)!=m["outputs"][name]:raise ValueError("history analysis identity")
    cfg=json.loads(Path(split).read_text());rows=[json.loads(x) for x in (p/name).read_text().splitlines()]
    if not rows:raise ValueError("no eligible stochastic-root observations")
    # One fixed protocol, schedule, budget, source: otherwise history could proxy an intervention.
    required=("mode","workers","budget","c_puct","samples_per_eval","law_id","source_h","source_action","opponent","kind")
    match=cfg["match"]
    if set(match)!=set(required):raise ValueError("explicit single-protocol match required")
    rows=[r for r in rows if all(r[k]==match[k] for k in required)]
    out=evaluate(rows,[tuple(x) for x in cfg["train"]],[tuple(x) for x in cfg["test"]],cfg.get("ridge",.01))
    out.update(match=match,source_manifest_sha256=sha256(p/"manifest.json"),split_sha256=sha256(split))
    atomic_json(output,out);return out
