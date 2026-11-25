"""Predeclared low-dimensional prospective Q-field closure.
Uses only h=0 search features at prediction time. Observed perturbed Q is
retained separately as a post-treatment diagnostic, not a predictor.
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from .client import sha256
from .observables import frozen_action
from .campaign import atomic_json

RIDGE=.05
def load_cases(path):
    path=Path(path);manifest=json.loads((path/"manifest.json").read_text())
    assert manifest["completed"]==manifest["expected"]
    for f,h in manifest["outputs"].items():
        if sha256(path/f)!=h:raise ValueError("analysis identity")
    grouped=defaultdict(dict)
    for line in (path/"local.jsonl").open():
        r=json.loads(line)
        if r["global_k"]!=r["budget"]:continue
        if r["mode"] not in ("serial","wave") or r.get("evaluator")!="rapfi_nnue_lookahead_v1":raise ValueError("wrong cohort")
        key=(r["position_id"],r["state_hash"],r["mode"],r["budget"],r["replicate"],r["source_action"])
        if r["source_h"] in grouped[key]:raise ValueError("duplicate arm")
        grouped[key][r["source_h"]]=r
    out=[]
    for key,arms in sorted(grouped.items()):
        if 0 not in arms:raise ValueError("missing zero")
        for h in sorted(x for x in arms if x>0):
            if -h not in arms:raise ValueError("missing paired arm")
            a,b,c=arms[-h],arms[0],arms[h]
            if any(x["actions"]!=b["actions"] or x["k"]!=b["k"] for x in (a,c)):raise ValueError("support mismatch")
            out.append((key,h,a,b,c))
    return out

def field_features(base,h):
    p=np.array(base["prior"],float);p/=p.sum()
    q=np.array(base["action_q"],float);pi=np.array(base["pi"],float)
    i=base["actions"].index(base["source_action"]);e=np.zeros(len(p));e[i]=1
    k=base["k"];c=base["c_puct"]
    def prior(hh):
        pt=p*np.exp(hh*e);return pt/pt.sum()
    def opt(qh,hh):
        return np.array(frozen_action(qh,prior(hh),k,c,pi)["optimum"])
    f=(opt(q,h)-opt(q,-h))/(2*h)
    return q,pi,i,np.column_stack((e,f)),(p+pi)/2,opt

def fit(cases):
    by=defaultdict(list)
    for key,h,a,b,c in cases:by[b["mode"]].append((key,h,a,b,c))
    result={}
    for mode,rr in by.items():
        A=np.zeros((2,2));v=np.zeros(2);num=den=0.
        for key,h,a,b,c in rr:
            q,pi,i,X,w,opt=field_features(b,h)
            y=(np.array(c["action_q"])-a["action_q"])/(2*h)
            A+=X.T@(w[:,None]*X)/len(rr);v+=X.T@(w*y)/len(rr)
            f=X[i,1];obs=(c["pi"][i]-a["pi"][i])/(2*h)
            num+=f*obs/len(rr);den+=f*f/len(rr)
        coef=np.linalg.solve(A+RIDGE*np.eye(2),v)
        result[mode]={"q_coef":coef.tolist(),"gain":float(num/(den+RIDGE)),"n_train":len(rr)}
    return result

def predict(base,h,coef):
    q,pi,i,X,w,opt=field_features(base,h)
    g=X@np.asarray(coef["q_coef"])
    qm=np.clip(q-h*g,-1,1);qp=np.clip(q+h*g,-1,1)
    return {"frozen":float(X[i,1]),"gain":float(coef["gain"]*X[i,1]),"prospective":float((opt(qp,h)[i]-opt(qm,-h)[i])/(2*h)),
            "q_derivative":(qp-qm)/(2*h)}
def evaluate(cases,model):
    out=[]
    for key,h,a,b,c in cases:
        q,pi,i,X,w,opt=field_features(b,h)
        pred=predict(b,h,model[b["mode"]])
        y=(np.array(c["action_q"])-a["action_q"])/(2*h)
        obs=(c["pi"][i]-a["pi"][i])/(2*h)
        diagnostic=(opt(np.asarray(c["action_q"]),h)[i]-opt(np.asarray(a["action_q"]),-h)[i])/(2*h)
        n=round(obs*2*h*b["k"])
        if abs(obs*2*h*b["k"]-n)>1e-8:raise ValueError("occupancy lattice")
        out.append(dict(position_id=key[0],state_hash=key[1],mode=b["mode"],budget=b["budget"],replicate=b["replicate"],source_action=key[-1],h=h,observed=obs,
            frozen=pred["frozen"],gain=pred["gain"],prospective=pred["prospective"],observed_q_diagnostic=float(diagnostic),
            q_mae_zero=float(w@abs(y)),q_mae_prospective=float(w@abs(y-pred["q_derivative"])),lattice_step=1/(2*h*b["k"]),
            q_abs_shift=float(np.max(abs(np.array(c["action_q"])-a["action_q"]))),baseline_job=b["job"],minus_job=a["job"],plus_job=c["job"]))
    return out

def metrics(rows):
    return {k:float(np.mean([abs(r["observed"]-(0 if k=="zero" else r[k])) for r in rows])) for k in ("zero","frozen","gain","prospective","observed_q_diagnostic")} | {k:float(np.mean([r[k] for r in rows])) for k in ("q_mae_zero","q_mae_prospective")}
