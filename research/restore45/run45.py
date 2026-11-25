"""Bounded RESTORE45 actual-cohort and conditional physics reduction."""
import json,math,statistics as st
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics44 import native_record,exact_wave_full_state,projected_information
from mqft.physics45 import lattice_action,protocol_entropy,probe_from_words,centered_estimates
from mqft.closure import evaluate as closure_evaluate
from mqft.conditional import density
from raw44 import audit_cohort,exact_native_draw_info,graph_ref,fit_scale,jsonable
S=Path(__file__).resolve().parent
prev=S.parent/"mqft_restore44_20261126"
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.open()]
def summary(x):
 a=np.asarray(x,float);return dict(mean=float(a.mean()),se=float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else None,n=len(a))
def assess(rows):
 arms=defaultdict(dict)
 for r in rows:
  key=(r["position"],r["law"],r["replicate"])
  assert r["h"] not in arms[key];arms[key][r["h"]]=r
 groups=defaultdict(list)
 for (pos,law,rep),a in sorted(arms.items()):
  assert set(a)=={-.3,0.,.3} and len({r["seed"] for r in a.values()})==1
  x=a[0.];z=dict(rep=rep,G=x["G"],F=x["F"],fisher_gap=x["G"]**2-x["F"],Lm=x["weights"]["-0.3"],Lp=x["weights"]["0.3"])
  for name in ("occupancy","mean_mark"):
   z[name]=x[name];z[name+"_direct"]=(a[.3][name]-a[-.3][name])/.6
  groups[pos,law].append(z)
 out=[]
 for (pos,law),a in sorted(groups.items()):
  g=np.array([x["G"] for x in a]);lm=np.array([x["Lm"] for x in a]);lp=np.array([x["Lp"] for x in a]);D=(lp-lm)/.6
  q=dict(position=pos,law=law,n=len(a),G=summary(g),F_gap=summary([x["fisher_gap"] for x in a]),Lm=summary(lm),Lp=summary(lp),
         ESSm=float(lm.sum()**2/(lm@lm)),ESSp=float(lp.sum()**2/(lp@lp)))
  for name in ("occupancy","mean_mark"):
   A=np.array([x[name] for x in a]);direct=np.array([x[name+"_direct"] for x in a]);v=centered_estimates(A,g,D)
   q[name]={k:summary(x) for k,x in v.items()};q[name]["direct"]=summary(direct)
   q[name]["paired_raw_minus_direct"]=summary(v["raw_fd"]-direct);q[name]["paired_centered_minus_direct"]=summary(v["centered_fd"]-direct)
   q[name]["centered_minus_direct_sd"]=float(np.std(v["centered_fd"]-direct,ddof=1))
   # jackknife covariance estimator: LOO summands are dependent; do not
   # interpret naive SE of centered products as valid. Recompute group-level
   # leave-one-out estimates to get a jackknife SE.
   for label,z in (("centered",g),("centered_fd",D)):
    est=[]
    for i in range(len(a)):
     mask=np.arange(len(a))!=i;est.append(float(np.cov(A[mask],z[mask],ddof=1)[0,1]))
    q[name][label]["se"]=float(np.sqrt((len(a)-1)/len(a)*np.sum((np.array(est)-np.mean(est))**2)))
   est=[]
   for i in range(len(a)):
    mask=np.arange(len(a))!=i;est.append(float(np.cov(A[mask],D[mask],ddof=1)[0,1]-direct[mask].mean()))
   q[name]["paired_centered_minus_direct"]["se"]=float(np.sqrt((len(a)-1)/len(a)*np.sum((np.array(est)-np.mean(est))**2)))
  out.append(q)
 return out

def explicit_probe_check(words,p,alpha=.5,echo=True):
 """Independent matrix-channel calculation for same model, all supplied histories."""
 X=np.array([[0,1],[1,0]],complex);psi=np.array([1,np.exp(1j*np.pi/4)])/math.sqrt(2);rho0=np.outer(psi,psi.conj())
 av=np.zeros((2,2),complex);bv=av.copy()
 for word,w in zip(words,p):
  a=rho0.copy();bs=[rho0.copy(),rho0.copy()]
  for t,m in enumerate(word):
   sign=-1 if echo and t>=len(word)//2 else 1
   us=[np.diag(np.exp(-1j*alpha*sign*int(m)*e*np.array([1,-1]))) for e in (-1,1)]
   a=sum(U@a@U.conj().T for U in us)/2
   bs=[U@r@U.conj().T for U,r in zip(us,bs)]
  av+=w*a;bv+=w*sum(bs)/2
 for r in (av,bv):density(r)
 return av,bv

def main():
 cg,games,_=audit_cohort("games");cm,_,tr=audit_cohort("marked")
 assert len(tr)==288
 old=read(prev/"PHYSICS_RESULTS.json");reg=read(S/"REGISTRATION.json");assert sha256(prev/"PHYSICS_RESULTS.json")==reg["previous_physics_sha"]
 rows=[];refs={};score_groups=defaultdict(list)
 for j,t,r in tr:
  sp=j["spec"];law=sp["root_law"];v=native_record(t,sp["source_action"])
  di,sc=exact_native_draw_info(t)
  v.update(position=j["position_id"],law=law["kind"],replicate=j["replicate"],seed=sp["seed"],job=j["id"],shard_sha=r["sha256"],**di)
  rows.append(v)
  if v["h"]==0:
   score_groups[v["position"],v["law"]].extend(sc)
   if v["law"]=="iid" and v["replicate"]==0:
    ed=next(x["edges"] for x in t["replicas"][0]["cuts"] if x["local_k"]==8)
    source=sp["source_action"];chosen=[source]+[x["action"] for x in sorted(ed,key=lambda x:(-x["prior"],x["action"])) if x["action"]!=source][:4]
    refs[v["position"]]=dict(edges=ed,chosen=chosen,source=source,job=j["id"],sha=r["sha256"])
 assert len(refs)==4
 fresh=assess(rows);past=assess(old["native"]);allgroups=past+fresh
 metrics={}
 for law in ("iid","pl","reserve"):
  for cohort,rr in (("old",past),("fresh",fresh),("both",allgroups)):
   a=[x for x in rr if x["law"]==law]
   for A in ("occupancy","mean_mark"):
    z={key:summary([x[A][key]["mean"] for x in a]) for key in ("raw","centered","raw_fd","centered_fd","direct","paired_raw_minus_direct","paired_centered_minus_direct")}
    z["G"]=summary([x["G"]["mean"] for x in a]);z["F_gap"]=summary([x["F_gap"]["mean"] for x in a]);z["Lm"]=summary([x["Lm"]["mean"] for x in a]);z["Lp"]=summary([x["Lp"]["mean"] for x in a])
    metrics[f"{cohort}/{law}/{A}"]=z
 # New independent opening split: fit existing closure using old h0 reservation histories, evaluate new only.
 oldjobs={x["job"] for x in old["native"] if x["law"]=="reserve" and x["h"]==0};jobs={x["job"] for x in rows if x["law"]=="reserve" and x["h"]==0}
 train=[x for x in lines(prev/"analysis/marked/history_features.jsonl") if x["job"] in oldjobs]
 test=[x for x in lines(S/"analysis/marked/history_features.jsonl") if x["job"] in jobs]
 train_keys=sorted({(x["kind"],x["position_id"]) for x in train});test_keys=sorted({(x["kind"],x["position_id"]) for x in test})
 assert len(train_keys)==len(test_keys)==4 and not set(train_keys)&set(test_keys)
 closure=closure_evaluate(train+test,train_keys,test_keys)
 # Conditional selection score-scale held out across cohorts (old data actual raw trace).
 # New heldout NLL at declared beta compared with uniform; no refit old beta fits,
 # which were obtained from two old roots.
 nll={}
 for law in ("iid","pl","reserve"):
  a=sum([score_groups[pos,law] for pos in sorted(refs)],[])
  from scipy.special import logsumexp
  def loss(beta):return float(np.mean([logsumexp(beta*s[v])-beta*s[i] for s,v,i in a]))
  nll[law]=dict(n_draws=len(a),old_fit=old["scale_fits"][law]["fitted"],test_nll_old_fit=loss(old["scale_fits"][law]["fitted"]),test_nll_input=loss(1/.3),test_nll_uniform=loss(0))
 exact=[];fluct=[];coarse=[];graph=[];probes=[];native_probes=[];state_arrays={}
 for pos,v in sorted(refs.items()):
  for law in ("iid","pl","reserve"):
   z=exact_wave_full_state(v["edges"],v["chosen"],4,law,v["source"])
   for dim,ms in ((False,(8,16)),(True,(8,16,32,64))):
    for x in lattice_action(z,ms,dim):exact.append(dict(position=pos,law=law,**x))
   for a in (0.,.3,.6):
    fluct.append(dict(position=pos,law=law,**protocol_entropy(v["edges"],v["chosen"],law,v["source"],a)))
   coarse.append(dict(position=pos,law=law,maps=projected_information(z)))
   # Same exact conditional word law, two noise models and actual intervention.
   for echo in (False,True):
    x=probe_from_words(z["words"],z["p"],0,.5,echo)
    a,b=explicit_probe_check(np.array(z["words"])==0,z["p"],.5,echo)
    for name,mat in (("iid",a),("static",b)):
     assert abs((mat[0,1].real+.5)-x[name]["X_plus"])<1e-10
     assert abs((-.0-mat[0,1].imag+.5)-x[name]["Y_plus"])<1e-10
     state_arrays[f"{pos}_{law}_{echo}_{name}"]=mat
    probes.append(dict(position=pos,law=law,echo=echo,**x))
   # Actual classical marked paths drive a declared qubit, without empirical
   # reconstruction of native quantum state. Both controls use the SAME paths.
   a=[r["target_ind"] for r in rows if r["position"]==pos and r["law"]==law and r["h"]==0]
   for echo in (False,True):
    x=probe_from_words(a,np.ones(len(a))/len(a),1,.5,echo)
    u,vv=explicit_probe_check(np.array(a)==1,np.ones(len(a))/len(a),.5,echo)
    assert abs(u[0,1].real+.5-x["iid"]["X_plus"])<1e-10
    assert abs(vv[0,1].real+.5-x["static"]["X_plus"])<1e-10
    native_probes.append(dict(position=pos,law=law,echo=echo,n_records=len(a),**x))
   tr0=next(t for j,t,r in tr if j["position_id"]==pos and j["replicate"]==0 and j["spec"]["root_law"]["kind"]==law and j["spec"]["root_law"]["logit_h"]==0)
   graph.append(dict(position=pos,law=law,depths=graph_ref(tr0)))
 assert len(exact)==72 and len(fluct)==36 and len(probes)==24 and len(native_probes)==24
 # Conservative recommendation: per-root finite-response residual variance;
 # jackknife SE for centered U-statistic, not naive dependent products.
 power=[]
 for x in fresh:
  n=x["n"];d=x["occupancy"]["paired_centered_minus_direct"];se=d["se"]
  power.append(dict(position=x["position"],law=x["law"],n=n,jackknife_se=se,normal_n_halfwidth_005=math.ceil(n*(1.96*se/.05)**2) if se>0 else None))
 out=dict(counts={"games":cg,"marked":cm},games=games,native=rows,paired=fresh,prior_pairs=past,metrics=metrics,closure=closure,score_scale=nll,
   references=refs,multivariate=exact,fluctuation=fluct,coarse=coarse,graph=graph,probes=probes,native_driven_probes=native_probes,power=power,
   scope="full-support native score and FD pilot; exact restricted frozen wave; declared coherent qubit and nonthermal order reversal, not native physical phase or heat")
 atomic_json(S/"PHYSICS_RESULTS.json",jsonable(out));np.savez_compressed(S/"PROBE_STATES.npz",**state_arrays)
 print("COUNTS",json.dumps(out["counts"]))
 print("GAMES",games);print("NEW METRICS",json.dumps({k:v for k,v in metrics.items() if k.startswith("fresh/") and k.endswith("occupancy")}))
 print("LOOP",len(exact),"max residual",max(x["dp_residual"] for x in exact))
 print("CLOSURE",json.dumps(closure));print("POWER",power)
if __name__=="__main__":main()
