"""Streaming independent trace and RNG replay, grouped physical results.
Run only when complete receipts exist. Prior cohorts are re-audited, not
recounted as fresh. All statistics use complete adaptive histories."""
from pathlib import Path
from collections import defaultdict,Counter
import json,gzip,math,subprocess,statistics as st
import numpy as np
from scipy.stats import kstest
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.path_law import replay,softmax
from mqft.physics44 import native_record
from mqft.physics46 import actual_protocol,score_evidence,fit_count_memory,causal_break,log_ratio
from mqft.physics45 import centered_estimates
from mqft.analysis import check_trace
from mqft.closure import evaluate as closure_evaluate
S=Path(__file__).resolve().parent
B=S.parent
def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as f:
  for l in f:yield json.loads(l)
def stats(v):
 a=np.asarray(v,float)
 assert len(a)>0 and np.isfinite(a).all()
 return dict(n=len(a),mean=float(a.mean()),se=float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else None,min=float(a.min()),max=float(a.max()))
def group_stats(records):
 assert len(records)==16
 a={k:stats([r[k] for r in records]) for k in ("G","F","G2minusF","occupancy","mean_mark")}
 for key,h in (("Lminus",-.3),("Lplus",.3)):
  z=np.array([r["weights"][str(h)] for r in records])
  a[key]=stats(z);a[key+"_ESS"]=float(z.sum()**2/(z@z))
 return a

def read_cohort(stage,name,epoch,reg,helper):
 d=stage/"data"/name;pl=read(stage/"plans"/(name+".json"));rp=read(next(d.glob("report-*.json")));lock=read(d/"run_lock.json")
 assert not rp["failures"] and not rp["unattempted"] and len(rp["completed"])==len(pl["jobs"])
 assert lock["plan_hash"]==pl["plan_hash"]==rp["plan_hash"] and lock["backend_sha"]==reg["backend_sha"] and lock["external"]==reg["model"]
 a=stage/"analysis"/name
 m=read(a/"manifest.json")
 assert m["completed"]==m["expected"]==len(pl["jobs"])
 assert all(sha256(a/n)==h for n,h in m["outputs"].items())
 jobs={x["id"]:x for x in pl["jobs"]};out=[];cnt=Counter();maxerr=0.;logerr=0.;minmargin=1.
 for r in rp["completed"]:
  j=jobs[r["job_id"]];f=d/r["file"];assert sha256(f)==r["sha256"] and read(d/(j["id"]+".receipt.json"))==r
  with gzip.open(f,"rt") as g:
   hdr,x,end=[json.loads(l) for l in g]
  assert hdr["job"]==j and hdr["evaluator_runtime"]==reg["model"] and x["type"]=="root" and end["status"]=="root_complete" and end["moves"]==j["opening"]
  t=x["trace"];sp=t["spec"];rep=t["replicas"][0]
  expected={**j["spec"],"c_puct":float(np.float32(j["spec"]["c_puct"]))}
  assert sp==expected and sp["budget"]==16 and sp["evaluator"]=="rapfi_nnue_lookahead_v1"
  # Full independent check, incl legal paths, moments, conditional law.
  # Old and new full analysis already validates; recheck root and sampler
  # directly here, with independent stochastic replay.
  audit=replay(rep,sp)
  ev=sorted(rep["events"],key=lambda x:x["root_selection"]["clock"])
  laws=[x["root_selection"]["law"] for x in ev];u=[x["u"] for x in laws]
  msg=dict(spec_seed=sp["seed"],replica_seed=rep["seed"],root_uniforms=u,evaluations=rep["evaluations"])
  helper.stdin.write(json.dumps(msg,separators=(",",":"))+"\n");helper.stdin.flush()
  line=helper.stdout.readline()
  if not line:raise RuntimeError("RNG helper failed")
  v=json.loads(line);cnt["rng_root_draws"]+=v["root_draws"];cnt["rng_sample_draws"]+=v["sample_draws"]
  maxerr=max(maxerr,v["max_root_u_error"]);logerr=max(logerr,v["max_sample_log_error"]);minmargin=min(minmargin,v["min_sample_cdf_margin"])
  z=native_record(t,sp["source_action"])
  assert abs(z["G"]-audit["score_p"])<1e-9 and abs(z["F"]-audit["conditional_fisher_p"])<1e-9
  assert abs(z["log_root_prob"]-audit["log_p"])<1e-8
  # Independently recompute the full conditional source LR at alternate h,
  # using the recorded pre-reservation scores and identical support.
  ratios={}
  for h in (-.3,0.,.3):
   lp=0.
   for a0 in laws:
    p=softmax(a0["scores"],a0["beta_p"],a0["active"],a0["actions"],a0["source"],h)
    lp+=math.log(p[a0["picked"]])
   ll,kl=log_ratio(z["conditional_p"],z["target_ind"],h-z["h"])
   assert abs(lp-z["log_root_prob"]-ll)<1e-8
   ratios[str(h)]=actual_protocol(z,h)
  waves=defaultdict(list)
  for e,I in zip(ev,z["target_ind"]):waves[e["phase"]].append(I)
  full=[a for _,a in sorted(waves.items()) if len(a)==4]
  assert len(full)==3
  cnt["jobs"]+=1;cnt["backups"]+=r["backups"];cnt["compressed_bytes"]+=r["bytes"]
  cnt["cuts"]+=len(rep["cuts"]);cnt["parent_queries"]+=len(rep["evaluations"])
  cnt["sample_queries"]+=sum(not x["terminal"] for a in rep["evaluations"] for x in a["samples"])
  cnt["samples"]+=sum(len(a["samples"]) for a in rep["evaluations"])
  z.update(position=j["position_id"],law=sp["root_law"]["kind"],replicate=j["replicate"],seed=sp["seed"],job=j["id"],epoch=epoch,root_hash=t["root"]["state_hash"],
    G2minusF=z["G"]**2-z["F"],uniforms=u,full_wave_counts=[sum(a) for a in full],full_wave_marks=full,protocol=ratios,shard_sha=r["sha256"])
  out.append(z)
 diag=[f.read_text() for f in (d/"diagnostics").glob("*.log") if f.stat().st_size]
 return out,dict(**dict(cnt),max_root_u_error=maxerr,max_sample_log_error=logerr,min_sample_cdf_margin=minmargin,
   analysis_manifest_sha=sha256(a/"manifest.json"),runtime_retries=sum(x.count("MQFT_RAPFI_RETRY") for x in diag),runtime_diagnostics=diag)

def response_group(arms):
 out={}
 z=arms[0.]
 g=np.array([r["G"] for r in z]);L1=np.array([r["weights"]["0.3"] for r in z]);L0=np.array([r["weights"]["-0.3"] for r in z]);D=(L1-L0)/.6
 for A in ("occupancy","mean_mark"):
  a=np.array([r[A] for r in z]);direct=np.array([(p[A]-m[A])/.6 for p,m in zip(arms[.3],arms[-.3])]);v=centered_estimates(a,g,D)
  q={k:stats(x) for k,x in v.items()};q["direct"]=stats(direct)
  for label,t in (("centered",g),("centered_fd",D)):
   jk=[]
   for i in range(len(a)):
    mask=np.arange(len(a))!=i;jk.append(float(np.cov(a[mask],t[mask],ddof=1)[0,1]))
   q[label]["se"]=float(np.sqrt((len(a)-1)/len(a)*sum((np.array(jk)-np.mean(jk))**2)))
  jk=[]
  for i in range(len(a)):
   mask=np.arange(len(a))!=i;jk.append(float(np.cov(a[mask],D[mask],ddof=1)[0,1]-direct[mask].mean()))
  q["difference"]={"mean":float(v["centered_fd"].mean()-direct.mean()),"jackknife_se":float(np.sqrt((len(a)-1)/len(a)*sum((np.array(jk)-np.mean(jk))**2)))}
  out[A]=q
 return out

def main():
 reg=read(S/"REGISTRATION.json")
 prev44=B/"mqft_restore44_20261126";prev45=B/"mqft_restore45_20261126"
 for p in (prev44,prev45):assert sha256(p/"PHYSICS_RESULTS.json")==reg["previous_physics"][p.name]
 exe=S/"runtime/mqft-rng-replay"
 helper=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 cohort=[];counts={};allfeatures=[]
 try:
  for stage,name,epoch in ((prev44,"marked","44"),(prev45,"marked","45"),(S,"old_roots","46-old"),(S,"new_roots","46-new")):
   a,c=read_cohort(stage,name,epoch,reg,helper);cohort.extend(a);counts[epoch]=c
   ids={r["job"] for r in a if r["h"]==0 and r["law"]=="reserve"}
   allfeatures.extend(r for r in rows(stage/"analysis"/name/"history_features.jsonl") if r["job"] in ids)
  helper.stdin.close();assert helper.wait(timeout=10)==0
 finally:
  if helper.poll() is None:helper.kill();helper.wait()
 assert len(cohort)==1152
 # Unique seed per root/law/replicate, paired across h. No reuse across roots/laws.
 seeds=defaultdict(list)
 for r in cohort:seeds[r["seed"]].append(r)
 assert len(seeds)==384 and all(len(a)==3 and {r["h"] for r in a}=={-.3,0.,.3} and len({(r["position"],r["law"]) for r in a})==1 for a in seeds.values())
 groups=defaultdict(dict)
 for (seed,a) in sorted(seeds.items()):
  key=(a[0]["position"],a[0]["law"])
  for r in a:groups[key].setdefault(r["h"],[]).append(r)
 assert len(groups)==24 and all(set(a)=={-.3,0.,.3} and all(len(x)==16 for x in a.values()) for a in groups.values())
 agg=[];protocol=[];power=[]
 for (pos,law),a in sorted(groups.items()):
  assert all([r["seed"] for r in a[h]]==[r["seed"] for r in a[0.]] for h in (-.3,.3))
  q=response_group(a)
  agg.append(dict(position=pos,law=law,stats=group_stats(a[0.]),response=q))
  se=q["occupancy"]["difference"]["jackknife_se"]
  power.append(dict(position=pos,law=law,se=se,n_005=math.ceil(16*(1.96*se/.05)**2) if se>0 else None))
  for h in (-.3,0.,.3):
   for h1 in (-.3,0.,.3):
    if h1==h:continue
    z=[r["protocol"][str(h1)] for r in a[h]]
    w=np.array([x["weight"] for x in z])
    protocol.append(dict(position=pos,law=law,source=h,target=h1,sigma=stats([x["sigma"] for x in z]),
       conditional_KL=stats([x["conditional_KL"] for x in z]),compensated=stats([x["compensated"] for x in z]),weight=stats(w),ESS=float(w.sum()**2/(w@w))))
 zero=sorted([x for x in cohort if x["h"]==0],key=lambda x:(x["position"],x["law"],x["seed"]))
 # Nonnegative mixture evidence uses only h0, never paired +/- draws as independent.
 diag={}
 for name in ("iid","pl","reserve","all"):
  a=[x for x in zero if name=="all" or x["law"]==name];u=np.concatenate([r["uniforms"] for r in a]);ks=kstest(u,"uniform")
  diag[name]=dict(**score_evidence(a),uniform_mean=float(u.mean()),uniform_KS=float(ks.statistic),uniform_KS_descriptive_p=float(ks.pvalue))
 # Frozen heldout opening split (old roots vs new roots).
 train=[r for r in zero if not r["position"].startswith("r45_")];test=[r for r in zero if r["position"].startswith("r45_")]
 memory=[fit_count_memory([r for r in train if r["law"]==l],[r for r in test if r["law"]==l],l) for l in ("iid","pl","reserve")]
 a=sorted({(r["kind"],r["position_id"]) for r in allfeatures if not r["position_id"].startswith("r45_")});b=sorted({(r["kind"],r["position_id"]) for r in allfeatures if r["position_id"].startswith("r45_")})
 closure=closure_evaluate(allfeatures,a,b)
 # Conditional causal-break: exactly SAME recorded first and third full waves.
 # Condition per word, then summarize across independent histories.
 cb=[]
 for r in zero:
  w=r["full_wave_marks"][0]+r["full_wave_marks"][-1]
  cb.append(dict(position=r["position"],law=r["law"],seed=r["seed"],job=r["job"],**causal_break(w)))
 assert len(cb)==384
 metrics={}
 for l in ("iid","pl","reserve"):
  a=[r for r in agg if r["law"]==l]
  metrics[l]={"G":stats([r["stats"]["G"]["mean"] for r in a]),"F_gap":stats([r["stats"]["G2minusF"]["mean"] for r in a]),
    "Lm":stats([r["stats"]["Lminus"]["mean"] for r in a]),"Lp":stats([r["stats"]["Lplus"]["mean"] for r in a]),
    "occupancy":{k:stats([r["response"]["occupancy"][k]["mean"] for r in a]) for k in ("raw","centered","raw_fd","centered_fd","direct","difference")},
    "mark":{k:stats([r["response"]["mean_mark"][k]["mean"] for r in a]) for k in ("raw","centered","raw_fd","centered_fd","direct","difference")}}
 # Reuse previous exact conditional action, thermal, Q results, no relabel.
 out=dict(counts=counts,total_histories=len(cohort),fresh_histories=720,unique_seed_groups=len(seeds),groups=agg,metrics=metrics,score_diagnostics=diag,
    actual_protocol=protocol,memory=memory,closure=closure,causal_break=cb,power=power,
    rng_replay_sha256=sha256(exe),native=cohort,
    scope="16 histories per 8 selected roots/law/arm; replay deterministic RNG, inferential checks conditional on ideal independent RNG, source protocol not physical reversal; fitted empirical pair model and declared CP intervention")
 atomic_json(S/"PHYSICS_RESULTS.json",out)
 print("COUNTS",json.dumps(counts))
 print("METRICS",json.dumps(metrics))
 print("DIAGNOSTICS",json.dumps(diag))
 print("MEMORY",json.dumps([{k:v for k,v in r.items() if k not in ("train_table","memory_table","projected_generator")} for r in memory]))
 print("CLOSURE",json.dumps(closure))
 print("POWER",json.dumps(power))
if __name__=="__main__":main()
