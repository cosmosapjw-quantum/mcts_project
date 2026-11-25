"""Actual trace reducer and grouped physical computations, RESTORE47.
The empirical CGF and fitted closure are separate from declared quantum
and thermal models. No synthetic MCTS or additional path generation."""
import json,gzip,math,subprocess,statistics as st
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.path_law import replay,softmax
from mqft.physics44 import native_record
from mqft.physics47 import joint_action,scale_response,clock_closure,process_probe,thermal_cycle,categorical_clock_memory
from mqft.physics46 import log_ratio
S=Path(__file__).resolve().parent
P=S.parent/"mqft_restore39_20261125/repo";prev=S.parent/"mqft_restore46_20261126"
def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as f:
  for line in f:yield json.loads(line)
def jsonable(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 return x
def mean_se(v):
 a=np.asarray(v,float)
 return dict(mean=float(a.mean()),se=float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else None,n=len(a))
def cohort(name,reg,helper):
 d=S/"data"/name;pl=read(S/"plans"/(name+".json"));lock=read(d/"run_lock.json");report=read(next(d.glob("report-*.json")))
 m=read(S/"analysis"/name/"manifest.json")
 assert m["completed"]==m["expected"]==len(pl["jobs"])==len(report["completed"]) and not report["failures"] and not report["unattempted"]
 assert lock["plan_hash"]==pl["plan_hash"]==m["plan_hash"] and lock["backend_sha"]==reg["backend_sha"] and lock["external"]==reg["model"]
 assert all(sha256(S/"analysis"/name/f)==h for f,h in m["outputs"].items())
 jobs={x["id"]:x for x in pl["jobs"]};cnt=Counter();native=[];games=[];err=logerr=0.;minmargin=1.
 for r in report["completed"]:
  j=jobs[r["job_id"]];q=d/r["file"];assert sha256(q)==r["sha256"] and q.stat().st_size==r["bytes"] and read(d/(j["id"]+".receipt.json"))==r
  cnt["jobs"]+=1;cnt["bytes"]+=r["bytes"]
  with gzip.open(q,"rt") as g:
   hd=json.loads(next(g));assert hd["job"]==j and hd["evaluator_runtime"]==reg["model"]
   moves=j["opening"].copy();b=0;nr=0;end=None
   for line in g:
    x=json.loads(line)
    if x["type"]=="end":end=x;break
    t=x["trace"];sp=t["spec"];rep=t["replicas"][0]
    assert sp["evaluator"]=="rapfi_nnue_lookahead_v1" and sp["samples_per_eval"]==4
    assert len(rep["events"])==sp["budget"]
    for a in rep["evaluations"]:
     assert a["nnue_scores"] and len(a["samples"])==4
     cnt["parent"]+=1;cnt["samples"]+=4
     for sm in a["samples"]:
      if not sm["terminal"]:
       w,l,dr=sm["nnue_wdl"];assert abs(a["side"]*(w-l)/(w+l+dr)-sm["value"])<1e-7;cnt["child"]+=1
    cnt["searches"]+=1;cnt["cuts"]+=len(rep["cuts"]);cnt["backups"]+=sp["budget"];b+=sp["budget"];nr+=1
    if name=="games":
     assert x["action"] in t["root"]["legal"];moves.append(x["action"]);continue
    # independent root chronology + original Rust RNG replay
    a=replay(rep,sp);ev=sorted(rep["events"],key=lambda z:z["root_selection"]["clock"])
    ls=[z["root_selection"]["law"] for z in ev];u=[z["u"] for z in ls]
    msg=dict(spec_seed=sp["seed"],replica_seed=rep["seed"],root_uniforms=u,evaluations=rep["evaluations"])
    helper.stdin.write(json.dumps(msg,separators=(",",":"))+"\n");helper.stdin.flush();l=helper.stdout.readline()
    if not l:raise RuntimeError("RNG replay failed")
    v=json.loads(l);cnt["rng_root"]+=v["root_draws"];cnt["rng_samples"]+=v["sample_draws"]
    err=max(err,v["max_root_u_error"]);logerr=max(logerr,v["max_sample_log_error"]);minmargin=min(minmargin,v["min_sample_cdf_margin"])
    z=native_record(t,sp["source_action"]);assert abs(z["G"]-a["score_p"])<1e-9 and abs(z["F"]-a["conditional_fisher_p"])<1e-9
    phases=defaultdict(list)
    for i,e in enumerate(ev):phases[e["phase"]].append(i)
    full=[]
    for phase,ids in sorted(phases.items()):
     if len(ids)!=4:continue
     full.append(dict(phase=phase,k_end=ids[-1]+1,count=sum(z["target_ind"][i] for i in ids),mark=st.mean(z["marks"][i] for i in ids),p=st.mean(z["conditional_p"][i] for i in ids)))
    assert len(full)==sp["budget"]//4-1
    proto=[]
    for h in (-.2,0.,.2):
     if h==z["h"]:continue
     lp=0.
     for d0 in ls:
      pp=softmax(d0["scores"],d0["beta_p"],d0["active"],d0["actions"],d0["source"],h);lp+=math.log(pp[d0["picked"]])
     ll,kl=log_ratio(z["conditional_p"],z["target_ind"],h-z["h"])
     assert abs(lp-z["log_root_prob"]-ll)<1e-8
     proto.append(dict(target=h,weight=math.exp(ll),sigma=-ll,conditional_KL=kl))
    z.update(campaign=name,position=j["position_id"],law=sp["root_law"]["kind"],seed=sp["seed"],job=j["id"],replicate=j["replicate"],root_hash=t["root"]["state_hash"],complete_waves=full,protocol=proto,shard_sha=r["sha256"])
    native.append(z)
   assert end and end["moves"]==moves and nr==r["records"] and b==r["backups"]
   if name=="games":
    assert end["status"]=="natural_terminal" and end["terminal"]
    games.append(dict(position=j["position_id"],plies=len(moves),winner=end["winner"],searches=nr))
 di=[x.read_text() for x in (d/"diagnostics").glob("*.log") if x.stat().st_size]
 return dict(**dict(cnt),runtime_retries=sum(x.count("MQFT_RAPFI_RETRY") for x in di),runtime_diagnostics=di,max_root_u_error=err,max_sample_log_error=logerr,min_sample_cdf_margin=minmargin if name!="games" else None),native,games

def old_wave(r):
 # RESTORE46 records have exactly three complete waves after four
 # completed selections at initial short barriers. Verify count/marks.
 z={**r,"campaign":"prior16"};full=[]
 for j,ind in enumerate(r["full_wave_marks"]):
  k=4+4*j;assert r["target_ind"][k:k+4]==ind
  full.append(dict(phase=j,k_end=k+4,count=sum(ind),mark=st.mean(r["marks"][k:k+4]),p=st.mean(r["conditional_p"][k:k+4])))
 assert len(full)==3 and full[-1]["k_end"]==16
 z["complete_waves"]=full
 return z
def main():
 reg=read(S/"REGISTRATION.json");rreg=read(S/"ROOT_REGISTRATION.json")
 assert sha256(prev/"PHYSICS_RESULTS.json")==reg["previous_physics_sha"]
 old=read(prev/"PHYSICS_RESULTS.json");assert len(old["native"])==1152
 # Reuse unchanged previously verified 46 helper and exact trace identity.
 helper_path=prev/"runtime/mqft-rng-replay";assert sha256(helper_path)==old["rng_replay_sha256"]
 helper=subprocess.Popen([str(helper_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 cnt={};native=[];games=[]
 try:
  for name in ("games","scale32","scale64","heldout64"):
   c,a,g=cohort(name,reg,helper);cnt[name]=c;native.extend(a);games.extend(g)
  helper.stdin.close();assert helper.wait(timeout=10)==0
 finally:
  if helper.poll() is None:helper.kill();helper.wait()
 assert len(native)==240 and len(games)==4
 assert sum(x.get("rng_root",0) for x in cnt.values())==11520
 fresh=[r for r in native if r["campaign"]=="heldout64"];scale=[r for r in native if r["campaign"]!="heldout64"]
 anchors={r["position"] for r in scale};assert len(anchors)==4
 old_seeds={r["seed"] for r in old["native"]};assert not old_seeds&{r["seed"] for r in native}
 assert not {r["position"] for r in fresh}&{r["position"] for r in old["native"]}
 grouped=defaultdict(lambda:defaultdict(list))
 for r in scale:grouped[r["position"],r["law"],r["k"]][r["h"]].append(r)
 for r in old["native"]:
  if r["position"] in anchors:grouped[r["position"],r["law"],r["k"]][r["h"]].append(r)
 res=scale_response(grouped);assert len(res)==48
 # 24 B16 empirical action fits, split each root/law's 16 h0 seeds 8+8.
 old0=[old_wave(r) for r in old["native"] if r["h"]==0]
 action=[]
 for pos in sorted({r["position"] for r in old0}):
  for l in ("iid","pl","reserve"):
   a=sorted([r for r in old0 if r["position"]==pos and r["law"]==l],key=lambda r:r["seed"]);assert len(a)==16
   action.append(dict(position=pos,law=l,**joint_action(a[:8],a[8:])))
 assert len(action)==24
 train=old0+[r for r in scale if r["h"]==0];test=fresh
 memory=[clock_closure(train,test,l) for l in ("iid","pl","reserve")]
 categorical=[categorical_clock_memory(train,test,l) for l in ("iid","pl","reserve")]
 # For each actual fresh h0 path, fixed first and last complete wave
 # drive qubit channels; re-use exact same hit-count channels by key.
 probe=[];cycle=[];states={};cache={}
 for r in fresh:
  w=r["complete_waves"];n1,n2=w[0]["count"],w[-1]["count"];key=(n1,n2)
  if key not in cache:
   q,mat=process_probe(*key);cache[key]=q
   states.update({f"hits{n1}_{n2}_{k}":a for k,a in mat.items()})
  probe.append(dict(position=r["position"],law=r["law"],seed=r["seed"],job=r["job"],**cache[key]))
  pp=[w[0]["p"],w[len(w)//2]["p"],w[-1]["p"]]
  for eta in (.25,.5,1.):
   cycle.append(dict(position=r["position"],law=r["law"],seed=r["seed"],job=r["job"],**thermal_cycle(pp,eta)))
 assert len(probe)==48 and len(cycle)==144
 # Native occupancy score-scale/entropy time evolution, no physical beta.
 temp=[]
 for name in ("games","scale32","scale64","heldout64"):
  ids={j["id"] for j in read(S/"plans"/(name+".json"))["jobs"] if name=="games" or j["spec"]["root_law"]["logit_h"]==0}
  for x in rows(S/"analysis"/name/"local.jsonl"):
   if x["job"] not in ids:continue
   temp.append({k:x[k] for k in ("job","position_id","ply","k","budget","beta_status","beta","beta_MF","H","fit_KL")}|{"campaign":name})
 # Separate observed native LR from conditional thermal cycles.
 out=dict(counts=cnt,games=games,native=native,scale_response=res,joint_action=action,clock_memory=memory,categorical_clock_memory=categorical,
   probes=probe,thermal_cycles=cycle,quantum_unique_channel_pairs=len(cache),temperature=temp,
   prior_physics_sha256=reg["previous_physics_sha"],rng_helper_sha256=sha256(helper_path),
   scope="fixed same MixNet scale pilot; native LR and fitted empirical measure; conditional CP qubit tomography and declared two-level thermal cycles; no physical native phase, heat or spatial criticality")
 atomic_json(S/"PHYSICS_RESULTS.json",jsonable(out));np.savez_compressed(S/"CHOI_STATES.npz",**states)
 print("COUNTS",json.dumps({k:{x:y for x,y in v.items() if x!="runtime_diagnostics"} for k,v in cnt.items()}))
 print("GAMES",games)
 print("MEMORY",json.dumps([{**{k:v for k,v in x.items() if k!="models"},"models":{k:v["mse"] for k,v in x["models"].items()}} for x in memory]))
 print("ACTION",len(action),sum(x["status"]=="full_rank" for x in action),max(x.get("duality_error",0) for x in action))
 print("PROBES",len(probe),"unique",len(cache),"CYCLES",len(cycle))
if __name__=="__main__":main()
