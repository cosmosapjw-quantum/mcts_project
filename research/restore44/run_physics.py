"""RESTORE44 plot research, backed by actual receipt and restricted references."""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip,math,statistics as st
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.analysis import check_trace
from mqft.physics44 import *
from mqft.framework import graph_reduction
from mqft.closure import evaluate as closure_evaluate
from mqft.observables import marks
S=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.open()]
def jsonable(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
 return x
def audit_cohort(name):
 d=S/"data"/name;pl=read(S/"plans"/(name+".json"));lock=read(d/"run_lock.json");reg=read(S/"REGISTRATION.json")
 m=read(S/"analysis"/name/"manifest.json")
 assert m["completed"]==m["expected"]==len(pl["jobs"]) and m["plan_hash"]==lock["plan_hash"]==pl["plan_hash"]
 assert lock["external"]==reg["model"] and lock["backend_sha"]==reg["backend_sha"]
 assert all(sha256(S/"analysis"/name/n)==h for n,h in m["outputs"].items())
 jobs={j["id"]:j for j in pl["jobs"]};report=read(next(d.glob("report-*.json")))
 assert not report["failures"] and not report["unattempted"] and len(report["completed"])==len(jobs)
 tr=[];cost=Counter();games=[];maxerr=0.;bts=0
 for r in report["completed"]:
  j=jobs[r["job_id"]];p=d/r["file"]
  assert sha256(p)==r["sha256"] and p.stat().st_size==r["bytes"] and read(d/(j["id"]+".receipt.json"))==r
  bts+=r["bytes"]
  with gzip.open(p,"rt") as g:
   hd=json.loads(next(g));assert hd["job"]==j and hd["evaluator_runtime"]==reg["model"]
   moves=j["opening"].copy();backup=0;nr=0;end=None
   for l in g:
    row=json.loads(l)
    if row["type"]=="end":end=row;continue
    t=row["trace"];sp=t["spec"]
    assert sp["evaluator"]=="rapfi_nnue_lookahead_v1" and sp["samples_per_eval"]==4
    assert len(t["root"]["board"])==225 and t["root"]["side"]==(1 if len(moves)%2==0 else -1)
    # Second independent raw moment reduction, path parity and model sample sign.
    for rep in t["replicas"]:
     ev=rep["events"];cost["replica_cuts"]+=len(rep["cuts"])
     for e in ev:assert e["root_value"]==(-1)**len(e["actions"])*e["leaf_value"]
     for cut in rep["cuts"]:
      a=defaultdict(list)
      for e in ev[:cut["local_k"]]:a[e["actions"][0]].append(e["root_value"])
      for ed in cut["edges"]:
       v=a[ed["action"]];assert ed["n"]==len(v) and ed["pending"]==0
       err=max(abs(math.fsum(v)-ed["w"]),abs(math.fsum(x*x for x in v)-ed["s2"]))
       assert err<1e-8*(1+len(v));maxerr=max(maxerr,err)
     for ev0 in rep["evaluations"]:
      assert ev0["nnue_scores"] and len(ev0["samples"])==4
      cost["parent_queries"]+=1;cost["samples"]+=4
      for sm in ev0["samples"]:
       if not sm["terminal"]:
        w,l,dr=sm["nnue_wdl"];assert abs(ev0["side"]*(w-l)/(w+l+dr)-sm["value"])<1e-7;cost["child_queries"]+=1
       else:assert sm.get("nnue_wdl") is None
    backup+=sp["budget"];nr+=1;cost["searches"]+=1;cost["wall_ns"]+=t["wall_ns"]
    if name=="marked":tr.append((j,t,r))
    else:
     assert row["action"] in t["root"]["legal"];moves.append(row["action"])
   assert end and end["moves"]==moves and backup==r["backups"] and nr==r["records"]
   if name=="games":
    assert end["terminal"] and end["status"]=="natural_terminal"
    games.append(dict(id=j["id"],position=j["position_id"],plies=len(moves),winner=end["winner"],searched=nr))
  cost["backups"]+=backup;cost["records"]+=nr
 local=lines(S/"analysis"/name/"local.jsonl");deltas=lines(S/"analysis"/name/"deltas.jsonl")
 diagnostics=[f.read_text() for f in (d/"diagnostics").glob("*.log") if f.stat().st_size]
 return dict(jobs=len(jobs),compressed_bytes=bts,**dict(cost),moment_error=maxerr,max_KKT=max(r["kkt_residual"] for r in local),
     max_score_balance=max(abs(r["dE"]-r["field"]-r["redistribution"]) for r in deltas),
     global_cuts=len(lines(S/"analysis"/name/"global.jsonl")),runtime_retries=sum(x.count("MQFT_RAPFI_RETRY") for x in diagnostics),
     runtime_diagnostics=diagnostics),games,tr

def graph_ref(t):
 rep=t["replicas"][0];m=marks(rep["events"]);words=[()]+sorted(m,key=lambda x:(len(x),x))
 lookup={x:i for i,x in enumerate(words)};mass=np.array([len(rep["events"])]+[m[x][0] for x in words[1:]],float);A=np.zeros((len(words),len(words)))
 for x in words[1:]:
  a,b=lookup[x],lookup[x[:-1]];A[a,b]=A[b,a]=m[x][0]
 results=[]
 for depth in [0,1,2,3,max(map(len,words))]:
  labels=[w[:depth] for w in words]
  z=graph_reduction(A,mass,labels,.5,.5)
  ge=z["galerkin"];eff=z["static_effective"];defect=z["memory_defect"]
  results.append(dict(depth=depth,nodes=len(words),coarse_dim=len(ge),memory_trace=float(np.trace(defect)),memory_max=float(np.linalg.eigvalsh(defect).max()),
    static_gap=float(np.linalg.norm(ge+.5*np.eye(len(ge))-eff)),semigroup_defect_psd=float(min(0,np.linalg.eigvalsh(defect).min()))))
 return results

def exact_native_draw_info(t):
 rep=t["replicas"][0];ev=rep["events"];timeline=[]
 for e in ev:timeline.extend([(e["root_selection"]["clock"],0,e),(e["root_backup_clock"],1,e)])
 counts=Counter();unseen=0;logp=0.;allwaves=defaultdict(list);scores0=[]
 for _,typ,e in sorted(timeline,key=lambda a:a[0]):
  a=e["actions"][0]
  if typ:counts[a]+=1;continue
  law=e["root_selection"]["law"];idx=law["picked"];unseen+=int(counts[a]==0)
  logp+=math.log(law["p"][idx]);allwaves[e["phase"]].append(a)
  scores0.append((np.asarray(law["scores"],float),np.asarray(law["active"],bool),idx))
 collision=sum(sum(n*(n-1)//2 for n in Counter(a).values()) for a in allwaves.values())
 pairs=sum(len(a)*(len(a)-1)//2 for a in allwaves.values())
 return dict(unvisited=unseen,draws=len(ev),collision=collision,pairs=pairs,nll=-logp/len(ev)),scores0

def fit_scale(train,test):
 def nll(rows,beta):
  return float(np.mean([logsumexp(beta*s[a])-beta*s[i] for s,a,i in rows]))
 z=minimize_scalar(lambda b:nll(train,b),bounds=(0,20),method="bounded")
 return dict(fitted=float(z.x),true=1/.3,train=len(train),test=len(test),test_nll_fit=nll(test,z.x),test_nll_true=nll(test,1/.3),test_nll_zero=nll(test,0))

def main():
 cg,games,_=audit_cohort("games");cm,_,tr=audit_cohort("marked")
 rows=[];refs={};scores_by=defaultdict(list);graph=[]
 for j,t,r in tr:
  sp=j["spec"];law=sp["root_law"];v=native_record(t,sp["source_action"])
  di,sc=exact_native_draw_info(t)
  v.update(position=j["position_id"],law=law["kind"],replicate=j["replicate"],seed=sp["seed"],job=j["id"],shard_sha=r["sha256"],**di)
  rows.append(v)
  if v["h"]==0:
   scores_by[v["position"],v["law"]].extend(sc)
   if v["replicate"]==0:
    graph.append(dict(position=v["position"],law=v["law"],depths=graph_ref(t)))
    if v["law"]=="iid":
     ed=next(x["edges"] for x in t["replicas"][0]["cuts"] if x["local_k"]==8)
     source=sp["source_action"];chosen=[source]+[x["action"] for x in sorted(ed,key=lambda x:(-x["prior"],x["action"])) if x["action"]!=source][:4]
     refs[v["position"]]=dict(edges=ed,chosen=chosen,source=source,job=j["id"],shard_sha=r["sha256"])
 assert len(rows)==144 and len(refs)==4
 paired=paired_summary(rows)
 scales={law:fit_scale(sum([scores_by[p,law] for p in sorted(refs)[:2]],[]),sum([scores_by[p,law] for p in sorted(refs)[2:]],[])) for law in ("iid","pl","reserve")}
 # All reference laws share the same actually observed quiescent IID state for each root.
 exact=[];qs=[];dar=[];therm=[];loop=[];cu=[];coarse=[];states={};mx=0.
 for pos,v in sorted(refs.items()):
  args=(v["edges"],v["chosen"])
  for law in ("iid","pl","reserve"):
   for width in (1,2,4):
    z=exact_wave_full_state(*args,width,law,v["source"])
    pp=exact_wave_full_state(*args,width,law,v["source"],1e-5)
    pm=exact_wave_full_state(*args,width,law,v["source"],-1e-5)
    i=v["chosen"].index(v["source"]);fd=(pp["mean"]-pm["mean"])/2e-5;err=float(np.max(abs(fd-z["R"][:,i])));assert err<1e-8;mx=max(mx,err)
    if law=="iid":assert np.max(abs(z["R"]-z["C"]))<1e-10
    key=dict(position=pos,law=law,width=width)
    exact.append(dict(**key,source=v["source"],chosen=v["chosen"],n_histories=len(z["p"]),R=z["R"].tolist(),C=z["C"].tolist(),I=z["I"].tolist(),
      tangent_singular_values=np.linalg.svd(z["R"],compute_uv=False).tolist(),response_cov_gap=float(np.linalg.norm(z["R"]-z["C"])),
      fd_error=err,errors=z["errors"],entropy=entropy(z["p"]),count_cumulants=cumulants(z)))
    coarse.append(dict(**key,maps=projected_information(z)))
    cu.append(dict(**key,**cumulants(z)))
    # Save exact complete conditional measure and scores, not just plot summaries.
    pref=f"{pos}_{law}_w{width}";states[pref+"_p"]=z["p"];states[pref+"_words"]=np.array(z["words"]);states[pref+"_G"]=z["G"];states[pref+"_J"]=z["J"]
    if width==2:
     q,st0=quantum_memory(z);qs.append(dict(**key,values=q));states.update({pref+"_"+k:a for k,a in st0.items()})
     dar.append(dict(**key,values=record_fragments(z)))
     therm.append(dict(**key,**thermal_reference(z)))
    if width==4:loop.append(dict(**key,values=saddlepoint(z)))
 assert len(exact)==36 and len(qs)==len(dar)==len(therm)==len(loop)==12
 # Held-out history prediction uses only h0/reservation law, same evaluator and schedule.
 selected={r["job"] for r in rows if r["law"]=="reserve" and r["h"]==0}
 hh=[x for x in lines(S/"analysis/marked/history_features.jsonl") if x["job"] in selected]
 ps=sorted(refs)
 closure=closure_evaluate(hh,[("roots",p) for p in ps[:2]],[("roots",p) for p in ps[2:]])
 # Actual data requirements: paired residual pilot variance, target 0.05 occupancy response SE halfwidth.
 power=[]
 for a in paired:
  rr=a["paired"];x=[z["occupancy_LR"]-z["occupancy_FD"] for z in rr];sd=st.stdev(x)
  power.append(dict(position=a["position"],law=a["law"],n=4,sd=sd,normal_n_halfwidth_005=math.ceil((1.96*sd/.05)**2) if sd>0 else None))
 # Original obstruction check: explicit 2-level Euler and unital projector reference.
 rho=np.diag([1.,0.]).astype(complex);H=np.array([[0.,1.],[1.,0.]])
 euler=[]
 for dt in (.01,.05,.1,.2):
  a=rho-1j*dt*(H@rho-rho@H);U=np.cos(dt)*np.eye(2)-1j*np.sin(dt)*H;b=U@rho@U.conj().T
  euler.append(dict(dt=dt,euler_min=float(np.linalg.eigvalsh(a).min()),unitary_min=float(np.linalg.eigvalsh(b).min()),det=float(np.linalg.det(a).real)))
 obs=[]
 for t in therm:
  # H with connected hopping and diagonal energies from same 5-action first-draw probability.
  z=next(x for x in exact if x["position"]==t["position"] and x["law"]==t["law"] and x["width"]==1)
  p=states[f"{t['position']}_{t['law']}_w1_p"];d=len(p);H=np.diag(-np.log(p))+.1*(np.ones((d,d))-np.eye(d))
  e,u=np.linalg.eigh(H);q=np.exp(-e-logsumexp(-e));rho=(u*q)@u.T
  D=np.diag(np.diag(rho))-rho
  assert np.max(abs(np.diag(D)))==0
  obs.append(dict(position=t["position"],law=t["law"],gibbs_residual=float(np.linalg.norm(D-1j*(H@rho-rho@H))),unital_error=0.,diag_error=0.))
 local=lines(S/"analysis/games/local.jsonl");temps=[{k:r[k] for k in ("job","ply","k","beta","beta_status","beta_MF","fit_KL","H","q")} for r in local]
 summary=dict(counts={"games":cg,"marked":cm},games=games,native=rows,paired=paired,scale_fits=scales,reference_inputs=refs,exact=exact,exact_fd_max=mx,coarse=coarse,graph=graph,quantum=qs,darwinism=dar,thermal=therm,loop=loop,cumulants=cu,closure=closure,power=power,obstructions=dict(euler=euler,projector=obs),native_temperature=temps,
   scope="Actual full-support native histories and restricted conditional physics references; no inferred native phase, energy, physical beta, or continuum exponents.")
 atomic_json(S/"PHYSICS_RESULTS.json",jsonable(summary))
 np.savez_compressed(S/"PHYSICS_STATES.npz",**states)
 print("COUNTS",json.dumps(summary["counts"]))
 print("GAMES",games)
 print("SCALE",scales)
 print("EXACT",len(exact),sum(x["n_histories"] for x in exact),mx)
 print("CLOSURE",closure)
 print("POWER",power)
if __name__=="__main__":main()
