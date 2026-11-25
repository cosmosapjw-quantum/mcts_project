"""Receipt-checked streaming analysis of future actual data. No synthetic fallbacks."""
from collections import defaultdict,Counter
import gzip
import json
import math
import random
from pathlib import Path
import numpy as np
from . import SCHEMA
from .client import sha256,digest
from .campaign import PLAN_SCHEMA,read,verify_state,valid_moves,choose,seed,atomic_json,win,win_at
from .path_law import replay as replay_root_law,path_weight_statistics
from .observables import root_observables,path_entropy,graph_and_coarse,marks,delta,empirical_mi,entropy,frozen_action

def nnue_wdl_value(wdl,side):
    a=np.asarray(wdl,float)
    if a.shape!=(3,) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1) or abs(float(a.sum())-1)>0.001:raise ValueError("NNUE WDL")
    return side*(a[0]-a[1])/a.sum()

def nnue_policy(scores,board):
    legal=[i for i,x in enumerate(board) if x==0]
    if [x[0] for x in scores]!=legal or any(type(x[1])!=int for x in scores):raise ValueError("NNUE raw policy support")
    x=np.array([z[1] for z in scores],float);w=np.exp((x-x.max())/32.0)
    return dict(zip(legal,((1-1e-6)*w/w.sum()+1e-6/len(legal)).tolist()))

def check_trace(tr,moves,spec):
    if tr["schema"]!=SCHEMA or tr.get("quartz_commit")!="65e7d3e9596ca259ebfe523b0c74058692b6a755":raise ValueError("trace schema or engine source")
    verify_state(tr["root"],moves)
    actual=tr["spec"]
    if set(spec)-set(actual):raise ValueError("missing spec fields")
    for k,v in spec.items():
        if isinstance(v,float):
            if not math.isclose(actual[k],v,abs_tol=1e-7,rel_tol=1e-7):raise ValueError(f"spec mismatch {k}")
        elif actual[k]!=v:raise ValueError(f"spec mismatch {k}")
    nnue=actual.get("evaluator","uniform_rollout")=="rapfi_nnue_lookahead_v1"
    if actual.get("evaluator","uniform_rollout") not in ("uniform_rollout","rapfi_nnue_lookahead_v1"):raise ValueError("unknown evaluator identity")
    base_board=tr["root"]["board"];legal=tr["root"]["legal"];c=actual["c_puct"];cuts=actual["cuts"]
    replicas=tr["replicas"];nr=actual["workers"] if actual["mode"]=="root" else 1
    if len(replicas)!=nr or [r["replica"] for r in replicas]!=list(range(nr)):raise ValueError("replica identities")
    local=[];prevs={};deltas=[];global_rows=[];frag=[];laws=[];features=[]
    for rep in replicas:
        ri=rep["replica"];ev=rep["events"];
        law_audit=replay_root_law(rep,actual);features.extend(dict(replica=ri,**z) for z in law_audit.pop("history_features",[]));laws.append(dict(replica=ri,**law_audit));evaluations={e["id"]:e for e in rep["evaluations"]}
        if len(evaluations)!=len(rep["evaluations"]) or sorted(evaluations)!=list(range(len(evaluations))):raise ValueError("evaluator ids")
        if [e["complete_seq"] for e in ev]!=list(range(len(ev))) or sorted(e["select_seq"] for e in ev)!=list(range(len(ev))):raise ValueError("event sequence")
        used=[];eval_states={0:(base_board,len(moves))}
        for e in ev:
            word=e["actions"];v=e["leaf_value"];r=e["root_value"]
            if not word or not math.isfinite(v) or not -1<=v<=1 or r!=((-1)**len(word))*v:raise ValueError("negamax mark")
            if len(e["parent_hashes"])!=len(word) or len(e["edge_priors"])!=len(word) or len(e["vl_visit"])!=len(word) or len(e["vl_value"])!=len(word):raise ValueError("path schema")
            b=base_board.copy();w=None
            for j,a in enumerate(word):
                if w is not None or type(a)!=int or not 0<=a<225 or b[a]:raise ValueError("invalid native path")
                b[a]=1+(len(moves)+j)%2
                if win_at(b,a,b[a]):w=1 if b[a]==1 else -1
            terminal=w is not None or len(moves)+len(word)==225
            if terminal!=e["terminal"]:raise ValueError("leaf terminal mismatch")
            if terminal:
                side=1 if (len(moves)+len(word))%2==0 else -1
                if e["eval_id"] is not None or v!=(w*side if w is not None else 0):raise ValueError("terminal return mismatch")
            else:
                eid=e["eval_id"]
                if eid is None or eid not in evaluations:raise ValueError("missing actual evaluation receipt")
                a=evaluations[eid]
                if a["state_hash"]!=e["leaf_hash"] or abs(a["value"]-v)>1e-7:raise ValueError("evaluation identity")
                used.append(eid);eval_states[eid]=(b,len(moves)+len(word))
        if sorted(used)!=list(range(1,len(evaluations))) or evaluations[0]["state_hash"]!=tr["root"]["state_hash"]:
            raise ValueError("expected root expansion eval plus exact nonterminal eval bindings")
        for a in evaluations.values():
            sam=a["samples"]
            if len(sam)!=actual["samples_per_eval"] or any(not math.isfinite(s["value"]) or not -1<=s["value"]<=1 or not 0<=s["plies"]<=actual["rollout_depth"] or (not nnue and (s["value"] not in (-1.,0.,1.) or (not s["terminal"] and s["value"]!=0.))) for s in sam):raise ValueError("sample provenance")
            if abs(a["value"]-sum(s["value"] for s in sam)/len(sam))>1e-7:raise ValueError("leaf mean")
            board,ply=eval_states[a["id"]]
            if a["side"]!=(1 if ply%2==0 else -1):raise ValueError("evaluator frame")
            if nnue:
                policy=nnue_policy(a["nnue_scores"],board)
                nnue_wdl_value(a["nnue_wdl"],a["side"])
                if a["id"]==0:
                    p=dict(policy);source=actual.get("source_action")
                    if source is not None:p[source]*=math.exp(actual["source_h"])
                    den=sum(p.values())
                    for cut in rep["cuts"]:
                        if any(abs(ed["prior"]-p[ed["action"]]/den)>2e-6 for ed in cut["edges"]):raise ValueError("NNUE root prior mismatch")
            elif a.get("nnue_scores") is not None:raise ValueError("unregistered NNUE evaluation")
            for r in sam:
                if nnue:
                    word=r["actions"]
                    if r["plies"]!=1 or not isinstance(word,list) or len(word)!=1 or word[0] not in policy:raise ValueError("NNUE sample action")
                    exact=math.log(policy[word[0]])
                else:exact=-sum(math.log(225-ply-i) for i in range(r["plies"]))
                if abs(r["log_path_prob"]-exact)>1e-8:raise ValueError("rollout log probability")
                if actual.get("record_rollouts"):
                    word=r["actions"]
                    if not isinstance(word,list) or len(word)!=r["plies"]:raise ValueError("missing rollout path")
                    z=board.copy();winner=None
                    for i,move in enumerate(word):
                        if winner is not None or type(move)!=int or not 0<=move<225 or z[move]:raise ValueError("illegal rollout path")
                        z[move]=1+(ply+i)%2
                        if win_at(z,move,z[move]):winner=1 if z[move]==1 else -1
                    terminal=winner is not None or ply+len(word)==225
                    val=winner*a["side"] if winner is not None else 0
                    if nnue and not terminal:
                        # Actual WDL is in black POV; value transported to original leaf.
                        val=nnue_wdl_value(r["nnue_wdl"],a["side"])
                    if terminal!=r["terminal"] or abs(val-r["value"])>1e-7 or (not terminal and len(word)!=actual["rollout_depth"]):raise ValueError("rollout mark replay")
                    if nnue and terminal and r.get("nnue_wdl") is not None:raise ValueError("terminal sample must use exact rule")
                elif r["actions"] is not None:raise ValueError("unregistered rollout path")
        if [x["global_k"] for x in rep["cuts"]]!=cuts:raise ValueError("cuts mismatch")
        for cut in rep["cuts"]:
            k=cut["local_k"];K=cut["global_k"];edges=cut["edges"]
            expected=K//nr+int(ri<K%nr)
            if k!=expected or len(ev)<k or [x["action"] for x in edges]!=sorted(legal):raise ValueError("cut shape")
            m=marks(ev[:k])
            if sum(x["n"] for x in edges)!=k or abs(sum(x["w"] for x in edges)-cut["root_w"])>1e-8*(1+k):raise ValueError("root conservation")
            if abs(sum(x["prior"] for x in edges)-1)>2e-6:raise ValueError("prior sum")
            for x in edges:
                n,w,s2=m.get((x["action"],),(0,0.,0.))
                if x["n"]!=n or abs(x["w"]-w)>1e-9*(1+k) or abs(x["s2"]-s2)>1e-9*(1+k) or x["pending"]!=0 or abs(x["pending_value"])>1e-9:
                    raise ValueError("independent edge mark replay failed")
            # All evaluations at completed cut: root initialization plus linked events.
            ids=[0]+[x["eval_id"] for x in ev[:k] if x["eval_id"] is not None]
            if len(ids)!=cut["eval_calls"] or sum(len(evaluations[i]["samples"]) for i in ids)!=cut["rollout_samples"] or sum(s["plies"] for i in ids for s in evaluations[i]["samples"])!=cut["rollout_plies"]:
                raise ValueError("cost accounting at barrier")
            if k==0:continue
            a=root_observables(edges,k,c);pe=path_entropy(ev[:k]);gd=graph_and_coarse(ev[:k],seed=seed(actual["seed"],ri,K))
            # Completed paths are non-IID, real prefix-tree measure; no physical entropy identification.
            row=dict(replica=ri,k=k,global_k=K,allocated_nodes=cut["allocated_nodes"],
                     eval_calls=cut["eval_calls"],rollout_samples=cut["rollout_samples"],rollout_plies=cut["rollout_plies"],
                     mean_depth=float(np.mean([len(x["actions"]) for x in ev[:k]])),max_depth=max(len(x["actions"]) for x in ev[:k]),
                     **a,**pe,**gd)
            if ri in prevs:
                b=prevs[ri]
                deltas.append(dict(replica=ri,k0=b["k"],k=k,global_k=K,**delta(b,a)))
            prevs[ri]={**a,"k":k};local.append(row)
        # Fragment chronology: completed-event blocks; tree observations not linearized.
        # For wave, phases are actual wave barriers; in root mode no cross-replica event ordering.
        if len(ev)!=rep["cuts"][-1]["local_k"]:raise ValueError("final event count")
        blocks=defaultdict(list)
        for e in ev:blocks[e["phase"]].append(e)
        for phase,events in sorted(blocks.items()):
            cts=Counter(x["actions"][0] for x in events)
            frag.append(dict(replica=ri,phase=phase,n=len(events),leader=min(a for a,n in cts.items() if n==max(cts.values())),
                root_H=entropy(np.array(list(cts.values()),float)/len(events)),duplicate_root_pairs=sum(n*(n-1)//2 for n in cts.values()),
                same_path_pairs=sum(n*(n-1)//2 for n in Counter(tuple(x["actions"]) for x in events).values())))
    for i,K in enumerate(cuts):
        # Root parallel aggregate only counts and moments. No fictitious pooled PUCT law.
        es=[r["cuts"][i] for r in replicas]
        e0=es[0]["edges"];n=np.sum([[e["n"] for e in x["edges"]] for x in es],axis=0)
        w=np.sum([[e["w"] for e in x["edges"]] for x in es],axis=0);s2=np.sum([[e["s2"] for e in x["edges"]] for x in es],axis=0)
        if any([e["prior"] for e in x["edges"]]!=[e["prior"] for e in e0] for x in es) or int(n.sum())!=K:raise ValueError("root replica support/normalization")
        pi=n/K;mu=w.sum()/K;leaders=[e["action"] for e,ct in zip(e0,n) if ct==n.max()]
        global_rows.append(dict(k=K,actions=[e["action"] for e in e0],counts=n.tolist(),pi=pi.tolist(),q=float(mu),H=entropy(pi),
            variance=max(0.,float(s2.sum()/K-mu**2)),order_parameter=float(np.sort(pi)[-1]-(np.sort(pi)[-2] if len(pi)>1 else 0.)),effective_actions=float(1/(pi@pi)),
            H_fraction=entropy(pi)/math.log(min(K,len(pi))) if min(K,len(pi))>1 else None,
            leaders=leaders,allocated_nodes=sum(x["allocated_nodes"] for x in es),eval_calls=sum(x["eval_calls"] for x in es),
            rollout_samples=sum(x["rollout_samples"] for x in es),rollout_plies=sum(x["rollout_plies"] for x in es)))
    if tr["move_counts"]!=[[a,n] for a,n in zip(global_rows[-1]["actions"],global_rows[-1]["counts"])]:raise ValueError("selected distribution not backed by events")
    return dict(local=local,global_rows=global_rows,deltas=deltas,fragments=frag,path_laws=laws,history_features=features)

def load_receipted(path,plan):
    folder=Path(path);lock=read(folder/"run_lock.json")
    if lock["plan_hash"]!=plan["plan_hash"]:raise ValueError("plan lock mismatch")
    jobs={j["id"]:j for j in plan["jobs"]}
    seen=set()
    for f in sorted(folder.glob("*.receipt.json")):
        r=read(f);jid=r["job_id"]
        if jid not in jobs or jid in seen or r["plan_hash"]!=plan["plan_hash"] or r["backend_sha"]!=lock["backend_sha"] or r["job_hash"]!=digest(jobs[jid]):raise ValueError("receipt identity")
        seen.add(jid);p=folder/r["file"]
        if p.name!=f"{jid}.jsonl.gz" or p.stat().st_size!=r["bytes"] or sha256(p)!=r["sha256"]:raise ValueError("shard bytes")
        yield jobs[jid],p,r,lock

def analyze(plan,datadir,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():raise FileExistsError("analysis directory already sealed")
    files={name:(out/(name+".jsonl.part")).open("x") for name in ("local","global","deltas","fragments","games","root_repeats","source_response","path_laws","history_features","law_summary")}
    completed=0;root_groups=defaultdict(list);law_groups=defaultdict(list);all_status=Counter()
    def emit(name,r):files[name].write(json.dumps(r,allow_nan=False,separators=(",",":"))+"\n")
    try:
        for job,p,receipt,lock in load_receipted(datadir,plan):
            with gzip.open(p,"rt",encoding="utf-8") as f:
                head=json.loads(next(f))
                if head["type"]!="header" or head.get("schema")!=PLAN_SCHEMA or head["job"]!=job or head["plan_hash"]!=plan["plan_hash"] or head["backend_sha"]!=receipt["backend_sha"]:raise ValueError("header provenance")
                if job["spec"].get("evaluator")=="rapfi_nnue_lookahead_v1":
                    if head.get("evaluator_runtime")!=lock["external"] or head.get("evaluator_runtime") is None:raise ValueError("common NNUE runtime receipt mismatch")
                moves=list(job["opening"]);rng=random.Random(job["game_seed"]);backups=records=0;end=None
                for line in f:
                    row=json.loads(line);typ=row["type"]
                    if end is not None:raise ValueError("trailing rows after end")
                    if typ=="end":end=row;continue
                    tr=row.get("trace");ply=len(moves)
                    if typ not in ("root","ply"):raise ValueError("unknown row")
                    if typ=="root" and (job["kind"]!="roots" or row["moves"]!=moves or records!=0):raise ValueError("root provenance")
                    if typ=="ply":
                        if job["kind"]!="games" or row["ply"]!=ply or ply>=job["max_plies"]:raise ValueError("ply chronology")
                        verify_state(row["state"],moves)
                        a=row["action"]
                        if a not in row["state"]["legal"]:raise ValueError("illegal actual move")
                    if typ=="ply":
                        use_ext=(job["opponent"]=="rapfi_black" and row["state"]["side"]==1) or (job["opponent"]=="rapfi_white" and row["state"]["side"]==-1)
                        if (row["engine"]=="rapfi")!=use_ext:raise ValueError("opponent turn mismatch")
                        if (tr is not None)!=(row["engine"]=="quartz"):raise ValueError("engine trace mismatch")
                    if tr is not None:
                        spec=job["spec"].copy()
                        if typ=="ply":
                            spec["seed"]=seed(spec["seed"],ply);spec["source_action"]=None;spec["source_h"]=0.
                            if (job["opponent"]=="serial_black" and ply%2==0) or (job["opponent"]=="serial_white" and ply%2==1):
                                spec["mode"]="serial";spec["workers"]=1
                        q=check_trace(tr,moves,spec);backups+=spec["budget"]
                        meta=dict(job=job["id"],block=job["block"],kind=job["kind"],position_id=job["position_id"],replicate=job["replicate"],
                                  mode=spec["mode"],workers=spec["workers"],budget=spec["budget"],samples_per_eval=spec["samples_per_eval"],
                                  c_puct=spec["c_puct"],source_h=spec["source_h"],source_action=spec["source_action"],law_id=digest(spec.get("root_law"))[:12],opponent=job["opponent"],ply=ply,
                                  state_hash=tr["root"]["state_hash"],source_file=p.name,backend_sha=receipt["backend_sha"],evaluator=spec.get("evaluator","uniform_rollout"),candidate_mode=job["spec"]["mode"],player_side=tr["root"]["side"])
                        for name in ("local","deltas","fragments","path_laws","history_features"):
                            for z in q[name]:emit(name,{**meta,**z})
                        if typ=="root":
                            for z in q["path_laws"]:
                                if z.get("proposal")=="path":
                                    key=tuple(meta[x] for x in ("position_id","state_hash","mode","workers","budget","samples_per_eval","c_puct","source_h","source_action","law_id"))
                                    law_groups[key].append({**meta,**z})
                        for z in q["global_rows"]:
                            emit("global",{**meta,**z,"actual_move":row.get("action"),"wall_ns":tr["wall_ns"] if z["k"]==spec["budget"] else None})
                        if typ=="ply":
                            if choose(tr["move_counts"],job["move_temperature"],rng)!=a:raise ValueError("move RNG replay")
                        else:
                            # Pre-registered equal completed-event windows, one ordered sequence per replica.
                            # For pooled root mode define fragment as joint tuple of per-replica leader, not artificial total chronology.
                            k=spec["budget"];fr=[]
                            for j in range(4):
                                labs=[]
                                for rep in tr["replicas"]:
                                    ev=rep["events"];lo=len(ev)*j//4;hi=len(ev)*(j+1)//4;cts=Counter(x["actions"][0] for x in ev[lo:hi])
                                    labs.append(min(a for a,n in cts.items() if n==max(cts.values())) if cts else -1)
                                fr.append(tuple(labs))
                            fin=q["global_rows"][-1];decision=min(fin["leaders"])
                            key=tuple(meta[x] for x in ("position_id","state_hash","mode","workers","budget","samples_per_eval","c_puct","source_h","source_action","law_id"))
                            root_groups[key].append(dict(meta=meta,decision=decision,fragments=fr,pi=fin["pi"],actions=fin["actions"],q=fin["q"],locals=[z for z in q["local"] if z["global_k"]==k]))
                    elif typ!="ply" or row["engine"]!="rapfi" or row["external"]["identity"]!=lock["external"] or row["external"]["action"]!=a:
                        raise ValueError("missing native trace or external identity")
                    else:
                        from .rapfi import parse
                        if parse(row["external"]["stdout"],moves)!=a:raise ValueError("external raw response replay")
                    if typ=="ply":moves.append(a)
                    records+=1
                if end is None or end["moves"]!=moves or end["backups"]!=backups or end["searched_records"]!=records or receipt["backups"]!=backups or receipt["records"]!=records or receipt["status"]!=end["status"]:
                    raise ValueError("end accounting")
                b=valid_moves(moves);w=1 if win(b,1) else -1 if win(b,2) else None;term=bool(w is not None or len(moves)==225)
                if end["terminal"]!=term or end["winner"]!=w:raise ValueError("terminal outcome")
                if job["kind"]=="games" and (end["status"]!=("natural_terminal" if term else "truncated") or (not term and len(moves)!=job["max_plies"])):raise ValueError("game terminal status")
                if job["kind"]=="roots" and (records!=1 or end["status"]!="root_complete"):raise ValueError("root shard status")
                emit("games",dict(job=job["id"],block=job["block"],mode=job["spec"]["mode"],workers=job["spec"]["workers"],budget=job["spec"]["budget"],opponent=job["opponent"],kind=job["kind"],status=end["status"],winner=w,plies=len(moves),opening=job["opening"],moves=moves,backups=backups))
                completed+=1;all_status[end["status"]]+=1
        if completed==0:raise ValueError("no receipted actual data")
        for key,rr in law_groups.items():
            emit("law_summary",{**dict(zip(("position_id","state_hash","mode","workers","budget","samples_per_eval","c_puct","source_h","source_action","law_id"),key)),"jobs":[r["job"] for r in rr],"n_paths":len(rr),"targets":path_weight_statistics(rr),
                                "scope":"fixed-root replicated path proposal; no physical work identity"})
        for key,rr in root_groups.items():
            if len(rr)<2:continue
            meta=rr[0]["meta"].copy();meta.pop("job");meta.pop("block");meta.pop("replicate");meta.pop("source_file")
            ds=[x["decision"] for x in rr];metrics=[]
            for j in range(4):
                ys=[tuple(x["fragments"][:j+1]) for x in rr];m=empirical_mi(ds,ys)
                null=[];rng=random.Random(seed(key,j))
                for _ in range(199):
                    z=ds.copy();rng.shuffle(z);null.append(empirical_mi(z,ys)["I"])
                disjoint=empirical_mi(ds,[x["fragments"][j] for x in rr])
                metrics.append(dict(fragment_count=j+1,**m,disjoint_I=disjoint["I"],increment=m["I"]-(metrics[-1]["I"] if metrics else 0.),majority_baseline_accuracy=max(Counter(ds).values())/len(ds),null_mean=float(np.mean(null)),p_permutation=(1+sum(x>=m["I"]-1e-14 for x in null))/200))
            a=np.array([x["pi"] for x in rr]);mean=a.mean(0);cov=(a-mean).T@(a-mean)/(len(a)-1)
            emit("root_repeats",{**meta,"n":len(rr),"metrics":metrics,"actions":rr[0]["actions"],"mean_pi":mean.tolist(),
                "covariance":cov.tolist(),"distinct_final":len(set(ds)),"mean_q":float(np.mean([x["q"] for x in rr])),
                "records":[{"replicate":x["meta"]["replicate"],"decision":x["decision"],"fragments":x["fragments"],"pi":x["pi"]} for x in rr]})
        # Finite, paired source response at fixed root and declared perturbation.
        paired=defaultdict(dict)
        for key,rr in root_groups.items():paired[key[:7]+key[8:]][key[7]]=rr
        for key,groups in paired.items():
            if 0 not in groups or key[-2] is None:continue
            for h in sorted(x for x in groups if x>0 and -x in groups):
                by=[{x["meta"]["replicate"]:x for x in groups[g]} for g in (-h,0,h)]
                common=sorted(set(by[0])&set(by[1])&set(by[2]))
                for rep in common:
                    minus,base,plus=(d[rep] for d in by)
                    a=key[-2];idx=base["actions"].index(a)
                    measured=(plus["pi"][idx]-minus["pi"][idx])/(2*h)
                    pred=[];adapt=[];q_shift=[];faces=[]
                    for hh,arm in ((-h,minus),(h,plus)):
                        total=0.;adaptive=0.;shifts=[];changed=0
                        for z,za in zip(base["locals"],arm["locals"]):
                            p=np.array(z["prior"],float);pt=p.copy();pt[idx]*=math.exp(hh);pt*=p.sum()/pt.sum()
                            f=frozen_action(z["action_q"],pt,z["k"],base["meta"]["c_puct"],z["pi"])
                            total+=z["k"]*f["optimum"][idx]
                            af=frozen_action(za["action_q"],pt,z["k"],base["meta"]["c_puct"],za["pi"])
                            adaptive+=z["k"]*af["optimum"][idx]
                            shifts.append(float(np.max(abs(np.array(za["action_q"])-z["action_q"]))))
                            f0=frozen_action(z["action_q"],p,z["k"],base["meta"]["c_puct"],z["pi"])
                            changed+=int(f["active_mask"]!=f0["active_mask"])
                        pred.append(total/base["meta"]["budget"]);adapt.append(adaptive/base["meta"]["budget"]);q_shift.append(max(shifts));faces.append(changed)
                    emit("source_response",{**base["meta"],"h":h,"measured":measured,"frozen_kkt":(pred[1]-pred[0])/(2*h),"adaptive_kkt_diagnostic":(adapt[1]-adapt[0])/(2*h),"max_q_shift":max(q_shift),"face_changes":sum(faces),"source":"paired finite difference; not FDT or one-loop proof"})
    finally:
        for f in files.values():f.close()
    outputs={}
    for name in files:
        src=out/(name+".jsonl.part");dst=out/(name+".jsonl");src.rename(dst);outputs[dst.name]=sha256(dst)
    manifest=dict(schema="mqft.analysis.v2",plan_hash=plan["plan_hash"],completed=completed,expected=len(plan["jobs"]),statuses=dict(all_status),outputs=outputs,
                  limitations=["partial campaigns remain partial","root replicas not pooled into a PUCT law","physical phase and temperature not identified","non-IID path/ply measurements"])
    atomic_json(out/"manifest.json",manifest)
    return manifest
