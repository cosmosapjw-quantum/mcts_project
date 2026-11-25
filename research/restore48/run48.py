"""Receipt-checked scale data, paired jackknife, causal time block response.
Prior traces reused via recorded summary identity, not counted as new."""
from pathlib import Path
import json,subprocess,statistics as st,math
from collections import defaultdict,Counter
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics48 import response_precision,causal_kernel,prefix_finite
from mqft.physics47 import clock_closure,categorical_clock_memory,joint_action
from raw47 import cohort,old_wave,jsonable
S=Path(__file__).resolve().parent
B=S.parent
def read(p):return json.loads(p.read_text())
def aggregate(a):
    # Fixed chosen roots; conditional sampling SE separate from across-root
    # descriptive variation. Seed groups independent across root/law.
    v=np.asarray([x[0] for x in a]);s=np.asarray([x[1] for x in a])
    return dict(mean=v.mean(0).tolist(),fixed_root_se=(np.sqrt((s*s).sum(0))/len(a)).tolist(),
                root_SE=(v.std(0,ddof=1)/math.sqrt(len(a))).tolist(),n_roots=len(a))

def main():
    r=read(S/"REGISTRATION.json");p47=B/"mqft_restore47_20261126";p46=B/"mqft_restore46_20261126"
    assert sha256(p47/"PHYSICS_RESULTS.json")==r["previous_physics_sha"] and sha256(p46/"PHYSICS_RESULTS.json")==r["previous46_sha"]
    old=read(p47/"PHYSICS_RESULTS.json");prev=read(p46/"PHYSICS_RESULTS.json")
    exe=p46/"runtime/mqft-rng-replay";assert sha256(exe)==prev["rng_replay_sha256"]
    counts={};fresh=[]
    for name in ("scale32a","scale32b","scale64"):
        x=read(S/"audit"/(name+".json"))
        assert x["helper_sha"]==sha256(exe) and x["reducer_sha"]==sha256(S/"raw47.py")
        assert x["plan_sha"]==sha256(S/"plans"/(name+".json")) and x["manifest_sha"]==sha256(S/"analysis"/name/"manifest.json")
        assert x["report_sha"]==sha256(next((S/"data"/name).glob("report-*.json")))
        counts[name]=x["counts"];fresh.extend(x["native"])
        print("REUSED AUDIT",name,x["counts"]["jobs"],flush=True)
    assert len(fresh)==1344 and sum(c["backups"] for c in counts.values())==59136
    assert sum(c["rng_root"] for c in counts.values())==59136
    previous=[x for x in old["native"] if x["campaign"]!="heldout64"]
    assert len(previous)==192 and not {x["seed"] for x in fresh}&{x["seed"] for x in old["native"]+prev["native"]}
    scale=previous+fresh;assert len(scale)==1536
    anchors={x["position"] for x in scale};assert len(anchors)==4
    prior=[old_wave(x) for x in prev["native"] if x["position"] in anchors]
    groups=defaultdict(lambda:defaultdict(list));seeds=defaultdict(list)
    for x in scale+prior:
        key=(x["position"],x["law"],x["k"]);groups[key][x["h"]].append(x)
        seeds[x["seed"]].append(x)
    assert len(groups)==36 and all(all(len(x)==16 for x in a.values()) for a in groups.values())
    assert all(len({(x["position"],x["law"],x["k"]) for x in a})==1 for a in seeds.values())
    result=[];kernels=[];rich=[];action=[];prefix=[]
    for (pos,law,b),a in sorted(groups.items()):
        assert set(a)==({-.3,0.,.3} if b==16 else {-.2,-.1,0.,.1,.2} if b==32 else {-.2,0.,.2})
        z=sorted(a[0.],key=lambda x:x["seed"])
        q,t=response_precision(a)
        result.extend(dict(position=pos,law=law,B=b,**x) for x in q)
        if t:rich.append(dict(position=pos,law=law,B=b,**t))
        kernels.append(dict(position=pos,law=law,**causal_kernel(z)))
        prefix.extend(dict(position=pos,law=law,B=b,**x) for x in prefix_finite(a))
        # Same fixed split and four J vectors as R47, but new B32/64 data.
        # Not used to tune any models. Rank-deficient is a recorded result.
        if b!=16:
            action.append(dict(position=pos,law=law,B=b,**joint_action(z[:8],z[8:])))
    assert len(prefix)==48
    assert len(result)==48 and len(kernels)==36 and len(rich)==12 and len(action)==24
    summary=[]
    for B0,h in ((16,.3),(32,.1),(32,.2),(64,.2)):
        for law in ("iid","pl","reserve"):
            a=[x for x in result if x["B"]==B0 and x["h"]==h and x["law"]==law];assert len(a)==4
            summary.append(dict(B=B0,h=h,law=law,n_per_arm=16,metrics={k:aggregate([(x["values"][k],x["se"][k]) for x in a]) for k in a[0]["values"]},
                mean_Lplus=st.mean(x["Lplus"] for x in a),mean_Lminus=st.mean(x["Lminus"] for x in a),
                mean_ESS=st.mean(x["ESSplus"] for x in a)))
    # Temperature/entropy cut summary from actual h0 native traces.
    # Average repeats within root, then equal-weight roots; report fit
    # status rather than forcing finite beta values.
    rootbins=defaultdict(list)
    for stg,name in ((p47,"scale32"),(p47,"scale64"),(S,"scale32a"),(S,"scale32b"),(S,"scale64")):
        pl=read(stg/"plans"/(name+".json"));js={j["id"]:(j["position_id"],j["spec"]["root_law"]["kind"],j["spec"]["budget"]) for j in pl["jobs"] if j["spec"]["root_law"]["logit_h"]==0}
        with (stg/"analysis"/name/"local.jsonl").open() as f:
            for line in f:
                x=json.loads(line)
                if x["job"] in js:
                    pos,law,bu=js[x["job"]];rootbins[pos,law,bu,x["k"]].append((x["H"],x["beta_status"],x["beta"]))
    temperature=[]
    for (pos,law,bu,k),a in sorted(rootbins.items()):
        assert len(a)==16
        b=[x[2] for x in a if x[1]=="finite"]
        temperature.append(dict(position=pos,law=law,B=bu,k=k,H=st.mean(x[0] for x in a),status=dict(Counter(x[1] for x in a)),finite_beta_median=st.median(b) if b else None))
    # No fresh independent test roots in this iteration. Reuse frozen R47
    # test only as repeated evaluation with unchanged hyperparameters.
    train=[old_wave(x) for x in prev["native"] if x["h"]==0]+[x for x in scale if x["h"]==0]
    test=[x for x in old["native"] if x["campaign"]=="heldout64"]
    memory=[clock_closure(train,test,l) for l in ("iid","pl","reserve")]
    cate=[categorical_clock_memory(train,test,l) for l in ("iid","pl","reserve")]
    out=dict(counts=counts,fresh_histories=len(fresh),scale_histories=len(scale),cumulative_marked_histories=1392+len(fresh),fresh=fresh,
        response=result,summary=summary,richardson=rich,causal_kernels=kernels,prefix_finite=prefix,joint_action=action,memory=memory,categorical_memory=cate,temperature=temperature,
        conditional_sha=sha256(S/"CONDITIONAL_RESULTS.json"),size_sha=sha256(S/"SIZE_SUPPORT.json"),rng_helper_sha=sha256(exe),
        scope="same four fixed anchors, 16 repeats per arm, budget-specific seeds; causal response from actual baseline, no time-varying direct intervention; old holdout repeated evaluation, not fresh")
    atomic_json(S/"PHYSICS_RESULTS.json",jsonable(out))
    for x in summary:
        print("SCALE",x["B"],x["h"],x["law"],{k:v["mean"] for k,v in x["metrics"].items() if k in ("LR","direct","LR2","direct2","LR_minus_direct","LR2_minus_direct2")},flush=True)
    print("COUNTS",json.dumps({k:{x:y for x,y in v.items() if x!="runtime_diagnostics"} for k,v in counts.items()}))
if __name__=="__main__":main()
