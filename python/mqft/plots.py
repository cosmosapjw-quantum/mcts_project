"""Original MQFT observable renderers. Never fabricate missing measurements."""
from collections import defaultdict
import json
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .client import sha256
from .campaign import atomic_json

def rows(p):
    with open(p,encoding="utf-8") as f:
        for line in f:yield json.loads(line)

def group(r):
    return (r["kind"],r["mode"],r["workers"],r["budget"],r["c_puct"],r["samples_per_eval"],r["opponent"],r["source_h"],r.get("law_id","legacy"),r.get("evaluator","uniform_rollout"),r.get("candidate_mode",r["mode"]))

def label(g):
    kind,mode,w,b,c,s,opp,h,law,evaluator,candidate=g
    return f"{kind}/{mode} W{w} B{b} c={c:g} L{s} {opp} h={h:g} law={law} {evaluator} vs {candidate}"

def summarize(iterator,field,x,transform=None,predicate=lambda r:True):
    # Average within job first, then over jobs, never over plies as independent samples.
    a=defaultdict(lambda:[0.,0])
    for r in iterator:
        if not predicate(r):continue
        y=transform(r) if transform else r.get(field)
        if y is None or not math.isfinite(y):continue
        key=(group(r),r[x],r["job"]);a[key][0]+=y;a[key][1]+=1
    b=defaultdict(list)
    for (g,k,_),(v,n) in a.items():b[(g,k)].append(v/n)
    return b

def draw(ax,b,ylabel,xlabel,logx=False):
    if not b:
        ax.text(.5,.5,"No eligible measurements",ha="center",va="center",transform=ax.transAxes)
        ax.set_axis_off();return False
    for g in sorted({k[0] for k in b}):
        xs=sorted(k[1] for k in b if k[0]==g);ys=[b[(g,x)] for x in xs]
        m=[np.mean(v) for v in ys];lo=[min(v) for v in ys];hi=[max(v) for v in ys]
        ax.plot(xs,m,marker=".",label=label(g));ax.fill_between(xs,lo,hi,alpha=.10)
    if logx and all(k[1]>0 for k in b):ax.set_xscale("log",base=2)
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2)
    return True

def render(indir,outdir):
    p=Path(indir);out=Path(outdir)
    m=json.loads((p/"manifest.json").read_text())
    if m["schema"]!="mqft.analysis.v2" or not m["completed"]:raise ValueError("no valid analysis")
    if any(sha256(p/n)!=h for n,h in m["outputs"].items()):raise ValueError("analysis hash mismatch")
    out.mkdir(parents=True,exist_ok=False)
    registry=[]
    def panel(name,items,title):
        fig,axs=plt.subplots(len(items),1,figsize=(10,3.1*len(items)),squeeze=False,constrained_layout=True)
        eligible=[]
        for ax,(stream,field,x,yl,xl,logx,fn,pred) in zip(axs[:,0],items):
            d=summarize(rows(p/(stream+".jsonl")),field,x,fn,pred)
            eligible.append(draw(ax,d,yl,xl,logx))
            if d:
                ax.legend(fontsize=6,bbox_to_anchor=(1.01,1),loc="upper left",frameon=False)
        fig.suptitle(title+f" | {m['completed']}/{m['expected']} jobs; bands = job min–max",fontsize=11)
        if any(eligible):
            for ext in ("png","pdf"):fig.savefig(out/(name+"."+ext),dpi=160,bbox_inches="tight")
            registry.append(dict(name=name,status="observed_partial" if not all(eligible) else "observed",eligible=eligible))
        else:registry.append(dict(name=name,status="BLOCKED_NO_ELIGIBLE_DATA",eligible=eligible))
        plt.close(fig)
    allr=lambda r:True
    last=lambda r:r["k"]==r["budget"]
    it=lambda s,f,x,y,xx,l=False,t=None,pr=allr:(s,f,x,y,xx,l,t,pr)
    panel("T_temperature_entropy",[
        it("local","beta","global_k","Fitted inverse score scale","Completed global backups",True),
        it("local","beta_MF","global_k","Mean-field comparator sqrt(k_local)/c","Completed global backups",True),
        it("local","fit_KL","global_k","KL residual (nats)","Completed global backups",True),
        it("global","H_fraction","k","Occupancy H / attainable ceiling","Completed global backups",True),
        it("global","q","ply","E = -mean Q (reward units)","Game ply",t=lambda r:-r["q"],pr=lambda r:last(r) and r["kind"]=="games")],
        "T: Thermometry and actual occupancy (finite fits only; boundary states in local.jsonl)")
    panel("A_action_curvature",[
        it("local","kkt_gap","global_k","Frozen objective gap","Completed global backups",True),
        it("local","tangent_logdet","global_k","Tangent log det (frozen active set)","Completed global backups",True),
        it("local","tangent_inverse_trace","global_k","Inverse tangent curvature trace","Completed global backups",True),
        it("local","variance","global_k","Actual backup-mark variance","Completed global backups",True)],"A: Finite action and curvature, not a calibrated one-loop law")
    panel("R_budget_and_coarse_flow",[
        it("global","q","k","Root-frame Q","Completed global backups",True),
        it("global","allocated_nodes","k","Allocated native nodes","Completed global backups",True),
        it("deltas","dQ","global_k","Adjacent-cut dQ","Completed global backups",True),
        it("deltas","TV","global_k","Adjacent-cut total variation","Completed global backups",True)],"R: Same-root budget flow; not an autonomous RG beta function")
    panel("D_decision_records",[
        it("global","effective_actions","k","Effective actions","Completed global backups",True),
        it("local","conditional_H","global_k","Path H conditional on first action (nats)","Completed global backups",True),
        it("deltas","plurality_overlap","global_k","Plurality-set overlap","Completed global backups",True)],"D: Classical decision records, not quantum coherence")
    panel("N_score_balance",[
        it("deltas","dE","global_k","Change in -score expectation","Completed global backups",True),
        it("deltas","field","global_k","Score-field contribution","Completed global backups",True),
        it("deltas","redistribution","global_k","Redistribution contribution","Completed global backups",True)],"N: Exact discrete product rule in score units, not heat or work")
    panel("C_finite_budget_response",[
        it("global","order_parameter","k","Occupancy order parameter pi_(1)-pi_(2)","Completed global backups",True),
        it("global","effective_actions","k","Effective actions","Completed global backups",True),
        it("local","mean_depth","global_k","Mean path depth","Completed global backups",True),
        it("global","variance","k","Actual mark variance","Completed global backups",True)],"C: Finite search crossover; no thermodynamic-limit extrapolation")
    # Graph distances and coarse-graining: within replica, normalize over independent jobs.
    def expand(field):
        for r in rows(p/"local.jsonl"):
            if r["global_k"]!=r["budget"]:continue
            for z in r[field]:yield {**r,**z}
    for name,field,x,y,yl in [("R_coarse_graining","coarse","width","unresolved_fraction","Unresolved variance fraction"),("C_graph_correlation","graph","distance","cov","Connected Q covariance")]:
        d=summarize(expand(field),y,x)
        fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
        if draw(ax,d,yl,x):
            ax.legend(fontsize=6,bbox_to_anchor=(1.01,1),loc="upper left")
            fig.savefig(out/(name+".png"),dpi=160,bbox_inches="tight");fig.savefig(out/(name+".pdf"),bbox_inches="tight");registry.append(dict(name=name,status="observed"))
        else:registry.append(dict(name=name,status="BLOCKED_NO_ELIGIBLE_DATA"))
        plt.close(fig)
    # Paired source and record groups, separate from longitudinal game plots.
    if "source_response.jsonl" in m["outputs"]:
        panel("A_C_paired_source_response",[
            it("source_response","measured","h","Measured d occupancy / dh","Source amplitude"),
            it("source_response","frozen_kkt","h","Frozen KKT finite response","Source amplitude"),
            it("source_response","residual","h","Measured - frozen response","Source amplitude",t=lambda r:r["measured"]-r["frozen_kkt"])],"A/C: Matched root source contrast, not FDT")
    laws=list(rows(p/"law_summary.jsonl"))
    if laws:
        fig,axs=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
        for j,r in enumerate(laws):
            for k,target in enumerate(("p","q")):
                x=j+(-.13 if k==0 else .13);z=r["targets"][target]
                for ax,field in zip(axs,("weight_mean","ESS","response_IS")):
                    if z[field] is not None:ax.scatter(x,z[field],label=target if j==0 else None,marker="o" if k==0 else "x")
        axs[0].axhline(1,color="grey",ls=":");axs[0].set_ylabel("Mean path weight")
        axs[1].set_ylabel("Path ESS (count)");axs[2].set_ylabel("IS score response")
        for ax in axs:ax.set_xticks(range(len(laws)),[f"{r['position_id']} B{r['budget']} {r['law_id']} n={r['n_paths']}" for r in laws],rotation=90,fontsize=6);ax.grid(alpha=.2)
        axs[0].legend();fig.suptitle("N/A: bounded path proposal, fixed-root replicates; not physical work")
        for ext in ("png","pdf"):fig.savefig(out/("N_A_path_law."+ext),dpi=160,bbox_inches="tight")
        plt.close(fig);registry.append(dict(name="N_A_path_law",status="observed"))
    else:registry.append(dict(name="N_A_path_law",status="BLOCKED_NO_PATH_PROPOSAL_DATA"))
    rec=list(rows(p/"root_repeats.jsonl"))
    if rec:
        fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
        for r in rec:
            z=r["metrics"];xs=[v["fragment_count"] for v in z]
            ax.plot(xs,[v["I"] for v in z],marker="o",alpha=.65,label=f"{r['mode']} {r['position_id']} B{r['budget']} n={r['n']}")
            ax.plot(xs,[v["null_mean"] for v in z],linestyle=":",color="grey",alpha=.5)
            ax.plot(xs,[v["disjoint_I"] for v in z],linestyle="--",alpha=.65)
        ax.set(xlabel="Cumulative disjoint record fragments",ylabel="Empirical MI (nats)",title="D: Classical cumulative MI; disjoint (dashed); permutation null (dotted)")
        if len(rec)<=20:ax.legend(fontsize=6,bbox_to_anchor=(1.01,1),loc="upper left")
        fig.savefig(out/"D_repeated_root_information.png",dpi=160,bbox_inches="tight");fig.savefig(out/"D_repeated_root_information.pdf",bbox_inches="tight");plt.close(fig)
        registry.append(dict(name="D_repeated_root_information",status="observed"))
    else:registry.append(dict(name="D_repeated_root_information",status="BLOCKED_NO_REPEATED_ROOTS"))
    manifest=dict(schema="mqft.plots.v2",source_manifest_sha256=sha256(p/"manifest.json"),panels=registry,
                  files={f.name:sha256(f) for f in sorted(out.iterdir()) if f.is_file()},
                  blocked=["physical temperature/heat/FDT/Crooks/Jarzynski without calibrated protocol","native quantum phase, rho, SBS, envariance, Lindblad or tunneling without declared map and controls","physical RG fixed points/exponents without scale-family and holdout","actual one-loop coefficient without measure and predictive test"])
    atomic_json(out/"manifest.json",manifest);return manifest
