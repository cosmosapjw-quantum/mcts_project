"""Eight compact theme figures; fixed DAG, original target ownership retained."""
import json,hashlib,statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text())
out=S/"plots/grouped";out.mkdir(parents=True,exist_ok=False)
laws=["iid","pl","reserve"];ix=np.arange(3);reg={}
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ("png","pdf"):fig.savefig(out/(node+"."+e),dpi=170,bbox_inches="tight")
 plt.close(fig);reg[node]=dict(scope=scope,files={node+"."+e:sh(out/(node+"."+e)) for e in ("png","pdf")})
def grid(ax):ax.grid(alpha=.2)
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
a=[x for x in r["native"] if x["h"]==0]
for l in laws:
 v=np.sort(np.concatenate([x["uniforms"] for x in a if x["law"]==l]));axs[0].plot(v,np.arange(1,len(v)+1)/len(v),label=l)
axs[0].plot([0,1],[0,1],color="grey",ls=":");axs[0].set(xlabel="Recorded PRNG draw",ylabel="Empirical CDF");axs[0].legend();grid(axs[0])
axs[1].bar(ix,[r["score_diagnostics"][l]["score_over_sqrt_F"] for l in laws]);axs[1].axhline(0,color="grey",ls=":");axs[1].set_xticks(ix,laws);axs[1].set_ylabel("Total score / sqrt(predictable Fisher)")
fig.suptitle("Native RNG replay and descriptive score diagnostics, h=0")
save("REPLAN-01",fig,"native_replayed")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for l in laws:
 a=[x for x in r["native"] if x["law"]==l and x["h"]==0];axs[0].hist([-x["log_root_prob"] for x in a],bins=20,histtype="step",density=True,label=l)
axs[0].set(xlabel="Root path action -log P (nats)",ylabel="Empirical density");axs[0].legend()
for l in laws:
 a=[x for x in r["native"] if x["law"]==l and x["h"]==0]
 vals=np.array([x["full_wave_counts"] for x in a]);v=np.cov(vals.T);axs[1].plot([0,1,2],[np.mean(np.diag(v,k)) for k in (0,1,2)],marker="o",label=l)
axs[1].set(xlabel="Complete wave lag",ylabel="Pooled target-count covariance");axs[1].legend();grid(axs[1]);fig.suptitle("A: actual path action and descriptive temporal covariance (chosen roots)")
save("REPLAN-03",fig,"native_observed")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for x in r["memory"]:
 axs[0].plot(range(1,len(x["kernel_norms"])+1),x["kernel_norms"],marker="o",label=x["kind"])
 axs[1].plot(range(len(x["markov_defect"])),x["markov_defect"],marker="o",label=x["kind"])
axs[0].set(xlabel="Memory lag",ylabel="Projected memory kernel norm");axs[1].set(xlabel="Declared continuation steps",ylabel="Exact projected vs Markov norm")
for ax in axs:ax.legend();grid(ax)
fig.suptitle("R: fitted finite pair-chain, exact discrete projection recurrence")
save("REPLAN-04.R",fig,"fitted_conditional_reference")
fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for ax,A in zip(axs[0],("occupancy","mark")):
 for k,lab,dx in (("raw","Score",-.3),("centered","Centered derivative",-.1),("centered_fd","Centered finite LR",.1),("direct","Direct finite h",.3)):
  a=[r["metrics"][l][A][k] for l in laws];ax.errorbar(ix+dx,[x["mean"] for x in a],yerr=[x["se"] for x in a],fmt="o",capsize=3,label=lab)
 ax.set_xticks(ix,laws);ax.axhline(0,color="grey",ls=":");ax.set_ylabel(A+" response");ax.legend(fontsize=7);grid(ax)
for ax,key,y,ref in [(axs[1,0],"G","Mean path score",0),(axs[1,1],"Lp","Mean likelihood ratio, h=+.3",1)]:
 a=[r["metrics"][l][key] for l in laws];ax.errorbar(ix,[x["mean"] for x in a],yerr=[x["se"] for x in a],fmt="o",capsize=4);ax.set_xticks(ix,laws);ax.axhline(ref,color="grey",ls=":");ax.set_ylabel(y);grid(ax)
fig.suptitle("C: 8 fixed roots, 16 histories/arm; descriptive root-level SE")
save("REPLAN-04.C",fig,"native_replication")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
cb=r["causal_break"]
for l in laws:
 a=[x for x in cb if x["law"]==l];x=[v["static_MI"] for v in a];axs[0].hist(x,bins=np.linspace(0,.7,15),histtype="step",label=l)
axs[0].set(xlabel="Conditional past-future MI (nats)",ylabel="Actual drive histories");axs[0].legend()
for noise,lab,dx in (("static_MI","Static",-.16),("iid_MI","Independent",.16)):
 axs[1].bar(ix+dx,[st.mean(x[noise] for x in cb if x["law"]==l) for l in laws],width=.3,label=lab)
axs[1].set_xticks(ix,laws);axs[1].set_ylabel("Mean conditional MI");axs[1].legend();fig.suptitle("D: declared qubit records after causal break; same fixed drive")
save("REPLAN-05",fig,"conditional_CP_records")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
pairs=[(-.3,.3),(.3,-.3),(0.,.3),(.3,0.)];xx=np.arange(len(pairs))
for i,l in enumerate(laws):
 a=[[x for x in r["actual_protocol"] if x["law"]==l and x["source"]==h and x["target"]==h1] for h,h1 in pairs]
 axs[0].plot(xx,[st.mean(x["sigma"]["mean"] for x in v) for v in a],marker="o",label=l)
 axs[0].plot(xx,[st.mean(x["conditional_KL"]["mean"] for x in v) for v in a],marker="x",ls=":",color=axs[0].lines[-1].get_color())
 axs[1].plot(xx,[st.mean(x["weight"]["mean"] for x in v) for v in a],marker="o",label=l)
for ax,y in zip(axs,("Log path ratio (solid), KL compensator (dotted)","Mean exp(-path ratio)")):
 ax.set_xticks(xx,[f"{h:g}→{h1:g}" for h,h1 in pairs]);ax.set_ylabel(y);ax.legend();grid(ax)
axs[1].axhline(1,color="grey",ls=":");fig.suptitle("N: actual source-protocol ratios, not physical time reversal or heat")
save("REPLAN-06",fig,"native_protocol_ratio")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for i,l in enumerate(laws):
 a=[x for x in cb if x["law"]==l];axs[0].scatter(np.full(len(a),i)+np.linspace(-.15,.15,len(a)),[x["correlation"] for x in a],s=9,alpha=.5)
axs[0].set_xticks(ix,laws);axs[0].set_ylabel("Conditional Y₁Y₂ correlation");axs[0].axhline(0,color="grey",ls=":")
axs[1].bar(ix,[st.mean(x["future_Y_plus_gap"] for x in cb if x["law"]==l) for l in laws]);axs[1].set_xticks(ix,laws);axs[1].set_ylabel("Future Y+ probability gap after equal reset")
fig.suptitle("Q: explicit measurement/repreparation causal-break intervention")
save("X-REP",fig,"conditional_CP_intervention")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for k,lab,dx in (("markov","First-order",-.16),("memory","Pair history",.16)):
 axs[0].bar(ix+dx,[next(x[k] for x in r["memory"] if x["kind"]==l) for l in laws],width=.3,label=lab)
axs[0].set_xticks(ix,laws);axs[0].set_ylabel("Held-out final wave count NLL");axs[0].legend()
a=r["closure"]["models"];ks=list(a);axs[1].bar(range(len(ks)),[a[k]["nll"] for k in ks]);axs[1].set_xticks(range(len(ks)),["State","History","Geometry","Both"],rotation=12);axs[1].set_ylabel("Held-out return-sign NLL")
fig.suptitle("Memory: held-out four roots, not universal Markov rejection")
save("X-MEMORY",fig,"native_heldout")
(out/"manifest.json").write_text(json.dumps(dict(source_sha256=sh(S/"PHYSICS_RESULTS.json"),groups=reg),indent=2)+"\n")
print("grouped figures",len(reg))
