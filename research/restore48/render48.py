"""Nine grouped figures; uncertainty at fixed roots, no new DAG nodes."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text());c=json.loads((S/"CONDITIONAL_RESULTS.json").read_text());sz=json.loads((S/"SIZE_SUPPORT.json").read_text())
out=S/"plots/grouped";out.mkdir(exist_ok=False);reg={}
laws=["iid","pl","reserve"];ix=np.arange(3)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ("png","pdf"):fig.savefig(out/(node+"."+e),dpi=160,bbox_inches="tight")
 plt.close(fig);reg[node]=dict(scope=scope,files={node+"."+e:h(out/(node+"."+e)) for e in ("png","pdf")})
def grid(ax):ax.grid(alpha=.2)
def mean(a,k):return st.mean(x[k] for x in a)
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
ns=list(r["counts"]);axs[0].bar(ns,[r["counts"][n]["jobs"] for n in ns]);axs[0].set_ylabel("New verified native histories")
v=sz["actual_probes"];axs[1].plot([x["L"] for x in v],[x.get("legal_scores",0) for x in v],marker="o",label="actual NNUE policy support")
axs[1].set(xlabel="Board side L",ylabel="Legal policy scores");axs[1].legend(fontsize=8);grid(axs[1])
fig.suptitle("Native scale replication and direct evaluator size support (not multi-size MCTS)");save("REPLAN-01",fig,"native_and_direct_evaluator")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for B in (32,64):
 for law in laws:
  a=[x for x in r["temperature"] if x["B"]==B and x["law"]==law];ks=sorted({x["k"] for x in a})
  axs[0].plot(ks,[mean([x for x in a if x["k"]==k],"H") for k in ks],marker=".",label=f"{law} B{B}")
  axs[1].plot(ks,[st.mean(x["status"].get("finite",0)/16 for x in a if x["k"]==k) for k in ks],marker=".",label=f"{law} B{B}")
for ax,y in zip(axs,("Occupancy entropy (nats)","Finite inverse-score fit fraction")):
 ax.set(xscale="log",xlabel="Completed backups",ylabel=y);ax.legend(fontsize=7);grid(ax)
fig.suptitle("T: actual native entropy and score-scale fit, no physical beta");save("REPLAN-02",fig,"native_observed")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
a=r["joint_action"]
for B in (32,64):
 z=[x for x in a if x["B"]==B];axs[0].scatter([x["eigenvalues"][0] for x in z],[x["eigenvalues"][1] for x in z],label=f"B{B}")
axs[0].set(xscale="symlog",yscale="log",xlabel="Small covariance eigenvalue",ylabel="Large eigenvalue");axs[0].legend();grid(axs[0])
for k,lab,dx in (("zero_error","Zero shift",-.25),("quadratic_error","Quadratic",0),("empirical_error","Empirical CGF",.25)):
 axs[1].bar(np.arange(2)+dx,[mean([q for x in a if x["B"]==B for q in x["predictions"]],k) for B in (32,64)],width=.23,label=lab)
axs[1].set_xticks(range(2),["B32","B64"]);axs[1].set_ylabel("Held-out output tilt mean L2 error");axs[1].legend(fontsize=8)
fig.suptitle("A: empirical Legendre action, fixed 8+8 history split");save("REPLAN-03",fig,"empirical_measure")
fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for i,k in enumerate((0,1)):
 a=[x for x in r["causal_kernels"] if x["B"]==64];mat=[np.mean([x["kernel"][k] for x in a if x["law"]==l],0) for l in laws];lim=max(abs(x).max() for x in mat)
 for j,l in enumerate(laws):
  im=axs[i,j].imshow(mat[j],origin="lower",cmap="RdBu_r",vmin=-lim,vmax=lim)
  axs[i,j].set(xlabel="Source block",ylabel="Observable block",title=l+" "+("count" if k==0 else "mark"))
 fig.colorbar(im,ax=axs[i].tolist(),shrink=.75)
fig.suptitle("R: actual time-block score response; upper triangle (future source) is finite-sample null");save("REPLAN-04.R",fig,"native_causal_score")
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True);cs=[(16,.3),(32,.1),(32,.2),(64,.2)]
for j,ax in enumerate(axs[0]):
 for l in laws:
  for k,ls in (("LR",":"),("direct","-")):
   a=[next(x for x in r["summary"] if x["law"]==l and (x["B"],x["h"])==t)["metrics"][k] for t in cs]
   ax.errorbar(np.arange(4),[x["mean"][j] for x in a],yerr=[x["fixed_root_se"][j] for x in a],fmt="o",ls=ls,capsize=2,label=l+" "+k)
 ax.set_xticks(range(4),["16/.3","32/.1","32/.2","64/.2"]);ax.set_ylabel(("Count" if j==0 else "Mark")+" first finite contrast");ax.legend(fontsize=7);grid(ax)
for j,ax in enumerate(axs[1]):
 for l in laws:
  a=[next(x for x in r["summary"] if x["law"]==l and (x["B"],x["h"])==t)["metrics"]["LR2_minus_direct2"] for t in cs]
  ax.errorbar(range(4),[x["mean"][j] for x in a],yerr=[x["fixed_root_se"][j] for x in a],fmt="o-",capsize=3,label=l)
 ax.axhline(0,color="grey",ls=":");ax.set_xticks(range(4),["16/.3","32/.1","32/.2","64/.2"]);ax.set_ylabel(("Count" if j==0 else "Mark")+" second finite LR − direct");ax.legend(fontsize=8);grid(ax)
fig.suptitle("C: 16 complete paired histories per arm; conditional fixed-root jackknife SE");save("REPLAN-04.C",fig,"native_scale_precision")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);q=c["tomography"]
for k,lab in (("observed","Persistent vs reset"),("same_channel_null","Same-channel null")):
 v=[[x["contrast"][k] for x in q if "contrast" in x and x["shots"]==n] for n in (128,512,2048)]
 axs[0].plot([128,512,2048],[st.mean(a) for a in v],marker="o",label=lab)
axs[0].set(xscale="log",xlabel="Shots / setting",ylabel="Reconstructed Choi distance");axs[0].legend(fontsize=8);grid(axs[0])
for k,lab in (("raw_error","Raw"),("repaired_error","CP repair")):
 axs[1].plot([128,512,2048],[mean([x for x in q if x["shots"]==n],k) for n in (128,512,2048)],marker="o",label=lab)
axs[1].set(xscale="log",yscale="log",xlabel="Shots / setting",ylabel="Trace distance to exact channel");axs[1].legend();grid(axs[1])
fig.suptitle("D: finite-shot conditional memory discrimination; simulated measurements");save("REPLAN-05",fig,"conditional_shot_simulation")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);a=c["thermal"];ds=(1.,4.,16.,64.)
for k,lab in (("work","Work"),("heat","Heat into system"),("delta_U","Delta U")):
 axs[0].plot(ds,[mean([x for x in a if x["duration"]==T and x["subdiv"]==128],k) for T in ds],marker="o",label=lab)
for k,lab in (("sigma","Entropy production"),("lag","Final lag KL")):
 axs[1].plot(ds,[mean([x for x in a if x["duration"]==T and x["subdiv"]==128],k) for T in ds],marker="o",label=lab)
for ax,y in zip(axs,("Energy in E* units","Dimensionless entropy")):ax.set(xscale="log",xlabel="Dimensionless duration",ylabel=y);ax.legend();grid(ax)
fig.suptitle("N: conditional thermal cycles, fixed actual propensity knots");save("REPLAN-06",fig,"conditional_thermal")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for i,kind in enumerate(("static","reset_static","dephased")):
 a=[[x for x in q if x["shots"]==n and x["kind"]==kind] for n in (128,512,2048)]
 axs[0].plot([128,512,2048],[sum(x["min_eig"]<-1e-10 for x in z)/len(z) for z in a],marker="o",label=kind)
 axs[1].plot([128,512,2048],[mean(z,"repair") for z in a],marker="o",label=kind)
for ax,y in zip(axs,("Non-PSD raw fraction","Depolarizing repair weight")):ax.set(xscale="log",xlabel="Shots / setting",ylabel=y);ax.legend();grid(ax)
fig.suptitle("Q: finite-shot Choi reconstruction and CP repair");save("X-REP",fig,"conditional_tomography")
fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True);mods=["state","clock","memory","clock_memory"]
for j in (0,1):
 for i,m in enumerate(mods):axs[j].bar(ix+(i-1.5)*.18,[next(x for x in r["memory"] if x["law"]==l)["models"][m]["mse"][j] for l in laws],width=.17,label=m)
 axs[j].set_xticks(ix,laws);axs[j].set_ylabel(("Count" if j==0 else "Mark")+" MSE")
for i,m in enumerate(mods):axs[2].bar(ix+(i-1.5)*.18,[next(x for x in r["categorical_memory"] if x["law"]==l)["nll"][i] for l in laws],width=.17,label=m)
axs[2].set_xticks(ix,laws);axs[2].set_ylabel("Categorical NLL");axs[2].legend(fontsize=7)
fig.suptitle("Memory: larger training, repeated R47 holdout (NOT new test roots)");save("X-MEMORY",fig,"repeated_heldout")
(out/"manifest.json").write_text(json.dumps(dict(source_sha256=h(S/"PHYSICS_RESULTS.json"),conditional_sha=h(S/"CONDITIONAL_RESULTS.json"),size_sha=h(S/"SIZE_SUPPORT.json"),groups=reg),indent=2)+"\n")
print("grouped figures",len(reg))
