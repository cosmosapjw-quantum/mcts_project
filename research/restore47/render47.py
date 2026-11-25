"""Nine compact grouped outputs. Actual data vs conditional models labelled."""
from pathlib import Path
import json,hashlib,statistics as st
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text())
out=S/"plots/grouped";out.mkdir(exist_ok=False)
laws=["iid","pl","reserve"];ix=np.arange(3);reg={}
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ("png","pdf"):fig.savefig(out/(node+"."+e),dpi=160,bbox_inches="tight")
 plt.close(fig);reg[node]=dict(scope=scope,files={node+"."+e:h(out/(node+"."+e)) for e in ("png","pdf")})
def grid(ax):ax.grid(alpha=.2)
def mean(a,k):return st.mean(x[k] for x in a)
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
axs[0].bar([x["position"] for x in r["games"]],[x["plies"] for x in r["games"]]);axs[0].set_ylabel("Natural-terminal plies")
names=list(r["counts"]);axs[1].bar(names,[r["counts"][x]["backups"] for x in names]);axs[1].tick_params(axis="x",rotation=20);axs[1].set_ylabel("Actual native backups")
fig.suptitle("Native common MixNet data: new D4 blocks and scale pilot");save("REPLAN-01",fig,"native")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
a=r["temperature"]
for name in ("scale32","scale64","heldout64"):
 v=[x for x in a if x["campaign"]==name];ks=sorted({x["k"] for x in v})
 axs[0].plot(ks,[mean([x for x in v if x["k"]==k],"H") for k in ks],marker="o",label=name)
 axs[1].plot(ks,[sum(x["beta_status"]=="finite" for x in v if x["k"]==k)/sum(x["k"]==k for x in v) for k in ks],marker="o",label=name)
for ax,y in zip(axs,("Occupancy entropy (nats)","Finite inverse-score fit fraction")):
 ax.set(xscale="log",xlabel="Completed backups",ylabel=y);ax.legend(fontsize=8);grid(ax)
fig.suptitle("T: native entropy and score-scale status, not physical thermometry");save("REPLAN-02",fig,"native_observed")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
a=r["joint_action"]
for i,l in enumerate(laws):
 q=[x for x in a if x["law"]==l];axs[0].scatter([x["eigenvalues"][0] for x in q],[x["eigenvalues"][1] for x in q],label=l)
axs[0].set(xscale="log",yscale="log",xlabel="Small covariance eigenvalue",ylabel="Large covariance eigenvalue");axs[0].legend();grid(axs[0])
for k,lab,dx in (("zero_error","Zero shift",-.25),("quadratic_error","Quadratic",0),("empirical_error","Empirical CGF",.25)):
 axs[1].bar(ix+dx,[mean([z for x in a if x["law"]==l for z in x["predictions"]],k) for l in laws],width=.23,label=lab)
axs[1].set_xticks(ix,laws);axs[1].set_ylabel("Held-out tilted mean L2 error");axs[1].legend(fontsize=8)
fig.suptitle("A: empirical joint count/mark action; 8+8 seeds per old root");save("REPLAN-03",fig,"empirical_measure_holdout")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for x in r["categorical_clock_memory"]:
 q=x["projection"];axs[0].plot(range(len(q["markov_defect"])),q["markov_defect"],marker="o",label=x["law"])
 axs[1].plot(range(2,9),[max(a) for a in q["kernel_norms"][1:]],marker="o",label=x["law"])
for ax,y in zip(axs,("Projected vs memory-free norm","Max nonstationary kernel norm")):
 ax.set(xlabel="Declared model clock",ylabel=y);ax.legend();grid(ax)
fig.suptitle("R: exact nonstationary projection of fitted four-category pair chain");save("REPLAN-04.R",fig,"fitted_conditional_reference")
fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True);cs=[(16,.3),(32,.1),(32,.2),(64,.2)]
for i,ax in enumerate(axs[0]):
 for l in laws:
  for key,ls in (("finite_lr",":"),("direct","-")):
   q=[np.mean([x[key][i] for x in r["scale_response"] if x["law"]==l and x["budget"]==B and x["h"]==hh]) for B,hh in cs]
   ax.plot(range(4),q,marker="o",ls=ls,label=l+" "+key)
 ax.set_xticks(range(4),["16/.3","32/.1","32/.2","64/.2"]);ax.set_ylabel(("Occupancy" if i==0 else "Mean mark")+" response");ax.legend(fontsize=7);grid(ax)
for l in laws:
 a=[x for x in r["scale_response"] if x["law"]==l]
 axs[1,0].plot(range(4),[st.mean(x["F_mean"]/B for x in a if x["budget"]==B and x["h"]==hh) for B,hh in cs],marker="o",label=l)
 axs[1,1].plot(range(4),[st.mean(x["ESSplus"]/x["n"] for x in a if x["budget"]==B and x["h"]==hh) for B,hh in cs],marker="o",label=l)
for ax,y in zip(axs[1],("Fisher / budget","Positive-arm weight ESS / n")):ax.set_xticks(range(4),["16/.3","32/.1","32/.2","64/.2"]);ax.set_ylabel(y);ax.legend();grid(ax)
fig.suptitle("C: scale/source pilot; 16 repeats at B16, TWO at B32/64; no scaling fit");save("REPLAN-04.C",fig,"native_small_sample")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for k,lab,dx in (("static","Persistent",-.24),("reset_static","Block reset",0),("dephased","Dephased",.24)):
 axs[0].bar(ix+dx,[st.mean(x["channels"][k]["phase_QFI"] for x in r["probes"] if x["law"]==l) for l in laws],width=.23,label=lab)
axs[0].set_xticks(ix,laws);axs[0].set_ylabel("Declared phase QFI");axs[0].legend(fontsize=8)
for i,l in enumerate(laws):
 q=[x for x in r["probes"] if x["law"]==l];axs[1].scatter(np.full(len(q),i)+np.linspace(-.12,.12,len(q)),[x["choi_static_reset"] for x in q],s=14)
axs[1].set_xticks(ix,laws);axs[1].set_ylabel("Inter-block memory Choi distance")
fig.suptitle("D: coherent phase sensitivity and identical-block-marginal control");save("REPLAN-05",fig,"conditional_phase_probe")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
a=r["thermal_cycles"];es=(.25,.5,1.)
for k,lab in (("work","Work"),("heat","Heat into system"),("delta_U","Delta U")):axs[0].plot(es,[mean([x for x in a if x["eta"]==e],k) for e in es],marker="o",label=lab)
for k,lab in (("entropy_production","Sigma"),("final_lag_KL","Final lag KL"),("work","Beta W")):axs[1].plot(es,[mean([x for x in a if x["eta"]==e],k) for e in es],marker="o",label=lab)
for ax,y in zip(axs,("Energy in declared E* units","Dimensionless entropy")):ax.set(xlabel="Gibbs reset fraction",ylabel=y);ax.legend();grid(ax)
fig.suptitle("N: exact cyclic two-level protocol from actual propensity schedule");save("REPLAN-06",fig,"conditional_thermal_cycle")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for k,lab,dx in (("choi_static_iid","Independent per hit",-.25),("choi_static_reset","Independent per block",0),("choi_static_dephased","Dephased",.25)):
 axs[0].bar(ix+dx,[mean([x for x in r["probes"] if x["law"]==l],k) for l in laws],width=.23,label=lab)
axs[0].set_xticks(ix,laws);axs[0].set_ylabel("Choi trace distance to persistent");axs[0].legend(fontsize=8)
for kind in ("iid","static","reset_static","dephased"):
 a=[x["channels"][kind]["phase_QFI"] for x in r["probes"]];axs[1].hist(a,bins=np.linspace(0,1.01,15),histtype="step",label=kind)
axs[1].set(xlabel="Phase QFI",ylabel="Actual drive histories");axs[1].legend(fontsize=8)
fig.suptitle("Q: six-preparation exact process reconstruction, declared driven qubit");save("X-REP",fig,"conditional_CP_tomography")
fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True);mods=["state","clock","memory","clock_memory"]
for j,k in enumerate((0,1)):
 for i,m in enumerate(mods):
  axs[j].bar(ix+(i-1.5)*.18,[next(x for x in r["clock_memory"] if x["law"]==l)["models"][m]["mse"][k] for l in laws],width=.17,label=m)
 axs[j].set_xticks(ix,laws);axs[j].set_ylabel(("Count fraction" if k==0 else "Return mark")+" MSE")
for i,m in enumerate(mods):axs[2].bar(ix+(i-1.5)*.18,[next(x for x in r["categorical_clock_memory"] if x["law"]==l)["nll"][i] for l in laws],width=.17,label=m)
axs[2].set_xticks(ix,laws);axs[2].set_ylabel("Categorical NLL");axs[2].legend(fontsize=7)
fig.suptitle("Memory: train old roots, evaluate four genuinely new D4 roots, B64");save("X-MEMORY",fig,"native_heldout")
(out/"manifest.json").write_text(json.dumps(dict(source_sha256=h(S/"PHYSICS_RESULTS.json"),choi_sha256=h(S/"CHOI_STATES.npz"),groups=reg),indent=2)+"\n")
print("grouped figures",len(reg))
