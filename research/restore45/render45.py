"""One new multi-panel figure per executed theme. No new DAG nodes."""
import json,statistics as st,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text());records=json.loads((S/"RECORD_INFORMATION.json").read_text())["data"]
out=S/"plots/grouped";out.mkdir(parents=True,exist_ok=False)
laws=["iid","pl","reserve"];ix=np.arange(3);registry={}
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ("png","pdf"):fig.savefig(out/(node+"."+e),dpi=170,bbox_inches="tight")
 plt.close(fig);registry[node]=dict(scope=scope,files={node+"."+e:h(out/(node+"."+e)) for e in ("png","pdf")})
def grid(ax):ax.grid(alpha=.2)
def field(rr,k):return [st.mean(x[k] for x in rr if x["law"]==l) for l in laws]
# native and score scales
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
axs[0].bar([x["position"] for x in r["games"]],[x["plies"] for x in r["games"]]);axs[0].set_ylabel("Natural-terminal game length (plies)")
a=[x for x in r["native"] if x["h"]==0];axs[1].bar(ix,[sum(x["collision"] for x in a if x["law"]==l)/sum(x["pairs"] for x in a if x["law"]==l) for l in laws]);axs[1].set_xticks(ix,laws);axs[1].set_ylabel("Native collision fraction")
fig.suptitle("REPLAN-01: new common MixNet cohort, B16")
save("REPLAN-01",fig,"native_observed")
fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
for k,lab,dx in (("test_nll_old_fit","Previous fit",-.25),("test_nll_input","Input beta",0),("test_nll_uniform","Uniform",.25)):
 ax.bar(ix+dx,[r["score_scale"][l][k] for l in laws],width=.23,label=lab)
ax.set_xticks(ix,laws);ax.set_ylabel("Held-out NLL (nats/draw)");ax.legend();ax.set_title("T: independent-root conditional score scale, not physical temperature")
save("REPLAN-02",fig,"native_heldout")
# action
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,d in zip(axs,(3,5)):
 ms=sorted({x["M"] for x in r["multivariate"] if x["d"]==d})
 for key,lab in (("leading_error","Leading rate"),("one_loop_error","Lattice determinant")):
  v=[[x[key] for x in r["multivariate"] if x["d"]==d and x["M"]==m] for m in ms]
  ax.plot(ms,[np.median(a) for a in v],marker="o",label=lab);ax.fill_between(ms,[min(a) for a in v],[max(a) for a in v],alpha=.16)
 ax.set(xscale="log",yscale="log",xlabel="Independent frozen waves M",ylabel="Absolute log-mass error",title=f"{d-1}D count coordinates");ax.legend(fontsize=8);grid(ax)
fig.suptitle("A: exact positive-coefficient DP vs multivariate lattice saddle")
save("REPLAN-03",fig,"exact_restricted_frozen_wave")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for l in laws:
 a=[x for x in r["coarse"] if x["law"]==l];m=["ordered","prefix1","unordered","target_count","constant"]
 axs[0].plot(range(len(m)),[st.mean(x["maps"][n]["relative"] for x in a) for n in m],marker="o",label=l)
axs[0].set_xticks(range(len(m)),m,rotation=25);axs[0].set_ylabel("Retained Fisher trace fraction");axs[0].legend();grid(axs[0])
for a in r["graph"]:
 x=a["depths"];axs[1].plot([z["coarse_dim"]/z["nodes"] for z in x],[z["memory_trace"] for z in x],marker=".",alpha=.65)
axs[1].set(xlabel="Retained prefix dimension / nodes",ylabel="Projected semigroup defect trace");grid(axs[1])
fig.suptitle("R: count information and declared native-prefix graph reduction")
save("REPLAN-04.R",fig,"exact_restricted_and_declared_native_graph")
fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for ax,c in zip(axs[0],("fresh","both")):
 for key,lab,dx in (("raw","Score",-.25),("centered","Centered score",0),("direct","Direct finite h",.25)):
  a=[r["metrics"][f"{c}/{l}/occupancy"][key] for l in laws];ax.errorbar(ix+dx,[x["mean"] for x in a],yerr=[x["se"] for x in a],fmt="o",capsize=3,label=lab)
 ax.set_xticks(ix,laws);ax.axhline(0,color="grey",ls=":");ax.set_ylabel("Target occupancy response");ax.set_title("4 new roots" if c=="fresh" else "8 chosen roots (equal root weight)");ax.legend(fontsize=7);grid(ax)
for ax,c,key,y in [(axs[1,0],"fresh","G","Mean path score"),(axs[1,1],"fresh","F_gap","Mean G² - conditional Fisher")]:
 a=[r["metrics"][f"{c}/{l}/occupancy"][key] for l in laws];ax.errorbar(ix,[x["mean"] for x in a],yerr=[x["se"] for x in a],fmt="o",capsize=4);ax.set_xticks(ix,laws);ax.axhline(0,color="grey",ls=":");ax.set_ylabel(y);grid(ax)
fig.suptitle("C: full-support marked-law pilot; bars are descriptive root-level SE")
save("REPLAN-04.C",fig,"native_pilot")
# record
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);windows=["fragment1","fragment2","fragment3","union"]
for ax,noise in zip(axs,("iid","static")):
 for k,lab in (("Holevo","Holevo"),("X_readout_MI","Fixed X readout")):
  v=[[x[k] for x in records if x["window"]==w and x["noise"]==noise] for w in windows]
  ax.plot(range(4),[np.mean(a) for a in v],marker="o",label=lab)
 ax.set_xticks(range(4),windows,rotation=15);ax.set_ylabel("Information (nats)");ax.set_title(noise);ax.legend();grid(ax)
fig.suptitle("D: declared CQ records, last-wave label disjoint; empirical 8-path mixtures")
save("REPLAN-05",fig,"conditional_CQ_empirical_drives")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for l in laws:
 a=[x for x in r["fluctuation"] if x["law"]==l]
 for ax,key in zip(axs,("KL","coarse_KL")):
  ax.plot([0,.3,.6],[st.mean(x[key] for x in a if x["amplitude"]==h) for h in (0,.3,.6)],marker="o",label=l)
for ax,y in zip(axs,("Ordered path relative entropy","Unordered count relative entropy")):
 ax.set(xlabel="Logit protocol amplitude",ylabel=y);ax.legend();grid(ax)
fig.suptitle("N: declared reversed-field / reversed-word protocol, not thermal heat")
save("REPLAN-06",fig,"exact_nonthermal_path_ratio")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,key in zip(axs,("probes","native_driven_probes")):
 for echo,lab,dx in ((False,"No echo",-.16),(True,"Echo",.16)):
  a=[x for x in r[key] if x["echo"]==echo];ax.bar(ix+dx,field(a,"trace_distance"),width=.31,label=lab)
 ax.set_xticks(ix,laws);ax.set_ylabel("Trace distance iid vs static");ax.legend();ax.set_title("Exact restricted words" if key=="probes" else "Actual classical drive ensemble")
fig.suptitle("Q: declared qubit CP probe; same diagonal, different multi-time response")
save("X-REP",fig,"conditional_CP_probe")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);a=r["closure"]["models"];keys=list(a)
axs[0].bar(range(4),[a[k]["nll"] for k in keys]);axs[0].set_xticks(range(4),["State","History","Geometry","Both"],rotation=12);axs[0].set_ylabel("Held-out NLL")
for m in keys:
 axs[1].plot(range(4),[z["nll"] for z in a[m]["test_group_NLL"]],marker="o",label=m)
axs[1].set_xticks(range(4),["Root0","Root1","Root2","Root3"]);axs[1].set_ylabel("Held-out root NLL");axs[1].legend(fontsize=8);grid(axs[1])
fig.suptitle("Memory: trained on RESTORE44, tested on four fresh roots")
save("X-MEMORY",fig,"native_heldout")
(out/"manifest.json").write_text(json.dumps(dict(source_sha256=h(S/"PHYSICS_RESULTS.json"),record_info_sha256=h(S/"RECORD_INFORMATION.json"),groups=registry),indent=2)+"\n")
print("grouped figures",len(registry))
