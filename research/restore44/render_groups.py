"""Compact topic groups: one multi-panel research figure per executed physics
node, not one independent node per legacy plot. Native vs conditional labeled."""
import json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text())
out=S/"plots/grouped";out.mkdir(parents=True,exist_ok=False)
registry={};laws=["iid","pl","reserve"];X=np.arange(3)
def save(node,fig,scope,source):
 for ext in ("png","pdf"):fig.savefig(out/(node+"."+ext),dpi=160,bbox_inches="tight")
 plt.close(fig);registry[node]={"scope":scope,"source":source,"files":{node+"."+e:hashlib.sha256((out/(node+"."+e)).read_bytes()).hexdigest() for e in ("png","pdf")}}
def med(rows,key,axis,xx):
 return [np.median([z[key] for v in rows for z in v["values"] if z[axis]==x]) for x in xx]
def grid(ax):ax.grid(alpha=.2)
zero=[x for x in r["native"] if x["h"]==0]
fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
ax=axs[0,0];ax.bar(X,[r["scale_fits"][l]["fitted"] for l in laws]);ax.axhline(1/.3,ls=":",color="black",label="Input");ax.set_xticks(X,laws);ax.set_ylabel("Conditional inverse score scale");ax.legend()
ax=axs[0,1]
for k,lab,dx in [("test_nll_fit","Fit",-.25),("test_nll_true","Input",0),("test_nll_zero","Uniform",.25)]:ax.bar(X+dx,[r["scale_fits"][l][k] for l in laws],width=.23,label=lab)
ax.set_xticks(X,laws);ax.set_ylabel("Held-out NLL (nats/draw)");ax.legend(fontsize=8)
a=r["native_temperature"];ax=axs[1,0]
for job in sorted({x["job"] for x in a}):
 q=[x for x in a if x["job"]==job and x["k"]==16 and x["beta_status"]=="finite"]
 ax.scatter([x["ply"] for x in q],[x["beta"] for x in q],s=16,label=job[:6])
ax.set(xlabel="Actual game ply",ylabel="Finite occupancy inverse score scale");ax.legend(fontsize=7)
cnt=Counter(x["beta_status"] for x in a if x["k"]==16);axs[1,1].bar(list(cnt),list(cnt.values()));axs[1,1].set_ylabel("Final-search fit status count")
fig.suptitle("T · Native score scales and occupancy thermometry (not physical beta)")
save("REPLAN-02",fig,"native_observed","scale_fits,native_temperature")

fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
ax=axs[0,0]
for i,l in enumerate(laws):
 a=[x for x in zero if x["law"]==l];ax.scatter(np.full(len(a),i)+np.linspace(-.12,.12,len(a)),[-x["log_root_prob"] for x in a],s=14)
ax.set_xticks(X,laws);ax.set_ylabel("Native root path action (nats)")
ax=axs[0,1]
for l in laws:
 a=[x for x in r["exact"] if x["law"]==l];ax.plot([1,2,4],[np.mean([x["response_cov_gap"] for x in a if x["width"]==w]) for w in [1,2,4]],marker="o",label=l)
ax.set(xlabel="Restricted exact wave width",ylabel="||R-C|| Frobenius");ax.legend();grid(ax)
ax=axs[1,0];xx=[4,8,16,32,64]
for k,l in [("leading_error","Leading action"),("one_loop_error","Gaussian determinant")]:
 a=[[z[k] for v in r["loop"] for z in v["values"] if z["M"]==m] for m in xx]
 ax.plot(xx,[np.median(v) for v in a],marker="o",label=l);ax.fill_between(xx,[min(v) for v in a],[max(v) for v in a],alpha=.12)
ax.set(xscale="log",yscale="log",xlabel="Independent frozen-wave replicas M",ylabel="Absolute log-mass error");ax.legend(fontsize=8);grid(ax)
ax=axs[1,1];ax.bar(X,[sum(x["unvisited"] for x in zero if x["law"]==l)/sum(x["draws"] for x in zero if x["law"]==l) for l in laws]);ax.set_xticks(X,laws);ax.set_ylabel("Native zero-visit selection fraction")
fig.suptitle("A · Marked action, collective response and conditional saddle correction")
save("REPLAN-03",fig,"native_and_exact_restricted_reference","native,exact,loop")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
maps=["ordered","prefix1","unordered","target_count","constant"];x=np.arange(len(maps))
for l in laws:
 a=[v for v in r["coarse"] if v["width"]==4 and v["law"]==l];axs[0].plot(x,[np.mean([v["maps"][m]["relative"] for v in a]) for m in maps],marker="o",label=l)
axs[0].set_xticks(x,maps,rotation=25);axs[0].set_ylabel("Retained Fisher trace fraction");axs[0].legend();grid(axs[0])
for v in r["graph"]:
 a=v["depths"];axs[1].plot([x["coarse_dim"]/x["nodes"] for x in a],[x["static_gap"] for x in a],marker=".",alpha=.55)
axs[1].set(xlabel="Retained prefix dimension / tree nodes",ylabel="Schur vs Galerkin norm");grid(axs[1])
fig.suptitle("R · Exact pushforward and declared weighted native-prefix graph")
save("REPLAN-04.R",fig,"restricted_reference_and_native_graph_operator","coarse,graph")

fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for ax,A in zip(axs[0],["occupancy","mean_mark"]):
 for key,lab,dx in [(A+"_LR","Score",-.24),(A+"_FD","Direct finite h",0),(A+"_ISFD","LR finite h",.24)]:
  val=[];se=[]
  for l in laws:
   z=np.array([x["stats"][key]["mean"] for x in r["paired"] if x["law"]==l]);val.append(z.mean());se.append(z.std(ddof=1)/2)
  ax.errorbar(X+dx,val,yerr=se,fmt="o",capsize=3,label=lab)
 ax.axhline(0,color="grey",ls=":");ax.set_xticks(X,laws);ax.set_ylabel(A+" response");ax.legend(fontsize=7);grid(ax)
for l in laws:
 a=[v for v in r["cumulants"] if v["width"]==4 and v["law"]==l]
 for ax,k in zip(axs[1],["skew","excess"]):ax.plot([1,2,4,8,16,32,64],[np.median([abs(z[k]) for v in a for z in v["replicas"] if z["M"]==m]) for m in [1,2,4,8,16,32,64]],marker="o",label=l)
for ax,y in zip(axs[1],["Absolute standardized skewness","Absolute excess kurtosis"]):ax.set(xscale="log",yscale="log",xlabel="Independent frozen-wave replicas",ylabel=y);ax.legend();grid(ax)
fig.suptitle("C · Actual finite-source response (4 opening blocks) / exact replica scaling")
save("REPLAN-04.C",fig,"native_pilot_and_exact_reference","paired,cumulants")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);xx=[0,.25,.5,.75,1.]
for typ,l in [("iid","Independent"),("static","Quasistatic")]:axs[0].plot(xx,med(r["quantum"],"reduced_S_"+typ,"alpha",xx),marker="o",label=l)
axs[0].set(xlabel="Declared noise angle",ylabel="Reduced S_vN");axs[0].legend();grid(axs[0])
for k,lab in [("QMI","Quantum MI"),("Holevo","Holevo"),("basis_MI","Basis readout")]:
 y=[np.median([z[k]/z["H_pointer"] for a in r["darwinism"] for z in a["values"] if z["fragment"]==m]) for m in range(1,7)]
 axs[1].plot(range(1,7),y,marker="o",label=lab)
axs[1].axhline(1,color="grey",ls=":");axs[1].set(xlabel="Retained record fragments",ylabel="Information / pointer Shannon H");axs[1].legend();grid(axs[1])
fig.suptitle("D · Conditional quantum coarse state and controlled record isometry")
save("REPLAN-05",fig,"conditional_quantum","quantum,darwinism")

fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True);a=r["thermal"];xx=[.25,.5,1,2,4]
ax=axs[0,0];ax.scatter([z["JE_expected"] for z in a],[z["JE"] for z in a]);lo=min(z["JE"] for z in a);hi=max(z["JE"] for z in a);ax.plot([lo,hi],[lo,hi],ls=":",color="grey");ax.set(xlabel="exp(-Delta F)",ylabel="<exp(-W)>")
ax=axs[0,1];ax.plot(range(len(a)),[z["sigma"] for z in a],marker="o",label="Delta S-Q");ax.plot(range(len(a)),[z["KL"] for z in a],marker="x",ls="--",label="KL");ax.set(xlabel="Finite reference",ylabel="Entropy production (k_B units)");ax.legend()
for v in a:
 axs[1,0].plot(xx,[z["C_over_kB"] for z in v["temp"]],alpha=.5);axs[1,1].plot(xx,[z["S"] for z in v["temp"]],alpha=.5)
for ax,y in zip(axs[1],["C/k_B","Entropy/k_B"]):ax.set(xscale="log",xlabel="Declared beta E_unit",ylabel=y);grid(ax)
fig.suptitle("N · Declared Gibbs quench/reset: exact JE, entropy balance and heat capacity")
save("REPLAN-06",fig,"conditional_thermal_reference","thermal")

fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True);xx=[0,.25,.5,.75,1.]
for typ,l in [("iid","Independent"),("static","Quasistatic")]:
 axs[0,0].plot(xx,med(r["quantum"],"S_"+typ,"alpha",xx),marker="o",label=l)
 axs[0,1].plot(xx,med(r["quantum"],"negativity_"+typ,"alpha",xx),marker="o",label=l)
for ax,y in zip(axs[0],["von Neumann S","Ordered-draw negativity"]):ax.set(xlabel="Declared noise angle",ylabel=y);ax.legend();grid(ax)
a=r["obstructions"]["euler"];axs[1,0].plot([v["dt"] for v in a],[v["euler_min"] for v in a],marker="o",label="Euler");axs[1,0].plot([v["dt"] for v in a],[v["unitary_min"] for v in a],marker="o",label="Exact unitary");axs[1,0].set(xlabel="Dimensionless step",ylabel="Minimum eigenvalue");axs[1,0].legend()
a=r["obstructions"]["projector"];axs[1,1].scatter(range(len(a)),[v["gibbs_residual"] for v in a]);axs[1,1].set(xlabel="Actual-law reference",ylabel="Gibbs stationarity residual")
fig.suptitle("Q · Explicit complex state, entanglement and CP evolution obstructions")
save("X-REP",fig,"conditional_quantum_reference","quantum,obstructions")

fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True);xx=[0,.25,.5,.75,1.]
for k,l in [("trace_distance","Two-draw"),("reduced_trace_distance","First-draw reduced")]:axs[0,0].plot(xx,med(r["quantum"],k,"alpha",xx),marker="o",label=l)
for k,l in [("probe_iid","Independent"),("probe_static","Quasistatic")]:axs[0,1].plot(xx,med(r["quantum"],k,"alpha",xx),marker="o",label=l)
for ax,y in zip(axs[0],["Trace distance","Coherent projector probability"]):ax.set(xlabel="Declared noise angle",ylabel=y);ax.legend();grid(ax)
for v in r["graph"]:
 a=v["depths"];axs[1,0].plot([x["coarse_dim"]/x["nodes"] for x in a],[x["memory_trace"] for x in a],marker=".",alpha=.65)
axs[1,0].set(xlabel="Retained prefix fraction",ylabel="Projected semigroup defect trace");grid(axs[1,0])
c=r["closure"]["models"];names=list(c);axs[1,1].bar(range(len(names)),[c[n]["nll"] for n in names]);axs[1,1].set_xticks(range(len(names)),["State","History","Geometry","Both"],rotation=12);axs[1,1].set_ylabel("Held-out return-sign NLL")
fig.suptitle("Memory · Native history, projected graph and nonthermal two-step channel")
save("X-MEMORY",fig,"native_pilot_and_conditional_quantum","quantum,graph,closure")

fig,axs=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
axs[0].bar([g["position"] for g in r["games"]],[g["plies"] for g in r["games"]]);axs[0].set_ylabel("Natural-terminal game length")
axs[1].bar(X,[sum(x["collision"] for x in zero if x["law"]==l)/sum(x["pairs"] for x in zero if x["law"]==l) for l in laws]);axs[1].set_xticks(X,laws);axs[1].set_ylabel("Actual h=0 wave collision fraction")
fig.suptitle("Native common MixNet cohort: game and root history controls")
save("REPLAN-01",fig,"observed_native","games,native")
manifest=dict(source_sha256=hashlib.sha256((S/"PHYSICS_RESULTS.json").read_bytes()).hexdigest(),groups=registry)
(out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("grouped plots",len(registry))
