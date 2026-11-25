"""Nine grouped figures; native and declared conditional models separated."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text())
out=S/"plots/grouped";out.mkdir(exist_ok=False);reg={}
Ls=[13,17,21];laws=["iid","pl","reserve"];strata=["high","medium","rare"]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ("png","pdf"):fig.savefig(out/(node+"."+e),dpi=160,bbox_inches="tight")
 plt.close(fig);reg[node]=dict(scope=scope,files={node+"."+e:h(out/(node+"."+e)) for e in ("png","pdf")})
def grid(ax):ax.grid(alpha=.2)
def mean(a,k):return st.mean(x[k] for x in a)
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
axs[0].bar(Ls,[sum(x["jobs"] for k,x in r["counts"][f"L{L}"].items() if k!="game") for L in Ls]);axs[0].set(xlabel="Board side L",ylabel="New actual root histories")
a=[x for x in r["diagnostics"] if x["campaign"]=="confirm"]
for L in Ls:
 z=[x for x in a if x["L"]==L];axs[1].scatter([x["pilot"]["pilot_F"] for x in z],[x["Fisher"]["mean"] for x in z],label=f"L{L}")
u=max(max(x["pilot"]["pilot_F"],x["Fisher"]["mean"]) for x in a);axs[1].plot([0,u],[0,u],color="grey",ls=":");axs[1].set(xlabel="Pilot-predicted Fisher / history",ylabel="Independent confirmation Fisher");axs[1].legend();grid(axs[1])
fig.suptitle("Independent source selection and common geometry: same MixNet")
save("REPLAN-01",fig,"native_pilot_confirmation")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for l in laws:
 for L in Ls:
  a=[x for x in r["temperature"] if x["L"]==L and x["law"]==l];ks=sorted({x["k"] for x in a})
  axs[0].plot(ks,[mean([x for x in a if x["k"]==k],"H") for k in ks],marker=".",label=f"{l} L{L}")
  q=[x for x in a if x["k"]==16]
  if L==Ls[0]:v=[]
 for L in Ls:
  a=[x for x in r["temperature"] if x["L"]==L and x["law"]==l and x["k"]==16];v.append(st.mean(x["beta_status"]=="finite" for x in a))
 axs[1].plot(Ls,v,marker="o",label=l)
axs[0].set(xscale="log",xlabel="Completed backups",ylabel="Occupancy entropy (nats)");axs[0].legend(fontsize=6,ncol=2)
axs[1].set(xlabel="Board side",ylabel="Finite inverse score-fit fraction");axs[1].legend()
for ax in axs:grid(ax)
fig.suptitle("T: native entropy and score scale (not physical temperature)")
save("REPLAN-02",fig,"native_observed")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,l in (("leading_error","Leading"),("one_loop_error","Lattice determinant")):
 a=[[v[k] for x in r["loops"] for v in x["values"] if v["M"]==M] for M in (8,16,32)]
 axs[0].plot((8,16,32),[np.median(z) for z in a],marker="o",label=l)
axs[0].set(xscale="log",yscale="log",xlabel="Independent frozen waves",ylabel="Absolute log-mass error");axs[0].legend()
for key,lab in (("actions","n8 histories"),("actions_n16","n16 selected high roots")):
 a=r[key];axs[1].plot((.0001,.001,.01),[st.mean(y["delta"] for x in a for y in x["fits"] if y["ridge"]==lam) for lam in (.0001,.001,.01)],marker="o",label=lab)
axs[1].axhline(0,color="grey",ls=":");axs[1].set(xscale="log",xlabel="Fixed covariance ridge",ylabel="Heldout Gaussian NLL − isotropic");axs[1].legend(fontsize=8)
fig.suptitle("A: exact restricted action; independent tangent density test")
save("REPLAN-03",fig,"exact_reference_and_heldout")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for i,k in enumerate(("drift_trace","martingale_trace","cross_trace")):
 axs[0].bar(np.arange(3)+(i-1)*.25,[mean([x for x in r["noise"] if x["law"]==l],k) for l in laws],width=.23,label=k)
axs[0].set_xticks(range(3),laws);axs[0].set_ylabel("Sector covariance trace");axs[0].legend(fontsize=7)
for i,l in enumerate(laws):
 a=[x for x in r["score_bounds"] if x["law"]==l];v=np.mean([x["response_fraction"] for x in a],0)
 axs[1].bar(np.arange(5)+(i-1)*.25,v,width=.23,label=l)
axs[1].set_xticks(range(5),["tan1","tan2","tan3","target","mark"]);axs[1].set_ylabel("Squared normalized sample response");axs[1].set_ylim(0,1.02);axs[1].legend()
fig.suptitle("R: adaptive martingale decomposition and score covariance bound")
save("REPLAN-04.R",fig,"native_martingale_information")

fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for l in laws:
 a=[x for x in r["summary"] if x["law"]==l]
 for ax,k in ((axs[0,0],"cov_trace"),(axs[0,1],"S_low"),(axs[1,0],"central_mass")):ax.plot(Ls,[x[k] for x in a],marker="o",label=l)
for l in laws:
 a=[x for x in r["peripheral_variants"] if x["law"]==l];axs[1,1].plot(Ls,[st.mean(abs(x["cov_trace"]) for x in a if x["L"]==L) for L in Ls],marker="o",label=l)
for ax,yl in zip(axs.ravel(),("Connected covariance trace","Lowest nonzero Fourier mode","Central 5x5 mass","Peripheral pair absolute covariance difference")):
 ax.set(xlabel="Board side L",ylabel=yl);ax.legend();grid(ax)
fig.suptitle("C: three central templates, two peripheral variants; finite B16")
save("REPLAN-04.C",fig,"native_finite_size_geometry")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
ths=(0.,np.pi/2,np.pi)
for l in laws:
 a=[x for x in r["quantum"] if x["law"]==l]
 axs[0].plot(ths,[mean([x for x in a if x["theta"]==t],"MI") for t in ths],marker="o",label=l)
 axs[1].plot(ths,[mean([x for x in a if x["theta"]==t],"future_gap") for t in ths],marker="o",label=l)
for ax,yl in zip(axs,("Past-future conditional MI (nats)","Future probability gap after same system reset")):
 ax.axhline(0,color="grey",ls=":",label="dephased/reset");ax.set_xticks(ths,["0","π/2","π"]);ax.set(xlabel="Environment phase angle",ylabel=yl);ax.legend();grid(ax)
fig.suptitle("D: explicit two-qubit environment memory with causal break")
save("REPLAN-05",fig,"conditional_coherent_records")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
ts=(16.,64.,256.);a=r["thermal"]
for k,lab in (("work","Work"),("sigma","Entropy production"),("lag","Final lag KL")):
 axs[0].plot(ts,[mean([x for x in a if x["tau"]==t],k) for t in ts],marker="o",label=lab)
axs[0].set(xscale="log",yscale="log",xlabel="Duration",ylabel="Dimensionless work / entropy");axs[0].legend()
axs[1].plot(ts,[mean([x for x in a if x["tau"]==t],"tau_work") for t in ts],marker="o",label="τ W")
axs[1].axhline(mean([x for x in a if x["tau"]==16.],"friction"),color="grey",ls=":",label="Friction integral")
axs[1].set(xscale="log",xlabel="Duration",ylabel="Scaled work");axs[1].legend()
fig.suptitle("N: continuous conditional Gibbs reset; slow-driving friction")
save("REPLAN-06",fig,"conditional_thermal")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
a=[x for x in r["quantum"] if x["theta"]==0]
for l in laws:
 z=[x for x in a if x["law"]==l];axs[0].scatter([x["negativity"] for x in z],[x["MI"] for x in z],s=12,alpha=.4,label=l)
axs[0].set(xlabel="First-block S:E negativity",ylabel="Later record MI");axs[0].legend()
for l in laws:
 z=[x for x in r["quantum"] if x["law"]==l];axs[1].plot(ths,[mean([x for x in z if x["theta"]==t],"correlation") for t in ths],marker="o",label=l)
axs[1].set_xticks(ths,["0","π/2","π"]);axs[1].set(xlabel="Environment phase",ylabel="Signed measurement correlation");axs[1].legend()
fig.suptitle("Q: complex coherent environmental intervention; conditional model only")
save("X-REP",fig,"conditional_quantum_state")

fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for j,L in enumerate((13,21)):
 ax=axs[0,j]
 for k,tag in enumerate(strata):
  a=[x for x in r["temporal"] if x["L"]==L and x["B"]==16 and x["stratum"]==tag and x["block"]==0]
  v=np.mean([x["values"] for x in a],0)
  ax.plot(range(1,5),v[1,:,4],marker="o",ls=":",label=tag+" LR")
  ax.plot(range(1,5),v[2,:,4],marker="s",label=tag+" direct")
 ax.axhline(0,color="grey",ls=":");ax.set(xlabel="Observable block",ylabel="Target finite contrast",title=f"L{L} early pulse");ax.legend(fontsize=7)
for j,k in enumerate(("sector","mark")):
 ax=axs[1,j]
 for i,mode in enumerate(("state","clock","memory","clock_memory")):
  ax.bar(np.arange(3)+(i-1.5)*.18,[st.mean(x["models"][mode][k] for x in r["memory"] if x["law"]==l) for l in laws],width=.17,label=mode)
 ax.set_xticks(range(3),laws);ax.set_ylabel(k+" MSE");ax.legend(fontsize=7)
fig.suptitle("Memory: informative pulse and leave-one-central-geometry-out prediction")
save("X-MEMORY",fig,"native_temporal_and_geometry_heldout")
(out/"manifest.json").write_text(json.dumps(dict(source_sha256=h(S/"PHYSICS_RESULTS.json"),states_sha256=h(S/"CONDITIONAL_ENV_STATES.npz"),groups=reg),indent=2)+"\n")
print("grouped figures",len(reg))
