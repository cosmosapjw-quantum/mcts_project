"""Nine original-topic grouped figures; spatial density, time and beta distinct."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text())
out=S/"plots/grouped";out.mkdir(exist_ok=False);reg={}
Ls=[13,15,17,19,21];laws=["iid","pl","reserve"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ("png","pdf"):fig.savefig(out/(node+"."+e),dpi=160,bbox_inches="tight")
 plt.close(fig);reg[node]=dict(scope=scope,files={node+"."+e:sha(out/(node+"."+e)) for e in ("png","pdf")})
def grid(ax):ax.grid(alpha=.2)
def mean(a,k):return st.mean(x[k] for x in a)
def subset(L,law,kind):return next(x for x in r["summary"] if x["L"]==L and x["law"]==law and x["kind"]==kind)
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for k in ("fixed","dense"):
 axs[0].plot(Ls,[subset(L,"iid",k)["density"] for L in Ls],marker="o",label=k)
axs[0].axhline(.06,color="grey",ls=":");axs[0].set(xlabel="Board side",ylabel="Initial stone fraction");axs[0].legend()
axs[1].bar(Ls,[x["plies"] for x in r["games"]]);axs[1].set(xlabel="Board side",ylabel="Completed game plies")
fig.suptitle("Native template and peripheral-density control; one complete game per L")
save("REPLAN-01",fig,"native_templates_and_games")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for l in laws:
 for kind,ls in (("fixed",":"),("dense","-")):
  axs[0].plot(Ls,[subset(L,l,kind)["H"] for L in Ls],marker="o",ls=ls,label=l+" "+kind)
  a=[x for x in r["temperature"] if x["B"]==16 and x["beta"]==10/3 and kind in x["position"] and x["law"]==l and x["k"]==16]
  # Equal mean over actual root variants; do not treat reused histories as new roots.
  v=[]
  for L in Ls:
   b=[x for x in a if x["L"]==L];ps=sorted({x["position"] for x in b});v.append(st.mean(st.mean(y["beta_status"]=="finite" for y in b if y["position"]==p) for p in ps))
  axs[1].plot(Ls,v,marker="o",ls=ls,label=l+" "+kind)
axs[0].set(xlabel="Board side",ylabel="Occupancy entropy (nats)");axs[1].set(xlabel="Board side",ylabel="Finite inverse score-fit fraction")
for ax in axs:ax.legend(fontsize=7);grid(ax)
fig.suptitle("T: actual score-scale and entropy, B16, beta=10/3; not physical temperature")
save("REPLAN-02",fig,"native_observed")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,l in (("leading_error","Leading"),("one_loop_error","Lattice determinant")):
 a=[[v[k] for x in r["loops"] for v in x["values"] if v["M"]==M] for M in (8,16,32)]
 axs[0].plot((8,16,32),[np.median(z) for z in a],marker="o",label=l)
 axs[0].fill_between((8,16,32),[min(z) for z in a],[max(z) for z in a],alpha=.12)
axs[0].set(xscale="log",yscale="log",xlabel="Independent frozen waves",ylabel="Absolute log-mass error");axs[0].legend()
for kind,ls in (("fixed",":"),("dense","-")):
 a=[x for x in r["actions"] if x["B"]==16 and x["beta"]==10/3 and kind in x["position"]]
 axs[1].plot((.0001,.001,.01),[st.mean(y["delta"] for x in a for y in x["fits"] if y["ridge"]==lam) for lam in (.0001,.001,.01)],marker="o",ls=ls,label=kind)
axs[1].axhline(0,color="grey",ls=":");axs[1].set(xscale="log",xlabel="Fixed covariance ridge",ylabel="Heldout Gaussian NLL − isotropic");axs[1].legend()
fig.suptitle("A: exact restricted action and regularized tangent predictive test")
save("REPLAN-03",fig,"exact_reference_and_rank_limited_prediction")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for kind,ls in (("fixed",":"),("dense","-")):
 for k,lab in (("drift_trace","predictable drift"),("martingale_trace","martingale"),("cross_trace","cross")):
  axs[0].plot(Ls,[mean([x for x in r["noise"] if x["L"]==L and kind in x["position"]],k) for L in Ls],marker="o",ls=ls,label=kind+" "+lab)
axs[0].axhline(0,color="grey",ls=":");axs[0].set(xlabel="Board side",ylabel="Sector covariance trace contributions");axs[0].legend(fontsize=7)
for l in laws:
 for kind,ls in (("fixed",":"),("dense","-")):
  axs[1].plot(Ls,[mean([x for x in r["noise"] if x["L"]==L and kind in x["position"] and x["law"]==l],"bracket_gap") for L in Ls],marker="o",ls=ls,label=l+" "+kind)
axs[1].axhline(0,color="grey",ls=":");axs[1].set(xlabel="Board side",ylabel="Sample martingale − predictable bracket trace");axs[1].legend(fontsize=7)
fig.suptitle("R: exact finite-sample noise decomposition; bracket equality only in expectation")
save("REPLAN-04.R",fig,"native_martingale_decomposition")

fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for law in laws:
 for kind,ls in (("fixed",":"),("dense","-")):
  for ax,k in ((axs[0,0],"cov_trace"),(axs[0,1],"S_low"),(axs[1,0],"central_mass")):
   ax.plot(Ls,[subset(L,law,kind)[k] for L in Ls],marker="o",ls=ls,label=law+" "+kind)
for beta,ls in ((1.,":"),(10/3,"-")):
 for L in (13,17,21):
  a=[x for x in r["spatial"] if x["L"]==L and x["position"].endswith("dense0") and x["beta"]==beta]
  axs[1,1].plot((16,32,64),[mean([x for x in a if x["B"]==B],"cov_trace") for B in (16,32,64)],marker="o",ls=ls,label=f"L{L} beta{beta:.2g}")
for ax,yl in zip(axs.ravel(),("Connected covariance trace","Lowest Fourier mode","Central 5x5 mass","Dense0 covariance trace")):
 ax.set_ylabel(yl);ax.set_xlabel("Board side" if ax!=axs[1,1] else "Backup budget");ax.legend(fontsize=6);grid(ax)
fig.suptitle("C: separate spatial/density and budget/score-scale pilots")
save("REPLAN-04.C",fig,"native_finite_size_and_time")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for l in laws:
 a=[x for x in r["quantum"] if x["law"]==l]
 for k,lab,ls in (("coherent_readout","coherent","-"),("dephased_readout","dephased",":")):
  axs[0].plot((0,.3,.6,1.2),[mean([v for x in a for v in x["rows"] if v["theta"]==t],k) for t in (0,.3,.6,1.2)],marker="o",ls=ls,label=l+" "+lab)
 axs[1].plot((0,.3,.6,1.2),[mean([v for x in a for v in x["rows"] if v["theta"]==t],"herald_prob") for t in (0,.3,.6,1.2)],marker="o",label=l)
axs[0].set(xlabel="Declared fragment measurement angle",ylabel="Conditional coherent projector probability");axs[1].set(xlabel="Declared fragment angle",ylabel="Herald probability")
for ax in axs:ax.legend(fontsize=7);grid(ax)
fig.suptitle("D: two separated fragment readouts, postselected vs dephased")
save("REPLAN-05",fig,"conditional_record_erasure")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,lab in (("work","Work"),("sigma","Entropy production"),("lag","Final lag")):
 for kind,ls in (("fixed",":"),("dense","-")):
  axs[0].plot((1,4,16,64),[mean([x for x in r["thermal"] if x["duration"]==T and kind in x["position"]],k) for T in (1,4,16,64)],marker="o",ls=ls,label=kind+" "+lab)
for l in laws:
 for kind,ls in (("fixed",":"),("dense","-")):
  axs[1].plot(Ls,[mean([x for x in r["protocol"] if x["L"]==L and x["law"]==l and kind in x["position"]],"mean_weight") for L in Ls],marker="o",ls=ls,label=l+" "+kind)
axs[0].set(xscale="log",xlabel="Declared cycle duration",ylabel="Dimensionless work / entropy");axs[1].axhline(1,color="grey",ls=":");axs[1].set(xlabel="Board side",ylabel="Actual source LR mean")
for ax in axs:ax.legend(fontsize=7);grid(ax)
fig.suptitle("N: declared thermal cycle (left), actual likelihood ratio (right)")
save("REPLAN-06",fig,"conditional_thermal_and_native_LR")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for l in laws:
 a=[x for x in r["quantum"] if x["law"]==l]
 for kind,ls in (("fixed",":"),("dense","-")):
  b=[x for x in a if kind in x["position"]]
  axs[0].plot(Ls,[mean([v for x in b if x["L"]==L for v in x["rows"] if v["theta"]==.6],"QFI") for L in Ls],marker="o",ls=ls,label=l+" "+kind)
  axs[1].plot(Ls,[mean([v for x in b if x["L"]==L for v in x["rows"] if v["theta"]==.6],"imag_norm") for L in Ls],marker="o",ls=ls,label=l+" "+kind)
axs[0].set(xlabel="Board side",ylabel="Conditional system phase QFI");axs[1].set(xlabel="Board side",ylabel="Complex conditional state imaginary norm")
for ax in axs:ax.legend(fontsize=7);grid(ax)
fig.suptitle("Q: explicit postselected complex state, no native quantum-phase claim")
save("X-REP",fig,"conditional_complex_states")

fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for j,LL in enumerate((15,21)):
 ax=axs[0,j]
 for block,ls in ((0,"-"),(1,"--"),(3,":")):
  a=[x for x in r["temporal"] if x["L"]==LL and x["block"]==block]
  v=np.mean([x["values"] for x in a],0)
  ax.plot(range(1,5),v[1,:,4],marker="o",ls=ls,label=f"b{block+1} LR")
  ax.plot(range(1,5),v[2,:,4],marker="s",ls=ls,label=f"b{block+1} direct")
 ax.axhline(0,color="grey",ls=":");ax.set(xlabel="Observable block",ylabel="Target finite response",title=f"L{LL}, 8 seeds/law");ax.legend(fontsize=7)
mods=["state","clock","memory","clock_memory"]
for j,k in enumerate(("sector_mse","mark_mse")):
 ax=axs[1,j]
 for i,m in enumerate(mods):ax.bar(np.arange(3)+(i-1.5)*.18,[next(x for x in r["memory"] if x["law"]==l)["models"][m][k] for l in laws],width=.17,label=m)
 ax.set_xticks(range(3),laws);ax.set_ylabel(k);ax.legend(fontsize=7)
fig.suptitle("Memory: early/middle/late direct pulse and heldout second geometry")
save("X-MEMORY",fig,"native_pulses_and_geometry_holdout")
(out/"manifest.json").write_text(json.dumps(dict(source_sha256=sha(S/"PHYSICS_RESULTS.json"),states_sha256=sha(S/"COHERENT_STATES.npz"),groups=reg),indent=2)+"\n")
print("grouped figures",len(reg))
