"""Nine grouped outputs, explicit group response vs conditional physical references."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text())
out=S/"plots/grouped";out.mkdir(exist_ok=False);reg={}
Ls=[13,21];laws=["iid","pl","reserve"]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ("png","pdf"):fig.savefig(out/(node+"."+e),dpi=160,bbox_inches="tight")
 plt.close(fig);reg[node]=dict(scope=scope,files={node+"."+e:h(out/(node+"."+e)) for e in ("png","pdf")})
def mean(a,k):return st.mean(x[k] for x in a)
def grid(ax):ax.grid(alpha=.2)
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for i,l in enumerate(laws):
 z=[x for x in r["compare"] if x["law"]==l];axs[0].scatter([x["scalar_F"] for x in z],[x["sector_F"] for x in z],label=l)
axs[0].plot([0,4],[0,4],ls=":",color="grey");axs[0].set(xlabel="Old scalar-source Fisher",ylabel="New group-source Fisher");axs[0].legend()
axs[1].bar(Ls,[x["plies"] for x in r["games"]]);axs[1].set(xlabel="Board side",ylabel="Actual game plies")
fig.suptitle("Native same-MixNet source comparison: different observables, new seeds")
save("REPLAN-01",fig,"native_group_source")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for L in Ls:
 for B in (16,32):
  z=[x for x in r["temperature"] if x["L"]==L and x["B"]==B];ks=sorted({x["k"] for x in z})
  axs[0].plot(ks,[mean([x for x in z if x["k"]==k],"H") for k in ks],marker=".",label=f"L{L} B{B}")
  axs[1].plot(ks,[st.mean(x["beta_status"]=="finite" for x in z if x["k"]==k) for k in ks],marker=".",label=f"L{L} B{B}")
for ax,y in zip(axs,("Occupancy entropy (nats)","Finite inverse score fit fraction")):ax.set(xscale="log",xlabel="Completed backups",ylabel=y);ax.legend();grid(ax)
fig.suptitle("T: native score-scale diagnostics, not physical temperature");save("REPLAN-02",fig,"native")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for k,l in (("leading_error","Leading"),("one_loop_error","Lattice determinant")):
 a=[[v[k] for x in r["loops"] for v in x["values"] if v["M"]==M] for M in (8,16,32)]
 axs[0].plot((8,16,32),[np.median(v) for v in a],marker="o",label=l)
axs[0].set(xscale="log",yscale="log",xlabel="Independent restricted waves",ylabel="Absolute log-mass error");axs[0].legend()
for B in (16,32):
 a=[x for x in r["actions"] if x["B"]==B]
 axs[1].plot((.0001,.001,.01),[st.mean(y["delta"] for x in a for y in x["fits"] if y["ridge"]==l) for l in (.0001,.001,.01)],marker="o",label=f"B{B}")
axs[1].axhline(0,color="grey",ls=":");axs[1].set(xscale="log",xlabel="Fixed ridge",ylabel="Heldout NLL − isotropic");axs[1].legend()
fig.suptitle("A: restricted exact reference and separate native projected action");save("REPLAN-03",fig,"exact_reference_and_empirical")
fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for i,L in enumerate(Ls):
 for j,l in enumerate(laws):
  a=[x for x in r["spatial"] if x["L"]==L and x["B"]==16 and x["law"]==l and x["position"].endswith("g0v0")][0]
  z=np.asarray(a["response"]).reshape(L,L);lim=max(abs(z).max(),1e-12)
  im=axs[i,j].imshow(z,cmap="RdBu_r",origin="lower",vmin=-lim,vmax=lim);axs[i,j].set_title(f"L{L} {l}, sector {a['sector']}");fig.colorbar(im,ax=axs[i,j],shrink=.8)
fig.suptitle("R: action occupancy response to group field, actual baseline score");save("REPLAN-04.R",fig,"native_group_response")
fig,axs=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for j,k in enumerate(("cov_trace","S_low","group_response")):
 for B in (16,32):
  for law in laws:
   a=[x for x in r["spatial"] if x["B"]==B and x["law"]==law and x["position"].endswith("g0v0")]
   axs[j].plot(Ls,[mean([x for x in a if x["L"]==L],k) for L in Ls],marker="o",label=f"{law} B{B}")
 axs[j].set(xlabel="Board side",ylabel=k);axs[j].legend(fontsize=6);grid(axs[j])
fig.suptitle("C: same-root size and budget pilot; no critical exponent");save("REPLAN-04.C",fig,"native_finite_size")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
q=r["quantum"];k=["none","X","Z","XZ","ZX"]
axs[0].bar(k,[st.mean(abs(x["correlations"][s]) for x in q) for s in k]);axs[0].set_ylabel("Mean absolute past-future correlation")
axs[1].scatter([x["n1"] for x in q],[x["entropy_S"] for x in q],s=13,alpha=.4);axs[1].set(xlabel="Actual first-block sector count",ylabel="Conditional system vN entropy")
fig.suptitle("D: declared coherent environment, nonselective intervention controls");save("REPLAN-05",fig,"conditional_quantum")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);ts=(4.,16.,64.)
for tau in ts:
 a=[x for x in r["thermal"] if x["tau"]==tau];f=np.mean([x["forward"] for x in a],0);rev=np.mean([x["reverse"] for x in a],0)[::-1]
 axs[0].plot(a[0]["lambdas"],f,marker="o",label=f"F {tau:g}");axs[0].plot(a[0]["lambdas"],rev,ls=":",label=f"R reflected {tau:g}")
axs[0].set(xlabel="Tilt λ",ylabel="Work generating function");axs[0].legend(fontsize=7)
axs[1].plot(ts,[mean([x for x in r["thermal"] if x["tau"]==t],"W") for t in ts],marker="o",label="Forward W");axs[1].plot(ts,[mean([x for x in r["thermal"] if x["tau"]==t],"W_reverse") for t in ts],marker="s",label="Reverse W");axs[1].set(xscale="log",xlabel="Duration",ylabel="Declared work / E*");axs[1].legend()
fig.suptitle("N: continuous tilted generator; Crooks symmetry and Jarzynski");save("REPLAN-06",fig,"conditional_thermal")
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
axs[0].scatter([x["n1"] for x in q],[x["CHSH"] for x in q],s=13,alpha=.4,label="coherent");axs[0].scatter([x["n1"] for x in q],[x["dephased_CHSH"] for x in q],s=13,alpha=.4,label="dephased");axs[0].axhline(2,color="grey",ls=":");axs[0].set(xlabel="Actual first-block count",ylabel="Declared optimal CHSH");axs[0].legend()
axs[1].scatter([x["entropy_S"] for x in q],[x["CHSH"] for x in q],s=13,alpha=.5);axs[1].set(xlabel="Declared S:E entanglement entropy",ylabel="Optimal CHSH")
fig.suptitle("Q: explicit complex-state witness, not native quantum experiment");save("X-REP",fig,"conditional_complex_state")
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for i,L in enumerate(Ls):
 for j,B in enumerate((16,32)):
  ax=axs[i,j]
  for law in laws:
   a=[x for x in r["temporal"] if x["L"]==L and x["B"]==B and x["law"]==law and x["block"]==B//4-1]
   v=np.mean([x["values"] for x in a],0);ax.plot(range(1,B//4+1),v[1,:,4],marker="o",ls=":",label=law+" LR");ax.plot(range(1,B//4+1),v[2,:,4],marker="s",label=law+" direct")
  ax.axhline(0,color="grey",ls=":");ax.set(xlabel="Observable block",ylabel="Group mass finite contrast",title=f"L{L} B{B} late pulse");ax.legend(fontsize=6)
fig.suptitle("Memory: actual causal group pulses, exact pre-intervention null");save("X-MEMORY",fig,"native_temporal")
(out/"manifest.json").write_text(json.dumps(dict(source_sha256=h(S/"PHYSICS_RESULTS.json"),states_sha256=h(S/"CONDITIONAL_STATES.npz"),groups=reg),indent=2)+"\n")
print("grouped figures",len(reg))
