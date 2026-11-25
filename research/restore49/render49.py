"""Compact original-topic multi-panel figures, real size family and source pulse."""
import json,statistics as st,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/"PHYSICS_RESULTS.json").read_text());ref=json.loads((S/"SPATIAL_REFERENCE_RESULTS.json").read_text());ph=json.loads((S/"PHASE_RESULTS.json").read_text())
out=S/"plots/grouped";out.mkdir(exist_ok=False);reg={}
L=[13,15,17,19,21];laws=["iid","pl","reserve"]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ("png","pdf"):fig.savefig(out/(node+"."+e),dpi=160,bbox_inches="tight")
 plt.close(fig);reg[node]=dict(scope=scope,files={node+"."+e:h(out/(node+"."+e)) for e in ("png","pdf")})
def grid(ax):ax.grid(alpha=.2)
def mean(a,k):return st.mean(x[k] for x in a)
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
axs[0].bar(L,[x["plies"] for x in r["games"]]);axs[0].set(xlabel="Board side",ylabel="Natural-terminal game plies")
axs[1].bar(L,[r["counts"][f"L{x}"]["roots"]["jobs"] for x in L]);axs[1].set(xlabel="Board side",ylabel="Verified root histories")
fig.suptitle("Common MixNet native size-family pilot, one self game per size")
save("REPLAN-01",fig,"native_observed")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for law in laws:
 for LL in (13,15,21):
  a=[x for x in ref["temperature"] if x["L"]==LL and x["law"]==law];ks=sorted({x["k"] for x in a})
  axs[0].plot(ks,[mean([x for x in a if x["k"]==k],"H") for k in ks],marker=".",label=f"{law} L{LL}")
 for ax in [axs[1]]:
  a=[x for x in r["summary"] if x["law"]==law]
  ax.plot(L,[x["H_over_log_min"] for x in a],marker="o",label=law)
axs[0].set(xscale="log",xlabel="Completed backups",ylabel="Native occupancy entropy (nats)");axs[0].legend(fontsize=6,ncol=2)
axs[1].set(xlabel="Board side",ylabel="H / log(min(B, legal))");axs[1].legend()
for ax in axs:grid(ax)
fig.suptitle("T: fixed 8-stone geometry, varying density; score entropy not physical temperature")
save("REPLAN-02",fig,"native_observed")

fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,l in (("leading_error","Leading rate"),("one_loop_error","Lattice determinant")):
 a=[[v[k] for x in ref["loops"] for v in x["values"] if v["M"]==M] for M in (8,16,32)]
 axs[0].plot((8,16,32),[np.median(z) for z in a],marker="o",label=l);axs[0].fill_between((8,16,32),[min(z) for z in a],[max(z) for z in a],alpha=.15)
axs[0].set(xscale="log",yscale="log",xlabel="Independent restricted waves",ylabel="Absolute log-mass error");axs[0].legend(fontsize=8)
for law in laws:
 a=[x for x in ref["loops"] if x["law"]==law]
 axs[1].plot(L,[st.median(v["one_loop_error"] for x in a if x["L"]==LL for v in x["values"] if v["M"]==32) for LL in L],marker="o",label=law)
axs[1].set(xlabel="Board side",ylabel="M32 corrected log-mass error");axs[1].legend()
for ax in axs:grid(ax)
fig.suptitle("A: exact same-geometry 5-action frozen-wave references; not native adaptive 1PI")
save("REPLAN-03",fig,"exact_restricted_reference")

fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for i,LL in enumerate((15,21)):
 for j,law in enumerate(laws):
  a=[x for x in r["spatial"] if x["L"]==LL and x["law"]==law]
  z=np.mean([x["spatial_response"] for x in a],0).reshape(LL,LL);lim=max(abs(z).max(),1e-12)
  im=axs[i,j].imshow(z,cmap="RdBu_r",vmin=-lim,vmax=lim,origin="lower")
  axs[i,j].set_title(f"{law} L{LL}");fig.colorbar(im,ax=axs[i,j],shrink=.8)
fig.suptitle("R: spatial response of occupancy to local target logit; sum of entries = 0")
save("REPLAN-04.R",fig,"native_small_sample_spatial_response")

fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for law in laws:
 a=[x for x in r["summary"] if x["law"]==law]
 axs[0,0].plot(L,[x["cov_trace"] for x in a],marker="o",label=law)
 axs[0,1].plot(L,[x["S_low"] for x in a],marker="o",label=law)
 axs[1,0].plot(L,[x["central_mass"] for x in a],marker="o",label=law)
 axs[1,1].plot(L,[x["finite_response"][1][0] for x in a],marker="o",ls=":",label=law+" LR")
 axs[1,1].plot(L,[x["finite_response"][2][0] for x in a],marker="s",label=law+" direct")
for ax,y in zip(axs.ravel(),("Trace connected covariance","Lowest nonzero Fourier mode","Central 5x5 occupation mass","Target finite response")):
 ax.set(xlabel="Board side L",ylabel=y);ax.legend(fontsize=7);grid(ax)
fig.suptitle("C: genuine spatial-size pilot, B16, 2 fixed geometries, 4 histories per arm")
save("REPLAN-04.C",fig,"native_fixed_geometry_small_sample")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for law in laws:
 a=[x for x in r["conditional_records"] if x["law"]==law]
 for k,ls in (("Holevo","-"),("basis_MI",":")):
  axs[0].plot([1,2,4,8],[st.mean(v[k]/v["H"] for x in a for v in x["values"] if v["m"]==m and v["H"]>0) for m in (1,2,4,8)],marker="o",ls=ls,label=law+" "+k)
 a=[x for x in ph["data"] if x["law"]==law]
 axs[1].plot(L,[st.mean(x["values"][0]["QFI"] for x in a if x["L"]==LL) for LL in L],marker="o",label=law)
axs[0].set(xlabel="Declared environment fragments",ylabel="Information / pointer Shannon H");axs[0].legend(fontsize=7)
axs[1].set(xlabel="Board side",ylabel="Conditional phase QFI");axs[1].legend()
fig.suptitle("D: declared spatial sector record isometry and coherent phase model")
save("REPLAN-05",fig,"conditional_quantum_record")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for k,lab in (("work","Work"),("sigma","Entropy production"),("lag","Final lag KL")):
 axs[0].plot((1,4,16,64),[mean([x for x in ref["thermal"] if x["duration"]==t],k) for t in (1,4,16,64)],marker="o",label=lab)
axs[0].set(xscale="log",xlabel="Declared cycle duration",ylabel="Dimensionless energy / entropy");axs[0].legend(fontsize=8)
for law in laws:
 a=[x for x in ref["protocol"] if x["law"]==law and x["protocol"]=="constant:0.2"]
 axs[1].plot(L,[mean([x for x in a if x["L"]==LL],"weight") for LL in L],marker="o",label=law)
axs[1].axhline(1,color="grey",ls=":");axs[1].set(xlabel="Board side",ylabel="Actual path LR mean");axs[1].legend()
fig.suptitle("N: conditional four-sector thermal cycle (left); actual source LR (right)")
save("REPLAN-06",fig,"conditional_thermal_and_native_source_ratio")

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for law in laws:
 a=[x for x in ph["data"] if x["law"]==law]
 axs[0].plot((0,.3,.6),[st.mean(v["probe"] for x in a for v in x["values"] if v["theta"]==t) for t in (0,.3,.6)],marker="o",label=law)
 axs[1].plot(L,[st.mean(x["values"][1]["imag_norm"] for x in a if x["L"]==LL) for LL in L],marker="o",label=law)
axs[0].axhline(.25,color="grey",ls=":",label="dephased");axs[0].set(xlabel="Declared phase angle",ylabel="Coherent projector probability")
axs[1].set(xlabel="Board side",ylabel="Complex matrix imaginary norm")
for ax in axs:ax.legend(fontsize=8);grid(ax)
fig.suptitle("Q: explicit phase-sensitive conditional sector density, not native phase")
save("X-REP",fig,"conditional_complex_state")

fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for i,LL in enumerate((15,21)):
 for j,law in enumerate(laws):
  ax=axs[i,j]
  for kind,ls in (("early","-"),("late",":")):
   a=[x for x in r["responses"] if x["L"]==LL and x["law"]==law and x["kind"]==kind]
   v=np.mean([x["values"] for x in a],0)
   ax.plot(range(1,5),v[1,:,0],marker="o",ls=ls,label=kind+" LR")
   ax.plot(range(1,5),v[2,:,0],marker="s",ls=ls,label=kind+" direct")
  ax.axhline(0,color="grey",ls=":");ax.set(xlabel="Completed block",ylabel="Target finite response",title=f"{law} L{LL}");ax.legend(fontsize=6)
fig.suptitle("Memory: direct temporal source pulses, four seed groups per root; exact late-prefix control")
save("X-MEMORY",fig,"native_scheduled_source_pilot")
(out/"manifest.json").write_text(json.dumps(dict(source_sha256=h(S/"PHYSICS_RESULTS.json"),reference_sha256=h(S/"SPATIAL_REFERENCE_RESULTS.json"),phase_sha256=h(S/"PHASE_RESULTS.json"),groups=reg),indent=2)+"\n")
print("grouped figures",len(reg))
