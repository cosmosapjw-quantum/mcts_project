"""Explicit representation/protocol diagnostics, not inferred quantum physics.

Adapted from RESTORE35--37: validate positive complex states, partial trace,
Holevo versus basis information, and finite Gibbs reset balance. The caller
must supply independently declared state/model/protocol inputs. No automatic
phase lift or physical energy identification from visit counts.
"""
from itertools import combinations
import math
import numpy as np
from .observables import entropy

def density(r,tol=1e-9):
    r=np.asarray(r,complex)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or not np.isfinite(r).all() or np.max(abs(r-r.conj().T))>tol or abs(np.trace(r)-1)>tol:
        raise ValueError("invalid density matrix")
    if np.linalg.eigvalsh(r).min()<-tol:raise ValueError("non-positive density matrix")
    return (r+r.conj().T)/2

def von_neumann(r):
    e=np.linalg.eigvalsh(density(r));p=np.maximum(e,0);p/=p.sum()
    return entropy(p)

def reduce(r,dims,keep):
    dims=tuple(dims);keep=list(keep)
    if any(type(d)!=int or d<1 for d in dims) or not keep or len(set(keep))!=len(keep) or any(not 0<=i<len(dims) for i in keep):raise ValueError("partition")
    r=density(r);n=len(dims);rest=[i for i in range(n) if i not in keep]
    if math.prod(dims)!=len(r):raise ValueError("factor dimension")
    d=math.prod(dims[i] for i in keep);k=math.prod(dims[i] for i in rest)
    t=r.reshape(*dims,*dims).transpose(keep+rest+[n+i for i in keep+rest])
    return density(np.trace(t.reshape(d,k,d,k),axis1=1,axis2=3))

def information(r,dims,fragment):
    """System is axis 0, fixed basis; quantum MI, Holevo, basis MI separated."""
    if not fragment or 0 in fragment:raise ValueError("record fragment")
    joint=reduce(r,dims,[0]+list(fragment));d=dims[0];n=len(joint)//d
    s=reduce(joint,[d,n],[0]);f=reduce(joint,[d,n],[1]);t=joint.reshape(d,n,d,n)
    p=np.maximum(s.diagonal().real,0);p/=p.sum()
    ss,sf,sj=map(von_neumann,(s,f,joint))
    chi=sf-sum(p[i]*von_neumann(t[i,:,i,:]/p[i]) for i in range(d) if p[i]>1e-12)
    a=np.maximum(joint.diagonal().real.reshape(d,n),0);a/=a.sum()
    basis=entropy(a.sum(0))+entropy(a.sum(1))-entropy(a.ravel())
    qmi=ss+sf-sj
    if min(qmi,chi,basis)<-1e-8 or basis>chi+1e-8 or chi>entropy(p)+1e-8 or chi>qmi+1e-8:raise ArithmeticError("information inequalities")
    return dict(S_system=ss,S_fragment=sf,S_joint=sj,QMI=max(0,qmi),Holevo=max(0,chi),basis_MI=max(0,basis),H_pointer=entropy(p),coherence_l1=float(np.sum(abs(s-np.diag(s.diagonal())))),imag_norm=float(np.linalg.norm(joint.imag)))

def finite_reset(r_sm,ds,energies,beta,theta):
    """RESTORE37 explicit equal-H memory/bath partial swap. Q into SM.
    Hamiltonian units and beta are declared, no native energy meaning.
    Analytic reduced states avoid materializing the full SM-bath matrix.
    """
    e=np.asarray(energies,float);n=len(e);r=density(r_sm)
    if n<2 or ds*n!=len(r) or not np.isfinite(e).all() or not math.isfinite(beta) or beta<0 or not math.isfinite(theta):raise ValueError("bath contract")
    x=-beta*(e-e.min());tau=np.exp(x);tau/=tau.sum()
    if tau.min()<=0:raise ValueError("unresolved Gibbs support")
    t=np.diag(tau);s=reduce(r,[ds,n],[0]);m=reduce(r,[ds,n],[1]);op=np.kron(np.eye(ds),t)
    c,v=np.cos(theta),np.sin(theta)
    sm=density(c*c*r+v*v*np.kron(s,t)+1j*c*v*(r@op-op@r))
    b=density(c*c*t+v*v*m+1j*c*v*(t@m-m@t))
    mf=reduce(sm,[ds,n],[1])
    q=float(e@(mf.diagonal().real-m.diagonal().real));qb=float(e@(b.diagonal().real-tau))
    ds_sm=von_neumann(sm)-von_neumann(r);sb=von_neumann(b);info=von_neumann(sm)+sb-von_neumann(r)-entropy(tau)
    db=-sb-float(b.diagonal().real@np.log(tau));sigma=ds_sm-beta*q
    if abs(q+qb)>1e-8 or abs(sigma-info-db)>1e-8 or min(info,db,sigma)<-1e-8:raise ArithmeticError("finite bath balance")
    return dict(Q_into_SM=q,delta_S_SM=ds_sm,Sigma=sigma,I_SM_B=info,D_B_Gibbs=db,first_law_residual=q+qb,balance_residual=sigma-info-db)

def render_model(contract_path,output):
    """Input JSON points at a hash-pinned NPZ of explicitly supplied densities.
    This is a conditional model renderer, not a native MCTS analysis.
    """
    import json
    from pathlib import Path
    from .client import sha256
    from .campaign import atomic_json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p=Path(contract_path);c=json.loads(p.read_text())
    if c.get("schema")!="mqft.declared_density.v1" or not c.get("model") or not c.get("measure") or not c.get("basis") or not c.get("provenance"):
        raise ValueError("explicit model/measure/basis/provenance required")
    f=p.parent/c["npz"]
    if sha256(f)!=c["sha256"]:raise ValueError("state identity")
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    data=[];dims=c["dims"]
    with np.load(f,allow_pickle=False) as z:
        for row in c["states"]:
            r=density(z[row["key"]]);v=[]
            for j in range(1,len(dims)):
                v.append(dict(fragment_count=j,**information(r,dims,list(range(1,j+1)))))
            item=dict(parameter=row["parameter"],state=row["key"],metrics=v)
            if "bath" in c:
                bath=c["bath"]
                if len(dims)!=2 or not bath.get("energy_unit"):raise ValueError("two-part state and explicit energy units required")
                item["reset"]=[dict(theta=theta,**finite_reset(r,dims[0],bath["energies"],bath["beta"],theta)) for theta in bath["theta"]]
            data.append(item)
    if not data or not any(x["metrics"] for x in data):raise ValueError("no eligible density states")
    fig,axs=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for r in data:
        m=r["metrics"];x=[v["fragment_count"] for v in m]
        axs[0].plot(x,[v["QMI"] for v in m],label=str(r["parameter"]),marker="o")
        axs[1].plot(x,[v["Holevo"] for v in m],marker="o")
        axs[1].plot(x,[v["basis_MI"] for v in m],ls="--")
    axs[0].set(ylabel="Quantum mutual information (nats)",xlabel="Cumulative fragments")
    axs[1].set(ylabel="Holevo / basis MI (dashed)",xlabel="Cumulative fragments")
    axs[0].legend(fontsize=8);fig.suptitle("Declared representation only: "+c["model"])
    for ext in ("png","pdf"):fig.savefig(out/("D_X_declared_information."+ext),dpi=160)
    plt.close(fig)
    if "bath" in c:
        fig,axs=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
        for row in data:
            r=row["reset"];x=[v["theta"] for v in r]
            axs[0].plot(x,[v["Q_into_SM"] for v in r],label=str(row["parameter"]))
            axs[1].plot(x,[v["Sigma"] for v in r],label=str(row["parameter"]))
            axs[1].plot(x,[v["I_SM_B"]+v["D_B_Gibbs"] for v in r],ls="--")
        axs[0].set(xlabel="Declared swap angle",ylabel="Heat into SM ("+c["bath"]["energy_unit"]+")")
        axs[1].set(xlabel="Declared swap angle",ylabel="Entropy balance (nats)")
        axs[0].legend(fontsize=8);fig.suptitle("Conditional finite bath; Sigma and I+D (dashed)")
        for ext in ("png","pdf"):fig.savefig(out/("N_declared_finite_bath."+ext),dpi=160)
        plt.close(fig)
    atomic_json(out/"model_metrics.json",dict(contract_sha256=sha256(p),state_sha256=c["sha256"],scope="conditional model, not native quantum identification",results=data))
    return dict(status="conditional_model",states=len(data))
