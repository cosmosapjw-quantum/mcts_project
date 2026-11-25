"""Group source response and explicitly declared continuous thermal and quantum models."""
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import softmax
from .physics48 import cov
from .physics50 import sectors
from .observables import entropy

def group_record(trace,L):
    rep=trace["replicas"][0];sp=trace["spec"];law=sp["root_law"];B=sp["budget"]
    group=law["source_actions"];assert sp["source_action"] is None and sp["source_h"]==0 and law["proposal"]=="p"
    assert group==sorted(set(group)) and set(group)<set(trace["root"]["legal"])
    ev=sorted(rep["events"],key=lambda e:e["root_selection"]["clock"])
    assert len(ev)==B and B%4==0 and trace["root"]["board_size"]==L
    d=[e["root_selection"]["law"] for e in ev];a=[e["actions"][0] for e in ev]
    assert all(x["source_actions"]==group and x["source"] is None for x in d)
    I=np.isin(a,group).astype(int);p=np.array([np.sum(np.asarray(x["p"])[np.isin(x["actions"],group)]) for x in d])
    h=np.array([x["logit_h"] for x in d]);assert np.array_equal(h,law.get("logit_schedule",[law["logit_h"]]*B))
    m=np.array([e["root_value"] for e in ev]);X=np.column_stack((I.reshape(-1,4).mean(1),m.reshape(-1,4).mean(1)))
    assert all(max(e["root_backup_clock"] for e in ev[i-4:i])<ev[i]["root_selection"]["clock"] for i in range(4,B,4))
    counts=np.zeros(L*L)
    for i,n in trace["move_counts"]:counts[i]=n
    assert counts.sum()==B and counts[group].sum()==I.sum()
    pri=np.zeros(L*L)
    for x in rep["cuts"][0]["edges"]:pri[x["action"]]=x["prior"]
    assert abs(pri.sum()-1)<2e-6
    n=counts/B
    return dict(L=L,B=B,law=law["kind"],group=group,sector=int(sectors(L,[group[0]])[0]),seed=sp["seed"],I=I.tolist(),p=p.tolist(),G=(I-p).reshape(-1,4).sum(1).tolist(),h=h.tolist(),marks=m.tolist(),X=X.tolist(),counts=counts.astype(int).tolist(),prior=pri.tolist(),H=entropy(n[n>0]),occupancy=float(I.mean()),mark=float(m.mean()),action_history=a,root_log_prob=float(sum(x["log_p"] for x in d)),parent_queries=len(rep["evaluations"]),child_queries=sum(not s["terminal"] for e in rep["evaluations"] for s in e["samples"]),legal=trace["root"]["legal"])

def spatial_group(records):
    L=records[0]["L"];B=records[0]["B"];A=np.array([r["counts"] for r in records])/B
    C=cov(A,A);G=np.array([sum(r["G"]) for r in records]);R=cov(A,G).ravel()
    assert np.max(abs(C.sum(0)))<1e-10 and abs(R.sum())<1e-10
    f=np.fft.fft2((A-A.mean(0)).reshape(-1,L,L),axes=(-2,-1))
    S=(abs(f)**2).sum(0)/(len(A)-1);assert abs(S[0,0])<1e-12 and abs(S.sum()/L**2-np.trace(C))<1e-12
    group=records[0]["group"];assert all(r["group"]==group for r in records)
    r=float(R[group].sum());assert abs(r-cov(A[:,group].sum(1),G).item())<1e-10
    return dict(L=L,B=B,n=len(A),sector=records[0]["sector"],cov_trace=float(np.trace(C)),rank=int(np.linalg.matrix_rank(C,tol=1e-10)),S0=float(S[0,0]),S_low=float((S[0,1]+S[0,-1]+S[1,0]+S[-1,0])/4),H=float(np.mean([r["H"] for r in records])),group_response=r,response=R.tolist(),structure_factor=S.tolist())

def tilted_cycle(probs,tau,method="DOP853",rtol=2e-10,atol=2e-12):
    """Continuous-time four-state reset generator, normalized s in [0,1].
    Column tilted vector g'=tau(pi 1^T-I)g-lambda E' g; E=-log pi,
    beta E*=1. Exact Jarzynski solution g_lambda=1(s)=pi(s)."""
    p=np.asarray(probs,float);assert p.shape==(3,4) and np.min(p)>0
    p=p/p.sum(1)[:,None];z=np.log(np.vstack((p,p[0])))
    lamb=np.array([0.,.25,.5,.75,1.])
    def run(z):
        g=np.tile(np.exp(z[0]),(5,1));mu=g[0].copy();W=0.;je_path=0.;mass=0.
        for j in range(3):
            d=z[j+1]-z[j]
            def rhs(u,y):
                pi=softmax(z[j]+u*d);v=d-pi@d
                a=y[:20].reshape(5,4)
                da=(tau/3)*(a.sum(1)[:,None]*pi-a)+lamb[:,None]*v*a
                return np.r_[da.ravel(),-a[0]@v]
            sol=solve_ivp(rhs,(0,1),np.r_[g.ravel(),0.],method=method,rtol=rtol,atol=atol)
            assert sol.success
            for u,y in zip(sol.t,sol.y.T):
                q=softmax(z[j]+u*d);a=y[:20].reshape(5,4)
                je_path=max(je_path,float(np.max(abs(a[-1]-q))))
                mass=max(mass,abs(a[0].sum()-1))
            g=sol.y[:20,-1].reshape(5,4);W+=sol.y[-1,-1]
        return g.sum(1),W,je_path,mass
    a,W,je,mass=run(z);b,Wr,jr,mr=run(z[::-1])
    err=max(abs(a-b[::-1]));assert err<1e-8 and max(je,jr,mass,mr)<1e-8
    return dict(tau=tau,lambdas=lamb.tolist(),forward=a.tolist(),reverse=b.tolist(),W=W,W_reverse=Wr,JE=float(a[-1]),crooks_error=float(err),je_path_error=max(je,jr),mass_error=max(mass,mr),method=method)

def quantum_noncommuting(n1,n2,alpha=.35):
    """Declared S:E two-qubit unitary, coherent entanglement witness and
    sequential environment nonselective measurements. Native counts
    specify angles only; no claim of native entanglement."""
    from .conditional import density,von_neumann
    I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
    Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.])
    a,b=alpha*n1,alpha*n2
    plus=np.array([1.,1.])/np.sqrt(2);minus=np.array([1.,-1.])/np.sqrt(2);zero=np.array([1.,0.]);one=np.array([0.,1.])
    psi=np.cos(a)*np.kron(plus,zero)-1j*np.sin(a)*np.kron(minus,one)
    rho=density(np.outer(psi,psi.conj()));R=rho.reshape(2,2,2,2)
    S=np.trace(R,axis1=1,axis2=3);E=np.trace(R,axis1=0,axis2=2)
    # Direct optimal CHSH from singular directions of correlation matrix.
    pa=[X,Y,Z];T=np.array([[np.trace(rho@np.kron(x,y)).real for y in pa] for x in pa])
    u,s,vh=np.linalg.svd(T);norm=np.hypot(s[0],s[1]);ang=math.atan2(s[1],s[0])
    op=lambda v:sum(x*y for x,y in zip(v,pa))
    A,A1=op(u[:,0]),op(u[:,1]);b0=op(np.cos(ang)*vh[0]+np.sin(ang)*vh[1]);b1=op(np.cos(ang)*vh[0]-np.sin(ang)*vh[1])
    ch=float(np.trace(rho@(np.kron(A,b0+b1)+np.kron(A1,b0-b1))).real)
    ref=2*np.sqrt(1+np.sin(2*a)**2);assert abs(ch-ref)<1e-10 and abs(ch-2*norm)<1e-10
    # Matched dephasing in Schmidt basis gives separable state.
    deph=np.cos(a)**2*np.outer(np.kron(plus,zero),np.kron(plus,zero))+np.sin(a)**2*np.outer(np.kron(minus,one),np.kron(minus,one))
    Td=np.array([[np.trace(deph@np.kron(x,y)).real for y in pa] for x in pa]);sd=np.linalg.svd(Td,compute_uv=False);cd=2*np.hypot(sd[0],sd[1])
    assert cd<=2+1e-10
    # Condition on first S Y outcome; environment remains coherent.
    # Nonselective X or Z intervention then the same S reset/readout.
    zx=np.kron(Z,X);U=np.cos(b)*np.eye(4)-1j*np.sin(b)*zx;ps=np.outer(plus,plus)
    q={}
    for name,o in (("none",None),("X",X),("Z",Z),("XZ",Z),("ZX",X)):
        arr=np.zeros((2,2))
        for j,y in enumerate((1,-1)):
            yy=np.array([1.,y*1j])/np.sqrt(2);e=np.einsum("i,iajb,j->ab",yy.conj(),R,yy)
            def dephase(e,o):return (e+o@e@o)/2
            if name=="XZ":e=dephase(dephase(e,X),Z)
            elif name=="ZX":e=dephase(dephase(e,Z),X)
            elif o is not None:e=dephase(e,o)
            rr=U@np.kron(ps,e)@U.conj().T;ss=np.trace(rr.reshape(2,2,2,2),axis1=1,axis2=3)
            for k,y0 in enumerate((1,-1)):
                v=np.array([1.,y0*1j])/np.sqrt(2);arr[j,k]=np.real(v.conj()@ss@v)
        assert abs(arr.sum()-1)<1e-10
        q[name]=float(arr[0,0]+arr[1,1]-arr[0,1]-arr[1,0])
    c=np.sin(2*a)*np.sin(2*b);err=max(abs(q["none"]-c),abs(q["X"]-c),abs(q["Z"]),abs(q["XZ"]),abs(q["ZX"]))
    assert err<1e-10
    return dict(n1=n1,n2=n2,CHSH=ch,dephased_CHSH=float(cd),correlations=q,entropy_S=von_neumann(density(S)),entropy_E=von_neumann(density(E)),residual=float(err)),rho
