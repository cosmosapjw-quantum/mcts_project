"""Exact classical syndrome posteriors and full quantum logical instruments."""
from base61 import *
from itertools import product
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]).astype(complex);P=[I,X,Y,Z]
V=np.eye(8,dtype=complex)[:,[0,7]]
def kron(a):return np.kron(np.kron(a[0],a[1]),a[2])
masks=list(product((0,1),repeat=3));reps=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
def synd(m):return m[0]^m[1],m[1]^m[2]
SYN=[synd(m) for m in reps]
E=[kron([X if b else I for b in m]) for m in masks];G1=kron([Z,Z,I]);G2=kron([I,Z,Z]);Pj=[(np.eye(8)+(-1)**a*G1)@(np.eye(8)+(-1)**b*G2)/4 for a,b in SYN];Cj=[kron([X if b else I for b in m]) for m in reps]
dag=lambda x:x.conj().T
assert np.array_equal(sum(Pj),np.eye(8)) and all(np.array_equal(p@p,p) for p in Pj)
assert all(np.array_equal(V.conj().T@dag(c)@d@V,I if i==j else np.zeros((2,2))) for i,c in enumerate(Cj) for j,d in enumerate(Cj))
assert np.array_equal(dag(V)@kron([Z,I,I])@V,Z)
def wlaw(p,mu):
 return np.array([(1-mu)*p**sum(m)*(1-p)**(3-sum(m))+mu*((1-p) if sum(m)==0 else p if sum(m)==3 else 0.) for m in masks])
def post(p,mu):
 w=wlaw(p,mu);assert abs(w.sum()-1)<1e-12;out=[];maj=opt=entropy=0.
 for r,s in zip(reps,SYN):
  k=masks.index(r);q=masks.index(tuple(1-b for b in r));a,b=w[k],w[q];z=a+b
  maj+=b;opt+=min(a,b)
  if z>0:
   conf=b/z if r==(0,0,0) else 1.
   h=-sum(t*np.log(t) for t in [a/z,b/z] if t>0);entropy+=z*h
   out.append(dict(syndrome=s,prob=float(z),fault_posterior=float(conf),majority_residual=float(b/z),map_residual=float(min(a,b)/z),majority_success=float(a/z)))
  else:out.append(dict(syndrome=s,prob=0.,fault_posterior=None,majority_residual=None,map_residual=None,majority_success=None))
 he=-sum(t*np.log(t) for t in w if t>0);hs=-sum(x['prob']*np.log(x['prob']) for x in out if x['prob']>0)
 assert abs(he-hs-entropy)<1e-12
 assert abs(maj-((1-mu)*(3*p*p-2*p**3)+mu*p))<1e-12
 if p<=.5:assert abs(opt-maj)<1e-12
 return dict(p=p,mu=mu,syndromes=out,majority_error=float(maj),map_error=float(opt),single_error=p,conditional_mask_entropy=float(entropy),syndrome_information=float(hs),mask_entropy=float(he),undetected_fault=float(w[-1]))
def channel(K,r):return sum(k@r@dag(k) for k in K)
def metrics(K):
 tp=float(abs(sum(dag(k)@k for k in K)-I).max());choi=sum(np.outer(k.ravel(),k.ravel().conj()) for k in K);ev=np.linalg.eigvalsh(choi)
 assert tp<1e-10 and ev.min()>-1e-10
 A=np.array([[np.trace(b@channel(K,a)).real/2 for a in P] for b in P]);return A,tp,float(ev.min())
def instr(t):
 c,s=np.cos(t),np.sin(t);U=kron([c*I-1j*s*X]*3)
 return [dag(V)@C@p@U@V for C,p in zip(Cj,Pj)]
def inv(K,t):
 W=instr(t);out=[]
 for a,b in zip(K,W):
  q=float(np.trace(dag(b)@b).real/2)
  # Undefined zero-probability branch under estimated model: choose identity, explicitly record fallback.
  out.append(dag(b)@a/np.sqrt(q) if q>1e-28 else a)
 return out
def fid(A,n):return float(np.trace(np.linalg.matrix_power(A,n))/4)
def main():
 posteriors=[post(p,mu) for p in (0.,.01,.05,.12,.25,.5,.75,1.) for mu in (0.,.5,1.)]
 out=[];err=0.;tp=0.;min_ev=0.;ind_err=0.
 for t in np.linspace(0,.6,31):
  c,s=np.cos(t),np.sin(t);K=instr(t);theory=[c**3*I+1j*s**3*X]+[-1j*s*c*c*I-s*s*c*X]*3
  e=float(max(abs(a-b).max() for a,b in zip(K,theory)));err=max(err,e);assert e<1e-12
  qs=[float(np.trace(dag(a)@a).real/2) for a in K]
  for a,q in zip(K,qs):assert abs(dag(a)@a-q*I).max()<1e-12
  p=s*s;pl=3*p*p-2*p**3;twirl=[np.sqrt(1-pl)*I,np.sqrt(pl)*X]
  arms={'ordinary':K,'known_angle':inv(K,t),'estimate_plus':inv(K,t+.03),'estimate_minus':inv(K,t-.03),'pauli':twirl}
  r={}
  for name,A in arms.items():
   M,er,ev=metrics(A);tp=max(tp,er);min_ev=min(min_ev,ev)
   r[name]={str(n):fid(M,n) for n in (1,10,100)}
   if name=='known_angle':assert abs(M-np.eye(4)).max()<1e-10
   # Independently compute complex transverse eigenvalue from aI+bX, in X eigenbasis.
   a=np.array([np.trace(k)/2 for k in A]);b=np.array([np.trace(k@X)/2 for k in A]);lam=np.sum((a+b)*np.conj(a-b))
   for n in (1,10,100):
    f=(1+(lam**n).real)/2;ind_err=max(ind_err,abs(f-r[name][str(n)]));assert abs(f-r[name][str(n)])<1e-10
  out.append(dict(theta=float(t),syndrome_prob=qs,conditional_unitary=True,logical_flip=float(pl),results=r))
 dump(S/'QEC89.json',dict(status='PASS_SCOPED',posteriors=posteriors,coherent=out,max_formula_error=err,max_tp=tp,min_choi_eigenvalue=min_ev,independent_repeated_fidelity_error=ind_err,scope='Exact ideal auxiliary quantum repetition channel and classical syndrome MAP; known-angle inverse assumes exact coherent noise calibration, not arbitrary-error QEC or native device.'))
 print('QEC',len(posteriors),len(out),err,tp,ind_err,flush=True)
if __name__=='__main__':main()
