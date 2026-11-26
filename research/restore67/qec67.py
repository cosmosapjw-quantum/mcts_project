from base61 import *
from itertools import product
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Z=np.diag([1,-1]);Y=np.array([[0,-1j],[1j,0]])
def kron(xs):
 a=np.array([[1]],complex)
 for x in xs:a=np.kron(a,x)
 return a
V=np.eye(8,dtype=complex)[:,[0,7]];dag=lambda x:x.conj().T
g1=kron([Z,Z,I]);g2=kron([I,Z,Z]);masks=list(product([0,1],repeat=3))
P=[];C=[]
for mask in [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]:
 a,b,c=mask;P.append((np.eye(8)+(-1)**(a^b)*g1)@(np.eye(8)+(-1)**(b^c)*g2)/4)
 C.append(kron([X if i else I for i in mask]))
assert np.max(abs(sum(P)-np.eye(8)))==0
def recover(K):
 return [dag(V)@c@p@k@V for k in K for p,c in zip(P,C)]
def channel(K,r):return sum(k@r@dag(k) for k in K)
def metrics(K):
 tp=float(abs(sum(dag(k)@k for k in K)-I).max());choi=sum(np.outer(k.ravel(),k.ravel().conj()) for k in K)
 assert tp<1e-12 and np.linalg.eigvalsh(choi).min()>-1e-12
 return float(sum(abs(np.trace(k))**2 for k in K)/4),tp
def logical(p,mu):
 return (1-mu)*(3*p*p-2*p**3)+mu*p
def main():
 rng=np.random.default_rng(670101);rows=[];maxerr=0.;tpmax=0.
 for p in np.linspace(0,.5,51):
  for mu in (0.,.5,1.):
   K=[]
   for mask in masks:
    w=(1-mu)*p**sum(mask)*(1-p)**(3-sum(mask))+mu*(1-p if sum(mask)==0 else p if sum(mask)==3 else 0.)
    K.append(np.sqrt(w)*kron([X if i else I for i in mask]))
   L=recover(K);fe,tp=metrics(L);tpmax=max(tpmax,tp);err=abs(fe-(1-logical(p,mu)));maxerr=max(maxerr,err)
   assert err<1e-12
   rows.append(dict(p=float(p),mu=mu,Fe=fe,pL=logical(p,mu),uncoded=1-p))
 coh=[]
 for angle in np.linspace(0,.6,31):
  c=np.cos(angle);s=np.sin(angle);p=s*s;L=recover([kron([c*I-1j*s*X]*3)]);fe,tp=metrics(L);tpmax=max(tpmax,tp)
  K=[np.sqrt(1-logical(p,0))*I,np.sqrt(logical(p,0))*X];err=0.
  for r in [(I+a)/2 for a in (X,Y,Z,-X,-Y,-Z)]:
   ref=(1-logical(p,0))*r+logical(p,0)*X@r@X-2j*c**3*s**3*(X@r-r@X);err=max(err,float(abs(channel(L,r)-ref).max()))
  assert err<1e-12
  def repeat(k):
   # entanglement fidelity via trace of PTM for trace-preserving channel.
   a=np.array([[np.trace(b@channel(k,r)).real/2 for r in (I,X,Y,Z)] for b in (I,X,Y,Z)])
   return [(j,float(np.trace(np.linalg.matrix_power(a,j))/4)) for j in (1,10,100)]
  coh.append(dict(angle=float(angle),p=float(p),Fe=fe,logical_coherent_coefficient=float(-2*c**3*s**3),formula_error=err,repeated_coherent=repeat(L),repeated_pauli=repeat(K)))
 z=recover([kron([Z,I,I])]);assert abs(metrics(z)[0])<1e-14
 # Independent finite noise draws, full syndrome decoding; bit-flip model only.
 cal=[]
 for mu in (0.,.5,1.):
  n=2000000;p=.12
  flips=rng.random((n,3))<p;common=rng.random(n)<mu;flips[common]=flips[common,:1]
  sy=flips[:,0]^flips[:,1];sz=flips[:,1]^flips[:,2];rec=flips[:,0]^(sy&~sz);major=flips.sum(1)>=2
  assert np.array_equal(rec,major)
  rate=float(rec.mean());ex=logical(p,mu);assert abs(rate-ex)<5*np.sqrt(ex*(1-ex)/n)
  cal.append(dict(mu=mu,n=n,logical=rate,expected=ex,physical=float(flips.mean()),syndrome=float((sy|sz).mean()),success_conditional=float((~rec[sy|sz]).mean()) if (sy|sz).any() else None))
 dump(S/'QEC67.json',dict(scope='explicit auxiliary 3-qubit bit flip quantum code, ideal syndrome/recovery; calibration software only',stochastic=rows,coherent=coh,phase_Z_Fe=metrics(z)[0],max_formula_error=maxerr,max_tp=tpmax,calibration=cal,physical_code_distance=1,restricted_bit_flip_distance=3))
 print('QEC',maxerr,tpmax,cal,flush=True)
if __name__=='__main__':main()
