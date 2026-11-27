"""Full-history empirical 1PI and controlled multivariate lattice local expansion."""
from base61 import *
from scipy.special import logsumexp
from scipy.optimize import root,minimize_scalar
from scipy.fft import rfft2,irfft2
from itertools import product
from math import gcd,comb
import time
FAMILIES=['g0','g3','Rapfi'];NREP=[8,16,32,64,128,256];ANCHORS=np.array([[0.,0.],[2.,-2.],[-2.,2.],[4.,4.],[-4.,-4.]])
GRID=np.array(list(product([-4.,-2.,0.,2.,4.],repeat=2))+list(product([-8.,8.],repeat=2)))
def cgf(X,J,logp=None):
 if logp is None:logp=np.full(len(X),-np.log(len(X)))
 l=logp+X@J;W=float(logsumexp(l));q=np.exp(l-W);mu=q@X;d=X-mu;H=(d*q[:,None]).T@d
 return W,mu,H,q
def source_test(X,M):
 # Uniform root mixture with equal number of seeds. Pointwise geometry-bootstrap on finite grid.
 a=X[(M[:,0]==0)];b=X[(M[:,0]==1)&(M[:,2]<32)];c=X[(M[:,0]==1)&(M[:,2]>=32)]
 m0=M[M[:,0]==0,2];m1=M[(M[:,0]==1)&(M[:,2]<32),2];m2=M[(M[:,0]==1)&(M[:,2]>=32),2]
 assert len(a)==len(b)==256 and len(c)==128
 rng=np.random.default_rng(800019);i0=rng.integers(0,32,size=(2000,32));i2=rng.integers(0,16,size=(2000,16));i1=rng.integers(0,32,size=(2000,32))
 out=[]
 for J in GRID:
  zs=[]
  for D in (a,b,c):
   W,mu,H,q=cgf(D,J);eig=np.linalg.eigvalsh(H)
   assert min(eig)>0
   Gamma=float(J@mu-W);KL=float(q@np.log(q*len(q)));assert abs(Gamma-KL)<1e-10 and Gamma>=-1e-12
   inv=np.linalg.inv(H);assert abs(inv@H-np.eye(2)).max()<1e-10
   # Inverse gradient solution and inverse susceptibility finite difference at image.
   sol=root(lambda j:cgf(D,j)[1]-mu,np.zeros(2),jac=lambda j:cgf(D,j)[2],tol=1e-10)
   assert np.max(abs(sol.fun))<1e-9
   delta=np.linalg.norm(sol.x-J);assert delta<1e-6
   zs.append(dict(W=W,phi=mu.tolist(),H=H.tolist(),inverse_H=inv.tolist(),gamma=Gamma,KL=KL,eig=eig.tolist(),ess=float(1/(q@q)),dual_error=delta))
  A=np.array([np.exp(a[m0==i]@J).mean() for i in range(32)]);B1=np.array([np.exp(b[m1==i]@J).mean() for i in range(32)]);C=np.array([np.exp(c[m2==i]@J).mean() for i in range(32,48)])
  ab=np.log(B1[i0].mean(1))-np.log(A[i0].mean(1));ac=np.log(C[i2].mean(1))-np.log(A[i1].mean(1))
  out.append(dict(J=J.tolist(),train=zs[0],matched=zs[1],novel=zs[2],matched_delta_W=zs[1]['W']-zs[0]['W'],novel_delta_W=zs[2]['W']-zs[0]['W'],matched_bootstrap=np.quantile(ab,[.025,.975]).tolist(),novel_bootstrap=np.quantile(ac,[.025,.975]).tolist()))
 return out
def lattice_index(Y):
 D=np.unique(Y,axis=0);D=D-D[0];k=0
 for i in range(len(D)):
  for j in range(i):
   k=gcd(k,abs(int(D[i,0]*D[j,1]-D[i,1]*D[j,0])))
   if k==1:return 1
 return k
def coeff(Y,q,H,mu):
 eig,V=np.linalg.eigh(H);Z=(Y-mu)@((V/np.sqrt(eig))@V.T)
 assert abs((Z*q[:,None]).T@Z-np.eye(2)).max()<1e-9
 def mom(k):return 0. if k%2 else float(math.prod(range(1,k,2)))
 c3=np.array([comb(3,i)*np.dot(q,Z[:,0]**i*Z[:,1]**(3-i)) for i in range(4)])
 c4=np.array([comb(4,i)*np.dot(q,Z[:,0]**i*Z[:,1]**(4-i)) for i in range(5)])
 c4[[0,2,4]]-=np.array([3.,6.,3.])
 e4=sum(c4[i]*mom(i)*mom(4-i) for i in range(5))
 s=np.convolve(c3,c3);e33=sum(s[i]*mom(i)*mom(6-i) for i in range(7))
 return float(e4/24-e33/72),float(e4),float(e33)
def alias(Y,q,M,k,shape):
 # Each other lattice point contributing to wrapped mass lies in one of four tails.
 logp=np.log(q);l=[];lo=Y.min(0);hi=Y.max(0)
 for i in range(2):
  for sign in [-1,1]:
   t=sign*k[i]+shape[i];v=sign*Y[:,i]
   if t>M*v.max():l.append(float('-inf'));continue
   fun=lambda u:float(M*logsumexp(logp+u*v)-u*t)
   opt=minimize_scalar(fun,bounds=(0.,8.),method='bounded',options={'xatol':1e-8})
   # Any chosen positive u is a valid Chernoff bound; optimizer need not be exact.
   assert opt.x>0; l.append(min(0.,fun(opt.x)))
 return float(np.exp(logsumexp(l)))
def torus(Y,q,M,k,shape):
 a=np.zeros(shape,float);ix=np.mod(Y,np.array(shape));np.add.at(a,(ix[:,0],ix[:,1]),q);assert abs(a.sum()-1)<1e-12
 F=rfft2(a,workers=1);F**=M;z=irfft2(F,s=shape,workers=1);p=float(z[k[0]%shape[0],k[1]%shape[1]])
 return p,float(z.min()),float(abs(z.sum()-1))
def local(Y,M,J0):
 # Normalized Y/B coordinates for anchor, integer coordinates for exact saddle.
 assert lattice_index(Y)==1
 logp=np.full(len(Y),-np.log(len(Y)));_,mu0,_,_=cgf(Y,J0/128,logp);k=np.rint(M*mu0).astype(int);target=k/M
 sol=root(lambda s:cgf(Y,s,logp)[1]-target,J0/128,jac=lambda s:cgf(Y,s,logp)[2],tol=1e-11)
 s=sol.x;K,mu,H,q=cgf(Y,s,logp);res=float(abs(mu-target).max())
 if res>1e-8 or min(np.linalg.eigvalsh(H))<=1e-10:return dict(status='SADDLE_NOT_RESOLVED',M=M,J0=J0.tolist(),k=k.tolist(),residual=res)
 C,e4,e33=coeff(Y,q,H,mu);lead=float(1/(2*np.pi*M*np.sqrt(np.linalg.det(H))));nextp=lead*(1+C/M)
 shape=tuple(2**int(np.ceil(np.log2(max(32,10*np.sqrt(M*H[i,i])+8)))) for i in range(2));attempt=0
 while True:
  assert max(shape)<=8192 and math.prod(shape)<=2**26,(M,shape)
  p,neg,norm=torus(Y,q,M,k,shape);bound=alias(Y,q,M,k,shape)
  if p>0 and bound<=p*1e-12:break
  shape=tuple(2*q0 for q0 in shape);attempt+=1
  if attempt>3:return dict(status='ALIAS_OR_SUPPORT_BLOCKED',M=M,J0=J0.tolist(),k=k.tolist(),wrapped=p,alias_bound=bound)
 gamma=float(s@target-K)
 assert p>0 and neg>-1e-10 and norm<1e-9
 return dict(status='OK',M=M,J0=J0.tolist(),k=k.tolist(),s=s.tolist(),K=K,mu=mu.tolist(),H=H.tolist(),gamma=gamma,tilt_ESS=float(1/(q@q)),saddle_residual=res,coeff=C,gaussian_e4=e4,gaussian_e33=e33,shape=list(shape),wrapped_mass=p,alias_bound=bound,fft_min=neg,fft_normalization_error=norm,log_mass=float(-M*gamma+np.log(p)),leading=lead,corrected=nextp,leading_rel=lead/p-1,corrected_rel=nextp/p-1,exact_action_per_replica=gamma-np.log(p)/M,leading_action_per_replica=gamma-np.log(lead)/M,corrected_action_per_replica=gamma-np.log(nextp)/M if nextp>0 else None)
def main():
 t=time.monotonic();a=read(S/'AUDIT80.json');assert sha(S/'MARKED80.npz')==a['array_sha']
 with np.load(S/'MARKED80.npz') as z:D={k:z[k] for k in z.files}
 M=D['meta'];X=D['X'];Y=D['Y'];rows=[];sources=[];summ=[]
 for g in range(3):
  i=M[:,1]==g;s=source_test(X[i],M[i]);sources.append(dict(family=FAMILIES[g],cases=s))
  for cohort in (0,1):
   ix=i&(M[:,0]==cohort)&(M[:,2]<32);y=Y[ix];assert len(y)==256
   idx=lattice_index(y);summ.append(dict(family=FAMILIES[g],cohort=cohort,distinct_support=len(np.unique(y,axis=0)),lattice_index=idx,range_min=y.min(0).tolist(),range_max=y.max(0).tolist()))
   if idx!=1:continue
   for n in NREP:
    for J in ANCHORS:
     r=local(y,n,J);r.update(family=FAMILIES[g],cohort=cohort);rows.append(r)
    print('LATTICE',FAMILIES[g],cohort,n,'elapsed',time.monotonic()-t,flush=True)
 dump(S/'ACTION80.json',dict(scope='Empirical whole-history finite 1PI and bivariate independent-replica local lattice mass; fixed finite law not true population or physical hbar',sources=sources,lattice_laws=summ,saddles=rows,anchors=ANCHORS.tolist(),M=NREP,source_grid=GRID.tolist(),input_sha=a['array_sha'],seconds=time.monotonic()-t))
 print('DONE',len(rows),sum(r['status']=='OK' for r in rows),flush=True)
if __name__=='__main__':main()
