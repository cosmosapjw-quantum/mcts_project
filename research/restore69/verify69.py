from records69 import *
from scipy.linalg import sqrtm
def isom(K):return np.stack(K,axis=1).reshape(6,2)
def explicit(K,L,r,phi,mode):
 v=isom(K);x=v@r@v.conj().T
 if mode=='actual_rotation':V=np.diag([np.exp(-.5j*phi),np.exp(.5j*phi)])
 else:V=I
 if mode=='commuting':ks=[I]
 elif mode=='pointer_dephased':ks=[U.conj().T@np.diag([1,0])@U,U.conj().T@np.diag([0,1])@U]
 else:ks=[np.sqrt((1+eta)/2)*V,np.sqrt((1-eta)/2)*Z@V]
 a=sum(np.kron(k,np.eye(3))@x@np.kron(k,np.eye(3)).conj().T for k in ks)
 # tensor order (S,R2,R1), swap to (S,R1,R2)
 Q=np.kron(isom(L),np.eye(3));b=(Q@a@Q.conj().T).reshape(2,3,3,2,3,3).transpose(0,2,1,3,5,4)
 R=np.einsum('aijakl->ijkl',b).reshape(9,9)
 # direct last-pointer projection produces explicit separable decomposition
 ar=a.reshape(2,3,2,3);t=np.einsum('ab,bicj,cd->aidj',U,ar,U.conj().T)
 diag=[t[z,:,z,:] for z in range(2)]
 lam=np.array([[np.trace(k@U.conj().T@np.diag([int(z==0),int(z==1)])@U).real for k in L] for z in range(2)])
 dec=sum(np.kron(diag[z],np.outer(lam[z],lam[z])) for z in range(2))
 assert min(np.linalg.eigvalsh(diag[z]).min() for z in range(2))>-1e-11
 err=float(abs(dec-R).max());assert err<1e-12
 A=np.einsum('aibi->ab',R.reshape(3,3,3,3));B=np.einsum('aiaj->ij',R.reshape(3,3,3,3))
 p=np.diag(R).real.reshape(3,3);vals=[eigen_entropy(A)+eigen_entropy(B)-eigen_entropy(R),H(p.sum(0))+H(p.sum(1))-H(p.ravel()),neg(R,3,3)]
 return vals,err
def bell(r):
 T=np.array([[np.trace(r@np.kron(a,b)).real for b in PAULI] for a in PAULI]);_,v=np.linalg.eigh(T.T@T)
 b0,b1=v[:,-1],v[:,-2];t0,t1=np.linalg.norm(T@b0),np.linalg.norm(T@b1)
 if min(t0,t1)<1e-14:return chsh(r),0
 norm=np.hypot(t0,t1);a0=T@b0/t0;a1=T@b1/t1;u=(t0*b0+t1*b1)/norm;w=(t0*b0-t1*b1)/norm
 obs=lambda v:np.einsum('i,iab->ab',v,PAULI)
 B=np.kron(obs(a0),obs(u)+obs(w))+np.kron(obs(a1),obs(u)-obs(w))
 actual=np.trace(r@B).real;return float(actual),float(abs(actual-chsh(r)))
def main():
 with np.load(S/'NATIVE_RECORDS69.npz') as z:na=z['native'];rp=z['rapfi']
 with np.load(S/'MULTITIME69.npz') as z:cross=z['cross']
 checks=[];errs=[];bchecks=[];rng=np.random.default_rng(690311)
 jobs=[(na[i,0,:3],na[i,:,3:6]) for i in range(len(na))]+[(np.array([0,.8,0]),x) for x in rp]
 for i in [*range(0,256,8),*range(256,288,4)]:
  v,pars=jobs[i]
  for k,mode in enumerate(('actual_rotation','no_rotation','commuting','pointer_dephased')):
   state=mat(v)
   for t,(m,r,phi) in enumerate(pars):
    if t:state=free(state,phi,mode)
    K=operators(m,r)
    for _ in range(3):state=channel(K,state)
    if t in (0,7,15,30):
     m2,r2,phi2=pars[t+1];L=operators(m2,r2);q,e=explicit(K,L,state,phi2,mode)
     err=float(abs(np.asarray(q)-cross[i,k,t,3:6]).max());assert err<1e-10
     errs.append([e,err]);checks.append((i,k,t))
    state=channel(K,state)
 with np.load(S/'RECORD_STATES69.npz') as z:
  d=z['fresh_R69_d'];c=z['fresh_R69_c'];s=z['fresh_R69_overlap'];target=z['fresh_R69_CHSH'][:,3]
  choose=np.r_[rng.choice(len(d),64,replace=False),rng.choice(np.flatnonzero(target>2),32,replace=False)]
  for i in choose:
   r=compact(np.array(d[i]),np.array(c[i]),np.array(s[i]),4,4);v,err=bell(r);assert err<1e-10 and abs(v-target[i])<1e-10;bchecks.append(err)
 report=dict(explicit_crosswave_cases=len(errs),max_separable_decomposition_error=float(np.max(errs,axis=0)[0]),max_pair_metric_error=float(np.max(errs,axis=0)[1]),explicit_CHSH_operator_checks=len(bchecks),max_CHSH_operator_error=float(max(bchecks)))
 dump(S/'INDEPENDENT69.json',report);print(report,flush=True)
if __name__=='__main__':main()
