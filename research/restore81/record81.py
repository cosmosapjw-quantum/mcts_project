"""Explicit frozen three-operation coherent record dilation; not actual adaptive coherent histories."""
from instrument81 import *
import itertools
from scipy.linalg import sqrtm
H=list(itertools.product(range(3),repeat=3));BATH=list(itertools.product(range(2),repeat=3))
def negativity(r,da,db):
 e=np.linalg.eigvalsh(r.reshape(da,db,da,db).transpose(2,1,0,3).reshape(da*db,da*db))
 return float(-e[e<0].sum())
def entropy(r):
 e=np.linalg.eigvalsh(r);assert e.min()>-1e-10;return float(-e[e>0]@np.log(e[e>0]))
def ops(info,mode):
 out=[]
 for t,(m,r) in zip((32,36,40),info):
  axis,lam,th,eta,erase=schedule(mode,t)
  U=np.diag([np.exp(-.5j*th),np.exp(.5j*th)])
  if erase:eta=0.
  Fs=[np.sqrt((1+eta)/2)*U,np.sqrt((1-eta)/2)*Z@U]
  A,B,O,_,_=operators(m,r,axis,lam)
  out.append(([A,B,O],Fs))
 return out
def joint(ops0,initial,blind=False):
 states={():initial}
 prefix=[]
 for K,F in ops0:
  nd={}
  for h,r in states.items():
   v=sum(f@r@adj(f) for f in F)
   if blind:v=np.diag(np.diag(v))
   for i,k in enumerate(K):nd[h+(i,)]=k@v@adj(k)
  states=nd;prefix.append(np.array([np.trace(v).real for v in states.values()]))
 p=np.array([np.trace(states[h]).real for h in H]);return p,prefix
def coherent(ops0,initial):
 # A chosen common bath label across qutrit record alternatives.
 mats=[]
 for b in BATH:
  m=np.array([ops0[2][0][h[2]]@ops0[2][1][b[2]]@ops0[1][0][h[1]]@ops0[1][1][b[1]]@ops0[0][0][h[0]]@ops0[0][1][b[0]] for h in H])
  mats.append(m)
 sq=np.linalg.eigh(initial);rt=(sq[1]*np.sqrt(sq[0]))@sq[1].conj().T
 # Gram over bath, final system and initial purification.
 a=np.transpose(np.asarray(mats)@rt,(1,0,2,3)).reshape(27,-1)
 return a@a.conj().T
def independent_isometry(ops0,initial):
 # Recursively add record/environment indices; independently form final density.
 a=np.eye(2,dtype=complex)[None,None,:,:]
 for K,F in ops0:
  blocks=[]
  for k in K:
   blocks.append(np.concatenate([np.einsum('ij,abjk->abik',k@f,a) for f in F],axis=1))
  a=np.stack(blocks,axis=1).reshape(a.shape[0]*3,a.shape[1]*2,2,2)
 return np.einsum('aeij,jk,beik->ab',a,initial,a.conj())
def one(info,g,c):
 o=ops(info,'plus81');p,pre=joint(o,RHO0);r=coherent(o,RHO0);r2=independent_isometry(o,RHO0);assert abs(r-r2).max()<1e-11
 assert np.min(np.linalg.eigvalsh(r))>-1e-11 and abs(np.trace(r)-1)<1e-12 and abs(r-r.conj().T).max()<1e-12
 assert abs(r.diagonal().real-p).max()<1e-12 and abs(p.sum()-1)<1e-12
 causal=max(abs(p.reshape(3,3,3).sum((1,2))-pre[0]).max(),abs(p.reshape(3,3,3).sum(2).ravel()-pre[1]).max())
 assert causal<1e-12
 pm,_=joint(ops(info,'minus81'),RHO0);ep,_=joint(ops(info,'eraseplus81'),RHO0);em,_=joint(ops(info,'eraseminus81'),RHO0);sw,_=joint(ops(info,'swap81'),RHO0);bl,_=joint(o,RHO0,blind=True)
 assert abs(ep-em).max()<1e-12
 de=np.diag(p).astype(complex);n=negativity(r,3,9);n2=negativity(de,3,9);assert n2<1e-12
 # First vs rest quantum information for explicit record partition.
 ra=np.einsum('abcb->ac',r.reshape(3,9,3,9));rb=np.einsum('abad->bd',r.reshape(3,9,3,9));mi=entropy(ra)+entropy(rb)-entropy(r)
 # Noncommutation of the actual-conditioned a effects at three frozen settings.
 eff=[adj(q[0][0])@q[0][0] for q in o];cm=[float(np.linalg.norm(eff[i]@eff[j]-eff[j]@eff[i])) for i,j in [(0,1),(1,2),(0,2)]]
 tv=lambda x,y:float(.5*abs(x-y).sum())
 return dict(family=g,core=c,baseinfo=info.tolist(),trace_error=float(abs(np.trace(r)-1)),hermiticity=float(abs(r-r.conj().T).max()),min_eig=float(np.linalg.eigvalsh(r).min()),diagonal_error=float(abs(r.diagonal().real-p).max()),isometry_error=float(abs(r-r2).max()),causal_error=float(causal),joint_prob=p.tolist(),phase_reverse_TV=tv(p,pm),phase_erase_TV=tv(p,ep),erase_sign_TV=tv(ep,em),swap_TV=tv(p,sw),all_readout_dephase_TV=tv(p,bl),negativity=n,dephased_record_negativity=n2,record_QMI=mi,record_entropy=entropy(r),offdiagonal_l1=float(abs(r-de).sum()),effect_commutator_norm=cm)
def main():
 a=read(S/'AUDIT81.json');assert sha(S/'PATHS81.npz')==a['array_sha']
 with np.load(S/'PATHS81.npz') as z:d={k:z[k] for k in ('meta','baseinfo')}
 out=[]
 for g in range(3):
  for c in range(24):
   i=np.flatnonzero(np.all(d['meta']==[g,c,0,5],axis=1));assert len(i)==1
   out.append(one(d['baseinfo'][i[0]],g,c))
 assert len(out)==72
 dump(S/'RECORD81.json',dict(scope='Explicit three-time frozen instrument using actual base history snapshots t32/36/40. Each of 27 grouped outcomes is a declared frozen-model record; bath label convention specified. Not a native adaptive coherent record reconstruction; actual conditional instrument in AUDIT81.',contexts=out,model='three qutrit records, final qubit and shared per-time bath labels; trace final qubit and bath',input_sha=a['array_sha']))
 print('RECORD',len(out),'max errors',max(x['isometry_error'] for x in out),max(x['causal_error'] for x in out),'negative',sum(x['negativity']>1e-10 for x in out),'max negativity',max(x['negativity'] for x in out),flush=True)
if __name__=='__main__':main()
