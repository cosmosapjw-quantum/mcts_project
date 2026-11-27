"""Native selective 3-wave record and explicit distinct pair-restricted diagnostic dilation."""
from instrument81 import *
from itertools import product
H=[''.join(map(str,b)) for b in product(range(2),repeat=3)]
HAD=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2);HX=np.kron(np.kron(HAD,HAD),HAD)
HAM=np.array([[sum(a!=b for a,b in zip(h,k)) for k in H] for h in H])
def norm(A):return float(np.max(abs(A)))
def sq(A):
 e,v=np.linalg.eigh(A);assert e.min()>-1e-12
 return (v*np.sqrt(np.maximum(e,0)))@v.conj().T
def tr(A):return float(np.trace(A).real)
def td(A,B):return float(.5*sum(abs(np.linalg.eigvalsh(A-B))))
def tv(a,b):return float(.5*sum(abs(np.array(a)-np.array(b))))
def entropy(A):
 e=np.linalg.eigvalsh(A);assert e.min()>-1e-10
 return float(-e[e>0]@np.log(e[e>0]))
def reduced(A,axis):
 v=A.reshape([2]*6);axes=[i for i in range(3) if i!=axis];perm=[axis]+axes+[axis+3]+[i+3 for i in axes]
 v=v.transpose(perm).reshape(2,4,2,4)
 a=np.einsum('abcb->ac',v);b=np.einsum('abad->bd',v)
 pt=v.transpose(2,1,0,3).reshape(8,8);en=np.linalg.eigvalsh(pt)
 return dict(negativity=float(-en[en<0].sum()),QMI=entropy(a)+entropy(b)-entropy(A),first_entropy=entropy(a),rest_entropy=entropy(b))
def setup(nodes,protocol,mode,frozen=False):
 out={}
 for h in sorted(nodes,key=lambda h:(len(h),h)):
  t=32+4*len(h);m,r=nodes['0'*len(h) if frozen else h];assert 0<m<1 and 0<r<1
  axis,lam,theta,eta,erase=schedule(mode,t);U=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)])
  if erase:eta=0.
  F=[np.sqrt((1+eta)/2)*U,np.sqrt((1-eta)/2)*Z@U]
  Ka,Kb,_,_,_=operators(m,r,axis,lam)
  L=[np.linalg.matrix_power(K,4) if protocol=='native_repeat' else K/np.sqrt(m) for K in (Ka,Kb)]
  eff=sum(adj(K)@K for K in L);D=I-eff;minfail=float(np.linalg.eigvalsh(D).min());assert minfail>-1e-12
  if protocol=='pair_diagnostic':assert norm(D)<1e-12;fail=np.zeros((2,2),complex)
  else:fail=sq(D)
  assert norm(eff+adj(fail)@fail-I)<1e-12
  out[h]=dict(F=F,L=L,fail=fail,minfail=minfail,complete=norm(eff+adj(fail)@fail-I))
 return out
def evaluate(nodes,protocol,mode='plus81',frozen=False,keep=False):
 op=setup(nodes,protocol,mode,frozen)
 # Lexicographic bath ordering, shared bath convention across the record branches.
 M={'':np.eye(2,dtype=complex)[None,:,:]};abort={};mass={'':1.};abort_prob={};cause=0.;efftotal=np.zeros((2,2),complex)
 for h,v in op.items():
  old=M[h];F=v['F'];q=[np.array([k@f@p for p in old for f in F]) for k in v['L']+[v['fail']]]
  for i,K in enumerate(q[:2]):
   M[h+str(i)]=K;mass[h+str(i)]=sum(tr(a@RHO0@adj(a)) for a in K)
  abort[h]=q[2];abort_prob[h]=sum(tr(a@RHO0@adj(a)) for a in q[2])
  cause=max(cause,abs(mass[h]-mass[h+'0']-mass[h+'1']-abort_prob[h]))
  efftotal+=sum(adj(a)@a for a in q[2])
 for h in H:efftotal+=sum(adj(a)@a for a in M[h])
 complete=norm(efftotal-I);assert complete<1e-12 and cause<1e-12
 A=np.array([M[h] for h in H]);R=np.einsum('hbik,kl,gbil->hg',A,RHO0,A.conj())
 success=tr(R);assert success>0
 p=np.array([mass[h] for h in H]);assert norm(R.diagonal().real-p)<1e-12 and abs(success+sum(abort_prob.values())-1)<1e-12
 r=R/success;assert norm(r-adj(r))<1e-12 and np.linalg.eigvalsh(r).min()>-1e-12
 de=np.diag(p/success).astype(complex);read=np.diag(HX@r@HX.T).real
 variants=[]
 for xi in (0.,.5,1.):
  C=r*xi**HAM;assert np.linalg.eigvalsh(C).min()>-1e-12 and norm(C.diagonal()-r.diagonal())<1e-12 and abs(tr(C)-1)<1e-12
  read0=np.diag(HX@C@HX.T).real
  variants.append(dict(overlap=xi,readout=read0.tolist(),readout_TV=tv(read0,de.diagonal().real*0+np.ones(8)/8),trace_distance=td(C,de),splits=[reduced(C,i) for i in range(3)]))
 v=dict(success=success,abort=1-success,leaf_prob=p.tolist(),conditional_prob=(p/success).tolist(),prefix_mass=mass,abort_by_prefix=abort_prob,completeness_error=complete,causality_error=cause,node_completeness=max(x['complete'] for x in op.values()),min_abort_eig=min(x['minfail'] for x in op.values()),conditional_min_eig=float(np.linalg.eigvalsh(r).min()),conditional_entropy=entropy(r),diagonal_entropy=entropy(de),offdiag_l1=float(abs(r-de).sum()),conditional_trace_distance=td(r,de),conditional_Hadamard_TV=tv(read,np.ones(8)/8),unconditional_trace_distance=success*td(r,de),unconditional_Hadamard_TV=success*tv(read,np.ones(8)/8),splits=[reduced(r,i) for i in range(3)],overlap_controls=variants)
 return v,r
def main():
 a=read(S/'AUDIT83.json');assert sha(S/'TREE83.json')==a['tree_sha'];q=read(S/'TREE83.json');out=[];mat=[]
 for c in q['contexts']:
  nodes=c['nodes'];actual=np.array([l['probability'] for l in c['leaves']])
  # Variation of actual m,r settings across siblings at equal depth; pair law not branch-frozen.
  spread=[np.ptp(np.array([v for h,v in nodes.items() if len(h)==dep]),axis=0).tolist() for dep in (1,2)]
  row=dict(family=c['family'],core=c['core'],settings_spread=spread,protocols={});rr=[]
  for protocol in ('native_repeat','pair_diagnostic'):
   v,r=evaluate(nodes,protocol);f,rf=evaluate(nodes,protocol,frozen=True)
   minus,rm=evaluate(nodes,protocol,'minus81');ep,re=evaluate(nodes,protocol,'eraseplus81');em,rem=evaluate(nodes,protocol,'eraseminus81');base,rb=evaluate(nodes,protocol,'base81')
   assert norm(re-rem)<1e-12 and abs(ep['success']-em['success'])<1e-12
   if protocol=='native_repeat':
    p=np.array(v['leaf_prob']);v['native_leaf_relative_error']=float(max(abs(p/actual-1)));assert v['native_leaf_relative_error']<1e-10
   else:assert abs(v['success']-1)<1e-12 and max(abs(v0['success']-1) for v0 in (f,minus,ep,em,base))<1e-12
   v['frozen_probability_TV']=tv(v['conditional_prob'],f['conditional_prob']);v['frozen_trace_distance']=td(r,rf);v['frozen_success']=f['success']
   # TV of stopped distributions including 7 explicit abort-prefix labels (not just global abort).
   stop=lambda a:np.r_[a['leaf_prob'],[a['abort_by_prefix'][h] for h in sorted(nodes)]]
   v['frozen_stopped_TV']=tv(stop(v),stop(f));v['phase_reversal_TV']=tv(v['conditional_prob'],minus['conditional_prob']);v['phase_reversal_trace_distance']=td(r,rm);v['phase_reversal_stopped_TV']=tv(stop(v),stop(minus));v['phase_erasure_TV']=tv(v['conditional_prob'],ep['conditional_prob']);v['erase_sign_TV']=tv(ep['conditional_prob'],em['conditional_prob']);v['base_TV']=tv(v['conditional_prob'],base['conditional_prob']);v['minus_success']=minus['success'];v['erased_success']=ep['success'];v['base_success']=base['success']
   row['protocols'][protocol]=v;rr.extend([r,rf,rm,re,rb])
  row['protocol_difference_TV']=tv(row['protocols']['native_repeat']['conditional_prob'],row['protocols']['pair_diagnostic']['conditional_prob'])
  out.append(row);mat.append(rr)
 np.savez_compressed(S/'RECORD_MATRICES83.npz',meta=np.array([[r['family'],r['core']] for r in out]),R=np.array(mat),labels=np.array([p+'_'+k for p in ('native_repeat','pair_diagnostic') for k in ('adaptive','frozen','minus','erase','base')]))
 dump(S/'RECORD83.json',dict(contexts=out,matrix_sha=sha(S/'RECORD_MATRICES83.npz'),tree_sha=a['tree_sha'],scope='Native repeat event conditional H32 with exact survival and explicit abort; pair-restricted diagnostic distinct. Shared bath labels and uncomputed native workspace declared, not inferred. Full stopped state block diagonal between success and abort flags.'))
 for g in range(3):
  for p in ('native_repeat','pair_diagnostic'):
   z=[x['protocols'][p] for x in out if x['family']==g]
   print(FAMS[g],p,'success median',float(np.median([x['success'] for x in z])),'TV',np.mean([x['conditional_Hadamard_TV'] for x in z]),'unconditional TV',np.mean([x['unconditional_Hadamard_TV'] for x in z]),'freeze TV',np.mean([x['frozen_probability_TV'] for x in z]),'negative',sum(max(k['negativity'] for k in x['splits'])>1e-10 for x in z),flush=True)
if __name__=='__main__':main()
