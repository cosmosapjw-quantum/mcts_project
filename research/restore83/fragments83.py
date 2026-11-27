"""Exploratory pointer-information readout on explicit adaptive record, not native Darwinism."""
from base61 import *
import itertools
H=list(itertools.product(range(2),repeat=3))
def ent(p):
 p=np.asarray(p);p=p[p>1e-16];return float(-p@np.log(p))
def vn(a):
 e=np.linalg.eigvalsh(a);assert min(e)>-1e-12;return ent(e)
def red(a,keep):
 rest=[i for i in range(2) if i not in keep];v=a.reshape([2]*4).transpose(keep+rest+[i+2 for i in keep]+[i+2 for i in rest]).reshape(2**len(keep),2**len(rest),2**len(keep),2**len(rest))
 return np.einsum('abcb->ac',v)
R=read(S/'RECORD83.json');assert sha(S/'RECORD_MATRICES83.npz')==R['matrix_sha']
with np.load(S/'RECORD_MATRICES83.npz') as z:D={k:z[k] for k in z.files}
rows=[]
for j,c in enumerate(R['contexts']):
 for n,p in enumerate(('native_repeat','pair_diagnostic')):
  r=D['R'][j,5*n];# measured first bit; condition remaining qubits on that readout
  prob=np.array([np.trace(r[i*4:(i+1)*4,i*4:(i+1)*4]).real for i in range(2)])
  assert min(prob)>0
  v=[r[i*4:(i+1)*4,i*4:(i+1)*4]/prob[i] for i in range(2)]
  h=ent(prob);q=[]
  for keep in ([0],[1],[0,1]):
   f=[red(a,keep) for a in v];avg=prob[0]*f[0]+prob[1]*f[1];chi=vn(avg)-sum(prob[i]*vn(f[i]) for i in range(2));mi=ent(avg.diagonal().real)-sum(prob[i]*ent(f[i].diagonal().real) for i in range(2))
   assert -1e-12<=mi<=chi+1e-12 and chi<=h+1e-12
   q.append(dict(fragment=[i+2 for i in keep],holevo=chi,classical_MI=mi,holevo_fraction=chi/h,classical_fraction=mi/h))
  rows.append(dict(family=c['family'],core=c['core'],protocol=p,pointer_entropy=h,fragments=q,nontrivial=h>1e-6,disjoint_90_Holevo=bool(h>1e-6 and min(a['holevo_fraction'] for a in q[:2])>=.9),disjoint_90_classical=bool(h>1e-6 and min(a['classical_fraction'] for a in q[:2])>=.9)))
dump(S/'FRAGMENTS83.json',dict(scope='Secondary exploratory finite pointer readout of first wave bit and two later single-bit fragments in chosen model. Native repeat conditioned on entire success, pair diagnostic different intervention. No physical environment redundancy/objectivity or accessible-information equality with Holevo.',contexts=rows))
for g in range(3):
 for p in ('native_repeat','pair_diagnostic'):
  q=[x for x in rows if x['family']==g and x['protocol']==p];print(g,p,'pointer H median',np.median([x['pointer_entropy'] for x in q]),'mean fractions',np.mean([[a['holevo_fraction'] for a in x['fragments']] for x in q],0),'90%',sum(x['disjoint_90_Holevo'] for x in q),sum(x['disjoint_90_classical'] for x in q),flush=True)
