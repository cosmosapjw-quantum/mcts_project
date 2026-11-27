"""Independent raw branch reduction, spectral Kraus products, record partial trace and source audit."""
from base61 import *
import gzip,zipfile,itertools
A=read(S/'AUDIT83.json');T=read(S/'TREE83.json');R=read(S/'RECORD83.json');assert sha(S/'TREE83.json')==A['tree_sha']==R['tree_sha'] and sha(S/'RECORD_MATRICES83.npz')==R['matrix_sha']
with np.load(S/'RECORD_MATRICES83.npz') as z:D={k:z[k] for k in z.files}
I=np.eye(2,dtype=complex);x=np.array([[0,1],[1,0]],complex);y=np.array([[0,-1j],[1j,0]],complex);z=np.diag([1.,-1.]).astype(complex);rho=(I+.8*x)/2
H=list(itertools.product(range(2),repeat=3));err=collections.defaultdict(float)
def chk(k,a,b,tol=1e-10):
 v=float(abs(np.asarray(a)-np.asarray(b)).max());err[k]=max(err[k],v);assert v<tol,(k,v)
def sq(a):
 e,v=np.linalg.eigh(a);assert e.min()>-1e-12
 return (v*np.sqrt(np.maximum(e,0)))@v.conj().T
def trace(a):return float(np.trace(a).real)
def adj(a):return a.conj().T
def F(dep,bit,mode):
 th=[np.pi/2,.37,-.51][dep]
 if dep==0 and mode=='minus':th=-th
 eta=0. if dep==0 and mode=='erase' else .9
 U=np.diag([np.exp(-.5j*th),np.exp(.5j*th)])
 return np.sqrt((1+(-1)**bit*eta)/2)*(I if bit==0 else z)@U
def effect(m,r,dep,bit,mode):
 ax=[y,z,x][dep];lam=0. if mode=='base' else .65
 E=m*((1-lam)*r*I+lam*(I+ax)/2)
 return E if bit==0 else m*I-E
def wave(m,r,dep,bit,mode,protocol):
 E=effect(m,r,dep,bit,mode);K=sq(E)
 return K@K@K@K if protocol=='native_repeat' else sq(E/m)
def negate(R,axis):
 pt=np.zeros((8,8),complex)
 for a,h in enumerate(H):
  for b,k in enumerate(H):
   u=list(h);v=list(k);u[axis],v[axis]=v[axis],u[axis]
   pt[H.index(tuple(u)),H.index(tuple(v))]=R[a,b]
 ev=np.linalg.eigvalsh(pt);return float(-ev[ev<0].sum())
def independent(c,protocol,mode='plus',frozen=False):
 nodes=c['nodes'];M=[]
 for h in H:
  q=[]
  for b in H:
   a=I.copy()
   for d in range(3):
    key='0'*d if frozen else ''.join(map(str,h[:d]));m,r=nodes[key]
    a=wave(m,r,d,h[d],mode,protocol)@F(d,b[d],mode)@a
   q.append(a)
  M.append(q)
 A=np.array(M);R=np.zeros((8,8),complex)
 for i in range(8):
  for j in range(8):R[i,j]=sum(np.trace(a@rho@adj(b)) for a,b in zip(A[i],A[j]))
 # Independent density recursion and trace-decrease complement at each retained prefix.
 pre={'':rho};abort={};total={}
 for d in range(3):
  for key in [k for k in pre if len(k)==d]:
   m,r=nodes['0'*d if frozen else key];state=sum(F(d,b,mode)@pre[key]@adj(F(d,b,mode)) for b in range(2))
   for b in (0,1):
    K=wave(m,r,d,b,mode,protocol);pre[key+str(b)]=K@state@adj(K)
   abort[key]=trace(pre[key])-trace(pre[key+'0'])-trace(pre[key+'1']);assert abort[key]>-1e-12
 p=np.array([trace(pre[''.join(map(str,h))]) for h in H]);s=p.sum()
 chk('relative_diagonal',R.diagonal().real/p,np.ones(8),1e-10)
 chk('stopped_mass',s+sum(abort.values()),1.)
 return R/s,s,p,abort
raw=0;branches=0;matrices=0
for n,(c,r) in enumerate(zip(T['contexts'],R['contexts'])):
 assert [c['family'],c['core']]==D['meta'][n].tolist()==[r['family'],r['core']]
 for l in c['leaves']:
  p=B/l['path'];assert sha(p)==l['sha256']
  with gzip.open(p,'rt') as f:tr=json.load(f)['trace']
  es=tr['events'];a=rho.copy();lp=0.
  for t in range(32,44):
   d=(t-32)//4;law=es[t]['root_selection']['law'];sc=np.asarray(law['scores']);ps=np.exp((10/3)*(sc-sc.max()));ps/=ps.sum();ac=np.array(law['actions']);ia,ib=[int(np.flatnonzero(ac==v)[0]) for v in c['pair']];m=ps[ia]+ps[ib];rr=ps[ia]/m;key=''.join(map(str,l['bits'][:d]));chk('raw_node_settings',[m,rr],c['nodes'][key])
   if t%4==0:a=sum(F(d,b,'plus')@a@adj(F(d,b,'plus')) for b in range(2))
   K=sq(effect(m,rr,d,l['bits'][d],'plus'));a=K@a@adj(K);p0=trace(a)
   lp+=math.log(law['p'][law['picked']]);chk('raw_log_mass',np.log(p0),lp)
   v=[trace(a@q)/p0 for q in (x,y,z)];chk('raw_state',v,l['state_after'][t-32])
  chk('raw_leaf_relative_mass',trace(a)/l['probability'],1.)
  raw+=1
 for j,protocol in enumerate(('native_repeat','pair_diagnostic')):
  q=r['protocols'][protocol]
  for k,(mode,frozen) in enumerate([('plus',False),('plus',True),('minus',False),('erase',False),('base',False)]):
   a,s,p,abort=independent(c,protocol,mode,frozen);chk('record_matrix',a,D['R'][n,j*5+k],1e-9);matrices+=1
   if mode=='plus' and not frozen:
    chk('success_relative',s/q['success'],1.);chk('leaf_relative',p/np.array(q['leaf_prob']),np.ones(8));chk('abort_prefix',[abort[h] for h in sorted(abort)],[q['abort_by_prefix'][h] for h in sorted(abort)])
    for ax in range(3):chk('partial_transpose',negate(a,ax),q['splits'][ax]['negativity'])
    branches+=8
assert raw==576 and branches==1152 and matrices==720
# Reauthenticate original R69 archive and relevant source spans.
O79=B/'mqft_restore79_20261127';O82=B/'mqft_restore82_20261127';O69=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O82/f)==sha(O69/f),f
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'};assert all(sha(paths[k])==v for k,v in orig['archives'].items())
ids=['L15','L16','L20','Q24','Q25','Q35','Q37','Q42','Q43','Q45'];spans=[s for s in orig['spans'] if s['id'] in ids];assert len(spans)==len(ids)
want={s['member'] for s in spans}|{'mcts_study/v3.0/qft-mcts-decoherence.md','mcts_study/effective-action-mcts.md','mcts_study/v5.0/new_quantum_mcts.md'}
members=[]
for s in orig['verified_members']:
 if s['member'] not in want:continue
 with zipfile.ZipFile(paths[s['archive']]) as f:v=f.read(s['member'])
 assert hashlib.sha256(v).hexdigest()==s['sha256'];members.append(s)
assert {s['member'] for s in members}==want
m='mcts_study/v6.0/quantum_mcts_foundation.md'
with zipfile.ZipFile(paths['mcts_study.zip']) as f:v=f.read(m)
h=hashlib.sha256(v).hexdigest();members.append(dict(archive='mcts_study.zip',member=m,sha256=h))
dump(S/'ORIGINAL_SOURCE83.json',dict(archives=orig['archives'],verified_members=members,spans=spans,original_attachment_receipt=orig['original_attachment_receipt'],scope='Original quantum record/entanglement/action and information objectives; actual branch-dependent model, not physical phase identification.'))
dump(S/'INDEPENDENT83.json',dict(status='PASS_SCOPED',raw_forced_paths=raw,independent_leaf_branches=branches,independent_record_matrices=matrices,errors=dict(err),scope='Separate spectral square roots, unnormalized fine CP branch products from all actual forced raw paths; explicit bath cross-trace, density recursion, partial transpose index map. No independent native subagent.'))
print('VERIFIED',raw,branches,matrices,dict(err),flush=True)
