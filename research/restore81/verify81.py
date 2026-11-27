"""Independent unnormalized CP branch products; audit original source and convergence."""
from base61 import *
import gzip,zipfile,csv
A=read(S/'AUDIT81.json');R=read(S/'EFFECTS81.json');V=read(S/'RECORD81.json');U=read(S/'RNG_AUDIT81.json')
assert A['array_sha']==R['input_sha']==V['input_sha']==sha(S/'PATHS81.npz')
assert U['streams']==384 and U['independent_uniforms']==49152 and U['max_error']<1e-12
with np.load(S/'PATHS81.npz') as z:D={k:z[k] for k in z.files}
I=np.eye(2,dtype=complex);x=np.array([[0,1],[1,0]],complex);y=np.array([[0,-1j],[1j,0]],complex);z=np.diag([1.,-1.]).astype(complex)
M=D['meta'];arms=R['arms'];err=collections.defaultdict(float)
def chk(k,a,b,tol=1e-10):
 v=float(abs(np.asarray(a)-np.asarray(b)).max());err[k]=max(err[k],v);assert v<tol,(k,v)
def sq(A):
 v,w=np.linalg.eigh(A);assert min(v)>-1e-13
 return (w*np.sqrt(np.maximum(v,0)))@w.conj().T
def branch(raw,mode,src):
 # Unnormalized branch state; no normalized step updates in this implementation.
 rho=(I+.8*x)/2;lp=0.;lp0=0.;increments=[];states=[]
 for t in range(32,44):
  law=raw[t]['root_selection']['law'];ac=np.asarray(law['actions']);sc=np.array(law['scores']);base=np.exp((10/3)*(sc-sc.max()));base/=base.sum()
  a,b=[int(np.flatnonzero(ac==s)[0]) for s in src];pick=law['picked'];m=base[a]+base[b];r=base[a]/m
  if t in (32,36,40):
   theta=((-1 if mode in ('minus81','eraseminus81') else 1)*np.pi/2) if t==32 else .37 if t==36 else -.51
   phase=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)]);rho=phase@rho@phase.conj().T
   eta=0. if t==32 and mode.startswith('erase') else .9
   rho[0,1]*=eta;rho[1,0]*=eta
  axis=y if t<36 else (x if mode=='swap81' else z) if t<40 else (z if mode=='swap81' else x)
  lam=0. if mode=='base81' else .65
  E=m*((1-lam)*r*I+lam*(I+axis)/2)
  K=sq(E) if pick==a else sq(m*I-E) if pick==b else np.sqrt(base[pick])*I
  rho=K@rho@K.conj().T;prob=float(np.trace(rho).real);assert prob>0
  new=np.log(prob);lp0+=np.log(base[pick]);increments.append(new-lp-np.log(base[pick]));lp=new
  states.append([float(np.trace(rho@s).real/prob) for s in (x,y,z)])
 return lp-lp0,np.array(increments),np.array(states)
n=0
for i,a in enumerate(A['raw_receipts']):
 if a['rep']!=0:continue
 f=B/a['path'];assert sha(f)==a['sha256']
 with gzip.open(f,'rt') as h:tr=json.load(h)['trace']
 es=tr['events'];src=a['sources']
 for j,mode in enumerate(arms):
  lp,inc,states=branch(es,mode,src);chk('branch_log_ratio',lp,D['ratios'][i,:,j].sum(),1e-9);chk('conditional_increments',inc,D['ratios'][i,:,j],1e-9)
  if j==a['arm']:chk('branch_final_and_intermediate_state',states,D['states'][i],1e-9)
 n+=1
assert n==432
# Recompute all native pair contrasts using explicit indexed root blocks.
index={tuple(r):i for i,r in enumerate(M)}
for g,f in enumerate(R['families']):
 vals=[];weights=[];cost=[]
 for c in range(24):
  a=[];b=[]
  for rep in range(8):
   ix=[index[g,c,rep,j] for j in range(6)]
   act=D['actions'][ix];src=D['sources'][ix];sig=(act==src[:,0,None]).astype(float)-(act==src[:,1,None]).astype(float)
   a.append(np.column_stack([sig[:,32],sig[:,32:44].mean(1),D['marks'][ix].mean(1)]))
   b.append(np.exp(D['ratios'][ix[5]].sum(0)))
  vals.append(np.array(a).mean(0));weights.append(np.array(b).mean(0))
 vals=np.array(vals);means=vals.mean(0);weights=np.array(weights)
 chk('weight_means',weights.mean(0),f['importance_weight_ci'][0])
 for j,arm in enumerate(arms):chk('native_means',means[j],np.array(f['arm_mean_ci'][arm][0])[[0,1,7]])
 for c in f['comparisons']:
  i,j=map(arms.index,c['arms']);chk('native_contrast',means[i]-means[j],np.array(c['direct_ci'][0])[[0,1,7]])
# Source archive/member identity and original diagonal entropy counterexample.
O=B/'mqft_restore80_20261127';A69=B/'mqft_restore69_20261126';O79=B/'mqft_restore79_20261127'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==v for k,v in orig['archives'].items())
ids=['L09','L15','L16','L20','Q24','Q25','Q35','Q37','Q42','Q43','Q45'];spans=[s for s in orig['spans'] if s['id'] in ids];assert len(spans)==len(ids)
want={s['member'] for s in spans}|{'mcts_study/v3.0/qft-mcts-decoherence.md','mcts_study/effective-action-mcts.md','mcts_study/v5.0/new_quantum_mcts.md'}
members=[]
for s in orig['verified_members']:
 if s['member'] not in want:continue
 with zipfile.ZipFile(paths[s['archive']]) as f:v=f.read(s['member'])
 assert hashlib.sha256(v).hexdigest()==s['sha256'];members.append(s)
assert {s['member'] for s in members}==want
m='mcts_study/v6.0/quantum_mcts_foundation.md'
with zipfile.ZipFile(paths['mcts_study.zip']) as f:v=f.read(m)
s=v.decode();assert 'Selection phase: Unitary evolution under H' in s and 'The von Neumann entropy quantifies superposition:' in s and 'The entropy follows:' in s
h=hashlib.sha256(v).hexdigest();members.append(dict(archive='mcts_study.zip',member=m,sha256=h))
rho=(I+.72*y)/2;de=np.diag(rho.diagonal());e=np.linalg.eigvalsh(rho);se=float(-e@np.log(e));sd=float(np.log(2));assert np.allclose(rho.diagonal(),de.diagonal()) and se<sd
dump(S/'ORIGINAL_SOURCE81.json',dict(archives=orig['archives'],verified_members=members,spans=spans,original_attachment_receipt=orig['original_attachment_receipt'],scope='Exact original sources and v6 thesis. Diagonal policy entropy equals vN entropy only for diagonal density in that basis; not generic coherent state. Direct counterexample in declared register; no universal entropy-law refutation beyond scope.',diagonal_entropy=sd,coherent_entropy=se,erased_entropy=sd,coherent_spectrum=e.tolist()))
dump(S/'INDEPENDENT81.json',dict(status='PASS_SCOPED',raw_paths=n,branch_checks=n*6,active_steps_per_branch=12,errors=dict(err),frozen_contexts=len(V['contexts']),max_frozen_isometry_error=max(q['isometry_error'] for q in V['contexts']),max_frozen_causal_error=max(q['causal_error'] for q in V['contexts']),scope='Independent unnormalized branch products with spectral square roots, 432 raw histories x6 alternatives, aggregate effect rederivation; frozen Gram versus independent isometry. Same-agent separate implementation.'))
print('VERIFY',n,n*6,dict(err),'entropy',se,sd,flush=True)
