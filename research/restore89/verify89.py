"""Independent closed-form logical channel, source audit and prior-artifact identities."""
from base61 import *
import zipfile,ast,re
from itertools import product
O67=B/'mqft_restore67_20261126';O68=B/'mqft_restore68_20261126';O79=B/'mqft_restore79_20261127';O88=B/'mqft_restore88_20261128';O69=B/'mqft_restore69_20261126'
Q=read(S/'QEC89.json');N=read(S/'NATIVE89.json');A=read(S/'SUMMARY89.json');old=read(O67/'QEC67.json')
err=collections.defaultdict(float)
def ck(k,a,b,tol=1e-10):
 e=float(abs(np.asarray(a)-np.asarray(b)).max());err[k]=max(e,err[k]);assert e<tol,(k,e)
# Syndrome instrument in X basis: conditional scalar eigenvalues are a±b. Direct complex multiplication rather than PTM.
def eigen(t):
 c,s=np.cos(t),np.sin(t)
 a=np.array([c**3]+[-1j*s*c*c]*3);b=np.array([1j*s**3]+[-s*s*c]*3);return a+b,a-b
for r,o in zip(Q['coherent'],old['coherent']):
 t=r['theta'];ck('angle',t,o['angle']);e0,e1=eigen(t);q=abs(e0)**2
 ck('syndrome_mass',q,r['syndrome_prob']);ck('state_independence',q,abs(e1)**2)
 z=np.sum(e0*e1.conj())
 for name in r['results']:
  if name=='ordinary':lam=z
  elif name=='pauli':lam=1-2*r['logical_flip']
  else:
   te=t+({'known_angle':0.,'estimate_plus':.03,'estimate_minus':-.03}[name]);v0,v1=eigen(te);ok=abs(v0)**2>1e-28
   u=np.ones(4,complex);u[ok]=v0[ok].conj()*v1[ok]/abs(v0[ok])**2
   lam=np.sum(e0*e1.conj()*u)
  for j in (1,10,100):
   ck('independent_coherence_eigenvalue',(1+np.real(lam**j))/2,r['results'][name][str(j)])
 for name,ref in [('ordinary','repeated_coherent'),('pauli','repeated_pauli')]:
  for j,v in o[ref]:ck('R67_channel_reproduction',r['results'][name][str(j)],v)
for r in Q['posteriors']:
 p,mu=r['p'],r['mu'];a=(1-mu)*(1-p)**3+mu*(1-p);b=(1-mu)*p**3+mu*p;c=(1-mu)*p*(1-p)**2;d=(1-mu)*p*p*(1-p)
 ck('mask_pair_mass',a+b+3*(c+d),1);ck('map_error',min(a,b)+3*min(c,d),r['map_error']);ck('majority_error',b+3*d,r['majority_error'])
 for s in r['syndromes']:
  z=a+b if s['syndrome']==[0,0] else c+d;x=b if s['syndrome']==[0,0] else d
  ck('syndrome_prob',z,s['prob'])
  if z:ck('conditional_residual',x/z,s['majority_residual'])
  else:assert s['majority_residual'] is None
# Verify source member hashes from R79 and exact producer. Inspect actual stabilizer sets by symplectic parity.
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==v for k,v in orig['archives'].items())
ids=['L21','L23'];contracts=[x for x in read(O79/'ITEM_ACCEPTANCE79.json')['items'] if x['id'] in ids];assert len(contracts)==2
src=[]
for c in contracts:
 s=c['original_source'];m=s['member'];archive=s['archive']
 with zipfile.ZipFile(paths[archive]) as z:v=z.read(m)
 assert hashlib.sha256(v).hexdigest()==s['member_sha'];lines=v.decode().splitlines()
 src.append(dict(id=c['id'],original_source=s,acceptance=c['final_acceptance'],source_excerpt='\n'.join(lines[int(s['start'])-1:int(s['end'])])))
m='omoknuni_quantum-main/python/mcts/quantum/phenomena/quantum_error_correction.py'
with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:v=z.read(m)
s=v.decode();lines=s.splitlines();root=ast.parse(s);codes=[]
for n in ast.walk(root):
 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='QuantumCode':
  kw={x.arg:ast.literal_eval(x.value) for x in n.value.keywords};st=kw['stabilizers'];bad=[[i,j] for i in range(len(st)) for j in range(i+1,len(st)) if sum(x!=y and x!='I' and y!='I' for x,y in zip(st[i],st[j]))%2]
  codes.append(dict(name=kw['name'],n=kw['n'],stabilizers=st,noncommuting_pairs=bad))
assert len(codes)==3 and next(x for x in codes if x['name']=='Shor')['noncommuting_pairs']
assert 'fidelity_reduction = len(errors) * 0.1' in s and 'syndrome_bit = 1 if np.random.random() < prob_error else 0' in s and 'fidelity_improvement = 0.8' in s
dump(S/'SOURCE_CORRECTIONS89.json',dict(member=m,sha256=hashlib.sha256(v).hexdigest(),code_tables=codes,segments=[dict(start=a,end=b,text='\n'.join(lines[a-1:b])) for a,b in [(159,198),(252,307),(309,394),(396,440)]],defects=['Two amplitudes rather than n-qubit encoded space','Noise changes scalar fidelity instead of actual state','Random fidelity-based syndrome rather than stabilizer outcome','Recovery changes scalar fidelity rather than channel','Shor listed stabilizer set includes anticommuting pair(s)'],scope='Static source audit only, corrected model separate, originals unchanged.'))
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O88/f)==sha(O69/f)
# Verify R88 source artifacts for integrated three rows, not new joint data.
r=read(O88/'ITEM_RECONSTRUCTION88.json');assert all(next(x for x in r['items'] if x['id']==k)['decision']=='ACCEPTED_CORRECTED_SCOPED_ARTIFACT' for k in ('L22','L24'))
for k in ('L22','L24'):
 x=next(x for x in r['items'] if x['id']==k)
 assert sha(O88/'plots'/f"{x['figure']}.png")==x['png_sha'] and sha(O88/'plots'/f"{x['figure']}.pdf")==x['pdf_sha']
 assert all(sha(O88/f)==h for f,h in x['evidence'].items())
dump(S/'ORIGINAL_SOURCE89.json',dict(archives=orig['archives'],original_attachment_receipt=orig['original_attachment_receipt'],items=src,acceptance_contract_sha=sha(O79/'ITEM_ACCEPTANCE79.json'),scope='L21 and L23 exact source spans and R79 contracts. R88 accepted component artifact identities verified.'))
dump(S/'INDEPENDENT89.json',dict(status='PASS_SCOPED',coherent_cases=len(Q['coherent']),coherent_repeated_checks=31*5*3,posterior_cases=len(Q['posteriors']),prior_QEC67_cases=31,root_paths=3200,game_paths=80,eval_records=579026,raw_cut_error=N['max_cut_error'],errors=dict(err),scope='Exact conditional syndrome posterior, independent X-eigenbasis logical channels and prior R67 recurrence, original source and R88 component identity; prior full root-law/evaluator replay inherited, new sign and cut audit all records. Same-agent separate implementation.'))
print('VERIFIED',dict(err),'original Shor',next(x for x in codes if x['name']=='Shor')['noncommuting_pairs'],flush=True)
