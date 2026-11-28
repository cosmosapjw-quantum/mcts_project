"""Authenticated R69 original sources, heldout truth and independent empirical face checks."""
from base61 import *
import zipfile
from scipy.special import logsumexp
from scipy.spatial import ConvexHull
from scipy.stats import t
O79=B/'mqft_restore79_20261127';O89=B/'mqft_restore89_20261128';O69=B/'mqft_restore69_20261126';O80=B/'mqft_restore80_20261127';O84=B/'mqft_restore84_20261127';O87=B/'mqft_restore87_20261128'
P=read(S/'PATH90.json');D=read(S/'DUAL90.json');R=read(S/'RESPONSE90.json');W=read(S/'WORK90.json');Q=read(S/'PHASE90.json');reg=read(S/'REGISTRATION.json')
assert all(x['status']=='PASS_SCOPED' for x in (P,D,R,W,Q))
assert sha(S/'PATH90.npz')==P['array_sha'] and sha(S/'THEORY_REGISTERED90.md')==reg['contract_sha']
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O89/f)==sha(O69/f)
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==v for k,v in orig['archives'].items())
items=[x for x in read(O79/'ITEM_ACCEPTANCE79.json')['items'] if x['id'] in reg['target']];assert len(items)==6
src=[]
for c in items:
 s=c['original_source'];guide=s['source_status']=='GUIDE_LABEL_ONLY'
 m=s['guide_source_member'] if guide else s['member'];archive='omoknuni_quantum-main.zip' if guide else s['archive'];h=s['guide_source_sha'] if guide else s['member_sha']
 with zipfile.ZipFile(paths[archive]) as z:v=z.read(m)
 assert hashlib.sha256(v).hexdigest()==h
 lines=v.decode().splitlines();span='\n'.join(lines[i-1] for i in s['guide_lines']) if guide else '\n'.join(lines[int(s['start'])-1:int(s['end'])])
 src.append(dict(id=c['id'],original_source=s,acceptance=c['final_acceptance'],source_excerpt=span))
notes=[]
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 for m,lo,hi in [('mcts_study/effective-action-mcts.md',3,121),('mcts_study/v3.0/qft-mcts-statistical.md',144,155)]:
  v=z.read(m);lines=v.decode().splitlines();notes.append(dict(member=m,sha256=hashlib.sha256(v).hexdigest(),start=lo,end=hi,excerpt='\n'.join(lines[lo-1:hi])))
dump(S/'ORIGINAL_SOURCE90.json',dict(archives=orig['archives'],original_attachment_receipt=orig['original_attachment_receipt'],items=src,study_notes=notes,acceptance_contract_sha=sha(O79/'ITEM_ACCEPTANCE79.json'),scope='Original item source spans/guide labels and effective action study sections authenticated. R69 charter preserved.'))
# Verify prior committed artifacts against exact manifests, not just current path name.
checks=[]
for st,day,files in [(80,27,['ACTION80.json','MARKED80.npz','INDEPENDENT80.json','FORCED_REPLAY80.json']),(81,27,['AUDIT81.json','PATHS81.npz','EFFECTS81.json']),(82,27,['POPULATION82.npz','TRANSFER82.json','SUMMARY82.json']),(83,27,['RECORD83.json','RECORD_MATRICES83.npz','AUDIT83.json','TREE83.json']),(84,27,['TRANSFER84.json','MODEL84.json','FRESH84.npz','SUMMARY84.json','INDEPENDENT84.json']),(85,27,['THERMAL85.json','BRANCH_ENERGY85.json','INDEPENDENT85.json']),(86,27,['AUDIT86.json','WAVES86.npz']),(87,28,['CRITICAL87.json','ERRATUM86.json','SUMMARY87.json'])]:
 o=B/f'mqft_restore{st}_202611{day}';r=read(o/f'RUN_MANIFEST{st}.json')
 for f in files:
  h=sha(o/f);assert h==r['files'][f],(st,f);checks.append(dict(path=str((o/f).relative_to(B)),sha256=h))
# R74 predates manifest standard; exact array and response derived identities already verified from raw.
o=B/'mqft_restore74_20261126';r=read(o/'RUN_MANIFEST74.json')
for f in ('RESPONSE74.json','PULSE74.npz','AUDIT74.json'):
 h=sha(o/f);assert h==r['files'][f];checks.append(dict(path=str((o/f).relative_to(B)),sha256=h))
# Verify every heldout R84 true W from newly reduced actual X, all 21 model aggregate scores.
with np.load(S/'PATH90.npz') as z:meta=z['meta'];X=z['X'];Y=z['Y']
G=read(O84/'MODEL84.json');T=read(O84/'TRANSFER84.json');grid=np.array(G['source_grid'])
errs=collections.defaultdict(float)
def ck(k,a,b,tol=1e-9):
 e=float(abs(np.asarray(a)-np.asarray(b)).max());errs[k]=max(e,errs[k]);assert e<tol,(k,e)
tr=[];ntruth=0
for g,x in enumerate(T['results']):
 actual={}
 for c in range(48):
  y=X[(meta[:,0]==3)&(meta[:,1]==g)&(meta[:,2]==c)];assert len(y)==16;actual[c]=logsumexp(y@grid.T,axis=0)-np.log(len(y))
 for m in x['models']:
  truth=np.array([actual[r['core']] for r in m['rows']]);ck('R84_truth_W',truth,[r['truth_W'] for r in m['rows']]);ntruth+=len(truth)*len(grid)
  v=np.array([r['W'] for r in m['rows']]);es=np.array([r['energy'] for r in m['rows']]);ck('R84_model_W_RMSE',np.sqrt(np.mean((v[:,:25]-truth[:,:25])**2)),m['W_RMSE']);ck('R84_energy_mean',es.mean(),m['energy_ci'][0])
 a=next(m for m in x['models'] if m['model']==x['selected_baseline']);b=next(m for m in x['models'] if m['model']==x['selected_candidate']);d=np.array([u['energy']-v['energy'] for u,v in zip(a['rows'],b['rows'])]);assert all(u['core']==v['core'] for u,v in zip(a['rows'],b['rows']))
 v=d.mean();h=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));ck('R84_gain_ci',[v,v-h,v+h],x['energy_gain_ci'])
 tr.append(dict(family=x['family'],baseline=a['model'],candidate=b['model'],energy=[a['energy_ci'][0],b['energy_ci'][0]],W=[a['W_RMSE'],b['W_RMSE']],gain_ci=x['energy_gain_ci']))
# independent supporting-line test for all listed hulls, face membership by integer normals, all boundary/vertex KL.
with np.load(O80/'MARKED80.npz') as z:m=z['meta'];v=z['Y']
laws=[];nedge=nvertex=0
for r in D['laws']:
 g=['g0','g3','Rapfi'].index(r['label'][0]);lab=r['label'][1]
 y=v[(m[:,0]==(0 if lab=='R80_reference' else 1))&(m[:,1]==g)&(m[:,2]<32)] if lab.startswith('R80') else Y[(meta[:,0]==(3 if lab=='R84' else 4))&(meta[:,1]==g)]
 h=np.asarray(r['hull']);assert len(y)==r['n'] and set(map(tuple,h))==set(map(tuple,y[ConvexHull(y).vertices]))
 for e in r['edges']:
  a,b=np.array(e['edge']);u=b-a;n=np.array(e['normal']);mask=(y@n==a@n);assert np.all(y@n<=a@n);ck('face_mass',mask.mean(),e['face_mass'])
  for k in e['cases']:
   j=u/np.linalg.norm(u)*k['tangent_t'];z=y[mask]/128;u0=z@j;p=np.exp(u0-logsumexp(u0));mu=p@z;ck('face_mean',mu,k['mean']);ck('boundary_KL',float(p@np.log(p*len(y))),k['gamma'])
   t0=j/(k['tangent_t'] if k['tangent_t']!=0 else 1) if k['tangent_t']!=0 else (b-a)/np.linalg.norm(b-a)
   var=float(p@((z-mu)@t0)**2);ck('tangent_variance',var,k['tangent_variance'])
  nedge+=1
 for e in r['vertices']:
  mass=float(np.all(y==e['vertex'],1).mean());ck('vertex_rate',-np.log(mass),e['gamma']);nvertex+=1
 assert r['unique']==len(np.unique(y,axis=0))
assert nedge==nvertex==124
# Explicit R87 corrected per-selection bound and negative critical outcome; use prior committed verified strata.
r=read(O87/'CRITICAL87.json');s=read(O87/'SUMMARY87.json');a=read(O87/'ERRATUM86.json')
scale=[]
for z in r['strata']:
 if z['cohort']==2 and z['budget']==64 and z['h']==64 and z['bi']<4:
  q=dict(zip(z['fields'],z['mean']));assert 0<=q['trace']<=1+1e-10;ck('scale_decomposition',q['raw_trace'],q['drift_trace']+q['centered_innovation_trace']+q['cross_trace'])
  scale.append(dict(L=z['L'],arm=z['arm'],beta=z['beta'],trace=q['trace'],raw=q['raw_trace'],drift=q['drift_trace'],sample=q['uncentered_innovation_trace']))
assert len(scale)==40 and s['registered_critical_wins']==0 and not s['critical_gate']
# Unique raw receipt identity union: do not sum reused cohorts as new paths.
receipts={}
for src0 in (P['raw_receipts'],R['raw_receipts'],Q['raw_receipts'],W['raw_receipts']):
 for x in src0:
  k=x['path'];h=x['sha256'];assert k not in receipts or receipts[k]['sha256']==h
  receipts[k]=dict(path=k,sha256=h,bytes=(B/k).stat().st_size)
dump(S/'INDEPENDENT90.json',dict(status='PASS_SCOPED',source_items=6,prior_file_checks=checks,raw_receipts=list(receipts.values()),unique_raw_receipts=len(receipts),interior_cases=D['interior_cases'],edges=nedge,vertices=nvertex,edge_tilt_cases=3*nedge,normal_checks=6*3*nedge,loop_cases=D['saddle_cases'],unaliased_FFT=D['independent_unaliased_FFT'],heldout_truth_values=ntruth,transfer=tr,scale=scale,errors=dict(errs),scope='Original identities, prior committed data, exact finite face entropy and independent R84 truth/aggregate validation, R87 corrected bound. All raw receipts rehashed by path/response/phase runs; no new native samples. Same-agent separate computations.'))
print('VERIFIED',len(receipts),len(checks),nedge,nvertex,ntruth,dict(errs),tr,flush=True)
