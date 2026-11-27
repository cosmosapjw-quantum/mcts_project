"""Actual branch-conditioned energy landscapes; hash-authenticated forced diagnostics."""
from base61 import *
import gzip
O73=B/'mqft_restore73_20261126';O83=B/'mqft_restore83_20261127'
def load(path,sha0):
 p=B/path;assert sha(p)==sha0
 with gzip.open(p,'rt') as f:return json.load(f)
def law(tr,t):
 l=tr['events'][t]['root_selection']['law'];a=np.asarray(l['actions']);ix=np.argsort(a);return a[ix],np.asarray(l['scores'])[ix]
def prefix(tr,k):
 return [(e['actions'],e['root_value'],e['leaf_hash']) for e in tr['events'][:k]]
out=[];raw=[]
M=read(O73/'RUN_MANIFEST73.json');index={x['path']:x['sha256'] for x in M['raw_receipts']}
for c in read(O73/'INSTRUMENT73.json')['contexts']:
 if c['arm']!='persist_plus':continue
 source=c['context_source'];h=c['context_source_sha'];assert index[source]==h
 d=load(source,h);tr=d['trace'];a,u=law(tr,15);rec=dict(path=source,sha256=h,role='R73_observed_prefix');raw.append(rec)
 em=[];branch=[]
 for r in sorted(c['raw_receipts'],key=lambda r:r['forced_action']):
  assert index[r['path']]==r['sha256']
  v=load(r['path'],r['sha256']);t=v['trace'];assert prefix(t,15)==prefix(tr,15) and t['events'][15]['actions'][0]==r['forced_action']
  assert np.array_equal(law(t,15)[0],a) and np.array_equal(law(t,15)[1],u)
  a1,u1=law(t,16);assert np.array_equal(a1,a)
  em.append((-u1).tolist());branch.append(r['forced_action']);raw.append(dict(path=r['path'],sha256=r['sha256'],role='R73_all_legal_forced_branch'))
 assert branch==a.tolist() and len(em)==len(a)
 out.append(dict(protocol='R73_fine_detector',family=int(c['generation']==3),core=c['core'],legal=a.tolist(),e0=(-u).tolist(),em=em,group=list(range(len(a))),source=source,branches=branch))
assert len(out)==32 and len(raw)==2304
M=read(O83/'RUN_MANIFEST83.json');index={x['path']:x['sha256'] for x in M['diagnostic_receipts']}
for c in read(O83/'TREE83.json')['contexts']:
 em=[];a=None;u=None;pre=None;refs=[]
 for label in ([0,0,0],[1,0,0]):
  r=next(x for x in c['leaves'] if x['bits']==label);assert index[r['path']]==r['sha256']
  d=load(r['path'],r['sha256']);tr=d['trace'];a0,u0=law(tr,32);a1,u1=law(tr,36)
  assert np.array_equal(a0,a1) and [e['actions'][0] for e in tr['events'][32:36]]==[c['pair'][label[0]]]*4
  if a is None:a=a0;u=u0;pre=prefix(tr,32)
  else:assert np.array_equal(a,a0) and np.array_equal(u,u0) and prefix(tr,32)==pre
  em.append((-u1).tolist());refs.append(r['path']);raw.append(dict(path=r['path'],sha256=r['sha256'],role='R83_actual_repeat_branch'))
 L=15 if c['family']==2 else 9;pos=np.array([divmod(int(v),L) for v in a]);p0,p1=[np.array(divmod(v,L)) for v in c['pair']]
 group=((pos-p1)**2).sum(1)<((pos-p0)**2).sum(1);assert group.any() and not group.all()
 out.append(dict(protocol='R83_binary_detector',family=c['family'],core=c['core'],legal=a.tolist(),e0=(-u).tolist(),em=em,group=group.astype(int).tolist(),source=refs,branches=c['pair']))
assert len(out)==104 and len(raw)==2448 and len({x['path'] for x in raw})==2448
dump(S/'BRANCH_ENERGY85.json',dict(contexts=out,raw_receipts=raw,scope='R73 32 full-action feedback landscapes from 2272 authenticated all-fine-choice forced traces and32 observed sources; R83 72 binary groups from144 actual forced repeat branches. All bare scores, not CQ readout probabilities. Separate declared protocols.'))
print('BRANCH COMPLETE',len(out),len(raw),flush=True)
