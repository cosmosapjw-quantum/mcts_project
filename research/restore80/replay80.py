"""Counterfactual forced-root-history replay with different seeds; not independent stochastic draws."""
from base61 import *
import gzip,subprocess,os
from arena68 import P
O=B/'mqft_restore79_20261127';R=read(S/'REGISTRATION.json');A=read(O/'AUDIT79.json')
old={x['path']:x for x in A['raw_receipts'] if x['cohort']==1}
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns','seed','id','u','forced80') } # eval IDs checked separately below; preserve all fields other than job/seed/random/timing
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
def compare(a,b):
 # only raw RNG draws and per-worker/timing metadata excluded; evaluation IDs retained.
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns','u')}
  if isinstance(x,list):return [clean(v) for v in x]
  return x
 for k in ('events','cuts','evaluations','state_hash','side'):
  assert clean(a[k])==clean(b[k]),k
 assert [e['root_selection']['law']['u'] for e in a['events']]!=[e['root_selection']['law']['u'] for e in b['events']]
 return hashlib.sha256(json.dumps({k:clean(a[k]) for k in ('events','cuts','evaluations','state_hash','side')},sort_keys=True,separators=(',',':')).encode()).hexdigest()
jobs={}
for tag in ('schedule0','schedule3','rapfi'):
 z=[]
 for j in read(O/'plans'/f'{tag}.json'):
  if j['core'] not in (0,1,2,3) or j['rep'] not in (0,1):continue
  p=O/'data'/tag/(j['id']+'.json.gz');r=old[str(p.relative_to(B))];assert sha(p)==r['sha256']
  with gzip.open(p,'rt') as f:d=json.load(f)
  assert d['input']==j
  jj=dict(j,seed=int.from_bytes(hashlib.sha256(('replay80'+j['id']).encode()).digest()[:8],'little')%(2**63),id='force80_'+j['id'],forced80=[e['actions'][0] for e in d['trace']['events']])
  assert jj['seed']!=j['seed'];z.append((jj,d['trace'],r))
 assert len(z)==8;jobs[tag]=z
reg=dict(scope='24 forced diagnostic replays, original R79 root sequences, changed seed; no counts of fresh probability histories',binary_sha=sha(S/'mqft_replay80'),base_binary_sha=sha(S/'mqft_pulse74'),patch_sha=sha(S/'replay80.patch'),source_receipts=[r for z in jobs.values() for j,t,r in z])
dump(S/'REPLAY_REGISTRATION80.json',reg)
out=[]
for tag in ('schedule0','schedule3'):
 z=jobs[tag];p=S/'plans'/f'forced_{tag}.json';dump(p,[j for j,t,r in z]);dd=S/'data'/('forced_'+tag)
 with (S/'logs'/f'forced_{tag}.log').open('w') as f:subprocess.run([str(S/'mqft_replay80'),'schedule66',str(B/R['weights'][tag[-1]]['path']),str(p),str(dd),'4'],stdout=f,stderr=subprocess.STDOUT,check=True)
 for j,t,r in z:
  f=dd/(j['id']+'.json.gz')
  with gzip.open(f,'rt') as q:d=json.load(q)
  assert d['input']==j;sem=compare(t,d['trace'])
  out.append(dict(family=tag,source=r['path'],source_sha=r['sha256'],path=str(f.relative_to(B)),sha256=sha(f),bytes=f.stat().st_size,semantic_sha=sem,seed=j['seed'],budget=128))
z=jobs['rapfi'];dd=S/'data/forced_rapfi';dd.mkdir();p=S/'plans/forced_rapfi.json';requests=[dict(moves=j['moves'],seed=j['seed'],arm='clean',budget=128,source74=j['sources'][0],pulse74=0.,forced80=j['forced80']) for j,t,r in z];dump(p,requests)
env=dict(os.environ,MQFT_RAPFI_DIR=str(P/'.external/rapfi'),MQFT_RAPFI_WORKERS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with (S/'logs/forced_rapfi.log').open('w') as f:
 proc=subprocess.Popen([str(S/'mqft_replay80'),'arena68'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=f,text=True,env=env)
 try:
  for (j,t,r),req in zip(z,requests):
   proc.stdin.write(json.dumps(req)+'\n');proc.stdin.flush();resp=json.loads(proc.stdout.readline());assert resp['ok'],resp
   rr=resp['result'];sem=compare(t,rr);f=dd/(j['id']+'.json.gz')
   with gzip.open(f,'wt',compresslevel=1) as q:json.dump(dict(input=req,trace=rr),q,separators=(',',':'),allow_nan=False)
   out.append(dict(family='rapfi',source=r['path'],source_sha=r['sha256'],path=str(f.relative_to(B)),sha256=sha(f),bytes=f.stat().st_size,semantic_sha=sem,seed=j['seed'],budget=128))
  proc.stdin.close();assert proc.wait(timeout=15)==0
 finally:
  if proc.poll() is None:proc.kill();proc.wait()
assert len(out)==24
dump(S/'FORCED_REPLAY80.json',dict(status='PASS_SCOPED',cases=out,diagnostic_backups=3072,scope='All marked path, evaluation, cut, score and probability fields exactly equal under forced root action sequence and changed seed; excludes random uniforms/worker/time only. Not actual sampled histories or universal proof of PRNG law.'))
print('FORCED_REPLAY',len(out),'all exact scientific fields',flush=True)
