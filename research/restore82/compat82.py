"""Predeclared clean-law compatibility; same-seed diagnostic replays, not new independent histories."""
from base61 import *
from arena68 import Native
import gzip,subprocess
O=B/'mqft_restore81_20261127';R=read(S/'REGISTRATION.json');a=read(O/'AUDIT81.json')
refs={(r['family'],r['core'],r['rep']):r for r in a['raw_receipts'] if r['arm']==5};out=[]
def readgz(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def semantic(tr):
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns','phase65','source')}
  if isinstance(x,list):return [clean(v) for v in x]
  return x
 return {k:clean(tr[k]) for k in ('events','cuts','evaluations','state_hash','side')}
for g,tag in enumerate(('schedule0','schedule3','rapfi')):
 jobs=[];raw=[]
 for core in (0,1,2,3):
  r=refs[g,core,0];p=B/r['path'];assert sha(p)==r['sha256'];d=readgz(p);j=dict(d['input']);assert j['arm']=='base81'
  j['arm']='zero74';j['id']='compat82_'+j['id'].replace('base81','zero74');j.pop('phase81',None)
  if g==2:j['pulse74']=0.
  jobs.append(j);raw.append((r,d['trace']))
 p=S/'plans'/f'compat_{tag}.json';dump(p,jobs);dest=S/'data'/('compat_'+tag)
 if g<2:
  subprocess.run([str(S/'mqft_pulse74'),'schedule66',str(B/R['weights'][str(3*g)]['path']),str(p),str(dest),'4'],check=True,stdout=subprocess.DEVNULL)
 else:
  dest.mkdir();n=Native(S/'logs/compat_rapfi.stderr.log')
  try:
   for j in jobs:
    tr=n.query(j['moves'],j['seed'],'clean',128,j['sources'][0],0.)
    with gzip.open(dest/(j['id']+'.json.gz'),'wt',compresslevel=1) as f:json.dump(dict(input=j,trace=tr),f,separators=(',',':'),allow_nan=False)
  finally:n.close()
 for j,(r,t) in zip(jobs,raw):
  q=dest/(j['id']+'.json.gz');v=readgz(q);assert v['input']==j
  s0=semantic(t);s1=semantic(v['trace'])
  for k in s0:assert s0[k]==s1[k],(g,j['core'],k)
  h=hashlib.sha256(json.dumps(s0,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  out.append(dict(family=g,core=j['core'],source=r['path'],source_sha=r['sha256'],path=str(q.relative_to(B)),sha256=sha(q),bytes=q.stat().st_size,semantic_sha=h,seed=j['seed'],backups=128))
assert len(out)==12
dump(S/'COMPAT82.json',dict(status='PASS_SCOPED',cases=out,diagnostic_backups=1536,scope='12 same-seed clean replays of R81 lambda0 paths on pinned pulse74 binary. All events/cuts/evals equal excluding source metadata, phase register, worker/time. Diagnostic replays excluded from fresh search count.'))
print('COMPAT',len(out),flush=True)
