"""Pinned native persistent JSONL campaign. Same spec and trace format as R72, explicit receipts."""
from base61 import *
import sys,gzip,os,time,subprocess
from mqft.client import Backend,digest,canonical
from mqft.campaign import load_plan,atomic_json,verify_state,validate_receipt
from mqft.rapfi import verify
L=int(sys.argv[1]);O=B/'mqft_restore49_20261126';P=B/'mqft_restore39_20261125/repo';R=read(S/f'REGISTRATION_L{L}.json')
exe=O/'runtime_verified'/f'mqft-L{L}';ext=P/'.external/rapfi';assert sha(exe)==R['backend_sha'] and verify(ext)==R['model'] and sha(O/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==R['source_manifest_sha']
plans={n:load_plan(S/'plans'/f'L{L}_{n}.json') for n in ('base',)}
for n,a in plans.items():assert sha(S/'plans'/f'L{L}_{n}.json')==R['plans'][n]['sha256'] and a['plan_hash']==R['plans'][n]['hash']
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns')}
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
d=S/'data'/f'L{L}';d.mkdir(exist_ok=False)
t0=time.monotonic();n=0;receipts=[]
with Backend(exe,90,ext,4,diagnostic_path=S/'logs'/f'L{L}-native.log') as be:
 assert be.sha==R['backend_sha'] and be.rapfi_identity==R['model']
 # Fresh reference of first registered job, then persistent repeat after an unrelated job.
 pilot=plans['base']['jobs'][0];other=plans['base']['jobs'][-1]
 with Backend(exe,90,ext,4,diagnostic_path=S/'logs'/f'L{L}-pilot.log') as fresh:ref=fresh.search(pilot['opening'],pilot['spec'])
 be.search(other['opening'],other['spec']);actual=be.search(pilot['opening'],pilot['spec'])
 assert scrub(ref)==scrub(actual),'persistent semantic mismatch'
 dump(S/f'PILOT_L{L}.json',dict(job=pilot['id'],semantic_sha=digest(scrub(ref)),fresh_wall_ns=ref['wall_ns'],persistent_wall_ns=actual['wall_ns'],status='exact_science_fields_equal'))
 print('PILOT',L,ref['wall_ns'],actual['wall_ns'],flush=True)
 for name,plan in plans.items():
  out=d/name;out.mkdir();atomic_json(out/'run_lock.json',dict(plan_hash=plan['plan_hash'],backend_sha=be.sha,external=be.rapfi_identity))
  rec=[]
  for j in plan['jobs']:
   st=be.state(j['opening']);verify_state(st,j['opening']);tr=be.search(j['opening'],j['spec'])
   expect=dict(j['spec'],c_puct=float(np.float32(j['spec']['c_puct'])))
   assert tr['spec']==expect and tr['root']['state_hash']==st['state_hash'],[(k,expect.get(k),tr['spec'].get(k)) for k in set(expect)|set(tr['spec']) if expect.get(k)!=tr['spec'].get(k)]
   rows=[dict(type='header',schema=plan['schema'],plan_hash=plan['plan_hash'],job=j,backend_sha=be.sha,backend=be.identity,external=None,evaluator_runtime=be.rapfi_identity),dict(type='root',moves=j['opening'],trace=tr),dict(type='end',status='root_complete',moves=j['opening'],terminal=False,winner=None,searched_records=1,backups=j['spec']['budget'])]
   p=out/(j['id']+'.jsonl.gz');tmp=p.with_name(p.name+'.part');b=b''.join((canonical(x)+'\n').encode() for x in rows)
   with tmp.open('xb') as f:
    with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=1,mtime=0) as z:z.write(b)
    f.flush();os.fsync(f.fileno())
   assert tmp.stat().st_size<=plan['max_output_bytes']//len(plan['jobs'])
   r=dict(job_id=j['id'],job_hash=digest(j),plan_hash=plan['plan_hash'],backend_sha=be.sha,status='root_complete',sha256=sha(tmp),file=p.name,bytes=tmp.stat().st_size,uncompressed_bytes=len(b),records=1,backups=j['spec']['budget'])
   tmp.replace(p);atomic_json(out/(j['id']+'.receipt.json'),r);validate_receipt(r,p,j,plan['plan_hash'],be.sha);rec.append(r);n+=1
   if n%64==0:print('GEN',L,n,time.monotonic()-t0,flush=True)
  report=dict(plan_hash=plan['plan_hash'],backend_sha=be.sha,expected=len(plan['jobs']),completed=rec,failures=[],unattempted=0)
  atomic_json(out/'report.json',report);receipts+=rec
assert n==160
dump(S/f'GEN_L{L}.json',dict(L=L,paths=n,backups=sum(x['backups'] for x in receipts),raw_bytes=sum(x['bytes'] for x in receipts),seconds=time.monotonic()-t0))
print('DONE',L,n,time.monotonic()-t0,flush=True)
