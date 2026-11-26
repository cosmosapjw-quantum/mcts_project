"""Independent fresh native processes, compare high-beta and pulse records to persistent acquisition."""
from base61 import *
import sys,gzip
from mqft.client import Backend,digest
from mqft.campaign import load_plan
from mqft.rapfi import verify
L=int(sys.argv[1]);O=B/'mqft_restore49_20261126';P=B/'mqft_restore39_20261125/repo';R=read(S/f'REGISTRATION_L{L}.json')
exe=O/'runtime_verified'/f'mqft-L{L}';ext=P/'.external/rapfi';assert sha(exe)==R['backend_sha'] and verify(ext)==R['model']
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns')}
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
out=[]
for name in ['base','pulse']:
 plan=load_plan(S/'plans'/f'L{L}_{name}.json')
 if name=='base':j=next(x for x in plan['jobs'] if x['spec']['budget']==64 and x['spec']['root_law']['beta_p']==[16.] and x['position_id']=='c00_dense' and x['replicate']==0)
 else:j=next(x for x in plan['jobs'] if x['spec']['root_law']['logit_schedule'][-1]==.35 and x['position_id']=='c00_dense' and x['replicate']==0)
 p=S/'data'/f'L{L}'/name/(j['id']+'.jsonl.gz');r=read(p.with_name(j['id']+'.receipt.json'));assert sha(p)==r['sha256']
 with gzip.open(p,'rt') as f:h,t,e=map(json.loads,f)
 assert h['job']==j
 with Backend(exe,90,ext,4,diagnostic_path=S/'logs'/f'L{L}-fresh-{name}.log') as be:actual=be.search(j['opening'],j['spec'])
 assert scrub(actual)==scrub(t['trace'])
 out.append(dict(campaign=name,id=j['id'],raw_sha=r['sha256'],semantic_sha=digest(scrub(actual)),new_wall_ns=actual['wall_ns'],original_wall_ns=t['trace']['wall_ns']))
dump(S/f'FRESH_L{L}.json',dict(L=L,status='exact_science_fields_equal',cases=out));print('FRESH',L,out,flush=True)
