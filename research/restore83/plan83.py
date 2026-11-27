from base61 import *
from arena68 import verify,P
import gzip,itertools
O=B/'mqft_restore81_20261127';R=read(O/'REGISTRATION.json');a=read(O/'AUDIT81.json')
assert sha(S/'mqft_train')==R['binary_sha'] and sha(S/'mqft_phase81')==R['phase_binary_sha'] and verify(P/'.external/rapfi')==R['external']
assert all(sha(B/x['path'])==x['sha256'] for x in R['weights'].values())
refs={r['path']:r for r in a['raw_receipts'] if r['arm']==0 and r['rep']==0};assert len(refs)==72
contexts=[];plans={}
for g,tag in enumerate(('schedule0','schedule3','rapfi')):
 jobs=[]
 for j in read(O/'plans'/f'{tag}.json'):
  if j['arm']!='plus81' or j['rep']!=0:continue
  p=O/'data'/tag/(j['id']+'.json.gz');r=refs[str(p.relative_to(B))];assert sha(p)==r['sha256']
  with gzip.open(p,'rt') as f:d=json.load(f)
  assert d['input']==j
  prefix=[e['actions'][0] for e in d['trace']['events'][:32]]
  c=dict(family=g,core=j['core'],source=str(p.relative_to(B)),source_sha=r['sha256'],seed=j['seed'],pair=j['sources'],prefix=prefix,leaf_ids=[])
  for bits in itertools.product(range(2),repeat=3):
   label=''.join(map(str,bits));suffix=[j['sources'][i] for i in bits for _ in range(4)]
   jj=dict(j,id='branch83_'+j['id']+'_'+label,forced83=prefix+suffix);jobs.append(jj);c['leaf_ids'].append(jj['id'])
  jj=dict(j,id='check83_'+j['id'],forced83=[]);jobs.append(jj);c['reference_id']=jj['id'];contexts.append(c)
 assert len(jobs)==216
 p=S/'plans'/f'{tag}.json';dump(p,jobs);plans[tag]=dict(n=len(jobs),sha256=sha(p))
 if g<2:
  for i,q in enumerate([jobs[:18],jobs[18:]]):dump(S/'plans'/f'{tag}_shard{i}.json',q)
assert len(contexts)==72
dump(S/'CONTEXTS83.json',contexts);dump(S/'ROOT_PREPARATION.json',read(O/'ROOT_PREPARATION.json'))
R.update(parent='fda7af649e105cd14cdc5099f0ea654c83267cb4',branch_binary_sha=sha(S/'mqft_branch83'),branch_patch_sha=sha(S/'branch83.patch'),contexts_sha=sha(S/'CONTEXTS83.json'),plans=plans,workers=12,scope='72 observed H32 contexts, all eight repeat-wave leaves forced plus one exact unforced replay each; 648 diagnostic paths, no new independent sample.')
dump(S/'REGISTRATION.json',R);print('REGISTER',plans,flush=True)
