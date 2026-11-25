import json,gzip
from pathlib import Path
from mqft.client import sha256
from mqft.campaign import atomic_json
S=Path(__file__).resolve().parent;B=S.parent
def read(p):return json.loads(p.read_text())
r47=read(B/'mqft_restore47_20261126/PHYSICS_RESULTS.json');r46=read(B/'mqft_restore46_20261126/PHYSICS_RESULTS.json')
anchors={r['position'] for r in r47['native'] if r['campaign']!='heldout64'}
a=[r for r in r46['native'] if r['position'] in anchors and r['h']==0]
b=[r for r in r47['native'] if r['position'] in anchors and r['h']==0]
n=0;gaps=[]
for r in a+b:
 if 'epoch' in r:
  epoch=r['epoch'];path=B/('mqft_restore'+epoch[:2]+'_20261126')
  campaign={'44':'marked','45':'marked','46-old':'old_roots','46-new':'new_roots'}[epoch]
 else:path=B/'mqft_restore47_20261126';campaign=r['campaign']
 f=path/'data'/campaign/(r['job']+'.jsonl.gz');assert sha256(f)==r['shard_sha']
 with gzip.open(f,'rt') as x:hd,root,end=[json.loads(l) for l in x]
 ev=sorted(root['trace']['replicas'][0]['events'],key=lambda x:x['root_selection']['clock'])
 assert [x['root_value'] for x in ev]==r['marks']
 for i in range(4,len(ev),4):
  gap=ev[i]['root_selection']['clock']-max(x['root_backup_clock'] for x in ev[i-4:i])
  assert gap>0;gaps.append(gap)
 n+=1
assert n==len(a)+len(b)==240
atomic_json(S/'PRIOR_CLOCK_AUDIT.json',dict(histories=n,boundaries=len(gaps),min_gap=min(gaps),scope='old h0 scale anchors only; all prior marks match trace and prior block backup before future selection'))
print(n,len(gaps),min(gaps))
