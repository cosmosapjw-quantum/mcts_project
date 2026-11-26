from pathlib import Path
import json,sys,math,hashlib
import numpy as np
from mqft.campaign import make_plan,atomic_json,valid_moves,win
from mqft.client import sha256
from mqft.rapfi import verify
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore49_20261126';R=B/'mqft_restore58_20261126'
L=int(sys.argv[1]);c=L//2
read=lambda p:json.loads(p.read_text())
def orbit(m,L):
 a=np.zeros((L,L),int)
 for i,x in enumerate(m):a.ravel()[x]=1+i%2
 return min(tuple(x.ravel()) for k in range(4) for x in (np.rot90(a,k),np.fliplr(np.rot90(a,k))))
def center(a,L):y,x=divmod(a,15);return (y-7+L//2)*L+x-7+L//2
rng=np.random.default_rng(590001);source=113;allowed=[y*15+x for y in range(4,11) for x in range(4,11) if y*15+x not in (112,source)]
seen=set()
for q in sorted(B.glob('mqft_restore*/REGISTRATION_L13.json')):
 d=read(q)
 for m in d.get('templates15',[d.get('template15')]):
  if m and len(m)==8:seen.add(orbit(m,15))
templates=[]
while len(templates)<4:
 m=[112]+rng.choice(allowed,7,replace=False).tolist()
 if orbit(m,15) in seen:continue
 seen.add(orbit(m,15));templates.append(m)
positions=[];dense_n=2*math.floor(.03*L*L+.5)
for i,m in enumerate(templates):
 core=[center(a,L) for a in m];src=center(source,L);valid_moves(core)
 rr=np.random.default_rng(591000+100*L+i)
 pool=[y*L+x for y in range(L) for x in range(L) if max(abs(y-c),abs(x-c))>=4]
 for attempt in range(1000):
  d=core+rr.choice(pool,dense_n-8,replace=False).tolist()
  try:
   b=valid_moves(d);assert not win(b,1) and not win(b,2) and b[src]==0
  except (ValueError,AssertionError):continue
  break
 else:raise RuntimeError('template construction')
 positions.append(dict(id=f'L{L}_core{i}',moves=d,source_action=src))
reg=read(O/f'REGISTRATION_L{L}.json');exe=O/'runtime_verified'/f'mqft-L{L}'
assert sha256(exe)==reg['backend_sha'] and sha256(O/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==reg['source_manifest_sha']
assert reg['model']==verify(P/'.external/rapfi')
base=dict(kind='roots',c_puct=[1.4],sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**29,search_timeout=90,max_plies=L*L)
law=dict(kind='iid',beta_p=[10/3],beta_q=[10/3],proposal='p',logit_h=0.)
a=make_plan({**base,'seed':592000+L,'replicates':8,'budgets':[32],'positions':positions,'modes':[dict(mode='wave',workers=4,samples_per_eval=4,root_law=law)]})
assert len(a['jobs'])==32
atomic_json(S/'plans'/f'heldout-L{L}.json',a)
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,positions=positions,templates15=templates,backend_sha=reg['backend_sha'],model=reg['model'],source_manifest_sha=reg['source_manifest_sha'],plan_hash=a['plan_hash'],jobs=32,scope='prospective four new cores, 8 replicates, B32 beta10/3 iid; training R58 only'))
print('REGISTERED',L,len(a['jobs']),a['plan_hash'],flush=True)
