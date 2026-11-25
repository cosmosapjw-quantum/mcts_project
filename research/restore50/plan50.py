from pathlib import Path
import json,sys,math,hashlib
import numpy as np
from mqft.campaign import make_plan,atomic_json,valid_moves,win
from mqft.client import sha256
from mqft.rapfi import verify
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore49_20261126'
L=int(sys.argv[1]);c=L//2
read=lambda p:json.loads(p.read_text())
def orbit(m,L):
 a=np.zeros((L,L),int)
 for i,x in enumerate(m):a.ravel()[x]=1+i%2
 return min(tuple(x.ravel()) for k in range(4) for x in (np.rot90(a,k),np.fliplr(np.rot90(a,k))))
def center(a,L):y,x=divmod(a,15);return (y-7+L//2)*L+x-7+L//2
# Deterministic new template pair across L, verify against previous root
# and opening geometries, not selected on outcomes.
rng=np.random.default_rng(50001);source=7*15+8;allowed=[y*15+x for y in range(4,11) for x in range(4,11) if y*15+x not in (112,source)]
old=[]
for n in (44,45):
 old+=read(B/f'mqft_restore{n}_20261126/plans/marked.json')['config']['positions']
old+=read(B/'mqft_restore47_20261126/plans/heldout64.json')['config']['positions']
seen={orbit(x['moves'],15) for x in old};templates=[]
while len(templates)<2:
 m=[112]+rng.choice(allowed,7,replace=False).tolist()
 if orbit(m,15) in seen:continue
 seen.add(orbit(m,15));templates.append(m)
positions=[];dense_n=2*math.floor(.03*L*L+.5);assert abs(dense_n/L**2-.06)<=1/L**2+1e-15
for i,m in enumerate(templates):
 core=[center(a,L) for a in m];src=center(source,L);valid_moves(core)
 rng=np.random.default_rng(50100+100*L+i)
 pool=[y*L+x for y in range(L) for x in range(L) if max(abs(y-c),abs(x-c))>=4]
 for attempt in range(1000):
  d=core+rng.choice(pool,dense_n-8,replace=False).tolist()
  try:
   b=valid_moves(d);assert not win(b,1) and not win(b,2) and b[src]==0
  except (ValueError,AssertionError):continue
  break
 else:raise RuntimeError("template construction")
 for tag,mv in [('fixed',core),('dense',d)]:
  b=valid_moves(mv);assert b[src]==0 and not win(b,1) and not win(b,2)
  positions.append(dict(id=f'L{L}_{tag}{i}',moves=mv,source_action=src))
assert len({orbit(x['moves'],L) for x in positions})==4
reg=read(O/f'REGISTRATION_L{L}.json');exe=O/'runtime_verified'/f'mqft-L{L}'
assert sha256(exe)==reg['backend_sha'] and sha256(O/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==reg['source_manifest_sha']
assert reg['model']==verify(P/'.external/rapfi')
prior={orbit(x['moves'],L) for x in reg['anchors']}
assert not prior&{orbit(x['moves'],L) for x in positions}
plans={}
base=dict(kind='roots',c_puct=[1.4],sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**29,search_timeout=90,max_plies=L*L)
def roots(name,pos,B,beta,rep,pulses=False,seed=0):
 modes=[]
 for law in ('iid','pl','reserve'):
  common=dict(kind=law,beta_p=[beta],beta_q=[beta],proposal='p',logit_h=0.)
  modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law=common))
  if pulses:
   for t in (0,4,12):
    for h in (-.2,.2):
     sched=[h if t<=k<t+4 else 0. for k in range(B)]
     modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law={**common,'logit_schedule':sched}))
 cfg={**base,'seed':seed,'replicates':rep,'budgets':[B],'positions':pos,'modes':modes}
 a=make_plan(cfg);assert len(a['jobs'])==len(pos)*3*rep*(7 if pulses else 1)
 if pulses:
  group={}
  for j in a['jobs']:group.setdefault((j['position_id'],j['replicate'],j['spec']['root_law']['kind']),[]).append(j)
  assert all(len(g)==7 and len({x['spec']['seed'] for x in g})==1 for g in group.values())
 atomic_json(S/'plans'/f'{name}-L{L}.json',a);plans[name]=dict(hash=a['plan_hash'],jobs=len(a['jobs']),B=B,beta=beta)
roots('base',[x for x in positions if not (L in (15,21) and x['id'].endswith('dense0'))],16,10/3,4,seed=50200+L)
dense=[x for x in positions if x['id'].endswith('dense0')]
if L in (15,21):roots('pulse',dense,16,10/3,8,True,50300+L)
if L in (13,17,21):
 for B0,bs in [(16,[1.]),(32,[1.,10/3]),(64,[1.,10/3])]:
  for beta in bs:roots(f'B{B0}b{1 if beta==1 else 3}',dense,B0,beta,4,seed=51000+100*L+2*B0+int(beta))
cfg=dict(kind='games',seed=52000+L,replicates=1,budgets=[16],c_puct=[1.4],modes=[dict(mode='serial',workers=1,samples_per_eval=4)],openings=[dense[0]['moves']],opponents=['self'],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,move_temperature=0.,max_plies=L*L,jobs=1,cpu_limit=2,max_output_bytes=2**28,search_timeout=90)
a=make_plan(cfg);atomic_json(S/'plans'/f'game-L{L}.json',a);plans['game']=dict(hash=a['plan_hash'],jobs=1,B=16)
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,positions=positions,templates15=templates,density_target=.06,dense_count=dense_n,density=dense_n/L**2,backend_sha=reg['backend_sha'],model=reg['model'],source_manifest_sha=reg['source_manifest_sha'],plans=plans,scope='new legal fixed templates and deterministic peripheral padding, not population sample; size-specific seed namespaces'))
print(L,dense_n,[(k,v['jobs']) for k,v in plans.items()],flush=True)
