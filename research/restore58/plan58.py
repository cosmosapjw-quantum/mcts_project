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
rng=np.random.default_rng(580001);source=7*15+8;allowed=[y*15+x for y in range(4,11) for x in range(4,11) if y*15+x not in (112,source)]
old=[]
for n in (44,45):old+=read(B/f'mqft_restore{n}_20261126/plans/marked.json')['config']['positions']
old+=read(B/'mqft_restore47_20261126/plans/heldout64.json')['config']['positions']
seen={orbit(x['moves'],15) for x in old}
for n in (50,51,53,54,55,56,57):
 q=read(B/f'mqft_restore{n}_20261126'/('REGISTRATION_L15.json' if n==50 else 'REGISTRATION_L13.json'))
 for m in q.get('templates15',[q.get('template15')]):
  if m:seen.add(orbit(m,15))
templates=[]
while len(templates)<2:
 m=[112]+rng.choice(allowed,7,replace=False).tolist()
 if orbit(m,15) in seen:continue
 seen.add(orbit(m,15));templates.append(m)
positions=[];dense_n=2*math.floor(.03*L*L+.5);assert abs(dense_n/L**2-.06)<=1/L**2+1e-15
for i,m in enumerate(templates):
 core=[center(a,L) for a in m];src=center(source,L);valid_moves(core)
 rng=np.random.default_rng(581000+100*L+i)
 pool=[y*L+x for y in range(L) for x in range(L) if max(abs(y-c),abs(x-c))>=4]
 for attempt in range(1000):
  d=core+rng.choice(pool,dense_n-8,replace=False).tolist()
  try:
   b=valid_moves(d);assert not win(b,1) and not win(b,2) and b[src]==0
  except (ValueError,AssertionError):continue
  break
 else:raise RuntimeError("template construction")
 for tag,mv in [('dense',d)]:
  b=valid_moves(mv);assert b[src]==0 and not win(b,1) and not win(b,2)
  positions.append(dict(id=f'L{L}_{tag}{i}',moves=mv,source_action=src))
assert len({orbit(x['moves'],L) for x in positions})==2
reg=read(O/f'REGISTRATION_L{L}.json');exe=O/'runtime_verified'/f'mqft-L{L}'
assert sha256(exe)==reg['backend_sha'] and sha256(O/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==reg['source_manifest_sha']
assert reg['model']==verify(P/'.external/rapfi')
prior={orbit(x['moves'],L) for x in reg['anchors']}
assert not prior&{orbit(x['moves'],L) for x in positions}
plans={}
base=dict(kind='roots',c_puct=[1.4],sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**29,search_timeout=90,max_plies=L*L)
def roots(name,B0,beta,n,pulse,seed):
 modes=[]
 for h in ((0.,-.2,.2) if pulse else (0.,)):
  law=dict(kind='iid',beta_p=[beta],beta_q=[beta],proposal='p',logit_h=0.)
  if h:law['logit_schedule']=[0.]*(B0-4)+[h]*4
  modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law=law))
 a=make_plan({**base,'seed':seed,'replicates':n,'budgets':[B0],'positions':positions,'modes':modes})
 assert len(a['jobs'])==2*n*len(modes)
 atomic_json(S/'plans'/f'{name}-L{L}.json',a);plans[name]=dict(hash=a['plan_hash'],jobs=len(a['jobs']),B=B0,beta=beta,reps=n)
for B0 in (16,32):
 for i,beta in enumerate((1.,10/3,8.)):
  roots(f'B{B0}b{i}',B0,beta,6,B0==16 and i==1,582000+100*L+2*B0+i)
assert sum(x['jobs'] for x in plans.values())==96
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,positions=positions,templates15=templates,density_target=.06,dense_count=dense_n,density=dense_n/L**2,backend_sha=reg['backend_sha'],model=reg['model'],source_manifest_sha=reg['source_manifest_sha'],plans=plans,scope='two common cores, matched approximate density; IID B16/32 x beta1,10/3,8, n6; final scalar source +/- .2 at B16 beta10/3; fixed before results; no critical exponent'))
print(L,dense_n,[(k,v['jobs']) for k,v in plans.items()],flush=True)
