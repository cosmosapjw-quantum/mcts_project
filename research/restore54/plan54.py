from pathlib import Path
import json,sys,math
import numpy as np
from scipy.linalg import helmert
from mqft.campaign import make_plan,atomic_json,valid_moves,win
from mqft.client import sha256
from mqft.rapfi import verify
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore53_20261126'
L=int(sys.argv[1]);c=L//2;U=helmert(4,full=False).T
read=lambda p:json.loads(p.read_text())
def orbit(m,L):
 a=np.zeros((L,L),int)
 for i,x in enumerate(m):a.ravel()[x]=1+i%2
 return min(tuple(x.ravel()) for k in range(4) for x in (np.rot90(a,k),np.fliplr(np.rot90(a,k))))
seen=set()
for n in (44,45):
 for x in read(B/f'mqft_restore{n}_20261126/plans/marked.json')['config']['positions']:seen.add(orbit(x['moves'],15))
for x in read(B/'mqft_restore47_20261126/plans/heldout64.json')['config']['positions']:seen.add(orbit(x['moves'],15))
for n in (50,51,53):
 f=B/f'mqft_restore{n}_20261126'/('REGISTRATION_L15.json' if n==50 else 'REGISTRATION_L13.json')
 for m in read(f)['templates15']:seen.add(orbit(m,15))
rng=np.random.default_rng(540001);allow=[y*15+x for y in range(4,11) for x in range(4,11) if y*15+x!=112]
while True:
 m=[112]+rng.choice(allow,7,replace=False).tolist()
 if orbit(m,15) not in seen:break
core=[(a//15-7+c)*L+a%15-7+c for a in m];dense=2*math.floor(.03*L*L+.5)
rng=np.random.default_rng(541000+L);pool=[i for i in range(L*L) if max(abs(i//L-c),abs(i%L-c))>=4]
seenL={orbit(x['moves'],L) for n in (50,51,53) for x in read(B/f'mqft_restore{n}_20261126/REGISTRATION_L{L}.json')['positions']}
pos=[]
for v in range(2):
 for attempt in range(1000):
  mv=core+rng.choice(pool,dense-8,replace=False).tolist()
  try:
   b=valid_moves(mv);o=orbit(mv,L);assert o not in seenL and not win(b,1) and not win(b,2)
  except (ValueError,AssertionError):continue
  break
 else:raise RuntimeError('geometry')
 seenL.add(o);pos.append(dict(id=f'L{L}_newv{v}',moves=mv,source_action=None))
old=read(O/f'REGISTRATION_L{L}.json');rt=O/'runtime'/f'mqft-L{L}';family=O/'family'/f'L{L}'
assert sha256(rt)==old['backend_sha'] and sha256(family/'SOURCE_MANIFEST.json')==old['source_manifest_sha'] and old['model']==verify(P/'.external/rapfi')
prior=next(x for x in old['positions'] if x['id']==f'L{L}_g0v0');prior={**prior,'id':f'L{L}_precision'}
base=dict(kind='roots',c_puct=[1.4],sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**30,search_timeout=90,max_plies=L*L)
plans={}
zero=(0.,0.,0.)
def axis(i,a):
 x=[0.,0.,0.];x[i]=a;return tuple(x)
def mixed(i,j):
 return [tuple(.3*(s if k==i else t if k==j else 0.) for k in range(3)) for s in (-1,1) for t in (-1,1)]
five=[zero]+[axis(0,h) for h in (-.3,-.15,.15,.3)]
all19=[zero]+[axis(i,h) for i in range(3) for h in (-.3,.3)]+sum([mixed(i,j) for i,j in ((0,1),(0,2),(1,2))],[])
assert len(set(all19))==19
def roots(name,positions,B0,n,laws,arms,seed):
 modes=[]
 for law in laws:
  for x in arms:
   h=(U@x).tolist();schedule=[[0.]*4 if i<B0-4 else h for i in range(B0)]
   modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law=dict(kind=law,beta_p=[10/3],beta_q=[10/3],proposal='p',logit_h=0.,sector_schedule=schedule)))
 a=make_plan({**base,'seed':seed,'replicates':n,'budgets':[B0],'positions':positions,'modes':modes})
 assert len(a['jobs'])==len(positions)*n*len(laws)*len(arms)
 atomic_json(S/'plans'/f'{name}-L{L}.json',a)
 plans[name]=dict(hash=a['plan_hash'],jobs=len(a['jobs']),B=B0,reps=n,arms=arms,laws=laws)
# 32 paired seed groups for one old root IID; two untested mixed pairs.
if L==13:
 roots('precision',[prior],16,32,['iid'],five+mixed(0,2)+mixed(1,2),54100+L)
roots('new',[pos[0]],16,4,['iid'] if L==13 else ['iid','pl','reserve'],all19,54200+L)
# Same fresh geometry for h comparison across B; B16 independent from new.
for budget in (16,32,64):roots(f'scale{budget}',[pos[0]],budget,4,['iid'],five,54300+100*L+budget)
roots('variant',[pos[1]],16,4,['iid'],[zero],54400+L)
cfg=dict(kind='games',seed=54500+L,replicates=1,budgets=[16],c_puct=[1.4],modes=[dict(mode='serial',workers=1,samples_per_eval=4)],openings=[pos[1]['moves']],opponents=['self'],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,move_temperature=0.,max_plies=L*L,jobs=1,cpu_limit=2,max_output_bytes=2**28,search_timeout=90)
a=make_plan(cfg);atomic_json(S/'plans'/f'game-L{L}.json',a);plans['game']=dict(hash=a['plan_hash'],jobs=1,B=16)
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,positions=pos+[prior],template15=m,density=dense/L**2,basis=U.tolist(),backend_sha=sha256(rt),model=old['model'],source_manifest_sha=old['source_manifest_sha'],plans=plans,scope='precision old root L13 IID 32 seeds; fresh common core and peripheral variant; all mixed pairs on fresh roots, B16/32/64 and h .15/.3 at fixed fresh geometry; bounded pilot precision, fixed arms before outcomes'))
print(L,[(n,q['jobs']) for n,q in plans.items()],flush=True)
