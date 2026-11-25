from pathlib import Path
import json,sys,copy,math,gzip
import numpy as np
from scipy.linalg import helmert
from mqft.campaign import make_plan,atomic_json,valid_moves,win
from mqft.client import Backend,sha256
from mqft.analysis import check_trace
from mqft.rapfi import verify
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore52_20261126'
L=int(sys.argv[1]);c=L//2;U=helmert(4,full=False).T
read=lambda p:json.loads(p.read_text())
def orbit(m,L):
 a=np.zeros((L,L),int)
 for i,x in enumerate(m):a.ravel()[x]=1+i%2
 return min(tuple(x.ravel()) for k in range(4) for x in (np.rot90(a,k),np.fliplr(np.rot90(a,k))))
def center(a):y,x=divmod(a,15);return (y-7+c)*L+x-7+c
seen=set()
for n in (44,45):
 for x in read(B/f'mqft_restore{n}_20261126/plans/marked.json')['config']['positions']:seen.add(orbit(x['moves'],15))
for x in read(B/'mqft_restore47_20261126/plans/heldout64.json')['config']['positions']:seen.add(orbit(x['moves'],15))
for n in (50,51):
 f=B/f'mqft_restore{n}_20261126'/('REGISTRATION_L15.json' if n==50 else 'REGISTRATION_L13.json')
 for m in read(f)['templates15']:seen.add(orbit(m,15))
rng=np.random.default_rng(530001);allowed=[y*15+x for y in range(4,11) for x in range(4,11) if y*15+x!=112];templates=[]
while len(templates)<2:
 m=[112]+rng.choice(allowed,7,replace=False).tolist()
 if orbit(m,15) in seen:continue
 seen.add(orbit(m,15));templates.append(m)
pos=[];dense=2*math.floor(.03*L*L+.5);seenL=set()
for n in (50,51):
 for x in read(B/f'mqft_restore{n}_20261126/REGISTRATION_L{L}.json')['positions']:seenL.add(orbit(x['moves'],L))
for g,m in enumerate(templates):
 core=[center(i) for i in m];rng=np.random.default_rng(531000+100*L+g)
 pool=[i for i in range(L*L) if max(abs(i//L-c),abs(i%L-c))>=4]
 for v in range(2):
  for attempt in range(1000):
   moves=core+rng.choice(pool,dense-8,replace=False).tolist()
   try:
    b=valid_moves(moves);o=orbit(moves,L);assert not win(b,1) and not win(b,2) and o not in seenL
   except (AssertionError,ValueError):continue
   break
  else:raise RuntimeError('geometry')
  seenL.add(o);pos.append(dict(id=f'L{L}_g{g}v{v}',moves=moves,source_action=None))
old=read(O/f'REGISTRATION_L{L}.json');rt=S/'runtime'/f'mqft-L{L}'
assert old['model']==verify(P/'.external/rapfi')
# Actual semantic gates on a prior root, independent of new samples.
j=read(O/'plans'/f'B16-L{L}.json')['jobs'][0];sp=copy.deepcopy(j['spec']);sp.update(budget=8,cuts=[1,2,4,8]);sp['root_law'].pop('logit_schedule',None);sp['root_law']['logit_h']=.25
moves=j['opening'];group=sp['root_law']['source_actions']
def sector(a):return 2*(a//L>=c)+int(a%L>=c)
k=sector(group[0]);assert all(sector(x)==k for x in group)
v=copy.deepcopy(sp);v['root_law'].pop('source_actions');v['root_law']['logit_h']=0.;v['root_law']['sector_schedule']=[[.25 if i==k else 0. for i in range(4)]]*8
def clean(z):
 if isinstance(z,dict):return {k:clean(v) for k,v in z.items() if k not in ('worker','wall_ns','elapsed_ns','schema','spec','source','source_actions','logit_h','sector_logits')}
 if isinstance(z,list):return [clean(x) for x in z]
 return z
def eq(a,b):
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 if isinstance(a,float) or isinstance(b,float):return bool(np.isclose(a,b,rtol=1e-10,atol=1e-12))
 return a==b
with Backend(rt,90,P/'.external/rapfi',4,diagnostic_path=S/'logs'/f'gate-L{L}.log') as be:
 a=be.search(moves,sp);check_trace(a,moves,sp)
 b=be.search(moves,v);check_trace(b,moves,v);assert eq(clean(a),clean(b))
 w=copy.deepcopy(v);w['root_law']['sector_schedule']=[[x+1.25 for x in y] for y in v['root_law']['sector_schedule']]
 z=be.search(moves,w);check_trace(z,moves,w);assert eq(clean(b),clean(z))
 for bad in ([[0.]*4],[[0.,1.,2.]]*8,[[0.,0.,0.,11.]]*8):
  w=copy.deepcopy(v);w['root_law']['sector_schedule']=bad
  try:be.search(moves,w)
  except ValueError:pass
  else:raise AssertionError('bad vector')
atomic_json(S/f'GATE_L{L}.json',dict(backend_sha=sha256(rt),group_equiv=True,common_shift=True,rejected=3))
base=dict(kind='roots',c_puct=[1.4],sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**30,search_timeout=90,max_plies=L*L)
plans={};delta=.3
arms=[('base',None,[0,0,0])]
for i in range(3):
 for sign in (-1,1):
  x=np.zeros(3);x[i]=sign;arms.append((f'd{i}_{sign}',3,x.tolist()))
for s in (-1,1):
 for t in (-1,1):arms.append((f'mix{s}_{t}',3,[s,t,0]))
for s in (-1,1):arms.append((f'early{s}',0,[s,0,0]))
assert len(arms)==13
def roots(name,positions,reps,seed,pulse):
 modes=[]
 for law in ('iid','pl','reserve'):
  for tag,block,x in arms if pulse else arms[:1]:
   h=(delta*U@x).tolist();schedule=[h if block is not None and i//4==block else [0.]*4 for i in range(16)]
   modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law=dict(kind=law,beta_p=[10/3],beta_q=[10/3],proposal='p',logit_h=0.,sector_schedule=schedule)))
 pl=make_plan({**base,'seed':seed,'replicates':reps,'budgets':[16],'positions':positions,'modes':modes})
 assert len(pl['jobs'])==len(positions)*3*reps*(13 if pulse else 1)
 atomic_json(S/'plans'/f'{name}-L{L}.json',pl);plans[name]=dict(hash=pl['plan_hash'],jobs=len(pl['jobs']),reps=reps)
roots('pilot',pos,2,53000+L,False)
roots('g0',[pos[0]],8,53100+L,True)
roots('g1',[pos[2]],4,53200+L,True)
roots('variants',[pos[1],pos[3]],4,53300+L,False)
cfg=dict(kind='games',seed=53400+L,replicates=1,budgets=[16],c_puct=[1.4],modes=[dict(mode='serial',workers=1,samples_per_eval=4)],openings=[pos[1]['moves']],opponents=['self'],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,move_temperature=0.,max_plies=L*L,jobs=1,cpu_limit=2,max_output_bytes=2**28,search_timeout=90)
pl=make_plan(cfg);atomic_json(S/'plans'/f'game-L{L}.json',pl);plans['game']=dict(hash=pl['plan_hash'],jobs=1)
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,positions=pos,templates15=templates,density=dense/L**2,basis=U.tolist(),delta=delta,arms=arms,backend_sha=sha256(rt),model=old['model'],source_manifest_sha=sha256(S/'family'/f'L{L}'/'SOURCE_MANIFEST.json'),plans=plans,scope='two fresh common core templates and two peripheral variants, pilot h0 separate; fixed vector directions no pilot-driven selection; B16 late axes and mixed01, early axis0; n8/n4'))
print(L,plans,flush=True)
