from pathlib import Path
import json,sys,math
import numpy as np
from mqft.campaign import make_plan,atomic_json,valid_moves,win
from mqft.client import sha256
from mqft.rapfi import verify
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore56_20261126';F=B/'mqft_restore53_20261126'
L=int(sys.argv[1]);c=L//2
read=lambda p:json.loads(p.read_text())
def orbit(m,L):
 a=np.zeros((L,L),int)
 for i,x in enumerate(m):a.ravel()[x]=1+i%2
 return min(tuple(x.ravel()) for k in range(4) for x in (np.rot90(a,k),np.fliplr(np.rot90(a,k))))
seen=set()
for n in (50,51,53,54,55,56):
 f=B/f'mqft_restore{n}_20261126'/('REGISTRATION_L15.json' if n==50 else 'REGISTRATION_L13.json')
 a=read(f)
 for m in a.get('templates15',[a.get('template15')]):
  if m:seen.add(orbit(m,15))
for n in (44,45):
 for x in read(B/f'mqft_restore{n}_20261126/plans/marked.json')['config']['positions']:seen.add(orbit(x['moves'],15))
for x in read(B/'mqft_restore47_20261126/plans/heldout64.json')['config']['positions']:seen.add(orbit(x['moves'],15))
rng=np.random.default_rng(570001);allowed=[y*15+x for y in range(4,11) for x in range(4,11) if y*15+x!=112];templates=[]
while len(templates)<2:
 m=[112]+rng.choice(allowed,7,replace=False).tolist()
 if orbit(m,15) in seen:continue
 seen.add(orbit(m,15));templates.append(m)
pos=[];dense=2*math.floor(.03*L*L+.5)
seenL={orbit(x['moves'],L) for n in (50,51,53,54,55,56) for x in read(B/f'mqft_restore{n}_20261126/REGISTRATION_L{L}.json')['positions']}
for g,m in enumerate(templates):
 core=[(a//15-7+c)*L+a%15-7+c for a in m];rng=np.random.default_rng(571000+100*L+g)
 pool=[a for a in range(L*L) if max(abs(a//L-c),abs(a%L-c))>=4]
 for v in range(1000):
  moves=core+rng.choice(pool,dense-8,replace=False).tolist()
  try:
   b=valid_moves(moves);o=orbit(moves,L);assert o not in seenL and not win(b,1) and not win(b,2)
  except (AssertionError,ValueError):continue
  break
 else:raise RuntimeError('geometry')
 seenL.add(o);pos.append(dict(id=f'L{L}_g{g}',moves=moves,source_action=None))
old=read(O/f'REGISTRATION_L{L}.json');rt=F/'runtime'/f'mqft-L{L}';q=B/'mqft_restore54_20261126/python'/f'L{L}';assert sha256(rt)==old['backend_sha'] and old['model']==verify(P/'.external/rapfi')
pm=read(q/'PATCH_MANIFEST.json');assert all(sha256(q/f)==v for f,v in pm['files'].items())
base=dict(c_puct=[1.4],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,jobs=1,cpu_limit=8,max_output_bytes=2**30,search_timeout=90,max_plies=L*L)
plans={}
def register(name,cfg):
 a=make_plan(cfg);atomic_json(S/'plans'/f'{name}-L{L}.json',a);plans[name]=dict(hash=a['plan_hash'],jobs=len(a['jobs']))
for g in range(2):
 modes=[]
 for law in ('iid','pl','reserve'):
  common=dict(kind=law,beta_p=[10/3],beta_q=[10/3],proposal='p',logit_h=0.)
  for j,h in ((-1,0.),(0,-.25),(0,.25)):
   # Exact binary group contrast h*(I_sector0 - I_sector1),
   # no need for irrational serialized Helmert coefficient.
   sched=[[h,-h,0.,0.] if i//4==j else [0.]*4 for i in range(16)]
   modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law={**common,'sector_schedule':sched}))
 register(f'root{g}',{**base,'kind':'roots','seed':57200+100*L+g,'replicates':8,'budgets':[16],'sources':[0.],'positions':[pos[g]],'paired_schedule_seed':True,'modes':modes})
assert sum(v["jobs"] for v in plans.values())==144
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,positions=pos,templates15=templates,density=dense/L**2,backend_sha=sha256(rt),model=old['model'],source_manifest_sha=old['source_manifest_sha'],analysis_patch_manifest_sha=sha256(q/'PATCH_MANIFEST.json'),plans=plans,scope='two new common core geometries; two roots per size n8/law/arm early contrast; frozen R56 mark models with R55 training only; no selector tuning'))
print(L,plans,flush=True)
