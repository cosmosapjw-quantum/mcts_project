from pathlib import Path
import json,gzip,sys,copy
import numpy as np
from mqft.campaign import make_plan,atomic_json,valid_moves
from mqft.client import Backend,sha256
from mqft.analysis import check_trace
from mqft.rapfi import verify
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore51_20261126';F=B/'mqft_restore49_20261126'
L=int(sys.argv[1]);c=L//2
read=lambda p:json.loads(p.read_text())
reg=read(O/f'REGISTRATION_L{L}.json');rt=S/'runtime'/f'mqft-L{L}';old=F/'runtime_verified'/f'mqft-L{L}'
assert sha256(old)==reg['backend_sha'] and verify(P/'.external/rapfi')==reg['model']
def sector(a):return 2*(a//L>=c)+int(a%L>=c)
pos=[];selection={}
for g in (0,1):
 x=next(x for x in reg['positions'] if x['id']==f'L{L}_g{g}v0');p=valid_moves(x['moves']);legal=[i for i,v in enumerate(p) if not v]
 # use only independent R51 h0 confirmation, never treatment arms
 plan=read(O/'plans'/f'confirm-L{L}.json');d=O/'data'/f'confirm-L{L}'
 J=[j for j in plan['jobs'] if j['position_id']==x['id']];assert len(J)==24
 A=[];ids=[]
 for j in J:
  f=d/(j['id']+'.jsonl.gz');rc=read(d/(j['id']+'.receipt.json'));assert sha256(f)==rc['sha256']
  with gzip.open(f,'rt') as z:hd,t,end=[json.loads(l) for l in z]
  assert hd['job']==j and j['spec']['root_law']['logit_h']==0
  ev=sorted(t['trace']['replicas'][0]['events'],key=lambda e:e['root_selection']['clock'])
  A.append([np.bincount([sector(a) for a in e['root_selection']['law']['actions']],weights=e['root_selection']['law']['p'],minlength=4).tolist() for e in ev]);ids.append(rc['sha256'])
 A=np.array(A);H=A.sum(1).mean(0);I=(A*(1-A)).sum(1).mean(0);q=int(np.argmax(I))
 group=[a for a in legal if sector(a)==q]
 selection[x['id']]=dict(sector=q,actions=group,pilot_F=I.tolist(),pilot_hits=H.tolist(),source_jobs=[j['id'] for j in J],shards=ids,absolute_design_pass=bool(I[q]>=1.0 and H[q]>=2.))
 pos.append(dict(id=x['id'],moves=x['moves'],source_actions=group))
# Live semantic regression: scalar vs singleton with exact old archived seed,
# group vs complement negative logit, and neutral group vs no-field.
x=pos[0];j=next(j for j in plan['jobs'] if j['position_id']==x['id'] and j['spec']['root_law']['kind']=='iid')
sp=copy.deepcopy(j['spec']);sp['budget']=8;sp['cuts']=[1,2,4,8];sp['root_law']['logit_h']=.2
def compare(a,b):
 # Ignore trace identity, requested source metadata, time and worker only.
 def clean(z):
  if isinstance(z,dict):return {k:clean(v) for k,v in z.items() if k not in ('worker','wall_ns','elapsed_ns','schema','spec','source','source_actions','logit_h')}
  if isinstance(z,list):return [clean(v) for v in z]
  return z
 def eq(a,b):
  if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
  if isinstance(a,list):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
  if isinstance(a,float) or isinstance(b,float):return bool(np.isclose(a,b,rtol=1e-10,atol=1e-12))
  return a==b
 assert eq(clean(a),clean(b)),'semantic mismatch'
with Backend(rt,90,P/'.external/rapfi',4,diagnostic_path=S/'logs'/f'gate-L{L}.log') as be:
 a=be.search(x['moves'],sp);check_trace(a,x['moves'],sp)
 b=copy.deepcopy(sp);b['root_law']['source_actions']=[b['source_action']];b['source_action']=None
 t=be.search(x['moves'],b);check_trace(t,x['moves'],b);compare(a,t)
 b['root_law']['source_actions']=x['source_actions'];t=be.search(x['moves'],b);check_trace(t,x['moves'],b)
 k=copy.deepcopy(b);k['root_law']['source_actions']=[a for a in t['root']['legal'] if a not in x['source_actions']];k['root_law']['source_actions'].sort();k['root_law']['logit_h']=-.2
 u=be.search(x['moves'],k);check_trace(u,x['moves'],k);compare(t,u)
 # malformed group must be rejected by Rust (not just Python).
 for bad in ([x['source_actions'][0]]*2,[L*L],sorted(t['root']['legal'])):
  k=copy.deepcopy(b);k['root_law']['source_actions']=bad
  try:be.search(x['moves'],k)
  except ValueError:pass
  else:raise AssertionError('invalid group accepted')
atomic_json(S/f'GATE_L{L}.json',dict(singleton_scalar=True,complement_gauge=True,invalid_groups_rejected=3,backend_sha=sha256(rt)))
base=dict(kind='roots',c_puct=[1.4],sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**29,search_timeout=90,max_plies=L*L)
plans={}
for B0,reps,ps in [(16,8,pos),(32,4,pos[:1])]:
 modes=[]
 for law in ('iid','pl','reserve'):
  com=dict(kind=law,beta_p=[10/3],beta_q=[10/3],proposal='p',logit_h=0.)
  modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law=com))
  for j in ((0,B0//4-1) if B0==16 else (B0//4-1,)):
   for h in (-.2,.2):
    modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law={**com,'logit_schedule':[h if 4*j<=i<4*j+4 else 0. for i in range(B0)]}))
 cfg={**base,'seed':52000+100*L+B0,'replicates':reps,'budgets':[B0],'positions':ps,'modes':modes}
 pl=make_plan(cfg);assert len(pl['jobs'])==len(ps)*3*reps*(5 if B0==16 else 3)
 atomic_json(S/'plans'/f'B{B0}-L{L}.json',pl);plans[f'B{B0}']=dict(hash=pl['plan_hash'],jobs=len(pl['jobs']),B=B0,reps=reps)
# same actual common evaluator full game, independent prefix
x=next(x for x in reg['positions'] if x['id']==f'L{L}_g2v1')
cfg=dict(kind='games',seed=52500+L,replicates=1,budgets=[16],c_puct=[1.4],modes=[dict(mode='serial',workers=1,samples_per_eval=4)],openings=[x['moves']],opponents=['self'],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,move_temperature=0.,max_plies=L*L,jobs=1,cpu_limit=2,max_output_bytes=2**28,search_timeout=90)
pl=make_plan(cfg);atomic_json(S/'plans'/f'game-L{L}.json',pl);plans['game']=dict(hash=pl['plan_hash'],jobs=1)
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,positions=pos,selection=selection,backend_sha=sha256(rt),model=reg['model'],source_manifest_sha=sha256(S/'family'/f'L{L}'/'SOURCE_MANIFEST.json'),plans=plans,scope='prior independent h0 pilot, fixed absolute geometry roots g0v0/g1v0, group field not single action; new seeds, finite pulse pilot'))
print('REGISTERED',L,plans,[(k,v['sector'],v['pilot_F'][v['sector']],v['pilot_hits'][v['sector']]) for k,v in selection.items()],flush=True)
