from pathlib import Path
import json,sys,math,gzip
import numpy as np
from mqft.campaign import make_plan,atomic_json,valid_moves,win
from mqft.client import sha256
from mqft.rapfi import verify
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore49_20261126'
L=int(sys.argv[1]);mode=sys.argv[2];c=L//2
read=lambda p:json.loads(p.read_text())
def orbit(m,L):
 a=np.zeros((L,L),int)
 for i,x in enumerate(m):a.ravel()[x]=1+i%2
 return min(tuple(x.ravel()) for k in range(4) for x in (np.rot90(a,k),np.fliplr(np.rot90(a,k))))
def center(a,L):y,x=divmod(a,15);return (y-7+L//2)*L+x-7+L//2
base=dict(kind='roots',c_puct=[1.4],sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**30,search_timeout=90,max_plies=L*L)
def roots(name,pos,B0,rep,seed,pulses=False):
 modes=[]
 for law in ('iid','pl','reserve'):
  com=dict(kind=law,beta_p=[10/3],beta_q=[10/3],proposal='p',logit_h=0.)
  modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law=com))
  if pulses:
   for j in (0,B0//8,B0//4-1):
    for h in (-.2,.2):
     sched=[h if 4*j<=k<4*j+4 else 0. for k in range(B0)]
     modes.append(dict(mode='wave',workers=4,samples_per_eval=4,root_law={**com,'logit_schedule':sched}))
 cfg={**base,'seed':seed,'replicates':rep,'budgets':[B0],'positions':pos,'modes':modes}
 a=make_plan(cfg);assert len(a['jobs'])==len(pos)*3*rep*(7 if pulses else 1)
 groups={}
 for j in a['jobs']:groups.setdefault((j['position_id'],j['replicate'],j['spec']['root_law']['kind']),[]).append(j)
 assert all(len({x['spec']['seed'] for x in g})==1 for g in groups.values())
 atomic_json(S/'plans'/f'{name}-L{L}.json',a)
 return dict(hash=a['plan_hash'],jobs=len(a['jobs']),B=B0,reps=rep)
old=read(O/f'REGISTRATION_L{L}.json');exe=O/'runtime_verified'/f'mqft-L{L}'
assert sha256(exe)==old['backend_sha'] and old['model']==verify(P/'.external/rapfi')
regp=S/f'REGISTRATION_L{L}.json'
if mode=='pilot':
 rng=np.random.default_rng(510001);allowed=[y*15+x for y in range(4,11) for x in range(4,11) if (y,x)!=(7,7)]
 prev=[]
 for n in (44,45):prev+=read(B/f'mqft_restore{n}_20261126/plans/marked.json')['config']['positions']
 prev+=read(B/'mqft_restore47_20261126/plans/heldout64.json')['config']['positions']
 seen={orbit(x['moves'],15) for x in prev}
 seen|={orbit(m,15) for m in read(B/'mqft_restore50_20261126/REGISTRATION_L15.json')['templates15']}
 templates=[]
 while len(templates)<3:
  m=[112]+rng.choice(allowed,7,replace=False).tolist()
  if orbit(m,15) in seen:continue
  seen.add(orbit(m,15));templates.append(m)
 positions=[];dense_n=2*math.floor(.03*L*L+.5)
 seenL={orbit(x['moves'],L) for x in old['anchors']}|{orbit(x['moves'],L) for x in read(B/f'mqft_restore50_20261126/REGISTRATION_L{L}.json')['positions']}
 for i,m in enumerate(templates):
  core=[center(x,L) for x in m];rng=np.random.default_rng(511000+L*10+i)
  pool=[y*L+x for y in range(L) for x in range(L) if max(abs(y-c),abs(x-c))>=4]
  for v in range(2):
   for attempt in range(1000):
    mv=core+rng.choice(pool,dense_n-8,replace=False).tolist()
    try:
     b=valid_moves(mv);o=orbit(mv,L);assert o not in seenL and not win(b,1) and not win(b,2)
    except (AssertionError,ValueError):continue
    break
   else:raise RuntimeError('geometry')
   seenL.add(o);positions.append(dict(id=f'L{L}_g{i}v{v}',moves=mv,source_action=next(j for j,x in enumerate(b) if x==0)))
 assert len(positions)==6
 plan=roots('pilot',positions,16,2,51100+L)
 reg=dict(L=L,parent='3ac8d282abdb15231e7c7cc8050c4bf48f0ab824',positions=positions,templates15=templates,density=dense_n/L**2,backend_sha=old['backend_sha'],source_manifest_sha=old['source_manifest_sha'],model=old['model'],plans={'pilot':plan},
 scope='3 new central templates, two distinct peripheral variants each, approximate density .06. pilot h0 only, select by predictable Fisher before any perturbed data. Confirm independent seed namespace.')
 atomic_json(regp,reg)
else:
 assert mode=='confirm'
 reg=read(regp);assert set(reg['plans'])=={'pilot'}
 d=S/'data'/f'pilot-L{L}';rp=read(next(d.glob('report-*.json')));assert len(rp['completed'])==36 and not rp['failures'] and not rp['unattempted']
 by={x['id']:[] for x in reg['positions']}
 for t in rp['completed']:
  f=d/t['file'];assert sha256(f)==t['sha256']
  with gzip.open(f,'rt') as g:hd,x,e=[json.loads(l) for l in g]
  assert hd['job']['spec']['root_law']['logit_h']==0 and hd['backend']['board_size']==L
  tr=x['trace'];rep=tr['replicas'][0];ev=sorted(rep['events'],key=lambda x:x['root_selection']['clock'])
  a=np.zeros((16,L*L))
  for i,z in enumerate(ev):
   q=z['root_selection']['law'];a[i,q['actions']]=q['p']
  assert np.allclose(a.sum(1),1)
  by[hd['job']['position_id']].append(dict(prob=a,seed=hd['job']['spec']['seed'],job=t['job_id'],sha=t['sha256']))
 choices={};high=[]
 for x in reg['positions']:
  q=by[x['id']];assert len(q)==6
  A=np.array([z['prob'] for z in q]);F=(A*(1-A)).sum(1).mean(0);H=A.sum(1).mean(0)
  legal=[i for i,z in enumerate(valid_moves(x['moves'])) if z==0]
  order=sorted(legal,key=lambda i:(-F[i],i));hi=order[0]
  med=min(order[1:],key=lambda i:(abs(F[i]-.5*F[hi]),i))
  rare=min([i for i in order if i not in (hi,med)],key=lambda i:(abs(F[i]-.02*F[hi]),i))
  selected={k:dict(action=i,pilot_F=float(F[i]),pilot_hits=float(H[i])) for k,i in [('high',hi),('medium',med),('rare',rare)]}
  choices[x['id']]=dict(strata=selected,pilot_jobs=[{k:z[k] for k in ('seed','job','sha')} for z in q])
  high.append({**x,'source_action':hi})
 # Registration is sealed before execution of confirmation and pulse.
 reg['source_selection']=choices;reg['selection_rule']='F_a=mean pilot sum_t p_t(a)(1-p_t(a)); maximum, nearest .5 max, nearest .02 max (distinct, deterministic action tie-break); 2 seeds/law, h0 only'
 reg['plans']['confirm']=roots('confirm',high,16,8,51200+L)
 if L in (13,21):
  x=reg['positions'][0]
  for i,tag in enumerate(('high','medium','rare')):
   pos=[{**x,'source_action':choices[x['id']]['strata'][tag]['action']}]
   reg['plans'][f'pulse_{tag}']=roots(f'pulse_{tag}',pos,16,8,51300+10*L+i,True)
  pos=[{**x,'source_action':choices[x['id']]['strata']['high']['action']}]
  reg['plans']['pulse32_high']=roots('pulse32_high',pos,32,4,51400+L,True)
 cfg=dict(kind='games',seed=51500+L,replicates=1,budgets=[16],c_puct=[1.4],modes=[dict(mode='serial',workers=1,samples_per_eval=4)],openings=[high[0]['moves']],opponents=['self'],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,move_temperature=0.,max_plies=L*L,jobs=1,cpu_limit=2,max_output_bytes=2**28,search_timeout=90)
 a=make_plan(cfg);atomic_json(S/'plans'/f'game-L{L}.json',a);reg['plans']['game']=dict(hash=a['plan_hash'],jobs=1,B=16,reps=1)
 atomic_json(regp,reg)
print(L,mode,[(k,v['jobs']) for k,v in reg['plans'].items()],flush=True)
