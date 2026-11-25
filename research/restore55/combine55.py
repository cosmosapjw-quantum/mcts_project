"""Receipt-checked native longitudinal physics and conditional references."""
from pathlib import Path
import json,gzip,math,statistics as st
from collections import defaultdict,Counter
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics53 import U
from mqft.physics50 import sectors
from mqft.physics55 import paired_controls,learn_mark,restricted_vector,quantum_records,pointer_dynamics,gibbs_fdt,noneq_fdt_control
from mqft.physics44 import exact_wave_full_state,projected_information
from mqft.physics45 import lattice_action
S=Path(__file__).resolve().parent;B=S.parent;read=lambda p:json.loads(p.read_text())
def raw(r):
 p=S/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,x,e=[json.loads(l) for l in f]
 assert h['job']['position_id']==r['position'];return x['trace']
def main():
 a=[read(S/f'PHYSICS_L{L}.json') for L in (13,21)];native=sum([x['native'] for x in a],[]);games=sum([x['games'] for x in a],[]);long=sum([x['longitudinal'] for x in a],[])
 counts={f"L{x['L']}":x['counts'] for x in a}
 assert len(native)==480 and len(games)==8 and len(long)==sum(x['searched'] for x in games)
 assert sum(v.get('exact_prefix_pairs',0) for c in counts.values() for v in c.values())==192
 assert sum(v.get('exact_prefix_draws',0) for c in counts.values() for v in c.values())==2304
 seeds=defaultdict(list)
 for r in native:seeds[r['seed']].append(r)
 assert len(seeds)==96 and all(len({(r['L'],r['campaign'],r['position'],r['law']) for r in v})==1 for v in seeds.values())
 for n in (53,54):
  for L in (13,21):assert not {r['seed'] for r in native}&{r['seed'] for r in read(B/f'mqft_restore{n}_20261126'/f'PHYSICS_L{L}.json')['native']}
 arms=defaultdict(lambda:defaultdict(list))
 for r in native:arms[r['L'],r['campaign'],r['law']][tuple(r['arm'])].append(r)
 assert len(arms)==12 and all(len(x)==5 and all(len(v)==8 for v in x.values()) for x in arms.values())
 control=[];refs=[];loops=[];models=[]
 for (L,c,law),rr in sorted(arms.items()):
  train=arms[L,'root1' if c=='root0' else 'root0',law][-1,0.]
  assert not {r['seed'] for r in train}&{r['seed'] for r in rr[-1,0.]}
  m=learn_mark(train);models.append(dict(L=L,campaign=c,law=law,train_jobs=[r['job'] for r in train],mark_model=m.tolist()))
  control.extend(dict(L=L,campaign=c,law=law,**z) for z in paired_controls(rr,m))
  # Independent finite restricted wave on common IID completed k8 root;
  # select 1 top-prior legal action/sector plus next global action.
  ref=sorted(arms[L,c,'iid'][-1,0.],key=lambda r:r['seed'])[0];t=raw(ref)
  edges=next(x['edges'] for x in t['replicas'][0]['cuts'] if x['local_k']==8)
  aa=np.array([e['action'] for e in edges]);pr=np.array([e['prior'] for e in edges]);ss=sectors(L,aa)
  chosen=[int(aa[np.flatnonzero(ss==k)[np.argmax(pr[ss==k])]]) for k in range(4)]
  chosen+=[int(aa[i]) for i in np.argsort(-pr) if aa[i] not in chosen][:1];assert len(set(chosen))==5
  eta=U.T@np.array([1.,-1.,0.,0.]);z=restricted_vector(edges,chosen,L,law,np.zeros(3))
  plus,minus=[restricted_vector(edges,chosen,L,law,eta*h) for h in (.25,-.25)]
  d=(plus['mean']-minus['mean'])/.5
  dd=[restricted_vector(edges,chosen,L,law,eta*h)['mean'] for h in (1e-5,-1e-5)]
  err=float(np.max(abs((dd[0]-dd[1])/2e-5-z['R']@eta)));assert err<1e-8
  # Naive iid frozen reference is evaluated on the same support/parent.
  f0=restricted_vector(edges,chosen,L,'iid',eta*.25);f1=restricted_vector(edges,chosen,L,'iid',-eta*.25)
  fd=(f0['mean']-f1['mean'])/.5
  refs.append(dict(L=L,campaign=c,law=law,source_job=ref['job'],support=chosen,words=len(z['p']),response=(z['R']@eta).tolist(),finite=d.tolist(),frozen_iid=fd.tolist(),frozen_gap=float(np.linalg.norm(d-fd)),derivative_error=err,F=z['F'].tolist()))
  q=exact_wave_full_state(edges,chosen,4,law,chosen[0]);v=lattice_action(q,(8,16,32),True)
  loops.append(dict(L=L,campaign=c,law=law,support=chosen,source_job=ref['job'],values=v,Fisher=projected_information(q),errors=q['errors']))
 assert len(control)==60 and len(refs)==len(loops)==12
 # Actual complete game plies as observational units; same opening
 # across budgets is matched, no claim of iid ply-level observations.
 long.sort(key=lambda x:(x['L'],x['B'],x['opening_id'],x['ply']))
 gs=[];phase=[];qstates={};quantum=[];thermal=[]
 for g in sorted(games,key=lambda x:(x['L'],x['B'],x['opening_id'])):
  rr=[r for r in long if r['game']==g['job']];assert len(rr)==g['searched'] and all(b['ply']==a['ply']+1 for a,b in zip(rr,rr[1:]))
  assert rr[0]['ply']==len(g['opening']) and rr[-1]['ply']==g['plies']-1
  n=len(rr)
  summary={k:st.mean(x[k] for x in rr) for k in ('H','H_fraction','q','black_q','variance','order','prior_KL','support','conditional_H','kkt_gap')}
  fits={f:dict(Counter(x['fits'][f]['status'] for x in rr)) for f in ('Q_all','Q_visited','PUCT_all')}
  gs.append(dict(**g,**summary,fit_status=fits,max_depth=max(len(x['depth']) for x in rr)))
  for k in range(3):
   ix=[i for i in range(n) if min(2,3*i//n)==k]
   if not ix:continue
   z=[rr[i] for i in ix];phase.append(dict(game=g['job'],L=g['L'],B=g['B'],opening_id=g['opening_id'],phase=k,n=len(z),**{k0:st.mean(x[k0] for x in z) for k0 in ('H','H_fraction','q','black_q','variance','order','prior_KL','support','conditional_H')},finite_Q=st.mean(x['fits']['Q_all']['status']=='finite' for x in z),finite_PUCT=st.mean(x['fits']['PUCT_all']['status']=='finite' for x in z)))
  # Three prospectively defined clock locations, first/middle/last.
  for tag,i in [('first',0),('middle',n//2),('last',n-1)]:
   r=rr[i];p=np.array(r['sector_probability']);assert min(p)>0
   key=f"{g['job']}_{tag}";pd,states=pointer_dynamics(p)
   for t,s0 in zip(pd['rows'],states):qstates[key+f"_pointer_{t['t']:g}"]=s0
   for strength in (.2,.5,.8):
    q,s0=quantum_records(p,strength);qstates[key+f"_record_{strength:g}"]=s0
    quantum.append(dict(game=g['job'],L=g['L'],B=g['B'],tag=tag,ply=r['ply'],source_sha=r['shard_sha'],p=p.tolist(),**q))
   thermal.append(dict(game=g['job'],L=g['L'],B=g['B'],tag=tag,ply=r['ply'],source_sha=r['shard_sha'],equilibrium=gibbs_fdt(p),nonstationary=noneq_fdt_control(p),pointer=pd))
 assert len(quantum)==72 and len(thermal)==24 and len(qstates)==216
 np.savez_compressed(S/'CONDITIONAL_STATES.npz',**qstates)
 out=dict(counts=counts,games=gs,longitudinal=long,phase=phase,control=control,control_models=models,restricted=refs,loops=loops,quantum=quantum,thermal=thermal,states_sha=sha256(S/'CONDITIONAL_STATES.npz'),fresh_histories=len(native),unique_seed_groups=len(seeds),size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in a},scope='native longitudinal and sequential martingale controls, restricted exact wave laws, declared quantum records and equilibrium Gibbs FDT')
 atomic_json(S/'PHYSICS_RESULTS.json',out);print('COMBINED',len(native),len(long),len(control),len(quantum),len(thermal),flush=True)
if __name__=='__main__':main()
