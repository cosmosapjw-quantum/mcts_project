"""Combine actual R56 games/roots and prior R55 training only.
Use exact archived R54 work samples for a dedicated original Q19 panel."""
from pathlib import Path
import json,gzip,statistics as st,math
from collections import defaultdict,Counter
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics56 import action_features,fit_mark,evaluate_mark,finite_control_samples,coefficients,score_controls,erasure,driven_response
from mqft.physics44 import exact_wave_full_state,projected_information
from mqft.physics45 import lattice_action
from mqft.physics50 import sectors
S=Path(__file__).resolve().parent;B=S.parent;O=B/'mqft_restore55_20261126';F=B/'mqft_restore54_20261126'
read=lambda p:json.loads(p.read_text())
def raw(stage,r):
 p=stage/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,x,e=[json.loads(l) for l in f]
 assert h['job']['id']==r['job'];return x['trace']
def group(rows):
 a=defaultdict(lambda:defaultdict(list))
 for r in rows:a[r['L'],r['campaign'],r['law']][tuple(r['arm'])].append(r)
 return a
def phase_summary(games,long,source):
 gs=[];ph=[]
 for g in games:
  rr=sorted([x for x in long if x['game']==g['job']],key=lambda r:r['ply']);n=len(rr)
  assert n==g['searched'] and n>0 and rr[0]['ply']==len(g['opening']) and rr[-1]['ply']==g['plies']-1
  metrics=('H','H_fraction','variance','order','black_q','prior_KL','support','conditional_H','kkt_gap')
  q={k:st.mean(x[k] for x in rr) for k in metrics};q.update({f'finite_{f}':st.mean(x['fits'][f]['status']=='finite' for x in rr) for f in ('Q_all','Q_visited','PUCT_all')})
  gs.append(dict(**{k:g[k] for k in ('L','B','job','opening_id','plies','searched','winner','status','opening','moves','shard_sha')},source=source,**q,max_depth=max(len(x['depth']) for x in rr)))
  for j in range(3):
   a=[x for i,x in enumerate(rr) if min(2,3*i//n)==j]
   if a:ph.append(dict(game=g['job'],source=source,L=g['L'],B=g['B'],opening_id=g['opening_id'],phase=j,n=len(a),**{k:st.mean(x[k] for x in a) for k in metrics},finite_Q=st.mean(x['fits']['Q_all']['status']=='finite' for x in a)))
 return gs,ph
def main():
 sizes=[read(S/f'PHYSICS_L{L}.json') for L in (13,21)]
 native=sum([x['native'] for x in sizes],[]);games=sum([x['games'] for x in sizes],[]);long=sum([x['longitudinal'] for x in sizes],[])
 assert len(native)==144 and len(games)==36 and len(long)==sum(g['searched'] for g in games)
 counts={f"L{x['L']}":x['counts'] for x in sizes};seeds=defaultdict(list)
 for r in native:seeds[r['seed']].append(r)
 assert len(seeds)==48 and all(len(v)==3 and len({(r['L'],r['campaign'],r['law']) for r in v})==1 for v in seeds.values())
 old=read(O/'PHYSICS_RESULTS.json');pub=read(O/'PUBLICATION_RECEIPT.json');assert sha256(O/'PHYSICS_RESULTS.json')==pub['physics_results_sha256']
 oldnative=[]
 for L in (13,21):
  p=O/f'PHYSICS_L{L}.json';assert sha256(p)==old['size_results_sha'][f'L{L}'];oldnative.extend(read(p)['native'])
 assert not {r['seed'] for r in native}&{r['seed'] for r in oldnative}
 tr=group(oldnative);te=group(native);models=[];controls=[];prediction=[]
 for L in (13,21):
  for law in ('iid','pl','reserve'):
   trains=[tr[L,c,law] for c in ('root0','root1')];tests=[te[L,c,law] for c in ('root0','root1')]
   train0=[r for t in trains for r in t[-1,0.]];rows=[]
   for r in train0:rows.extend(action_features(raw(O,r)))
   model=fit_mark(rows);assert model['n']==256
   train_e={r['job']:evaluate_mark(action_features(raw(O,r)),model,L,16) for t in trains for k in ((-1,0.),(0,-.25),(0,.25)) for r in t[k]}
   coef=coefficients([finite_control_samples(t,train_e) for t in trains])
   models.append(dict(L=L,law=law,train_jobs=sorted(train_e),model=model,coefficients=coef.tolist(),scope='R55 only; no R56 fitting'))
   for c,t in zip(('root0','root1'),tests):
    e={r['job']:evaluate_mark(action_features(raw(S,r)),model,L,16) for v in t.values() for r in v}
    controls.extend(dict(L=L,law=law,campaign=c,**z) for z in score_controls(finite_control_samples(t,e),coef))
    for r in t[-1,0.]:
     q=e[r['job']];pred=np.array(q['pred']);y=np.array(q['marks'])
     prediction.append(dict(L=L,law=law,campaign=c,job=r['job'],mse=np.mean((pred-y[:,None])**2,0).tolist(),names=['zero','sector','action'],n=len(y)))
 assert len(controls)==72 and len(prediction)==48 and len(models)==6
 # Conditional references on 4 new roots, all three laws, same actual IID snapshot.
 loops=[]
 for (L,c,law),t in sorted(te.items()):
  ref=sorted(te[L,c,'iid'][-1,0.],key=lambda r:r['seed'])[0];trace=raw(S,ref)
  edges=next(x['edges'] for x in trace['replicas'][0]['cuts'] if x['local_k']==8)
  aa=np.array([e['action'] for e in edges]);pr=np.array([e['prior'] for e in edges]);ss=sectors(L,aa)
  chosen=[int(aa[np.flatnonzero(ss==k)[np.argmax(pr[ss==k])]]) for k in range(4)]
  chosen+=[int(aa[i]) for i in np.argsort(-pr) if aa[i] not in chosen][:1]
  q=exact_wave_full_state(edges,chosen,4,law,chosen[0]);loops.append(dict(L=L,campaign=c,law=law,support=chosen,source_job=ref['job'],values=lattice_action(q,(8,16,32),True),Fisher=projected_information(q)))
 assert len(loops)==12
 gs,ph=phase_summary(games,long,'R56');oldg,oldph=phase_summary(old['games'],old['longitudinal'],'R55')
 allg=oldg+gs;allph=oldph+ph
 # Same opening at two budgets, pair by exact opening list and stage.
 # Report fixed-design bootstrap descriptive intervals, no iid plies.
 paired=[];rng=np.random.default_rng(566000)
 for L in (13,21):
  for low,high in ((16,32),(32,64)):
   lo={(g['source'],tuple(g['opening'])):g for g in allg if g['L']==L and g['B']==low}
   hi={(g['source'],tuple(g['opening'])):g for g in allg if g['L']==L and g['B']==high}
   keys=sorted(set(lo)&set(hi));assert len(keys)==(8 if low==16 else 6)
   for k in ('H_fraction','variance','finite_Q_all','finite_Q_visited','finite_PUCT_all','max_depth'):
    v=np.array([hi[i][k]-lo[i][k] for i in keys]);boot=v[rng.integers(0,len(v),(4096,len(v)))].mean(1)
    paired.append(dict(L=L,low=low,high=high,observable=k,n=len(v),mean=float(v.mean()),se=float(v.std(ddof=1)/np.sqrt(len(v))),interval=np.quantile(boot,[.025,.975]).tolist(),differences=v.tolist(),scope='fixed design opening-block bootstrap; not population confidence'))
 # Conditional environment and nonstationary reset driven by actual p.
 long.sort(key=lambda r:(r['L'],r['B'],r['opening_id'],r['ply']));quantum=[];thermal=[];states={}
 for g in gs:
  rr=[r for r in long if r['game']==g['job']];n=len(rr);a=[rr[0],rr[n//2],rr[-1]]
  probs=[x['sector_probability'] for x in a]
  # Middle snapshot only for 3 record strengths, all 36 game protocols
  p=probs[1];assert min(p)>0
  for g0 in (.2,.5,.8):
   q,z=erasure(p,g0);key=f"{g['job']}_g{g0:g}"
   q=dict(q); q['environment_L']=q.pop('L'); quantum.append(dict(game=g['job'],L=g['L'],B=g['B'],ply=a[1]['ply'],p=p,**q))
   for m0,r0 in z.items():states[key+'_'+m0]=r0
  thermal.append(dict(game=g['job'],L=g['L'],B=g['B'],knots=probs,**driven_response(probs)))
 assert len(quantum)==108 and len(states)==972 and len(thermal)==36
 np.savez_compressed(S/'CONDITIONAL_STATES.npz',**states)
 # Q19 dedicated cyclic work verification: reuse exact archived R54 raw
 # work arrays, not count as new work paths or native games.
 q54=read(F/'PHYSICS_RESULTS.json');pub54=read(F/'PUBLICATION_RECEIPT.json')
 assert sha256(F/'PHYSICS_RESULTS.json')==pub54['physics_results_sha256'] and sha256(F/'WORK_SAMPLES.npz')==q54['work_samples_sha']
 q19=[];w=np.load(F/'WORK_SAMPLES.npz')
 for a in q54['thermal']:
  key=f"{a['position']}_{a['law']}_{a['tau']:g}";v=w[key+'_F'];z=w[key+'_R']
  e=np.exp(-v);n=np.array([16,64,256,1024,4096]);avg=[float(e[:i].mean()) for i in n];df=[-math.log(x) for x in avg]
  q19.append(dict(key=key,tau=a['tau'],N=len(v),prefix_n=n.tolist(),JE=avg,deltaF_est=df,JE_SE=float(e.std(ddof=1)/np.sqrt(len(e))),
                  W_F=a['forward']['moments'][1],W_R=a['reverse']['moments'][1],Var_F=a['forward']['variance'],Var_R=a['reverse']['variance'],complex_crooks_error=a['complex_crooks_error']))
 assert len(q19)==12
 out=dict(counts=counts,games=gs,longitudinal=long,phase=ph,combined_game_summary=allg,combined_phase=allph,paired=paired,control=controls,control_models=models,mark_prediction=prediction,loops=loops,quantum=quantum,thermal=thermal,q19=q19,
          q19_source=dict(stage='RESTORE54',sha=sha256(F/'WORK_SAMPLES.npz'),scope='reused 98304 conditional work paths; no new generation'),states_sha=sha256(S/'CONDITIONAL_STATES.npz'),fresh_histories=len(native),unique_seed_groups=len(seeds),
          size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in sizes},prior_result_sha=sha256(O/'PHYSICS_RESULTS.json'),
          scope='actual matched complete games and independent fixed-root mark control; declared coherent erasure/driven reset; scoped Q19 replot')
 atomic_json(S/'PHYSICS_RESULTS.json',out);print('COMBINED',len(native),len(long),len(controls),len(quantum),len(thermal),flush=True)
if __name__=='__main__':main()
