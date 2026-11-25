"""R57 exact reuse of verified R55/R56 longitudinal raw and prospective
fixed R56 predictor, new noncyclic thermodynamic references."""
from pathlib import Path
import json,gzip,statistics as st,math
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics56 import action_features,evaluate_mark,finite_control_samples,score_controls
from mqft.physics57 import score_energy,levels,compare_noncyclic,cramer,heat_engines,protocol
S=Path(__file__).resolve().parent;B=S.parent;O=B/'mqft_restore56_20261126';A=B/'mqft_restore55_20261126'
read=lambda p:json.loads(p.read_text())
def checked(stage):
 p=stage/'PHYSICS_RESULTS.json';r=read(p);assert sha256(p)==read(stage/'PUBLICATION_RECEIPT.json')['physics_results_sha256'];return r
def raw(stage,r):
 p=stage/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,x,e=[json.loads(l) for l in f]
 assert h['job']['id']==r['job'];return x['trace']
def lawfit(scores,chosen,b0=10/3):
 # IID actual all action scores and known input beta; pooled log likelihood.
 s=np.asarray(scores);v=np.asarray(chosen);m=len(v)
 def fun(b):
  z=b*s;l=logsumexp(z,axis=1);p=np.exp(z-l[:,None]);mu=(p*s).sum(1)
  return float(np.sum(l-b*v)),float(np.sum(mu-v)),float(np.sum(p*(s-mu[:,None])**2))
 z=fun(0.)
 if z[1]>=0:beta=0.;status='zero'
 elif fun(100.)[1]<=0:beta=None;status='unresolved_or_infinite'
 else:beta=float(brentq(lambda b:fun(b)[1],0.,100.));status='finite'
 a=fun(b0);f=fun(beta) if beta is not None else None
 return dict(n=m,beta=beta,status=status,nll_at_input=a[0]/m,nll_uniform=z[0]/m,nll_fit=None if f is None else f[0]/m,input_score=-a[1],input_fisher=a[2])
def variable_lawfit(rows,b0=10/3):
 # Different legal support across game plies, no padding/illegal probabilities.
 def fun(b):
  vals=[]
  for s,v in rows:
   z=b*s;l=logsumexp(z);p=np.exp(z-l);mu=p@s
   vals.append((l-b*v,mu-v,p@(s-mu)**2))
  return np.sum(vals,axis=0)
 a=fun(b0);z=fun(0.)
 if z[1]>=0:beta=0.;status='zero'
 elif fun(100.)[1]<=0:beta=None;status='unresolved_or_infinite'
 else:beta=float(brentq(lambda b:fun(b)[1],0.,100.));status='finite'
 f=fun(beta) if beta is not None else None;m=len(rows)
 return dict(n=m,beta=beta,status=status,nll_at_input=float(a[0]/m),nll_uniform=float(z[0]/m),nll_fit=None if f is None else float(f[0]/m),input_score=float(-a[1]),input_fisher=float(a[2]))
def main():
 old=checked(O);prior=checked(A);sizes=[read(S/f'PHYSICS_L{L}.json') for L in (13,21)]
 native=sum([x['native'] for x in sizes],[]);assert len(native)==288
 counts={f"L{x['L']}":x['counts'] for x in sizes};groups=defaultdict(lambda:defaultdict(list));seeds=defaultdict(list)
 for r in native:groups[r['L'],r['campaign'],r['law']][tuple(r['arm'])].append(r);seeds[r['seed']].append(r)
 assert len(seeds)==96 and all(len(v)==3 for v in seeds.values()) and len(groups)==12
 for stg,q in ((A,prior),(O,old)):
  for L in (13,21):
   f=stg/f'PHYSICS_L{L}.json';assert sha256(f)==q['size_results_sha'][f'L{L}']
   assert not set(seeds)&{x['seed'] for x in read(f)['native']}
 # No fit or coefficient adjustment using current data.
 models={(x['L'],x['law']):x for x in old['control_models']}
 controls=[];prediction=[]
 for (L,c,law),rr in sorted(groups.items()):
  model=models[L,law];assert all(len(x)==8 for x in rr.values()) and len(rr)==3
  e={r['job']:evaluate_mark(action_features(raw(S,r)),model['model'],L,16) for v in rr.values() for r in v}
  controls.extend(dict(L=L,campaign=c,law=law,**z) for z in score_controls(finite_control_samples(rr,e),np.asarray(model['coefficients'])))
  for r in rr[-1,0.]:
   q=e[r['job']];p=np.asarray(q['pred']);y=np.asarray(q['marks'])
   prediction.append(dict(L=L,campaign=c,law=law,job=r['job'],mse=np.mean((p-y[:,None])**2,axis=0).tolist()))
 assert len(controls)==72 and len(prediction)==96
 # Recover original G01 definition from native actual game cuts, reusing
 # prior complete trace verification and checking each exact shard.
 energy=[];fit=[];games=[]
 for stage,q in ((A,prior),(O,old)):
  index={(x['game'],x['ply']):x for x in q['longitudinal']}
  for g in sorted(q['games'],key=lambda x:(x['L'],x['B'],x['opening_id'])):
   L=g['L'];budget=g['B'];f=stage/'data'/f'game{budget}-L{L}'/(g['job']+'.jsonl.gz')
   assert sha256(f)==g['shard_sha'];receipt=read(f.with_name(g['job']+'.receipt.json'))
   assert receipt['sha256']==g['shard_sha'] and receipt['bytes']==f.stat().st_size
   n=0;game_rows=[];m=[]
   with gzip.open(f,'rt') as fh:
    hd=json.loads(next(fh));assert hd['job']['id']==g['job']
    for line in fh:
     z=json.loads(line)
     if z['type']=='end':assert z['status']=='natural_terminal' and z['moves']==g['moves'];break
     t=z['trace'];ply=z['ply'];ref=index[g['job'],ply];a=score_energy(t,ply);last=a['rows'][-1]
     assert abs(last['E_Q']+ref['q'])<1e-10 and abs(last['H']-ref['H'])<1e-10 and t['root']['state_hash']==ref['root_hash']
     e=dict(game=g['job'],stage=stage.name,L=L,B=budget,opening_id=g['opening_id'],shard_sha=g['shard_sha'],**a)
     energy.append(e);game_rows.append(e);n+=1
     for event in t['replicas'][0]['events']:
      d=event['root_selection']['law'];s=np.asarray(d['scores']);p=np.asarray(d['p']);j=d['picked']
      assert all(d['active']) and np.max(abs(np.exp((10/3)*s-logsumexp((10/3)*s))-p))<1e-10
      m.append((s,s[j]))
   assert n==g['searched'];fit.append(dict(game=g['job'],L=L,B=budget,stage=stage.name,**variable_lawfit(m)))
   games.append(dict(game=g['job'],L=L,B=budget,stage=stage.name,opening_id=g['opening_id'],searched=n,
      mean_E_Q=st.mean(x['rows'][-1]['E_Q'] for x in game_rows),mean_E_PUCT=st.mean(x['rows'][-1]['E_PUCT'] for x in game_rows),
      mean_E_black=st.mean(x['rows'][-1]['E_Q']*x['rows'][-1]['side'] for x in game_rows)))
 assert len(games)==44 and len(energy)==533
 # Fixed eight R56 games: first two opening IDs per size and B16/64.
 # No selection on p, work, or response.
 selected=[]
 for L in (13,21):
  for b in (16,64):
   selected+=sorted([g for g in old['games'] if g['L']==L and g['B']==b],key=lambda g:g['opening_id'])[:2]
 assert len(selected)==8
 thermal=[];engines=[];rate=[];arrays={};registered=[]
 for j,g in enumerate(selected):
  a=sorted([r for r in old['longitudinal'] if r['game']==g['job']],key=lambda r:r['ply']);ix=[0,len(a)//2,len(a)-1]
  probs=[a[i]['sector_probability'] for i in ix];E=levels(probs)
  registered.append(dict(game=g['job'],L=g['L'],B=g['B'],plies=[a[i]['ply'] for i in ix],source_sha=g['shard_sha'],prob=probs,levels=E.tolist()))
  for tau in (4.,16.,64.):
   q,f,r=compare_noncyclic(E,tau,2048,577000+100*j+2*int(tau))
   key=f"g{j}_tau{tau:g}"
   thermal.append(dict(key=key,game=g['job'],L=g['L'],B=g['B'],**q))
   for tag,z in (('F',f),('R',r)):
    for k,v in z.items():arrays[key+'_'+tag+'_'+k]=v
   engines.append(dict(game=g['job'],L=g['L'],B=g['B'],**heat_engines(probs[1],tau/4)))
  rate.append(dict(game=g['job'],**cramer(E,4.)))
 assert len(thermal)==24 and len(engines)==24 and len(rate)==8 and len(arrays)==192
 np.savez_compressed(S/'NONCYCLIC_PATHS.npz',**arrays)
 # Gauge covariance fixture on actual first protocol, same ideal reset RNG.
 E=np.asarray(registered[0]['levels']);q,f=protocol(E,4.,512,579001);shift=np.array([-.2,.1,.7])
 a,b=protocol(E+shift[:,None],4.,512,579001)
 assert np.array_equal(f['X'],b['X'])
 gauge=float(max(np.max(abs(b['W']-f['W']-(np.linspace(0,1,9)*0+np.interp(np.linspace(0,1,9),[0,.5,1],shift)-shift[0]))),np.max(abs(b['Q']-f['Q'])),np.max(abs(b['sigma']-f['sigma']))))
 assert gauge<1e-10
 atomic_json(S/'THERMAL_REGISTRATION.json',dict(scope='deterministic first two R56 opening IDs per L/B16,64; declared E3=0, beta E*=1; fixed rates and seeds',protocols=registered))
 out=dict(counts=counts,fresh_histories=len(native),unique_seed_groups=len(seeds),controls=controls,mark_prediction=prediction,model_source_sha=old['physics_results_sha'] if 'physics_results_sha' in old else sha256(O/'PHYSICS_RESULTS.json'),
          games=games,energy=energy,lawfit=fit,thermal=thermal,engines=engines,rate=rate,gauge_error=gauge,paths_sha=sha256(S/'NONCYCLIC_PATHS.npz'),thermal_registration_sha=sha256(S/'THERMAL_REGISTRATION.json'),
          size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in sizes},
          reuse_source_sha={x.name:sha256(x/'PHYSICS_RESULTS.json') for x in (A,O)},scope='new native response, actual original score-energy, conditional thermodynamic references')
 atomic_json(S/'PHYSICS_RESULTS.json',out);print('COMBINED',len(native),len(energy),len(thermal),len(controls),flush=True)
if __name__=='__main__':main()
