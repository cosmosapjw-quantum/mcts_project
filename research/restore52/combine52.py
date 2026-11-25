"""Aggregate only audited native sector-field paths; independent conditional Q/N."""
from pathlib import Path
import json,gzip,statistics as st,math
from collections import defaultdict,Counter
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics50 import sectors,noise_decomposition
from mqft.physics51 import continuous_cycle
from mqft.physics52 import tilted_cycle,quantum_noncommuting
from mqft.physics44 import exact_wave_full_state,projected_information
from mqft.physics45 import lattice_action
S=Path(__file__).resolve().parent;O=S.parent/'mqft_restore51_20261126'
read=lambda p:json.loads(p.read_text())
def raw(r):
 p=S/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,x,e=[json.loads(l) for l in f]
 assert h['job']['position_id']==r['position'];return x['trace']
def main():
 a=[read(S/f'PHYSICS_L{L}.json') for L in (13,21)]
 native=sum([x['native'] for x in a],[]);games=sum([x['games'] for x in a],[])
 counts={f"L{x['L']}":x['counts'] for x in a}
 out={k:sum([x[k] for x in a],[]) for k in ('spatial','actions','temporal','diagnostics','bounds','temperature')}
 assert len(native)==552 and len(games)==2 and len(out['spatial'])==18 and len(out['temporal'])==30
 assert sum(v.get('exact_prefix_pairs',0) for c in counts.values() for v in c.values())==240
 groups=defaultdict(list);seedset=defaultdict(list)
 for r in native:
  seedset[r['seed']].append(r)
  if r['schedule']=='base':groups[r['L'],r['position'],r['law'],r['B']].append(r)
 assert len(groups)==18
 assert all(len({(r['L'],r['position'],r['law'],r['B']) for r in v})==1 for v in seedset.values())
 for L in (13,21):
  old=read(O/f'PHYSICS_L{L}.json')['native'];assert not {r['seed'] for r in native}&{r['seed'] for r in old}
 # Compare to same-root independent scalar-source R51 baseline, not
 # equality of observables. Prior confirmation used only to choose sector.
 compare=[]
 for x in out['diagnostics']:
  if x['B']!=16:continue
  old=read(O/f"PHYSICS_L{x['L']}.json")['diagnostics']
  q=next(y for y in old if y['campaign']=='confirm' and y['position']==x['position'] and y['law']==x['law'])
  compare.append(dict(L=x['L'],position=x['position'],law=x['law'],scalar_F=q['Fisher']['mean'],sector_F=x['Fisher']['mean'],scalar_hits=q['hits']['total'],sector_hits=x['hits']['total'],sector_expected=x['expected_hits']['total'],scalar_expected=q['expected_hits']['total']))
 # For each observed path, verify exact Bernoulli sufficient statistic of
 # same-history field tilt: categorical KL equals binary KL; conditionals
 # within group and outside unchanged. Directly evaluate actual p and shifted q.
 klmax=normmax=withinmax=0.;n_draws=0
 for r in native:
  t=raw(r)
  for e in t['replicas'][0]['events']:
   z=e['root_selection']['law'];p=np.asarray(z['p']);mask=np.isin(z['actions'],r['group']);m=p>0
   h=.2;Z=float(np.dot(p,np.exp(h*mask)));q=p*np.exp(h*mask)/Z;b=float(p[mask].sum())
   kl=float(np.dot(p[m],np.log(p[m]/q[m])));k0=math.log1p(b*math.expm1(h))-h*b
   klmax=max(klmax,abs(kl-k0));normmax=max(normmax,abs(q.sum()-1))
   for w in (mask,~mask):
    if p[w].sum()>1e-14:withinmax=max(withinmax,float(np.max(abs(p[w]/p[w].sum()-q[w]/q[w].sum()))))
   n_draws+=1
 assert max(klmax,normmax,withinmax)<1e-10
 quantum=[];states={};thermal=[];checks=[];noise=[];loops=[]
 # B16 baseline only for matched spatial reference; group field has
 # distinct support from the restricted reference's single source.
 for (L,pos,law,B),rr in sorted(groups.items()):
  if B!=16:continue
  actual=[];prop=[]
  for r in rr:
   t=raw(r);ev=sorted(t['replicas'][0]['events'],key=lambda e:e['root_selection']['clock'])
   p=[]
   for e in ev:
    d=e['root_selection']['law'];p.append(np.bincount(sectors(L,d['actions']),weights=d['p'],minlength=4))
   actual.append(np.eye(4)[sectors(L,r['action_history'])]);prop.append(p)
   n1=sum(r['I'][:4]);n2=sum(r['I'][-4:]);q,rho=quantum_noncommuting(n1,n2)
   quantum.append(dict(L=L,position=pos,law=law,job=r['job'],**q));states[r['job']]=rho
  noise.append(dict(L=L,position=pos,law=law,**noise_decomposition(actual,prop)))
  p=np.asarray(prop).reshape(len(rr),4,4,4).mean((0,2))[[0,2,3]]
  p/=p.sum(1)[:,None]
  for tau in (4.,16.,64.):
   q=tilted_cycle(p,tau);ref=continuous_cycle(p,tau)
   err=abs(q['W']-ref['work']);assert err<1e-8
   thermal.append(dict(L=L,position=pos,law=law,propensity=p.tolist(),mean_work_reference_error=err,**q))
   if pos.endswith('g0v0') and law=='iid':
    t=tilted_cycle(p,tau,method='Radau',rtol=2e-11,atol=2e-13)
    err=max(abs(q['W']-t['W']),np.max(abs(np.asarray(q['forward'])-t['forward'])),np.max(abs(np.asarray(q['reverse'])-t['reverse'])))
    assert err<1e-8;checks.append(dict(L=L,tau=tau,max_error=float(err)))
  # Actual common geometry reference: use existing R51 selected reference
  # support, same native snapshot for laws (B16 baseline IID first seed).
  ref=sorted(groups[L,pos,'iid',16],key=lambda r:r['seed'])[0];t=raw(ref)
  reg=read(O/'REFERENCE_REGISTRATION.json');offsets=reg['support'][pos.split('_')[1]]['offsets']
  support=[(dy+L//2)*L+dx+L//2 for dy,dx in offsets];assert all(i in ref['legal'] for i in support)
  edges=next(x['edges'] for x in t['replicas'][0]['cuts'] if x['local_k']==8)
  z=exact_wave_full_state(edges,support,4,law,support[0])
  loops.append(dict(L=L,position=pos,law=law,source_job=ref['job'],support=support,values=lattice_action(z,(8,16,32),True),Fisher=projected_information(z),errors=z['errors']))
 assert len(quantum)==96 and len(states)==96 and len(thermal)==36 and len(checks)==6 and len(loops)==12
 np.savez_compressed(S/'CONDITIONAL_STATES.npz',**states)
 out.update(counts=counts,games=games,compare=compare,quantum=quantum,thermal=thermal,thermal_check=checks,noise=noise,loops=loops,
            kl_identity=dict(draws=n_draws,max_KL_error=klmax,max_norm_error=normmax,max_within_group_error=withinmax),fresh_histories=len(native),unique_seed_groups=len(seedset),
            size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in a},states_sha=sha256(S/'CONDITIONAL_STATES.npz'),scope='native group-field intervention; separate conditional Q/N and frozen single-source references; no native phase or physical temperature')
 atomic_json(S/'PHYSICS_RESULTS.json',out);print('COMBINED',len(native),len(out['temporal']),len(quantum),len(thermal),flush=True)
if __name__=='__main__':main()
