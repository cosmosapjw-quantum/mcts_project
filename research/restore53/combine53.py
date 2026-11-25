"""Aggregate receipt-checked vector native data and conditional physical models."""
from pathlib import Path
import json,gzip,math,statistics as st
from collections import defaultdict
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics53 import U,prospective_response,thermal_work,ordered_environment
from mqft.physics50 import sectors,noise_decomposition
from mqft.physics51 import continuous_cycle
from mqft.physics52 import tilted_cycle,quantum_noncommuting
from mqft.physics44 import exact_wave_full_state,projected_information
from mqft.physics45 import lattice_action
S=Path(__file__).resolve().parent;B=S.parent
read=lambda p:json.loads(p.read_text())
def raw(r):
 p=S/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,x,e=[json.loads(l) for l in f]
 assert h['job']['position_id']==r['position'];return x['trace']
def main():
 a=[read(S/f'PHYSICS_L{L}.json') for L in (13,21)]
 native=sum([x['native'] for x in a],[]);games=sum([x['games'] for x in a],[]);groups=sum([x['groups'] for x in a],[]);T=sum([x['temperature'] for x in a],[])
 counts={f"L{x['L']}":x['counts'] for x in a}
 assert len(native)==1032 and len(games)==2 and len(groups)==48
 assert sum(v.get('exact_prefix_pairs',0) for c in counts.values() for v in c.values())==720
 assert sum(v.get('exact_prefix_draws',0) for c in counts.values() for v in c.values())==8640
 seedset=defaultdict(list)
 for r in native:seedset[r['seed']].append(r)
 assert len(seedset)==168 and all(len({(r['L'],r['campaign'],r['position'],r['law']) for r in v})==1 for v in seedset.values())
 for old in (51,52):
  for L in (13,21):
   p=B/f'mqft_restore{old}_20261126'/f'PHYSICS_L{L}.json'
   assert not {r['seed'] for r in native}&{r['seed'] for r in read(p)['native']}
 arms=defaultdict(lambda:defaultdict(list))
 for r in native:arms[r['L'],r['campaign'],r['position'],r['law']][tuple(r['arm'])].append(r)
 assert len(arms)==48
 confirmation={k:v[(-1,0,0,0)] for k,v in arms.items() if k[1]!='pilot'}
 assert len(confirmation)==24
 temporal=[];prediction=[]
 for (L,c,pos,l),v in sorted(arms.items()):
  if c not in ('g0','g1'):continue
  q=next(x for x in groups if (x['L'],x['campaign'],x['position'],x['law'])==(L,c,pos,l))
  temporal.extend(dict(L=L,campaign=c,position=pos,law=l,**x) for x in q['responses'])
  prediction.extend(dict(L=L,campaign=c,position=pos,law=l,**x) for x in prospective_response(v))
 assert len(temporal)==60 and len(prediction)==48
 # Periphery comparison only at h0; n8 vs n4 in g0 acknowledged.
 variants=[]
 for L in (13,21):
  for g in (0,1):
   for l in ('iid','pl','reserve'):
    z=[next(x for x in groups if x['L']==L and x['law']==l and x['position']==f'L{L}_g{g}v{v}' and x['campaign']!='pilot') for v in (0,1)]
    variants.append(dict(L=L,geometry=g,law=l,**{k:z[1][k]-z[0][k] for k in ('H','cov_trace','S_low')}))
 # Reference support: top five initial prior actions in central 7x7 on
 # L13 confirmation IID, separate from actual vector field.
 support={}
 for g in (0,1):
  r=sorted(confirmation[13,f'g{g}',f'L13_g{g}v0','iid'],key=lambda r:r['seed'])[0]
  c=6;legal=[i for i in r['legal'] if max(abs(i//13-c),abs(i%13-c))<=3]
  a0=sorted(legal,key=lambda i:(-r['prior'][i],i))[:5];assert len(a0)==5
  support[str(g)]=dict(offsets=[(i//13-c,i%13-c) for i in a0],job=r['job'],sha=r['shard_sha'])
 atomic_json(S/'REFERENCE_REGISTRATION.json',dict(support=support,scope='unperturbed initial prior common core support; single-source frozen reference, not vector native law'))
 loops=[];noise=[];thermal=[];quantum=[];samples={};states={}
 for (L,c,pos,l),rr in sorted(confirmation.items()):
  if c=='variants':continue
  g=int(pos.split('_g')[1][0]);ref=sorted(confirmation[L,c,pos,'iid'],key=lambda r:r['seed'])[0];t=raw(ref)
  sup=[(dy+L//2)*L+dx+L//2 for dy,dx in support[str(g)]['offsets']];assert all(i in ref['legal'] for i in sup)
  edges=next(x['edges'] for x in t['replicas'][0]['cuts'] if x['local_k']==8)
  z=exact_wave_full_state(edges,sup,4,l,sup[0])
  loops.append(dict(L=L,position=pos,law=l,source_job=ref['job'],support=sup,values=lattice_action(z,(8,16,32),True),Fisher=projected_information(z),errors=z['errors']))
  A=np.array([r['I'] for r in rr]);P=np.array([r['P'] for r in rr])
  noise.append(dict(L=L,position=pos,law=l,**noise_decomposition(A,P)))
  for r in rr:
   n1=int(sum(np.asarray(r['I'])[:4,0]));n2=int(sum(np.asarray(r['I'])[-4:,0]))
   q,rho=quantum_noncommuting(n1,n2);states[r['job']]=rho
   for theta in (0.,np.pi/6,np.pi/4,np.pi/3,np.pi/2):
    quantum.append(dict(L=L,position=pos,law=l,job=r['job'],entropy_S=q['entropy_S'],entropy_E=q['entropy_E'],CHSH=q['CHSH'],**ordered_environment(n1,n2,theta)))
  p=P.reshape(len(rr),4,4,4).mean((0,2))[[0,2,3]];p/=p.sum(1)[:,None]
  for tau in (4.,16.,64.):
   seed=535000+len(thermal);q,w=thermal_work(p,tau,4096,seed)
   ref=tilted_cycle(p,tau);err=abs(q['moments'][1]-ref['W']);assert err<1e-8
   # CF and mean numerical checks, do not silently reject stochastic draws.
   v=np.array(q['mc_se']);e=np.abs(np.array(q['cf_re'])+1j*np.array(q['cf_im'])-np.array(q['mc_re'])-1j*np.array(q['mc_im']))
   z=np.divide(e,v,out=np.zeros_like(e),where=v>1e-12)
   meanz=abs(q['mc_mean']-q['moments'][1])/max(q['mc_mean_se'],1e-12)
   samples[f'{pos}_{l}_{tau:g}']=w
   thermal.append(dict(L=L,position=pos,law=l,propensity=p.tolist(),max_cf_z=float(z.max()),mean_z=float(meanz),mean_work_reference_error=err,**q))
 assert len(loops)==12 and len(noise)==12 and len(thermal)==36 and len(quantum)==360 and len(samples)==36 and len(states)==72
 np.savez_compressed(S/'WORK_SAMPLES.npz',**samples)
 np.savez_compressed(S/'CONDITIONAL_STATES.npz',**states)
 out=dict(counts=counts,games=games,groups=groups,temperature=T,temporal=temporal,prediction=prediction,peripheral_variants=variants,loops=loops,noise=noise,quantum=quantum,thermal=thermal,
          fresh_histories=len(native),unique_seed_groups=len(seedset),max_tilt_error=max(r['max_tilt_error'] for r in native),
          size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in a},work_samples_sha=sha256(S/'WORK_SAMPLES.npz'),states_sha=sha256(S/'CONDITIONAL_STATES.npz'),
          scope='same actual MixNet; two new core geometries, vector pulses and fresh seed groups; explicit conditional environment and Gibbs reset work distribution')
 atomic_json(S/'PHYSICS_RESULTS.json',out);print('COMBINED',len(native),len(temporal),len(prediction),len(quantum),len(thermal),flush=True)
if __name__=='__main__':main()
