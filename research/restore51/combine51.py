"""Combine RESTORE51 actual cohorts and explicit physics references.
No baseline reuse as fresh data. Pilot selection independent of confirmation."""
from pathlib import Path
import json,gzip,math,statistics as st
from collections import defaultdict,Counter
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics49 import spatial_structure
from mqft.physics50 import sectors,noise_decomposition,projected_action
from mqft.physics51 import selection_diagnostics
from mqft.physics51 import causal_env,score_bound,geometry_block_memory,continuous_cycle,exact_fisher_projection
from mqft.physics44 import exact_wave_full_state,projected_information
from mqft.physics45 import lattice_action
S=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def raw(r):
 p=S/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,t,e=[json.loads(l) for l in f]
 assert h['job']['position_id']==r['position'];return t['trace']
def mean(a,k):return st.mean(x[k] for x in a)
def static():
 a=[read(S/'audits'/f'confirm-L{L}.json')['native'] if L!=17 else [r for r in read(S/'PHYSICS_L17.json')['native'] if r['campaign']=='confirm'] for L in (13,17,21)]
 confirm=sum(a,[]);assert len(confirm)==432
 sp=[];actions=[];diagnostics=[]
 groups=defaultdict(list)
 for r in confirm:groups[r['L'],r['position'],r['law']].append(r)
 assert len(groups)==54
 for (L,pos,law),rr in sorted(groups.items()):
  assert len(rr)==8
  r0=rr[0];q=spatial_structure(rr);q.update(position=pos,density=r0['density'],mark=st.mean(r['mark'] for r in rr),mean_parent=st.mean(r['parent_queries'] for r in rr),mean_child=st.mean(r['child_queries'] for r in rr));sp.append(q)
  actions.append(dict(L=L,position=pos,law=law,**projected_action(rr)))
  reg=read(S/f'REGISTRATION_L{L}.json');sel=reg['source_selection'][pos]['strata']['high'];assert r0['source']==sel['action']
  diagnostics.append(dict(L=L,position=pos,law=law,campaign='confirm',stratum='high',pilot=sel,**selection_diagnostics(rr)))
 # Analyze same-core peripheral variation, explicitly not independent size samples.
 variants=[]
 for L in (13,17,21):
  for g in range(3):
   for law in ('iid','pl','reserve'):
    p,q=[next(x for x in sp if x['L']==L and x['position']==f'L{L}_g{g}v{v}' and x['law']==law) for v in (0,1)]
    variants.append(dict(L=L,geometry=g,law=law,**{k:q[k]-p[k] for k in ('H','cov_trace','S_low','central_mass','mark')}))
 summaries=[]
 for L in (13,17,21):
  for law in ('iid','pl','reserve'):
   z=[x for x in sp if x['L']==L and x['law']==law]
   summaries.append(dict(L=L,law=law,geometries=3,variants=6,**{k:mean(z,k) for k in ('H','prior_H','cov_trace','S_low','central_mass','mark','mean_parent','mean_child')}))
 # Cross-size support: select 5 common core legal actions by L17
 # g,v unperturbed initial prior; source varies by L, so use common
 # reference target (first action) only for exact frozen-wave reference.
 # Not the actual intervention source. No result-based selection.
 sup={}
 for g in range(3):
  for v in range(2):
   z=sorted(groups[17,f'L17_g{g}v{v}','iid'],key=lambda r:r['seed'])[0]
   # L17 center=8. central 7x7 common legal (core same, peripheral outside).
   legal=[i for i in z['legal'] if max(abs(i//17-8),abs(i%17-8))<=3]
   order=sorted(legal,key=lambda i:(-z['prior'][i],i))[:5];assert len(order)==5
   sup[f'g{g}v{v}']=dict(offsets=[(i//17-8,i%17-8) for i in order],job=z['job'],sha=z['shard_sha'])
 atomic_json(S/'REFERENCE_REGISTRATION.json',dict(support=sup,scope='common restricted support mapped from L17 h0 prior, same core and peripheral variant label, no perturbation outcomes; reference target not necessarily actual source'))
 loops=[];noise=[];bounds=[];quantum=[];states={};thermal=[];thermal_check=[];fisher=[]
 for (L,pos,law),z in sorted(groups.items()):
  assert len(z)==8
  ref=sorted(groups[L,pos,'iid'],key=lambda r:r['seed'])[0];t=raw(ref);edges=next(c['edges'] for c in t['replicas'][0]['cuts'] if c['local_k']==8)
  support=[(dy+L//2)*L+dx+L//2 for dy,dx in sup[pos.split('_')[1]]['offsets']]
  assert all(i in ref['legal'] for i in support)
  q=exact_wave_full_state(edges,support,4,law,support[0]);l=lattice_action(q,(8,16,32),True)
  loops.append(dict(L=L,position=pos,law=law,source_job=ref['job'],support=support,values=l,errors=q['errors'],Fisher=projected_information(q)))
  actual=[];prop=[]
  for r in z:
   tr=raw(r);ev=sorted(tr['replicas'][0]['events'],key=lambda e:e['root_selection']['clock'])
   p=[]
   for e in ev:
    d=e['root_selection']['law'];p.append(np.bincount(sectors(L,d['actions']),weights=d['p'],minlength=4))
   prop.append(p);actual.append(np.eye(4)[sectors(L,r['action_history'])])
   # Explicit two-qubit environment memory: actual selected-source
   # sector count, not selected action count, in first/last block.
   sector=int(sectors(L,[r['source']])[0]);seq=sectors(L,r['action_history'])
   n1=int(np.sum(seq[:4]==sector));n2=int(np.sum(seq[-4:]==sector))
   for theta in (0.,np.pi/2,np.pi):
    q,E=causal_env(n1,n2,theta)
    quantum.append(dict(L=L,position=pos,law=law,job=r['job'],seed=r['seed'],sector=sector,**q))
    if theta==0:states[r['job']]=E
  noise.append(dict(L=L,position=pos,law=law,**noise_decomposition(actual,prop)))
  bounds.append(dict(L=L,position=pos,law=law,**score_bound(z)))
  P=np.asarray(prop);F=np.mean([exact_fisher_projection(p) for p in P],0)
  fisher.append(dict(L=L,position=pos,law=law,trace=float(np.trace(F)),eigenvalues=np.linalg.eigvalsh(F).tolist(),matrix=F.tolist()))
  p=P.reshape(8,4,4,4).mean((0,2))[[0,2,3]];p/=p.sum(1)[:,None]
  for tau in (16.,64.,256.):
   q=continuous_cycle(p,tau)
   thermal.append(dict(L=L,position=pos,law=law,propensity=p.tolist(),**q))
   if pos.endswith('g0v0') and law=='iid':
    a0=continuous_cycle(p,tau,rtol=2e-11,atol=2e-13,method='Radau')
    err=max(abs(q[k]-a0[k]) for k in ('work','heat','sigma','lag'))
    assert err<1e-8
    thermal_check.append(dict(L=L,tau=tau,max_error=err,method='independent Radau'))
 assert len(loops)==len(noise)==len(bounds)==len(fisher)==54 and len(thermal)==162 and len(quantum)==1296 and len(states)==432
 np.savez_compressed(S/'CONDITIONAL_ENV_STATES.npz',**states)
 memory=geometry_block_memory(confirm)
 out=dict(spatial=sp,actions=actions,diagnostics=diagnostics,summary=summaries,peripheral_variants=variants,
          loops=loops,noise=noise,score_bounds=bounds,sector_fisher=fisher,quantum=quantum,thermal=thermal,thermal_check=thermal_check,memory=memory,
          confirm_ids=sorted([r['job'] for r in confirm]),states_sha256=sha256(S/'CONDITIONAL_ENV_STATES.npz'),
          scope='confirmed baseline only; no pulse results')
 atomic_json(S/'STATIC_RESULTS.json',out);print('STATIC',len(confirm),len(sp),len(loops),len(quantum),len(thermal),flush=True)

def main():
 a=[read(S/f'PHYSICS_L{L}.json') for L in (13,17,21)]
 native=sum([x['native'] for x in a],[]);games=sum([x['games'] for x in a],[])
 counts={f"L{x['L']}":x['counts'] for x in a};temp=sum([x['temporal'] for x in a],[])
 dg=sum([x['diagnostics'] for x in a],[]);T=sum([x['temperature'] for x in a],[])
 assert len(native)==1716 and len(games)==3 and len(temp)==72 and len(dg)==78
 confirm=[r for r in native if r['campaign']=='confirm'];pilot=[r for r in native if r['campaign']=='pilot']
 assert len(confirm)==432 and len(pilot)==108
 seeds=defaultdict(list)
 for r in native:seeds[r['seed']].append(r)
 assert not {x['seed'] for x in pilot}&{x['seed'] for x in native if x['campaign']!='pilot'}
 assert all(len({(x['L'],x['position'],x['campaign'],x['law'],x['B'],x['source']) for x in v})==1 for v in seeds.values())
 groups=defaultdict(list)
 for r in confirm:groups[r['L'],r['position'],r['law']].append(r)
 st0=read(S/'STATIC_RESULTS.json')
 assert st0['confirm_ids']==sorted(r['job'] for r in confirm) and st0['states_sha256']==sha256(S/'CONDITIONAL_ENV_STATES.npz')
 # 16 independent B16 histories only for selected high-pulse roots.
 rank16=[]
 for L in (13,21):
  for law in ('iid','pl','reserve'):
   pos=f'L{L}_g0v0';z=groups[L,pos,law]+[r for r in native if r['L']==L and r['campaign']=='pulse_high' and r['law']==law and r['schedule']=='base']
   assert len(z)==16 and len({r['seed'] for r in z})==16 and len({r['source'] for r in z})==1
   rank16.append(dict(L=L,position=pos,law=law,**projected_action(z)))
 out=dict(**{k:v for k,v in st0.items() if k not in ('diagnostics','scope')},counts=counts,games=games,actions_n16=rank16,temporal=temp,diagnostics=dg,temperature=T,
          fresh_histories=len(native),unique_seed_groups=len(seeds),size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in a},
          static_sha=sha256(S/'STATIC_RESULTS.json'),scope='independent pilot-selected sources; three new templates and two peripheral variants across three sizes; finite native pulse and explicit conditional physical references')
 atomic_json(S/'PHYSICS_RESULTS.json',out);print('COMBINED',len(native),len(temp),len(dg),flush=True)
if __name__=='__main__':
 import sys
 if len(sys.argv)>1 and sys.argv[1]=='static':static()
 else:main()
