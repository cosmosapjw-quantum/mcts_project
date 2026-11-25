"""Combine audited size-specific outputs; actual density contrast,
heldout tangent prediction, common-support lattice, four-sector thermodynamics."""
from pathlib import Path
import sys,json,gzip,math,statistics as st
from collections import defaultdict,Counter
import numpy as np
from scipy.linalg import helmert
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics49 import actual_schedule_ratio
from mqft.physics50 import sector_blocks,sectors,coherent_records,noise_decomposition
from mqft.physics44 import exact_wave_full_state,projected_information
from mqft.physics45 import lattice_action
from mqft.observables import entropy
S=Path(__file__).resolve().parent;O=S.parent/'mqft_restore49_20261126'
sys.path.insert(0,str(O));from spatial_ref49 import cycle
read=lambda p:json.loads(p.read_text())
def raw(r):
 p=S/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz')
 assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,t,e=[json.loads(l) for l in f]
 assert h['job']['position_id']==r['position']
 return t['trace']
def fit_memory(native):
 # Two independent template geometries, cross-size variations are
 # correlated geometry blocks. Train template0, test template1 only.
 U=helmert(4,full=False).T
 out=[]
 for law in ('iid','pl','reserve'):
  rows=[]
  for r in native:
   if r['schedule']!='base' or r['B']!=16 or r['beta']!=10/3 or r['law']!=law:continue
   a=sector_blocks(r);a=np.column_stack((a[:,:4]@U,a[:,4:]))
   # j=1,2: same previous/current/next availability across alternatives.
   for j in (1,2):
    rows.append(dict(pos=r['position'],L=r['L'],rho=r['density'],clock=(j+1)/4,state=a[j],lag=a[j-1],target=a[j+1]))
  train=[r for r in rows if r['pos'].endswith('0')];test=[r for r in rows if r['pos'].endswith('1')]
  c=Counter(r['pos'] for r in train);w=np.array([1/(len(c)*c[r['pos']]) for r in train]);Y=np.array([r['target'] for r in train]);Z=np.array([r['target'] for r in test])
  assert len(c)==10 and len({r['pos'] for r in test})==10 and not set(c)&{r['pos'] for r in test}
  def f(r,mode):
   x=r['state'];v=[1.,(r['L']-17)/4,r['rho']/.06,*x]
   if 'clock' in mode:v.extend([r['clock'],*(r['clock']*x)])
   if 'memory' in mode:v.extend(r['lag'])
   return v
  models={}
  for mode in ('state','clock','memory','clock_memory'):
   X=np.array([f(r,mode) for r in train]);T=np.array([f(r,mode) for r in test]);pen=.05*np.eye(X.shape[1]);pen[0,0]=0
   K=np.linalg.solve(X.T@(w[:,None]*X)+pen,X.T@(w[:,None]*Y));v=T@K
   # Project 4-sector mass onto simplex (Euclidean), scalar count/mark
   # clipped to physical range. Fixed before viewing test.
   for j in range(len(v)):
    z=.25+U@v[j,:3];u=np.sort(z)[::-1];s=np.cumsum(u)-1;k=np.flatnonzero(u-s/np.arange(1,5)>0)[-1]
    q=np.maximum(z-s[k]/(k+1),0);assert abs(q.sum()-1)<1e-12
    v[j,:3]=q@U
   v[:,3]=np.clip(v[:,3],0,1);v[:,4]=np.clip(v[:,4],-1,1)
   e=(v-Z)**2;groups=[]
   for pos in sorted({r['pos'] for r in test}):
    i=[j for j,r in enumerate(test) if r['pos']==pos]
    groups.append(dict(position=pos,n=len(i),sector_mse=float(e[i,:3].sum(1).mean()),target_mse=float(e[i,3].mean()),mark_mse=float(e[i,4].mean())))
   models[mode]=dict(groups=groups,**{k:st.mean(r[k] for r in groups) for k in ('sector_mse','target_mse','mark_mse')})
  out.append(dict(law=law,train_transitions=len(train),test_transitions=len(test),train_roots=10,test_roots=10,models=models,scope='two geometric templates only; second template held out, 10 size/density variants not 10 independent geometries'))
 return out

def main():
 a=[read(S/f'PHYSICS_L{L}.json') for L in (13,15,17,19,21)];native=sum([x['native'] for x in a],[])
 counts={f"L{x['L']}":x['counts'] for x in a};g=sum([x['games'] for x in a],[]);sp=sum([x['spatial'] for x in a],[]);act=sum([x['actions'] for x in a],[]);temp=sum([x['temporal'] for x in a],[]);quantum=sum([x['quantum'] for x in a],[]);T=sum([x['temperature'] for x in a],[])
 assert len(native)==732 and len(g)==5 and len(sp)==105 and len(act)==105 and len(temp)==18 and len(quantum)==60
 assert sum(v.get('exact_prefix_pairs',0) for c in counts.values() for v in c.values())==192
 seed_groups=defaultdict(list)
 for r in native:seed_groups[r['seed']].append(r)
 assert all(len({(r['L'],r['position'],r['B'],r['beta'],r['law']) for r in v})==1 for v in seed_groups.values())
 # same template, 2 density conditions, per-root means; no pooled plies.
 summary=[];delta=[]
 for L in (13,15,17,19,21):
  for law in ('iid','pl','reserve'):
   for typ in ('fixed','dense'):
    q=[x for x in sp if x['L']==L and x['B']==16 and x['beta']==10/3 and x['law']==law and x['kind']==typ];assert len(q)==2
    summary.append(dict(L=L,law=law,kind=typ,n_roots=2,**{k:st.mean(x[k] for x in q) for k in ('density','H','prior_H','cov_trace','S_low','central_mass','central_variance','mark','mean_parent','mean_child')}))
   x,y=[next(x for x in summary if x['L']==L and x['law']==law and x['kind']==t) for t in ('fixed','dense')]
   delta.append(dict(L=L,law=law,**{k:y[k]-x[k] for k in ('H','prior_H','cov_trace','S_low','central_mass','central_variance','mark')}))
 # Exact common 5-action support in 7x7 central template, for both density
 # arms and all sizes. Fixed prior-only choice, not outcome tuned.
 support={}
 for idx in (0,1):
  q=sorted([r for r in native if r['L']==15 and r['position']==f'L15_fixed{idx}' and r['law']=='iid' and r['B']==16 and r['beta']==10/3 and r['schedule']=='base'],key=lambda r:r['seed'])[0]
  src=q['source'];opts=[i for i in q['legal'] if i!=src and max(abs(i//15-7),abs(i%15-7))<=3]
  support[idx]=dict(actions=[src]+sorted(opts,key=lambda i:(-q['prior'][i],i))[:4],job=q['job'],shard_sha=q['shard_sha'])
 atomic_json(S/'REFERENCE_REGISTRATION.json',dict(support=support,scope='source plus four highest unperturbed initial prior actions in common central 7x7, independently for each template; same mapped support in both density arms and all L'))
 baseline=defaultdict(list)
 for r in native:
  if r['B']==16 and r['beta']==10/3 and r['schedule']=='base':baseline[r['L'],r['position'],r['law']].append(r)
 loops=[];thermal=[];states={};phase=[];proto=[];noise=[]
 for (L,pos,law),rr in sorted(baseline.items()):
  ref=sorted(baseline[L,pos,'iid'],key=lambda r:r['seed'])[0];t=raw(ref);edges=next(c['edges'] for c in t['replicas'][0]['cuts'] if c['local_k']==8)
  def map_a(i):y,x=divmod(i,15);return (y-7+L//2)*L+x-7+L//2
  sup=[map_a(i) for i in support[int(pos[-1])]['actions']]
  assert sup[0]==ref['source'] and all(i in ref['legal'] for i in sup)
  z=exact_wave_full_state(edges,sup,4,law,sup[0]);l=lattice_action(z,(8,16,32),True)
  loops.append(dict(L=L,position=pos,law=law,source_job=ref['job'],support=sup,values=l,errors=z['errors'],Fisher=projected_information(z)))
  # Actual conditional sector propensity knots; full native support.
  C=[];actual=[];propensity=[]
  for r in rr:
   tr=raw(r);ev=sorted(tr['replicas'][0]['events'],key=lambda e:e['root_selection']['clock'])
   q=[]
   for e in ev:
    d=e['root_selection']['law'];q.append(np.bincount(sectors(L,d['actions']),weights=d['p'],minlength=4))
   C.append(np.array(q).reshape(4,4,4).mean(1)[[0,2,3]])
   actual.append(np.eye(4)[sectors(L,r["action_history"])]);propensity.append(q)
  noise.append(dict(L=L,position=pos,law=law,**noise_decomposition(actual,propensity)))
  p=np.mean(C,0);p/=p.sum(1)[:,None]
  for duration in (1.,4.,16.,64.):
   thermal.append(dict(L=L,position=pos,law=law,propensity=p.tolist(),**cycle(p,duration,32)))
  rr0,ss=coherent_records(rr);phase.append(dict(L=L,position=pos,law=law,**rr0))
  for v,R in zip(rr0['rows'],ss):states[f"{pos}_{law}_{v['theta']}"]=R
  proto.append(dict(L=L,position=pos,law=law,mean_sigma=st.mean(r['protocol']['sigma'] for r in rr),mean_weight=st.mean(r['protocol']['weight'] for r in rr),mean_KL=st.mean(r['protocol']['conditional_KL'] for r in rr)))
 assert len(loops)==60 and len(thermal)==240 and len(states)==240
 np.savez_compressed(S/'COHERENT_STATES.npz',**states)
 memory=fit_memory(native)
 out=dict(counts=counts,games=g,spatial=sp,actions=act,temporal=temp,quantum=phase,temperature=T,summary=summary,density_delta=delta,loops=loops,thermal=thermal,protocol=proto,memory=memory,noise=noise,
    unique_seed_groups=len(seed_groups),fresh_histories=len(native),size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in a},
    states_sha256=sha256(S/'COHERENT_STATES.npz'),
    scope='same actual MixNet, fixed count vs approximate density, 2 new geometric templates, B/beta pilot, exact causal prefix, rank-aware conditional action, declared coherent and thermal model')
 atomic_json(S/'PHYSICS_RESULTS.json',out)
 print('COMBINED',len(native),len(sp),len(temp),len(loops),len(thermal),len(states),flush=True)
if __name__=='__main__':main()
