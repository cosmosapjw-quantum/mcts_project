"""Aggregate fixed-size/size-family native pilot and dedicated diagnostics."""
from pathlib import Path
import json,gzip,math,statistics as st
from collections import defaultdict,Counter
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
S=Path(__file__).resolve().parent;B=S.parent;Ls=(13,15,17,19,21);read=lambda p:json.loads(p.read_text())
def cov(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 return (a-a.mean(0)).T@(b-b.mean(0))/(len(a)-1)
def raw(r):
 p=S/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,x,e=[json.loads(l) for l in f]
 assert h['job']['id']==r['job'];return x['trace']
def decomposition(rr):
 L=rr[0]['L'];B0=rr[0]['B'];a=np.asarray([r['counts'] for r in rr])/B0
 noise=np.asarray([r['action_score'] for r in rr])/B0;drift=a-noise
 traces=[]
 for r,d in zip(rr,drift):
  t=raw(r);e=t['replicas'][0]['events'];s=np.zeros(L*L);v=0.
  for z in e:
   q=z['root_selection']['law'];p=np.asarray(q['p']);s[q['actions']]+=p;v+=1-p@p
  assert max(abs(s/B0-d))<1e-10
  traces.append(v/B0**2)
 # Matrix identity, not assuming drift independent of martingale noise.
 C=cov(a,a);N=cov(noise,noise);D=cov(drift,drift);X=cov(noise,drift)
 err=float(np.max(abs(C-N-D-X-X.T)));assert err<1e-12
 return dict(total=float(np.trace(C)),martingale_sample=float(np.trace(N)),drift=float(np.trace(D)),cross=float(2*np.trace(X)),
             predictable_trace=float(np.mean(traces)),martingale_mean_norm=float(np.linalg.norm(noise.mean(0))),decomposition_error=err)
def main():
 sizes=[read(S/f'PHYSICS_L{L}.json') for L in Ls];native=sum([x['native'] for x in sizes],[]);spatial=sum([x['spatial'] for x in sizes],[]);response=sum([x['response'] for x in sizes],[])
 counts={f"L{x['L']}":x['counts'] for x in sizes};assert len(native)==480 and len(spatial)==60 and len(response)==10
 seed=defaultdict(list);grp=defaultdict(list)
 for r in native:
  seed[r['seed']].append(r)
  if r['amplitude']==0:grp[r['L'],r['campaign'],r['position']].append(r)
 assert len(seed)==360 and len(grp)==60 and all(len(x) in (1,3) and len({(r['L'],r['campaign'],r['position']) for r in x})==1 for x in seed.values())
 assert sum(len(x)==3 for x in seed.values())==60
 assert sum(v.get('exact_prefix_pairs',0) for c in counts.values() for v in c.values())==120
 assert sum(v.get('exact_prefix_draws',0) for c in counts.values() for v in c.values())==1440
 base=[r for r in native if r['amplitude']==0];assert len(base)==360
 rows=[]
 for x in spatial:
  q=decomposition(grp[x['L'],x['campaign'],x['position']]);assert abs(q['total']-x['trace'])<1e-12
  rows.append(dict(**x,**q))
 # Descriptive slope/window sensitivity only: fixed two cores, 3 beta
 # levels and 5 sizes. Test whether simple size-power trend predicts
 # heldout size better than a constant. No fitted critical beta/exponent.
 scaling=[]
 for B0 in (16,32):
  for i in range(3):
   for g in (0,1):
    a=sorted([x for x in rows if x['campaign']==f'B{B0}b{i}' and x['position'].endswith(f'dense{g}')],key=lambda x:x['L'])
    assert [x['L'] for x in a]==list(Ls)
    for key in ('trace','S_low','central_mass','H_fraction','order'):
     y=np.array([x[key] for x in a]);xx=np.log(Ls)
     if min(y)<=1e-15:
      scaling.append(dict(B=B0,beta=a[0]['beta'],core=g,observable=key,status='nonpositive_no_log_fit'));continue
     z=np.log(y);s0=float(np.polyfit(xx,z,1)[0]);s1=float(np.polyfit(xx[1:-1],z[1:-1],1)[0]);a0=[];a1=[]
     for k in range(5):
      m=np.arange(5)!=k;a0.append((z[k]-z[m].mean())**2);a1.append((z[k]-np.polyval(np.polyfit(xx[m],z[m],1),xx[k]))**2)
     scaling.append(dict(B=B0,beta=a[0]['beta'],core=g,observable=key,status='descriptive',slope_all=s0,slope_inner=s1,loo_constant_mse=float(np.mean(a0)),loo_power_mse=float(np.mean(a1))))
 # Survival / predictable first hit. No product of adaptive hazards
 # treated as unconditional survival. Descriptive pooling across roots.
 survival=[]
 for L in Ls:
  for B0 in (16,32):
   for b in range(3):
    rr=[r for r in base if r['L']==L and r['campaign']==f'B{B0}b{b}'];assert len(rr)==12
    I=np.array([r['I'] for r in rr]);A=np.array([r['at_risk'] for r in rr]);p=np.array([r['p'] for r in rr])
    hits=(I*A).sum(0);risk=A.sum(0);expect=(A*p).sum(0)
    assert hits.sum()==sum(r['detected'] for r in rr)
    survival.append(dict(L=L,B=B0,beta=rr[0]['beta'],n=len(rr),risk=risk.tolist(),hits=hits.tolist(),expected=expect.tolist(),
      survival=(1-np.cumsum(hits)/len(rr)).tolist(),hit_rate=float(hits.sum()/len(rr)),compensator=float(expect.sum()),residual=float(hits.sum()-expect.sum()),
      predictable_variance=float((A*p*(1-p)).sum()),wave4_observed=sum(r['firstwave_hit'] for r in rr),wave4_expected=sum(r['firstwave_prob'] for r in rr)))
 # Structure summaries all actual baseline and interventions. Full
 # per-history data in PHYSICS_L*.json; retain maps and sparse examples.
 faults=Counter();top=[];bar=[]
 for r in native:
  for x in r['fault']['results']:faults[x['kind']+'_'+x['status']]+=1
  q=r['topology_k8'];z=r['topology_visited'];b=r['barrier']
  top.append(dict(L=r['L'],B=r['B'],beta=r['beta'],position=r['position'],job=r['job'],amplitude=r['amplitude'],snapshot_b0=q['final']['b0'],snapshot_b1=q['final']['b1'],
   snapshot_max_b0=q['max_b0'],snapshot_max_b1=q['max_b1'],positive_H0=q['positive_H0'],max_persistence=q['max_persistence'],mean_persistence=q['mean_persistence'],
   visited_b0=z['final']['b0'],visited_b1=z['final']['b1'],visited_vertices=z['vertices'],trie=r['trie'],independent_thresholds=q['independent_thresholds']+z['independent_thresholds']))
  bar.append(dict(L=r['L'],B=r['B'],beta=r['beta'],position=r['position'],job=r['job'],amplitude=r['amplitude'],detected=r['detected'],first_hit=r['first_hit'],
   survived_k8=int(not any(r['I'][:8])),hit_after_k8=int(not any(r['I'][:8]) and any(r['I'][8:])),source_probability_k8=r['p'][8],**b))
 assert faults=={'single_corrected':1920,'double_detected_uncorrectable':1440}
 assert all(x['trie']['b1']==0 and x['trie']['E']==x['trie']['V']-1 for x in top)
 # Prespecified visual exemplar: baseline first replicate, L17 B32 beta=10/3,
 # core0 (not the maximum-barrier or persistence example).
 r=next(x for x in base if x['L']==17 and x['campaign']=='B32b1' and x['position'].endswith('dense0') and x['replicate']==0)
 example={k:r[k] for k in ('L','B','beta','position','job','source','root_k8_score','visited_Q','topology_k8','topology_visited','barrier','action_history','p','I','fault')}
 out=dict(counts=counts,fresh_histories=len(native),unique_seed_groups=len(seed),baseline_histories=len(base),spatial=rows,response=response,scaling=scaling,survival=survival,topology=top,barriers=bar,fault_counts=dict(faults),
          fault_max_error=max(x['error'] for r in native for x in r['fault']['results'] if x['kind']=='single'),example=example,
          size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in sizes},
          scope='native R49 verified same-MixNet size-family pilot; first detection not quantum tunneling; planar score filtration not quantum topology; offline checksum recovery not QEC')
 atomic_json(S/'PHYSICS_RESULTS.json',out);print('COMBINED',len(native),len(base),len(rows),len(response),len(top),dict(faults),flush=True)
if __name__=='__main__':main()
