"""Exact finite reference based on the same actual L-family snapshots.
Same geometric restricted support across L; full parent counts retained.
Thermal/quantum reference is conditional, not native physical heat."""
from pathlib import Path
import json,gzip,math,statistics as st
from collections import defaultdict,Counter
import numpy as np
from scipy.special import logsumexp
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics44 import exact_wave_full_state,projected_information
from mqft.physics45 import lattice_action
from mqft.observables import entropy
S=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def raw(L,r):
 p=S/'data'/f'roots-L{L}'/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:h,x,e=[json.loads(l) for l in f]
 assert h['job']['position_id']==r['position'];return x['trace']
def map_coord(a,L):
 y,x=divmod(a,15);y+=L//2-7;x+=L//2-7
 assert 0<=x<L and 0<=y<L
 return y*L+x

def cycle(p,duration,m=32):
 """Four-state explicit Gibbs cycle, fixed actual empirical propensity
 knots, beta E*=1. Linear interpolation in log weights, normalized.
 Forward quench/reset, reverse reset/quench. Exact tilted transfer."""
 p=np.asarray(p,float);assert p.shape==(3,4) and np.min(p)>0 and np.allclose(p.sum(1),1)
 z=np.log(p);z=np.vstack((z,z[0]))
 w=np.concatenate([np.linspace(z[i],z[i+1],m,endpoint=False) for i in range(3)]+[z[-1:]],axis=0)
 pi=np.exp(w-logsumexp(w,axis=1)[:,None]);E=-np.log(pi)
 eta=-math.expm1(-duration/(3*m));T=(1-eta)*np.eye(4)[None,:,:]+eta*np.repeat(pi[1:,None,:],4,axis=1)
 a=pi[0].copy();W=Q=0.
 for j,M in enumerate(T):
  W+=a@(E[j+1]-E[j]);b=a@M;Q+=(b-a)@E[j+1];a=b
 U=float(a@E[-1]-pi[0]@E[0]);sigma=entropy(a)-entropy(pi[0])-Q;lag=float(a@np.log(a/pi[0]))
 def Z(l,rev=False):
  v=pi[-1 if rev else 0].copy()
  if rev:
   for j in range(len(T)-1,-1,-1):v=(v@T[j])*np.exp(-l*(E[j]-E[j+1]))
  else:
   for j,M in enumerate(T):v=(v*np.exp(-l*(E[j+1]-E[j])))@M
  return float(v.sum())
 r=max(abs(Z(1)-1),abs(U-W-Q),abs(W-sigma-lag),max(abs(Z(l)-Z(1-l,True)) for l in (0.,.25,.5,.75,1.)))
 assert r<1e-9 and sigma>=-1e-10 and lag>=-1e-10
 return dict(duration=duration,subdivision=m,work=float(W),heat=float(Q),sigma=float(sigma),lag=lag,residual=float(r))

def main():
 r={L:read(S/f'PHYSICS_L{L}.json') for L in (13,15,17,19,21)}
 chosen={}
 for pos in sorted({x['position'] for x in r[15]['native']}):
  a=next(x for x in sorted(r[15]['native'],key=lambda x:x['seed']) if x['position']==pos and x['law']=='iid' and x['schedule']=='constant' and x['amplitude']==0.)
  src=a['source'];legal=a['legal'];p=a['prior']
  support=[src]+[x for x in sorted(legal,key=lambda x:(-p[x],x)) if x!=src and max(abs(x//15-7),abs(x%15-7))<=6][:4]
  assert len(support)==5
  chosen[pos.split('_',1)[1]]=dict(actions=support,job=a['job'],source_sha=a['shard_sha'])
 atomic_json(S/'SPATIAL_REFERENCE_REGISTRATION.json',dict(support=chosen,rule='source and top four initial L15 NNUE-prior legal actions in common central 13x13 support; no perturbed outcomes; fixed support maps for all L',replicas=[8,16,32],coarse=True))
 loops=[];thermal=[];temps=[];protocol=[]
 for L,rr in r.items():
  native=rr['native'];groups=defaultdict(list)
  for a in native:
   if a['schedule']=='constant' and a['amplitude']==0.:groups[a['position'],a['law']].append(a)
  for (pos,law),a in sorted(groups.items()):
   assert len(a)==4
   # Snapshot common to the three candidate laws: actual IID h0 k8 first
   # registered seed from that L/position, not post-treatment selection.
   ref=sorted(groups[pos,'iid'],key=lambda x:x['seed'])[0];t=raw(L,ref)
   edges=next(x['edges'] for x in t['replicas'][0]['cuts'] if x['local_k']==8)
   support=[map_coord(x,L) for x in chosen[pos.split('_',1)[1]]['actions']]
   assert support[0]==a[0]['source'] and all(x in a[0]['legal'] for x in support)
   z=exact_wave_full_state(edges,support,4,law,support[0])
   q=lattice_action(z,(8,16,32),True)
   maps=projected_information(z)
   loops.append(dict(L=L,position=pos,law=law,source_job=ref['job'],source_sha=ref['shard_sha'],support=support,histories=len(z['words']),errors=z['errors'],Fisher=maps,values=q))
   # Three fixed knots: mean actual target policy over spatial quadrants,
   # first, middle and final block. Average over complete h0 histories.
   C=[];c=L//2
   for v in a:
    tr=raw(L,v);ev=sorted(tr['replicas'][0]['events'],key=lambda e:e['root_selection']['clock'])
    x=[]
    for e in ev:
     d=e['root_selection']['law'];sector=[2*int(i//L>=c)+int(i%L>=c) for i in d['actions']]
     x.append(np.bincount(sector,weights=d['p'],minlength=4))
    C.append(np.array(x).reshape(4,4,4).mean(1)[[0,2,3]])
   p=np.mean(C,0);p/=p.sum(1)[:,None]
   for T in (1.,4.,16.,64.):
    thermal.append(dict(L=L,position=pos,law=law,propensity=p.tolist(),**cycle(p,T)))
   for label in ('constant:0.2','early:0.2','late:0.2'):
    if label not in a[0]['protocols']:continue
    q=[x['protocols'][label] for x in a]
    protocol.append(dict(L=L,position=pos,law=law,protocol=label,sigma=st.mean(x['sigma'] for x in q),KL=st.mean(x['conditional_KL'] for x in q),weight=st.mean(x['weight'] for x in q),max_balance=max(abs(x['balance']) for x in q)))
  ids={x['job']:(x['position'],x['law']) for x in native if x['schedule']=='constant' and x['amplitude']==0.}
  a=S/'analysis'/f'roots-L{L}'/'local.jsonl';by=defaultdict(list)
  with a.open() as f:
   for l in f:
    x=json.loads(l)
    if x['job'] in ids:
     pos,law=ids[x['job']];by[pos,law,x['k']].append(x)
  for (pos,law,k),a in sorted(by.items()):
   assert len(a)==4;temps.append(dict(L=L,position=pos,law=law,k=k,H=st.mean(x['H'] for x in a),beta_status=dict(Counter(x['beta_status'] for x in a))))
 assert len(loops)==30 and len(thermal)==120 and len(protocol)==54
 out=dict(loops=loops,thermal=thermal,protocol=protocol,temperature=temps,scope='actual snapshots and same restricted geometric support; exact conditional lattice action and thermal model separate from actual path ratio')
 atomic_json(S/'SPATIAL_REFERENCE_RESULTS.json',out)
 print('REF',len(loops),len(thermal),len(protocol),len(temps),flush=True)
if __name__=='__main__':main()
