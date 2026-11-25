"""Exact ideal conditional common-uniform final-wave contrast distribution."""
from pathlib import Path
import json,gzip,math
import numpy as np
from collections import defaultdict
from mqft.client import sha256
from mqft.campaign import atomic_json
S=Path(__file__).resolve().parent
def raw(r):
 p=S/'data'/f"{r['campaign']}-L{r['L']}"/(r['job']+'.jsonl.gz');assert sha256(p)==r['shard_sha']
 with gzip.open(p,'rt') as f:hd,x,e=[json.loads(l) for l in f]
 assert hd['job']['id']==r['job'];return sorted(x['trace']['replicas'][0]['events'],key=lambda e:e['root_selection']['clock'])
def dist_pair(a,b):
 # union of target action probability intervals on shared uniform
 p=[]
 for r in (a,b):
  law=[x['root_selection']['law'] for x in raw(r)[-4:]]
  assert all(x['actions']==law[0]['actions'] and np.array_equal(x['p'],law[0]['p']) for x in law)
  x=law[0];i=x['actions'].index(r['source']);c=np.r_[0.,np.cumsum(x['p'])];c[-1]=1.
  p.append((float(c[i]),float(c[i+1]),[v['u'] for v in law]))
 assert np.array_equal(p[0][2],p[1][2])
 lo,hi,u=p[0];l,h,_=p[1];c=max(0.,min(hi,h)-max(lo,l));q=hi-lo;v=h-l
 pr=np.array([[1-q-v+c,v-c],[q-c,c]]);assert pr.min()>-1e-12
 pr=np.maximum(pr,0);pr/=pr.sum()
 for rr,(l0,h0,uu) in zip((a,b),p):
  assert np.array_equal(rr['I'][-4:],(np.asarray(uu)>=l0)&(np.asarray(uu)<h0))
 d=np.zeros((5,5));d[0,0]=1
 for i in range(4):
  z=np.zeros_like(d)
  for x in (0,1):
   for y in (0,1):z[x:,y:]+=pr[x,y]*d[:5-x,:5-y]
  d=z
 assert abs(d.sum()-1)<1e-12
 v0=np.zeros(9);v1=np.zeros(3);s=int(not any(a['I'][:-4]));assert a['I'][:-4]==b['I'][:-4]
 for i in range(5):
  for j in range(5):v0[i-j+4]+=d[i,j];v1[s*(int(i>0)-int(j>0))+1]+=d[i,j]
 return v0,v1,int(sum(a['I'][-4:])-sum(b['I'][-4:])),int(a['detected']-b['detected'])
rows=[]
for L in (13,15,17,19,21):
 a=json.loads((S/f'PHYSICS_L{L}.json').read_text())['native'];groups=defaultdict(dict)
 for r in a:
  if r['campaign']=='B16b1':groups[(r['position'],r['seed'])][r['amplitude']]=r
 for (pos,seed),z in sorted(groups.items()):
  p,q,obs0,obs1=dist_pair(z[.2],z[-.2]);rows.append(dict(L=L,position=pos,seed=seed,occ_pmf=p.tolist(),hit_pmf=q.tolist(),obs_occ=obs0,obs_hit=obs1))
assert len(rows)==60
def aggregate(rr,key,obskey,offset,scale):
 p=np.ones(1)
 for r in rr:p=np.convolve(p,r[key])
 x=np.arange(len(p))-offset*len(rr);mu=p@x;var=p@((x-mu)**2);obs=sum(r[obskey] for r in rr)
 tail=p[np.abs(x-mu)>=abs(obs-mu)-1e-12].sum()
 return dict(mean=float(mu*scale/len(rr)),observed=float(obs*scale/len(rr)),conditional_SE=float(np.sqrt(var)*scale/len(rr)),two_sided_tail=float(tail),zero_contrast_probability=float(p[offset*len(rr)]),support=x.tolist(),pmf=p.tolist())
out=[]
for L in (13,15,17,19,21,None):
 rr=[r for r in rows if L is None or r['L']==L];out.append(dict(L=L,n=len(rr),occupancy=aggregate(rr,'occ_pmf','obs_occ',4,1/.4/4),detection=aggregate(rr,'hit_pmf','obs_hit',1,1/.4)))
atomic_json(S/'CRN_VALIDATION.json',dict(scope='conditional on actual independent prefixes; ideal shared uniform draws; no PRNG distribution proof or unconditioned CI',source_sha=sha256(S/'PHYSICS_RESULTS.json'),rows=rows,aggregates=out))
for r in out:print(r['L'],r['n'],{k:{a:r[k][a] for a in ('mean','observed','conditional_SE','two_sided_tail','zero_contrast_probability')} for k in ('occupancy','detection')})
