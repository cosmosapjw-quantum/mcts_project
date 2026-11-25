"""Exact conditional CRN stencil covariance using actual categorical thresholds.
Ideal iid last-wave future uniforms, conditioned on completed pre-wave history."""
from pathlib import Path
import json,gzip,hashlib,math
from collections import defaultdict
import numpy as np
from mqft.physics53 import U
from mqft.physics50 import sectors
from mqft.client import sha256
from mqft.campaign import atomic_json
S=Path(__file__).resolve().parent
R=json.loads((S/'PHYSICS_RESULTS.json').read_text())
L=json.loads((S/'PHYSICS_L13.json').read_text())
assert sha256(S/'PHYSICS_L13.json')==R['size_results_sha']['L13']
G=defaultdict(dict)
for r in L['native']:
 if r['campaign']!='precision':continue
 G[r['replicate']][tuple(r['arm'])]=r
assert len(G)==32 and all(len(x)==13 for x in G.values())
q=next(x for x in R['groups'] if x['campaign']=='precision')
def coeff(row):
 h=row['h'];i=row['i'];j=row['j'];zero=(0.,0.,0.)
 def v(s,t=0.):
  a=np.zeros(3);a[i]=s*h
  if i!=j:a[j]=t*h
  return tuple(a)
 if row['order']==1:return {v(1):1/(2*h),v(-1):-1/(2*h)}
 if i==j:return {v(1):1/h**2,v(-1):1/h**2,zero:-2/h**2}
 return {v(s,t):s*t/(4*h*h) for s in (-1,1) for t in (-1,1)}
C=[coeff(x) for x in q['rows']];d=len(C);M=16384
rng=np.random.default_rng(547777);sums=np.zeros((M,d));actual=[];means=[];covs=[];draws=0
for rep,arms in sorted(G.items()):
 tables={};us=[]
 for arm,r in arms.items():
  f=S/'data/precision-L13'/(r['job']+'.jsonl.gz');assert sha256(f)==r['shard_sha']
  with gzip.open(f,'rt') as f:h,x,e=[json.loads(l) for l in f]
  ev=sorted(x['trace']['replicas'][0]['events'],key=lambda e:e['root_selection']['clock'])[-4:]
  z=[e['root_selection']['law'] for e in ev]
  p=np.array(z[0]['p']);a=np.array(z[0]['actions']);assert all(np.array_equal(a,e['actions']) and np.max(abs(p-e['p']))<1e-12 for e in z)
  w=np.cumsum(p);w[-1]=1.;us.append([e['u'] for e in z]);tables[arm]=(w,U[sectors(13,a)])
  assert np.allclose(np.array(r['I'][-4:])@U,U[sectors(13,a[np.searchsorted(w,us[-1],side='right')])],atol=1e-12,rtol=0)
 assert np.max(abs(np.array(us)-us[0]))==0
 b=np.unique(np.clip(np.concatenate([np.array([0.,1.])]+[v[0] for v in tables.values()]),0,1));width=np.diff(b);mid=b[:-1]+width/2
 def stencil(x):
  vals={k:v[np.searchsorted(w,x,side='right')] for k,(w,v) in tables.items()}
  return np.stack([sum(c*vals[k][...,row['i']] for k,c in a.items()) for a,row in zip(C,q['rows'])],axis=-1)
 Z=stencil(mid);mu=width@Z;v=Z-mu;cov=(v.T*width)@v/4
 assert np.max(abs(mu-np.array([sum(c* (np.asarray(arms[k]['P'][-4])@U)[row['i']] for k,c in a.items()) for a,row in zip(C,q['rows'])])))<1e-10
 obs=stencil(np.asarray(us[0])).mean(0);actual.append(obs);means.append(mu);covs.append(cov)
 v=stencil(rng.random((M,4))).mean(1);sums+=v-mu;draws+=M*4
obs=np.mean(actual,0);mu=np.mean(means,0);cov=np.sum(covs,0)/32**2;se=np.sqrt(np.diag(cov));z=(obs-mu)/se;zsim=sums/32/se
assert np.max(abs(obs-np.array([x['direct'][x['i']] for x in q['rows']])))<1e-10
assert np.max(abs(mu-np.array([x['conditional'][x['i']] for x in q['rows']])))<1e-10
p=(1+np.sum(abs(zsim)>=abs(z),axis=0))/(M+1);pmax=(1+np.sum(np.max(abs(zsim),axis=1)>=np.max(abs(z))))/(M+1)
r=dict(scope='ideal conditional iid shared uniforms, actual action CDFs; simulated calibration not native histories; six prespecified scalar outputs, max statistic controls this finite family only',M=M,simulated_uniforms=draws,labels=[f"{x['label']} h={x['h']}" for x in q['rows']],observed=obs.tolist(),conditional=mu.tolist(),exact_conditional_SE=se.tolist(),z=z.tolist(),conditional_tail_MC=p.tolist(),max_abs_z=float(np.max(abs(z))),max_stat_tail_MC=float(pmax),conditional_covariance=cov.tolist(),source_sha=sha256(S/'PHYSICS_RESULTS.json'))
atomic_json(S/'CRN_CALIBRATION.json',r)
print(json.dumps(r,indent=2),flush=True)
