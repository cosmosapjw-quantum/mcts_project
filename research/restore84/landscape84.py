"""Evaluate all legal children before new stochastic searches."""
from base61 import *
from arena68 import verify,P
import subprocess,os,time
R=read(S/'REGISTRATION.json');O=B/'mqft_restore82_20261127';assert verify(P/'.external/rapfi')==R['external']
b=read(S/'BUILD84.json');assert sha(S/'mqft_landscape84')==b['binary_sha']
C=[]
for group,st in [(0,80),(1,81),(2,82),(3,84)]:
 q=read(B/f'mqft_restore{st}_20261127/REGISTRATION.json')
 for c in q['cores']:C.append(dict(group=group,core=c['id'],moves=c['moves'],pair=c['sources'],orbit=c['orbit']))
assert len(C)==152 and len({c['orbit'] for c in C})==152
dump(S/'LANDSCAPE_ROOTS84.json',C)
res=[];times=[]
for g in range(3):
 def emb(a):return (a//9+3)*15+a%9+3 if g==2 else a
 moves=[[emb(a) for a in c['moves']] for c in C]
 p=S/'plans'/f'landscape{g}.json';dump(p,moves)
 w='rapfi' if g==2 else str(B/R['weights'][str(3*g)]['path'])
 cmd=[str(S/'mqft_landscape84'),'landscape84',w,str(p),'8']
 env=dict(os.environ,MQFT_RAPFI_DIR=str(P/'.external/rapfi'),MQFT_RAPFI_WORKERS='8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 t=time.monotonic()
 with (S/'logs'/f'landscape{g}.log').open('w') as f:raw=subprocess.check_output(cmd,env=env,stderr=f,text=True)
 v=json.loads(raw);assert len(v)==len(C)
 for c,x,m in zip(C,v,moves):
  assert x['moves']==m and x['side']==1 and len(x['p'])==(225 if g==2 else 81)
  legal=set(range(len(x['p'])))-set(m);assert {a['a'] for a in x['children']}==legal and len(x['children'])==len(legal)
  assert x['calls']==1+sum(not a['terminal'] for a in x['children'])
  assert all(a['side']==-1 and a['root_v']==-a['child_v'] and abs(a['root_v'])<=1 for a in x['children'])
  assert all(x['p'][a]>0 for a in legal) and all(x['p'][a]==0 for a in m)
  x.update(group=c['group'],core=c['core'],family=g,pair=[emb(a) for a in c['pair']],orbit=c['orbit'])
  res.append(x)
 times.append(dict(family=g,roots=len(v),calls=sum(x['calls'] for x in v),seconds=time.monotonic()-t,command=cmd,fixture_sha=sha(p),terminal_children=sum(a['terminal'] for x in v for a in x['children'])))
 dump(S/'LANDSCAPE84.json',dict(status='COMPLETE' if g==2 else 'PARTIAL',rows=res,timing=times,binary_sha=b['binary_sha'],scope='All legal child states direct evaluator, before any R84 stochastic search; wall times concurrent, no search budget included.'))
 print('LANDSCAPE',times[-1],flush=True)
assert len(res)==456
