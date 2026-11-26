from base61 import *
import gzip,sys
sys.path.insert(0,str(O))
from train60 import load
import torch
rng=np.random.default_rng(650211);out=[]
with torch.no_grad():
 for g in (0,3):
  net,_=load(O/f'checkpoints/g{g}.pt')
  for arm in ('baseline','persist_plus','persist_minus'):
   js=[j for j in read(S/f'plans/phase{g}.json') if j['arm']==arm]
   js=[js[i] for i in rng.choice(len(js),16,replace=False)]
   xs=[];vs=[];ps=[]
   for j in js:
    with gzip.open(S/f'data/phase{g}'/(j['id']+'.json.gz'),'rt') as f:q=json.load(f)['trace']
    b=np.asarray(q['board']);side=q['side'];me=1 if side==1 else 2
    xs.append(np.stack([b==me,b==3-me]).astype(np.float32));vs.append(q['evaluations'][0]['value']);ps.append(q['cuts'][0]['prior'])
    es=[e for e in q['events'] if e['eval_id'] is not None]
    for i in rng.choice(len(es),min(5,len(es)),replace=False):
     e=es[i];bb=b.copy();s=side
     for a in e['actions']:bb[a]=1 if s==1 else 2;s=-s
     me=1 if s==1 else 2;xs.append(np.stack([bb==me,bb==3-me]).astype(np.float32));vs.append(e['leaf_value']);ps.append(None)
   l,v=net(torch.from_numpy(np.asarray(xs).reshape(-1,2,9,9)));pp=torch.softmax(l,1).numpy();vv=v.numpy()
   pe=max(float(abs(pp[i]-p).max()) for i,p in enumerate(ps) if p is not None);ve=float(abs(vv-vs).max())
   assert pe<3e-6 and ve<3e-5
   out.append(dict(generation=g,arm=arm,roots=sum(p is not None for p in ps),leaves=sum(p is None for p in ps),max_policy_error=pe,max_value_error=ve,checkpoint_sha=sha(O/f'checkpoints/g{g}.pt'),torch=torch.__version__))
dump(S/'EVALUATOR65.json',out);print(out)
