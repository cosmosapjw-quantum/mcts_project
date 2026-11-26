from base61 import *
import gzip,sys
sys.path.insert(0,str(O))
from train60 import load
import torch
rng=np.random.default_rng(620202);out=[]
with torch.no_grad():
 for g in (0,3):
  net,_=load(O/f'checkpoints/g{g}.pt')
  for d4,tag in [(False,f'response{g}w4'),(True,f'symmetry{g}')]:
   jobs=read(S/f'plans/{tag}.json');js=[jobs[i] for i in rng.choice(len(jobs),20,replace=False)]
   xs=[];vs=[];ps=[]
   for j in js:
    with gzip.open(S/'data'/tag/(j['id']+'.json.gz'),'rt') as f:q=json.load(f)['trace']
    b=np.asarray(q['board']);side=q['side'];me=1 if side==1 else 2
    xs.append(np.stack([b==me,b==3-me]).astype(np.float32));vs.append(q['evaluations'][0]['value']);ps.append(q['cuts'][0]['prior'])
    es=[e for e in q['events'] if e['eval_id'] is not None]
    for i in rng.choice(len(es),min(5,len(es)),replace=False):
     e=es[i];bb=b.copy();s=side
     for a in e['actions']:bb[a]=1 if s==1 else 2;s=-s
     me=1 if s==1 else 2;xs.append(np.stack([bb==me,bb==3-me]).astype(np.float32));vs.append(e['leaf_value']);ps.append(None)
   X=np.asarray(xs);pout=[];vout=[]
   for k in (range(8) if d4 else range(1)):
    p=PERM[k];z=X[:,:,np.argsort(p)].reshape(-1,2,9,9).copy();l,v=net(torch.from_numpy(z));pout.append(torch.softmax(l,1).numpy()[:,p]);vout.append(v.numpy())
   pp=np.mean(pout,axis=0);vv=np.mean(vout,axis=0)
   pe=max(float(abs(pp[i]-p).max()) for i,p in enumerate(ps) if p is not None);ve=float(abs(vv-vs).max())
   assert pe<3e-6 and ve<3e-5
   out.append(dict(generation=g,d4=d4,root_checks=sum(p is not None for p in ps),leaf_checks=sum(p is None for p in ps),max_policy_error=pe,max_value_error=ve,checkpoint_sha=sha(O/f'checkpoints/g{g}.pt'),torch=torch.__version__))
dump(S/'EVALUATOR62.json',out);print(out)
