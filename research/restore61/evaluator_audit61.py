from base61 import *
import sys
sys.path.insert(0,str(O))
from train60 import load
import torch
rng=np.random.default_rng(610202)
reg=read(S/'REGISTRATION.json')
rr=read(S/'CONTROL_AUDIT61.json')['jobs']
out=[]
with torch.no_grad():
 for g in (0,3):
  m,_=load(O/f'checkpoints/g{g}.pt')
  for d4 in (False,True):
   jobs=[x['job'] for x in rr if x['job']['generation']==g and x['job']['d4']==d4]
   jobs=[jobs[i] for i in rng.choice(len(jobs),24,replace=False)]
   features=[];values=[];priors=[]
   for j in jobs:
    q=read(S/'data'/f"{j['cohort']}{g}"/(j['id']+'.json'));b=np.asarray(q['board']);side=q['side'];me=1 if side==1 else 2
    features.append(np.stack([b==me,b==3-me]).astype(np.float32));values.append(q['evaluations'][0]['value']);priors.append(q['cuts'][0]['prior'])
    es=[e for e in q['events'] if e['eval_id'] is not None]
    for i in rng.choice(len(es),min(6,len(es)),replace=False):
     e=es[i];bb=b.copy();s=side
     for a in e['actions']:bb[a]=1 if s==1 else 2;s=-s
     me=1 if s==1 else 2;features.append(np.stack([bb==me,bb==3-me]).astype(np.float32));values.append(e['leaf_value']);priors.append(None)
   x=np.stack(features);pred=[];val=[]
   for k in (range(8) if d4 else range(1)):
    p=PERM[k];z=x[:,:,np.argsort(p)].reshape(-1,2,9,9).copy();l,v=m(torch.from_numpy(z));pred.append(torch.softmax(l,1).numpy()[:,p]);val.append(v.numpy())
   pp=np.mean(pred,axis=0);vv=np.mean(val,axis=0)
   pe=max(float(abs(pp[i]-p).max()) for i,p in enumerate(priors) if p is not None);ve=float(abs(vv-values).max())
   assert pe<3e-6 and ve<3e-5
   out.append(dict(generation=g,d4=d4,root_checks=sum(p is not None for p in priors),leaf_checks=sum(p is None for p in priors),max_policy_error=pe,max_value_error=ve,torch=torch.__version__,checkpoint_sha=sha(O/f'checkpoints/g{g}.pt')))
dump(S/'EVALUATOR_AUDIT61.json',out);print(out)
