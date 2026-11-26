"""Secondary exploratory diagnostics, not registered primary inference."""
import json, numpy as np
from pathlib import Path
from predict59 import sha,dump,predict
S=Path(__file__).resolve().parent
a=np.load(S/'TRAIN_FEATURES.npz'); b=np.load(S/'HELDOUT_PREDICTIONS.npz')
m=json.loads((S/'MODEL59.json').read_text());r=json.loads((S/'PHYSICS_RESULTS.json').read_text())
assert sha(S/'MODEL59.json')==r['model_sha256']
assert sha(S/'HELDOUT_PREDICTIONS.npz')==r['heldout_predictions_sha256']
assert sha(S/'TRAIN_FEATURES.npz')==m['train_features_sha256']
assert sha(S/'HELDOUT_MANIFEST.json')==r['heldout_manifest_sha256']
assert np.allclose(np.mean((b['pred']-b['y'][:,None])**2,axis=0),r['mse'],rtol=0,atol=1e-13)
audit={}
for key in ['state','history']:
 q=m['models']['180'][key];d=q['d'];x=(a['X'][:,:d]-q['mu'])/q['scale'];n=len(x)
 z=np.column_stack([np.ones(n),x])/np.sqrt(n)
 A=np.vstack([z,np.diag(np.sqrt([0]+[.02]*d))]); y=np.r_[a['y']/np.sqrt(n),np.zeros(d+1)]
 w=np.linalg.lstsq(A,y,rcond=None)[0];err=np.max(abs(w-q['w']))
 pred=np.clip(w[0]+((b['X'][:,:d]-q['mu'])/q['scale'])@w[1:],-1,1)
 pe=np.max(abs(pred-b['pred'][:,3 if key=='state' else 4]))
 assert err<1e-10 and pe<1e-10
 audit[key]=dict(coef_max_error=float(err),prediction_max_error=float(pe),augmented_condition=float(np.linalg.cond(A)))
def waves(z,y):
 out=[]
 for i in np.unique(z['job']):
  ix=z['job']==i;p=z['phase'][ix];ys=y[ix]
  out.append((int(z['L'][ix][0]),int(z['core'][ix][0]),float(z['X'][ix,1][0]),np.array([ys[p==k].mean() for k in np.unique(p)])))
 return out
def acf(w):
 return [float(np.corrcoef(np.array([(x[i],x[i+k]) for x in w for i in range(len(x)-k)]).T)[0,1]) for k in range(1,6)]
def within(w):
 out=[]
 for l,c in sorted({(x[0],x[1]) for x in w}):
  group=np.stack([x[3] for x in w if (x[0],x[1])==(l,c)])
  out.extend(group-group.mean(0))
 return acf(out)
W={k:waves(b,y) for k,y in [('raw',b['y']),('state_resid',b['y']-b['pred'][:,3]),('history_resid',b['y']-b['pred'][:,4])]}
centered={k:within(w) for k,w in W.items()}
root_acf=[dict(L=l,core=c,**{k:acf([x[3] for x in w if (x[0],x[1])==(l,c)])[0] for k,w in W.items()}) for l,c in sorted({(x[0],x[1]) for x in W['raw']})]
# Clock-conditioned categorical predictions, train only matched beta; compare on identical test contexts.
from collections import defaultdict
def examples(w):
 for l,c,beta,x in w:
  v=np.digitize(x,[-1/3,1/3])
  for t in range(2,len(v)):yield (t,int(v[t-2]),int(v[t-1]),int(v[t]),l,c)
train=[x for x in waves(a,a['y']) if abs(x[2]-10/3)<1e-9]
counts=[defaultdict(lambda:np.ones(3)) for _ in range(3)]
for t,u,v,y,l,c in examples(train):
 for j,k in enumerate([(t,),(t,v),(t,u,v)]):counts[j][k][y]+=1
err=[];js=[]
for t,u,v,y,l,c in examples(W['raw']):
 ps=[counts[j][k]/counts[j][k].sum() for j,k in enumerate([(t,),(t,v),(t,u,v)])]
 z=(ps[1]+ps[2])/2
 js.append(.5*sum(ps[1]*np.log(ps[1]/z))+.5*sum(ps[2]*np.log(ps[2]/z)))
 err.append([-np.log(p[y]) for p in ps])
q=dict(scope='exploratory post-primary analysis; centering uses same heldout root/phase sample means, not predictive model; finite wave lags only',audit=audit,root_centered_acf=centered,root_lag1=root_acf,transition_prediction=dict(n_train_histories=len(train),n_test_transitions=len(err),orders=[0,1,2],nll=np.mean(err,axis=0).tolist(),mean_js_nats=float(np.mean(js)),prior='Laplace 1 per bin, clock conditioned'))
dump(S/'DIAGNOSTICS59.json',q);print(json.dumps(q,indent=2))
