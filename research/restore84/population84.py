"""Frozen development-only model selection and external pre-search distribution prediction."""
from base61 import *
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import t as student
from itertools import product
FAMS=['g0','g3','Rapfi'];MOD=['pool','ply','root_kernel','root_ridge','child_kernel','combined_kernel','combined_ridge']
BASE=MOD[:4];CAND=MOD[4:];HS=[.35,.7,1.4,2.8];AL=[.01,.1,1.,10.]
GRID=np.array(list(product([-4.,-2.,0.,2.,4.],repeat=2))+list(product([-8.,8.],repeat=2)))
SCALE=np.array([1.,.5]);OFF=[0,48,72,104]
RF=['ply','root_value','log_pair_mass','log_prior_max','prior_entropy_norm','log_pair_ratio','pair_distance','mean_radial_stone_distance','same_color_adjacency']
CF=['child_mean','child_sd','child_min','child_max','child_q25','child_median','child_q75','prior_weighted_mean','prior_weighted_sd','source_mean','source_difference','prior_tilt_mean','value_softmax_pair_mass','prior_tilt_pair_mass','terminal_fraction']
def features(r):
 p=np.asarray(r['p'],float);L=int(np.sqrt(len(p)));a,b=r['pair'];legal=np.flatnonzero(p>0)
 q=p[legal]/p[legal].sum();ch={x['a']:x for x in r['children']};v=np.array([ch[int(i)]['root_v'] for i in legal]);assert len(ch)==len(legal)
 ix=np.isin(legal,[a,b]);m=r['moves'];pos=np.array([divmod(i,L) for i in m]);dist=np.linalg.norm(np.array(divmod(a,L))-np.array(divmod(b,L)))/(L-1)
 rad=np.mean(np.linalg.norm(pos-(L-1)/2,axis=1))/(L-1)
 same=[np.max(abs(pos[i]-pos[j]))<=1 for i in range(len(m)) for j in range(i) if (i-j)%2==0]
 f=np.array([len(m),r['v'],np.log(p[a]+p[b]),np.log(p.max()),entropy(q)/np.log(len(q)),np.log(p[a]/p[b]),dist,rad,np.mean(same)])
 e=np.exp((10/3)*(v-v.max()));w=q*e;w/=w.sum();mn=q@v
 u=np.r_[v.mean(),v.std(),v.min(),v.max(),np.quantile(v,[.25,.5,.75]),mn,np.sqrt(q@((v-mn)**2)),.5*(ch[a]['root_v']+ch[b]['root_v']),ch[a]['root_v']-ch[b]['root_v'],w@v,e[ix].sum()/e.sum(),w[ix].sum(),np.mean([x['terminal'] for x in r['children']])]
 assert len(f)==len(RF) and len(u)==len(CF) and np.isfinite(f).all() and np.isfinite(u).all()
 return f,u
def data():
 O=B/'mqft_restore82_20261127';a=read(O/'AUDIT82.json')
 assert sha(O/'POPULATION82.npz')==a['array_sha']
 with np.load(O/'POPULATION82.npz') as z:d={k:z[k] for k in ('meta','X','root')}
 l=read(S/'LANDSCAPE84.json');assert l['status']=='COMPLETE';r=l['rows']
 feature={};err=0.
 for x in r:
  c=OFF[x['group']]+x['core'];key=(x['family'],c);assert key not in feature;feature[key]=features(x)
 for m,x in zip(d['meta'],d['root']):
  key=(m[1],OFF[m[0]]+m[2]);err=max(err,float(abs(x-feature[key][0]).max()))
 assert err<2e-6,err
 X=d['X'];M=d['meta'];roots=np.array([OFF[m[0]]+m[2] for m in M])
 return feature,X,M,roots,err
def ci(v):
 v=np.asarray(v);m=v.mean(0);h=student.ppf(.975,len(v)-1)*v.std(0,ddof=1)/np.sqrt(len(v));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def prepare(g,F,X,M,C,train):
 ix=(M[:,1]==g)&np.isin(C,train);x=X[ix];c=C[ix];keys=np.unique(c);assert np.array_equal(keys,np.sort(train))
 r=np.stack([F[g,int(k)][0] for k in keys]);u=np.stack([F[g,int(k)][1] for k in keys]);w=np.array([1/(len(keys)*sum(c==k)) for k in c]);assert abs(w.sum()-1)<1e-12
 # Root-level standardization, each geometry weight equal.
 def norm(z):return z.mean(0),np.maximum(z.std(0),1e-8)
 rm,rs=norm(r);um,us=norm(u);rr=(r-rm)/rs;uu=(u-um)/us
 z=np.column_stack([rr,uu]);y=np.stack([x[c==k].mean(0) for k in keys]);i=np.searchsorted(keys,c)
 return dict(x=x,c=c,keys=keys,w=w,r=rr,u=uu,z=z,y=y,i=i,rm=rm,rs=rs,um=um,us=us,ply=r[:,0],dist=cdist(x*SCALE,x*SCALE))
def fit_ridge(C,model,alpha):
 z=C['r'] if model=='root_ridge' else C['z'];y=C['y'];ym=y.mean(0)
 coef=np.linalg.solve(z.T@z/len(z)+alpha*np.eye(z.shape[1]),z.T@(y-ym)/len(z))
 fitted=np.clip(z@coef+ym,[0,-1],[1,1]);res=C['x']-fitted[C['i']]
 return coef,ym,res
def predict(C,f,model,h=None,reg=None):
 r,u=f;rr=(r-C['rm'])/C['rs'];uu=(u-C['um'])/C['us'];p=C['w'].copy();x=C['x'];clip=0.
 if model=='ply':
  p*=C['ply'][C['i']]==r[0];assert p.sum()>0;p/=p.sum()
 elif model.endswith('kernel'):
  dr=np.mean((C['r']-rr)**2,1);du=np.mean((C['u']-uu)**2,1)
  ds=dr if model=='root_kernel' else du if model=='child_kernel' else .5*(dr+du)
  a=np.log(p)-ds[C['i']]/(2*h*h);p=.98*np.exp(a-logsumexp(a))+.02*p
 elif model.endswith('ridge'):
  coef,ym,res=reg if reg is not None else fit_ridge(C,model,h)
  z=rr if model=='root_ridge' else np.r_[rr,uu];mu=np.clip(z@coef+ym,[0,-1],[1,1]);raw=res+mu
  clipped=np.clip(raw,[0,-1],[1,1]);clip=float(p@np.any(raw!=clipped,axis=1));x=clipped.copy();x[:,0]=np.rint(x[:,0]*128)/128
 assert abs(p.sum()-1)<1e-12 and np.all(p>=0) and np.all((x[:,0]>=0)&(x[:,0]<=1)) and np.max(abs(x[:,1]))<=1 and abs(x[:,0]*128-np.rint(x[:,0]*128)).max()<1e-12
 return p,x,clip
def score(C,p,x,y):
 dy=cdist(y*SCALE,x*SCALE);D=C['dist'] if x is C['x'] else cdist(x*SCALE,x*SCALE)
 return dy@p-.5*p@D@p
def options(m):return HS if m.endswith('kernel') else AL if m.endswith('ridge') else [None]
def fit():
 F,X,M,C,err=data();out=[]
 for g in range(3):
  scores={m:{str(h):[] for h in options(m)} for m in MOD}
  for fold in range(8):
   train=np.array([i for i in range(104) if i%8!=fold]);test=np.array([i for i in range(104) if i%8==fold]);A=prepare(g,F,X,M,C,train)
   assert not set(train)&set(test)
   for m in MOD:
    for h in options(m):
     reg=fit_ridge(A,m,h) if m.endswith('ridge') else None
     for k in test:
      y=X[(M[:,1]==g)&(C==k)];p,x,clip=predict(A,F[g,int(k)],m,h,reg);s=score(A,p,x,y)
      scores[m][str(h)].append(float(s.mean()))
   print('CV',FAMS[g],fold,flush=True)
  cv={m:{k:float(np.mean(v)) for k,v in q.items()} for m,q in scores.items()}
  chosen={m:min(options(m),key=lambda h:(cv[m][str(h)],str(h))) for m in MOD}
  s={m:cv[m][str(chosen[m])] for m in MOD}
  b=min(BASE,key=lambda m:(s[m],m));a=min(CAND,key=lambda m:(s[m],m))
  out.append(dict(family=FAMS[g],cv=cv,hyperparameters=chosen,selected_baseline=b,selected_candidate=a,selected_cv_scores=s))
  print('FROZEN',FAMS[g],b,a,chosen,s,flush=True)
 dump(S/'MODEL84.json',dict(status='TRAIN_ONLY_FROZEN',development_sha=sha(B/'mqft_restore82_20261127/POPULATION82.npz'),landscape_sha=sha(S/'LANDSCAPE84.json'),root_feature_max_error=err,models=MOD,source_grid=GRID.tolist(),families=out,scope='104 development roots only; 8 orbit-block folds, best baseline and candidate frozen before fresh stochastic search. No R84 outcomes used.'))
def evaluate():
 F,X,M,C,err=data();model=read(S/'MODEL84.json');assert model['landscape_sha']==sha(S/'LANDSCAPE84.json')
 A=read(S/'AUDIT84.json');assert sha(S/'FRESH84.npz')==A['array_sha']
 with np.load(S/'FRESH84.npz') as z:Z=z['X'];T=z['meta']
 allres=[];pred=[]
 for g in range(3):
  tr=prepare(g,F,X,M,C,np.arange(104));fm=model['families'][g];q=[]
  for m in MOD:
   h=fm['hyperparameters'][m];reg=fit_ridge(tr,m,h) if m.endswith('ridge') else None;rows=[]
   for k in range(48):
    y=Z[(T[:,1]==g)&(T[:,2]==k)];assert len(y)==16;p,x,clip=predict(tr,F[g,k+104],m,h,reg);es=score(tr,p,x,y);ex=np.exp(x@GRID.T);mgf=p@ex;W=np.log(mgf)
    truth=np.log(np.exp(y@GRID.T).mean(0));split=np.log(np.exp(y[::2]@GRID.T).mean(0))-np.log(np.exp(y[1::2]@GRID.T).mean(0))
    group=np.stack([np.bincount(tr['i'],weights=p*ex[:,j],minlength=104)/mgf[j] for j in range(len(GRID))]);ess=1/(group*group).sum(1)
    # finite empirical marginal 90% prediction interval coverage
    cv=[]
    for j in range(2):
     o=np.argsort(x[:,j]);cum=np.cumsum(p[o]);lo=x[o[np.searchsorted(cum,.05)],j];hi=x[o[np.searchsorted(cum,.95)],j];cv.append(float(np.mean((y[:,j]>=lo-1e-12)&(y[:,j]<=hi+1e-12))))
    mu=p@x;cov=((x-mu)*p[:,None]).T@(x-mu)
    rows.append(dict(core=k,energy=float(es.mean()),W=W.tolist(),truth_W=truth.tolist(),split=split.tolist(),min_ESS_regular=float(ess[:25].min()),min_ESS_stress=float(ess[25:].min()),coverage90=cv,clipped_mass=clip,pred_mean=mu.tolist(),cov=cov.tolist()))
    pred.append(dict(family=g,model=m,**rows[-1]))
   W=np.array([r['W'] for r in rows]);V=np.array([r['truth_W'] for r in rows]);d=W-V;ss=np.array([r['split'] for r in rows]);pop=logsumexp(W,axis=0)-np.log(48);pop0=logsumexp(V,axis=0)-np.log(48)
   q.append(dict(model=m,hyperparameter=h,energy_ci=ci([r['energy'] for r in rows]),W_RMSE=float(np.sqrt(np.mean(d[:,:25]**2))),W_stress_RMSE=float(np.sqrt(np.mean(d[:,25:]**2))),population_W_RMSE=float(np.sqrt(np.mean((pop[:25]-pop0[:25])**2))),split_half_W_RMSE=float(np.sqrt(np.mean(ss[:,:25]**2))),coverage90=np.mean([r['coverage90'] for r in rows],0).tolist(),min_root_ESS_regular=min(r['min_ESS_regular'] for r in rows),min_root_ESS_stress=min(r['min_ESS_stress'] for r in rows),mean_clipped_mass=float(np.mean([r['clipped_mass'] for r in rows])),rows=rows))
  a=next(z for z in q if z['model']==fm['selected_candidate']);b=next(z for z in q if z['model']==fm['selected_baseline']);gain=ci(np.array([r['energy'] for r in b['rows']])-np.array([r['energy'] for r in a['rows']]))
  passed=gain[1]>0 and a['W_RMSE']<b['W_RMSE']
  allres.append(dict(family=FAMS[g],selected_baseline=b['model'],selected_candidate=a['model'],energy_gain_ci=gain,registered_pass=bool(passed),models=q))
  print('TEST',FAMS[g],b['model'],a['model'],gain,'W',b['W_RMSE'],a['W_RMSE'],'PASS',passed,flush=True)
 dump(S/'TRANSFER84.json',dict(status='EVALUATED_FROZEN_MODELS',model_sha=sha(S/'MODEL84.json'),fresh_array_sha=A['array_sha'],landscape_sha=sha(S/'LANDSCAPE84.json'),results=allres,scope='One independent new 48-root cohort. Pointwise root-block CI, no test refit; finite donor law not exact native history distribution.'))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='fit':fit()
 elif sys.argv[1]=='evaluate':evaluate()
 else:raise ValueError(sys.argv)
