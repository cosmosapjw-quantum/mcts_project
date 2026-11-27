"""Root-balanced conditional empirical laws; blocked CV and fixed external evaluation."""
from base61 import *
from scipy.spatial.distance import cdist
from scipy.stats import t as student
from scipy.special import logsumexp
from itertools import product
FAMS=['g0','g3','Rapfi'];MODELS=['pool','ply','root','prefix_pool','prefix_kernel','root_history']
KERNEL=['root','prefix_kernel','root_history'];HS=[.35,.7,1.4,2.8]
GRID=np.array(list(product([-4.,-2.,0.,2.,4.],repeat=2))+list(product([-8.,8.],repeat=2)))
SCALE=np.array([1.,.5])
def arrays():
 a=read(S/'AUDIT82.json');assert sha(S/'POPULATION82.npz')==a['array_sha']
 with np.load(S/'POPULATION82.npz') as z:return {k:z[k] for k in z.files}
def subset(d,ix):return {k:v[ix] for k,v in d.items() if k not in ('root_fields','history_fields')}
def balanced(c):
 _,i,n=np.unique(c,return_inverse=True,return_counts=True);w=1/n[i];return w/w.sum()
def stand(x,w):
 mu=w@x;s=np.sqrt(w@((x-mu)**2));return mu,np.maximum(s,1e-8)
def rootmeans(v,c):return np.stack([v[c==i].mean(0) for i in np.unique(c)])
def ci(a):
 a=np.asarray(a);m=a.mean(0);h=student.ppf(.975,len(a)-1)*a.std(0,ddof=1)/np.sqrt(len(a));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def setup(tr,te):
 w=balanced(tr['meta'][:,2]);rmu,rs=stand(tr['root'],w);hmu,hs=stand(tr['history'],w)
 r=(tr['root']-rmu)/rs;q=(te['root']-rmu)/rs;v=(tr['history']-hmu)/hs;u=(te['history']-hmu)/hs
 dr=cdist(q,r,'sqeuclidean')/r.shape[1];dh=cdist(u,v,'sqeuclidean')/v.shape[1]
 # Common prefix translation cancels in pairwise energy distances.
 e={}
 for name,k,f in [('root','X',1.),('prefix','V',.75)]:
  x=tr[k]*SCALE*f;y=te[k]*SCALE*f
  e[name]=(cdist(y,x),cdist(x,x))
 return dict(w=w,dr=dr,dh=dh,e=e,ply=(te['root'][:,0,None]==tr['root'][None,:,0]),rmu=rmu,rs=rs,hmu=hmu,hs=hs)
def weights(c,model,h=None):
 w=c['w'];N=len(c['dr'])
 if model in ('pool','prefix_pool'):return np.broadcast_to(w,(N,len(w))).copy()
 if model=='ply':
  p=c['ply']*w[None,:];assert np.all(p.sum(1)>0);return p/p.sum(1,keepdims=True)
 d=c['dr'] if model=='root' else c['dh'] if model=='prefix_kernel' else (c['dr']+c['dh'])/2
 assert model in KERNEL and h in HS
 z=np.log(w)[None,:]-d/(2*h*h);p=np.exp(z-logsumexp(z,axis=1)[:,None]);return .98*p+.02*w[None,:]
def es(c,p,model):
 d,D=c['e']['root' if model in ('pool','ply','root') else 'prefix']
 return np.einsum('ij,ij->i',p,d)-.5*np.einsum('ij,ij->i',p@D,p)
def fit():
 d=arrays();M=d['meta'];out=[]
 for g in range(3):
  a=subset(d,(M[:,0]==0)&(M[:,1]==g));c=a['meta'][:,2];scores={k:{h:[] for h in HS} for k in KERNEL};base={k:[] for k in ('pool','ply','prefix_pool')};folds=[]
  for f in range(6):
   tr=subset(a,c%6!=f);te=subset(a,c%6==f);C=setup(tr,te);core=te['meta'][:,2];assert len(set(core))==8 and not set(core)&set(tr['meta'][:,2]);folds.append(dict(fold=f,train_roots=sorted(set(tr['meta'][:,2].tolist())),test_roots=sorted(set(core.tolist()))))
   for k in base:base[k].extend(rootmeans(es(C,weights(C,k),k),core).tolist())
   for k in KERNEL:
    for h in HS:scores[k][h].extend(rootmeans(es(C,weights(C,k,h),k),core).tolist())
  # Tie resolved to larger h; never touch external labels for model selection.
  ms={k:{str(h):float(np.mean(v)) for h,v in s.items()} for k,s in scores.items()}
  best={k:min(HS,key=lambda h:(ms[k][str(h)],-h)) for k in KERNEL}
  C=setup(a,a);out.append(dict(family=FAMS[g],n_train=len(c),train_roots=len(set(c)),bandwidth=best,cv_es=ms,cv_baselines={k:float(np.mean(v)) for k,v in base.items()},folds=folds,root_mean=C['rmu'].tolist(),root_sd=C['rs'].tolist(),history_mean=C['hmu'].tolist(),history_sd=C['hs'].tolist()))
  print('FIT',FAMS[g],best,ms,flush=True)
 dump(S/'MODEL82.json',dict(status='TRAIN_ONLY_FROZEN',input_sha=sha(S/'POPULATION82.npz'),train_rows=1920,model_names=MODELS,bandwidth_grid=HS,uniform_floor=.02,source_grid=GRID.tolist(),families=out,scope='Fit only group0; blocked six-fold by complete orbit. No external cohort labels accessed for choice.'))
def mixture(a):
 # Exact tilted law, root-balanced empirical population.
 x=a['X'];c=a['meta'][:,2];p=balanced(c);z=[]
 for J in GRID:
  v=np.log(p)+x@J;W=float(logsumexp(v));q=np.exp(v-W);mu=q@x;cen=x-mu;H=(cen*q[:,None]).T@cen
  Wc=[];mc=[];Hc=[]
  for k in np.unique(c):
   y=x[c==k];s=y@J;w=float(logsumexp(s)-np.log(len(y)));r=np.exp(s-logsumexp(s));m=r@y;v0=y-m
   Wc.append(w);mc.append(m);Hc.append((v0*r[:,None]).T@v0)
  Wc=np.array(Wc);mc=np.array(mc);Hc=np.array(Hc);tilt=np.exp(Wc-logsumexp(Wc))
  within=np.einsum('i,ijk->jk',tilt,Hc);between=((mc-mu)*tilt[:,None]).T@(mc-mu)
  errs=[abs(W-(logsumexp(Wc)-np.log(len(Wc)))),abs(mu-tilt@mc).max(),abs(H-within-between).max()]
  assert max(errs)<1e-11 and W>=Wc.mean()-1e-12
  z.append(dict(J=J.tolist(),W=W,mean_conditional_W=float(Wc.mean()),jensen_gap=float(W-Wc.mean()),root_tilt_ESS=float(1/(tilt@tilt)),mean=mu.tolist(),cov=H.tolist(),within=within.tolist(),between=between.tolist(),error=float(max(errs))))
 return z
def predict(tr,te,C,k,h):
 p=weights(C,k,h);assert np.all(p>=0) and abs(p.sum(1)-1).max()<1e-12
 prefix=k.startswith('prefix') or k=='root_history';don=tr['V']*.75 if prefix else tr['X'];offset=te['A'] if prefix else np.zeros_like(te['X'])
 # Conditional support constraints and lattice identity; no independent component resampling.
 assert np.min(don[:,0])>=0 and np.max(don[:,0])<= (.75 if prefix else 1.)+1e-14
 assert abs(don[:,0]*128-np.rint(don[:,0]*128)).max()<1e-10
 mu=p@don+offset
 h2=np.einsum('ni,nj->nij',don,don).reshape(len(don),4);mu0=p@don;cov=(p@h2).reshape(-1,2,2)-np.einsum('ni,nj->nij',mu0,mu0)
 assert np.linalg.eigvalsh(cov).min()>-1e-10
 s=don@GRID.T;ex=np.exp(s);mgf=p@ex;W=np.log(mgf)+offset@GRID.T;ess=mgf**2/((p*p)@(ex*ex))
 # Marginal CRPS in original normalized ES coordinates, plus a dependence-sensitive variogram score.
 cr=[]
 for j in range(2):
  z=don[:,j]*SCALE[j];y=(te['X'][:,j]-offset[:,j])*SCALE[j];dd=abs(z[:,None]-z[None,:]);dy=abs(y[:,None]-z[None,:]);cr.append(np.einsum('ij,ij->i',p,dy)-.5*np.einsum('ij,ij->i',p@dd,p))
 # |z1-z2|^0.5 after mapping both coordinates to [0,1], offset-specific.
 gap=(don[:,0]-don[:,1]/2)[None,:]+(offset[:,0]-offset[:,1]/2-.5)[:,None]
 obs=abs(te['X'][:,0]-(te['X'][:,1]+1)/2)**.5;vg=(p*abs(gap)**.5).sum(1);variogram=(vg-obs)**2
 # Exact marginal coverage with discrete predictive quantiles; ties retained.
 cov90=[]
 for j in range(2):
  o=np.argsort(don[:,j]);cum=np.cumsum(p[:,o],axis=1);lo=don[o[np.argmax(cum>=.05,axis=1)],j]+offset[:,j];hi=don[o[np.argmax(cum>=.95,axis=1)],j]+offset[:,j]
  cov90.append((te['X'][:,j]>=lo-1e-12)&(te['X'][:,j]<=hi+1e-12))
 return dict(energy=es(C,p,k),crps=np.stack(cr,1),variogram=variogram,coverage=np.stack(cov90,1).astype(float),W=W,mu=mu,cov=cov,ess=1/(p*p).sum(1),tilt_ess=ess,nearest_root=np.sqrt(C['dr'].min(1)),nearest_history=np.sqrt(C['dh'].min(1)))
def evaluate():
 model=read(S/'MODEL82.json');assert model['status']=='TRAIN_ONLY_FROZEN' and model['input_sha']==sha(S/'POPULATION82.npz');msha=sha(S/'MODEL82.json')
 d=arrays();M=d['meta'];allres=[];predrows=[];mix=[]
 for g in range(3):
  tr=subset(d,(M[:,0]==0)&(M[:,1]==g));fm=model['families'][g];assert fm['family']==FAMS[g]
  for group in (0,1,2):mix.append(dict(family=FAMS[g],group=group,source=mixture(subset(d,(M[:,0]==group)&(M[:,1]==g)))))
  for group in (1,2):
   te=subset(d,(M[:,0]==group)&(M[:,1]==g));C=setup(tr,te);c=te['meta'][:,2];keys=np.unique(c)
   assert np.allclose(C['rmu'],fm['root_mean']) and np.allclose(C['rs'],fm['root_sd']) and np.allclose(C['hmu'],fm['history_mean']) and np.allclose(C['hs'],fm['history_sd'])
   truth=np.log(rootmeans(np.exp(te['X']@GRID.T),c));# finite empirical reference
   # independent seed split within each root, as noisy empirical-W benchmark, not a lower bound.
   split=[]
   for k in keys:
    x=te['X'][c==k];r=te['meta'][c==k,4];p0=np.exp(x[r%2==0]@GRID.T).mean(0);p1=np.exp(x[r%2==1]@GRID.T).mean(0);split.append(np.log(p0)-np.log(p1))
   split=np.asarray(split);res=[];pred={}
   for k in MODELS:
    h=fm['bandwidth'].get(k);v=predict(tr,te,C,k,h);pred[k]=v
    rw=np.log(rootmeans(np.exp(v['W']),c));delta=rw-truth;pop=np.log(np.exp(rw).mean(0));poptrue=np.log(np.exp(truth).mean(0));er=rootmeans(v['energy'],c);reg=delta[:,:25];stress=delta[:,25:]
    metrics=dict(model=k,bandwidth=h,energy_ci=ci(er),crps_ci=ci(rootmeans(v['crps'],c)),variogram_ci=ci(rootmeans(v['variogram'],c)),coverage90=rootmeans(v['coverage'],c).mean(0).tolist(),W_RMSE=float(np.sqrt(np.mean(reg*reg))),W_stress_RMSE=float(np.sqrt(np.mean(stress*stress))),W_population_RMSE=float(np.sqrt(np.mean((pop[:25]-poptrue[:25])**2))),W_population_stress_RMSE=float(np.sqrt(np.mean((pop[25:]-poptrue[25:])**2))),min_donor_ESS=float(v['ess'].min()),median_donor_ESS=float(np.median(v['ess'])),min_tilt_ESS_regular=float(v['tilt_ess'][:,:25].min()),min_tilt_ESS_stress=float(v['tilt_ess'][:,25:].min()),root_W=rw.tolist(),truth_W=truth.tolist(),root_energy=er.tolist(),mean_prediction_RMSE=np.sqrt(np.mean((v['mu']-te['X'])**2,0)).tolist(),mean_predicted_covariance=v['cov'].mean(0).tolist())
    res.append(metrics);predrows.append(dict(family=g,group=group,model=k,meta=te['meta'],**v))
   comparisons=[]
   for a,b in [('root','ply'),('root','pool'),('prefix_pool','ply'),('prefix_kernel','prefix_pool'),('root_history','prefix_pool'),('root_history','root'),('root_history','prefix_kernel')]:
    va,vb=pred[a],pred[b];# positive energy gain favors a
    comparisons.append(dict(candidate=a,baseline=b,energy_gain_ci=ci(rootmeans(vb['energy']-va['energy'],c)),variogram_gain_ci=ci(rootmeans(vb['variogram']-va['variogram'],c))))
   allres.append(dict(family=FAMS[g],group=group,roots=len(keys),paths=len(c),models=res,comparisons=comparisons,split_half_W_RMSE=float(np.sqrt(np.mean(split[:,:25]**2))),split_half_W_stress_RMSE=float(np.sqrt(np.mean(split[:,25:]**2))),nearest_root_distance_quantiles=np.quantile(pred['root']['nearest_root'],[0,.5,.9,1]).tolist(),nearest_history_distance_quantiles=np.quantile(pred['root_history']['nearest_history'],[0,.5,.9,1]).tolist()))
   print('TEST',FAMS[g],group,[(r['model'],round(r['energy_ci'][0],6),round(r['W_RMSE'],6)) for r in res],flush=True)
 # Output typed arrays separately (order preserved and explicit).
 fields=['energy','crps','variogram','coverage','W','mu','cov','ess','tilt_ess','nearest_root','nearest_history'];np.savez_compressed(S/'PREDICTIONS82.npz',meta=np.concatenate([np.column_stack([v['meta'],np.full(len(v['meta']),MODELS.index(v['model']))]) for v in predrows]),**{k:np.concatenate([v[k] for v in predrows]) for k in fields})
 assert sha(S/'MODEL82.json')==msha
 dump(S/'TRANSFER82.json',dict(status='EVALUATED_FROZEN_MODELS',model_sha=msha,array_sha=sha(S/'POPULATION82.npz'),prediction_sha=sha(S/'PREDICTIONS82.npz'),results=allres,mixtures=mix,scope='Root-balanced finite empirical distribution, test root W from finite seeds; prefix forecast W integrates actual prefixes and is not root-only prediction. Pointwise t intervals, no test refit.'))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='fit':fit()
 elif sys.argv[1]=='evaluate':evaluate()
 else:raise ValueError(sys.argv)
