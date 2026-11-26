"""Disjoint fragment predictive decision information on actual histories.
Train on R71 48 roots; old 16 holdout and 16 new R73 roots tested separately.
"""
from base61 import *
import gzip
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import t as student
from audit65 import audit_one
from audit68 import audit_search
O=B/'mqft_restore71_20261126'
def extract(arg):
 stage,tag,j,h=arg;f=stage/'data'/tag/(j['id']+'.json.gz')
 digest=sha(f);assert h is None or digest==h
 with gzip.open(f,'rt') as ff:q=json.load(ff)
 assert q['input']==j;tr=q['trace'];ev=tr['events'];assert len(ev)==128
 a,b=j['sources'];size=225 if tag=='rapfi' else 81;L=int(np.sqrt(size));g=4 if tag=='rapfi' else 0 if tag=='phase0' else 1
 arm=0 if tag=='rapfi' else ['persist_plus','persist_minus','persist_dephased'].index(j['arm'])
 n=np.asarray(tr['cuts'][-1]['n']);assert n.sum()==128 and len(n)==size
 actions=np.array([e['actions'][0] for e in ev]);marks=np.array([e['root_value'] for e in ev])
 assert np.array_equal(np.bincount(actions,minlength=size),n)
 assert np.all(abs(marks)<=1+1e-7) and all(e['root_value']==(-1)**len(e['actions'])*e['leaf_value'] for e in ev)
 D=int(n.argmax());tie=int(np.sum(n==n.max()));dc=0 if D==a else 1 if D==b else 2
 actual=int(tr['action']) if tag=='rapfi' else -1
 prior=np.array(tr['cuts'][0]['prior']);assert abs(prior.sum()-1)<2e-6
 vi=float(tr['evaluations'][0]['delivered' if tag=='rapfi' else 'value'])
 p=prior[prior>0];xy=np.array([(x//L,x%L) for x in j['moves']]);sxy=np.array([(x//L,x%L) for x in (a,b)])
 dist=np.linalg.norm(sxy[:,None]-xy[None,:],axis=-1)
 root=[vi,len(j['moves'])/size,prior[a],prior[b],float(-p@np.log(p)),float(np.linalg.norm(sxy[0]-sxy[1])/L),*list(dist.min(1)/L),*list(dist.mean(1)/L)]
 frag=[]
 for k in range(4):
  acts=actions[32*k:32*(k+1)];z=marks[32*k:32*(k+1)]
  lab=np.where(acts==a,0,np.where(acts==b,1,2))
  x=np.bincount(lab,minlength=3)/32;w=np.bincount(lab,weights=z,minlength=3)/32
  cnt=np.bincount(acts,minlength=size);occ=cnt[cnt>0]/32
  frag.append([*x[:2],*w,z.mean(),float(-occ@np.log(occ)),float(cnt.max()/32),float(len(occ)/32)])
 audit=None
 if stage==S:
  if tag=='rapfi':
   r=audit_search(tr,j['moves'],'clean',j['seed'],128);audit={k:r[k] for k in ('max_probability','max_score','max_log')}
  else:
   r=audit_one((tag,j));audit={k:r[k] for k in ('max_probability','max_state','max_score','max_log')}
  audit.update(path=str(f.relative_to(B)),sha256=digest,bytes=f.stat().st_size,seed=j['seed'])
 return [int(stage==S),g,arm,j['core'],j['rep'],dc,D,tie,actual],root,frag,audit
def fit(x,y):
 mu=x.mean(0);sd=np.maximum(x.std(0),1e-6);u=np.c_[np.ones(len(x)),(x-mu)/sd];d=u.shape[1]
 def loss(w):
  w=w.reshape(d,3);z=u@w;lp=z-logsumexp(z,axis=1)[:,None]
  p=np.exp(lp);p[np.arange(len(y)),y]-=1;reg=.02
  grad=u.T@p/len(y);grad[1:]+=reg*w[1:];grad[0]+=.001*w[0]
  return -lp[np.arange(len(y)),y].mean()+reg*np.sum(w[1:]**2)/2+.001*np.sum(w[0]**2)/2,grad.ravel()
 q=minimize(loss,np.zeros(d*3),jac=True,method='L-BFGS-B',options={'gtol':1e-9,'maxiter':500})
 assert q.success or np.max(abs(q.jac))<1e-5,q.message
 return mu,sd,q.x.reshape(d,3),float(np.max(abs(q.jac)))
def pred(m,x,y):
 mu,sd,w,_=m;z=np.c_[np.ones(len(x)),(x-mu)/sd]@w
 return logsumexp(z,axis=1)-z[np.arange(len(y)),y]
def block_ci(x):
 x=np.asarray(x);m=x.mean();h=student.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return [float(m),float(m-h),float(m+h)]
def main():
 old=read(O/'AUDIT71.json');assert sha(O/'JOINT71.npz')==old['array_sha256'];hs={r['path']:r['sha256'] for r in old['raw_receipts']}
 args=[]
 for stage in (O,S):
  reg=read(stage/'REGISTRATION.json')
  for tag in ('phase0','phase3','rapfi'):
   p=stage/'plans'/f'{tag}.json';js=read(p);assert sha(p)==reg['plans'][tag]['sha256']
   for j in js:args.append((stage,tag,j,hs[str((stage/'data'/tag/(j['id']+'.json.gz')).relative_to(B))] if stage==O else None))
 M=[];X=[];F=[];aud=[]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,(m,x,f,a) in enumerate(pool.map(extract,args,chunksize=8)):
   M.append(m);X.append(x);F.append(f)
   if a:aud.append(a)
   if (i+1)%1024==0:print('DECISION EXTRACT',i+1,flush=True)
 M=np.array(M);X=np.array(X);F=np.array(F);assert len(M)==10112 and len(aud)==896
 np.savez_compressed(S/'DECISION73.npz',meta=M,root=X,frag=F)
 out=[];rng=np.random.default_rng(730311)
 modes=[(),(0,),(1,),(2,),(3,),(0,1),(2,3),(0,1,2,3)]
 def design(ix,mode,fp=None):
  return np.c_[X[ix],(F[ix] if fp is None else fp)[:,list(mode),:].reshape(len(ix),-1)]
 for g in (0,1,4):
  # pooled sign arms, no phase label in predictors, separate sign/dephased reporting.
  train=np.flatnonzero((M[:,0]==0)&(M[:,1]==g)&(M[:,3]%16>=4));assert len(np.unique(M[train,3]))==48
  models=[fit(design(train,m),M[train,5]) for m in modes]
  for fresh in (0,1):
   for arm in ([0] if g==4 else [0,1] if fresh==0 else [0,1,2]):
    ix=np.flatnonzero((M[:,0]==fresh)&(M[:,1]==g)&(M[:,2]==arm)&((M[:,3]%16<4) if fresh==0 else True))
    assert len(np.unique(M[ix,3]))==16
    y=M[ix,5];co=M[ix,3];roots=np.unique(co);rootloss=[];gain=[];null=[]
    base=pred(models[0],design(ix,()),y)
    losses=np.array([pred(m,design(ix,mode),y) for m,mode in zip(models,modes)])
    for k,mode in enumerate(modes):
     rootscore=np.array([losses[k,co==c].mean() for c in roots])
     rootgain=np.array([(base[co==c]-losses[k,co==c]).mean() for c in roots])
     nullgain=[]
     for z in range(32):
      fp=F[ix].copy()
      for c in roots:
       jj=np.flatnonzero(co==c);fp[jj]=fp[rng.permutation(jj)].copy()
      ll=pred(models[k],design(ix,mode,fp),y);nullgain.append(float((base-ll).mean()))
     rootloss.append(block_ci(rootscore));gain.append(block_ci(rootgain));null.append([float(np.mean(nullgain)),float(np.quantile(nullgain,.025)),float(np.quantile(nullgain,.975))])
    empH=float(np.mean([entropy(np.bincount(y[co==c],minlength=3)/(co==c).sum()) for c in roots]))
    out.append(dict(generation=g,dataset='fresh_R73' if fresh else 'R71_holdout',arm=arm,paths=len(ix),root_blocks=16,decision_counts=np.bincount(y,minlength=3).tolist(),ties=int(np.sum(M[ix,7]>1)),native_Rapfi_decision_mismatch=int(np.sum((M[ix,8]>=0)&(M[ix,8]!=M[ix,6]))),empirical_root_entropy=empH,log_loss=rootloss,gain=gain,shuffle_gain=null,fit_gradient=max(m[-1] for m in models)))
 for r in out:print(r['generation'],r['dataset'],r['arm'],r['decision_counts'],'gain',[round(q[0],4) for q in r['gain']],flush=True)
 audit=dict(paths=896,backups=114688,reused_paths=9216,raw_bytes=sum(q['bytes'] for q in aud),max_probability=max(q['max_probability'] for q in aud),max_score=max(q['max_score'] for q in aud),max_log=max(q['max_log'] for q in aud),max_state=max(q.get('max_state',0) for q in aud),raw_receipts=aud,array_sha256=sha(S/'DECISION73.npz'))
 dump(S/'AUDIT73.json',audit);dump(S/'DECISION_RESULTS73.json',dict(scope='D is deterministic most-visited lowest-index tie recommendation, not pointer Z. Cross-entropy gain of fixed model, not exact MI or quantum Darwinism. R71 training 48 roots, R71 16 old heldout, R73 new 16. Within-root seed shuffle, pointwise t intervals; no simultaneous inference.',modes=[list(m) for m in modes],results=out,source_R71_audit_sha=sha(O/'AUDIT71.json'),array_sha256=audit['array_sha256']))
 print('AUDIT', {k:v for k,v in audit.items() if k!='raw_receipts'},flush=True)
if __name__=='__main__':main()
