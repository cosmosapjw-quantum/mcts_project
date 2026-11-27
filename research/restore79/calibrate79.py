"""R78 train, R79 independent native likelihood calibration and endpoint occupancy projection."""
from base61 import *
import gzip
from scipy.special import logsumexp
from scipy.optimize import brentq
from scipy.stats import t as student
from audit_native74 import one as audit9
from audit_rapfi74 import audit_search
O=B/'mqft_restore78_20261127';FAMS=['g0','g3','Rapfi'];TAGS=['schedule0','schedule3','rapfi'];CUTS=[4,8,16,32,64,128];BETA=10/3
def raw(arg):
 cohort,tag,j,receipt=arg;root=O if cohort==0 else S;f=root/'data'/tag/(j['id']+'.json.gz')
 if receipt:assert sha(f)==receipt['sha256']
 with gzip.open(f,'rt') as z:d=json.load(z)
 assert d['input']==j;tr=d['trace'];g=TAGS.index(tag);L=15 if g==2 else 9
 if cohort:a=audit_search(tr,j['moves'],'clean',j['seed'],128,j['sources'][0],0.) if g==2 else audit9((tag,j))
 else:a=receipt
 es=tr['events'];cuts={c['k']:c for c in tr['cuts']};legal=np.asarray(es[0]['root_selection']['law']['actions']);K=len(legal)
 assert K==L*L-len(j['moves'])
 N=np.zeros(L*L);W=np.zeros(L*L);pr=np.array(cuts[4]['prior'],float)[legal];pr/=pr.sum()
 scores=[];qs=[];probs=[];obs=[];occ=[];last=[];maxerr=0.
 for start in range(0,128,4):
  Q=np.divide(W,N,out=np.zeros_like(W),where=N>0)
  p=None
  for e in es[start:start+4]:
   law=e['root_selection']['law'];assert np.array_equal(law['actions'],legal) and law['beta_p']==BETA and law['logit_h']==0.
   if p is None:p=np.array(law['p']);sc=np.array(law['scores'])
   else:assert np.array_equal(p,law['p']) and np.array_equal(sc,law['scores'])
   obs.append(law['picked'])
  scores.append(sc);qs.append(Q[legal]);probs.append(p)
  ll=BETA*sc-logsumexp(BETA*sc);maxerr=max(maxerr,float(abs(np.exp(ll)-p).max()))
  for e in es[start:start+4]:
   a0=e['actions'][0];N[a0]+=1;W[a0]+=e['root_value']
  k=start+4
  if k in cuts:
   assert np.array_equal(N,cuts[k]['n']) and abs(W-cuts[k]['w']).max()<1e-9
   occ.append(N[legal]/k);last.append(np.divide(W,N,out=np.zeros_like(W),where=N>0)[legal])
 # Pad vectors, all illegal entries masked, never introduce illegal mass.
 def pad(x):
  a=np.zeros((len(x),L*L));a[:,legal]=x;return a
 ix=np.array(obs).reshape(32,4);counts=np.stack([np.bincount(a0,minlength=K) for a0 in ix])/4
 mask=np.zeros(L*L,bool);mask[legal]=True
 r=dict(cohort=cohort,family=g,core=j['core'],rep=j['rep'],seed=j['seed'],path=str(f.relative_to(B)),sha256=sha(f),bytes=f.stat().st_size,max_score=a['max_score'],max_probability=a['max_probability'],max_log=a['max_log'],conditional_recheck=maxerr)
 return r,mask,pad(scores),pad(qs),pad(probs),pad(counts),pad(occ),pad(last),np.where(mask,np.log(np.maximum(np.bincount(legal,weights=pr,minlength=L*L),1e-300)),0)
def fit(x,y,m):
 # x,y rows; y probability empirical or law; m bool; bounds retained.
 def v(b):
  l=np.where(m,b*x,-np.inf);z=logsumexp(l,axis=-1);p=np.exp(l-z[...,None]);grad=np.sum((p-y)*x);return float(grad),float(np.sum(p*(x-np.sum(p*x,axis=-1)[...,None])**2))
 if np.max(np.where(m,x,-np.inf),axis=-1).max()==np.min(np.where(m,x,np.inf),axis=-1).min():return None,'flat',0.
 lo,hi=v(-32)[0],v(32)[0]
 if abs(lo-hi)<1e-12:return None,'flat',0.
 if lo>=0:b=-32.;status='lower_bound'
 elif hi<=0:b=32.;status='upper_bound'
 else:b=float(brentq(lambda b:v(b)[0],-32,32,xtol=1e-11));status='interior'
 return b,status,v(b)[1]
def loss(x,y,m,b):
 l=np.where(m,b*x,-np.inf);l=l-logsumexp(l,axis=-1)[...,None];l=np.where(m,l,0.);return -np.sum(y*l,axis=-1)
def ci(a):
 a=np.asarray(a);m=a.mean(0);h=student.ppf(.975,len(a)-1)*a.std(0,ddof=1)/np.sqrt(len(a));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def main():
 R=read(S/'REGISTRATION.json');R0=read(O/'REGISTRATION.json');old={r['path']:r for r in read(O/'AUDIT78.json')['raw_receipts'] if r['orientation']==0}
 jobs=[]
 for cohort,root,reg in [(0,O,R0),(1,S,R)]:
  for tag in TAGS:
   p=root/'plans'/f'{tag}.json';assert sha(p)==reg['plans'][tag]['sha256']
   for j in read(p):
    if cohort==0 and '_o0_' not in j['id']:continue
    f=root/'data'/tag/(j['id']+'.json.gz');r=old[str(f.relative_to(B))] if cohort==0 else None
    jobs.append((cohort,tag,j,r))
 assert len(jobs)==960
 out=[]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for q in pool.map(raw,jobs,chunksize=8):out.append(q)
 assert len(out)==960
 rec=[a[0] for a in out];summ=[];fitrows=[]
 for g in range(3):
  aa=[q for q in out if q[0]['family']==g];r=[q[0] for q in aa];M=np.array([[x['cohort'],x['core'],x['rep']] for x in r]);mask,sc,Q,p,y,occ,qcut,prior=map(np.stack,zip(*(x[1:] for x in aa)))
  tr=M[:,0]==0;te=~tr;assert tr.sum()==64 and te.sum()==256
  # actual laws in pre-wave histories
  m=mask[:,None,:];assert np.max(abs(sc[:,0]))==0
  ref=loss(sc,y,m,BETA);ent=loss(sc,p,m,BETA)
  models=[]
  for name,x in [('score',sc),('Q',Q),('log_prior',np.broadcast_to(prior[:,None,:],sc.shape)),('uniform',np.zeros_like(sc)),('known_score',sc)]:
   b,status,cur=(0.,'fixed',0.) if name=='uniform' else (BETA,'known',0.) if name=='known_score' else fit(x[tr,1:],y[tr,1:],m[tr])
   assert b is not None
   l=loss(x[te],y[te],m[te],b);kl=loss(x[te],p[te],m[te],b)-ent[te];assert kl.min()>-1e-10
   diff=l-ref[te];cores=M[te,1]
   nll=[];regret=[];kls=[]
   for k in CUTS:
    nll.append(ci([l[cores==c,:k//4].mean() for c in range(32)]));regret.append(ci([diff[cores==c,:k//4].mean() for c in range(32)]));kls.append(ci([kl[cores==c,:k//4].mean() for c in range(32)]))
   models.append(dict(model=name,beta=b,status=status,train_curvature=cur,nll=nll,excess_nll=regret,conditional_KL=kls))
  # deterministic conditional scale identification from log p, separately count projection.
  sb=[];sk=[];occfits=[]
  for i in np.flatnonzero(te):
   K=mask[i].sum();x=sc[i,:,mask[i]].T;v=np.log(p[i,:,mask[i]].T);xc=x-x.mean(-1,keepdims=True);vc=v-v.mean(-1,keepdims=True);ss=(xc*xc).sum(-1);ok=ss>1e-24
   assert not ok[0]
   beta=(xc[ok]*vc[ok]).sum(-1)/ss[ok];sb.extend(beta.tolist());sk.extend(np.flatnonzero(ok).tolist())
   for k,xx,yy in zip(CUTS,qcut[i],occ[i]):
    b,status,cur=fit(xx[None,:],yy[None,:],mask[i,None,:]);occfits.append(dict(core=int(M[i,1]),rep=int(M[i,2]),B=k,beta=b,status=status,curvature=cur,occupancy_KL=None if b is None else float(loss(xx[None,:],yy[None,:],mask[i,None,:],b)[0]-entropy(yy))))
  sb=np.array(sb);assert abs(sb-BETA).max()<1e-8
  of=[]
  for k in CUTS:
   z=[a for a in occfits if a['B']==k];v=[a['beta'] for a in z if a['status']=='interior'];ct=collections.Counter(a['status'] for a in z)
   of.append(dict(B=k,n=len(z),statuses=dict(ct),interior_quantiles=np.quantile(v,[.05,.5,.95]).tolist() if v else None,positive_interior=sum(a>0 for a in v),mean_occupancy_KL=float(np.mean([a['occupancy_KL'] for a in z if a['occupancy_KL'] is not None]))))
  summ.append(dict(family=FAMS[g],train_paths=64,test_paths=256,models=models,score_log_slope_count=len(sb),score_log_slope_max_error=float(abs(sb-BETA).max()),unidentifiable_waves=256*32-len(sb),occupancy=of,occupancy_fits=occfits))
  np.savez_compressed(S/f'CALIB_{FAMS[g]}.npz',meta=M,mask=mask,score=sc,Q=Q,p=p,y=y,occ=occ,Qcut=qcut,logprior=prior)
  print(FAMS[g],[(a['model'],a['beta'],a['conditional_KL'][-1][0],a['excess_nll'][-1][0]) for a in models],of[-1],flush=True)
 audit=dict(paths=768,train_paths=192,backups=98304,train_backups=24576,raw_bytes=sum(a['bytes'] for a in rec if a['cohort']),max_score=max(a['max_score'] for a in rec),max_probability=max(a['max_probability'] for a in rec),max_log=max(a['max_log'] for a in rec),raw_receipts=rec,arrays={f'CALIB_{f}.npz':sha(S/f'CALIB_{f}.npz') for f in FAMS})
 dump(S/'AUDIT79.json',audit);dump(S/'CALIBRATION79.json',dict(scope='R78 identity orientation train, new R79 test; conditional score likelihood versus Q/prior/occupancy projection, no physical T, pointwise geometry block intervals',cuts=CUTS,known_beta=BETA,results=summ))
 print('AUDIT', {k:v for k,v in audit.items() if k not in ('raw_receipts','arrays')},flush=True)
if __name__=='__main__':main()
