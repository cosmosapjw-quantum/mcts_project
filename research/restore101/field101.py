from base61 import *
from scipy.special import logsumexp
from scipy.stats import t as student
import time,sys,platform
R93=B/'mqft_restore93_20261128';R74=B/'mqft_restore74_20261126';R94=B/'mqft_restore94_20261128';R95=B/'mqft_restore95_20261128';R81=B/'mqft_restore81_20261127'
def ck(e,k,a,b,tol=1e-9):
 d=float(np.max(abs(np.asarray(a)-np.asarray(b))));e[k]=max(e.get(k,0.),d);assert d<tol,(k,d)
def ci(z):
 z=np.asarray(z);m=z.mean(0);w=student.ppf(.975,len(z)-1)*z.std(0,ddof=1)/np.sqrt(len(z));return np.stack([m,m-w,m+w])
def pin(stage,f,src):
 p=next(B.glob(f'mqft_restore{stage}_20*'));m=read(p/f'RUN_MANIFEST{stage}.json');k=str((p/f).relative_to(B));h=m['files'][f];assert sha(p/f)==h;src[k]=h;return p/f
def pin_input(p,m,src):
 k=str(p.relative_to(B));h=m['inputs'][k];assert sha(p)==h;src[k]=h;return p
def rehash(x):k,h=x;assert sha(B/k)==h,k;return k,h
def soft(v):
 z=np.exp(v-logsumexp(v));return z
def W(X,J,lr):
 a=X@J+lr;z=logsumexp(a)-logsumexp(lr);w=soft(a);mu=w@X;d=X-mu;cov=(d.T*w)@d;return float(z),w,mu,cov
def main():
 t=time.monotonic();src={};err={};r93=read(R93/'RUN_MANIFEST93.json');a=read(pin_input(R74/'AUDIT74.json',r93,src));raw={r['path']:r['sha256'] for r in a['raw_receipts']};assert len(raw)==2304 and all(r93['prior_raw_receipts'][k]==h for k,h in raw.items())
 with cf.ThreadPoolExecutor(max_workers=6) as pool:assert dict(pool.map(rehash,raw.items()))==raw
 for stage,f in [(93,'RESPONSE93.json'),(98,'SUMMARY98.json'),(99,'ACTION99.json'),(100,'SUMMARY100.json')]:pin(stage,f,src)
 p=pin_input(R74/'PULSE74.npz',r93,src)
 with np.load(p) as z:
  meta=z['meta'];v=z['features'];assert v.shape==(2304,32,10)
 X=np.empty((3,16,16,3,32,10));seen=set()
 for m,y in zip(meta,v):
  k=tuple(m);assert k not in seen;seen.add(k);X[k]=y
 assert len(seen)==2304
 # R74 exact hit fraction, mark mean and base conditional probability are columns0,1,2.
 O=np.concatenate([X[...,:8,:2].mean(-2),X[...,8:16,:2].mean(-2)],-1);q=X[:,:,:,0,8,2];k=4*X[:,:,:,0,8,0];assert abs(k-np.rint(k)).max()<1e-10 and min(q.ravel())>0 and max(q.ravel())<1
 ck(err,'matched_pre',O[:,:,:,0,:2],O[:,:,:,1,:2]);ck(err,'matched_pre_minus',O[:,:,:,0,:2],O[:,:,:,2,:2])
 ck(err,'matched_pulse_q',q,X[:,:,:,1,8,2]);ck(err,'matched_pulse_q_minus',q,X[:,:,:,2,8,2])
 Js=[np.zeros(4)]+[s*np.eye(4)[j] for j in range(4) for s in [-1,1]]+[s*np.array(d) for d in ([1,1,-1,-1],[2,0,2,0],[0,2,0,2]) for s in [-1,1]]
 rows=[];rankcount=collections.Counter();nfd=0;hs=[-.35,0.,.35]
 for g in range(3):
  for c in range(16):
   x=O[g,c,:,0];qq=q[g,c];kk=k[g,c];score=kk-4*qq
   def lr(h):return h*kk-4*np.log1p(qq*np.expm1(h))
   v0=[]
   for h in hs:
    s=kk-4*qq*np.exp(h)/(1+qq*np.expm1(h));l=lr(h);basew=soft(l)
    for J in Js:
     w,ww,mu,C=W(x,J,l);G=float(J@mu-w);K=float(np.sum(ww*np.log(ww/basew)));ck(err,'dual_KL',G,K);assert np.linalg.eigvalsh(C).min()>-1e-10 and G>=-1e-10
     evals,U=np.linalg.eigh(C);rank=int((evals>1e-12).sum());rankcount[rank]+=1;P=U[:,evals>1e-12]@U[:,evals>1e-12].T;inv=np.linalg.pinv(C,rcond=1e-12/max(evals[-1],1e-12));ck(err,'Hessian_inverse_on_range',C@inv,P,1e-6)
     # Mixed derivative d_h mean at finite empirical normalization, independent centered difference.
     response=(ww[:,None]*(x-mu)*(s-ww@s)[:,None]).sum(0);eps=1e-5;mu1=W(x,J,lr(h+eps))[2];mu0=W(x,J,lr(h-eps))[2];ck(err,'mixed_response_fd',response,(mu1-mu0)/(2*eps),1e-6)
     # independent directional derivative of W and gradient gives Hessian.
     direction=np.array([.3,-.7,.2,.5]);z1=W(x,J+eps*direction,l);z0=W(x,J-eps*direction,l)
     ck(err,'gradient_fd',mu@direction,(z1[0]-z0[0])/(2*eps),1e-6);ck(err,'Hessian_fd',C@direction,(z1[2]-z0[2])/(2*eps),1e-6);nfd+=1
     # prediction of held-out empirical logMGF from disjoint training reps; no fit.
     tr,te=x[:8],x[8:];a0=l[:8];b0=l[8:];j1=J*np.array([1,1,0,0]);j2=J-j1
     train=W(tr,J,a0)[0];test=W(te,J,b0)[0];fac=W(tr,j1,a0)[0]+W(tr,j2,a0)[0]
     # comparison at J=h=0 to actual direct finite pulse contrast, conditional score estimate
     v0.append(dict(h=h,J=J.tolist(),W=w,dual=G,rank=rank,mean=mu.tolist(),Hessian=C.tolist(),response=response.tolist(),train=train,test=test,factorized_train=fac))
   direct=(O[g,c,:,1]-O[g,c,:,2]).mean(0)/.7
   reweight=(soft(lr(.35))@x-soft(lr(-.35))@x)/.7
   r0=(x-x.mean(0)).T@(score-score.mean())/16
   zero=next(a for a in v0 if a['h']==0 and not any(a['J']));ck(err,'score_cov_direct',r0,zero['response'])
   rows.append(dict(family=g,core=c,values=v0,direct=direct.tolist(),reweighted=reweight.tolist(),score=r0.tolist(),raw_Lplus=float(np.exp(lr(.35)).mean()),raw_Lminus=float(np.exp(lr(-.35)).mean())))
 assert len(rows)==48 and nfd==2160
 groups=[]
 for g in range(3):
  rr=[r for r in rows if r['family']==g];z=np.array([[[v['train']-v['test'],v['factorized_train']-v['test']] for v in r['values'] if any(v['J'])] for r in rr]);mse=(z*z).mean(1)
  groups.append(dict(family=g,rmse=np.sqrt(mse.mean(0)).tolist(),joint_minus_factorized_MSE_ci=ci(mse[:,0]-mse[:,1]).tolist(),direct_ci=ci([r['direct'] for r in rr]).tolist(),reweighted_ci=ci([r['reweighted'] for r in rr]).tolist(),score_ci=ci([r['score'] for r in rr]).tolist()))
 dump(S/'FIELD101.json',dict(status='PASS_SCOPED',paths=len(raw),root_contexts=48,evaluations=nfd,rank_counts={str(k):v for k,v in rankcount.items()},groups=groups,rows=rows,checks=err,raw_receipts=raw,source_hashes=src,elapsed_s=time.monotonic()-t,scope='Finite empirical normalized tilted path law; disjoint rep halves, not independent new campaign or continuum QFT.'))
 print('FIELD',nfd,dict(rankcount),groups,err,flush=True)
if __name__=='__main__':main()
