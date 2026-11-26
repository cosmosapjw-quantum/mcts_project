"""R70 adaptive marked path CGF and lattice replica expansion.
Distinct empirical path CGF, exact conditional count kernel, and learned closure."""
from base61 import *
import gzip,sys,csv
from scipy.special import logsumexp,gammaln
from scipy.optimize import root,brentq
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=B/'mqft_restore69_20261126'
G=np.array([(i,j,4-i-j) for i in range(5) for j in range(5-i)],int)
LCO=gammaln(5)-gammaln(G+1).sum(1)
def mass(p):
 return np.exp(np.log(np.maximum(p,1e-300))@G.T+LCO)
def one(arg):
 src,tag,j=arg; path=src/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(path,'rt') as f:r=json.load(f)
 assert r['input']==j;es=r['trace']['events'];assert len(es)==128
 a,b=j['sources'];Y=[];P=[];F=[];Z=[]
 for t in range(32):
  ev=es[4*t:4*t+4];law=ev[0]['root_selection']['law'];acts=law['actions'];p=np.asarray(law['p']);ia,ib=acts.index(a),acts.index(b)
  p=np.array([p[ia],p[ib],1-p[ia]-p[ib]])
  c=np.zeros(3,int);marks=np.zeros(3)
  for e in ev:
   ac=e['actions'][0];i=0 if ac==a else 1 if ac==b else 2
   c[i]+=1;z=e['root_value'];assert abs(z)<=1+1e-7 and abs(z-(-1)**len(e['actions'])*e['leaf_value'])<1e-7;marks[i]+=z
  if tag=='rapfi':
   for e in ev:
    l=e['root_selection']['law'];q=np.asarray(l['p']);i=l['actions'].index(a);k=l['actions'].index(b);assert np.max(abs(np.array([q[i],q[k],1-q[i]-q[k]])-p))<1e-10
   m=mass(p)
  else:
   h=law['phase65'];v=h['prepared'];m0=h['pair_mass'];u=(1+v[1]*np.sin(.7)+v[2]*np.cos(.7))/2
   p0=np.array([m0*h['e0'],m0*(1-h['e0']),1-m0]);p1=np.array([m0*h['e1'],m0*(1-h['e1']),1-m0])
   assert np.max(abs(u*p0+(1-u)*p1-p))<2e-10
   m=u*mass(p0)+(1-u)*mass(p1)
  assert abs(m.sum()-1)<1e-10 and np.all(m>0);Y.append(c);P.append(p);F.append(m);Z.append(marks)
 family=4 if tag=='rapfi' else 2*(j['generation']==3)+int(j['arm']=='persist_minus')
 return [family,j['core'],j['rep']],Y,P,F,Z,{'path':str(path.relative_to(B)),'sha256':sha(path),'bytes':path.stat().st_size,'seed':j['seed']}
def extract(src):
 jobs=[(src,t,j) for t in ('phase0','phase3','rapfi') for j in read(src/'plans'/f'{t}.json')]
 with cf.ProcessPoolExecutor(max_workers=8) as pool:r=list(pool.map(one,jobs,chunksize=4))
 return [np.asarray([x[i] for x in r]) for i in range(5)], [x[5] for x in r]
def cgf(x,j):
 s=x@j;w=logsumexp(s)-np.log(len(s));p=np.exp(s-logsumexp(s));mu=p@x;v=x-mu;h=(v.T*p)@v
 return w,mu,h,p
def stat(x):
 x=np.asarray(x);n=len(x);return dict(mean=float(x.mean()),ci95=(x.mean()+np.array([-1,1])*student.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)).tolist(),n_blocks=n)
def ridge(x,y,z):
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1
 A=np.column_stack([np.ones(len(x)),(x-mu)/sd]);T=np.column_stack([np.ones(len(z)),(z-mu)/sd])
 R=np.eye(A.shape[1])*.02;R[0,0]=0
 return T@np.linalg.solve(A.T@A/len(A)+R,A.T@y/len(A))
def feats(P,Y,Z):
 n=len(P);t=np.broadcast_to(np.arange(32)[None,:,None]/31,(n,32,1))
 prev=np.concatenate([np.zeros_like(Y[:,:1]),np.cumsum(Y,axis=1)[:,:-1]],axis=1)/(4*np.maximum(1,np.arange(32))[None,:,None])
 zm=Z.sum(-1)/4;lag=np.concatenate([np.zeros_like(zm[:,:1]),zm[:,:-1]],1)[...,None]
 cum=np.concatenate([np.zeros_like(zm[:,:1]),np.cumsum(zm,1)[:,:-1]],1)[...,None]/np.maximum(1,np.arange(32))[None,:,None]
 return [t,np.concatenate([t,P[:,:,:2]],-1),np.concatenate([t,P[:,:,:2],prev[:,:,:2],lag,cum],-1)],prev
def bins(prev):
 return np.digitize(prev[...,0],[.01,.03,.10])*4+np.digitize(prev[...,1],[.01,.03,.10])
def fit_hist(prev,pmf):
 s=bins(prev);M=np.zeros((32,16,15))
 for t in range(32):
  for b in range(16):
   q=pmf[s[:,t]==b,t];M[t,b]=(q.sum(0)+2*pmf[:,t].mean(0))/(len(q)+2)
 return M
def saddle(p,M,j0):
 x=np.arange(len(p));ix=p>0;x=x[ix];lp=np.log(p[ix]);span=int(np.gcd.reduce(np.diff(x)))
 if len(x)<2 or span!=1:return None
 def law(j):
  a=lp+j*x;w=logsumexp(a);q=np.exp(a-w);m=q@x;v=x-m;k2=q@(v**2);k3=q@(v**3);k4=q@(v**4)-3*k2*k2
  return w,m,k2,k3,k4,q
 k=int(round(M*law(j0)[1]));target=k/M
 if not x[0]<target<x[-1]:return None
 j=brentq(lambda j:law(j)[1]-target,-20,20);w,m,v,c3,c4,q=law(j)
 dense=np.zeros(int(x[-1])+1);dense[x]=q
 # Exact tilted convolution, no un-tilting underflow. q_M(k) same as original mass times exp(j*k-M*w).
 qM=np.array([1.])
 for _ in range(M):qM=np.convolve(qM,dense)
 exact=qM[k];lead=1/np.sqrt(2*np.pi*M*v);cor=(c4/(8*v*v)-5*c3*c3/(24*v**3))/M
 if exact<=0:return None
 return dict(M=M,j=j,k=k,var=v,c3=c3,c4=c4,correction=cor,leading_rel=float(lead/exact-1),next_rel=float(lead*(1+cor)/exact-1),exact_tilted=float(exact),log_prob=float(np.log(exact)+M*w-j*k),normalization=float(qM.sum()),tail_mass_near_edge=float(q[0]+q[-1]))
def main():
 # SSOT data and weights have separate replay receipts; same geometries, independent seed schedules.
 a,oldrec=extract(O);b,newrec=extract(S)
 meta,Y,P,F,Z=b;om,oy,op,of,oz=a
 assert not ({x['seed'] for x in oldrec}&{x['seed'] for x in newrec})
 assert len(meta)==len(om)==288 and np.array_equal(meta,om)
 prior=np.load(B/'mqft_restore65_20261126/GENERATOR65.npz')
 oldmeta=prior['meta'];oldy=prior['counts'];oldprev=np.concatenate([np.zeros_like(oldy[:,:1]),np.cumsum(oldy,axis=1)[:,:-1]],axis=1)/(4*np.maximum(1,np.arange(32))[None,:,None])
 # R63 is a separate ordinary trained search closure, not phase-selective evidence.
 r63=np.load(B/'mqft_restore63_20261126/WAVES63.npz');assert r63['wave_y'].shape==(7680,32,3)
 idx=np.argmax(np.all(Y[:,:,None,:]==G,axis=-1),axis=-1);assert np.array_equal(G[idx],Y)
 freeze=np.broadcast_to(F[:,:1],F.shape);iid=mass(P)
 f0,prev=feats(P,Y,Z);f1,oprev=feats(op,oy,oz)
 hist=np.zeros_like(F);mark_preds=np.zeros((3,len(meta),32));results=[];curves=[];saddles=[]
 source=np.linspace(-4,4,41);labels=['g0 +','g0 -','g3 +','g3 -','Rapfi NNUE']
 for g in range(5):
  ids=np.flatnonzero(meta[:,0]==g)
  # Geometry cross-fitting: R69 disjoint cores for marks and Rapfi closure; R65 trained closure independent of new seeds.
  for fold in (0,1):
   test=ids[(meta[ids,1]%2)==fold];train=np.flatnonzero((om[:,0]==g)&(om[:,1]%2!=fold))
   if g<4:
    train65=(oldmeta[:,0]==(0 if g<2 else 3))&(oldmeta[:,3]==g%2)
    M=fit_hist(oldprev[train65],prior['pred'][train65])
   else:M=fit_hist(oprev[train],of[train])
   hist[test]=M[np.arange(32)[None,:],bins(prev[test])]
   for k,(xx,zz) in enumerate(zip(f1,f0)):
    mark_preds[k,test]=ridge(xx[train].reshape(-1,xx.shape[-1]),oz[train].sum(-1).ravel()/4,zz[test].reshape(-1,zz.shape[-1])).reshape(len(test),32)
  X=np.column_stack([Y[ids].sum(1)[:,:2],Z[ids].sum((1,2))])/128
  Xold=np.column_stack([oy[ids].sum(1)[:,:2],oz[ids].sum((1,2))])/128
  # Per-family empirical marked CGF: exact finite empirical measure, not exact population law.
  wa=[];wb=[];wf=[];wn=[];ess=[];hess=[];c3=[];c4=[]
  for j in source:
   v=np.array([j,-j,.5*j]);w,m,H,p=cgf(X,v);wo,*_=cgf(Xold,v)
   wa.append(w);wb.append(wo);ess.append(1/(p@p));hess.append(np.linalg.eigvalsh(H).tolist())
   dv=(X-m)@np.array([1,-1,.5]);c3.append(p@(dv**3));c4.append(p@(dv**4)-3*(p@(dv**2))**2)
   # Frozen and realized adaptive conditional normalizers, count-only matched source.
   L=np.exp(G[:,:2]@v[:2]/128)
   wf.append(float(np.mean(np.log(freeze[ids]@L).sum(1))))
   wn.append(float(np.mean(np.log(F[ids]@L).sum(1))))
  inversions=[]
  for j0 in [np.zeros(3),np.array([2,-2,1]),np.array([-2,2,-1]),np.array([50,-50,25])]:
   w,m,H,p=cgf(X,j0);ev=np.linalg.eigvalsh(H);fit=root(lambda j:cgf(X,j)[1]-m,np.zeros(3),jac=lambda j:cgf(X,j)[2])
   hstep=1e-4;Hfd=np.column_stack([(cgf(X,j0+np.eye(3)[i]*hstep)[1]-cgf(X,j0-np.eye(3)[i]*hstep)[1])/(2*hstep) for i in range(3)])
   assert np.max(abs(H-Hfd))<2e-8
   inversions.append(dict(J=j0.tolist(),mean=m.tolist(),gamma=float(j0@m-w),eig=ev.tolist(),det=float(np.linalg.det(H)),ess=float(1/(p@p)),inversion_success=bool(fit.success),inversion_source_error=float(np.linalg.norm(fit.x-j0)),inversion_mean_error=float(np.linalg.norm(fit.fun)),hessian_fd_error=float(np.max(abs(H-Hfd)))))
  # Sequential exact count tilt: product of ratios is a mean-one martingale, not an unconditional MGF.
  mart=[]
  for j in [-.5,-.2,.2,.5]:
   v=np.array([j,-j])/4;den=np.log(F[ids]@np.exp(G[:,:2]@v));obs=Y[ids,:,:2]@v
   L=np.exp((obs-den).sum(1))
   blocks=[L[meta[ids,1]==c].mean() for c in np.unique(meta[ids,1])]
   mart.append(dict(j=j,mean=stat(blocks),ess=float(L.sum()**2/(L@L))))
  total=Y[ids,:,0].sum(1);p=np.bincount(total,minlength=129)/len(total)
  for M in (8,16,32,64,128,256):
   for j in (-.1,0,.1):
    rr=saddle(p,M,j)
    if rr is not None:saddles.append(dict(family=g,**rr))
  ms=np.stack([freeze[ids],iid[ids],F[ids],hist[ids]])
  nll=-np.log(np.take_along_axis(ms,idx[ids][None,:,:,None],axis=-1)[...,0])
  cmean=ms@G;res=Y[ids][None]-cmean
  blocks=np.unique(meta[ids,1]);ns=[stat([nll[k,meta[ids,1]==c].mean() for c in blocks]) for k in range(4)]
  rm=[float(np.sqrt(np.mean(res[k]**2))) for k in range(4)]
  marks=(Z[ids].sum(-1)/4)
  err=(mark_preds[:,ids]-marks)**2;mr=[float(np.sqrt(e.mean())) for e in err]
  md=[stat([float((err[k,meta[ids,1]==c]-err[0,meta[ids,1]==c]).mean()) for c in blocks]) for k in range(3)]
  # Count source only: valid comparison of path empirical W and the average realized normalizers.
  v=np.array([2.,-2.]);q=X[:,:2]@v;Wcount=float(logsumexp(q)-np.log(len(q)));cond=float(np.mean(np.log(F[ids]@np.exp(G[:,:2]@v/128)).sum(1)))
  results.append(dict(family=g,label=labels[g],paths=len(ids),waves=len(ids)*32,nll=ns,count_rmse=rm,mark_rmse=mr,mark_mse_diff_vs_clock=md,mean_counts=Y[ids].mean((0,1)).tolist(),inversion=inversions,martingale=mart,count_path_W_at_2=Wcount,mean_conditional_log_normalizers_at_2=cond,marked_W_repeat_max_abs=float(np.max(abs(np.array(wa)-wb)))))
  curves.append(dict(family=g,J=source.tolist(),W=wa,W_prior=wb,mean_frozen_count_logZ=wf,mean_adaptive_count_logZ=wn,ess=ess,hessian_eig=hess,k3=c3,k4=c4))
 np.savez_compressed(S/'MARKED70.npz',meta=meta,counts=Y,state=P,conditional_pmf=F,marks=Z,history_pmf=hist,mark_predictions=mark_preds,grid=G,prior_meta=om,prior_counts=oy,prior_marks=oz)
 report=dict(scope='empirical path law; no physical hbar; R65 count-only reduced history and R69 crossfit marks; independent replica saddle of empirical marginal',families=results,curves=curves,saddles=saddles,fresh_paths=len(newrec),reused_paths=len(oldrec),fresh_backups=len(newrec)*128,R65_waves=int(oldy.shape[0]*oldy.shape[1]),R63_reference_waves=int(r63['wave_y'].shape[0]*32),raw_receipts=newrec,prior_raw_receipts=oldrec,source_hashes={str(p.relative_to(B)):sha(p) for p in [B/'mqft_restore65_20261126/GENERATOR65.npz',B/'mqft_restore63_20261126/WAVES63.npz',O/'REGISTRATION.json',S/'REGISTRATION.json']},array_sha=sha(S/'MARKED70.npz'))
 dump(S/'PHYSICS_RESULTS.json',report)
 print('RESULT',[(r['label'],[round(x['mean'],4) for x in r['nll']],r['mark_rmse'],r['count_path_W_at_2']-r['mean_conditional_log_normalizers_at_2']) for r in results],flush=True)
 print('SADDLE',len(saddles),flush=True)
 figman=[]
 def plot(name,fig,ids):
  fig.tight_layout();p=S/'plots'/name
  fig.savefig(str(p)+'.png',dpi=150);fig.savefig(str(p)+'.pdf');plt.close(fig)
  figman.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
 for name,field,ylabel in [('marked_path_W','W','Empirical W(J)'),('tangent_hessian','hessian_eig','Smallest covariance eigenvalue'),('marked_cumulants','k4','Fourth directional cumulant')]:
  fig,ax=plt.subplots(figsize=(8,5))
  for c,r in zip(curves,results):
   vals=np.asarray(c[field]);vals=vals[:,0] if field=='hessian_eig' else vals
   ax.plot(c['J'],vals,label=r['label'])
   if field=='W':ax.plot(c['J'],c['W_prior'],ls=':',alpha=.65)
  ax.set(xlabel='source amplitude (dimensionless)',ylabel=ylabel,title=name.replace('_',' '));ax.legend(fontsize=8);plot(name,fig,['L16','Q21'])
 fig,ax=plt.subplots(figsize=(9,5));x=np.arange(5)
 for i,l in enumerate(['initial frozen','adaptive IID','exact adaptive','history reduced']):
  ax.bar(x+(i-1.5)*.19,[r['nll'][i]['mean'] for r in results],.18,label=l)
 ax.set_xticks(x,labels);ax.set(ylabel='count NLL per wave (nat)',title='Held-out actual conditional count laws');ax.legend(fontsize=8);plot('conditional_count_closure',fig,['L16'])
 fig,ax=plt.subplots(figsize=(9,5))
 for i,l in enumerate(['clock','state','state + history']):ax.bar(x+(i-1)*.25,[r['mark_rmse'][i] for r in results],.24,label=l)
 ax.set_xticks(x,labels);ax.set(ylabel='RMSE of wave mean root-value mark',title='Geometry cross-fitted mark predictions');ax.legend();plot('marked_history_closure',fig,['L16','Q21'])
 fig,ax=plt.subplots(figsize=(8,5))
 for g,l in enumerate(labels):
  for field,ls in [('leading_rel','--'),('next_rel','-')]:
   xx=[];yy=[]
   for M in (8,16,32,64,128,256):
    rs=[abs(r[field]) for r in saddles if r['family']==g and r['M']==M]
    if rs:xx.append(M);yy.append(max(max(rs),1e-16))
   if xx:ax.loglog(xx,yy,ls,marker='o',label=l+' '+field.split('_')[0])
 ax.set(xlabel='independent replicas M',ylabel='maximum absolute relative error',title='Exact tilted convolution vs saddle expansion');ax.legend(fontsize=7,ncol=2);plot('independent_replica_saddle',fig,['L16','Q21'])
 fig,ax=plt.subplots(figsize=(9,5))
 for g,r in enumerate(results):
  js=[v['j'] for v in r['martingale']];ms=[v['mean']['mean'] for v in r['martingale']];ci=np.array([v['mean']['ci95'] for v in r['martingale']])
  ax.errorbar(np.array(js)+(.006*(g-2)),ms,yerr=[np.array(ms)-ci[:,0],ci[:,1]-np.array(ms)],fmt='o-',label=r['label'],capsize=2)
 ax.axhline(1,ls=':');ax.set(xlabel='count source amplitude',ylabel='mean sequential likelihood ratio',title='Conditional exponential martingale; core-block 95% intervals');ax.legend(fontsize=8);plot('conditional_tilt_martingale',fig,['L16','Q21'])
 dump(S/'PLOT_MANIFEST70.json',dict(figures=figman,sources={'PHYSICS_RESULTS.json':sha(S/'PHYSICS_RESULTS.json'),'MARKED70.npz':sha(S/'MARKED70.npz')}))
if __name__=='__main__':main()
