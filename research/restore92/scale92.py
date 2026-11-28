from base61 import *
import gzip,time
from scipy.fft import rfft2,irfft2,next_fast_len
from scipy.stats import t as student
from itertools import combinations
O=B/'mqft_restore87_20261128';LS=[13,15,17,19,21];CUTS=[16,32,64];BETAS=[0.,10/3,8.,16.]
def geometry(L):
 y,x=np.divmod(np.arange(L*L),L);xy=np.stack([x,y],1).astype(float);z=np.arange(-L+1,L);d=(z[:,None]**2+z[None,:]**2).ravel()
 side=next_fast_len(2*L-1);ix=z%side
 return xy,d,side,ix,2*(L-1)**2+1
def shell(v,L,g=None):
 xy,dist,side,ix,ns=geometry(L) if g is None else g
 v=np.asarray(v);a=v.reshape(*v.shape[:-1],L,L);q=rfft2(a,s=(side,side),axes=(-2,-1));s=irfft2(q*q.conj(),s=(side,side),axes=(-2,-1))
 c=s[...,ix[:,None],ix].reshape(-1,len(dist));out=np.stack([np.bincount(dist,weights=x,minlength=ns) for x in c])
 return out.reshape(*v.shape[:-1],ns)
def dct(L):
 y,x=np.divmod(np.arange(L*L),L);return np.stack([np.cos(np.pi*k*(a+.5)/L) for k in (1,2) for a in (x,y)],1)
def one(arg):
 L,r=arg;f=B/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
 tr=row['trace'];sp=tr['spec'];k=sp['budget'];beta=sp['root_law']['beta_p'][0];rep=tr['replicas'][0];assert end['backups']==k==r['budget'] and sp['seed']==r['seed']
 es=sorted(rep['events'],key=lambda e:e['root_selection']['clock']);assert len(es)==k
 n=L*L;g=geometry(L);xy=g[0];F=dct(L);mask=np.array(tr['root']['board'])==0;legal=np.flatnonzero(mask);m=len(legal)
 p=np.zeros((k,n));act=[];logs=[];scoregap=[];maxerr=0.
 for j,e in enumerate(es):
  law=e['root_selection']['law'];ac=np.array(law['actions']);q=np.array(law['p']);s=np.array(law['scores']);a=e['actions'][0]
  assert law['t']==j and law['logit_h']==0 and all(law['active']) and np.array_equal(np.sort(ac),legal) and ac[law['picked']]==a and e['root_value']==(-1)**len(e['actions'])*e['leaf_value']
  v=np.exp(beta*(s-s.max()));v/=v.sum();maxerr=max(maxerr,float(abs(v-q).max()));assert min(q)>0 and abs(q.sum()-1)<1e-10
  p[j,ac]=q;act.append(a);logs.append(-math.log(q[law['picked']]));ss=np.partition(s,-2);scoregap.append(float(ss[-1]-ss[-2]))
 assert maxerr<1e-9 and beta==(BETAS+[math.log(L*L)])[r['beta_index']]
 # shell sums of conditional covariance. Zero-padding avoids wraparound.
 ac=shell(p,L,g);ac[:,0]=1-ac[:,0];ac[:,1:]*=-1
 u=mask.astype(float)/m;nu=-shell(u,L,g);nu[0]+=1
 pairs0=shell(mask.astype(float),L,g);pairs=np.rint(pairs0).astype(int);assert abs(pairs-pairs0).max()<1e-8 and pairs.sum()==m*m and pairs[0]==m
 mu=p@xy;geo=2*(p@np.sum(xy*xy,1)-np.sum(mu*mu,1));d=np.arange(len(nu));err=max(float(abs(ac.sum(1)).max()),float(abs(-ac@d-geo).max()),abs(float(nu.sum())))
 assert err<1e-9
 # independent direct shell check of first/middle/last for each fresh core rep0 at one beta and both densities.
 direct=0.
 if r['cohort']==2 and r['rep']==0 and r['beta_index']==3:
  zz=xy[legal];dd=np.sum((zz[:,None]-zz[None,:])**2,axis=-1).astype(int)
  pc=np.bincount(dd.ravel(),minlength=len(nu));assert np.array_equal(pc,pairs)
  for t in [0,k//2,k-1]:
   q=p[t,legal];K=np.diag(q)-np.outer(q,q)
   direct=max(direct,float(abs(np.bincount(dd.ravel(),weights=K.ravel(),minlength=len(nu))-ac[t]).max()))
  assert direct<1e-9
 if beta==0:assert abs(ac-nu).max()<1e-9
 a=np.array(act);V=[];M=[];P=[];ORD=[];NULL=[];Q=[];G=[]
 centered=F[:,0:2]-u@F[:,0:2]
 # independent uniform legal null and scalar concentration.
 for h in CUTS:
  if h>k:continue
  x=np.bincount(a[:h],minlength=n)/h;dr=p[:h].mean(0);assert abs(x.sum()-1)<1e-12 and abs(dr.sum()-1)<1e-10
  occ=x[x>0];ss=np.sort(x);mraw=(m*(x@x)-1)/(m-1);mc=(h*mraw-1)/(h-1)
  ent=float(-occ@np.log(occ)/np.log(m));mode=x@centered
  pp=p[:h];fp=pp@F;vv=(pp@(F*F)-fp*fp).mean(0)
  V.append([x,dr]);M.append([r['cohort'],r['core'],r['arm'],r['beta_index'],r['rep'],k,h]);P.append(ac[:h].mean(0));NULL.append(nu)
  Q.append([float(ac[:h,0].mean()),float(vv[:2].mean()),float(vv[2:].mean())])
  ORD.append([mraw,mc,ent,float(ss[-1]-ss[-2]),*mode,float(np.mean(logs[:h])),float(np.mean(scoregap[:h])),float(np.mean(geo[:h]))/L**2, float((m*(dr@dr)-1)/(m-1))])
  G.append(float(-nu@d)/L**2)
 sec=None
 if k==64:
  y,xx=np.divmod(np.arange(n),L);sector=np.digitize(np.maximum(abs(xx-L//2),abs(y-L//2)),[2,4]);sec=(p@np.eye(3)[sector],sector[a])
 return dict(meta=np.array(M),vectors=np.array(V),pred=np.array(P),null=np.array(NULL),pairs=pairs,order=np.array(ORD),Q=np.array(Q),geom_null=G,sec=sec,mask=mask,err=err,direct=direct,lawerr=maxerr)
def ci(a):
 a=np.asarray(a);mu=a.mean(0);h=student.ppf(.975,len(a)-1)*a.std(0,ddof=1)/np.sqrt(len(a));return np.stack([mu,mu-h,mu+h])
def main():
 begin=time.monotonic();allrows=[];temps=[];receipts=[];audit=[];total=0;direct_n=0
 for L in LS:
  q=read(O/f'AUDIT_L{L}.json');rec=q['raw_receipts'];assert len(rec)==864
  assert sha(O/f'OBS_L{L}.npz')==q['array_sha'] and sha(O/f'SECTOR_L{L}.npz')==q['sector_sha']
  with cf.ProcessPoolExecutor(max_workers=10) as pool:out=list(pool.map(one,[(L,r) for r in rec],chunksize=4))
  z=np.load(O/f'OBS_L{L}.npz');sec=np.load(O/f'SECTOR_L{L}.npz');M=np.concatenate([x['meta'] for x in out]);V=np.concatenate([x['vectors'] for x in out]);Pred=np.concatenate([x['pred'] for x in out]);Null=np.concatenate([x['null'] for x in out]);Q=np.concatenate([x['Q'] for x in out]);Order=np.concatenate([x['order'] for x in out]);Geom=np.concatenate([x['geom_null'] for x in out]);Pairs=np.concatenate([np.repeat(x['pairs'][None,:],len(x['meta']),axis=0) for x in out])
  assert np.array_equal(M,z['meta']) and abs(V[:,0]-z['X']).max()<1e-12 and abs(V[:,1]-z['A']).max()<1e-11 and abs(Q-z['Q']).max()<1e-10 and len(M)==2080
  sp=np.stack([x['sec'][0] for x in out if x['sec'] is not None]);sa=np.stack([x['sec'][1] for x in out if x['sec'] is not None]);sm=np.array([x['meta'][0,:5] for x in out if x['sec'] is not None])
  assert np.array_equal(sm,sec['meta']) and np.array_equal(sa,sec['selected']) and abs(sp-sec['p']).max()<1e-12
  assert abs(Null[:,0]-z['null'][:,0]).max()<1e-10
  # group matched root replicas; covariance shell of pair differences; direct matrix check.
  geom=geometry(L);maxcov=0.;maxtrace=0.;maxsum=0.;maxdirect=0.;max_temp=0.;groups=sorted(set(tuple(m[[0,2,3,5,6]]) for m in M))
  for co,arm,bi,budget,h in groups:
   ix=(M[:,0]==co)&(M[:,2]==arm)&(M[:,3]==bi)&(M[:,5]==budget)&(M[:,6]==h);root=[];ps=[];allmask=[]
   for core in range(8):
    ii=np.flatnonzero(ix&(M[:,1]==core));x=V[ii,0];a=V[ii,1];d=x-a;nr=len(ii);assert nr==(2 if co==2 else 4) and np.all(Pairs[ii]==Pairs[ii[0]]) and np.allclose(Null[ii],Null[ii[0]],atol=1e-12)
    # E pair difference products /2 yields unbiased sample covariance.
    ds=[(x[j]-x[k],a[j]-a[k],d[j]-d[k]) for j,k in combinations(range(nr),2)]
    xx,aa,dd=map(np.stack,zip(*ds));sc=h/2
    raw,dr,inn=[sc*shell(v,L,geom).mean(0) for v in [xx,aa,dd]]
    cross=sc*(shell(aa+dd,L,geom)-shell(aa,L,geom)-shell(dd,L,geom)).mean(0)
    pred=Pred[ii].mean(0);nu=Null[ii].mean(0);row=np.stack([raw,dr,inn,cross,pred,nu])
    e=float(abs(raw-dr-inn-cross).max());maxcov=max(maxcov,e)
    maxsum=max(maxsum,float(abs(row.sum(1)).max()))
    maxtrace=max(maxtrace,abs(raw[0]-h*np.sum((x-x.mean(0))**2)/(nr-1)),abs(pred[0]-Q[ii,0].mean()))
    if co==2 and core==0 and h==64:
     zz=geom[0];dist=np.sum((zz[:,None]-zz[None,:])**2,-1).astype(int)
     K=h*(x-x.mean(0)).T@(x-x.mean(0))/(nr-1);d0=np.bincount(dist.ravel(),weights=K.ravel(),minlength=geom[-1])
     maxdirect=max(maxdirect,float(abs(d0-raw).max()))
    assert max(e,maxsum,maxtrace,maxdirect)<1e-9
    ps.append(Pairs[ii[0]]);root.append(row);allmask.append(np.r_[Order[ii].mean(0),Geom[ii].mean(),Q[ii].mean(0)])
   root=np.array(root);pair=np.array(ps);metric=np.array(allmask)
   # per-root pair means, undefined shells remain NaN until serialized as mask/zeros.
   valid=pair>0;avg=np.divide(root,pair[:,None,:],out=np.zeros_like(root),where=valid[:,None,:]);vc=valid.sum(0);avgm=np.divide(avg.sum(0),vc[None,:],out=np.zeros_like(avg[0]),where=vc[None,:]>0)
   # Root uncertainty on shell mass, per-pair estimates separate.
   allrows.append(dict(L=L,cohort=int(co),arm=int(arm),bi=int(bi),beta=(BETAS+[math.log(L*L)])[bi],budget=int(budget),h=int(h),roots=8,reps=nr,metric=metric.tolist(),metric_ci=ci(metric).tolist(),shell_ci=ci(root).tolist(),pairmean=avgm.tolist(),pair_root_count=vc.tolist(),pairs_mean=pair.mean(0).tolist(),shell_root_values=root.tolist()))
  # temporal within-root sample covariance across independent repetitions at fixed t,s, no stationarity assumption
  for co,arm,bi in sorted(set(tuple(m[[0,2,3]]) for m in sm)):
   root=[];start=[]
   for core in range(8):
    ii=np.flatnonzero((sm[:,0]==co)&(sm[:,1]==core)&(sm[:,2]==arm)&(sm[:,3]==bi));nr=len(ii);a=np.eye(3)[sa[ii]];p=sp[ii];d=a-p
    vv=[v-v.mean(0,keepdims=True) for v in (a,p,d)]
    def cov(x,y):return np.einsum('rtc,rsc->ts',x,y)/(nr-1)
    ra,dr,inn=[cov(v,v) for v in vv];cross=cov(vv[1],vv[2])+cov(vv[2],vv[1])
    assert abs(ra-dr-inn-cross).max()<1e-12
    if co==2 and core==0:
     # direct two-rep identity
     max_temp=max(max_temp,float(abs(ra-(a[0]-a[1])@(a[0]-a[1]).T/2).max()))
    c=np.stack([ra,dr,inn,cross])
    root.append(np.stack([np.diagonal(c[:,4:,4:],offset=t,axis1=1,axis2=2).mean(-1) for t in range(17)],axis=1))
    start.append(np.stack([np.diagonal(c[:,:4,:4],offset=t,axis1=1,axis2=2).mean(-1) for t in range(4)],axis=1))
   temps.append(dict(L=L,cohort=int(co),arm=int(arm),bi=int(bi),beta=(BETAS+[math.log(L*L)])[bi],roots=8,reps=nr,lag_ci=ci(root).tolist(),startup_ci=ci(start).tolist(),root_lag=np.array(root).tolist()))
  audit.append(dict(L=L,paths=len(rec),rows=len(M),B64=len(sm),max_path_identity=max(x['err'] for x in out),max_direct_path=max(x['direct'] for x in out),max_law=max(x['lawerr'] for x in out),max_array=float(abs(Q-z['Q']).max()),max_cov=maxcov,max_trace=maxtrace,max_zero=maxsum,max_direct_cov=maxdirect,max_temporal=max_temp,raw_receipts=rec))
  receipts.extend(rec);total+=len(M);print('SIZE',L,len(rec),len(M),len(sm),'errors',maxcov,maxsum,maxdirect,flush=True)
 assert len(receipts)==4320 and total==10400 and len({x['path'] for x in receipts})==4320 and sum(x['B64'] for x in audit)==3040
 # exact finite iid concentration sanity by enumeration, n=3 B=4, biased law too
 from itertools import product
 errs=[]
 for p in (np.ones(3)/3,np.array([.2,.3,.5])):
  m=0.;mc=0.
  for seq in product(range(3),repeat=4):
   x=np.bincount(seq,minlength=3)/4;v=(3*x@x-1)/2;w=np.prod(p[list(seq)]);m+=w*v;mc+=w*(4*v-1)/3
  target=(3*p@p-1)/2;errs.append(max(abs(m-(.25+.75*target)),abs(mc-target)))
 assert max(errs)<1e-12
 dump(S/'SCALE92.json',dict(status='PASS_SCOPED',paths=len(receipts),prefix_rows=total,B64_paths=sum(x['B64'] for x in audit),strata=allrows,temporal=temps,metric_fields=['concentration','corrected_collision','normalized_entropy','top_gap','signed_mode_x','signed_mode_y','action_per_selection','score_gap','geometric_spread_over_L2','drift_concentration','null_geometric_spread_over_L2','trace','low','high'],shell_fields=['raw','drift','innovation','cross','predictable','uniform_null'],temporal_fields=['raw','drift','innovation','cross'],iid_check=max(errs),elapsed_s=time.monotonic()-begin,scope='Ordered legal endpoint squared Euclidean displacement shells; no physical correlation length. Root blocks; conditional instantaneous response only.'))
 dump(S/'INDEPENDENT92.json',dict(status='PASS_SCOPED',audit=audit,independence='same-agent FFT vs direct pair enumeration and sample covariance, prior R87 full replay inherited; no independent subagent',erratum_sha=sha(O/'ERRATUM86.json'),critical_sha=sha(O/'CRITICAL87.json')))
 print('COMPLETE',len(allrows),len(temps),total,round(time.monotonic()-begin,2),flush=True)
if __name__=='__main__':main()
