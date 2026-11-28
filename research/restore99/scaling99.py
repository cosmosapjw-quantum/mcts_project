from base61 import *
from scipy.special import xlogy,expit
from scipy.stats import t as student
import gzip,time,sys,platform
O=B/'mqft_restore98_20261128';R87=B/'mqft_restore87_20261128';R92=B/'mqft_restore92_20261128';LS=[13,15,17,19,21];BS=[1,2,4,8,16,32]
FIELDS=['n','pmax','KL_norm','concentration','Fisher','KL','binary_KL','tail_KL','spike_concentration','tail_concentration','spike_TV','entropy_A','entropy_X','corrected_collision_X','X_concentration','X_KL','geometric_spread_A','geometric_spread_spike','geometric_spread_uniform','action','score_gap','conditional_trace','conditional_low','conditional_high','normalized_entropy_X']
MOM=['mean','variance','central3','central4','binder','binder_defined']
def ck(e,k,a,b,tol=1e-9):
 d=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));e[k]=max(e.get(k,0.),d);assert d<tol,(k,d)
def H(p):return float(-np.sum(xlogy(p,p)))
def D(p,u):return float(np.sum(xlogy(p,p/u)))
def tv(p,q):return .5*float(abs(p-q).sum())
def ci(z):
 z=np.asarray(z);m=z.mean(0);w=student.ppf(.975,len(z)-1)*z.std(0,ddof=1)/np.sqrt(len(z));return np.stack([m,m-w,m+w])
def moment(p,f):
 m=float(p@f);z=f-m;v=float(p@(z*z));c3=float(p@(z**3));c4=float(p@(z**4))
 return [m,v,c3,c4,1-c4/(3*v*v) if v>1e-14 else 0.,float(v>1e-14)]
def block(p,u,L,legal,b):
 y,x=np.divmod(legal,L);i=(y//b)*((L+b-1)//b)+x//b
 q=np.bincount(i,weights=p);w=np.bincount(i,weights=u);ix=w>0
 return q[ix],w[ix]
def one(L,x,a,f0,rg,h):
 e={};mask=a>0;legal=np.flatnonzero(mask);p=a[mask];x=x[mask];n=len(p);u=np.ones(n)/n;assert n>1 and abs(p.sum()-1)<1e-10 and abs(x.sum()-1)<1e-10
 j=int(p.argmax());m=float(p[j]);tail=np.delete(p,j)/(1-m);sp=np.full(n,(1-m)/(n-1));sp[j]=m
 assert m<1 and abs(tail.sum()-1)<1e-9
 kl=D(p,u);binary=D(np.array([m,1-m]),np.array([1/n,1-1/n]));dk=(1-m)*D(tail,np.ones(n-1)/(n-1))
 c=(n*float(p@p)-1)/(n-1);cs=((n*m-1)/(n-1))**2;ct=n/(n-1)*(1-m)**2*float(np.sum((tail-1/(n-1))**2))
 ck(e,'KL_tail_identity',kl,binary+dk);ck(e,'collision_tail_identity',c,cs+ct);ck(e,'KL_spike',D(sp,u),binary);ck(e,'concentration_spike',(n*(sp@sp)-1)/(n-1),cs)
 assert dk>=-1e-10 and ct>=0 and m>=1/n-1e-12
 # R98 actual A and X coarse KL, with peak-matched spike reference distinct.
 scale=[]
 for b,rv in zip(BS,rg):
  q,w=block(p,u,L,legal,b);z,_=block(sp,u,L,legal,b);xx,_=block(x,u,L,legal,b)
  k0,k1=D(q,w),D(z,w);ck(e,'R98_A_KL',k0,rv[0]);ck(e,'R98_X_KL',D(xx,w),rv[1])
  scale.append([k0,k1,tv(q,z),float(rv[3])])
 scale=np.array(scale);assert np.max(np.diff(scale[:,:3],axis=0))<1e-9
 y,xx=np.divmod(legal,L);xy=np.c_[xx,y]/L
 def geom(z):return float(2*(z@(xy*xy).sum(1)-np.sum((z@xy)**2)))
 f=f0[mask].sum(1)/np.sqrt(2);f-=u@f
 moms=np.array([moment(v,f) for v in [p,x,sp,u]])
 # Independent raw-to-central moment identities, absolute tolerances only on moments not ill-conditioned ratios.
 for v,mom in zip([p,x,sp,u],moms):
  raw=np.array([sum(float(pi)*float(fi)**k for pi,fi in zip(v,f)) for k in range(1,5)]);mu=raw[0];cen=[mu,raw[1]-mu**2,raw[2]-3*mu*raw[1]+2*mu**3,raw[3]-4*mu*raw[2]+6*mu**2*raw[1]-3*mu**4]
  ck(e,'moment_raw_identity',mom[:4],cen)
  if mom[1]>1e-5:ck(e,'binder_independent',mom[4],1-cen[3]/(3*cen[1]**2),1e-5)
 cx=(n*(x@x)-1)/(n-1);cxcor=(h*cx-1)/(h-1);hx=H(x)
 v=[n,m,kl/np.log(n),c,float(rg[0,3]),kl,binary,dk,cs,ct,tv(p,sp),H(p),hx,cxcor,cx,D(x,u),geom(p),geom(sp),geom(u),0,0,0,0,0,hx/np.log(n)]
 # "action,score_gap,conditional_*" populated at stratum level from R92; not invented per path.
 return np.array(v),moms,scale,e
def rehash(arg):
 k,h=arg;assert sha(B/k)==h,k;return k,h
def main():
 start=time.monotonic();m=read(O/'RUN_MANIFEST98.json');m92=read(R92/'RUN_MANIFEST92.json');raw=m['prior_raw_receipts'];assert len(raw)==4320 and raw==m92['prior_raw_receipts']
 with cf.ThreadPoolExecutor(max_workers=6) as pool:assert dict(pool.map(rehash,raw.items()))==raw
 print('RAW HASHES',len(raw),flush=True)
 sources={};err={};out={};groups=[]
 def pin(p,h):k=str(p.relative_to(B));assert sha(p)==h,k;sources[k]=h
 for f in ['RG98.npz','SUMMARY98.json','RG98.json','ORIGINAL_SOURCE98.json']:pin(O/f,m['files'][f])
 for f in ['SCALE92.json.gz','SUMMARY92.json']:pin(R92/f,m92['files'][f])
 for f in ['CRITICAL87.json','ERRATUM86.json']:
  pin(R87/f,m92['inputs'][str((R87/f).relative_to(B))])
 z92=read(R92/'SUMMARY92.json');key=lambda x:tuple(x[k] for k in ['L','cohort','arm','bi','budget','h'])
 s92={key(g):g for g in z92['strata']}
 with np.load(O/'RG98.npz') as rg:
  for L in LS:
   p=R87/f'OBS_L{L}.npz';pin(p,m['inputs'][str(p.relative_to(B))]);z=np.load(p);M=z['meta'];V=rg[f'V{L}'];assert np.array_equal(M,rg[f'M{L}'])
   a=z['A'];x=z['X'];f=z['modes'][:,:2];R=[];J=[];K=[]
   for i in range(len(M)):
    r,j,k,e=one(L,x[i],a[i],f,V[i],int(M[i,-1]));R.append(r);J.append(j);K.append(k)
    for n,v in e.items():err[n]=max(err.get(n,0.),v)
   R=np.array(R);J=np.array(J);K=np.array(K)
   out.update({f'M{L}':M,f'V{L}':R,f'J{L}':J,f'K{L}':K})
   for co,arm,bi,bud,h in sorted(set(tuple(v[[0,2,3,5,6]]) for v in M)):
    ii=np.all(M[:,[0,2,3,5,6]]==[co,arm,bi,bud,h],axis=1);root=[];mom=[];scale=[]
    for c in range(8):
     ix=ii&(M[:,1]==c);assert ix.sum()==(2 if co==2 else 4)
     root.append(R[ix].mean(0));mj=J[ix];# valid Binder only if all constituent draws have positive variance; record denominator
     m0=mj.mean(0);m0[:,4]=np.divide((mj[:,:,4]*mj[:,:,5]).sum(0),mj[:,:,5].sum(0),out=np.zeros(4),where=mj[:,:,5].sum(0)>0);mom.append(m0);scale.append(K[ix].mean(0))
    root=np.array(root);pr=s92[(L,co,arm,bi,bud,h)]['metric_ci'][0]
    for idx,j in [(3,9),(13,1),(14,0),(24,2),(18,10)]:ck(err,'R92_metric',root[:,idx].mean(),pr[j])
    # Do not store placeholders as scientific path results. Group-level inherited comparators only.
    groups.append(dict(L=L,cohort=int(co),arm=int(arm),bi=int(bi),beta=[0.,10/3,8.,16.,math.log(L*L)][bi],budget=int(bud),h=int(h),roots=8,reps=int(ii.sum()/8),root_values=root[:,:19].tolist(),ci=ci(root[:,:19]).tolist(),moment_root=np.array(mom).tolist(),moment_ci=ci(mom).tolist(),scale_ci=ci(scale).tolist(),R92_metrics=pr,normalized_entropy_X_ci=ci(root[:,24]).tolist()))
   print('L',L,len(M),flush=True)
 assert len(groups)==400 and sum(len(out[f'M{L}']) for L in LS)==10400
 # Actual prior shell bins independently rederive from raw root shells.
 with gzip.open(R92/'SCALE92.json.gz','rt') as f:raw92=json.load(f)
 b92={key(g):g for g in z92['binned']};nb=0;bins=[]
 for g in raw92['strata']:
  L=g['L'];p=b92[key(g)];c=np.asarray(p['bin_centers']);edges=np.r_[0,(c[1:]+c[:-1])/2,np.sqrt(2)+1e-12]
  sh=np.asarray(g['shell_root_values']);r=np.sqrt(np.arange(sh.shape[-1]))/L;idx=np.digitize(r,edges)-1
  # R92 excludes self (d=0) from radial bins.
  v=np.stack([np.stack([np.bincount(idx[1:],weights=v0[1:],minlength=14)[:14] for v0 in row]) for row in sh])
  ck(err,'R92_radial_bins',v,p['bin_root']);nb+=1
  if g['cohort']==2 and g['budget']==64 and g['arm']==1 and g['bi']==3:bins.append(p)
 assert nb==400 and len(bins)==15
 # exact independent one-spike control on finite arrays for reasonable n; analytic large n.
 controls=[]
 for n in [16,64,169,441,1000,1000000]:
  for x0 in np.linspace(-6,6,25):
   p=float(expit(x0));beta=math.log(n-1)+float(x0);f=p*(1-p);u4=2-1/(3*f);mu4=(1-p)*p**4+p*(1-p)**4;ck(err,'Bernoulli_fourth',u4,1-mu4/(3*f*f))
   if n<=1000:
    q=np.r_[np.exp(beta),np.ones(n-1)];q/=q.sum();ck(err,'one_spike_control',q[0],p);ck(err,'one_spike_Fisher',np.sum(q*(np.r_[1.,np.zeros(n-1)]-p)**2),f)
   controls.append(dict(n=n,x=float(x0),beta=beta,M=p,Fisher=f,U4=u4))
 # Only meaningful per-prefix fields in stored NPZ; leave derived comparisons separate.
 for L in LS:out[f'V{L}']=out[f'V{L}'][:,:19]
 np.savez_compressed(S/'SCALE99.npz',**out)
 dump(S/'SUMMARY99.json',dict(groups=groups,fields=FIELDS[:19],moment_fields=MOM,scales=BS,moment_distributions=['A','X','matched_spike','uniform'],scale_fields=['KL_A','KL_spike','TV_A_spike','Fisher'],radial_bins=bins,controls=controls,prior_critical_gates=z92['critical']['gates'],scope='Finite categorical mixture A_h and visit X_h; matched spike is diagnostic, not independent prediction; one-draw centered fourth moment, not physical Binder'))
 dump(S/'INDEPENDENT99.json',dict(status='PASS_SCOPED',checks=err,raw_receipts=raw,source_hashes=sources,software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),independence='All prefix rederived from authenticated A/X; raw-to-central moments vs direct sums, R98 KL, R92 native metrics and exact radial shells; no fresh raw path replay'))
 dump(S/'SCALE99.json',dict(status='PASS_SCOPED',paths=4320,prefix_rows=10400,scale_rows=62400,root_strata=400,moment_cases=41600,radial_strata=400,analytic_controls=len(controls),array_sha=sha(S/'SCALE99.npz'),elapsed_s=time.monotonic()-start))
 print('RESULT',read(S/'SCALE99.json'),err,flush=True)
if __name__=='__main__':main()
