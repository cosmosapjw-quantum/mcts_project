from base61 import *
from scipy.special import logsumexp,xlogy
from scipy.optimize import brentq
import gzip,csv,time
O=B/'mqft_restore85_20261127';A=B/'mqft_restore79_20261127';R=B/'mqft_restore61_20261126'
CUT=np.array([4,8,16,32,64,128]);FAM=['g0','g3','Rapfi'];BETA=10/3
def H(p):return -np.sum(xlogy(p,p),axis=-1)
def kl(p,q):return np.sum(xlogy(p,p)-xlogy(p,q),axis=-1)
def gibbs(e,b):
 l=-b*e;l-=logsumexp(l,axis=-1,keepdims=True);return np.exp(l)
def energycheck(r):
 f=B/r['path'];assert sha(f)==r['sha256'] and f.stat().st_size==r['bytes']
 with gzip.open(f,'rt') as z:d=json.load(z)
 es=d['trace']['events'];cuts={c['k']:c for c in d['trace']['cuts']};assert len(es)==128
 n=np.zeros(len(cuts[4]['n']),int);w=np.zeros(len(n));vals=[];marks=[]
 for k,e in enumerate(es,1):
  i=e['actions'][0];z=e['root_value'];assert z==(-1)**len(e['actions'])*e['leaf_value'];n[i]+=1;w[i]+=z;marks.append(z)
  if k in CUT:
   c=cuts[k];assert np.array_equal(n,c['n']) and np.max(abs(w-c['w']))<1e-9
   q=np.divide(w,n,out=np.zeros(len(n)),where=n>0);a=-n@q/k;b=-np.mean(marks)
   assert abs(a-b)<1e-12;vals.append(a)
 return (r['family'],r['cohort'],r['core'],r['rep']),vals
def native():
 au=read(A/'AUDIT79.json');rec=au['raw_receipts'];assert len(rec)==960
 with cf.ProcessPoolExecutor(max_workers=10) as pool:res=dict(pool.map(energycheck,rec,chunksize=8))
 assert len(res)==960
 ret={};maxe=0.;maxb=0.;checks=0;max_fit_beta=0.;max_fit_prob=0.;max_fit_KL=0.
 for g,f in enumerate(FAM):
  name=f'CALIB_{f}.npz';assert sha(A/name)==au['arrays'][name]
  with np.load(A/name) as z:
   meta=z['meta'];mask=z['mask'];score=z['score'];p=z['p'];occ=z['occ'];q=z['Qcut']
  e=-(occ*q).sum(-1);truth=np.array([res[(g,*m)] for m in meta]);maxe=max(maxe,float(abs(e-truth).max()))
  m=mask[:,None,:];assert np.max(abs(p.sum(-1)-1))<1e-12 and np.max(abs(occ.sum(-1)-1))<1e-12
  lp=np.where(m,np.log(np.maximum(p,1e-300)),0);nn=m.sum(-1)
  sc=np.where(m,score-(score*m).sum(-1)[...,None]/nn[...,None],0)
  lc=np.where(m,lp-(lp*m).sum(-1)[...,None]/nn[...,None],0)
  v=(sc*sc).sum(-1);ok=v>1e-24;beta=(sc*lc).sum(-1)[ok]/v[ok];maxb=max(maxb,float(abs(beta-BETA).max()));checks+=int(ok.sum())
  cal=read(A/'CALIBRATION79.json')['results'][g];assert cal['family']==f
  te=meta[:,0]==1
  # exact projection onto restricted exponential family at each observed cut; independent bisection.
  fits=[]
  for i in np.flatnonzero(te):
   for k,x,y in zip(CUT,q[i],occ[i]):
    x=x[mask[i]];y=y[mask[i]]
    def grad(b):return float(gibbs(-x,b)@x-y@x)
    if np.ptp(x)<1e-12:b=None;status='flat';err=None
    elif grad(-32)>=0:b=-32.;status='lower_bound';err=float(kl(y,gibbs(-x,b)))
    elif grad(32)<=0:b=32.;status='upper_bound';err=float(kl(y,gibbs(-x,b)))
    else:
     lo,hi=-32.,32.
     for _ in range(45):
      mid=(lo+hi)/2
      if grad(mid)>0:hi=mid
      else:lo=mid
     b=(lo+hi)/2;status='interior';err=float(kl(y,gibbs(-x,b)))
    fits.append(dict(core=int(meta[i,1]),rep=int(meta[i,2]),B=int(k),beta=b,status=status,KL=err))
  prev={(x['core'],x['rep'],x['B']):x for x in cal['occupancy_fits']};idx={(int(m[1]),int(m[2])):i for i,m in enumerate(meta) if m[0]==1}
  for x in fits:
   a=prev[x['core'],x['rep'],x['B']]
   assert x['status']==a['status'],(x,a)
   if x['beta'] is not None:
    i=idx[x['core'],x['rep']];xx=q[i,np.flatnonzero(CUT==x['B'])[0],mask[i]]
    pd=float(abs(gibbs(-xx,x['beta'])-gibbs(-xx,a['beta'])).max());kd=abs(x['KL']-a['occupancy_KL'])
    max_fit_beta=max(max_fit_beta,abs(x['beta']-a['beta']));max_fit_prob=max(max_fit_prob,pd);max_fit_KL=max(max_fit_KL,kd)
    assert pd<1e-7 and kd<1e-9,(x,a,pd,kd)
  ret[f]=dict(energy_by_cut=e[te].tolist(),root_ids=meta[te,1].tolist(),fits=fits,conditional_beta_fit=cal['models'][0]['beta'],flat_waves=int((~ok).sum()),conditional_beta_min=float(beta.min()),conditional_beta_max=float(beta.max()))
 assert maxe<1e-12 and maxb<1e-8
 r=read(R/'R60_REDUCTION61.json');assert sha(R/'R60_PREFIX61.npz')==r['array_sha256']
 with np.load(R/'R60_PREFIX61.npz') as z:
  values=z['values'];meta=z['meta'];fields=list(z['fields']);assert len(values)==63438
  E=values[:,fields.index('E_Q')];assert np.max(abs(E))<=1+1e-12
  game=[];valid=meta[:,4]==32
  for i in sorted(set(meta[valid,0])):
   ix=np.flatnonzero(valid&(meta[:,0]==i));ix=ix[np.argsort(meta[ix,3])]
   game.append(dict(game=int(i),ply=meta[ix,3].tolist(),energy=E[ix].tolist(),tag=int(meta[ix[0],1])))
 dump(S/'NATIVE91.json',dict(status='PASS_SCOPED',histories=len(res),backup_events=len(res)*128,cuts=len(res)*6,max_energy_error=maxe,max_slope_error=maxb,identified_waves=checks,max_projection_beta_difference=max_fit_beta,max_projection_probability_difference=max_fit_prob,max_projection_KL_difference=max_fit_KL,families=ret,game_rows=len(values),game_trajectories=game,game_scope='R61 hashed reduction reused, not new game or raw reread',scope='Native score units; temperature proxy versus conditional beta versus endpoint projection'))
 print('NATIVE',len(res),maxe,maxb,checks,flush=True)
def branch(c,b,eps,alpha):
 e=np.array(c['e0']);em=np.array(c['em']);g=np.array(c['group']);n=len(e);m=len(em)
 p=gibbs(e,b);pi=gibbs(em,b);M=np.full((n,m),eps/m);M[np.arange(n),g]+=1-eps
 joint=(p[:,None]*M).T;pm=joint.sum(1);assert np.min(pm)>0
 a=joint/pm[:,None];bb=(1-alpha)*a+alpha*pi;end=pm@bb
 Q=float(pm@np.sum((bb-a)*em,axis=1));W=float(pm@np.sum((a-bb)*(em-e),axis=1));DE=float((end-p)@e)
 i0=float(H(p)-pm@H(a));i1=float(H(end)-pm@H(bb));ds=float(pm@(H(bb)-H(a)))
 # relative entropy loss of conditional relaxation; independent signed entropy balance.
 sr=float(pm@(kl(a,pi)-kl(bb,pi)));d=float(kl(end,p));total=b*W+i0
 err=max(abs(DE-W-Q),abs(sr-ds+b*Q),abs(total-sr-d-i1))
 assert min(sr,d,i0,i1)>-1e-10 and err<1e-9
 return dict(W=W,Q=Q,DE=DE,I0=i0,I1=i1,sigma_rel=sr,D_final=d,total=total,dS_cond=ds,dS_sys=float(H(end)-H(p)),residual=err), (e,em,p,pi,M,pm,a,bb)
def explicit(t,b,alpha):
 e,em,p,pi,M,pm,a,bb=t;n=len(e);res=np.zeros(6);end=np.zeros(n)
 for k in range(len(pm)):
  P=(1-alpha)*np.eye(n)+alpha*np.broadcast_to(pi[k],(n,n))
  F=(p*M[:,k])[:,None]*P;Rev=pm[k]*p[None,:]*P.T;u=F>0
  w=(em[k]-e)[:,None]-(em[k]-e)[None,:];q=em[k][None,:]-em[k][:,None]
  ss=np.log(F[u]/Rev[u]);res+=np.array([F.sum(),Rev.sum(),Rev[u].sum(),np.sum(F[u]*ss),np.sum(F*w),np.sum(F*q)])
  end+=F.sum(0)
 return res,end
def therm():
 source=read(O/'BRANCH_ENERGY85.json');old=read(O/'THERMAL85.json');assert sha(O/'BRANCH_ENERGY85.json')==old['energy_sha'] and sha(O/'THERMAL_SAMPLES85.npz')==old['raw_array_sha']
 for r in source['raw_receipts']:assert sha(B/r['path'])==r['sha256']
 ctx={(c['protocol'],c['family'],c['core']):c for c in source['contexts']};assert len(ctx)==104
 out=[];maxe=maxmc=maxdir=0.
 with np.load(O/'THERMAL_SAMPLES85.npz') as z:
  F=z['forward'];Rv=z['reverse']
 for k,x in enumerate(old['cases']):
  c=ctx[x['protocol'],x['family'],x['core']];eps=x['eps'];alpha=x['alpha'];r,t=branch(c,BETA,eps,alpha);e,em,p,pi,M,pm,a,bb=t
  err=max(abs(r[v]-x[w]) for v,w in [('W','mean_work'),('Q','mean_heat'),('DE','mean_delta_E'),('I0','mean_info'),('D_final','mean_final_KL'),('total','mean_sigma')])
  maxe=max(maxe,err);assert err<1e-9
  i,m,j=F[k].T;ii,mm,jj=Rv[k].T;w=em[m,i]-e[i]+e[j]-em[m,j];q=em[m,j]-em[m,i];inf=np.log(M[i,m]/pm[m]);wt=np.exp(-BETA*w-inf);sup=M[ii,mm]>0
  mc=dict(ift=float(wt.mean()),reverse_support=float(sup.mean()),work=float(w.mean()),heat=float(q.mean()),ess=float(wt.sum()**2/(wt@wt)),ift_se=float(wt.std(ddof=1)/np.sqrt(len(wt))))
  err=max(abs(mc[v]-x[w]) for v,w in [('ift','mc_ift'),('reverse_support','mc_reverse_support'),('work','mc_work'),('heat','mc_heat'),('ess','mc_ess'),('ift_se','mc_ift_se')])
  maxmc=max(maxmc,err);assert err<1e-8
  if k%64==0 or k%64==7:
   v,end=explicit(t,BETA,alpha);err=max(abs(v-np.array([1,1,x['exact_ift'],r['total'],r['W'],r['Q']])).max(),abs(end-pm@bb).max());maxdir=max(maxdir,float(err));assert err<1e-9
  out.append(dict(protocol=x['protocol'],family=x['family'],core=x['core'],eps=eps,alpha=alpha,**r,**mc,exact_ift=x['exact_ift'],uncorrected_ift=x['uncorrected_ift'],work_sd=x['work_sd'],singular=x['lambda_singular']))
 # fixed-H heat capacity with explicit temperature in score-energy units and independent finite derivative/pair variance
 T=np.geomspace(.08,3,41);capacity=[];maxc=0.;pairerr=0.;time_rows=[];limerr=0.
 for c in ctx.values():
  e=np.array(c['e0']);em=np.array(c['em']);E=np.vstack([e,em]);C=[]
  for theta in T:
   prob=gibbs(E,1/theta);u=(prob*E).sum(1);v=(prob*(E-u[:,None])**2).sum(1)/theta**2
   h=theta*1e-4;du=((gibbs(E,1/(theta+h))*E).sum(1)-(gibbs(E,1/(theta-h))*E).sum(1))/(2*h)
   maxc=max(maxc,float(abs(v-du).max()))
   # independent pairwise variance on first Hamiltonian, all contexts and T
   vv=.5*np.sum(prob[0,:,None]*prob[0,None,:]*(e[:,None]-e[None,:])**2)/theta**2;pairerr=max(pairerr,float(abs(v[0]-vv)))
   C.append([float(v[0]),float(v[1:].mean()),float(u[0]),float(H(prob[0]))])
  capacity.append(dict(protocol=c['protocol'],family=c['family'],core=c['core'],curve=C))
  for b in [1/3,1,10/3,10]:
   r,t=branch(c,b,.2,.25);e,em,p,pi,M,pm,a,bb=t
   z=a;rr=[]
   for k in range(8):
    zz=.75*z+.25*pi;ds=float(pm@(H(zz)-H(z)));q=float(pm@np.sum((zz-z)*em,axis=1));sig=float(pm@(kl(z,pi)-kl(zz,pi)))
    er=abs(sig-ds+b*q);assert sig>=-1e-10 and er<1e-9
    rr.append([float(pm@H(z)),float(pm@H(zz)),ds,q,sig,er]);z=zz
   time_rows.append(dict(protocol=c['protocol'],family=c['family'],core=c['core'],beta=b,steps=rr))
  r,_=branch(c,BETA,.2,0);limerr=max(limerr,abs(r['sigma_rel']),abs(r['W']),abs(r['Q']))
  r,_=branch(c,BETA,1,.25);limerr=max(limerr,abs(r['I0']));assert abs(r['I0'])<1e-10
 # I1 need not vanish for eps=1 if branch H differ! I0=0, I1 generated by branch-dependent dynamics.
 assert maxc<2e-6 and pairerr<1e-10
 dump(S/'THERMAL91.json',dict(status='PASS_SCOPED',contexts=len(ctx),cases=out,raw_receipts=len(source['raw_receipts']),samples_per_direction=len(F[0]),max_old_error=maxe,max_sample_error=maxmc,max_direct_error=maxdir,capacity_derivative_error=maxc,capacity_pair_error=pairerr,theta=T.tolist(),capacity=capacity,time=time_rows,limit_probe=limerr,scope='Declared finite feedback reference, signed entropy and information; not physical native bath'))
 print('THERMAL',len(out),maxe,maxmc,maxdir,maxc,pairerr,limerr,flush=True)
if __name__=='__main__':
 native();therm()
