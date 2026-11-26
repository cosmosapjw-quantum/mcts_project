"""Finite native geometry scaling with exact IID null, matched-root uncertainty and held-size predictions."""
from base61 import *
from scipy.stats import t as student
from itertools import combinations
LS=[13,15,17,19,21];BETAS=[0.,10/3,8.,16.];BS=[16,64];NS=2000
PAIRS=list(combinations(range(4),2));DIS=[(0,1,2,3),(0,2,1,3),(0,3,1,2)]
def kstats(x):
 # x: (...,4) independent replica scalar means; no IID assumption across root geometries.
 d=np.stack([(x[...,i]-x[...,j])**2 for i,j in PAIRS],-1)
 m22=np.mean(np.stack([(x[...,i]-x[...,j])**2*(x[...,k]-x[...,l])**2 for i,j,k,l in DIS],-1),-1)/4
 return np.mean(d,-1)/2,.5*np.mean(d*d,-1)-6*m22,m22
def binder(a,b):
 a=np.asarray(a);b=np.asarray(b);return np.divide(-a,3*b,out=np.full_like(a,np.nan,dtype=float),where=b>0)
def ratio(lo,hi):
 lo=np.asarray(lo);hi=np.asarray(hi);return np.divide(lo,lo+hi,out=np.full_like(lo,np.nan,dtype=float),where=lo+hi>0)
def ci(a):
 a=np.asarray(a);m=a.mean(0);h=student.ppf(.975,len(a)-1)*a.std(0,ddof=1)/np.sqrt(len(a));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def quant(a):
 a=np.asarray(a);v=a[np.isfinite(a)];return dict(valid=len(v),lo=float(np.quantile(v,.025)) if len(v) else None,hi=float(np.quantile(v,.975)) if len(v) else None,median=float(np.median(v)) if len(v) else None)
def modes(L):
 yy,xx=np.indices((L,L));return np.array([np.cos(np.pi*(xx+.5)/L),np.cos(np.pi*(yy+.5)/L),np.cos(2*np.pi*(xx+.5)/L),np.cos(2*np.pi*(yy+.5)/L)]).reshape(4,L*L)
def cov(x):
 u=x-x.mean(0);return u.T@u/(len(x)-1)
def rootstats(x,a,board,stat,prior,L,B):
 legal=board==0;K=legal.sum();p=legal/K;mode=modes(L);u=x-a;C=cov(x);D=cov(u);A=cov(a);cross=C-D-A
 assert abs(C.sum())<1e-11 and abs(C.sum(0)).max()<1e-11
 vv=np.diag(mode@C@mode.T);vnull=(mode*mode)@p-(mode@p)**2;vnull/=B
 x0=x@mode[0];m2,k4,m22=kstats(x0);assert abs(m2-vv[0])<1e-12
 v=mode[0,legal];v-=v.mean();mu2=np.mean(v*v);mu4=np.mean(v**4);kn=(mu4-3*mu2**2)/B**3;m22n=(mu2/B)**2
 xy=np.column_stack(np.divmod(np.arange(L*L),L));dist=abs(xy[:,None]-xy[None,:]).sum(-1);sel=legal[:,None]&legal[None,:]
 count=np.bincount(dist[sel],minlength=2*L-1);rad=np.bincount(dist[sel],weights=C[sel],minlength=2*L-1);radnull=np.bincount(dist[sel],weights=((np.diag(p)-p[:,None]*p[None,:])/B)[sel],minlength=2*L-1)
 # fixed source and top-prior pair order statistics.
 assert np.max(abs(prior-prior[0]))<1e-7 if prior.ndim==2 else True
 return dict(trace=float(np.trace(C)),low=float(vv[:2].mean()),high=float(vv[2:].mean()),k4=float(k4),mu22=float(m22),mu2=float(m2),null_trace=float((1-1/K)/B),null_low=float(vnull[:2].mean()),null_high=float(vnull[2:].mean()),null_k4=float(kn),null_mu22=float(m22n),drift=float(np.trace(A)),martingale=float(np.trace(D)),cross=float(np.trace(cross)),bracket=float(stat[:,8].mean()),radial_sum=rad.tolist(),radial_null_sum=radnull.tolist(),pair_counts=count.tolist(),fixed_pair_m=float(stat[:,0].mean()),top_pair_m=float(stat[:,1].mean()),top_pair_abs=float(abs(stat[:,1]).mean()),prior_gap=float(stat[:,2].mean()),top=float(stat[:,3].mean()),entropy=float(stat[:,4].mean()),root_mark=float(stat[:,5].mean()),capacity=float(stat[:,6].mean()),conditional_H=float(stat[:,7].mean()),chi_fixed=float(B*stat[:,0].var(ddof=1)),chi_top=float(B*stat[:,1].var(ddof=1)),path_score_response=float(np.cov(stat[:,0],stat[:,12],ddof=1)[0,1]),legal=int(K))
def null_draw(boards,L,B,seed):
 rng=np.random.default_rng(seed);mode=modes(L);out=[]
 for board in boards:
  legal=board==0;K=legal.sum();p=np.ones(K)/K;f=mode[:,legal]
  x=rng.multinomial(B,p,size=(NS,4)).astype(np.float64)/B
  tr=np.sum(x.var(1,ddof=1),axis=-1);q=x@f.T;v=q.var(1,ddof=1)
  _,k4,m22=kstats(q[:,:,0])
  out.append(np.stack([tr*B,v[:,:2].mean(1),v[:,2:].mean(1),k4,m22],-1))
 a=np.array(out).mean(0)
 return dict(trace_B=quant(a[:,0]),ratio=quant(ratio(a[:,1],a[:,2])),binder=quant(binder(a[:,3],a[:,4])),seed=seed,synthetic_cohorts=NS)
def crossings(beta,y):
 out=[]
 for i in range(len(beta)-1):
  a,b=y[i:i+2]
  if not np.isfinite(a+b):continue
  if a==0:out.append(float(beta[i]))
  elif a*b<0:out.append(float(beta[i]-a*(beta[i+1]-beta[i])/(b-a)))
 if y[-1]==0:out.append(float(beta[-1]))
 return out
def fit_predict(y,train,hold,kind):
 # y(L,beta) root aggregate. Fit on four sizes only; no threshold claim.
 sizes=np.array(LS,float);bt=np.array(BETAS)
 l,b=np.meshgrid(sizes,bt,indexing='ij')
 keep=np.isin(l,train);target=np.isin(l,hold)
 if not np.all(np.isfinite(y)):return None
 configs=[(0.,0.,0.)] if kind=='analytic' else [(c,w,p) for c in np.linspace(0,16,17) for w in [1.,2.,4.,8.] for p in ([0.] if kind=='crossover' else [.25,.5,1.,2.])]
 best=None
 for c,w,p in configs:
  if kind=='analytic':d=np.stack([np.ones_like(l),b/16,(b/16)**2,17/l-1],-1)
  else:d=np.stack([np.ones_like(l),np.tanh((b-c)*(l/17)**p/w),17/l-1],-1)
  A=d[keep];coeff=np.linalg.lstsq(A,y[keep],rcond=None)[0];loss=float(np.mean((A@coeff-y[keep])**2))
  if best is None or loss<best[0]:best=(loss,c,w,p,coeff,d)
 loss,c,w,p,coeff,d=best;pred=d[target]@coeff
 return dict(model=kind,train_rmse=math.sqrt(loss),test_rmse=float(np.sqrt(np.mean((pred-y[target])**2))),pred=pred.tolist(),actual=y[target].tolist(),beta_center=c,width=w,scaling_y=p,coeff=coeff.tolist())
def main():
 groups=[];aux=[];aud=[];data={};nulls=[];rng=np.random.default_rng(760991);boot=rng.integers(0,8,size=(NS,8));base_stats={}
 for L in LS:
  aa=read(S/f'AUDIT_L{L}.json');assert sha(S/f'SCALE_L{L}.npz')==aa['array_sha256'];aud.append(aa)
  with np.load(S/f'SCALE_L{L}.npz') as z:a={k:z[k] for k in z.files}
  data[L]=a;M=a['meta'];X=a['X'];A=a['A'];boards=a['board'];stat=a['stat'];prior=a['prior']
  def sel(core,arm,B,beta,c=1.4,h=0.):
   z=np.flatnonzero(np.all(M[:,:6]==[core,arm,B,beta,c,h],axis=1));assert len(z)==4
   return z[np.argsort(M[z,6])]
  for arm in (0,1):
   for B0 in BS:
    bo=[]
    for core in range(8):
     z=sel(core,arm,B0,0.);assert np.all(boards[z]==boards[z][0]);bo.append(boards[z][0])
    null=null_draw(bo,L,B0,760800+100*L+10*arm+B0);null.update(L=L,arm=arm,B=B0);nulls.append(null)
    for beta in BETAS:
     rs=[]
     for core in range(8):
      z=sel(core,arm,B0,beta);assert np.all(boards[z]==bo[core])
      r=rootstats(X[z],A[z],boards[z[0]],stat[z],prior[z],L,B0);r.update(core=core);rs.append(r)
     keys=[k for k in rs[0] if k not in ('core','radial_sum','radial_null_sum','pair_counts')]
     vals={k:np.array([r[k] for r in rs]) for k in keys};mean={k:float(v.mean()) for k,v in vals.items()}
     ratio0=float(ratio(mean['low'],mean['high']));binder0=float(binder(mean['k4'],mean['mu22']))
     rb=ratio(vals['low'][boot].mean(1),vals['high'][boot].mean(1));ub=binder(vals['k4'][boot].mean(1),vals['mu22'][boot].mean(1))
     radial=np.sum([r['radial_sum'] for r in rs],0);rnull=np.sum([r['radial_null_sum'] for r in rs],0);cnt=np.sum([r['pair_counts'] for r in rs],0)
     assert abs(radial.sum())<1e-10 and abs(rnull.sum())<1e-10
     rr=dict(L=L,arm=arm,B=B0,beta=beta,roots=rs,means=mean,ci={k:ci(v) for k,v in vals.items()},ratio=ratio0,binder=binder0,ratio_bootstrap=quant(rb),binder_bootstrap=quant(ub),null_ratio=float(ratio(mean['null_low'],mean['null_high'])),null_binder=float(binder(mean['null_k4'],mean['null_mu22'])),radial_sum=radial.tolist(),radial_null_sum=rnull.tolist(),radial_pair_counts=cnt.tolist())
     groups.append(rr);base_stats[L,arm,B0,beta]=(rr,rb,ub)
   # independent budget seed check (beta shared seed across fixed B/arm; B not shared)
  # c_puct independent condition and paired pulse within dense B16.
  for c in (.7,1.4,2.8):
   v=[rootstats(X[z],A[z],boards[z[0]],stat[z],prior[z],L,16) for core in range(8) for z in [sel(core,1,16,10/3,c)]]
   aux.append(dict(kind='cpuct',L=L,c=c,means={k:float(np.mean([q[k] for q in v])) for k in ['top_pair_m','top_pair_abs','top','capacity','chi_top','entropy','trace']},ci={k:ci([q[k] for q in v]) for k in ['top_pair_m','top_pair_abs','top','capacity','chi_top','entropy','trace']}))
  pulse=[]
  for core in range(8):
   a0=stat[sel(core,1,16,10/3)];plus=stat[sel(core,1,16,10/3,1.4,.35)];minus=stat[sel(core,1,16,10/3,1.4,-.35)]
   pp=a0[:,11];exact=lambda h:pp*np.exp(h)/(1+pp*np.expm1(h))
   direct=(plus[:,9]-minus[:,9])/.7;cond=(exact(.35)-exact(-.35))/.7
   # All three prefixes equal; last-wave observable is source fraction.
   pulse.append([direct.mean(),cond.mean(),np.mean(pp*(1-pp)),float((direct-cond).mean()),float(np.mean((plus[:,10]-minus[:,10])/.7)),float(np.cov(a0[:,9],a0[:,13],ddof=1)[0,1])])
  aux.append(dict(kind='pulse',L=L,root_stats=pulse,fields=['direct_source','exact_conditional_fd','exact_derivative','difference','direct_mark','sample_score_response'],ci=ci(pulse)))
  print('SIZE',L,'groups',len(groups),flush=True)
 # all crossings and held-size prediction, never select best crossing.
 cr=[];fits=[];peaks=[];joint=[]
 for arm in (0,1):
  for B0 in BS:
   for ob,idx in [('ratio',1),('binder',2)]:
    z=np.array([[base_stats[L,arm,B0,b][0][ob] for b in BETAS] for L in LS])
    boots=np.array([[base_stats[L,arm,B0,b][idx] for b in BETAS] for L in LS])
    for i in range(4):
     ys=z[i]-z[i+1];cs=crossings(BETAS,ys)
     cbs=[crossings(BETAS,boots[i,:,j]-boots[i+1,:,j]) for j in range(NS)]
     cr.append(dict(arm=arm,B=B0,observable=ob,L=LS[i],Lnext=LS[i+1],crossings=cs,bootstrap_fraction_any=sum(bool(x) for x in cbs)/NS,bootstrap_fraction_multiple=sum(len(x)>1 for x in cbs)/NS,all_crossing_quantiles=quant([v for x in cbs for v in x])))
   def bracket_common(z):
    # z shape (5,4) or (5,4,N); use finite bracket sign changes, including exact zeros.
    d=z[:-1]-z[1:]
    hit=np.isfinite(d[:,:-1])&np.isfinite(d[:,1:])&((d[:,:-1]*d[:,1:])<=0)
    return np.all(hit,axis=0)
   both=[]
   for ob,idx in [('ratio',1),('binder',2)]:
    z=np.array([[base_stats[L,arm,B0,b][0][ob] for b in BETAS] for L in LS])
    bt=np.array([[base_stats[L,arm,B0,b][idx] for b in BETAS] for L in LS])
    bc=bracket_common(bt);observed=bracket_common(z)
    joint.append(dict(arm=arm,B=B0,observable=ob,observed_common_brackets=np.flatnonzero(observed).tolist(),bootstrap_fraction_common=float(np.any(bc,axis=0).mean()),per_bracket_fraction=bc.mean(1).tolist()))
    both.append(bc)
   joint.append(dict(arm=arm,B=B0,observable='both',bootstrap_fraction_common=float(np.any(both[0]&both[1],axis=0).mean()),per_bracket_fraction=(both[0]&both[1]).mean(1).tolist()))
   for ob in ('ratio','log_B_trace'):
    y=np.array([[base_stats[L,arm,B0,b][0]['ratio'] if ob=='ratio' else np.log(B0*base_stats[L,arm,B0,b][0]['means']['trace']) for b in BETAS] for L in LS])
    for hold in ([21],[13]):
     tr=[L for L in LS if L not in hold];fits.append(dict(arm=arm,B=B0,observable=ob,holdout=hold,models=[fit_predict(y,tr,hold,k) for k in ('analytic','crossover','scaling')]))
   for L in LS:
    for key in ('capacity','chi_top','chi_fixed'):
     y=[base_stats[L,arm,B0,b][0]['means'][key] for b in BETAS];i=int(np.argmax(y));peaks.append(dict(L=L,arm=arm,B=B0,observable=key,beta=BETAS[i],value=y[i],interior=i not in (0,len(BETAS)-1),values=y))
 out=dict(scope='actual independent full B16/B64 and five sizes, two density protocols; beta0 multinomial null; finite root block ratios and finite-size fits, no critical exponent admission',groups=groups,nulls=nulls,aux=aux,crossings=cr,joint_crossings=joint,heldout_fits=fits,peaks=peaks,bootstrap_seed=760991,bootstrap_reps=NS,paths=sum(a['paths'] for a in aud),backups=sum(a['backups'] for a in aud),raw_bytes=sum(a['raw_bytes'] for a in aud),candidate_samples=sum(a['candidate_samples'] for a in aud))
 dump(S/'CRITICAL76.json',out)
 print('COUNTS',out['paths'],out['backups'],out['candidate_samples'],out['raw_bytes'],len(groups),len(cr),len(fits),flush=True)
 for arm in (0,1):
  for B0 in BS:
   for beta in BETAS:
    q=[g for g in groups if g['arm']==arm and g['B']==B0 and g['beta']==beta]
    print('PROFILE',arm,B0,beta,'Btrace',[round(B0*g['means']['trace'],4) for g in q],'R',[round(g['ratio'],4) for g in q],'U',[round(g['binder'],4) for g in q],flush=True)
 for q in aux:
  if q['kind']=='pulse':print('PULSE',q['L'],q['ci'],flush=True)
if __name__=='__main__':main()
