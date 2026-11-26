"""Actual block-law deviations from an exact independent-sum probability RG."""
from base61 import *
from scipy.special import logsumexp
from scipy.stats import binom
LS=[13,15,17,19,21];BT=[0.,10/3,16.];JS=np.array([-2.,-1.,-.5,0.,.5,1.,2.]);SCALES=[1,2,4,8]
def W(x,J=JS):
 x=np.ravel(x);return logsumexp(x[:,None]*np.asarray(J)[None,:],axis=0)-np.log(len(x))
def cumul(x):
 x=np.ravel(x);m=x.mean();z=x-m;v=np.mean(z*z)
 return np.array([m,v,np.mean(z**3),np.mean(z**4)-3*v*v])
def aggregate(y,h):return y.reshape(len(y),8//h,h).sum(-1)/np.sqrt(h)
def one(train,test):
 center=train.mean(0);scale=float(np.std(train-center));assert scale>1e-12
 a=(train-center)/scale;b=(test-center)/scale
 result=[];errors=[]
 for h in SCALES:
  x=aggregate(b,h);u=aggregate(a,h);wi=h*W(a,JS/np.sqrt(h));ws=h*W(b,JS/np.sqrt(h));wa=W(x);ca=cumul(x);c1=cumul(b);pred=c1*np.power(h,1-np.arange(1,5)/2)
  row=dict(h=h,empirical_W=wa.tolist(),iid_train_W=wi.tolist(),iid_same_sample_W=ws.tolist(),W_train_error=float(np.sqrt(np.mean((wa-wi)**2))),W_same_error=float(np.sqrt(np.mean((wa-ws)**2))),cumulants=ca.tolist(),iid_cumulants=pred.tolist(),train_cumulants=cumul(u).tolist(),gaussian_W= (JS*ca[0]+.5*ca[1]*JS**2).tolist())
  if h<8:
   A=x[:,::2].ravel();B0=x[:,1::2].ravel();z=(A+B0)/np.sqrt(2);joint=W(z);prod=W(A,JS/np.sqrt(2))+W(B0,JS/np.sqrt(2));iid=2*W(x,JS/np.sqrt(2))
   cv=float(np.mean((A-A.mean())*(B0-B0.mean())));res=abs(np.var(z)-.5*(np.var(A)+np.var(B0))-cv);assert res<1e-12
   # Exact identity of actual next block with paired sum, not independence.
   exact=float(abs(W(aggregate(b,2*h))-joint).max());assert exact<1e-12
   row.update(adjacent_cov=cv,variance_identity_error=res,exact_block_error=exact,joint_W=joint.tolist(),independent_marginals_W=prod.tolist(),iid_marginal_W=iid.tolist(),dependence_rmse=float(np.sqrt(np.mean((joint-prod)**2))),marginal_asymmetry_rmse=float(np.sqrt(np.mean((prod-iid)**2))))
  result.append(row)
 return dict(training_center=center.tolist(),training_scale=scale,blocks=result)
def old_witnesses():
 O=B/'mqft_restore72_20261126';R=read(O/'SCALE_RESULTS72.json');assert len(R['witnesses'])==112
 arr={};inputs={str((O/'SCALE_RESULTS72.json').relative_to(B)):sha(O/'SCALE_RESULTS72.json')}
 for L in LS:
  p=O/f'SCALE_L{L}.npz';assert sha(p)==read(O/f'AUDIT_L{L}.json')['array_sha256'];inputs[str(p.relative_to(B))]=sha(p)
  with np.load(p) as z:arr[L]={k:z[k] for k in z.files}
 rows=[]
 for w in R['witnesses']:
  L=w['L'];A=arr[L];rg=np.digitize(np.maximum(abs(np.arange(L*L)//L-L//2),abs(np.arange(L*L)%L-L//2)),[2,4,7]);t=w['t'];vals=[]
  for rep in (w['rep1'],w['rep2']):
   i=np.flatnonzero(np.all(A['meta']==[w['core'],w['arm'],rep],axis=1));assert len(i)==1;i=int(i[0])
   c=np.bincount(rg[A['actions'][i,:t]],minlength=4);p=np.bincount(rg,weights=A['p'][i,t],minlength=4);z=A['marks'][i,:t]
   assert c.tolist()==w['counts'];vals.append(dict(count=c.tolist(),p=p.tolist(),mark=float(z.sum()),mark_bin=int(np.digitize(z.mean(),[-.2,.2])),prob_bin=int(np.digitize(p[0],[.1,.4])),lag_bin=int(np.digitize(z[-4:].mean(),[-.2,.2]))))
  a,b=vals;tv=.5*np.sum(abs(np.array(a['p'])-b['p']));assert abs(tv-w['tv'])<1e-12
  rows.append(dict(L=L,core=w['core'],arm=w['arm'],t=t,reps=[w['rep1'],w['rep2']],tv=float(tv),values=vals,exact_mark_equal=a['mark']==b['mark'],mark_bin_equal=a['mark_bin']==b['mark_bin'],mark_prob_bin_equal=a['mark_bin']==b['mark_bin'] and a['prob_bin']==b['prob_bin'],all_bin_equal=all(a[k]==b[k] for k in ('mark_bin','prob_bin','lag_bin'))))
 return dict(rows=rows,inputs=inputs,counts={k:sum(x[k] for x in rows) for k in ('exact_mark_equal','mark_bin_equal','mark_prob_bin_equal','all_bin_equal')},max_tv=float(max(x['tv'] for x in rows)),max_remaining_tv=float(max((x['tv'] for x in rows if x['all_bin_equal']),default=0.)))
def main():
 out=[];inputs={}
 for L in LS:
  f=S/f'PATHS_L{L}.npz';assert sha(f)==read(S/f'AUDIT_L{L}.json')['array_sha256'];inputs[str(f.relative_to(B))]=sha(f)
  with np.load(f) as z:d={k:z[k] for k in z.files}
  for arm in (0,1):
   for bi,beta in enumerate(BT):
    m=d['meta'];i=(m[:,2]==arm)&(m[:,3]==bi);train=i&(m[:,0]==0);test=i&(m[:,0]==1);assert train.sum()==test.sum()==32
    p=d['prob'][:,8:16,0];r=d['counts'][:,8:16,0]/4-p;mark=d['marks'][:,8:16];rr=one(r[train],r[test]);mm=one(mark[train],mark[test])
    bracket=[]
    for h in SCALES:
     xx=aggregate(r[test],h);bb=p[test]*(1-p[test])/4;target=bb.reshape(32,8//h,h).mean(-1)
     # Population martingale second moment identity tested as finite sample difference, not forced equality.
     bracket.append(dict(h=h,observed_second_moment=float(np.mean(xx**2)),mean_predictable_bracket=float(target.mean()),difference=float(np.mean(xx**2-target))))
    control=[]
    if beta==0:
     # Exact conditional binomial null for each root; different masks need not yield a common IID law.
     for core in range(8):
      ix=test&(m[:,1]==core);q=float(p[ix][0,0]);assert abs(p[ix]-q).max()<1e-13
      for h in SCALES:
       k=np.arange(4*h+1);w=binom.pmf(k,4*h,q);x=(k/4-h*q)/np.sqrt(h)
       c2=float(w@(x*x));c4=float(w@(x**4)-3*c2*c2);ref2=q*(1-q)/4;ref4=q*(1-q)*(1-6*q*(1-q))/(64*h)
       assert max(abs(c2-ref2),abs(c4-ref4))<1e-12
       control.append(dict(core=core,h=h,p=q,var=c2,k4=c4,variance_error=abs(c2-ref2),k4_error=abs(c4-ref4)))
    out.append(dict(L=L,arm=arm,beta=beta,central_residual=rr,mark=mm,raw_martingale_bracket=bracket,exact_uniform_controls=control))
  print('RG',L,flush=True)
 # Algebraic independent-sum map for an explicit non-Gaussian Rademacher law, exact cumulant eigenvalues.
 ref=[]
 for n in (1,2,4,8,16,32,64):
  k=np.arange(n+1);p=binom.pmf(k,n,.5);x=(2*k-n)/np.sqrt(n);mu=float(p@x);v=float(p@(x*x));k4=float(p@(x**4)-3*v*v)
  assert max(abs(mu),abs(v-1),abs(k4+2/n))<1e-12
  ref.append(dict(block=n,k2=v,k4=k4,expected_k4=-2/n,W=np.log(p@np.exp(x[:,None]*JS)).tolist(),gaussian_W=(.5*JS**2).tolist()))
 old=old_witnesses()
 dump(S/'RG77.json',dict(scope='independent-sum reference map versus actual normalized late block laws; no native fixed-point admission; temporal blocks not spatial RG',strata=out,old_witnesses=old,inputs=inputs,source_J=JS.tolist(),cumulant_eigenvalues={str(k):2**(1-k/2) for k in range(1,9)},reference_control=ref))
 print('OLD',old['counts'],old['max_remaining_tv'],flush=True)
if __name__=='__main__':main()
