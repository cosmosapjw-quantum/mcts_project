"""Exact conditional multinomial compensator and dyadic predictable variance."""
from base61 import *
from itertools import product
from scipy.stats import t as student
O=B/'mqft_restore77_20261127';LS=[13,15,17,19,21];BETA=[0.,10/3,16.];J=np.array([.15,-.10,0.])
H=np.array([np.bincount(a,minlength=3) for a in product(range(3),repeat=4)])
I=np.eye(3)
def ci(x):
 a=np.array(x);m=a.mean(0);h=student.ppf(.975,len(a)-1)*a.std(0,ddof=1)/np.sqrt(len(a));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def rm(a,c):return np.stack([a[c==i].mean(0) for i in np.unique(c)])
def calc(c,p,root,hs):
 assert c.shape==p.shape and c.ndim==3 and c.shape[-1]==3 and np.all(c.sum(-1)==4) and np.all(p>0)
 err=float(abs(p.sum(-1)-1).max());assert err<1e-12
 d=c-4*p;v=4*(p[:,:,:,None]*I[None,None]-p[:,:,:,None]*p[:,:,None,:]);assert abs(v.sum(-1)).max()<1e-10
 assert abs(d.sum(-1)).max()<1e-10
 out=[]
 for h in hs:
  D=d.reshape(len(c),-1,h,3).sum(2);A=c.reshape(len(c),-1,h,3).sum(2);V=v.reshape(len(c),-1,h,3,3).sum(2);S=D[:,:,:,None]*D[:,:,None,:];sq=np.trace(S,axis1=-2,axis2=-1)/(4*h);qv=np.trace(V,axis1=-2,axis2=-1)/(4*h)
  # Average blocks within path then root, retain fixed root ensemble variance.
  delta=rm((sq-qv).mean(1),root);z=rm(sq.mean(1),root);pred=rm(qv.mean(1),root)
  raw=rm((A/(np.sqrt(4*h))).mean(1),root)
  # raw count across paths + time blocks: variance of count/sqrt(4h) around common empirical mean.
  a0=A/np.sqrt(4*h);rawvar=float(np.mean(np.sum((a0-a0.mean((0,1)))**2,-1)))
  # exact conditional compensator produces nonnegative exponential martingale, not IID count law.
  e=[]
  for j in (J,-J):
   ell=c@j-4*np.log(p@np.exp(j));ell=ell.reshape(len(c),-1,h).sum(2);wt=np.exp(ell);mean=rm(wt.mean(1),root);e.append(dict(J=j.tolist(),mean_ci=ci(mean),ESS=float(wt.sum()**2/np.sum(wt*wt)),n_blocks=int(wt.size),max_weight=float(wt.max())))
  out.append(dict(h=h,martingale_trace_ci=ci(z),predictable_trace_ci=ci(pred),difference_ci=ci(delta),raw_count_population_trace=rawvar,mean_cov=(S/(4*h)).mean((0,1)).tolist(),mean_predictable_cov=(V/(4*h)).mean((0,1)).tolist(),exponential=e))
 return out
def main():
 out=[];receipts=[];arrays={};err=collections.defaultdict(float);wave_count=0
 for L in LS:
  a=read(O/f'AUDIT_L{L}.json');f=O/f'PATHS_L{L}.npz';assert sha(f)==a['array_sha256'];arrays[str(f.relative_to(B))]=sha(f);receipts+=a['raw_receipts']
  with np.load(f) as z:d={k:z[k] for k in z.files}
  m=d['meta'];c=d['counts'];p=d['prob']
  # enumerate the 81 fine sector strings, including duplicate histograms, for each actual wave.
  p0=p.reshape(-1,3);w=np.exp(np.log(p0)@H.T);means=w@H;second=np.einsum('ki,ij,il->kjl',w,H,H);cov=second-means[:,:,None]*means[:,None,:]
  exact=4*(p0[:,:,None]*I[None]-p0[:,:,None]*p0[:,None,:])
  for k,v in [('mass',abs(w.sum(1)-1).max()),('mean',abs(means-4*p0).max()),('cov',abs(cov-exact).max()),('logMGF',abs(np.log(w@np.exp(H@J))-4*np.log(p0@np.exp(J))).max())]:err[k]=max(err[k],float(v))
  wave_count+=len(p0)
  for arm in (0,1):
   for bi,beta in enumerate(BETA):
    for cohort in (0,1):
     ix=(m[:,2]==arm)&(m[:,3]==bi)&(m[:,0]==cohort);assert ix.sum()==32
     root=m[ix,1];pp=p[ix];cc=c[ix];board=d['board'][ix]
     if beta==0:
      rad=np.maximum(abs(np.arange(L*L)//L-L//2),abs(np.arange(L*L)%L-L//2));sector=np.digitize(rad,[2,4])
      legal=board==0;fixed=legal@np.eye(3)[sector]/legal.sum(1)[:,None];assert abs(pp-fixed[:,None]).max()<1e-12
      # Frozen conditional root iid control at any block length: exact mgf and covariance, not pooled root law.
      for h in [1,2,4,8,16]:
       exact_mgf=(4*h)*np.log(fixed@np.exp(J));e=np.log(pp[:,:h]@np.exp(J)).sum(1)*4;err['beta0_block_logMGF']=max(err['beta0_block_logMGF'],float(abs(e-exact_mgf).max()))
     out.append(dict(L=L,arm=arm,beta=beta,cohort=cohort,roots=8,paths=32,blocks=calc(cc,pp,root,[1,2,4,8,16])))
  print('SCALE',L,flush=True)
 assert len(out)==60 and len(receipts)==1920 and wave_count==30720 and max(err.values())<1e-9
 dump(S/'SCALE86.json',dict(strata=out,arrays=arrays,raw_receipts=receipts,waves=wave_count,enumerated_sector_strings=wave_count*81,errors=dict(err),scope='Actual R76/R77 radial-sector count laws. Exact conditional multinomial and martingale identities; empirical dyadic second moments and pointwise root-block intervals, no native CLT or physical spatial RG.'))
 print('SCALE COMPLETE',wave_count,dict(err),flush=True)
if __name__=='__main__':main()
