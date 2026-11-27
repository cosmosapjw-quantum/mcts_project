"""Correct R86 R49 startup grouping without modifying historical artifacts."""
from base61 import *
from scipy.stats import t
O=B/'mqft_restore86_20261127';R=B/'mqft_restore77_20261127';old=read(O/'SCALE86.json');J=np.array([.15,-.1,0.]);I=np.eye(3)
def ci(a):
 x=np.array(a);m=x.mean(0);h=t.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return [float(m),float(m-h),float(m+h)]
def rootci(a,core):return ci([a[core==i].mean() for i in np.unique(core)])
rows=[];errors=collections.defaultdict(float);startup=collections.Counter();maxdef=0.;min_eig=0.;sample_waves=0
def chk(k,a,b,tol=1e-10):
 e=float(abs(np.asarray(a)-np.asarray(b)).max());errors[k]=max(errors[k],e);assert e<tol,(k,e)
for L in (13,15,17,19,21):
 a=read(S/f'AUDIT_L{L}.json');f=S/f'SECTOR_L{L}.npz';assert sha(f)==a['sector_sha']
 with np.load(f) as z:d={k:z[k] for k in z.files}
 p=R/f'PATHS_L{L}.npz';assert sha(p)==read(R/f'AUDIT_L{L}.json')['array_sha256']
 with np.load(p) as z:old_d={k:z[k] for k in z.files}
 idx={tuple(m):i for i,m in enumerate(d['meta'])}
 ids=[]
 for m in old_d['meta']:
  q=m.copy();q[3]=[0,1,3][q[3]];ids.append(idx[tuple(q)])
 P=d['p'][ids];C=np.eye(3)[d['selected'][ids]];W=P.reshape(-1,16,4,3);C4=C.reshape(-1,16,4,3).sum(2);P4=W.mean(2)
 chk('R86_wave_mean',P4,old_d['prob']);chk('R86_wave_counts',C4,old_d['counts'])
 D=C-P;V=P[:,:,:,None]*I[None,None]-P[:,:,:,None]*P[:,:,None,:];V4=V.reshape(-1,16,4,3,3).sum(2)
 oldV=4*(P4[:,:,:,None]*I[None,None]-P4[:,:,:,None]*P4[:,:,None,:]);delta=oldV-V4
 delta0=W-P4[:,:,None,:];chk('covariance_defect',delta,np.einsum('ntki,ntkj->ntij',delta0,delta0));assert abs(delta[:,1:]).max()<1e-10
 maxdef=max(maxdef,float(np.trace(delta[:,0],axis1=-2,axis2=-1).max()));min_eig=min(min_eig,float(np.linalg.eigvalsh(delta[:,0]).min()))
 for bi,beta in enumerate((0.,10/3,16.)):
  for co in (0,1):
   for arm in (0,1):
    m=old_d['meta'];ix=(m[:,0]==co)&(m[:,2]==arm)&(m[:,3]==bi);assert ix.sum()==32
    pr=P[ix];c=C[ix];v=V[ix];d0=D[ix];roots=m[ix,1]
    startup[(bi,co)]+=int((abs(pr[:,:4]-pr[:,0:1]).max((1,2))>1e-12).sum())
    # t=0,1 singleton and t=2,3 pair; t>=4 width-four.
    chk('startup_pair',pr[:,2:4],pr[:,2:3])
    chk('later_wave_constant',pr[:,4:].reshape(-1,15,4,3),pr[:,4:].reshape(-1,15,4,3)[:,:,:1])
    a0=next(x for x in old['strata'] if [x[k] for k in ('L','arm','beta','cohort')]==[L,arm,beta,co])
    out=[]
    for h in (1,2,4,8,16):
     B0=4*h;z=d0.reshape(-1,16//h,B0,3).sum(2);vv=v.reshape(-1,16//h,B0,3,3).sum(2)
     sq=(z*z).sum(2)/B0;qv=np.trace(vv,axis1=-2,axis2=-1)/B0
     prior=next(b for b in a0['blocks'] if b['h']==h);chk('unchanged_compensated',sq.mean(),prior['martingale_trace_ci'][0])
     ee=[]
     for j0 in (J,-J):
      ell=(c@j0)-np.log(pr@np.exp(j0));w=np.exp(ell.reshape(-1,16//h,B0).sum(2));ee.append(dict(J=j0.tolist(),mean_ci=rootci(w.mean(1),roots),ESS=float(w.sum()**2/(w*w).sum())))
     out.append(dict(h=h,martingale_trace_ci=rootci(sq.mean(1),roots),predictable_trace_ci=rootci(qv.mean(1),roots),difference_ci=rootci((sq-qv).mean(1),roots),prior_predictable_trace=prior['predictable_trace_ci'][0],exponential=ee))
    rows.append(dict(L=L,cohort=co,arm=arm,beta=beta,blocks=out))
 sample_waves+=len(P)*16
assert len(rows)==60 and sample_waves==30720
q=dict(status='CORRECTED_STARTUP_SCOPE',R86_scale_sha=sha(O/'SCALE86.json'),paths=1920,waves=sample_waves,regular_width4_waves=28800,beta0_startup_uniform_paths=640,nonuniform_startup_sector_paths={str(k):int(v) for k,v in startup.items()},max_startup_trace_overestimate=maxdef,min_defect_eigenvalue=min_eig,errors=dict(errors),strata=rows,variance_interval_failures=sum(not b['difference_ci'][1]<=0<=b['difference_ci'][2] for a in rows for b in a['blocks']),exponential_interval_failures=sum(not e['mean_ci'][1]<=1<=e['mean_ci'][2] for a in rows for b in a['blocks'] for e in b['exponential']),scope='R49 startup waves t0 singleton, t1 singleton, t2-3 pair. R86 averaging probabilities preserves D but overestimates bracket by sum(p_i-pbar)(p_i-pbar)^T; replace logMGF by sum actual per-draw logMGF. R86 B128 selector/memory results unaffected; R86 archived raw and report immutable. Exact one-step theorem and regular width-four waves retained.')
dump(S/'ERRATUM86.json',q)
print('ERRATUM',q['nonuniform_startup_sector_paths'],maxdef,q['variance_interval_failures'],q['exponential_interval_failures'],dict(errors),flush=True)
