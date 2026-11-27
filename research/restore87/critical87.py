"""Root-balanced finite-size observables, competing heldout models, and crossing diagnostics."""
from base61 import *
from scipy.stats import t as student
from itertools import product
LS=[13,15,17,19,21];BETA=[0.,10/3,8.,16.];HS=[16,32,64]
def ci(x):
 x=np.asarray(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.array([m,m-h,m+h]).tolist()
def pair_cov(x,y):
 return np.sum((x-x.mean(0))*(y-y.mean(0)))/(len(x)-1)
def load():
 out={};audit={}
 for L in LS:
  a=read(S/f'AUDIT_L{L}.json');p=S/f'OBS_L{L}.npz';assert sha(p)==a['array_sha']
  with np.load(p) as z:out[L]={k:z[k] for k in z.files}
  audit[L]=a
 return out,audit
def aggregate(d):
 rows=[]
 for L,z in d.items():
  M=z['meta'];F=z['modes']
  for co,arm,bi,budget,h in sorted(set(tuple(m[[0,2,3,5,6]]) for m in M)):
   idx=(M[:,0]==co)&(M[:,2]==arm)&(M[:,3]==bi)&(M[:,5]==budget)&(M[:,6]==h);r=[]
   for core in range(8):
    sel=idx&(M[:,1]==core);x=z['X'][sel];a=z['A'][sel];D=x-a;assert len(x)==(2 if co==2 else 4)
    q=z['Q'][sel].mean(0);nu=z['null'][sel].mean(0)
    raw=h*pair_cov(x,x);dr=h*pair_cov(a,a);innov=h*pair_cov(D,D);cross=2*h*pair_cov(a,D);unc=h*np.sum(D*D,axis=1).mean()
    assert abs(raw-dr-innov-cross)<1e-10
    assert q[0]<=nu[0]+1e-12 and min(q)>=-1e-12
    modes=[]
    for k in (0,2):
     f=F[:,k:k+2];v=(x@f);w=(a@f);u=v-w
     r0=h*pair_cov(v,v)/2;d0=h*pair_cov(w,w)/2;i0=h*pair_cov(u,u)/2;c0=h*pair_cov(w,u);u0=h*np.sum(u*u,axis=1).mean()/2
     assert abs(r0-d0-i0-c0)<1e-10
     modes.extend([r0,d0,i0,c0,u0])
    r.append(np.r_[q,nu,raw,dr,innov,cross,unc,z['aux'][sel].mean(0),modes])
   x=np.array(r);mu=x.mean(0);ratio=mu[1]/(mu[1]+mu[2])
   rows.append(dict(L=L,cohort=int(co),arm=int(arm),bi=int(bi),beta=(BETA+[math.log(L*L)])[bi],budget=int(budget),h=int(h),roots=8,replicates=2 if co==2 else 4,fields=['trace','low','high','null_trace','null_low','null_high','raw_trace','drift_trace','centered_innovation_trace','cross_trace','uncentered_innovation_trace','pmax','score_range','H32_pmax','raw_low','drift_low','innovation_low','cross_low','uncentered_low','raw_high','drift_high','innovation_high','cross_high','uncentered_high'],root_values=x.tolist(),mean=mu.tolist(),ci=ci(x),ratio=ratio))
 return rows
def get(rows,L,co,arm,bi,h,budget=64):
 a=[x for x in rows if (x['L'],x['cohort'],x['arm'],x['bi'],x['h'],x['budget'])==(L,co,arm,bi,h,budget)];assert len(a)==1;return a[0]
def target(a,name):return a['ratio'] if name=='ratio' else a['mean'][0]
def design(L,b,kind,params):
 L=np.asarray(L,dtype=float);b=np.asarray(b,dtype=float);x=b/16
 if kind=='smooth':return np.column_stack([np.ones(len(L)),x,x*x,17/L])
 bc,w,y=params;return np.column_stack([np.ones(len(L)),np.tanh((b-bc)*(L/17)**y/w),17/L])
def fit(L,b,z,kind,params):
 X=design(L,b,kind,params);r=np.eye(X.shape[1])*1e-8;r[0,0]=0.;return np.linalg.solve(X.T@X+r,X.T@z)
def predict(L,b,kind,params,c):return np.clip(design(L,b,kind,params)@c,0,1)
def select(rows,arm,h,name,kind):
 L=np.repeat(LS[:4],4);b=np.tile(BETA,4);z=np.array([target(get(rows,l,0,arm,k,h),name) for l in LS[:4] for k in range(4)])
 opts=[None] if kind=='smooth' else list(product(np.arange(0,17,2,dtype=float),[1.,2.,4.,8.],[0.] if kind=='crossover' else [.25,.5,1.,2.]))
 results=[]
 for p in opts:
  pred=np.zeros_like(z)
  for ell in LS[:4]:
   ix=L!=ell;c=fit(L[ix],b[ix],z[ix],kind,p);pred[~ix]=predict(L[~ix],b[~ix],kind,p,c)
  results.append((float(np.sqrt(np.mean((pred-z)**2))),p))
 cv,p=min(results,key=lambda q:(q[0],str(q[1])));c=fit(L,b,z,kind,p)
 out=dict(arm=arm,h=h,observable=name,model=kind,params=list(p) if p else None,coef=c.tolist(),cv_rmse=cv,grid_count=len(opts),external=[])
 for co in (1,2):
  bis=(0,1,3) if co==1 else (0,1,2,3)
  for name0,ll,ii in [('L21',LS[-1:],bis),('all_sizes',LS,bis)]+([('log_size',LS,(4,))] if co==2 else []):
   v=[get(rows,l,co,k,bi,h) for l in ll for k in [arm] for bi in ii];x=np.array([target(a,name) for a in v]);p0=predict([a['L'] for a in v],[a['beta'] for a in v],kind,p,c)
   out['external'].append(dict(cohort=co,scope=name0,RMSE=float(np.sqrt(np.mean((p0-x)**2))),truth=x.tolist(),prediction=p0.tolist(),L=[a['L'] for a in v],beta=[a['beta'] for a in v]))
 return out
def bracket(v):
 # Only strict sign change or an exactly zero endpoint; retain bracket labels, not invented roots.
 out=[]
 for i in range(3):
  if v[i]==0 or v[i+1]==0 or v[i]*v[i+1]<0:out.append(i)
 return out
def crossing(rows):
 rng=np.random.default_rng(871200);out=[]
 for arm in (0,1):
  for h in HS:
   for name in ('ratio','trace'):
    a=np.array([[get(rows,L,2,arm,k,h)['root_values'] for k in range(4)] for L in LS])
    def values(x):return x[...,0] if name=='trace' else x[...,1]/(x[...,1]+x[...,2])
    u=values(a.mean(2));br=[bracket(u[i+1]-u[i]) for i in range(4)];common=set(br[0]).intersection(*map(set,br[1:]))
    count=0;cross=np.zeros(4,int)
    for _ in range(2000):
     ix=rng.integers(0,8,8);v=values(a[:,:,ix,:].mean(2));b=[bracket(v[i+1]-v[i]) for i in range(4)];count+=bool(set(b[0]).intersection(*map(set,b[1:])));cross+=np.array([bool(q) for q in b])
    out.append(dict(arm=arm,h=h,observable=name,brackets=br,common_brackets=sorted(common),bootstrap_common_fraction=count/2000,bootstrap_adjacent_fractions=(cross/2000).tolist()))
 return out
def main():
 d,a=load();rows=aggregate(d);fits=[]
 for arm,h,n,k in product((0,1),HS,('ratio','trace'),('smooth','crossover','critical')):fits.append(select(rows,arm,h,n,k))
 cross=crossing(rows)
 gates=[]
 for arm,n,co in product((0,1),('ratio','trace'),(1,2)):
  a0=[x for x in fits if x['arm']==arm and x['h']==64 and x['observable']==n]
  errs={x['model']:next(z['RMSE'] for z in x['external'] if z['scope']=='L21' and z['cohort']==co) for x in a0}
  gates.append(dict(arm=arm,observable=n,cohort=co,errors=errs,critical_win=errs['critical']<min(errs['smooth'],errs['crossover'])))
 # Frozen actual H32 scores vs the actual adaptive H32 pmax, fresh only.
 frozen=[]
 for L in LS:
  q=a[L]['frozen'];assert len(q)==32;z=d[L];m=z['meta']
  for arm in (0,1):
   for bi in range(5):
    x=[r['values'][bi] for r in q if r['arm']==arm];ix=(m[:,0]==2)&(m[:,2]==arm)&(m[:,3]==bi)&(m[:,6]==64)
    frozen.append(dict(L=L,arm=arm,bi=bi,beta=(BETA+[math.log(L*L)])[bi],frozen_mean=np.mean(x,0).tolist(),actual_H32_mean_pmax=float(z['aux'][ix,2].mean()),frozen_count=len(x)))
 entropy_null=[]
 for n in (100,1000,10000,100000,1000000):
  for s in (-4,-2,0,2,4):
   beta=np.log(n-1)+s;p=1/(1+np.exp(-s));v=1-p*p-(1-p)**2/(n-1)
   entropy_null.append(dict(n=n,s=s,beta=beta,pmax=p,trace=v))
 dump(S/'CRITICAL87.json',dict(strata=rows,fits=fits,crossings=cross,gates=gates,frozen=frozen,entropy_null=entropy_null,scope='Descriptive noncritical/critical-type prediction models trained only R76 L13–19; fixed beta heldout R77/R87 L21; log-size schedule separately. Not RG fixed point evidence.'))
 print('GATES',gates,flush=True);print('CROSS',cross,flush=True)
if __name__=='__main__':main()
