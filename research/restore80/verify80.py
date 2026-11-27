"""Independent direct moments, Wick/Gauss-Hermite coefficient, finite convolution and raw-path crosschecks."""
from base61 import *
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from scipy.stats import binom
from scipy.fft import rfft2,irfft2
import gzip
A=read(S/'AUDIT80.json');R=read(S/'ACTION80.json');F=read(S/'FORCED_REPLAY80.json');U=read(S/'RNG_AUDIT80.json')
assert sha(S/'MARKED80.npz')==A['array_sha']==R['input_sha'] and F['status']=='PASS_SCOPED'
with np.load(S/'MARKED80.npz') as z:D={k:z[k] for k in z.files}
M=D['meta'];X=D['X'];Y=D['Y'];err=collections.defaultdict(float)
def chk(k,a,b,tol=1e-10):
 v=float(np.max(abs(np.asarray(a)-np.asarray(b))));err[k]=max(err[k],v);assert v<tol,(k,v)
def direct(x,j):
 z=x@j;m=z.max();q=np.exp(z-m);q/=q.sum();w=m+np.log(np.exp(z-m).mean());mu=np.einsum('i,ij->j',q,x);v=x-mu
 H=sum(p*np.outer(a,a) for p,a in zip(q,v))
 return w,mu,H,q
def gh_coeff(x,q,mu,H):
 # Independent Cholesky (not symmetric whitening) and Gaussian quadrature.
 L=np.linalg.cholesky(H);d=np.linalg.solve(L,(x-mu).T).T;z,w=hermgauss(4);z=z*np.sqrt(2);w=w/np.sqrt(np.pi)
 c=0.;e4=e33=0.
 for a,wa in zip(z,w):
  for b,wb in zip(z,w):
   v=np.array([a,b]);d0=d@v;k3=q@(d0**3);k4=q@(d0**4)-3*(q@(d0**2))**2
   e4+=wa*wb*k4;e33+=wa*wb*k3*k3
 return e4/24-e33/72,e4,e33
def convolve_direct(Y,q,n,k):
 # Exact finite-support coefficient by repeated nonnegative direct shifts, no Fourier method.
 lo=Y.min(0);Z=Y-lo;hi=Z.max(0);y=np.array(k)-n*lo
 pts,ix=np.unique(Z,axis=0,return_inverse=True);p=np.bincount(ix,weights=q)
 a=np.ones((1,1),np.float64)
 for t in range(n):
  z=np.zeros(tuple((t+1)*hi+1))
  for v,w in zip(pts,p):z[v[0]:v[0]+a.shape[0],v[1]:v[1]+a.shape[1]]+=w*a
  a=z
 assert np.all(y>=0) and np.all(y<np.array(a.shape))
 chk('direct_mass_normalization',a.sum(),1.,1e-11);return a[tuple(y)]
# Recompute all empirical generating functions and inverse Hessian, full moment checks.
for g,group in enumerate(R['sources']):
 for row in group['cases']:
  j=np.array(row['J'])
  for name,ix in [('train',(M[:,0]==0)),('matched',(M[:,0]==1)&(M[:,2]<32)),('novel',(M[:,0]==1)&(M[:,2]>=32))]:
   x=X[ix&(M[:,1]==g)];w,mu,H,q=direct(x,j);v=row[name]
   chk('W',w,v['W']);chk('mean',mu,v['phi']);chk('covariance',H,v['H']);chk('Gamma',np.dot(j,mu)-w,v['gamma']);chk('KL',sum(q*np.log(q*len(q))),v['KL']);chk('inverse_H',np.array(v['inverse_H'])@H,np.eye(2))
# Independent coefficient all resolved cases; nonnegative convolution at M8 center.
cases=0;dp=[]
for row in R['saddles']:
 if row['status']!='OK':continue
 g=['g0','g3','Rapfi'].index(row['family']);ix=(M[:,1]==g)&(M[:,0]==row['cohort'])&(M[:,2]<32);y=Y[ix];s=np.array(row['s']);n=row['M'];k=np.array(row['k']);w,mu,H,q=direct(y,s)
 chk('saddle_mean',mu,k/n,1e-8);chk('saddle_H',H,row['H'],1e-8);chk('saddle_K',w,row['K']);chk('gamma',s@(k/n)-w,row['gamma']);c,e4,e33=gh_coeff(y,q,mu,H)
 chk('wick_coefficient',c,row['coeff'],1e-8);chk('wick_e4',e4,row['gaussian_e4'],1e-7);chk('wick_e33',e33,row['gaussian_e33'],1e-7)
 assert row['alias_bound']<=row['wrapped_mass']*1e-12 and row['fft_normalization_error']<1e-9
 if n==8 and row['J0']==[0.,0.]:
  v=convolve_direct(y,q,n,k);p=row['wrapped_mass'];chk('direct_convolution_rel',v/p,1.,1e-8);dp.append(dict(family=row['family'],cohort=row['cohort'],M=n,k=k.tolist(),direct=v,wrapped=p,relative_error=abs(v/p-1)))
 cases+=1
# Independent factorized binomial exact 2D distribution (each copy 2 independent Bernoulli draws).
controls=[]
for p,q in [(.2,.7),(.5,.5),(.8,.1)]:
 for n in [8,32,128]:
  k=np.rint(n*np.array([p,q])).astype(int);target=k/n
  if min(target)<=0 or max(target)>=1:continue
  pp=np.array([p,q]);s=np.log(target/(1-target))-np.log(pp/(1-pp));v=np.array([[0,0],[0,1],[1,0],[1,1]])
  prob=np.prod(np.where(v,pp,1-pp),axis=1);qw=prob*np.exp(v@s);Z=qw.sum();qw/=Z;H=np.diag(target*(1-target));c,e4,e33=gh_coeff(v,qw,target,H)
  # Independent 1D correction sums for product law.
  sig=target*(1-target);c1=np.sum((1-6*sig)/(8*sig)-5*(1-2*target)**2/(24*sig))
  chk('binomial_Wick',c,c1,1e-10)
  exact=float(np.prod(binom.pmf(k,n,target)));leading=1/(2*np.pi*n*np.sqrt(np.linalg.det(H)));nextp=leading*(1+c/n);d=convolve_direct(v,qw,n,k);chk('binomial_direct_mass',d/exact,1.,1e-10)
  controls.append(dict(p=p,q=q,M=n,exact=exact,leading_rel=leading/exact-1,corrected_rel=nextp/exact-1))
# Raw subset of 24 fresh + 24 prior histories per family? fixed 4 roots x2 reps each cohort.
raw=0
for i,r in enumerate(A['raw_receipts']):
 if r['core'] not in (0,1,2,3) or r['rep'] not in (0,1):continue
 f=B/r['path'];assert sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:d=json.load(z)
 j=d['input'];es=d['trace']['events'];a=np.array([e['actions'][0] for e in es]);z=np.array([e['root_value'] for e in es])
 lp=[]
 for e in es:
  l=e['root_selection']['law'];lp.append(math.log(l['p'][l['picked']]))
 chk('raw_action',-sum(lp),D['action'][i],1e-9);chk('raw_X',[np.mean(np.isin(a,j['sources'])),np.mean(z)],X[i]);chk('raw_Y',[sum(np.isin(a,j['sources'])),sum(z>0)],Y[i]);raw+=1
assert raw==48 and len(dp)==6
# Verify forced receipts independently byte-for-byte and action sequences.
for q in F['cases']:
 assert sha(B/q['path'])==q['sha256'] and sha(B/q['source'])==q['source_sha']
 with gzip.open(B/q['path'],'rt') as z:a=json.load(z)['trace']
 with gzip.open(B/q['source'],'rt') as z:b=json.load(z)['trace']
 for field in ('actions','root_value','leaf_value','parent_hashes'):
  assert [x[field] for x in a['events']]==[x[field] for x in b['events']]
assert len(F['cases'])==24 and U['streams']==768 and U['independent_uniforms']==98304
dump(S/'INDEPENDENT80.json',dict(status='PASS_SCOPED',source_cases=3*len(R['source_grid'])*3,saddles_checked=cases,direct_convolutions=dp,binomial_controls=controls,raw_paths=raw,forced_replays=24,errors=dict(err),scope='Independent direct log sums and moment covariance, Cholesky/Gauss-Hermite cumulant coefficient, 6 nonnegative direct convolutions and factorized binomial controls. No native independent subagent, no rigorous floating roundoff bound.'))
print('VERIFIED',cases,'direct',len(dp),'raw',raw,dict(err),flush=True)
