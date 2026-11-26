"""Separate verification of finite marked law and local saddle; direct exact count convolution."""
from base61 import *
from scipy.special import logsumexp
from scipy.optimize import brentq
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv,sys
from marked70 import cgf,saddle
A=np.load(S/'MARKED70.npz');r=read(S/'PHYSICS_RESULTS.json');pm=read(S/'PLOT_MANIFEST70.json')
Y=A['counts'];Z=A['marks'];F=A['conditional_pmf'];meta=A['meta'];G=A['grid'];hist=A['history_pmf'];labels=[x['label'] for x in r['families']]
assert np.all(Y.sum(-1)==4) and np.all(np.abs(Z)<=Y+1e-7)
assert np.max(abs(F.sum(-1)-1))<1e-11 and np.max(abs(hist.sum(-1)-1))<1e-11
assert len(r['raw_receipts'])==288 and all(sha(B/x['path'])==x['sha256'] for x in r['raw_receipts']+r['prior_raw_receipts'])
a=read(S/'AUDIT69.json');assert a['native_paths']==256 and a['rapfi_paths']==32 and a['backups']==36864
# Known analytic Bernoulli and binomial coefficient verify independent replica calculation.
bchecks=[]
for p in (.1,.5,.8):
 for M in (8,32,128):
  q=saddle(np.array([1-p,p]),M,0)
  if q is None:continue
  exact=binom.logpmf(q['k'],M,p)
  assert abs(exact-q['log_prob'])<1e-10
  bchecks.append(abs(exact-q['log_prob']))
# Independent literal sums at moderate sources; directional third/fourth via finite differences
# on tilted third and second cumulants rather than unstable fourth difference of W.
errors=[];curves=[];gammas=[]
J=np.linspace(-4,4,41);v=np.array([1.,-1.,.5])
for g in range(5):
 ix=meta[:,0]==g;X=np.column_stack([Y[ix].sum(1)[:,:2],Z[ix].sum((1,2))])/128
 c=[];gg=[]
 for t in J:
  j=t*v;w,m,h,p=cgf(X,j)
  q=np.exp(X@j);q=q/q.sum();mm=(q[:,None]*X).sum(0)
  hh=sum(q[k]*np.outer(x-mm,x-mm) for k,x in enumerate(X));d=(X-mm)@v
  k3=float(q@(d**3));k4=float(q@(d**4)-3*(q@(d**2))**2)
  def third(t):
   _,m0,_,q0=cgf(X,t*v);return q0@(((X-m0)@v)**3)
  def second(t):
   _,_,h0,_=cgf(X,t*v);return v@h0@v
  dh=1e-4;e3=abs((second(t+dh)-second(t-dh))/(2*dh)-k3);e4=abs((third(t+dh)-third(t-dh))/(2*dh)-k4)
  err=max(abs(w-np.log(np.mean(np.exp(X@j)))),np.max(abs(m-mm)),np.max(abs(h-hh)),e3,e4)
  assert err<2e-8;errors.append(float(err));gg.append([float(m@v),float(j@m-w)])
  s=t*np.array([1.,-1.]);d0=G[:,:2]@s/128
  x0=X[:,:2]@s;logz=np.log(F[ix]@np.exp(d0)).sum(1);logf=32*np.log(F[ix,0]@np.exp(d0));logh=np.log(hist[ix]@np.exp(d0)).sum(1)
  c.append([float(logsumexp(x0)-np.log(len(x0))),float(logsumexp(logf)-np.log(len(logf))),float(logz.mean()),float(logh.mean())])
 curves.append(np.asarray(c));gammas.append(np.asarray(gg))
# Distinguish log of mixture from mean log conditional products; not autonomous closure.
def plot(name,fig,ids):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=150);fig.savefig(str(p)+'.pdf');plt.close(fig)
 pm['figures'].append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True);axs=axs.ravel()
for g,(ax,c) in enumerate(zip(axs,curves)):
 for k,l in enumerate(['empirical adaptive W','initial frozen mixture W','mean adaptive conditional log Z','mean reduced teacher-forced log Z']):ax.plot(J,c[:,k],label=l)
 ax.set_title(labels[g]);ax.set_xlabel('count source');ax.set_ylabel('dimensionless W or mean log Z')
axs[-1].axis('off');handles,lab=axs[0].get_legend_handles_labels();axs[-1].legend(handles,lab,loc='center',fontsize=9)
plot('adaptive_path_vs_conditional_generators',fig,['L16','Q21'])
fig,ax=plt.subplots(figsize=(8,5))
for g,c in enumerate(gammas):ax.plot(c[:,0],c[:,1],label=labels[g])
ax.set(xlabel='directional marked mean v·x',ylabel='Gamma(x) on source ray',title='Empirical Legendre conjugate (source-ray restriction)');ax.legend(fontsize=8);plot('marked_legendre_conjugate',fig,['L16','Q21'])
fig,ax=plt.subplots(figsize=(8,5))
for g,l in enumerate(labels):
 ss=sorted([v for v in r['saddles'] if v['family']==g and v['M']==64],key=lambda z:z['k']);xs=[q['k']/64 for q in ss];ys=[-q['log_prob']/64 for q in ss];lead=[y-np.log(1+q['leading_rel'])/64 for y,q in zip(ys,ss)];nxt=[y-np.log(1+q['next_rel'])/64 for y,q in zip(ys,ss)]
 ax.plot(xs,ys,'o-',label=l);ax.plot(xs,lead,'x',color=ax.lines[-1].get_color());ax.plot(xs,nxt,'+',color=ax.lines[-1].get_color())
ax.set(xlabel='count per independent replica',ylabel='negative log mass per replica',title='Finite-replica action: exact (circle), Gaussian (x), next (+)');ax.legend(fontsize=8);plot('finite_replica_effective_action',fig,['L16','Q21'])
res=dict(status='verified_finite_empirical_law_and_lattice_convolution',max_direct_and_cumulant_check_error=max(errors),finite_source_cases=len(errors),max_binomial_log_mass_error=max(bchecks),binomial_checks=len(bchecks),full_native_replay_sha=sha(S/'AUDIT69.json'),raw_receipts_verified=576,count_curves=[c.tolist() for c in curves],gamma_curves=[c.tolist() for c in gammas],scope='Separate implementation of direct summation and analytic binomial reference; not native independent subagent; no autonomous joint generator verified.')
dump(S/'INDEPENDENT70.json',res)
pm['sources']['INDEPENDENT70.json']=sha(S/'INDEPENDENT70.json');dump(S/'PLOT_MANIFEST70.json',pm)
print('INDEPENDENT', {k:v for k,v in res.items() if not k.endswith('curves')}, 'figures',len(pm['figures']),flush=True)
