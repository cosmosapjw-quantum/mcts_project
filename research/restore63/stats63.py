from base61 import *
from scipy.stats import norm
from fractions import Fraction as Fr
R=B/'mqft_restore62_20261126'
def main():
 a=read(S/'AUDIT63.json');assert sha(S/'FINAL63.npz')==a['final_sha']
 with np.load(S/'FINAL63.npz') as d:z={k:d[k] for k in d.files}
 m=z['meta'];tags=list(z['tags']);arms=list(z['arms']);rng=np.random.default_rng(630401)
 def get(tag,arm='z'):
  ix=np.flatnonzero((m[:,1]==tags.index(tag))&(m[:,6]==arms.index(arm)))
  ix=ix[np.lexsort((m[ix,5],m[ix,4]))]
  n=int(m[ix,4].max()+1);r=int(m[ix,5].max()+1)
  assert len(ix)==n*r and np.array_equal(m[ix,4],np.repeat(np.arange(n),r)) and np.array_equal(m[ix,5],np.tile(np.arange(r),n))
  return {k:z[k][ix].reshape((n,r)+z[k].shape[1:]) for k in ('C','G','F','D','M','V','logw')}
 def stat(x):
  n=len(x);b=rng.integers(0,n,(10000,n));s=x[b].mean(1);return dict(mean=x.mean(0).tolist(),ci95=np.quantile(s,[.025,.975],axis=0).tolist())
 def fisher(q):
  M=q['M'];V=q['V'];n,r=M.shape[:2];assert abs(M-(q['G'][...,None]*q['G'][...,None,:]-q['F'])).max()<1e-9
  # Fixed geometry: actual independent seeds, mean bracket / npath^2.
  mean=M.mean((0,1));se=np.sqrt(V.sum((0,1)))/(n*r);se_emp=np.sqrt(M.var(1,ddof=1).sum(0)/r)/n
  assert np.all(se>0);Z=mean/se;ci=np.stack([mean-1.96*se,mean+1.96*se]);crit=float(norm.ppf(1-.05/(2*6)));sim=np.stack([mean-crit*se,mean+crit*se])
  ix=np.triu_indices(3);b=M.mean(1);block_se=b.std(0,ddof=1)/np.sqrt(n);rr=rng.integers(0,n,(10000,n))
  boot=(b[rr].mean(1)-mean)/block_se;cv=float(np.quantile(abs(boot[:,ix[0],ix[1]]).max(1),.95))
  block_sim=np.stack([mean-cv*block_se,mean+cv*block_se]);rootz=M.sum(1)/np.sqrt(V.sum(1))
  return dict(n_roots=n,seeds=r,residual=stat(b),bracket_se=se.tolist(),empirical_fixed_root_se=se_emp.tolist(),normal_approx95=ci.tolist(),bonferroni_critical=crit,bonferroni95=sim.tolist(),Z=Z.tolist(),block_maxT_critical=cv,block_simultaneous95=block_sim.tolist(),root_standardized=rootz.tolist(),mean_F=q['F'].mean((0,1)).tolist(),mean_GG=(q['G'][...,None]*q['G'][...,None,:]).mean((0,1)).tolist(),mean_G=stat(q['G'].mean(1)),residual_second_moment_minus_bracket=stat((M*M-V).mean(1)),scope='fixed-root conditional bracket normal approximation; block bootstrap is descriptive geometry generalization; family 6 unique symmetric entries, no exact finite coverage guarantee')
 f0=get('fisher0');old=get('response0w4');new=fisher(f0);baseline=fisher(old);combined=fisher({k:np.concatenate([old[k],f0[k]],axis=1) for k in old})
 l=get('legacy0');lo=get('legacy0','am35');hi=get('legacy0','ap35')
 C=l['C'][:,:,0];G=l['G'][:,:,0];p1=np.exp(l['logw'][:,:,4]);p2=np.exp(l['logw'][:,:,5]);base=np.empty_like(C);base[:,::2]=C[:,1::2].mean(1,keepdims=True);base[:,1::2]=C[:,::2].mean(1,keepdims=True)
 fd=(hi['C'][:,:,0]-lo['C'][:,:,0]).mean(1)/.7
 score=((C-C.mean(1,keepdims=True))*(G-G.mean(1,keepdims=True))).sum(1)/(C.shape[1]-1)
 rw=(C*(p2-p1)).mean(1)/.7;center=((C-base)*(p2-p1)).mean(1)/.7
 cross=((C-base)*G).mean(1)
 ess=lambda x:float(x.sum()**2/(x*x).sum())
 legacy=dict(n_roots=32,seeds=32,fd=stat(fd),score=stat(score),reweight=stat(rw),centered=stat(center),crossfit_score=stat(cross),fd_minus_score=stat(fd-score),fd_minus_rw=stat(fd-rw),fd_minus_center=stat(fd-center),weights_mean=[float(p1.mean()),float(p2.mean())],ess=[ess(p1),ess(p2)],weights_conditional_means=[p1.mean(1).tolist(),p2.mean(1).tolist()],R62=read(R/'PHYSICS_RESULTS.json')['legacy_replication'][-1])
 # Exact two-step three-category path enumeration: verifies E[M]=0 and E[M²]=E[V].
 p=[Fr(1,3)]*3;q=[[Fr(1,2),Fr(1,3),Fr(1,6)],[Fr(1,6),Fr(1,2),Fr(1,3)],[Fr(1,3),Fr(1,6),Fr(1,2)]]
 for i in range(3):
  for j in range(3):
   em=em2=ev=Fr(0)
   def step(g,p,a):
    x=[Fr(int(a==k))-p[k] for k in range(3)]
    return g[i]*x[j]+g[j]*x[i]+int(i==j)*x[i]-p[i]*x[j]-p[j]*x[i]
   for a in range(3):
    g=[Fr(int(a==k))-p[k] for k in range(3)]
    v0=sum(p[b]*step([Fr(0)]*3,p,b)**2 for b in range(3));v1=sum(q[a][b]*step(g,q[a],b)**2 for b in range(3))
    for b in range(3):
     d=step([Fr(0)]*3,p,a)+step(g,q[a],b);w=p[a]*q[a][b];em+=w*d;em2+=w*d*d;ev+=w*(v0+v1)
   assert em==0 and em2==ev
 out=dict(fisher=dict(old=baseline,new=new,combined_secondary=combined),legacy=legacy,exact_martingale_fraction_check=True)
 dump(S/'PHYSICS_RESULTS.json',out)
 for label,r in out['fisher'].items():print('FISHER',label,'mean',r['residual']['mean'],'Z',r['Z'],'fixedCIaa',np.asarray(r['normal_approx95'])[:,0,0].tolist(),'blockCIaa',np.asarray(r['residual']['ci95'])[:,0,0].tolist(),'simultaneous',r['block_simultaneous95'],flush=True)
 print('LEGACY',{k:v for k,v in legacy.items() if k not in ('R62','weights_conditional_means')},flush=True)
if __name__=='__main__':main()
