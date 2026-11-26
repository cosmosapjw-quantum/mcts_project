"""Independent pairwise covariance, finite null identities, raw and prediction audit."""
from base61 import *
import gzip,itertools
LS=[13,15,17,19,21];BETAS=[0.,10/3,8.,16.]
def load(p):
 with gzip.open(p,'rt') as f:return list(map(json.loads,f))
def run():
 q=read(S/'CRITICAL76.json');err=collections.defaultdict(float);n=0;raw_n=0
 def record(key,a,b):
  e=float(np.max(abs(np.asarray(a)-np.asarray(b))));err[key]=max(err[key],e)
 # Finite exact enumerated n=4 Bernoulli checks, including asymmetric distributions.
 for p in (.1,.3,.5,.9):
  E4=0.;EM=0.
  for xs in itertools.product([0.,1.],repeat=4):
   x=np.array(xs);w=p**x.sum()*(1-p)**(4-x.sum());d=[(a-b)**2 for i,a in enumerate(x) for b in x[i+1:]]
   m22=((x[0]-x[1])**2*(x[2]-x[3])**2+(x[0]-x[2])**2*(x[1]-x[3])**2+(x[0]-x[3])**2*(x[1]-x[2])**2)/12
   E4+=w*(np.mean(np.square(d))/2-6*m22);EM+=w*m22
  record('k4_exact',E4,p*(1-p)*(1-6*p*(1-p)));record('mu22_exact',EM,(p*(1-p))**2)
 for L in LS:
  au=read(S/f'AUDIT_L{L}.json');assert sha(S/f'SCALE_L{L}.npz')==au['array_sha256']
  with np.load(S/f'SCALE_L{L}.npz') as z:d={k:z[k] for k in z.files}
  M=d['meta'];N=L*L;y,x=np.divmod(np.arange(N),L)
  mode=np.array([np.cos(np.pi*(x+.5)/L),np.cos(np.pi*(y+.5)/L),np.cos(2*np.pi*(x+.5)/L),np.cos(2*np.pi*(y+.5)/L)])
  regs=read(S/f'REGISTRATION_L{L}.json');plan={}
  for k in ('base','cpuct','pulse'):
   plan.update({j['id']:j for j in read(S/'plans'/f'L{L}_{k}.json')['jobs']})
  for row in au['raw_receipts']:
   j=plan[row['id']];core=int(j['position_id'][1:3]);arm=int(j['position_id'].endswith('dense'));b=j['spec']['budget'];beta=j['spec']['root_law']['beta_p'][0];c=j['spec']['c_puct'];h=j['spec']['root_law'].get('logit_schedule',[0.])[-1]
   if core!=0 or j['replicate']!=0:continue
   f=B/row['path'];assert sha(f)==row['sha256'];hd,rr,e=load(f);tr=rr['trace'];es=sorted(tr['replicas'][0]['events'],key=lambda v:v['root_selection']['clock']);assert hd['job']==j and e['backups']==b
   z=np.flatnonzero(np.all(M==[core,arm,b,beta,c,h,0],axis=1));assert len(z)==1;z=z[0]
   X=np.zeros(N);A=np.zeros(N);p0=None
   for event in es:
    l=event['root_selection']['law'];act=np.array(l['actions']);pr=np.array(l['p']);X[event['actions'][0]]+=1/b;A[act]+=pr/b
    if beta==0:
     assert np.max(abs(pr-1/len(pr)))<1e-15
   record('raw_occupancy',X,d['X'][z]);record('raw_drift',A,d['A'][z]);raw_n+=1
  for g in [a for a in q['groups'] if a['L']==L]:
   b=g['B'];beta=g['beta'];arm=g['arm']
   for r in g['roots']:
    z=np.flatnonzero(np.all(M[:,:6]==[r['core'],arm,b,beta,1.4,0.],axis=1));assert len(z)==4
    X=d['X'][z];A=d['A'][z];D=X-A;legal=d['board'][z[0]]==0;K=legal.sum();p=legal/K
    # C = sum over unordered replica differences / n(n-1).
    dif=np.array([X[i]-X[j] for i,j in itertools.combinations(range(4),2)])
    C=np.einsum('ri,rj->ij',dif,dif)/12
    record('trace',np.trace(C),r['trace'])
    vv=(dif@mode.T)**2;v=vv.sum(0)/12
    record('low',v[:2].mean(),r['low']);record('high',v[2:].mean(),r['high'])
    record('zero_mode',C.sum(),0.);record('cov_sum_rows',C.sum(0),0.)
    def trace_pair(a):
     return sum(np.dot(a[i]-a[j],a[i]-a[j]) for i,j in itertools.combinations(range(4),2))/12
    record('drift',trace_pair(A),r['drift']);record('martingale',trace_pair(D),r['martingale']);record('cross',trace_pair(X)-trace_pair(A)-trace_pair(D),r['cross'])
    # Null derived by finite one-draw categorical second/fourth moments.
    c0=(np.diag(p)-p[:,None]*p[None,:])/b;v0=np.diag(mode@c0@mode.T)
    record('null_trace',np.trace(c0),r['null_trace']);record('null_mode',v0[:2].mean(),r['null_low']);record('null_mode',v0[2:].mean(),r['null_high'])
    f0=mode[0,legal];mu=f0.mean();k=np.mean((f0-mu)**4)-3*np.mean((f0-mu)**2)**2
    record('null_k4',k/b**3,r['null_k4'])
    assert sum(r['pair_counts'])==K*K
    record('radial_mass',sum(r['radial_sum']),0.);record('radial_null_mass',sum(r['radial_null_sum']),0.)
    n+=1
 assert n==640 and raw_n==100
 # Rebuild heldout predictions independently from saved linear coefficients; fit grid selection not rerun.
 for a in q['heldout_fits']:
  arm,b,ob,L=a['arm'],a['B'],a['observable'],a['holdout'][0]
  y=np.array([next(g['ratio'] if ob=='ratio' else np.log(b*g['means']['trace']) for g in q['groups'] if (g['L'],g['arm'],g['B'],g['beta'])==(L,arm,b,beta)) for beta in BETAS])
  for f in a['models']:
   assert f is not None
   if f['model']=='analytic':A=np.array([[1.,beta/16,(beta/16)**2,17/L-1] for beta in BETAS])
   else:A=np.array([[1.,np.tanh((beta-f['beta_center'])*(L/17)**f['scaling_y']/f['width']),17/L-1] for beta in BETAS])
   pred=A@np.array(f['coeff']);record('heldout_prediction',pred,f['pred']);record('heldout_actual',y,f['actual']);record('heldout_RMSE',np.sqrt(np.mean((pred-y)**2)),f['test_rmse'])
 assert max(err.values())<1e-10,dict(err)
 out=dict(status='PASS_SCOPED',root_group_checks=n,raw_paths_independently_reduced=raw_n,heldout_models=3*len(q['heldout_fits']),errors=dict(err),scope='Same-agent separate pairwise covariance, direct trace reductions, finite Bernoulli enumeration, reference categorical covariance and heldout model predictions; independent native subagent not run.')
 dump(S/'INDEPENDENT76.json',out);print(out,flush=True)
if __name__=='__main__':run()
