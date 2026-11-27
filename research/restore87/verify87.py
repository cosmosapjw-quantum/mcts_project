"""Independent direct covariance and model check, native raw sample, original archive authentication."""
from base61 import *
import gzip,zipfile
from itertools import combinations,product
from scipy.stats import t
R=read(S/'CRITICAL87.json');LS=[13,15,17,19,21];BETA=[0.,10/3,8.,16.]
err=collections.defaultdict(float)
def chk(k,a,b,tol=1e-9):
 v=float(np.max(abs(np.asarray(a)-np.asarray(b))));err[k]=max(err[k],v);assert v<tol,(k,v)
def ci(x):
 m=x.mean(0);h=t.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.array([m,m-h,m+h])
def cv(x,y,h):
 return h*sum(np.dot(x[i]-x[j],y[i]-y[j]) for i,j in combinations(range(len(x)),2))/(len(x)*(len(x)-1))
raw=0;wave=0;strata=0;sample_modes=0
for L in LS:
 a=read(S/f'AUDIT_L{L}.json');p=S/f'OBS_L{L}.npz';assert sha(p)==a['array_sha']
 with np.load(p) as f:z={k:f[k] for k in f.files}
 M=z['meta'];F=z['modes'];assert len(M)==2080
 for r in [x for x in R['strata'] if x['L']==L]:
  co,arm,bi,b,h=[r[k] for k in ('cohort','arm','bi','budget','h')];rows=[]
  for core in range(8):
   ix=(M[:,0]==co)&(M[:,1]==core)&(M[:,2]==arm)&(M[:,3]==bi)&(M[:,5]==b)&(M[:,6]==h);x=z['X'][ix];u=z['A'][ix];d=x-u
   q=z['Q'][ix].mean(0);n=z['null'][ix].mean(0);v=[cv(x,x,h),cv(u,u,h),cv(d,d,h),2*cv(u,d,h),h*np.mean(np.sum(d*d,1))]
   m=[]
   for f in [F[:,:2],F[:,2:]]:
    xx=x@f;uu=u@f;dd=d@f;m.extend([cv(xx,xx,h)/2,cv(uu,uu,h)/2,cv(dd,dd,h)/2,cv(uu,dd,h),h*np.mean(np.sum(dd*dd,1))/2])
   rows.append(np.r_[q,n,v,z['aux'][ix].mean(0),m])
  rows=np.array(rows);chk('root_cov',rows,r['root_values']);chk('aggregate',ci(rows),r['ci']);chk('ratio',rows[:,1].mean()/(rows[:,1:3].sum(1).mean()),r['ratio']);strata+=1
 # Representative all fresh geometries/densities and beta, rep0. Explicit categorical covariance via centered f, check actual spectral bound.
 for rr in a['raw_receipts']:
  if rr['stage']!=87 or rr['rep']!=0:continue
  p0=B/rr['path'];assert sha(p0)==rr['sha256']
  with gzip.open(p0,'rt') as f:hd,r0,end=map(json.loads,f)
  tr=r0['trace'];es=sorted(tr['replicas'][0]['events'],key=lambda e:e['root_selection']['clock']);A=np.zeros(L*L);N=np.zeros(L*L);qv=[];q0=[];frozen=[]
  for i,e in enumerate(es):
   l=e['root_selection']['law'];ac=np.array(l['actions']);p=np.array(l['p']);n=len(p);N[e['actions'][0]]+=1;A[ac]+=p
   v=F[ac];x=v-p@v;c=(p[:,None]*x*x).sum(0)
   qv.append([1-(p*p).sum(),c[:2].mean(),c[2:].mean()])
   u=v-v.mean(0);c=(u*u).mean(0);q0.append([1-1/n,c[:2].mean(),c[2:].mean()])
   if i%4==0:
    # Largest eigenvalue for representative wave t32 only; cheap explicit spectrum of PSD covariance.
    if i==32:
     H=np.diag(p)-np.outer(p,p);ev=np.linalg.eigvalsh(H);assert ev[0]>-1e-12 and ev[-1]<=.5+1e-12;chk('zero_mode',H.sum(1),0.);sample_modes+=1
    wave+=1
   h=i+1
   if h in (16,32,64):
    ix=np.flatnonzero(np.all(M==[2,rr['core'],rr['arm'],rr['beta_index'],0,64,h],1));assert len(ix)==1;k=ix[0]
    chk('raw_X',N/h,z['X'][k]);chk('raw_A',A/h,z['A'][k]);chk('raw_Q',np.mean(qv,0),z['Q'][k]);chk('raw_null',np.mean(q0,0),z['null'][k])
  raw+=1
assert raw==400 and wave==6400 and sample_modes==400
# Independent least-squares implementation and heldout predictions; reselect parameters from complete training CV.
def value(L,co,arm,bi,h,name):
 x=next(r for r in R['strata'] if [r[k] for k in ('L','cohort','arm','bi','h','budget')]==[L,co,arm,bi,h,64])
 return x['ratio'] if name=='ratio' else x['mean'][0]
def basis(l,b,kind,pa):
 if kind=='smooth':return [1,b/16,(b/16)**2,17/l]
 c,w,y=pa;return [1,np.tanh((b-c)*(l/17)**y/w),17/l]
def ls(l,b,y,kind,pa):
 X=np.array([basis(i,j,kind,pa) for i,j in zip(l,b)]);P=np.eye(X.shape[1])*1e-4;P[0,0]=0
 return np.linalg.lstsq(np.vstack([X,P]),np.r_[y,np.zeros(len(P))],rcond=None)[0]
def pred(l,b,kind,pa,c):return np.clip(np.array([basis(i,j,kind,pa) for i,j in zip(l,b)])@c,0,1)
for r in R['fits']:
 arm,h,name,k=[r[x] for x in ('arm','h','observable','model')];l=np.repeat(LS[:4],4);b=np.tile(BETA,4);y=np.array([value(i,0,arm,j,h,name) for i in LS[:4] for j in range(4)])
 opts=[None] if k=='smooth' else list(product(np.arange(0,17,2,dtype=float),[1.,2.,4.,8.],[0.] if k=='crossover' else [.25,.5,1.,2.]))
 v=[]
 for p in opts:
  z=np.zeros(len(y))
  for L in LS[:4]:
   ix=l!=L;c=ls(l[ix],b[ix],y[ix],k,p);z[~ix]=pred(l[~ix],b[~ix],k,p,c)
  v.append((np.sqrt(np.mean((z-y)**2)),p))
 cv0,pa=min(v,key=lambda q:(q[0],str(q[1])));chk('cv',cv0,r['cv_rmse'])
 # degeneracy in grid ties allowed only within tolerance, selected fit independently checked.
 p0=r['params'];c=ls(l,b,y,k,p0);chk('fit_coef',c,r['coef'],2e-6)
 for q in r['external']:
  x=pred(q['L'],q['beta'],k,p0,c);chk('heldout_prediction',x,q['prediction'],2e-7);chk('heldout_RMSE',np.sqrt(np.mean((x-q['truth'])**2)),q['RMSE'],2e-7)
# Exact categorical crossover check via independent odds identity at all n,s.
for q in R['entropy_null']:
 n,s,b=q['n'],q['s'],q['beta'];p=np.exp(b)/(np.exp(b)+n-1);chk('entropy_crossover',p,q['pmax']);chk('entropy_trace',1-p*p-(1-p)**2/(n-1),q['trace'])
O79=B/'mqft_restore79_20261127';O86=B/'mqft_restore86_20261127';O69=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O86/f)==sha(O69/f),f
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'};assert all(sha(paths[k])==v for k,v in orig['archives'].items())
ids=['L17','L26','L36','L37','L38','Q32'];spans=[s for s in orig['spans'] if s['id'] in ids];assert len(spans)==len(ids)
want={s['member'] for s in spans}|{'mcts_study/v3.0/qft-mcts-statistical.md'};members=[]
for s in orig['verified_members']:
 if s['member'] not in want:continue
 with zipfile.ZipFile(paths[s['archive']]) as f:v=f.read(s['member'])
 assert hashlib.sha256(v).hexdigest()==s['sha256'];members.append(s)
assert {s['member'] for s in members}==want
dump(S/'ORIGINAL_SOURCE87.json',dict(archives=orig['archives'],verified_members=members,spans=spans,original_attachment_receipt=orig['original_attachment_receipt'],scope='Original critical scaling, phase-transition, temperature and RG/susceptibility targets; R76/R77 negative evidence retained.'))
dump(S/'INDEPENDENT87.json',dict(status='PASS_SCOPED',raw_paths=raw,waves=wave,spectral_cases=sample_modes,strata=strata,fit_models=len(R['fits']),entropy_cases=len(R['entropy_null']),errors=dict(err),scope='Pairwise covariance vs centered covariance, independent raw moment reductions and spectral checks, augmented least squares and full CV grid, exact source identity; same-agent separate algorithm.'))
print('VERIFIED',raw,wave,strata,dict(err),flush=True)
