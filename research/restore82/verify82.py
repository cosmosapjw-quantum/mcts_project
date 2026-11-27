"""Separate direct empirical score and CGF evaluation; raw prefix feature and original source checks."""
from base61 import *
import gzip,zipfile,subprocess
from itertools import product
A=read(S/'AUDIT82.json');F=read(S/'MODEL82.json');R=read(S/'TRANSFER82.json');C=read(S/'COMPAT82.json')
assert A['array_sha']==F['input_sha']==R['array_sha']==sha(S/'POPULATION82.npz') and R['model_sha']==sha(S/'MODEL82.json') and R['prediction_sha']==sha(S/'PREDICTIONS82.npz')
with np.load(S/'POPULATION82.npz') as z:D={k:z[k] for k in z.files}
with np.load(S/'PREDICTIONS82.npz') as z:P={k:z[k] for k in z.files}
M=D['meta'];mp=P['meta'];assert len(M)==4032 and len(mp)==12672 and len(set(map(tuple,mp)))==12672
models=F['model_names'];grid=np.array(F['source_grid']);scale=np.array([1.,.5]);err=collections.defaultdict(float)
def chk(k,a,b,tol=1e-9):
 v=float(np.max(abs(np.asarray(a)-np.asarray(b))));err[k]=max(err[k],v);assert v<tol,(k,v)
def weight(c):
 return np.array([1/(len(set(c))*sum(c==i)) for i in c])
def standard(a,b,w):
 m=np.sum(a*w[:,None],0);s=np.maximum(np.sqrt(np.sum(w[:,None]*(a-m)**2,0)),1e-8);return np.mean(((a-b)/s)**2,1)
# independent direct outcome probabilities and pairwise Euclidean distances.
pred_count=0
for g in range(3):
 ix=(M[:,0]==0)&(M[:,1]==g);c=M[ix,2];w=weight(c);fm=F['families'][g]
 for k,h in fm['bandwidth'].items():
  assert h==min(F['bandwidth_grid'],key=lambda v:(fm['cv_es'][k][str(v)],-v))
 for fold in fm['folds']:
  assert not set(fold['train_roots'])&set(fold['test_roots']) and len(fold['test_roots'])==8 and len(fold['train_roots'])==40
 for j in np.flatnonzero((M[:,0]>0)&(M[:,1]==g)&(M[:,4]==0)):
  dr=standard(D['root'][ix],D['root'][j],w);dh=standard(D['history'][ix],D['history'][j],w)
  for i,k in enumerate(models):
   f=np.flatnonzero(np.all(mp==np.r_[M[j],i],axis=1));assert len(f)==1;f=int(f[0])
   if k in ('pool','prefix_pool'):p=w.copy()
   elif k=='ply':p=w*(D['root'][ix,0]==D['root'][j,0]);p/=p.sum()
   else:
    d=dr if k=='root' else dh if k=='prefix_kernel' else .5*(dr+dh);h=fm['bandwidth'][k];z=np.exp(-(d-d.min())/(2*h*h))*w;p=.98*z/z.sum()+.02*w
   assert abs(p.sum()-1)<1e-12 and min(p)>=0
   pred=(D['A'][j]+.75*D['V'][ix]) if k.startswith('prefix') or k=='root_history' else D['X'][ix]
   assert np.all((pred[:,0]>=-1e-12)&(pred[:,0]<=1+1e-12)) and np.all(abs(pred[:,1])<=1+1e-12) and abs(pred[:,0]*128-np.rint(pred[:,0]*128)).max()<1e-10
   y=D['X'][j];d=(pred-y)*scale;dist=np.linalg.norm(d,axis=1);dist2=np.sqrt(np.sum(((pred[:,None]-pred[None,:])*scale)**2,2))
   chk('energy',p@dist-.5*p@dist2@p,P['energy'][f]);e=np.exp(pred@grid.T);chk('W',np.log(p@e),P['W'][f]);chk('tilted_ESS',(p@e)**2/((p*p)@(e*e)),P['tilt_ess'][f]);mu=p@pred;chk('mu',mu,P['mu'][f]);cov=sum(q*np.outer(v-mu,v-mu) for q,v in zip(p,pred));chk('cov',cov,P['cov'][f])
   pred_count+=1
assert pred_count==1008
# Check all external aggregate scores and W without using model functions.
for a in R['results']:
 g=['g0','g3','Rapfi'].index(a['family']);gr=a['group'];roots=np.unique(M[(M[:,0]==gr)&(M[:,1]==g),2])
 for k,v in enumerate(a['models']):
  ix=(mp[:,0]==gr)&(mp[:,1]==g)&(mp[:,5]==k)
  ew=np.array([P['energy'][ix&(mp[:,2]==c)].mean() for c in roots]);chk('energy_root_means',ew,v['root_energy']);w=np.array([np.log(np.exp(P['W'][ix&(mp[:,2]==c)]).mean(0)) for c in roots]);chk('root_W',w,v['root_W']);t=np.array([np.log(np.exp(D['X'][(M[:,0]==gr)&(M[:,1]==g)&(M[:,2]==c)]@grid.T).mean(0)) for c in roots]);chk('truth_W',t,v['truth_W']);chk('W_RMSE',np.sqrt(np.mean((w[:,:25]-t[:,:25])**2)),v['W_RMSE'])
# Recompute every tilted empirical mixture via direct conditional second moments.
mix_count=0
for a in R['mixtures']:
 g=['g0','g3','Rapfi'].index(a['family']);gr=a['group'];ix=(M[:,0]==gr)&(M[:,1]==g);x=D['X'][ix];c=M[ix,2];w=weight(c)
 for v,J in zip(a['source'],grid):
  r=w*np.exp(x@J);q=r/r.sum();mu=q@x;second=np.einsum('i,ij,ik->jk',q,x,x);cov=second-np.outer(mu,mu);within=np.zeros((2,2));between=np.zeros((2,2))
  for c0 in set(c):
   sel=c==c0;t=q[sel].sum();p=q[sel]/t;xx=x[sel];mm=p@xx;within+=t*(np.einsum('i,ij,ik->jk',p,xx,xx)-np.outer(mm,mm));between+=t*np.outer(mm-mu,mm-mu)
  chk('mixture_W',np.log(r.sum()),v['W']);chk('mixture_cov',cov,v['cov']);chk('mixture_within',within,v['within']);chk('mixture_between',between,v['between']);assert v['jensen_gap']>=-1e-12;mix_count+=1
assert mix_count==261
# independent raw reduction, one seed per orbit/family, R80 only for training duplicates
raw=0
for i,r in enumerate(A['raw_receipts']):
 if r['rep']!=0 or r['stage']==79:continue
 f=B/r['path'];assert sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:a=json.load(z)
 j=a['input'];tr=a['trace'];es=tr['events'];src=j['sources'];p=np.asarray(tr['cuts'][0]['prior']);v=tr['evaluations'][0]['clean' if r['family']==2 else 'value'];acts=np.array([e['actions'][0] for e in es]);marks=np.array([e['root_value'] for e in es]);hit=np.isin(acts,src)
 chk('raw_X',[hit.mean(),marks.mean()],D['X'][i]);chk('raw_A',[hit[:32].sum()/128,marks[:32].sum()/128],D['A'][i]);chk('raw_V',[hit[32:].mean(),marks[32:].mean()],D['V'][i])
 chk('raw_root_core',[len(j['moves']),v,np.log(p[src].sum()),np.log(p.max())],D['root'][i,:4])
 n=np.bincount(acts[:32],minlength=len(p));w=np.bincount(acts[:32],weights=marks[:32],minlength=len(p));Q=np.divide(w,n,out=np.zeros(len(p)),where=n>0);m=[]
 for e in es[:33]:
  l=e['root_selection']['law'];ac=np.array(l['actions']);pp=np.array(l['p']);m.append(pp[np.isin(ac,src)].sum())
 l=es[32]['root_selection']['law'];ac=np.array(l['actions']);p1=np.array(l['p']);a0,b0=[p1[np.flatnonzero(ac==s)[0]] for s in src]
 chk('raw_prefix',[hit[:32].mean(),marks[:32].mean(),marks[:32].var(),entropy(n[n>0]/32)/math.log(np.count_nonzero(p)),np.count_nonzero(n)/np.count_nonzero(p),Q[src[0]],Q[src[1]],np.log(a0+b0),np.log(a0/b0),(hit[:32].sum()-sum(m[:32]))/32],D['history'][i],1e-7);raw+=1
assert raw==312
for r in C['cases']:assert sha(B/r['path'])==r['sha256'] and sha(B/r['source'])==r['source_sha']
assert C['status']=='PASS_SCOPED' and len(C['cases'])==12
# Independent RNG replay on fresh histories only.
p=B/'mqft_restore61_20261126/mqft_trained_rng61';q=subprocess.run([str(p),str(S/'RNG_INPUT82.json')],capture_output=True,text=True,timeout=20,check=True);u=json.loads(q.stdout);assert u['streams']==1024 and u['independent_uniforms']==131072 and u['max_error']<1e-12
u.update(binary_sha=sha(p),input_sha=sha(S/'RNG_INPUT82.json'));dump(S/'RNG_AUDIT82.json',u)
# Original archive and relevant member identities.
O79=B/'mqft_restore79_20261127';O81=B/'mqft_restore81_20261127';O69=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O81/f)==sha(O69/f),f
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'};assert all(sha(paths[k])==v for k,v in orig['archives'].items())
ids=['L16','Q32','G07'];spans=[s for s in orig['spans'] if s['id'] in ids];want={s.get('guide_source_member',s['member']) if s['source_status']=='GUIDE_LABEL_ONLY' else s['member'] for s in spans}|{'mcts_study/effective-action-mcts.md','mcts_study/v3.0/qft-mcts-statistical.md'}
members=[]
for s in orig['verified_members']:
 if s['member'] not in want:continue
 with zipfile.ZipFile(paths[s['archive']]) as f:v=f.read(s['member'])
 assert hashlib.sha256(v).hexdigest()==s['sha256'];members.append(s)
assert {s['member'] for s in members}==want
dump(S/'ORIGINAL_SOURCE82.json',dict(archives=orig['archives'],verified_members=members,spans=spans,original_attachment_receipt=orig['original_attachment_receipt'],scope='Original effective-action/statistical-field and source-response objectives, exact R79 source identity; conditional finite law and root mixture are not physical temperature or original full amplitude.'))
dump(S/'INDEPENDENT82.json',dict(status='PASS_SCOPED',raw_paths=raw,independent_predictions=pred_count,aggregate_predictions=len(mp),tilted_mixture_cases=mix_count,compatibility_cases=12,rng=u,errors=dict(err),scope='Same-agent independent direct weights, explicit pairwise energy and moment covariance, raw prefix/endpoint reduction and aggregate recalculation; no independent native subagent.'))
print('VERIFIED',raw,pred_count,len(mp),mix_count,dict(err),'RNG',u,flush=True)
