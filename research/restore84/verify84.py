"""Independent direct feature, ridge/predictive score, source and aggregate audit."""
from base61 import *
import zipfile,gzip
from scipy.spatial.distance import cdist
from scipy.stats import t as student
A=read(S/'AUDIT84.json');L0=read(S/'LANDSCAPE84.json');F=read(S/'MODEL84.json');R=read(S/'TRANSFER84.json')
assert sha(S/'LANDSCAPE84.json')==F['landscape_sha']==R['landscape_sha'] and sha(S/'MODEL84.json')==R['model_sha'] and sha(S/'FRESH84.npz')==A['array_sha']==R['fresh_array_sha']
O82=B/'mqft_restore82_20261127';assert sha(O82/'POPULATION82.npz')==F['development_sha']==read(O82/'AUDIT82.json')['array_sha']
with np.load(O82/'POPULATION82.npz') as f:D={k:f[k] for k in ('meta','X','root')}
with np.load(S/'FRESH84.npz') as f:T={k:f[k] for k in f.files}
grid=np.asarray(F['source_grid']);off=[0,48,72,104];core=np.array([off[m[0]]+m[2] for m in D['meta']]);feat={};err=collections.defaultdict(float)
def chk(k,a,b,tol=1e-9):
 v=float(np.max(abs(np.asarray(a)-np.asarray(b))));err[k]=max(err[k],v);assert v<tol,(k,v)
def board_win(m,a,L):
 b=np.zeros((L,L),int)
 for k,v in enumerate(m+[a]):b.flat[v]=1+k%2
 y,x=divmod(a,L)
 for dy,dx in ((1,0),(0,1),(1,1),(1,-1)):
  n=1
  for sign in (-1,1):
   yy=y+sign*dy;xx=x+sign*dx
   while 0<=yy<L and 0<=xx<L and b[yy,xx]==b[y,x]:n+=1;yy+=sign*dy;xx+=sign*dx
  if n>=5:return True
 return False
nchild=0
for a in L0['rows']:
 g=a['family'];i=off[a['group']]+a['core'];p=np.asarray(a['p'],float);L=int(np.sqrt(len(p)));moves=a['moves'];assert len(set(moves))==len(moves)
 ch=sorted(a['children'],key=lambda v:v['a']);legal=[j for j in range(L*L) if j not in moves];assert [v['a'] for v in ch]==legal
 q=p[legal];q/=q.sum();v=np.array([z['root_v'] for z in ch]);aa,bb=a['pair'];ix=[legal.index(aa),legal.index(bb)];zz=np.exp((10/3)*v);r=q*zz;r/=r.sum()
 for z in ch:
  terminal=board_win(moves,z['a'],L) or len(moves)+1==L*L
  assert terminal==z['terminal'] and z['root_v']==-z['child_v'] and z['side']==-1
  if g==2 and not terminal:
   w=z['wdl'];assert abs(sum(w)-1)<.001;chk('rapfi_child_sign',np.float32((w[0]-w[1])/sum(w)),z['root_v'],1e-7)
  nchild+=1
 pp=np.array([divmod(j,L) for j in moves]);same=[max(abs(pp[i0]-pp[j]))<=1 for i0 in range(len(pp)) for j in range(i0) if (i0-j)%2==0]
 v0=float(sum(q*v));var=float(sum(q*(v-v0)**2));xy=np.array(divmod(aa,L))-np.array(divmod(bb,L))
 rf=np.array([len(moves),a['v'],np.log(p[aa]+p[bb]),np.log(max(p)),-sum(q*np.log(q))/np.log(len(q)),np.log(p[aa]/p[bb]),np.sqrt(xy@xy)/(L-1),np.sqrt(((pp-(L-1)/2)**2).sum(1)).mean()/(L-1),np.mean(same)])
 cf=np.array([np.mean(v),np.std(v),min(v),max(v),*np.quantile(v,[.25,.5,.75]),v0,np.sqrt(var),(v[ix[0]]+v[ix[1]])/2,v[ix[0]]-v[ix[1]],sum(r*v),sum(zz[ix])/sum(zz),sum(r[ix]),sum(z['terminal'] for z in ch)/len(ch)])
 feat[g,i]=np.r_[rf,cf]
assert len(feat)==456 and nchild==54264
for m,f in zip(D['meta'],D['root']):chk('root_features',feat[m[1],off[m[0]]+m[2]][:9],f,2e-6)
# Selected models independent least squares via augmented system, not original normal equations.
cases=0
for g,z in enumerate(R['results']):
 m0=F['families'][g];cv=m0['selected_cv_scores'];assert m0['selected_baseline']==min(F['models'][:4],key=lambda m:(cv[m],m)) and m0['selected_candidate']==min(F['models'][4:],key=lambda m:(cv[m],m))
 c=core[D['meta'][:,1]==g];X=D['X'][D['meta'][:,1]==g];Y=np.array([X[c==i].mean(0) for i in range(104)]);w=np.array([1/(104*sum(c==i)) for i in c]);assert abs(w.sum()-1)<1e-12
 train=np.array([feat[g,i] for i in range(104)]);mu=train.mean(0);sd=np.maximum(train.std(0),1e-8);train=(train-mu)/sd
 for m in z['models']:
  es=np.array([r['energy'] for r in m['rows']]);W=np.array([r['W'] for r in m['rows']]);truth=np.array([r['truth_W'] for r in m['rows']]);chk('aggregate_score',es.mean(),m['energy_ci'][0]);chk('aggregate_W',np.sqrt(np.mean((W[:,:25]-truth[:,:25])**2)),m['W_RMSE'])
  if m['model'] not in (z['selected_baseline'],z['selected_candidate']):continue
  assert m['model'] in ('root_ridge','combined_ridge')
  dim=9 if m['model']=='root_ridge' else 24;Z=train[:,:dim];y=Y.mean(0);alpha=m['hyperparameter'];coef=np.linalg.lstsq(np.r_[Z,np.sqrt(104*alpha)*np.eye(dim)],np.r_[Y-y,np.zeros((dim,2))],rcond=None)[0]
  resid=X-np.clip(Z@coef+y,[0,-1],[1,1])[c]
  for k,row in enumerate(m['rows']):
   f=((feat[g,104+k]-mu)/sd)[:dim];pred=np.clip(resid+np.clip(f@coef+y,[0,-1],[1,1]),[0,-1],[1,1]);pred[:,0]=np.rint(pred[:,0]*128)/128
   yy=T['X'][(T['meta'][:,1]==g)&(T['meta'][:,2]==k)];assert len(yy)==16
   # independent explicit energy distances; chunk to bound memory
   d=pred*np.array([1,.5]);dy=yy*np.array([1,.5]);a0=np.array([w@np.linalg.norm(d-t,axis=1) for t in dy]).mean()
   a1=0.
   for start in range(0,len(d),128):a1+=w[start:start+128]@((np.sqrt(((d[start:start+128,None]-d[None,:])**2).sum(2)))@w)
   chk('selected_energy',a0-.5*a1,row['energy']);ex=np.exp(pred@grid.T);Z0=w@ex;chk('selected_W',np.log(Z0),row['W']);mm=w@pred;chk('selected_mean',mm,row['pred_mean']);chk('selected_cov',(pred*w[:,None]).T@pred-np.outer(mm,mm),row['cov'])
   ess=[]
   for j in range(len(grid)):
    p1=np.bincount(c,weights=w*ex[:,j],minlength=104)/Z0[j];ess.append(1/(p1@p1))
   chk('selected_ESS',[min(ess[:25]),min(ess[25:])],[row['min_ESS_regular'],row['min_ESS_stress']])
   chk('truth_W',np.log(np.exp(yy@grid.T).mean(0)),row['truth_W']);cases+=1
 b=next(m for m in z['models'] if m['model']==z['selected_baseline']);a=next(m for m in z['models'] if m['model']==z['selected_candidate']);delta=np.array([r['energy'] for r in b['rows']])-np.array([r['energy'] for r in a['rows']]);m=delta.mean();h=student.ppf(.975,47)*delta.std(ddof=1)/np.sqrt(48);chk('gain_ci',[m,m-h,m+h],z['energy_gain_ci']);assert z['registered_pass']==bool(m-h>0 and a['W_RMSE']<b['W_RMSE'])
assert cases==288
# All fresh seed streams already replayed; direct raw endpoint reduction on 144 independent root representatives.
raw=0
for j,r in enumerate(A['raw_receipts']):
 if r['rep']!=0:continue
 p=B/r['path'];assert sha(p)==r['sha256']
 with gzip.open(p,'rt') as h:x=json.load(h)
 es=x['trace']['events'];a=x['input']['sources'];z=[e['root_value'] for e in es];v=[e['actions'][0] for e in es]
 chk('raw_X',[np.mean(np.isin(v,a)),np.mean(z)],T['X'][j]);raw+=1
assert raw==144
# Direct terminal test, feature identity and original source/member authentication.
term=read(S/'TERMINAL_TEST84.json');assert term['status']=='PASS_SCOPED'
for r in term['cases']:assert sha(S/r['file'])==r['sha256'] and r['winning_child']['root_v']==1 and r['winning_child']['child_v']==-1
O83=B/'mqft_restore83_20261127';O69=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O83/f)==sha(O69/f),f
orig=read(O82/'ORIGINAL_SOURCE82.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'};assert all(sha(paths[k])==v for k,v in orig['archives'].items())
for r in orig['verified_members']:
 with zipfile.ZipFile(paths[r['archive']]) as f:v=f.read(r['member'])
 assert hashlib.sha256(v).hexdigest()==r['sha256']
dump(S/'ORIGINAL_SOURCE84.json',dict(orig,scope='Original finite action, statistical field and source-response items L16/Q32/G07 reauthenticated; direct child features and finite conditional prediction, not original full physical amplitude.'))
dump(S/'INDEPENDENT84.json',dict(status='PASS_SCOPED',child_checks=nchild,selected_predictions=cases,all_aggregate_models=21,raw_paths=raw,terminal_controls=3,errors=dict(err),scope='Independent root feature calculation, augmented ridge least squares, explicit chunked energy distances, all selected prediction moments/CGFs and root ESS, aggregate scores, raw endpoints and original archive verification.'))
print('VERIFIED',nchild,cases,raw,dict(err),flush=True)
