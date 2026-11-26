"""R59 pre-evaluation root mark prediction; train R58, holdout R59."""
from pathlib import Path
import json,gzip,hashlib,math,sys,collections,time
import numpy as np
S=Path(__file__).resolve().parent; B=S.parent; R=B/'mqft_restore58_20261126'
LOAD=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
STATE=['L','beta','density','clock_fraction','root_value_initial','initial_nnue_action','log_prior_rel','selected_score','log_p_rel','completed_action_logcount','completed_action_q','completed_global_mean','completed_global_sd','dist_center','dist_nearest','local_own','local_opp','samewave_prior_reservations']
HIST=['prev_wave_mean','prev_wave_sd','prev_wave_n','last_action_mark','first_action_mark','last_action_age','has_action_history','last_minus_action_mean','previous_two_wave_delta']
N=len(STATE)
def avg(x):return float(np.mean(x)) if len(x) else 0.
def sd(x):return float(np.std(x)) if len(x) else 0.
def features(t):
 sp=t['spec'];root=t['root'];L=root['board_size'];board=np.array(root['board']).reshape(L,L);side=root['side'];rep=t['replicas'][0]
 assert sp['budget']==32 and sp['mode']=='wave' and sp['root_law']['kind']=='iid' and sp['root_law']['logit_h']==0
 ev=sorted(rep['events'],key=lambda e:e['root_selection']['clock']);assert len(ev)==32
 pri={x['action']:x['prior'] for x in rep['cuts'][0]['edges']};sc=dict(rep['evaluations'][0]['nnue_scores']);initial=rep['evaluations'][0]['value'];legal=root['legal'];d=len(legal)
 assert set(pri)==set(legal)==set(sc) and all(x>0 for x in pri.values())
 assert all(abs(x['prior']-pri[x['action']])<1e-10 for c in rep['cuts'] for x in c['edges'])
 prior=[];waves=[];rows=[];same=collections.Counter();per=collections.defaultdict(list)
 for phase,group in __import__('itertools').groupby(ev,key=lambda e:e['phase']):
  group=list(group);same.clear()
  for e in group:
   sel=e['root_selection'];law=sel['law'];a=e['actions'][0];i=law['picked'];clock=sel['clock']
   assert law['actions'][i]==a and law['active'][i] and all(q['root_backup_clock']<clock for q in prior)
   assert e['root_value']==(-1)**len(e['actions'])*e['leaf_value'] and abs(e['root_value'])<=1+1e-7
   assert abs(sum(law['p'])-1)<1e-9 and law['p'][i]>0
   y,x=divmod(a,L);stones=np.argwhere(board!=0);dist=min((abs(int(yy)-y)+abs(int(xx)-x) for yy,xx in stones),default=L)
   local=board[max(0,y-2):min(L,y+3),max(0,x-2):min(L,x+3)];z=[q[1] for q in per[a]];allz=[q['root_value'] for q in prior];last=waves[-1] if waves else []
   state=[L,sp['root_law']['beta_p'][0],float(np.count_nonzero(board)/L**2),len(prior)/32,initial,sc[a]/1000,math.log(pri[a]*d),law['scores'][i],math.log(law['p'][i]*d),math.log1p(len(z)),avg(z),avg(allz),sd(allz),math.hypot(y-(L-1)/2,x-(L-1)/2)/L,dist/L,np.count_nonzero(local==side)/25,np.count_nonzero((local!=0)&(local!=side))/25,same[a]]
   hist=[avg(last),sd(last),len(last),z[-1] if z else 0.,z[0] if z else 0.,(phase-per[a][-1][0])/32 if z else 1.,float(bool(z)),(z[-1]-avg(z)) if z else 0.,avg(last)-avg(waves[-2]) if len(waves)>1 else 0.]
   rows.append((state+hist,e['root_value'],phase,a,clock));same[a]+=1
  for e in group:per[e['actions'][0]].append((phase,e['root_value']))
  prior.extend(group);waves.append([e['root_value'] for e in group])
 assert len(rows)==32 and np.isfinite(np.asarray([r[0] for r in rows])).all()
 # independent final snapshot: count and W reconstructed solely from events.
 final=rep['cuts'][-1];assert final['local_k']==32
 for e in final['edges']:
  v=[q[1] for q in per[e['action']]];assert len(v)==e['n'] and abs(math.fsum(v)-e['w'])<1e-8 and abs(math.fsum(x*x for x in v)-e['s2'])<1e-8
 return rows

def collect(train):
 dirs=[(R/'data'/f'B32b{i}-L{L}',R/'analysis'/f'B32b{i}-L{L}',L,True) for L in (13,15,17,19,21) for i in range(3)] if train else [(S/'data'/f'heldout-L{L}',S/'analysis'/f'heldout-L{L}',L,False) for L in (13,21)]
 X=[];Y=[];J=[];G=[];LS=[];PH=[];A=[];meta=[];checks=0;histories=[]
 for folder,an,L,is_train in dirs:
  rp=LOAD(next(folder.glob('report-*.json')));manifest=LOAD(an/'manifest.json');lock=LOAD(folder/'run_lock.json')
  reg=LOAD((R if is_train else S)/f'REGISTRATION_L{L}.json')
  expected=12 if is_train else 32
  assert len(rp['completed'])==expected==manifest['completed']==manifest['expected'] and not rp['failures'] and not rp['unattempted']
  assert lock['backend_sha']==reg['backend_sha'] and lock['external']==reg['model'] and lock['plan_hash']==manifest['plan_hash']
  if not is_train:assert manifest['plan_hash']==reg['plan_hash']
  assert all(sha(an/f)==h for f,h in manifest['outputs'].items())
  for rec in sorted(rp['completed'],key=lambda x:x['job_id']):
   p=folder/rec['file'];assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'] and LOAD(folder/(rec['job_id']+'.receipt.json'))==rec
   with gzip.open(p,'rt') as f:
    hd=json.loads(next(f));row=json.loads(next(f));end=json.loads(next(f));assert not f.read().strip()
   job=hd['job'];t=row['trace'];assert hd['evaluator_runtime']==reg['model'] and hd['backend_sha']==reg['backend_sha'] and end['backups']==32 and rec['backups']==32
   assert t['root']['board_size']==L
   rr=features(t);# perturb last wave marks only: features must be invariant through last wave
   if checks==0:
    import copy
    u=copy.deepcopy(t);es=u['replicas'][0]['events'];last=max(e['phase'] for e in es)
    # mutate evaluator-independent features by changing last marks and update final snapshot consistently
    for e in es:
     if e['phase']==last:e['root_value']=-e['root_value'];e['leaf_value']=-e['leaf_value']
    for cut in u['replicas'][0]['cuts']:
     if cut['local_k']==32:
      for x in cut['edges']:
       vv=[e['root_value'] for e in es if e['actions'][0]==x['action']];x['w']=math.fsum(vv);x['s2']=math.fsum(v*v for v in vv)
    r2=features(u);assert np.array_equal(np.asarray([r[0] for r in rr]),np.asarray([r[0] for r in r2]));checks+=1
   idx=len(histories);core=int(job['position_id'][-1]);histories.append(dict(job=job['id'],core=core,L=L,seed=job['spec']['seed'],root=t['root']['state_hash'],sha=rec['sha256'],path=str(p.relative_to(B))))
   for xx,y,ph,a,cl in rr:X.append(xx);Y.append(y);J.append(idx);G.append(core);LS.append(L);PH.append(ph);A.append(a)
   meta.append(rec['sha256'])
  print('COLLECT',folder.name,expected,flush=True)
 return dict(X=np.asarray(X),y=np.asarray(Y),job=np.asarray(J),core=np.asarray(G),L=np.asarray(LS),phase=np.asarray(PH),action=np.asarray(A)),histories,checks

def fit(X,y,d):
 x=X[:,:d];mu=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1.;z=(x-mu)/scale;z=np.column_stack([np.ones(len(x)),z]);pen=np.diag([0.]+[.02]*d);w=np.linalg.solve(z.T@z/len(z)+pen,z.T@y/len(z));assert np.isfinite(w).all()
 return dict(mu=mu.tolist(),scale=scale.tolist(),w=w.tolist(),d=d)
def predict(model,X):
 d=model['d'];z=(X[:,:d]-model['mu'])/model['scale'];w=np.asarray(model['w']);return np.clip(w[0]+z@w[1:],-1,1)
def models(a,idx):return {'constant':float(a['y'][idx].mean()),'state':fit(a['X'][idx],a['y'][idx],N),'history':fit(a['X'][idx],a['y'][idx],N+len(HIST))}
def predictions(m,X):return np.column_stack([np.zeros(len(X)),np.full(len(X),m['constant']),X[:,STATE.index('completed_action_q')],predict(m['state'],X),predict(m['history'],X)])
NAMES=['zero','constant','action_mean','state','history']
def train():
 a,h,c=collect(True);assert len(h)==180 and len(a['y'])==5760 and c==1
 # deterministic nested train sets ordered by SHA256(job id), no heldout inspection
 ids=sorted(range(len(h)),key=lambda i:hashlib.sha256(h[i]['job'].encode()).hexdigest());mm={str(n):models(a,np.isin(a['job'],ids[:n])) for n in (45,90,180)}
 np.savez_compressed(S/'TRAIN_FEATURES.npz',**a);dump(S/'TRAIN_MANIFEST.json',h);dump(S/'MODEL59.json',dict(state=STATE,history=HIST,models=mm,train_manifest_sha256=sha(S/'TRAIN_MANIFEST.json'),train_features_sha256=sha(S/'TRAIN_FEATURES.npz'),script_sha256=sha(Path(__file__)),heldout_plan_shas={str(L):sha(S/f'plans/heldout-L{L}.json') for L in (13,21)},n_histories=len(h),n_marks=len(a['y']),causality_test=c))
 print('TRAIN_FROZEN',sha(S/'MODEL59.json'),flush=True)
def acf(w):
 out=[]
 for lag in range(1,6):
  pairs=np.array([(x[i],x[i+lag]) for x in w for i in range(len(x)-lag)])
  out.append(float(np.corrcoef(pairs.T)[0,1]) if len(pairs) and np.std(pairs[:,0])*np.std(pairs[:,1])>0 else None)
 return out
def trans(w):
 m=np.zeros((3,3),int);q=np.zeros((3,3,3),int)
 for x in w:
  b=np.digitize(x,[-1/3,1/3]);
  for u,v in zip(b[:-1],b[1:]):m[u,v]+=1
  for u,v,z in zip(b[:-2],b[1:-1],b[2:]):q[u,v,z]+=1
 return m.tolist(),q.tolist()
def evaluate():
 fr=LOAD(S/'MODEL59.json');assert fr['script_sha256']==sha(Path(__file__)) and fr['train_manifest_sha256']==sha(S/'TRAIN_MANIFEST.json') and fr['train_features_sha256']==sha(S/'TRAIN_FEATURES.npz')
 assert all(sha(S/f'plans/heldout-L{L}.json')==v for L,v in fr['heldout_plan_shas'].items())
 a,h,c=collect(False);assert len(h)==64 and len(a['y'])==2048 and c==1
 tr=LOAD(S/'TRAIN_MANIFEST.json');assert not {x['root'] for x in tr}&{x['root'] for x in h} and not {x['seed'] for x in tr}&{x['seed'] for x in h}
 pr=predictions(fr['models']['180'],a['X']);err=(pr-a['y'][:,None])**2;assert np.isfinite(err).all() and np.max(abs(pr))<=1
 mse=err.mean(0);core=np.array([err[a['core']==g].mean(0) for g in range(4)]);byL={str(L):err[a['L']==L].mean(0).tolist() for L in (13,21)};diff=core[:,3]-core[:,4];rng=np.random.default_rng(595959);bs=diff[rng.integers(0,4,(10000,4))].mean(1)
 waves=[];res=[];res0=[]
 for i in range(len(h)):
  phases=np.unique(a['phase'][a['job']==i]);ix=[(a['job']==i)&(a['phase']==ph) for ph in phases];waves.append([avg(a['y'][z]) for z in ix]);res.append([avg(a['y'][z]-pr[z,4]) for z in ix]);res0.append([avg(a['y'][z]-pr[z,3]) for z in ix])
 conv={n:((predictions(m,a['X'])-a['y'][:,None])**2).mean(0).tolist() for n,m in fr['models'].items()}
 m,q=trans(waves);mr,qr=trans(res);cal=[]
 for j in (3,4):
  b=np.clip(np.digitize(pr[:,j],np.linspace(-1,1,9))-1,0,7);cal.append([dict(n=int((b==k).sum()),pred=avg(pr[b==k,j]),observed=avg(a['y'][b==k])) for k in range(8)])
 np.savez_compressed(S/'HELDOUT_PREDICTIONS.npz',**a,pred=pr);dump(S/'HELDOUT_MANIFEST.json',h)
 r=dict(n_train=180,train_marks=5760,n_holdout=64,heldout_marks=2048,model_sha256=sha(S/'MODEL59.json'),heldout_manifest_sha256=sha(S/'HELDOUT_MANIFEST.json'),heldout_predictions_sha256=sha(S/'HELDOUT_PREDICTIONS.npz'),names=NAMES,mse=mse.tolist(),core_mse=core.tolist(),by_size=byL,core_differences=diff.tolist(),mean_improvement=float(diff.mean()),cluster_bootstrap_95=np.quantile(bs,[.025,.975]).tolist(),clusters=4,bootstrap_scope='descriptive four paired core blocks; no general population inference',acf={'raw':acf(waves),'state_resid':acf(res0),'history_resid':acf(res)},transition_raw=m,transition_history_resid=mr,transition_two_lag_raw=q,transition_two_lag_history_resid=qr,convergence=conv,calibration=cal,causality_test=c)
 dump(S/'PHYSICS_RESULTS.json',r);print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':{'train':train,'evaluate':evaluate}[sys.argv[1]]()
