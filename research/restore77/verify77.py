"""Independent donor weighting and raw reduction; compare actual finite computed results."""
from base61 import *
import gzip
LS=[13,15,17,19,21];BT=[0.,10/3,16.]
C=read(S/'CLOSURE77.json');R=read(S/'RG77.json');errs=collections.defaultdict(float);cases=0;raw_count=0
def chk(k,a,b,tol=1e-10):
 e=float(np.max(abs(np.asarray(a)-np.asarray(b))));errs[k]=max(errs[k],e);assert e<tol,(k,e)
def features(d,m):
 counts=d['counts'];marks=d['marks'];prob=d['prob'];nn=[];ww=[];pp=[];lag=[]
 for t in range(1,16):
  nn.append(np.searchsorted([.1,.4],counts[:,:t,0].sum(1)/(4*t),side='right'))
  ww.append(np.searchsorted([-.2,.2],marks[:,:t].sum(1)/t,side='right'))
  pp.append(np.searchsorted([.1,.4],prob[:,t,0],side='right'))
  lag.append(np.searchsorted([-.2,.2],marks[:,t-1],side='right'))
 nn=np.array(nn).T;vv=[nn,np.array(ww).T,np.array(pp).T,np.array(lag).T]
 s=np.zeros_like(nn)
 for x in vv[:m]:s=3*s+x
 y=np.array([counts[:,1:15,0]/4,marks[:,1:15]]).transpose(1,2,0)
 return s,nn,y
def donors(s,sn,y,nc,N,J):
 s=s.flatten();sn=sn.flatten();y=y.reshape(-1,2);nc=nc.flatten()
 w=(np.arange(N)[:,None]==s[None,:]).astype(float)+8/len(s);w/=w.sum(1)[:,None]
 K=w@np.eye(N)[sn];E=w@y;P=w@np.eye(3)[nc]
 T=(w*np.exp(y@J/14))@np.eye(N)[sn]
 return K,E,P,T
for L in LS:
 a=read(S/f'AUDIT_L{L}.json');f=S/f'PATHS_L{L}.npz';assert sha(f)==a['array_sha256']
 with np.load(f) as z:d={k:z[k] for k in z.files}
 # direct raw audit of all strata, one path per core 0/rep0/cohort.
 m=d['meta'];rg=np.digitize(np.maximum(abs(np.arange(L*L)//L-L//2),abs(np.arange(L*L)%L-L//2)),[2,4])
 for row in a['raw_receipts']:
  p=B/row['path']
  if not row['id']:raise AssertionError
  # Use authentic header to match core, arm and beta; read only selected plan jobs.
  cohort=row['cohort'];root=B/'mqft_restore76_20261127' if cohort==0 else S
  # plans indexed outside raw data reads below.
  if 'plans' not in locals():plans={}
  key=(L,cohort)
  if key not in plans:plans[key]={j['id']:j for j in read(root/'plans'/f'L{L}_base.json')['jobs']}
  j=plans[key][row['id']]
  if j['replicate']!=0 or not j['position_id'].startswith('c00_'):continue
  assert sha(p)==row['sha256']
  with gzip.open(p,'rt') as z:hd,rr,en=map(json.loads,z)
  assert hd['job']==j and en['backups']==64
  t=rr['trace'];es=sorted(t['replicas'][0]['events'],key=lambda x:x['root_selection']['clock']);k=np.flatnonzero(np.all(m==[cohort,0,int(j['position_id'].endswith('dense')),BT.index(j['spec']['root_law']['beta_p'][0]),0],axis=1));assert len(k)==1;k=int(k[0])
  cs=[];ps=[];zs=[];ws=[]
  for q in range(16):
   e=es[q*4:(q+1)*4];act=[x['actions'][0] for x in e];mark=[x['root_value'] for x in e];cs.append(np.bincount(rg[act],minlength=3));zs.append(sum(mark)/4);ws.append(np.bincount(rg[act],weights=mark,minlength=3))
   v=[]
   for x in e:
    a0=x['root_selection']['law'];v.append(np.bincount(rg[a0['actions']],weights=a0['p'],minlength=3))
   ps.append(np.mean(v,0))
  chk('raw_counts',cs,d['counts'][k]);chk('raw_mark',zs,d['marks'][k]);chk('raw_sector_mark',ws,d['sector_marks'][k]);chk('raw_prob',ps,d['prob'][k]);raw_count+=1
 for arm in (0,1):
  for bi,beta in enumerate(BT):
   tr=(m[:,0]==0)&(m[:,2]==arm)&(m[:,3]==bi);te=(m[:,0]==1)&(m[:,2]==arm)&(m[:,3]==bi)
   a={k:v[tr] for k,v in d.items()};b={k:v[te] for k,v in d.items()}
   q=next(q for q in C['strata'] if (q['L'],q['arm'],q['beta'])==(L,arm,beta))
   for md,model in enumerate(q['models']):
    N=3**md;s,n,y=features(a,md);v,vn,vy=features(b,md)
    for mm in model['models']:
     kind=mm['kind'];J=np.array([0.,4.]);ji=C['J'].index(J.tolist())
     def fit(t0,h):
      if kind=='homogeneous':
       sx=np.concatenate([s[:,t] for t in range(15-h)]);sn=np.concatenate([s[:,t+h] for t in range(15-h)]);yy=np.concatenate([y[:,t:t+h].sum(1) for t in range(15-h)]);nc=np.concatenate([n[:,t+h] for t in range(15-h)])
      else:sx=s[:,t0];sn=s[:,t0+h];yy=y[:,t0:t0+h].sum(1);nc=n[:,t0+h]
      return donors(sx,sn,yy,nc,N,J)
     ds=[fit(t,1) for t in range(14)];pred=np.eye(N)[v[:,0]].mean(0)
     for K,E,P,T in ds:pred=pred@T
     chk('W_model',np.log(pred.sum()),mm['W'][ji]);chk('W_actual',np.log(np.mean(np.exp(vy.sum(1)@J/14))),mm['Wactual'][ji])
     P=np.stack([d0[2][v[:,t]] for t,d0 in enumerate(ds)],1);Y=np.stack([d0[1][v[:,t]] for t,d0 in enumerate(ds)],1)
     chk('one_count_brier',np.mean(np.sum((P-np.eye(3)[vn[:,1:]])**2,-1)),mm['one_step_ci'][0][0]);chk('one_emission_mse',np.mean((Y-vy)**2),mm['one_step_ci'][0][1])
     if kind=='homogeneous':
      fp=mm['fixed_point'];idx=fp['states'];A=ds[0][0][np.ix_(idx,idx)];pi=np.asarray(fp['pi']);chk('stationary',pi@A,pi);Kstar=np.tile(pi,(len(pi),1));chk('projector',Kstar@Kstar,Kstar)
      ev=np.linalg.eigvals(A);chk('eigen',max(abs(np.delete(ev,np.argmin(abs(ev-1)))),default=0.),fp['eigen_subdominant'])
     # One h=4 block prediction audit over all eligible starts.
     h=4;cs=[];es=[];ck=[]
     for t0 in range(15-h):
      M=np.eye(N);E0=np.zeros((N,2))
      for t in range(t0,t0+h):
       K,E,P,T=ds[t];E0+=M@E;P0=M@P;M=M@K
      dt=fit(t0,h);ss=v[:,t0];cs.append(np.sum((P0[ss]-np.eye(3)[vn[:,t0+h]])**2,1));es.append(np.mean((E0[ss]/h-vy[:,t0:t0+h].mean(1))**2,1));ck.append(.5*np.sum(abs(P0[ss]-dt[2][ss]),1))
     c=next(x for x in q['CK'] if x['model']==model['model'] and x['kind']==kind and x['h']==4)['ci'][0]
     chk('CK_brier',np.mean(cs),c[0]);chk('CK_mse',np.mean(es),c[1]);chk('CK_TV',np.mean(ck),c[4]);cases+=1
   z=next(x for x in R['strata'] if (x['L'],x['arm'],x['beta'])==(L,arm,beta))
   for field in ('central_residual','mark'):
    train=a['counts'][:,8:,0]/4-a['prob'][:,8:,0] if field=='central_residual' else a['marks'][:,8:]
    test=b['counts'][:,8:,0]/4-b['prob'][:,8:,0] if field=='central_residual' else b['marks'][:,8:]
    mean=train.mean(0);scale=np.std(train-mean);q=z[field];chk('rg_center',mean,q['training_center']);chk('rg_scale',scale,q['training_scale'])
    test=(test-mean)/scale
    for row in q['blocks']:
     h=row['h'];x=np.stack([test[:,i:i+h].sum(1)/np.sqrt(h) for i in range(0,8,h)],1).ravel();u=x-x.mean();v=np.mean(u*u)
     chk('rg_cumulants',[x.mean(),v,np.mean(u**3),np.mean(u**4)-3*v*v],row['cumulants']);chk('rg_W',np.log(np.mean(np.exp(x[:,None]*np.array(R['source_J'])),0)),row['empirical_W'])
 print('VERIFY',L,flush=True)
assert cases==300 and raw_count==60
dump(S/'INDEPENDENT77.json',dict(status='PASS_SCOPED',models_checked=cases,raw_paths=raw_count,errors=dict(errs),scope='Independent direct normalized donor weights, stationary/eigen check, one-source finite CGF, all h4 start predictions, raw subset and direct block cumulants. Same-agent separate implementation; no native subagent.'))
print(dict(errs),flush=True)
