"""Independent cross-tabulation, inclusion-exclusion memory, raw witness and scale checks."""
from base61 import *
import gzip,zipfile
from itertools import combinations
from scipy.stats import t as student
A=read(S/'AUDIT86.json');R=read(S/'MEMORY86.json');M0=read(S/'MODEL86.json');V=read(S/'SCALE86.json')
assert sha(S/'WAVES86.npz')==A['array_sha']==R['array_sha']==M0['wave_array_sha'] and sha(S/'MODEL86.json')==R['model_sha']
with np.load(S/'WAVES86.npz') as f:d={k:f[k] for k in f.files}
m=d['meta'];root=m[:,2]+np.array([0,48,72,104,152])[m[:,0]];cohort=np.where(m[:,0]<3,0,m[:,0]-2)
y=(np.cumsum(d['count'],axis=1)[:,1::2]/(8*np.arange(1,17))>=1/16).astype(int)+(np.cumsum(d['count'],axis=1)[:,1::2]/(8*np.arange(1,17))>=1/4).astype(int)
w=np.cumsum(d['mark'],axis=1)[:,1::2]/(8*np.arange(1,17));y=3*y+(w>=-.2).astype(int)+(w>=.2).astype(int)
err=collections.defaultdict(float)
def chk(k,a,b,tol=1e-9):
 v=float(abs(np.asarray(a)-np.asarray(b)).max());err[k]=max(err[k],v);assert v<tol,(k,v)
def ci(x):
 x=np.array(x);s=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.array([s,s-h,s+h])
def cut_simplex(a):
 if min(a)>=-1e-14:return np.maximum(a,0)/np.maximum(a,0).sum()
 # independent bisection of Lagrange multiplier
 lo=min(a)-1;hi=max(a)
 for _ in range(55):
  mid=(lo+hi)/2
  if np.maximum(a-mid,0).sum()>1:lo=mid
  else:hi=mid
 return np.maximum(a-(lo+hi)/2,0)
pairs=terms=pred_count=0
for g,model in enumerate(M0['models']):
 ix=(m[:,1]==g)&(cohort==0);a=y[ix];r=root[ix];wt=np.array([1/(104*sum(r==j)) for j in r]);sup=[np.unique(a[:,i]) for i in range(16)];assert [x.tolist() for x in sup]==model['support']
 G={};K={tuple(map(int,k.split(','))):np.array(v) for k,v in model['K'].items()};mass=[]
 for i in range(16):
  ni=np.searchsorted(sup[i],a[:,i]);mass.append(np.bincount(ni,weights=wt,minlength=len(sup[i])))
  for j in range(i,16):
   nj=np.searchsorted(sup[j],a[:,j]);tab=np.bincount(ni*len(sup[j])+nj,weights=wt,minlength=len(sup[i])*len(sup[j])).reshape(len(sup[i]),len(sup[j]));G[i,j]=tab/tab.sum(1,keepdims=True);chk('G_cross_tab',G[i,j],model['G'][f'{i},{j}']);pairs+=1
 # explicit inclusion-exclusion formula over inserted intermediate projections; all delays <=8.
 for i in range(15):
  for s in range(i+1,min(16,i+9)):
   v=G[i,s].copy()
   for n in range(1,s-i):
    for route in combinations(range(i+1,s),n):
     path=(i,)+route+(s,);q=G[path[0],path[1]]
     for j,k in zip(path[1:-1],path[2:]):q=q@G[j,k]
     v+=(-1)**n*q
   chk('K_inclusion_exclusion',v,K[i,s-1]);terms+=1
 # Independent forward recurrence: path compositions of allowed delays; for all measured horizons.
 for ext in R['results'][g]['external']:
  ix=(m[:,1]==g)&(m[:,0]==ext['group']);a=y[ix];r=root[ix]
  for h0 in ext['horizons']:
   h=h0['horizon'];o_values={}
   for order in (1,2,4,15):
    by=[]
    for i in range(16-h):
     s=i+h;acc={i:np.eye(len(sup[i]))}
     for j in range(i+1,s+1):
      acc[j]=sum((acc[k]@K[k,j-1] for k in range(max(i,j-order),j)),start=np.zeros((len(sup[i]),len(sup[j]))))
     f=acc[s];q=np.array([cut_simplex(x) for x in f]);raw=[];pred=[]
     for k in a[:,i]:
      if k not in sup[i]:v=mass[s];z=v
      else:l=int(np.searchsorted(sup[i],k));v=f[l];z=q[l]
      x=np.zeros(9);x[sup[s]]=v;w0=np.zeros(9);w0[sup[s]]=z;raw.append(x);pred.append(w0)
     raw=np.array(raw);pred=np.array(pred);t=np.eye(9)[a[:,s]]
     vals=np.stack([((pred-t)**2).sum(1),((pred.reshape(-1,3,3).sum(2)-np.eye(3)[a[:,s]//3])**2).sum(1),((pred.reshape(-1,3,3).sum(1)-np.eye(3)[a[:,s]%3])**2).sum(1),np.maximum(-raw,0).sum(1),.5*abs(pred-raw).sum(1)],1);by.append(vals);pred_count+=len(a)
    z=np.mean(by,0);v=np.stack([z[r==k].mean(0) for k in np.unique(r)]);chk('heldout_score_and_CI',ci(v),h0['scores'][str(order)]);o_values[order]=v
   for order in (2,4,15):chk('heldout_gain_CI',ci(o_values[1]-o_values[order]),h0['gains_vs_markov'][str(order)])
# Verify actual worst coarse collisions from raw native records, not just array references.
raw=0;witnesses=[]
def raw_at(i,t):
 rec=A['raw_receipts'][i];p=B/rec['path'];assert sha(p)==rec['sha256']
 with gzip.open(p,'rt') as f:q=json.load(f)
 es=q['trace']['events'];pair=q['input']['sources'];a=[e['actions'][0] for e in es[:t]];z=[e['root_value'] for e in es[:t]]
 l=es[t]['root_selection']['law'];prob=sum(p0 for a0,p0 in zip(l['actions'],l['p']) if a0 in pair);c=sum(v in pair for v in a);w=sum(z)
 return q['input']['moves'],c,w,prob,rec
for a in R['collisions']:
 w=a['models']['coarse_count_mark']['witness'];i,j,t=w['i'],w['j'],w['clock'];u=raw_at(i,t);v=raw_at(j,t)
 assert u[0]==v[0] and u[4]['family']==v[4]['family'] and u[4]['seed']!=v[4]['seed'];chk('collision_prob',[u[3],v[3]],w['prob']);chk('collision_mark',[u[2],v[2]],w['mark']);assert [u[1],v[1]]==w['count']
 assert np.digitize(u[1]/t,[1/16,1/4])==np.digitize(v[1]/t,[1/16,1/4]) and np.digitize(u[2]/t,[-.2,.2])==np.digitize(v[2]/t,[-.2,.2])
 # Exact one-wave binomial TV can exceed individual sector-pick TV.
 p,q=u[3],v[3];k=np.arange(5);b=lambda x:np.array([math.comb(4,int(n))*x**n*(1-x)**(4-n) for n in k]);tv=float(.5*abs(b(p)-b(q)).sum())
 witnesses.append(dict(family=a['family'],cohort=a['cohort'],clock=t,prob=[p,q],binomial_TV=tv,paths=[u[4]['path'],v[4]['path']],sha=[u[4]['sha256'],v[4]['sha256']]))
 raw+=2
# One independent raw trajectory per fresh geometry and evaluator; 72 endpoint and wave reductions.
for i,rec in enumerate(A['raw_receipts']):
 if rec['group']!=4 or rec['rep']!=0:continue
 p=B/rec['path'];assert sha(p)==rec['sha256']
 with gzip.open(p,'rt') as f:q=json.load(f)
 es=q['trace']['events'];pair=q['input']['sources'];c=[];z=[];prob=[]
 for t in range(0,128,4):
  v=es[t:t+4];c.append(sum(e['actions'][0] in pair for e in v));z.append(sum(e['root_value'] for e in v));l=v[0]['root_selection']['law'];prob.append(sum(p0 for a0,p0 in zip(l['actions'],l['p']) if a0 in pair))
 chk('fresh_raw_count',c,d['count'][i]);chk('fresh_raw_mark',z,d['mark'][i]);chk('fresh_raw_prob',prob,d['prob'][i]);raw+=1
assert raw==90
# Independent direct scale aggregates and lag-resolved compensated cross-moments. All 1920 raw receipt bytes verified.
O=B/'mqft_restore77_20261127';sraw=V['raw_receipts'];assert len(sraw)==1920 and all(sha(B/x['path'])==x['sha256'] for x in sraw)
scale_cases=0;autocorr=[]
for L in (13,15,17,19,21):
 p=O/f'PATHS_L{L}.npz';assert sha(p)==V['arrays'][str(p.relative_to(B))]
 with np.load(p) as f:z={k:f[k] for k in f.files}
 mm=z['meta']
 for c in [v for v in V['strata'] if v['L']==L]:
  bi=[0.,10/3,16.].index(c['beta']);sel=(mm[:,0]==c['cohort'])&(mm[:,2]==c['arm'])&(mm[:,3]==bi);cnt=z['counts'][sel];pr=z['prob'][sel];rr=mm[sel,1];D=cnt-4*pr;var=4*(1-(pr*pr).sum(2));lag=[]
  for h in (1,2,4,8,16):
   a=next(x for x in c['blocks'] if x['h']==h);ds=np.stack([D[:,i:i+h].sum(1) for i in range(0,16,h)],1);qv=np.stack([var[:,i:i+h].sum(1) for i in range(0,16,h)],1);v=(ds*ds).sum(-1)/(4*h);q=qv/(4*h)
   u=np.stack([(v-q)[rr==k].mean() for k in np.unique(rr)]);chk('scale_variance_CI',ci(u),a['difference_ci'])
   for v0 in a['exponential']:
    J=np.array(v0['J']);w0=np.exp(np.stack([(cnt[:,i:i+h]@J-4*np.log(pr[:,i:i+h]@np.exp(J))).sum(1) for i in range(0,16,h)],1));chk('scale_IFT_CI',ci([w0[rr==k].mean() for k in np.unique(rr)]),v0['mean_ci'])
   scale_cases+=1
  # Cross-moments, not a stationary ACF: compare pooled-clock centered raw and compensated increments.
  C=cnt-cnt.mean((0,1));s0=float(np.mean((C*C).sum(2)));s1=float(np.mean((D*D).sum(2)))
  for h in (1,2,4,8):
   raw0=(C[:,:-h]*C[:,h:]).sum(2)/s0;res=(D[:,:-h]*D[:,h:]).sum(2)/s1
   lag.append(dict(lag=h,raw_ci=ci([raw0[rr==k].mean() for k in np.unique(rr)]).tolist(),compensated_ci=ci([res[rr==k].mean() for k in np.unique(rr)]).tolist()))
  autocorr.append(dict(L=L,arm=c['arm'],beta=c['beta'],cohort=c['cohort'],lag=lag))
assert scale_cases==300
dump(S/'AUTOCORRELATION86.json',dict(strata=autocorr,scope='Nonstationary lag cross-moments normalized by pooled wave second moment; raw count centered over same stratum, compensated count uses known predictable probabilities. No fitted stationary exponential or correlation time. Secondary diagnostic, not preregistered gate.'))
dump(S/'COLLISION_WITNESSES86.json',dict(witnesses=witnesses,scope='Raw actual same-root/time coarse collisions; exact binomial next-wave law difference, not full-tree sufficiency.'))
# Original archive and exact relevant members, original R77 fixed-point limitations retained.
O79=B/'mqft_restore79_20261127';O85=B/'mqft_restore85_20261127';O69=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O85/f)==sha(O69/f),f
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'};assert all(sha(paths[k])==v for k,v in orig['archives'].items())
ids=['L16','L37','L38','L39'];spans=[s for s in orig['spans'] if s['id'] in ids];assert len(spans)==4
want={s['member'] for s in spans}|{'mcts_study/v3.0/qft-mcts-statistical.md'};members=[]
for s in orig['verified_members']:
 if s['member'] not in want:continue
 with zipfile.ZipFile(paths[s['archive']]) as f:v=f.read(s['member'])
 assert hashlib.sha256(v).hexdigest()==s['sha256'];members.append(s)
assert {s['member'] for s in members}==want
dump(S/'ORIGINAL_SOURCE86.json',dict(archives=orig['archives'],verified_members=members,spans=spans,original_attachment_receipt=orig['original_attachment_receipt'],scope='Original path, RG fixed point/scale and autocorrelation objectives; exact finite conditional projection and compensated fluctuations, no native RG fixed point.'))
dump(S/'INDEPENDENT86.json',dict(status='PASS_SCOPED',cross_tab_pairs=pairs,independent_kernel_cases=terms,heldout_path_start_predictions=pred_count,raw_reductions=raw,scale_cases=scale_cases,scale_raw_receipts_verified=len(sraw),errors=dict(err),scope='Separate cross-tab and inclusion-exclusion memory check; forward composition and independent simplex bisection, heldout score CIs, raw collision witnesses and scale reduction; same-agent separate implementation.'))
print('VERIFIED',pairs,terms,pred_count,raw,scale_cases,dict(err),flush=True)
