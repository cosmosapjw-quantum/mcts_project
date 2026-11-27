"""Independent rank-one reductions, sampled path checks, stationary flux witness, source identities."""
from base61 import *
import gzip,zipfile
from scipy.special import logsumexp
N=read(S/'NATIVE85.json');T=read(S/'THERMAL85.json');B0=read(S/'BRANCH_ENERGY85.json')
assert sha(S/'BRANCH_ENERGY85.json')==T['energy_sha'] and sha(S/'THERMAL_SAMPLES85.npz')==T['raw_array_sha'] and sha(S/'WAVE_SAMPLES85.npz')==N['samples_sha']
err=collections.defaultdict(float)
def chk(k,a,b,tol=1e-9):
 e=float(np.max(abs(np.asarray(a)-np.asarray(b))));err[k]=max(err[k],e);assert e<tol,(k,e)
def gib(e):
 a=np.exp(-(10/3)*(e-np.min(e,axis=-1,keepdims=True)));return a/a.sum(-1,keepdims=True)
# All native conditional finite count laws; independent sums and all actual auxiliary sample reductions.
by={(x['family'],x['core'],x['rep']):x for x in N['raw_receipts'] if x['arm']==0}
with np.load(S/'WAVE_SAMPLES85.npz') as z:F=z['forward'];R=z['reverse']
assert F.shape==R.shape==(768,2048,4)
flux=[]
for k,c in enumerate(N['contexts']):
 q=by[c['family'],c['core'],c['rep']];p=np.array(q['p']);a=q['source_index'];v=p.copy();v[a]*=np.exp(.35);v/=v.sum();x=p[a];y=v[a]
 # Probability of forward positive-hit conditioning via direct 2^4 enumeration, no binomial formula.
 f=[];r=[]
 from itertools import product
 for mask in product((0,1),repeat=4):
  n=sum(mask)
  if n:f.append(y**n*(1-y)**(4-n));r.append(x**n*(1-x)**(4-n))
 sf=sum(f);sr=sum(r);f=np.array(f)/sf;r=np.array(r);s=np.log(f/r);chk('native_exact_ift',f@np.exp(-s),c['exact_ift']);chk('native_exact_mean_sigma',f@s,c['exact_mean_sigma']);chk('native_support',sr,c['s_reverse'])
 assert np.all((F[k]>=0)&(F[k]<len(p))) and np.all((R[k]>=0)&(R[k]<len(p))) and np.all((F[k]==a).sum(1)>0)
 sig=(np.log(v[F[k]])-np.log(p[F[k]])).sum(1)-np.log(sf);wt=np.exp(-sig)
 chk('wave_mc_ift',wt.mean(),c['mc_ift']);chk('wave_mc_reverse_support',np.mean(np.any(R[k]==a,axis=1)),c['mc_reverse_support']);chk('wave_mc_ess',wt.sum()**2/(wt@wt),c['ess'],1e-8)
 # Stationary reversible baseline and a distinct nonreversible transition kernel on same actual marginal.
 flow=np.outer(p,p);tri=np.argsort(p)[-3:];pairs=list(zip(tri,np.roll(tri,-1)));delta=.5*min(flow[i,j] for i,j in pairs);assert delta>0
 C=flow.copy()
 for i,j in pairs:C[i,j]+=delta;C[j,i]-=delta
 K=C/p[:,None];assert np.all(K>0);chk('cycle_row_norm',K.sum(1),1);chk('cycle_stationarity',p@K,p);chk('cycle_column_mass',C.sum(0),p);assert abs(C-C.T).max()>0
 ep=float((C*np.log(C/C.T)).sum());closed=sum(2*delta*np.log((flow[i,j]+delta)/(flow[i,j]-delta)) for i,j in pairs);chk('cycle_EP',ep,closed);assert ep>0
 energy=-np.array(q['score']);chk('gibbs_scale_gauge',gib(energy),p);chk('gibbs_scale_gauge',np.exp(-(10/3)*energy-logsumexp(-(10/3)*energy)),p)
 # Rescale E_*, T together; b stays invariant. Common energy offset independent of microstate.
 chk('gibbs_offset',gib(energy+7.3),p)
 flux.append(dict(family=c['family'],core=c['core'],rep=c['rep'],cycle_labels=[q['legal'][i] for i in tri],circulation=delta,stationary_entropy_production=ep,detailed_balance_error=float(abs(C-C.T).max()),baseline_EP=0.))
dump(S/'STATIONARY85.json',dict(contexts=flux,scope='Explicit nonreversible stationary kernel with the same actual frozen Gibbs marginal. Does not reproduce native authenticated iid conditional transitions; marginal alone cannot identify detailed balance or heat.'))
# Independent rank-one feedback expectations. No n*n path matrices in this computation.
lookup={(x['protocol'],x['family'],x['core']):x for x in B0['contexts']}
with np.load(S/'THERMAL_SAMPLES85.npz') as z:F=z['forward'];R=z['reverse']
assert F.shape==R.shape==(832,2048,3)
for k,c in enumerate(T['cases']):
 a=lookup[c['protocol'],c['family'],c['core']];e=np.array(a['e0']);em=np.array(a['em']);g=np.array(a['group']);n=len(e);m=len(em);eps=c['eps'];alpha=c['alpha'];p=gib(e);pi=gib(em)
 D=np.eye(m)[g]*(1-eps)+eps/m;J=p[:,None]*D;pm=J.sum(0);final=(1-alpha)*J.T+alpha*pm[:,None]*pi
 d=em-e;w=float(((J.T-final)*d).sum());q=float(((final-J.T)*em).sum());info=float(np.sum(J[J>0]*np.log((D/pm)[J>0])));end=final.sum(0);rev_final=(1-alpha)*p[None,:]+alpha*pi
 lam=float(np.sum((pm[:,None]*rev_final)[D.T==0]));sig=10/3*w+info
 unc=(1-alpha)+alpha*np.sum((J.T*np.exp(-(10/3)*d)).sum(1)*(pi*np.exp((10/3)*d)).sum(1))
 for name,value in [('mean_work',w),('mean_heat',q),('mean_info',info),('mean_sigma',sig),('lambda_singular',lam),('exact_ift',1-lam),('uncorrected_ift',unc),('mean_delta_E',end@e-p@e),('mean_final_KL',end@np.log(end/p))]:chk('thermal_'+name,value,c[name])
 i,z,j=F[k].T;ir,mr,jr=R[k].T
 assert np.all((i>=0)&(i<n)&(j>=0)&(j<n)&(z>=0)&(z<m)) and np.all((ir>=0)&(ir<n)&(jr>=0)&(jr<n)&(mr>=0)&(mr<m))
 wf=d[z,i]-d[z,j];qf=em[z,j]-em[z,i];inf=np.log(D[i,z]/pm[z]);s=(10/3)*wf+inf;v=np.exp(-s)
 pfor=p[i]*D[i,z]*((1-alpha)*(i==j)+alpha*pi[z,j]);prev=pm[z]*p[j]*((1-alpha)*(i==j)+alpha*pi[z,i]);chk('sample_log_ratio',np.log(pfor/prev),s)
 chk('sample_IFT',v.mean(),c['mc_ift']);chk('sample_reverse_support',np.mean(D[ir,mr]>0),c['mc_reverse_support']);chk('sample_work',wf.mean(),c['mc_work']);chk('sample_heat',qf.mean(),c['mc_heat']);chk('sample_info',inf.mean(),c['mc_info'])
 wr=d[mr,jr]-d[mr,ir];qr=em[mr,ir]-em[mr,jr];chk('sample_reverse_work',wr.mean(),c['mc_reverse_work']);chk('sample_reverse_heat',qr.mean(),c['mc_reverse_heat'])
# Independent raw native forward ledger reductions for all 144 root/arm representatives.
raw=0
for c in N['raw_receipts']:
 if c['rep']!=0:continue
 p=B/c['path'];assert sha(p)==c['sha256']
 with gzip.open(p,'rt') as f:d=json.load(f)
 tr=d['trace'];es=tr['events'];legal=c['legal'];src=legal[c['source_index']];counts=collections.Counter()
 for e in es:counts[e['actions'][0]]+=1
 for k in (4,8,16,32,64,128):
  z=next(v for v in tr['cuts'] if v['k']==k);a=np.array([e['actions'][0] for e in es[:k]]);v=np.array([e['root_value'] for e in es[:k]])
  n=np.bincount(a,minlength=len(z['n']));w=np.bincount(a,weights=v,minlength=len(z['n']));s2=np.bincount(a,weights=v*v,minlength=len(z['n']))
  assert np.array_equal(n,z['n']);chk('forward_cut_w',w,z['w']);chk('forward_cut_s2',s2,z['s2'])
 assert sum(e['actions'][0]==src for e in es[32:36])==c['hits'];raw+=1
assert raw==144
# Verify exact original local archive identity and specific source spans. Read actual proposals even though not prior member list.
O79=B/'mqft_restore79_20261127';O84=B/'mqft_restore84_20261127';O69=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O84/f)==sha(O69/f),f
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'};assert all(sha(paths[k])==v for k,v in orig['archives'].items())
ids=['L08','L18','L32','Q19','Q20','Q21','Q23'];spans=[s for s in orig['spans'] if s['id'] in ids];assert len(spans)==len(ids)
want={s['member'] for s in spans}|{'mcts_study/v3.0/qft-mcts-statistical.md'}
members=[]
for s in orig['verified_members']:
 if s['member'] not in want:continue
 with zipfile.ZipFile(paths[s['archive']]) as f:v=f.read(s['member'])
 assert hashlib.sha256(v).hexdigest()==s['sha256'];members.append(s)
assert {s['member'] for s in members}==want
for k in (1,4):
 m=f'mcts_study/thermodynamic-mcts-research-proposal_{k}.md'
 with zipfile.ZipFile(paths['mcts_study.zip']) as f:v=f.read(m)
 s=v.decode();assert ('No thermodynamic claims' if k==1 else 'We do NOT claim physical energy') in s
 members.append(dict(archive='mcts_study.zip',member=m,sha256=hashlib.sha256(v).hexdigest()))
dump(S/'ORIGINAL_SOURCE85.json',dict(archives=orig['archives'],verified_members=members,spans=spans,original_attachment_receipt=orig['original_attachment_receipt'],scope='Original work/heat/Jarzynski/information and actual measurement objective; proposals explicitly distinguish information from physical thermodynamics. Exact source members authenticated.'))
dump(S/'INDEPENDENT85.json',dict(status='PASS_SCOPED',wave_cases=768,thermal_cases=832,raw_representatives=144,auxiliary_samples=N['auxiliary_samples']+T['auxiliary_samples'],stationary_witnesses=768,errors=dict(err),scope='Independent 16-pattern exact wave calculation, rank-one feedback reduction and all sample path sums, explicit stationary circulation, raw forward cut subset; same-agent separate algorithm.'))
print('VERIFIED',raw,768,832,dict(err),flush=True)
