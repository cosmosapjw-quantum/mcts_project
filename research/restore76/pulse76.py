"""Exact final-wave paired source-count calibration, conditional on recorded shared prefixes and ideal IID uniform root draws."""
from base61 import *
import gzip
LS=[13,15,17,19,21];out=[]
for L in LS:
 groups=collections.defaultdict(dict);audit=read(S/f'AUDIT_L{L}.json')
 for rc in audit['raw_receipts']:
  if rc['campaign'] not in ('base','pulse') or rc['budget']!=16:continue
  f=B/rc['path'];assert sha(f)==rc['sha256']
  with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
  j=hd['job'];sp=row['trace']['spec']
  if not j['position_id'].endswith('dense') or sp['root_law']['beta_p'][0]!=10/3:continue
  h=sp['root_law'].get('logit_schedule',[0.])[-1]
  k=(j['position_id'],j['replicate']);assert h not in groups[k]
  es=sorted(row['trace']['replicas'][0]['events'],key=lambda e:e['root_selection']['clock'])
  groups[k][h]=(sp['source_action'],sp['seed'],es,rc['sha256'])
 assert len(groups)==32
 probs=[];observed=0;maxerr=0.;prefix_hashes=[]
 for k,g in sorted(groups.items()):
  assert set(g)=={-.35,0.,.35};a,s,e0,z0=g[0.]
  prefix_hashes.append([g[h][3] for h in [-.35,0.,.35]])
  for h in [-.35,.35]:
   aa,ss,ee,zz=g[h];assert aa==a and ss==s
   for t in range(12):assert ee[t]['actions']==e0[t]['actions'] and ee[t]['root_selection']['law']==e0[t]['root_selection']['law']
  ref=e0[12]['root_selection']['law'];p0=np.array(ref['p']);acts=ref['actions'];i=acts.index(a)
  for t in range(12,16):
   intervals={};obs={}
   for h in [-.35,0.,.35]:
    e=g[h][2][t];law=e['root_selection']['law'];p=np.array(law['p'])
    assert law['actions']==acts and law['scores']==ref['scores'] and law['u']==e0[t]['root_selection']['law']['u']
    pred=p0*np.exp(h*(np.array(acts)==a));pred/=pred.sum();maxerr=max(maxerr,float(abs(pred-p).max()))
    l=float(p[:i].sum());r=l+float(p[i]);intervals[h]=(l,r);obs[h]=int(e['actions'][0]==a)
    assert obs[h]==int(l<=law['u']<r)
   l0,r0=intervals[-.35];l1,r1=intervals[.35]
   assert l1<=l0+1e-13 and r1>=r0-1e-13
   q=(r1-l1)-(r0-l0);assert 0<=q<=1
   probs.append(q);d=obs[.35]-obs[-.35];assert d in (0,1);observed+=d
 assert len(probs)==128 and maxerr<1e-12
 pmf=np.array([1.])
 for p in probs:pmf=np.convolve(pmf,[1-p,p])
 assert abs(pmf.sum()-1)<1e-12 and abs(pmf@np.arange(len(pmf))-sum(probs))<1e-12
 cdf=np.cumsum(pmf);lo,hi=[int(np.searchsorted(cdf,x)) for x in [.025,.975]]
 out.append(dict(L=L,paired_draws=len(probs),observed_difference_count=observed,expected_difference_count=sum(probs),p_zero=float(pmf[0]),predictive_count_95=[lo,hi],observed_inside=lo<=observed<=hi,prob_at_least_observed=float(pmf[observed:].sum()),prob_at_most_observed=float(pmf[:observed+1].sum()),max_reweight_error=maxerr,pmf=pmf.tolist(),raw_triple_sha=prefix_hashes))
dump(S/'PULSE_EXACT76.json',dict(status='PASS_SCOPED',scope='Conditional Poisson-binomial predictive distribution for paired counts with nested source CDF intervals; ideal IID uniform draws given shared prefixes. Not confidence intervals for population effects; mark response not covered.',cases=out))
print([(r['L'],r['observed_difference_count'],r['expected_difference_count'],r['p_zero'],r['predictive_count_95'],r['max_reweight_error']) for r in out],flush=True)
