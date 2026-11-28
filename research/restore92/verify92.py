from base61 import *
import gzip,csv,zipfile,platform
from scipy.stats import t as student
O=B/'mqft_restore87_20261128';P=B/'mqft_restore91_20261128';R=B/'mqft_restore69_20261126'
def gzread(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def gzsha(p):
 h=hashlib.sha256()
 with gzip.open(p,'rb') as f:
  for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
 return h.hexdigest()
def ci(x):
 x=np.asarray(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
 return np.stack([m,m-h,m+h])
def main():
 pkg=read(S/'SCALE_PACKING92.json');assert sha(S/'SCALE92.json.gz')==pkg['compressed_sha256'] and gzsha(S/'SCALE92.json.gz')==pkg['uncompressed_sha256']
 a=gzread(S/'SCALE92.json.gz');v=read(S/'INDEPENDENT92.json');assert a['status']==v['status']=='PASS_SCOPED';assert a['paths']==4320 and a['prefix_rows']==10400 and a['B64_paths']==3040 and len(a['strata'])==400 and len(a['temporal'])==120
 inputs={};deferral=[]
 for i in v['audit']:
  L=i['L'];old=read(O/f'AUDIT_L{L}.json');assert i['raw_receipts']==old['raw_receipts']
  for f,k in [(f'OBS_L{L}.npz','array_sha'),(f'SECTOR_L{L}.npz','sector_sha')]:
   p=O/f;assert sha(p)==old[k];inputs[str(p.relative_to(B))]=sha(p)
  inputs[str((O/f'AUDIT_L{L}.json').relative_to(B))]=sha(O/f'AUDIT_L{L}.json')
  assert max(vv for k,vv in i.items() if k.startswith('max_'))<1e-9
 assert v['erratum_sha']==sha(O/'ERRATUM86.json') and v['critical_sha']==sha(O/'CRITICAL87.json')
 for p in [O/'ERRATUM86.json',O/'CRITICAL87.json',P/'DAG.json',P/'CONVERGENCE91.json',P/'QUALITY_MATRIX91.csv',P/'PROGRESS91.json',P/'ORIGINAL_SOURCE91.json',P/'PLOT_INVENTORY91.csv',P/'ORIGINAL_OUTPUT_PROGRESS91.csv']:
  inputs[str(p.relative_to(B))]=sha(p)
 for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
  assert sha(S/f)==sha(P/f)==sha(R/f)
 # Independent aggregate and scale-shell check against retained arrays, plus finite spatial extent identity.
 maxerr=collections.defaultdict(float)
 def err(k,a,b):maxerr[k]=max(maxerr[k],float(np.max(abs(np.asarray(a)-np.asarray(b)))))
 nnonzero=0;bins=np.linspace(0,np.sqrt(2),15);binned=[];summary=[];temporal=[]
 for x in a['strata']:
  L=x['L'];root=np.asarray(x['shell_root_values']);m=np.asarray(x['metric']);q=np.asarray(x['shell_ci']);pr=np.asarray(x['pair_root_count']);pair=np.asarray(x['pairmean'])
  assert root.shape==(8,6,2*(L-1)**2+1) and m.shape==(8,14)
  err('root_aggregate_shell',q,ci(root));err('root_aggregate_metric',x['metric_ci'],ci(m))
  err('conservation',root.sum(-1),0);err('covariance_split',root[:,0],root[:,1]+root[:,2]+root[:,3])
  ds=np.arange(root.shape[-1]);err('geometric_predictable',-root[:,4]@ds/L**2,m[:,8]);err('geometric_uniform',-root[:,5]@ds/L**2,m[:,10]);err('trace',root[:,4,0],m[:,11])
  assert (root[:,4,1:]<1e-11).all() and (root[:,5,1:]<1e-11).all()
  assert (pr>=0).all() and (pr<=8).all() and np.all(pair[:,pr==0]==0)
  if x['bi']==0:
   err('uniform',root[:,4],root[:,5])
  # collision correction from stored means, exact algebra and independently reconstructed R87 X
  nnonzero+=int((pr>0).sum())
  h=x['h'];err('collision',m[:,1],(h*m[:,0]-1)/(h-1))
  # Bin by dimensionless distance; retain signed shell mass and uncertainty at root level.
  rr=np.sqrt(ds)/L;binidx=np.clip(np.digitize(rr,bins)-1,0,len(bins)-2)
  z=np.zeros((8,6,len(bins)-1))
  for j in range(len(bins)-1):
   z[:,:,j]=root[:,:, (binidx==j)&(ds>0)].sum(-1)
  binned.append(dict(L=L,cohort=x['cohort'],arm=x['arm'],bi=x['bi'],beta=x['beta'],budget=x['budget'],h=h,bin_centers=((bins[1:]+bins[:-1])/2).tolist(),bin_ci=ci(z).tolist(),bin_root=z.tolist()))
  summary.append({k:x[k] for k in ['L','cohort','arm','bi','beta','budget','h','roots','reps','metric_ci']})
 # Temporal check from original authenticated SECTOR arrays, reconstruct full covariance and clock-block lag=1.
 for L in [13,15,17,19,21]:
  with np.load(O/f'SECTOR_L{L}.npz') as z:
   meta=z['meta'];p=z['p'];s=z['selected']
  for x in [q for q in a['temporal'] if q['L']==L]:
   mats=[];window=[]
   for c in range(8):
    ix=(meta[:,0]==x['cohort'])&(meta[:,1]==c)&(meta[:,2]==x['arm'])&(meta[:,3]==x['bi'])
    aa=np.eye(3)[s[ix]];pp=p[ix];dd=aa-pp;n=len(aa);assert n==x['reps']
    # Pairwise difference identity, independent of centered einsum in scale92.
    out=[]
    for i,j in __import__('itertools').combinations(range(n),2):
     u=aa[i]-aa[j];w=pp[i]-pp[j];v0=dd[i]-dd[j]
     out.append(np.stack([u@u.T,w@w.T,v0@v0.T,w@v0.T+v0@w.T])/2)
    q=np.mean(out,0);mats.append(q)
    window.append([[q[k,t,t+1] for t in range(63)] for k in range(4)])
   mats=np.array(mats)
   # Each root lag average exactly as registered, compare alternate estimator.
   lag=np.stack([np.diagonal(mats[:,:,4:,4:],offset=l,axis1=2,axis2=3).mean(-1) for l in range(17)],axis=-1)
   startup=np.stack([np.diagonal(mats[:,:,:4,:4],offset=l,axis1=2,axis2=3).mean(-1) for l in range(4)],axis=-1)
   err('temporal_pairwise_lag',lag,x['root_lag']);err('temporal_aggregate',ci(lag),x['lag_ci']);err('startup_aggregate',ci(startup),x['startup_ci'])
   temporal.append(dict(L=L,cohort=x['cohort'],arm=x['arm'],bi=x['bi'],beta=x['beta'],reps=x['reps'],lag_ci=x['lag_ci'],startup_ci=x['startup_ci'],lag1_clock_ci=ci(window).tolist(),representative_matrix=mats.mean(0).tolist() if (L==21 and x['cohort']==2 and x['arm']==1 and x['bi']==3) else None))
 assert max(maxerr.values())<1e-9
 # Actual independent B16, development B64 prefix B16: do not infer paired-seed independent experiment.
 co0={(x['L'],x['arm'],x['bi'],x['budget'],x['h']):x for x in a['strata'] if x['cohort']==0}
 bdiff=[]
 for L in [13,15,17,19,21]:
  for arm in [0,1]:
   for bi in range(4):
    x=co0[L,arm,bi,16,16];y=co0[L,arm,bi,64,16]
    bdiff.append(dict(L=L,arm=arm,bi=bi,independent_B16=x['metric_ci'][0],B64_prefix16=y['metric_ci'][0],difference_ci=ci(np.array(x['metric'])-np.array(y['metric'])).tolist()))
 crit=read(O/'CRITICAL87.json'); assert len(crit['gates'])==8 and sum(r['critical_win'] for r in crit['gates'])==0 and len(crit['crossings'])==12 and sum(bool(r['common_brackets']) for r in crit['crossings'])==0
 # Reconfirm original sources exactly; study full selected text read and hash, preserve only relevant excerpts.
 paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
 orig=read(P/'ORIGINAL_SOURCE91.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
 def cr(p):
  with p.open() as f:return list(csv.DictReader(f))
 cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};qm={x['id']:x for x in cr(P/'QUALITY_MATRIX91.csv')};items=[]
 guide='omoknuni_quantum-main/docs/quantum_physics/STATISTICAL_MECHANICS_ANALYSIS_GUIDE.md'
 with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
  g=z.read(guide);gh=hashlib.sha256(g).hexdigest();ls=g.decode().splitlines()
  for k in read(S/'REGISTRATION.json')['targets']:
   q=qm[k];x=cat[k]
   if k.startswith('G'):
    assert gh==q['source_sha'];line=int(q['source_span']);items.append(dict(id=k,archive='omoknuni_quantum-main.zip',member=guide,member_sha=gh,span=[line,line],source_excerpt=ls[line-1],acceptance=q['final_acceptance']))
   else:
    with zipfile.ZipFile(paths[x['archive']]) as z0:
     b=z0.read(x['member']);hh=hashlib.sha256(b).hexdigest();assert hh==x['member_sha256'];lines=b.decode().splitlines();lo=int(x['start']);hi=int(x['end']);ss='\n'.join(lines[lo-1:hi]);items.append(dict(id=k,archive=x['archive'],member=x['member'],member_sha=hh,span=[lo,hi],span_sha=hashlib.sha256(ss.encode()).hexdigest(),source_excerpt=ss,acceptance=q['final_acceptance']))
 study=[]
 with zipfile.ZipFile(paths['mcts_study.zip']) as z:
  assert z.testzip() is None
  for k in ['mcts_study/v3.0/qft-mcts-statistical.md','mcts_study/v3.0/quantum-mcts-validation-plan.md','mcts_study/v5.0/COMPREHENSIVE_PLOT_DOCUMENTATION.md','mcts_study/v6.0/quantum_mcts_foundation.md','mcts_study/v6.0/quantum_mcts_validation.md']:
   b=z.read(k);ls=b.decode(errors='strict').splitlines();matches=[(i+1,v) for i,v in enumerate(ls) if any(t in v.lower() for t in ['correlation length','finite-size','finite size','critical','order parameter','susceptibility','상관','임계','유한크기','질서매개'])]
   study.append(dict(member=k,sha256=hashlib.sha256(b).hexdigest(),lines=len(ls),relevant_line_count=len(matches),excerpts=matches[:30]))
  members=len([n for n in z.namelist() if not n.endswith('/')])
 dump(S/'ORIGINAL_SOURCE92.json',dict(archives=orig,guide_sha=gh,items=items,study_members=members,study_selected=study,authority='R69 original archives, exact source spans and study notes reread; original folder as archived, not separate live directory'))
 maxoff=max(max(abs(np.asarray(x['lag_ci'])[0,2,1:])) for x in temporal);point=sum(int(np.sum((np.asarray(x['lag_ci'])[1,2,1:]>0)|(np.asarray(x['lag_ci'])[2,2,1:]<0))) for x in temporal)
 out=dict(status='PASS_SCOPED',strata=summary,binned=binned,temporal=temporal,budget_comparison=bdiff,metrics=a['metric_fields'],shell_fields=a['shell_fields'],temporal_fields=a['temporal_fields'],checks=dict(maxerr),source_hashes=inputs,raw_paths=a['paths'],prefix_rows=a['prefix_rows'],time_paths=a['B64_paths'],scope='Same-agent independent aggregate, direct path and pairwise temporal checks. Root t intervals descriptive, 8 matched cores per stratum, not global simultaneous CIs.',software={'python':platform.python_version(),'numpy':np.__version__},critical=dict(gates=crit['gates'],crossings=crit['crossings']),temporal_max_offlag_innovation=float(maxoff),temporal_pointwise_exclusions=point,temporal_pointwise_total=1920)
 dump(S/'SUMMARY92.json',out)
 print('VERIFY',dict(maxerr),'offlag',maxoff,point,'critical 0/8,0/12',flush=True)
if __name__=='__main__':main()
