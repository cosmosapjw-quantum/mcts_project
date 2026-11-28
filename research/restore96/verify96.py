from darwin96 import *
import csv,zipfile,platform
from scipy.stats import t as student
R69=B/'mqft_restore69_20261126'
def direct(a,keep,n):
 rest=[i for i in range(n) if i not in keep];d=1<<len(keep);e=1<<len(rest)
 idx=np.zeros((d,e),int)
 for i,j in product(range(d),range(e)):
  for b,k in enumerate(keep):idx[i,j]|=((i>>(len(keep)-1-b))&1)<<(n-1-k)
  for b,k in enumerate(rest):idx[i,j]|=((j>>(len(rest)-1-b))&1)<<(n-1-k)
 return sum(a[ix[:,None],ix[None,:]] for ix in idx.T)
def ci(x):
 x=np.array(x,float);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.zeros_like(m);return np.stack([m,m-h,m+h]).tolist()
def main():
 a=read(S/'DARWIN96.json');assert a['status']=='PASS_SCOPED';pr=read(P/'ENTANGLE95.json');assert a['raw_receipts']==pr['raw_receipts'];assert sha(P/'JOINT95.npz')==pr['joint_sha']
 with np.load(P/'JOINT95.npz') as z:D={k:z[k] for k in z.files}
 ix={(int(g),int(c),p,m):i for i,(g,c,p,m) in enumerate(D['meta'])};assert len(ix)==576
 err=collections.defaultdict(float);nfrag=0
 def ck(k,x,y,tol=1e-9):
  e=norm(np.asarray(x)-np.asarray(y));err[k]=max(err[k],e);assert e<tol,(k,e)
 for row in a['rows']:
  g,c,p,m,d,xi,s=[row[k] for k in ['family','core','protocol','mode','depth','xi','success']];J=D['J'+str(d)][ix[g,c,p,m]];n=d+1
  bits=np.arange(2**n);J=J*np.array([[xi**(((i^j)>>1).bit_count()) for j in bits] for i in bits]);assert abs(np.trace(J)-1)<1e-9
  for q in row['pointers']:
   ptr=0 if q['pointer']=='first_record' else d;fs=list(range(1,d)) if ptr==0 else list(range(d))
   for f in q['fragments']:
    sub=direct(J,[ptr]+f['fragment'],n);k=2**len(f['fragment']);w=np.stack([sub[:k,:k],sub[k:,k:]]);probs=np.trace(w,axis1=1,axis2=2).real
    ck('entropy_pointer',float(ent(probs)),f['H']);ck('classical_Z',MI(np.diagonal(w,axis1=1,axis2=2).real),f['classical'])
    if probs.min()>=1e-15:
     r=w/probs[:,None,None];avg=w.sum(0);chi=vn(avg)-sum(probs[i]*vn(r[i]) for i in range(2));ck('chi',chi,f['chi'])
     D0=.5*np.linalg.svd(r[0]-r[1],compute_uv=False).sum();ck('trace_distance',D0,f['trace_distance']);ck('upper',min(chi,f['H']*D0),f['upper'])
     # Reconstruct the reported attaining POVM, independent of grid search.
     kind=f['kind']
     if kind=='Z':j=np.diagonal(w,axis1=1,axis2=2).real
     elif kind=='sphere_projective':
      n0=np.array(f['axis']);ck('direction_norm',np.linalg.norm(n0),1)
      E=(I+np.einsum('i,ijk->jk',n0,PAULI))/2;j=np.stack([[np.trace(x@E).real,np.trace(x@(I-E)).real] for x in w])
     else:
      Q=w[0]-w[1] if kind.startswith('prior') else r[0]-r[1];ev,v=np.linalg.eigh(Q)
      j=np.stack([np.diag(v.conj().T@x@v).real for x in w])
      if kind.endswith('binary'):j=np.stack([j[:,ev>=0].sum(1),j[:,ev<0].sum(1)],1)
     ck('measurement_norm',j.sum(),1);ck('attained_information',MI(j),f['lower'])
    ck('weighted',np.array([f['weighted_lower'],f['weighted_upper']]),s*np.array([f['lower'],f['upper']]))
    nfrag+=1
   if q['packing']:
    for frac,got in zip(a['thresholds'],q['packing']):
     vals=[]
     for key,eps in [('lower',1e-9),('upper',-1e-9)]:
      valid=[set(f['fragment']) for f in q['fragments'] if f[key]>=frac*q['H']+eps]
      best=max([len(v) for k in range(len(valid)+1) for v in combinations(valid,k) if len(set().union(*v))==sum(map(len,v))],default=0)
      vals.append(best)
     assert vals==got
 assert nfrag==25920
 # Exact source records, actual software phase comparator and nonpooled classical predictive comparator.
 au=read(A/'AUDIT81.json');n94=read(R94/'NATIVE94.json')
 assert sha(A/'PATHS81.npz')==au['array_sha'] and sha(R94/'NATIVE94.npz')==n94['array_sha']
 r81={x['path']:x['sha256'] for x in au['raw_receipts']};assert len(r81)==3456
 allr=dict(a['raw_receipts']);assert all(allr[k]==v for k,v in r81.items() if k in allr);more={k:v for k,v in r81.items() if k not in allr}
 with cf.ProcessPoolExecutor(max_workers=10) as pool:allr.update(dict(pool.map(rehash,more.items(),chunksize=16)))
 assert len(allr)==4104
 with np.load(A/'PATHS81.npz') as z:m=z['meta'];act=z['actions'];src=z['sources'];counts=z['counts'];states=z['states']
 with np.load(R94/'NATIVE94.npz') as z:
  lookup={tuple(v):i for i,v in enumerate(z['meta'])};ck('R94_states',states,z['states'][[lookup[tuple(v)] for v in m]])
 idx={tuple(v):i for i,v in enumerate(m)};native=[]
 for g in range(3):
  ix0=np.array([[[idx[g,c,r,a] for a in range(6)] for r in range(8)] for c in range(24)]);a0=act[ix0];p0=src[ix0];rec=counts[ix0].argmax(-1)
  sg=(a0==p0[...,0,None]).astype(float)-(a0==p0[...,1,None]).astype(float)
  first=ci((sg[:,:,0,32]-sg[:,:,1,32]).mean(1));active=ci((sg[:,:,0,32:44]-sg[:,:,1,32:44]).mean((1,2)));dis=int((rec[:,:,0]!=rec[:,:,1]).sum())
  old=n94['families'][g];ck('phase_first',first,old['phase_first_signed_ci']);ck('phase_active',active,old['phase_active_signed_ci']);assert dis==old['recommend_disagreement']
  # separate actual categorical choice and final recommendation, not identity with model pointer
  c1=(a0[:,:,0,32]==p0[:,:,0,0]).astype(int)+2*(a0[:,:,0,32]==p0[:,:,0,1]).astype(int);c2=(rec[:,:,0]==p0[:,:,0,0]).astype(int)+2*(rec[:,:,0]==p0[:,:,0,1]).astype(int)
  table=np.zeros((3,3),int)
  for x,y in zip(c1.ravel(),c2.ravel()):table[x,y]+=1
  native.append(dict(family=FAMS[g],first_signed=first,active_signed=active,recommend_disagreement=dis,actual_first_vs_recommendation_table=table.tolist(),labels=['other','source_a','source_b'],scope='descriptive joint counts, heterogeneous root blocks; no plug-in information generalization'))
 # source authority and exact function spans
 paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
 orig=read(P/'ORIGINAL_SOURCE95.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
 for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(P/f)==sha(R69/f)
 qm={x['id']:x for x in csv.DictReader(open(P/'QUALITY_MATRIX95.csv'))};items=[];code='quantum/research/visualization/plot_decoherence_darwinism.py'
 with zipfile.ZipFile(paths['quantum.zip']) as z:
  b=z.read(code);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  for k in read(S/'REGISTRATION.json')['targets']:
   x=qm[k];assert h==x['source_sha'];l,r=map(int,x['source_span'].split('-'));assert x['source_function'] in ls[l-1]
   items.append(dict(id=k,archive='quantum.zip',member=code,member_sha=h,span=[l,r],source_function=x['source_function'],source_excerpt=ls[l-1:r],acceptance=x['final_acceptance']))
 study=[]
 with zipfile.ZipFile(paths['mcts_study.zip']) as z:
  assert z.testzip() is None
  for k in ['mcts_study/v3.0/qft-mcts-decoherence.md','mcts_study/v5.0/COMPREHENSIVE_PLOT_DOCUMENTATION.md','mcts_study/v6.0/quantum_mcts_foundation.md','mcts_study/v6.0/quantum_mcts_validation.md']:
   b=z.read(k);ls=b.decode().splitlines();match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['darwinism','redundancy','objectivity','pointer','holevo','fragment','decoherence'])]
   study.append(dict(member=k,sha256=hashlib.sha256(b).hexdigest(),lines=len(ls),relevant_line_count=len(match),excerpts=match[:30]))
 dump(S/'ORIGINAL_SOURCE96.json',dict(archives=orig,items=items,study_selected=study,authority='R69 archive identities and exact five original function spans; archived mcts_study; original source not edited'))
 r73=read(R73/'DECISION_RESULTS73.json');assert len(r73['results'])>0
 # compact groups with per-context packing intervals, no ratios for H<=1e-6
 groups=[]
 for g,p,m,d,xi,name in product(range(3),['native_repeat','pair_diagnostic'],['plus81','minus81','eraseplus81','base81'],range(1,4),[0.,.5,1.],['first_record','system_Z']):
  xs=[(r,q) for r in a['rows'] if (r['family'],r['protocol'],r['mode'],r['depth'],r['xi'])==(g,p,m,d,xi) for q in r['pointers'] if q['pointer']==name]
  if not xs:continue
  valid=[(r,q) for r,q in xs if q['packing']];assert len(xs)==24
  counts=[dict(lower_ge2=sum(q['packing'][k][0]>=2 for r,q in valid),upper_ge2=sum(q['packing'][k][1]>=2 for r,q in valid),lower_ge1=sum(q['packing'][k][0]>=1 for r,q in valid),upper_ge1=sum(q['packing'][k][1]>=1 for r,q in valid)) for k in range(3)]
  fields=['classical','lower','upper','chi','QMI']
  ratios=[];absinfo=[]
  for i in range(len(xs[0][1]['fragments'])):
   vv=[q['fragments'][i] for _,q in xs];absinfo.append(ci([[f[k] for k in fields] for f in vv]))
   ratios.append(ci([[q['fragments'][i][k]/q['H'] for k in fields] for _,q in valid]) if valid else None)
  groups.append(dict(family=FAMS[g],protocol=p,mode=m,depth=d,xi=xi,pointer=name,labels=xs[0][1]['labels'],fragments=[v['fragment'] for v in xs[0][1]['fragments']],valid=len(valid),degenerate=len(xs)-len(valid),packing_counts=counts,absolute_ci=absinfo,ratio_ci=ratios,H_ci=ci([q['H'] for r,q in xs]),pointer_dephase_ci=ci([q['pointer_dephase_TD'] for r,q in xs]),success_median=float(np.median([r['success'] for r,q in xs])),largest_product_TD_ci=ci([q['fragments'][-1]['conditional_product_TD'] for r,q in xs]) if len(xs[0][1]['fragments'][-1]['fragment'])>1 else None,weighted_largest_ci=ci([[q['fragments'][-1]['weighted_lower'],q['fragments'][-1]['weighted_upper']] for r,q in xs])))
 dump(S/'SUMMARY96.json',dict(groups=groups,fields=['classical','lower','upper','chi','QMI'],thresholds=a['thresholds'],native=native,R73_comparator=dict(scope=r73['scope'],results=r73['results']),scope='Root-block descriptive intervals, accessible information bracketed, exact disjoint packing; finite declared model only.'))
 inp=[P/'ENTANGLE95.json',P/'JOINT95.npz',P/'BASE_CONTROL95.json',A/'AUDIT81.json',A/'PATHS81.npz',R83/'FRAGMENTS83.json',R94/'RECORD94.json',R94/'NATIVE94.json',R94/'NATIVE94.npz',R73/'DECISION_RESULTS73.json',*[P/k for k in ['DAG.json','PROGRESS95.json','QUALITY_MATRIX95.csv','CONVERGENCE95.json','PLOT_INVENTORY95.csv','ORIGINAL_OUTPUT_PROGRESS95.csv','ORIGINAL_SOURCE95.json']]]
 dump(S/'INDEPENDENT96.json',dict(status='PASS_SCOPED',checks=dict(err),source_hashes={str(x.relative_to(B)):sha(x) for x in inp},raw_receipts=allr,unique_raw=4104,fragment_rechecks=nfrag,software={'python':platform.python_version(),'numpy':np.__version__},independence='Same-agent bit-index reduction, measurement probabilities and exhaustive subset packing; no native subagent'))
 print('VERIFY',dict(err),'groups',len(groups),'unique',len(allr),flush=True)
if __name__=='__main__':main()
