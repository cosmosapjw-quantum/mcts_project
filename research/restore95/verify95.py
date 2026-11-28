from entangle95 import *
import csv,zipfile,platform
from scipy.stats import t as student
P94=B/'mqft_restore94_20261128';R69=B/'mqft_restore69_20261126';O73=B/'mqft_restore73_20261126'
def ci(x):
 x=np.array(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.stack([m,m-h,m+h]).tolist()
def main():
 a=read(S/'ENTANGLE95.json');assert a['status']=='PASS_SCOPED' and sha(S/'JOINT95.npz')==a['joint_sha']
 with np.load(S/'JOINT95.npz') as z:D={k:z[k] for k in z.files}
 assert D['meta'].shape==(576,4)
 ix={(int(g),int(c),p,m):i for i,(g,c,p,m) in enumerate(D['meta'])};assert len(ix)==576
 # Rebuild all controlled states; direct index reduction, independently assemble PT and Pauli coefficient tensor.
 # Pairwise data and entire cut negativity checked separately; record-only marginal distinguished from joint.
 err=collections.defaultdict(float)
 def ck(k,x,y,tol=1e-9):
  e=norm(np.asarray(x)-np.asarray(y));err[k]=max(err[k],e);assert e<tol,(k,e)
 def transpose_bits(a,cut,n):
  # independently swap row/column bit mask at each subsystem
  z=np.arange(1<<n);i,j=np.meshgrid(z,z,indexing='ij');mask=sum(1<<(n-1-k) for k in cut);v=(i^j)&mask
  return a[i^v,j^v]
 r83=read(O/'RECORD83.json');recmap={(q['family'],q['core']):q for q in r83['contexts']}
 rcount=0;summary=[]
 for row in a['rows']:
  g,c,p,m,d,xi=[row[k] for k in ['family','core','protocol','mode','depth','xi']];J=D['J'+str(d)][ix[g,c,p,m]];n=d+1
  bits=np.array(list(product(range(2),repeat=d)));dist=np.sum(bits[:,None,:]!=bits[None,:,:],-1);J=J*np.kron(xi**dist,np.ones((2,2)))
  for j,pair in enumerate(row['pairs']):
   u=red_direct(J,pair,n);w=np.asarray(row['metrics'][j]);q=transpose_bits(u,[0],2);e=np.linalg.eigvalsh(q)
   ck('pair_PT',-e[e<0].sum(),w[2])
   p0=red_direct(u,[0],2);p1=red_direct(u,[1],2);ck('pair_QMI',vn(p0)+vn(p1)-vn(u),w[0])
   T=np.array([[np.vdot(np.kron(x,y),u).real for y in PAULI] for x in PAULI]);eig=np.linalg.eigvalsh(T.T@T)
   ck('CHSH_Horodecki_eigen',2*np.sqrt(max(0,eig[-2:].sum())),w[5])
   ck('weighted',w[6:],row['success']*np.array([w[2],w[3],max(w[5]-2,0)]))
   # diagonal measurement probabilities unchanged under record tagging; verify separately
   ref=red_direct(D['J'+str(d)][ix[g,c,p,m]],pair,n);ck('Z_diagonal_control',ref.diagonal(),u.diagonal())
  for cut,val in zip(row['cuts'],row['cut_negativity']):
   e=np.linalg.eigvalsh(transpose_bits(J,cut,n));ck('cut_PT_independent',-e[e<0].sum(),val)
  if d==3 and xi==1 and m=='plus81':
   R0=red_direct(J,[0,1,2],4);v=recmap[g,c]['protocols'][p]
   for k,s in enumerate(v['splits']):
    e=np.linalg.eigvalsh(transpose_bits(R0,[k],3));ck('R83_record_only_cut',-e[e<0].sum(),s['negativity']);rcount+=1
 assert rcount==432
 # verify original inputs and source spans
 paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
 orig=read(P94/'ORIGINAL_SOURCE94.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
 for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(P94/f)==sha(R69/f)
 qm={x['id']:x for x in csv.DictReader(open(P94/'QUALITY_MATRIX94.csv'))};items=[];guide='omoknuni_quantum-main/docs/quantum_physics/QUANTUM_PHENOMENA_ANALYSIS_GUIDE.md';code='omoknuni_quantum-main/python/mcts/quantum/phenomena/entanglement.py'
 with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
  b=z.read(guide);gh=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  for k in read(S/'REGISTRATION.json')['targets']:
   x=qm[k];assert gh==x['source_sha'];i=int(x['source_span']);items.append(dict(id=k,archive='omoknuni_quantum-main.zip',member=guide,member_sha=gh,span=[i,i],source_excerpt=ls[i-1],acceptance=x['final_acceptance']))
  bs=z.read(code);legacy=dict(member=code,sha256=hashlib.sha256(bs).hexdigest(),source=bs.decode())
  spec=[dict(line=i+1,text=ls[i]) for i in list(range(107,145))+list(range(366,371))]
 study=[]
 with zipfile.ZipFile(paths['mcts_study.zip']) as z:
  assert z.testzip() is None
  for k in ['mcts_study/v3.0/qft-mcts-decoherence.md','mcts_study/v5.0/COMPREHENSIVE_PLOT_DOCUMENTATION.md','mcts_study/v6.0/quantum_mcts_foundation.md','mcts_study/v6.0/quantum_mcts_validation.md']:
   b=z.read(k);ls=b.decode().splitlines();match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['entanglement','concurrence','bell','mutual information','negativity','얽힘','상호정보'])]
   study.append(dict(member=k,sha256=hashlib.sha256(b).hexdigest(),lines=len(ls),relevant_line_count=len(match),excerpts=match[:30]))
  members=len([k for k in z.namelist() if not k.endswith('/')])
 dump(S/'ORIGINAL_SOURCE95.json',dict(archives=orig,guide_sha=gh,items=items,guide_features=spec,legacy_code=legacy,study_members=members,study_selected=study,authority='R69 original archive byte identity; archived mcts_study; separate live folder not used'))
 # Explicitly distinguish native R94 state/phase and R73 prior compatible fine branch construction.
 q=read(O73/'INSTRUMENT73.json');assert q['branch_paths']==4544 and len(q['contexts'])==64
 n94=read(P94/'NATIVE94.json');assert n94['status']=='PASS_SCOPED'
 inp=[A/'AUDIT81.json',O/'AUDIT83.json',O/'TREE83.json',O/'RECORD83.json',O/'RECORD_MATRICES83.npz',P94/'RECORD94.json',P94/'NATIVE94.json',P94/'instrument81.py',O73/'INSTRUMENT73.json',O73/'RESULTS_KO.md',*[P94/k for k in ['DAG.json','PROGRESS94.json','QUALITY_MATRIX94.csv','CONVERGENCE94.json','PLOT_INVENTORY94.csv','ORIGINAL_OUTPUT_PROGRESS94.csv','ORIGINAL_SOURCE94.json']]]
 assert sha(S/'instrument81.py')==sha(P94/'instrument81.py')
 # Compact group-wise statistics; actual pair labels and cut masks kept.
 for g,p,m,d,xi in product(range(3),PROTOCOLS,MODES,range(1,4),XI):
  x=[r for r in a['rows'] if (r['family'],r['protocol'],r['mode'],r['depth'],r['xi'])==(g,p,m,d,xi)];assert len(x)==24
  v=np.array([r['metrics'] for r in x]);c=np.array([r['cut_negativity'] for r in x]);s=np.array([r['success'] for r in x])
  summary.append(dict(family=FAMS[g],protocol=p,mode=m,depth=d,xi=xi,labels=x[0]['labels'],pairs=x[0]['pairs'],cuts=x[0]['cuts'],metrics_ci=ci(v),cut_ci=ci(c),success_median=float(np.median(s)),pair_NPT=int((v[:,:,2]>1e-10).sum()),pair_C=int((v[:,:,3]>1e-7).sum()),pair_CHSH=int((v[:,:,5]>2+1e-9).sum()),cut_NPT=int((c>1e-10).sum()),contexts_any_pair_NPT=int((v[:,:,2].max(1)>1e-10).sum()),contexts_any_cut_NPT=int((c.max(1)>1e-10).sum()),max_CHSH=float(v[:,:,5].max()),max_C=float(v[:,:,3].max()),max_weighted_CHSH=float(v[:,:,8].max()),mean_weighted_CHSH=float(v[:,:,8].mean()),max_weighted_N=float(v[:,:,6].max())))
 dump(S/'SUMMARY95.json',dict(groups=summary,fields=a['fields'],scope='24 root contexts per family, pointwise descriptive t intervals, no population transfer; current prefix success not final selected.'))
 dump(S/'INDEPENDENT95.json',dict(status='PASS_SCOPED',checks=dict(err),source_hashes={str(x.relative_to(B)):sha(x) for x in inp},R73_comparator={'contexts':len(q['contexts']),'branch_paths':q['branch_paths'],'scope':'prior distinct compatible fine branch construction, not re-executed'},R94_phase_comparator=[dict(family=x['family'],phase_first_signed_ci=x['phase_first_signed_ci'],recommend_disagreement=x['recommend_disagreement']) for x in n94['families']],R83_record_only_cut_checks=rcount,software={'python':platform.python_version(),'numpy':np.__version__},independence='Same-agent independent explicit bit-index PT, reduced states, eigen CHSH; no native subagent'))
 print('VERIFY',dict(err),'groups',len(summary),flush=True)
if __name__=='__main__':main()
