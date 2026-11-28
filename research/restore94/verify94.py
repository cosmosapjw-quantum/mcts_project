from instrument81 import *
import csv,zipfile,platform
from scipy.special import xlogy
O=B/'mqft_restore93_20261128';A=B/'mqft_restore81_20261127';C=B/'mqft_restore83_20261127';R=B/'mqft_restore69_20261126'
n=read(S/'NATIVE94.json');r=read(S/'RECORD94.json');assert n['status']==r['status']=='PASS_SCOPED' and sha(S/'NATIVE94.npz')==n['array_sha']
assert n['raw_receipts']==({x['path']:x['sha256'] for x in read(A/'AUDIT81.json')['raw_receipts']}|{x['path']:x['sha256'] for x in read(C/'AUDIT83.json')['raw_receipts']})
with np.load(S/'NATIVE94.npz') as z:
 m=z['meta'];v=z['states'];f=z['features'];p=z['participation'];st=z['stability']
 assert m.shape==(3456,4) and len(set(map(tuple,m)))==3456 and v.shape==(3456,12,3)
 assert np.min(p[:,:,:3])>=1-1e-12 and np.max(p[:,:,:3]-p[:,:,3,None])<1e-10
 assert np.max(p[:,:,2]-np.arange(1,129)[None,:])<1e-10
 q=np.linalg.norm(v,axis=-1);assert q.max()<=1+1e-10
 e=(1+np.clip(q,0,1))/2;S0=-xlogy(e,e)-xlogy(1-e,1-e)
 e0=(1+v[:,:,2])/2;d=-xlogy(e0,e0)-xlogy(1-e0,1-e0)
 err=dict(entropy=float(abs(S0-f[:,:,8]).max()),coherence=float(abs(np.linalg.norm(v[:,:,:2],axis=-1)-f[:,:,2]).max()),relative_coherence=float(abs(d-S0-f[:,:,5]).max()),purity=float(abs((1+q*q)/2-f[:,:,21]).max()),entropy_split=float(abs(f[:,:,14]-f[:,:,11]+f[:,:,12]).max()),frozen_optimal=float(max(0,np.max(st[:,6:9]-st[:,0,None]))))
 assert max(err.values())<1e-9 and np.min(f[:,:,14])>=-1e-10 and np.min(f[:,:,12])>=-1e-10
 # explicit negative example: a pair of successive noncommuting local dephasing maps
 # pointer of first not fixed by second, with entropy-minimizing final directions determined by singular values.
 Gx=np.diag([1.,.6,.6]);Gz=np.diag([.7,.7,1.]);T=Gz@Gx
 assert np.linalg.norm(T@np.array([1.,0.,0.]))==.7 and np.linalg.norm(T@np.array([0.,0.,1.]))==.6
 assert np.linalg.svd(T,compute_uv=False)[0]==.7
 # independent check of observed phase comparison to R90 and coherent QEC control.
 p90=B/'mqft_restore90_20261128'/'PHASE90.json';old=read(p90)
 for x,y in zip(n['families'],old['families']):
  assert x['family']==y['family'] and abs(np.array(x['phase_first_signed_ci'])-y['first_ci']).max()<1e-10 and abs(np.array(x['phase_active_signed_ci'])-y['active_ci']).max()<1e-10
 q89=B/'mqft_restore89_20261128'/'QEC89.json';q=read(q89);assert q['status']=='PASS_SCOPED'
 control=next(x for x in q['coherent'] if abs(x['theta']-.2)<1e-12)
 assert abs(control['results']['known_angle']['100']-1)<1e-10
 p92=B/'mqft_restore92_20261128'/'SUMMARY92.json';q92=read(p92);assert q92['status']=='PASS_SCOPED'
 # Original input audit
 paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
 orig=read(O/'ORIGINAL_SOURCE93.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
 for k in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/k)==sha(O/k)==sha(R/k)
 qm={x['id']:x for x in csv.DictReader(open(O/'QUALITY_MATRIX93.csv'))};items=[];guide='omoknuni_quantum-main/docs/quantum_physics/QUANTUM_PHENOMENA_ANALYSIS_GUIDE.md';source='omoknuni_quantum-main/python/mcts/quantum/phenomena/decoherence.py'
 with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
  b=z.read(guide);gh=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  for k in read(S/'REGISTRATION.json')['targets']:
   x=qm[k];assert gh==x['source_sha'];i=int(x['source_span'])
   items.append(dict(id=k,archive='omoknuni_quantum-main.zip',member=guide,member_sha=gh,span=[i,i],source_excerpt=ls[i-1],acceptance=x['final_acceptance']))
  spec=[dict(line=i+1,text=ls[i]) for i in range(39,75)]
  bs=z.read(source);ss=bs.decode().splitlines();legacy=dict(member=source,sha256=hashlib.sha256(bs).hexdigest(),selected=[dict(line=i+1,text=ss[i]) for i in list(range(180,227))+list(range(356,487))])
 study=[]
 with zipfile.ZipFile(paths['mcts_study.zip']) as z:
  assert z.testzip() is None
  for k in ['mcts_study/v3.0/qft-mcts-decoherence.md','mcts_study/v5.0/COMPREHENSIVE_PLOT_DOCUMENTATION.md','mcts_study/v6.0/quantum_mcts_foundation.md','mcts_study/v6.0/quantum_mcts_validation.md']:
   b=z.read(k);ls=b.decode().splitlines();match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['coherence','decoherence','entropy','pointer','participation','darwinism'])]
   study.append(dict(member=k,sha256=hashlib.sha256(b).hexdigest(),lines=len(ls),relevant_line_count=len(match),excerpts=match[:30]))
  members=len([k for k in z.namelist() if not k.endswith('/')])
 dump(S/'ORIGINAL_SOURCE94.json',dict(archives=orig,guide_sha=gh,items=items,guide_features=spec,legacy_code=legacy,study_members=members,study_selected=study,authority='R69 locked original archives; no separate live mcts_study folder at archived Dropbox path'))
 inp=[A/'AUDIT81.json',A/'PATHS81.npz',A/'EFFECTS81.json',C/'AUDIT83.json',C/'TREE83.json',C/'RECORD83.json',C/'RECORD_MATRICES83.npz',C/'FRAGMENTS83.json',C/'instrument81.py',p90,q89,p92,*[O/k for k in ['DAG.json','PROGRESS93.json','QUALITY_MATRIX93.csv','CONVERGENCE93.json','PLOT_INVENTORY93.csv','ORIGINAL_OUTPUT_PROGRESS93.csv','ORIGINAL_SOURCE93.json']]]
 assert sha(S/'instrument81.py')==sha(C/'instrument81.py')
 dump(S/'INDEPENDENT94.json',dict(status='PASS_SCOPED',checks=err,source_hashes={str(p.relative_to(B)):sha(p) for p in inp},R89_control={'theta':.2,'results':control['results'],'scope':'separate declared coherent repetition channel; not R81 decoherence or physical calibration'},R92_comparator={'raw_paths':q92['raw_paths'],'scope':'native categorical covariance separate population'},noncommuting_example={'singular_values':np.linalg.svd(T,compute_uv=False).tolist(),'scope':'finite map; local pointer differs by axis'},software={'python':platform.python_version(),'numpy':np.__version__},independence='Same-agent scalar/eigenvalue, matrix and purification checks; no independent subagent.'))
 print('VERIFY',err)
