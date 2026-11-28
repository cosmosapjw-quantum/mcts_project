from base61 import *
import csv,zipfile,platform
from fractions import Fraction as F
from scipy.special import expit
P=B/'mqft_restore92_20261128';R69=B/'mqft_restore69_20261126';O74=B/'mqft_restore74_20261126';O66=B/'mqft_restore66_20261126';O90=B/'mqft_restore90_20261128';O87=B/'mqft_restore87_20261128'
def main():
 a=read(S/'RESPONSE93.json');assert a['status']=='PASS_SCOPED' and len(a['raw_receipts'])==5376
 inputs={}
 for p in [O74/'AUDIT74.json',O74/'PULSE74.npz',O74/'RESPONSE74.json',O66/'AUDIT66.json',O66/'NATIVE66.npz',O66/'PHYSICS_RESULTS.json',O90/'RESPONSE90.json',O87/'ERRATUM86.json',P/'SUMMARY92.json',P/'DAG.json',P/'PROGRESS92.json',P/'QUALITY_MATRIX92.csv',P/'CONVERGENCE92.json',P/'PLOT_INVENTORY92.csv',P/'ORIGINAL_OUTPUT_PROGRESS92.csv',P/'ORIGINAL_SOURCE92.json']:
  inputs[str(p.relative_to(B))]=sha(p)
 r=read(O74/'AUDIT74.json');t=read(O66/'AUDIT66.json');assert a['raw_receipts']==({x['path']:x['sha256'] for x in r['raw_receipts']}|{str((O66/'data'/x['tag']/(x['job']['id']+'.json.gz')).relative_to(B)):x['sha256'] for x in t['jobs']})
 assert sha(O74/'PULSE74.npz')==r['array_sha256'] and sha(O66/'NATIVE66.npz')==t['array_sha']
 # Separate explicitly declared equilibrium reference: frozen Bernoulli source probability at H8.
 # P_ij=pi_j, reversible, stationary, no physical time/temperature calibration.
 q=[];er=collections.defaultdict(float)
 with np.load(O74/'PULSE74.npz') as z:
  m=z['meta'];f=z['features'];ix=m[:,3]==0
  for i in np.flatnonzero(ix):
   p=float(f[i,8,2]);pi=np.array([1-p,p]);K=np.tile(pi,(2,1));flow=pi[:,None]*K
   e=np.array([0.,1.]);var=pi@(e-p)**2
   er['stationarity']=max(er['stationarity'],float(abs(pi@K-pi).max()))
   er['detailed_balance']=max(er['detailed_balance'],float(abs(flow-flow.T).max()))
   er['variance']=max(er['variance'],abs(float(var)-p*(1-p)))
   th=1e-5;h=lambda d:expit(np.log(p/(1-p))+d)
   er['static_fd']=max(er['static_fd'],abs((h(th)-h(-th))/(2*th)-var))
   er['lag1_cov']=max(er['lag1_cov'],abs(float((pi*(e-p))@K@(e-p))))
   q.append((int(m[i,0]),p,var))
 assert len(q)==768 and max(er.values())<1e-8
 # Exact rational two-wave adaptive example deliberately nonzero mixed derivative.
 p0=F(3,10);p1=[F(1,5),F(7,10)];resp=[F(0),F(0)];H=[[F(0),F(0)],[F(0),F(0)]]
 for x in [0,1]:
  for y in [0,1]:
   w=(p0 if x else 1-p0)*(p1[x] if y else 1-p1[x]);s=[x-p0,y-p1[x]];v=[p0*(1-p0),p1[x]*(1-p1[x])]
   for i in [0,1]:
    resp[i]+=w*y*s[i]
    for j in [0,1]:H[i][j]+=w*y*(s[i]*s[j]-(v[i] if i==j else 0))
 expect=[p0*(1-p0)*(p1[1]-p1[0]),(1-p0)*p1[0]*(1-p1[0])+p0*p1[1]*(1-p1[1])]
 cross=p0*(1-p0)*(p1[1]*(1-p1[1])-p1[0]*(1-p1[0]))
 assert resp==expect and H[0][1]==H[1][0]==cross and cross!=0
 # Independent finite-mixture identity with exact rational F/C histories, same alphabet, positive support.
 f=[F(1,10),F(2,10),F(3,10),F(4,10)];c=[F(4,10),F(3,10),F(2,10),F(1,10)];y=[-1,0,2,5]
 mix=[(a+b)/2 for a,b in zip(f,c)];wf=[2*a/(a+b) for a,b in zip(f,c)];wc=[2*b/(a+b) for a,b in zip(f,c)]
 assert all(0<w<2 for w in wf+wc) and sum(a*b for a,b in zip(mix,wf))==sum(a*b for a,b in zip(mix,wc))==1
 assert sum(m*(a-b)*v for m,a,b,v in zip(mix,wf,wc,y))==sum((a-b)*v for a,b,v in zip(f,c,y))
 # Original source contract and archived study notes. Read relevant full entries, retain excerpts.
 paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
 orig=read(P/'ORIGINAL_SOURCE92.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
 for k in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
  assert sha(S/k)==sha(P/k)==sha(R69/k)
 qm={x['id']:x for x in csv.DictReader(open(P/'QUALITY_MATRIX92.csv'))};items=[]
 guide='omoknuni_quantum-main/docs/quantum_physics/STATISTICAL_MECHANICS_ANALYSIS_GUIDE.md'
 with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
  b=z.read(guide);gh=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  for k in ['G10','G11','G12']:
   x=qm[k];assert gh==x['source_sha'];i=int(x['source_span'])
   items.append(dict(id=k,archive='omoknuni_quantum-main.zip',member=guide,member_sha=gh,span=[i,i],source_excerpt=ls[i-1],acceptance=x['final_acceptance']))
  spec=[dict(line=i+1,text=ls[i]) for i in range(138,155)]
 study=[]
 with zipfile.ZipFile(paths['mcts_study.zip']) as z:
  assert z.testzip() is None
  for k in ['mcts_study/v3.0/qft-mcts-statistical.md','mcts_study/v5.0/COMPREHENSIVE_PLOT_DOCUMENTATION.md','mcts_study/v6.0/quantum_mcts_foundation.md','mcts_study/v6.0/quantum_mcts_validation.md']:
   b=z.read(k);ls=b.decode().splitlines();words=['response','fluctuation','susceptibility','correlation','응답','요동','섭동','상관','온도']
   match=[(i+1,v) for i,v in enumerate(ls) if any(w in v.lower() for w in words)]
   study.append(dict(member=k,sha256=hashlib.sha256(b).hexdigest(),lines=len(ls),relevant_line_count=len(match),excerpts=match[:25]))
  members=len([n for n in z.namelist() if not n.endswith('/')])
 dump(S/'ORIGINAL_SOURCE93.json',dict(archives=orig,guide_sha=gh,items=items,guide_feature_contract=spec,study_members=members,study_selected=study,authority='R69 original archive byte identities; guide label and feature span; archived mcts_study, no separate live directory'))
 old90=read(O90/'RESPONSE90.json');assert old90['paths']==2304
 for new,old in zip(a['pulse'],old90['results']):
  assert new['family']==old['family'] and np.max(abs(np.asarray(new['direct'])-old['direct_fd']))<1e-10
 assert len(a['schedule'])==2
 # R92 is an independent population. Retain signed covariance check and original R87 startup correction, no pooling.
 scale=read(P/'SUMMARY92.json');assert scale['status']=='PASS_SCOPED'
 check=dict(status='PASS_SCOPED',source_hashes=inputs,reference_cases=len(q),reference_errors=dict(er),exact_nonzero_cross=str(cross),exact_response=[str(v) for v in resp],exact_hessian=[[str(v) for v in r] for r in H],exact_mixture_contrast=str(sum((a-b)*v for a,b,v in zip(f,c,y))),R92_independent_scope=dict(raw_paths=scale['raw_paths'],time_paths=scale['time_paths'],max_offlag=scale['temporal_max_offlag_innovation'],note='R92 different size/root law; no pooling or physical FDT'),software={'python':platform.python_version(),'numpy':np.__version__},independence='Same agent; pairwise covariance, exact rational two-wave and finite stationary reference; no native subagent')
 dump(S/'INDEPENDENT93.json',check);print('VERIFY',dict(er),'nonzero cross',cross,'mixture contrast',check['exact_mixture_contrast'],flush=True)
if __name__=='__main__':main()
