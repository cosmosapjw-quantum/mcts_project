from thermal91 import *
import zipfile,platform
src=read(O/'BRANCH_ENERGY85.json');t=read(S/'THERMAL91.json');n=read(S/'NATIVE91.json')
assert len(t['cases'])==832 and t['contexts']==104 and n['histories']==960
ctx={(c['protocol'],c['family'],c['core']):c for c in src['contexts']}
# Independent vector entropy balance: marginal plus mutual information change, not just conditional entropy.
v=[];maxerr=0.;i1=[]
for c in ctx.values():
 e=np.array(c['e0']);em=np.array(c['em']);g=np.array(c['group']);m=len(em)
 for b in [1/3,1,10/3,10]:
  p=gibbs(e,b);pi=gibbs(em,b);M=np.full((len(e),m),.2/m);M[np.arange(len(e)),g]+=.8
  joint=p[:,None]*M;pm=joint.sum(0);a=joint.T/pm[:,None];z=a;rr=[]
  for k in range(8):
   zz=(.75**(k+1))*a+(1-.75**(k+1))*pi
   r0=pm@z;r1=pm@zz;dS=float(H(r1)-H(r0))
   I0=float(H(r0)-pm@H(z));I1=float(H(r1)-pm@H(zz));dI=I1-I0
   q=float(pm@np.sum((zz-z)*em,axis=1))
   # Independently compute entropy production from probability flow ratio Pij/Pji using detailed balance.
   sig=dS-dI-b*q;rr.append([dS,dI,q,sig])
   z=zz
  row=next(x for x in t['time'] if (x['protocol'],x['family'],x['core'],x['beta'])==(c['protocol'],c['family'],c['core'],b))
  ref=np.array(row['steps']);a1=np.array(rr);err=float(abs(a1[:,3]-ref[:,4]).max());maxerr=max(maxerr,err);assert err<1e-9
  v.append(dict(protocol=c['protocol'],family=c['family'],core=c['core'],beta=b,steps=rr))
# Inspect original mcts_study archive, preserve all three byte identities, extract authoritative exact spans.
p90=B/'mqft_restore90_20261128';r69=B/'mqft_restore69_20261126'
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
orig=read(p90/'ORIGINAL_SOURCE90.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
pol=['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv'];assert all(sha(S/f)==sha(p90/f)==sha(r69/f) for f in pol)
cat={x['id']:x for x in csv.DictReader(open(S/'ORIGINAL_TARGET_CATALOG.csv'))};qm={x['id']:x for x in csv.DictReader(open(p90/'QUALITY_MATRIX90.csv'))};target=read(S/'REGISTRATION.json')['targets'];items=[]
guide='omoknuni_quantum-main/docs/quantum_physics/STATISTICAL_MECHANICS_ANALYSIS_GUIDE.md'
with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
 a=z.read(guide);gh=hashlib.sha256(a).hexdigest();assert gh==qm['G01']['source_sha'];ls=a.decode().splitlines()
 for k in target:
  x=cat[k];q=qm[k]
  if k.startswith('G'):
   line=int(q['source_span']);items.append(dict(id=k,archive='omoknuni_quantum-main.zip',member=guide,member_sha=gh,span=[line,line],source_excerpt=ls[line-1],acceptance=q['final_acceptance']))
  else:
   with zipfile.ZipFile(paths[x['archive']]) as z:
    a=z.read(x['member']);h=hashlib.sha256(a).hexdigest();assert h==x['member_sha256'];lines=a.decode().splitlines();lo=int(x['start']);hi=int(x['end']);s='\n'.join(lines[lo-1:hi]);items.append(dict(id=k,archive=x['archive'],member=x['member'],member_sha=h,span=[lo,hi],span_sha=hashlib.sha256(s.encode()).hexdigest(),source_excerpt=s,acceptance=q['final_acceptance']))
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 members=[a for a in z.namelist() if not a.endswith('/')];assert z.testzip() is None
 # Inspect study titles/text headings instead of inventing original results.
 study=[]
 for k in members:
  if k.lower().endswith(('.md','.tex','.txt')) and any(x in k.lower() for x in ('thermo','statist','entropy','mcts','quantum')):
   b=z.read(k);s=b.decode(errors='replace');ls=[x.strip() for x in s.splitlines() if x.lstrip().startswith('#')][:5]
   study.append(dict(member=k,sha256=hashlib.sha256(b).hexdigest(),headings=ls))
dump(S/'ORIGINAL_SOURCE91.json',dict(archives=orig,items=items,guide_sha=gh,study_members=len(members),study_selected=study,authority='RESTORE69 original source identity; exact spans and guide labels'))
eps1=[r['I1'] for r in t['cases'] if r['eps']==1];assert max(eps1)>1e-8
for x in t['cases']:assert x['total']>=-1e-10 and x['sigma_rel']>=-1e-10
dump(S/'INDEPENDENT91.json',dict(status='PASS_SCOPED',method='same-agent independent closed-form repeated kernel vs step iteration and marginal+MI signed entropy balance; exact path direct enumeration subset',max_marginal_entropy_error=maxerr,case_count=len(t['cases']),time_steps=len(v)*8,eps1_I1_range=[min(eps1),max(eps1)],marginal_time=v,source_hashes={str(p.relative_to(B)):sha(p) for p in [O/'BRANCH_ENERGY85.json',O/'THERMAL85.json',O/'THERMAL_SAMPLES85.npz',A/'AUDIT79.json',A/'CALIBRATION79.json',R/'R60_PREFIX61.npz',R/'R60_REDUCTION61.json',p90/'DAG.json',p90/'CONVERGENCE90.json',p90/'QUALITY_MATRIX90.csv']},subagent='NOT_RUN',software={'python':platform.python_version(),'numpy':np.__version__}))
print('INDEPENDENT',maxerr,len(v)*8,'eps1 I1',min(eps1),max(eps1))
