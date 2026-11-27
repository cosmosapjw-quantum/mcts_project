"""Original identity and finite convergence acceptance, derived from authenticated data."""
from base61 import *
import csv,zipfile
O=B/'mqft_restore79_20261127';A=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
old=read(O/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==v for k,v in old['archives'].items())
members=[];want=['mcts_study/effective-action-mcts.md','mcts_study/v3.0/qft-mcts-statistical.md']
ids=['L16','Q21','Q32','Q43','G03','G07'];spans=[x for x in old['spans'] if x['id'] in ids];assert len(spans)==6
for x in spans:want.append(x['guide_source_member'] if x['source_status']=='GUIDE_LABEL_ONLY' else x['member'])
for x in old['verified_members']:
 if x['member'] not in want:continue
 with zipfile.ZipFile(paths[x['archive']]) as z:b=z.read(x['member'])
 assert hashlib.sha256(b).hexdigest()==x['sha256'];members.append(x)
 if x['member']=='mcts_study/effective-action-mcts.md':
  s=b.decode();assert 'S_cl[π] = -log N[π]' in s and 'This recovers the UCT selection policy' in s and 'Γ^(1)[π̄]' in s
assert set(want)=={x['member'] for x in members}
dump(S/'ORIGINAL_SOURCE80.json',dict(archives=old['archives'],verified_members=members,spans=spans,original_attachment_receipt=old['original_attachment_receipt'],scope='Original archive and effective-action note, exact relevant source spans inherited from R79 and member bytes reauthenticated. Original visit action replaced only by scoped corrected derivation; original physical quantum loop not certified.'))
r=read(S/'ACTION80.json');a=read(S/'AUDIT80.json');v=read(S/'INDEPENDENT80.json');rng=read(S/'RNG_AUDIT80.json');f=read(S/'FORCED_REPLAY80.json');assert v['status']=='PASS_SCOPED' and len(r['saddles'])==180 and all(x['status']=='OK' for x in r['saddles'])
with np.load(S/'MARKED80.npz') as z:D={k:z[k] for k in z.files}
M=D['meta'];data=[]
for n in r['M']:
 q=[x for x in r['saddles'] if x['M']==n];assert len(q)==30
 data.append(dict(M=n,cases=len(q),max_leading=max(abs(x['leading_rel']) for x in q),max_corrected=max(abs(x['corrected_rel']) for x in q),mean_leading=float(np.mean([abs(x['leading_rel']) for x in q])),mean_corrected=float(np.mean([abs(x['corrected_rel']) for x in q])),improved=sum(abs(x['corrected_rel'])<abs(x['leading_rel']) for x in q),max_scaled_remainder=max(abs(x['corrected_rel'])*n*n for x in q)))
sources=[]
for g,group in enumerate(r['sources']):
 ix=(M[:,0]==1)&(M[:,1]==g);mass=D['unseen_mass'][ix];act=D['action'][ix];z=group['cases'];assert len(mass)==384
 obs=[]
 for name,q in [('regular',[x for x in z if max(abs(y) for y in x['J'])<=4]),('stress',[x for x in z if max(abs(y) for y in x['J'])>4])]:
  obs.append(dict(grid=name,n=len(q),matched_rmse=float(np.sqrt(np.mean([x['matched_delta_W']**2 for x in q]))),novel_rmse=float(np.sqrt(np.mean([x['novel_delta_W']**2 for x in q]))),min_ess=min(x[y]['ess'] for x in q for y in ['train','matched','novel']),matched_pointwise_zero_in_interval=sum(x['matched_bootstrap'][0]<=0<=x['matched_bootstrap'][1] for x in q),novel_pointwise_zero_in_interval=sum(x['novel_bootstrap'][0]<=0<=x['novel_bootstrap'][1] for x in q)))
 sources.append(dict(family=group['family'],paths=int(ix.sum()),unseen_positive_cases=int((mass>0).sum()),mean_unseen_mass=float(mass.mean()),mean_nongreedy=float(D['nongreedy'][ix].mean()),action_range=[float(act.min()),float(act.max())],source_metrics=obs))
q=dict(paths=a['paths'],backups=a['backups'],prior_paths=a['prior_paths'],prior_backups=a['prior_backups'],raw_bytes=a['raw_bytes'],forced_replays=len(f['cases']),forced_backups=f['diagnostic_backups'],rng_streams=rng['streams'],rng_draws=rng['independent_uniforms'],rng_max_error=rng['max_error'],max_probability_error=a['max_probability'],max_score_error=a['max_score'],max_log_error=a['max_log'],lattice_laws=r['lattice_laws'],saddle_convergence=data,sources=sources,max_alias_relative=max(x['alias_bound']/x['wrapped_mass'] for x in r['saddles']),max_torus_cells=max(math.prod(x['shape']) for x in r['saddles']),min_tilt_ess=min(x['tilt_ESS'] for x in r['saddles']),min_source_eig=min(min(x[y]['eig']) for f in r['sources'] for x in f['cases'] for y in ['train','matched','novel']),max_dual_error=max(x[y]['dual_error'] for f in r['sources'] for x in f['cases'] for y in ['train','matched','novel']),independent=v,scope='Exact finite empirical law; population generalization and native complex action separate; alias bound analytic conditional on exact arithmetic, floating reference numerically checked.')
dump(S/'SUMMARY80.json',q)
# Scientific convergence is at explicit subclaim level, not another coverage increase.
claims=[
 dict(id='STAT-ACTION-01',status='ACCEPTED_CORRECTED_SCOPED',items=['L16','Q21','Q43'],claim='For fixed clean deterministic downstream transition maps and ideal conditional categorical root choices, the complete marked history measure is normalized and its action is minus summed log conditional probability. Final visit counts are not this action.',evidence=['THEORY.md','AUDIT80.json','FORCED_REPLAY80.json','RNG_AUDIT80.json','ORIGINAL_SOURCE80.json'],limit='Algorithmic ideal randomized law; 24 forced sequences checked. Not a universal theorem for stochastic evaluator or asynchronous scheduling; no physical quantum action.'),
 dict(id='STAT-1PI-02',status='ACCEPTED_DERIVED_AND_CHECKED',items=['L16','Q21','Q32','G07'],claim='On finite affinely spanning support, exact normalized log-MGF has positive definite tilted covariance; Legendre gradient and Hessian are dual and Gamma equals tilted relative entropy at image.',evidence=['THEORY.md','ACTION80.json','INDEPENDENT80.json'],limit='Exact mathematical identity for fixed measure; 261 finite empirical source cases; true population law not known.'),
 dict(id='STAT-LOOP-03',status='ACCEPTED_DERIVED_AND_CHECKED',items=['L16','Q21','Q43'],claim='For fixed finite full-lattice law with compact tilts, local independent-replica bivariate saddle has determinant prefactor and explicit Wick correction with relative O(M^-2) remainder. All 180 registered finite cases evaluated; M256 maximum corrected relative error 3.07672e-5.',evidence=['THEORY.md','ACTION80.json','INDEPENDENT80.json','SUMMARY80.json'],limit='Replica M counts independent whole histories, not B or physical hbar. Uniform asymptotic constant not explicit; finite numerical comparison uses analytic alias control and separate numerical checks, not rigorous floating interval.'),
 dict(id='STAT-TRANSFER-04',status='OPEN_GENERALIZATION',items=['L16','Q32','G07'],claim='A fixed reference-mixture generating function predicts arbitrary new geometry populations.',evidence=['ACTION80.json','SUMMARY80.json'],limit='Matched geometry predictions much better than novel geometry; Rapfi novel regular-grid RMSE 1.53661; cannot transfer without root-conditioned population model. No equivalence or autonomous closure claim.')]
dump(S/'CONVERGENCE80.json',dict(status='THREE_BOUNDED_CORRECTED_SUBCLAIMS_ACCEPTED_ONE_OPEN',claims=claims,accepted_subclaims=3,final_original_item_certifications=0,prior_corrected_report_artifacts=['L31'],scope='Authored scientific disposition tied to explicit derivations and verification; not independent expert certification; does not retire original quantum objective.'))
print('SUMMARY',q['saddle_convergence'],q['sources'],'accepted',3,flush=True)
