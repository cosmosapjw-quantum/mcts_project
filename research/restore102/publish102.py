from base61 import *
import csv,shutil,subprocess
from PIL import Image
O=B/'mqft_restore101_20261129';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';n=102
parent='ca9453563c6b320627eedb201094a57fb86583cc'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED102.md') and r['parent_dag_sha']==sha(O/'DAG.json') and r['parent']==parent
for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f)
v=read(S/'INDEPENDENT102.json');u=read(S/'MARKOV102.json');q=read(S/'SUMMARY102.json');pm=read(S/'PLOT_MANIFEST102.json');src=read(S/'ORIGINAL_SOURCE102.json')
assert v['status']==u['status']=='PASS_SCOPED' and v['counts']['triples']==306 and v['counts']['root_triples']==17952 and v['counts']['quiescent_waves']==49440 and u['paths']==7488 and len(pm['figures'])==9 and len(pm['animations'])==0
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'L39': 'Actual within-root/clock lag covariance, valid normalized covariance and predictable innovation moments, all five sizes and distinct budget/cohort strata; no stationary exponential or physical tau.', 'L40': 'Exact finite triple Markovization, CMI and CK weighted TV bound, XOR and nonstationary controls, root-conditioned finite-sample permutation expectation and independent-root predictive tests; prior strong nonclosure retained, no blanket Markov verdict.', 'L41': 'Actual quiescent width4 conditional binomial count and factorial pair moments, observed conditional calibration, predictive variance/NLL/Brier, independent polynomial check, corrected startup1+1+2.', 'L42': 'Actual root-level distributions of lag covariance and innovation, valid normalization and clock mixture decomposition; no fabricated game durations or relaxation times.', 'L43': 'Explicit 9-state time-dependent stochastic kernels, positive hierarchical backoff and independent root one/multistep validation; direct vs composed and homogeneous controls, no autonomous exact closure.', 'L44': 'Partial: actual finite lag and backup clocks, clock-mixture decomposition, no separately identified environment/system relaxation time or physical tau.', 'L48': 'Comprehensive scoped Markovian report: exact triple CMI/CK, finite conditional null, actual lag and analytic count law, heldout root predictions and unresolved timescale, retained R86 strong counterexamples.', 'L49': 'Scoped summary figure and report of actual finite lag/CK, conditional law, heldout prediction and finite-sample CMI baseline; explicit unresolved tau, no Markov YES/NO or physical correlation time.'}
partial=['L44'];acc=[k for k in scopes if k not in partial]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS101.json');inv=cr(O/'PLOT_INVENTORY101.csv');assert len(inv)==771 and sha(O/'PLOT_INVENTORY101.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==780;cw(S/'PLOT_INVENTORY102.csv',inv)
ans=pm['animations'];assert sorted(a['frames'] for a in ans)==[]
for an in ans:
 assert sha(S/an['path'])==an['sha256']
 with Image.open(S/an['path']) as im:
  assert im.n_frames==an['frames'];hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert hs==an['frame_rgb_sha256'] and len(set(hs))==im.n_frames
c=read(O/'CONVERGENCE101.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==71 and not set(scopes)&set(old)
newclaims=[
 dict(id='MEM-TRIPLE-26',stage=n,items=['L40','L48','L49'],claim='Exact three-time Markovization Q=P_ab P_c|b preserves adjacent marginals, KL(P||Q)=CMI, weighted CK TV<=joint TV<=sqrt(CMI/2). XOR CK0 with CMI ln2, and exact time-dependent Markov control. 306 root-balanced triples and17952 root-specific triples checked; exact hypergeometric finite-sample conditional-permutation expectation retained.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','MARKOV102.json','CMI_NULL102.json','INDEPENDENT102.json']],limit='Finite empirical distributions, root mixture and plug-in bias; no population Markov order or universal exact closure.'),
 dict(id='MEM-PREDICT-27',stage=n,items=['L40','L43','L48','L49'],claim='Fixed 9-state development-only hierarchical clock-conditioned lag2 model improves clock1 on both previously independent root cohorts for all three evaluators, all6 pointwise paired intervals for log loss and Brier positive. Direct vs clock-composed multistep positive point gains18/18, intervals17/18 each. Independent verification165888 path/start/horizon cases; homogeneous model better than lag2 for g3 NLL.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','MARKOV102.json','STATE102.npz','SUMMARY102.json','INDEPENDENT102.json']],limit='Previously acquired heldout roots, fixed smoothing and state bins, pointwise unadjusted intervals; no new campaign, universal order or closure.'),
 dict(id='MEM-LAG-28',stage=n,items=['L39','L41','L42','L44','L48','L49'],claim='For predictable inner-sector p, D=X-p is a martingale difference; genuine width4 frozen-law count has exact binomial moments. Actual 4320 paths/215040 selections,49440 waves,160 strata,6080 root/lag conditions,755 undefined normalized covariances. Clock covariance split and startup1+1+2 defect verified;16/160 pointwise pair residual intervals exclude0 retained.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','LAG102.json','SELECTION102.npz','SUMMARY102.json','INDEPENDENT102.json']],limit='Finite algorithmic selection clock, not stationary exponential decay or physical environment/system relaxation.')
 ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='PARTIAL_CORRECTED_DIAGNOSTIC' if k in partial else 'ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Exact autonomous closure, population Markov order and physical timescale separation remain open.'))
for an in ans:next(x for x in items if x['id']==an['id'])['animations']=[an]
dump(S/'ITEM_RECONSTRUCTION102.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=78,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=52,new_accepted_subclaims=3,new_corrected_artifacts=7,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION102.json'),scope='Seven corrected scoped legacy Markov outputs, L44 partial; six open claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==52
dump(S/'CONVERGENCE102.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS101.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R102_evidence']=scopes.get(k,'');x['R102_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k])+(';'+';'.join(a['path'] for a in ans if a['id']==k) if False else ''),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R102_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS102.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE102.csv',ev)
qm=cr(O/'QUALITY_MATRIX101.csv')
for x in qm:
 k=x['id'];x['R102_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R102_open_subclaims']='MEM-CLOSURE-04' if k in scopes else '';x['R102_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='102';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R102'
cw(S/'QUALITY_MATRIX102.csv',qm)
dump(S/'ITEM_CONVERGENCE102.json',dict(items=[dict(id=x['id'],accepted=x['R102_convergence_subclaims'],scope=x['R102_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R102_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX101.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE101.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION102.json')))
prev.update(figure_pairs=780,distinct_png_sha=780,files_paired=1560,current_pairs=9,prior_pairs=771,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY102.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE102.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS102.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX102.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=52,final_scientific_certified=0,animation_scope='R102 no new animation; original Markov reports static; prior animations retained')
prev['stages'][S.name]=9;prev['original_output_evidence'].update(current=8,R55_R102_union=121,scope='121 scoped coverage; 78 corrected artifact acceptances,52 scoped claims; six open claims.')
dump(S/'PROGRESS102.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT102.md').write_text('# RESTORE102 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  ks=[k for k in scopes if any(x['id']==k and x['owner']==node['id'] for x in rows)]
  node['restore102_followup']=dict(evidence='11808 raw receipts, 306 empirical triples and 17952 root triples; 165888 heldout cases; 215040 selections,49440 quiescent waves',accepted_subclaims=[z['id'] for z in newclaims if set(ks)&set(z['items'])],corrected_artifacts=[k for k in acc if k in ks],partial=[k for k in partial if k in ks],remaining='Exact autonomous closure, population Markov order and separate physical relaxation clocks not identified.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT102.md,ITEM_RECONSTRUCTION102.json,CONVERGENCE102.json}')
dag.update(revision='RESTORE102_20261129',scope='Convergence: nonstationary triple Markovization, heldout lag prediction and actual quiescent count law',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE103-LEGACY-MEASUREMENT-QUALITY',mainline_priorities=['R103 L03/L04/L05/L25/L26/L34 authentic temperature and typed measurement quality; inspect acceptance list','Direct original reconstruction; no optimizer drift','Six open science claims, Q10 Q36 Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX102.csv',convergence_ledger='CONVERGENCE102.json')
rec=dict(status='seven_corrected_artifacts_one_partial_three_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_B128_paths=7488,reused_size_paths=4320,raw_receipts=11808,triple_cases=306,root_triples=17952,heldout_cases=165888,selection_events=215040,quiescent_waves=49440,figures=9,animations=0,retained_pairs=780,original_evidence_items=8,original_evidence_union_R55_R102=121,original_obligations=121,corrected_report_artifact_accepted=78,new_corrected_artifacts=7,accepted_corrected_subclaims=52,new_accepted_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='43 corrected artifact obligations; L44 partial; six open science claims; measurement quality next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED102.md','MARKOV102.json','STATE102.npz','LAG102.json','SELECTION102.npz','CMI_NULL102.json','SUMMARY102.json','INDEPENDENT102.json','ORIGINAL_SOURCE102.json','CONVERGENCE102.json','ITEM_RECONSTRUCTION102.json','ORIGINAL_OUTPUT_REPORT102.md','ITEM_CONVERGENCE102.json','QUALITY_MATRIX102.csv','PLOT_MANIFEST102.json','PLOT_INVENTORY102.csv','ORIGINAL_OUTPUT_EVIDENCE102.csv','ORIGINAL_OUTPUT_PROGRESS102.csv','PROGRESS102.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore102']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore102']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish102.log','backup102.log','RUN_MANIFEST102.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=v['raw_receipts'];assert len(raw)==11808
manifest=dict(schema='mqft_restore102_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute102.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent independent triple, exact hypergeometric baseline, heldout scores, polynomial count and root covariance; no population order or physical timescale.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST102.json',manifest)
dst=P/'research/restore102';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST102.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS102.csv','ORIGINAL_OUTPUT_EVIDENCE102.csv','PLOT_INVENTORY102.csv','PROGRESS102.json','QUALITY_MATRIX102.csv','CONVERGENCE102.json','ITEM_RECONSTRUCTION102.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE102_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE102_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore102','docs')
# Verify byte-locked preregistration.
assert sha(S/'THEORY_REGISTERED102.md')==r['contract_sha']==sha(dst/'THEORY_REGISTERED102.md')
git('diff','--cached','--check','--','.',':(exclude)research/restore102/logs',':(exclude)research/restore102/CORE_POLICY.md',':(exclude)research/restore102/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE102: triple Markovization and independent root memory prediction')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE102.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE102.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE102_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST102.json'),source_zip_sha256=sha(S/'MQFT_RESTORE102_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE102.git.bundle'),progress_sha256=sha(S/'PROGRESS102.json'),fresh_paths=0,fresh_backups=0,figures=9,animations=0,cumulative_retained_pairs=780,original_union=121,accepted_corrected_subclaims=52,corrected_report_artifact_accepted=78,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
