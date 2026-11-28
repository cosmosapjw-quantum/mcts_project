from base61 import *
import csv,shutil,subprocess
from PIL import Image
O=B/'mqft_restore102_20261129';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';n=103
parent='d616961b37a8fee5751ab90e33231dc619e596df'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED103.md') and r['parent_dag_sha']==sha(O/'DAG.json') and r['parent']==parent
for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f)
v=read(S/'INDEPENDENT103.json');u=read(S/'MEASURE103.json');q=read(S/'SUMMARY103.json');pm=read(S/'PLOT_MANIFEST103.json');src=read(S/'ORIGINAL_SOURCE103.json')
assert v['status']==u['status']=='PASS_SCOPED' and v['counts']['projection_cases']==62400 and v['counts']['prior_projection_cases']==5760 and v['counts']['conditional_heldout']==50 and u['paths']==4320 and v['cross_history_root_strata']==3200 and len(pm['figures'])==8 and len(pm['animations'])==0
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'L03': 'Authenticated per-selection log-probability slope versus actual recorded score, affine scale invariance and flat startup masks, independent R79 and controlled R87 cohorts; no physical T.', 'L04': 'Four explicit signed bounded occupancy-slope scaling candidates, development leave-one-size-out selection and existing independent L21 test; no universal scaling, retain negative comparisons.', 'L05': 'Typed actual selection/wave/trajectory/prefix/projection counts and quality, explicit invalid flat and constrained/face status, independent numerical validation and cross-history controls; no physical T count.', 'L25': 'Full legal support A/X projection onto endpoint Q, mean and last preselection score, exact moment and KL identities, explicit negative and boundary cases, heldout conditional score/Q grid; no physical T.', 'L26': 'Separate actual independent B16/B64 and correlated B64 prefix diagnostics across five sizes, four fixed scaling candidates, separate log-size control; no physical power law.', 'L34': 'Typed measurement quality summary, flat score and constrained projection statuses, actual cross-history endogeneity comparison and exact counterexample, heldout error and independent checks; no artificial physical observations.'}
partial=[];acc=[k for k in scopes if k not in partial]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS102.json');inv=cr(O/'PLOT_INVENTORY102.csv');assert len(inv)==780 and sha(O/'PLOT_INVENTORY102.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==788;cw(S/'PLOT_INVENTORY103.csv',inv)
ans=pm['animations'];assert sorted(a['frames'] for a in ans)==[]
for an in ans:
 assert sha(S/an['path'])==an['sha256']
 with Image.open(S/an['path']) as im:
  assert im.n_frames==an['frames'];hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert hs==an['frame_rgb_sha256'] and len(set(hs))==im.n_frames
c=read(O/'CONVERGENCE102.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==78 and not set(scopes)&set(old)
newclaims=[
 dict(id='MEASURE-LAW-29',stage=n,items=['L03','L05','L25','L34'],claim='Exact centered score/log-probability slope on nonflat legal support; affine score gauge preserves conditional probabilities, so independent physical energy calibration is required. Actual R87 215040 selections,210720 identifiable and4320 flat; R79 30720 waves,29760 identifiable and960 flat; controlled B16 training selects the programmed slope for all8 conditions, Q-only NLL worse in all36 nonzero-control L21 pointwise comparisons.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','MEASURE103.json','PRIOR103.json','FIT103.json','INDEPENDENT103.json']],limit='Algorithmic conditional score law, not physical temperature or hypothetical policy trajectories; existing independent cohorts.'),
 dict(id='MEASURE-PROJECTION-30',stage=n,items=['L03','L04','L05','L25','L26','L34'],claim='For fixed legal feature f, KL projection onto softmax(b f) obeys moment matching and constrained Pythagorean correction; temporal mixture gap separates changing laws from static projection residual. 62400 projections, all independently bisected; 62078 interior,110 finite-bound,212 support-face; 64 scaling fits, development-selected fresh L21 B64 wins/ties6/16 including two trivial uniform A controls (4/14 nontrivial).',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','MEASURE103.json','MEASURE103.npz','FIT103.json','SUMMARY103.json','INDEPENDENT103.json']],limit='Fixed feature and finite signed bounded slope, no population or physical temperature scaling.'),
 dict(id='MEASURE-ENDOGENEITY-31',stage=n,items=['L05','L25','L34'],claim='Predictable visit residual E(X-A)=0; across independent repeats cross feature moment has expectation zero, self-feature moment need not. Exact self-minus-cross empirical covariance identity tested across3200 root/stratum blocks and26400 ordered pairs. Fresh dense beta0 endpoint-Q mean self -.284899 vs cross -.010665; constructed uniform two-action example attains infinite signed projection despite beta0.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','CROSS_HISTORY_EXTENSION103.md','CROSS103.json','SUMMARY103.json','INDEPENDENT103.json']],limit='Post-main-run diagnostic, ordered pairs not independent, no claim of zero bias from finite samples; not negative physical temperature.')
 ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='PARTIAL_CORRECTED_DIAGNOSTIC' if k in partial else 'ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Independent physical energy and bath temperature calibration remain open.'))
for an in ans:next(x for x in items if x['id']==an['id'])['animations']=[an]
dump(S/'ITEM_RECONSTRUCTION103.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=84,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=55,new_accepted_subclaims=3,new_corrected_artifacts=6,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION103.json'),scope='Six corrected scoped measurement and scaling outputs; physical temperature unresolved; six open claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==55
dump(S/'CONVERGENCE103.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS102.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R103_evidence']=scopes.get(k,'');x['R103_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k])+(';'+';'.join(a['path'] for a in ans if a['id']==k) if False else ''),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R103_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS103.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE103.csv',ev)
qm=cr(O/'QUALITY_MATRIX102.csv')
for x in qm:
 k=x['id'];x['R103_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R103_open_subclaims']='THERMO-PHYSICAL-04;STAT-TRANSFER-04' if k in scopes else '';x['R103_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='103';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R103'
cw(S/'QUALITY_MATRIX103.csv',qm)
dump(S/'ITEM_CONVERGENCE103.json',dict(items=[dict(id=x['id'],accepted=x['R103_convergence_subclaims'],scope=x['R103_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R103_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX102.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE102.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION103.json')))
prev.update(figure_pairs=788,distinct_png_sha=788,files_paired=1576,current_pairs=8,prior_pairs=780,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY103.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE103.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS103.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX103.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=55,final_scientific_certified=0,animation_scope='R103 no new animation; original measurement reports static; prior animations retained')
prev['stages'][S.name]=8;prev['original_output_evidence'].update(current=6,R55_R103_union=121,scope='121 scoped coverage; 84 corrected artifact acceptances,55 scoped claims; six open claims.')
dump(S/'PROGRESS103.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT103.md').write_text('# RESTORE103 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  ks=[k for k in scopes if any(x['id']==k and x['owner']==node['id'] for x in rows)]
  node['restore103_followup']=dict(evidence='5280 raw receipts, 215040 selections and 30720 prior waves; 62400 projections and 5760 prior cuts; 50 heldout conditions and 3200 cross-history blocks',accepted_subclaims=[z['id'] for z in newclaims if set(ks)&set(z['items'])],corrected_artifacts=[k for k in acc if k in ks],partial=[k for k in partial if k in ks],remaining='Independent physical energy and bath temperature calibration not identified.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT103.md,ITEM_RECONSTRUCTION103.json,CONVERGENCE103.json}')
dag.update(revision='RESTORE103_20261129',scope='Convergence: conditional score identification, full-support projections and endogeneity',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE104-LEGACY-THERMAL-FEEDBACK',mainline_priorities=['R104 L06/L08/L18/L32/L33 thermal and feedback legacy; inspect acceptance list','Direct original reconstruction; no optimizer drift','Six open science claims, Q10 Q36 Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX103.csv',convergence_ledger='CONVERGENCE103.json')
rec=dict(status='six_corrected_artifacts_three_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_R79_paths=960,reused_size_paths=4320,raw_receipts=5280,projection_cases=62400,prior_projection_cases=5760,heldout_conditions=50,selection_events=215040,cross_history_blocks=3200,figures=8,animations=0,retained_pairs=788,original_evidence_items=6,original_evidence_union_R55_R103=121,original_obligations=121,corrected_report_artifact_accepted=84,new_corrected_artifacts=6,accepted_corrected_subclaims=55,new_accepted_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='37 corrected artifact obligations; six open science claims; thermal/feedback legacy next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED103.md','MEASURE103.json','MEASURE103.npz','PRIOR103.json','FIT103.json','CROSS103.json','CROSS_HISTORY_EXTENSION103.md','SUMMARY103.json','INDEPENDENT103.json','ORIGINAL_SOURCE103.json','CONVERGENCE103.json','ITEM_RECONSTRUCTION103.json','ORIGINAL_OUTPUT_REPORT103.md','ITEM_CONVERGENCE103.json','QUALITY_MATRIX103.csv','PLOT_MANIFEST103.json','PLOT_INVENTORY103.csv','ORIGINAL_OUTPUT_EVIDENCE103.csv','ORIGINAL_OUTPUT_PROGRESS103.csv','PROGRESS103.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore103']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore103']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish103.log','backup103.log','RUN_MANIFEST103.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=v['raw_receipts'];assert len(raw)==5280
manifest=dict(schema='mqft_restore103_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute103.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent independent bisection, score law, grid and scaling predictions, cross-history identity; no physical T.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST103.json',manifest)
dst=P/'research/restore103';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST103.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS103.csv','ORIGINAL_OUTPUT_EVIDENCE103.csv','PLOT_INVENTORY103.csv','PROGRESS103.json','QUALITY_MATRIX103.csv','CONVERGENCE103.json','ITEM_RECONSTRUCTION103.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE103_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE103_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore103','docs')
# Verify byte-locked preregistration.
assert sha(S/'THEORY_REGISTERED103.md')==r['contract_sha']==sha(dst/'THEORY_REGISTERED103.md')
git('diff','--cached','--check','--','.',':(exclude)research/restore103/logs',':(exclude)research/restore103/CORE_POLICY.md',':(exclude)research/restore103/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE103: conditional score law and endogenous occupancy projection')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE103.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE103.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE103_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST103.json'),source_zip_sha256=sha(S/'MQFT_RESTORE103_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE103.git.bundle'),progress_sha256=sha(S/'PROGRESS103.json'),fresh_paths=0,fresh_backups=0,figures=8,animations=0,cumulative_retained_pairs=788,original_union=121,accepted_corrected_subclaims=55,corrected_report_artifact_accepted=84,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
