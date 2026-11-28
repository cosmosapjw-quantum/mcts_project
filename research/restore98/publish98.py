from base61 import *
import csv,shutil,subprocess
from PIL import Image
O=B/'mqft_restore97_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';n=98
parent='42ecf3f9b4611e46ce9b099ebedd4590c281966e'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED98.md') and r['parent_dag_sha']==sha(O/'DAG.json') and r['parent']==parent
for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f)
v=read(S/'INDEPENDENT98.json');u=read(S/'RG98.json');q=read(S/'SUMMARY98.json');pm=read(S/'PLOT_MANIFEST98.json');src=read(S/'ORIGINAL_SOURCE98.json')
assert v['status']==u['status']=='PASS_SCOPED' and v['independent_prefix_checks']==u['prefix_rows']==10400 and u['paths']==4320 and len(pm['figures'])==7 and len(pm['animations'])==1
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'Q09':'Exact finite dyadic log-density field and conditional score/Fisher identities on actual legal probability; R77 original fixed-point refutation retained, no continuum beta ODE.',
'Q10':'Partial: actual size, density and score control map with finite KL, Fisher and max probability, prior critical gate retained; no independent physical coupling axes or identified phase boundary.',
'Q11':'Actual native exact probability pushforward across six spatial scales, weighted two-cosine coupling trajectories, nonclosure counterexample and measured TV/coordinate residuals, separate cohorts and budget; no autonomous physical RG.',
'Q12':'Six verified distinct scale frames of actual fixed-prefix L21 dense beta16 distribution, exact and projected lifts; spatial scale is not physical or search time.',
'Q13':'Comprehensive scoped report of exact map and information loss, projected beta field and closure defects, finite fixed-point projector control and prior negative critical/WF evidence; no physical fixed point or universal critical exponents.'}
partial=['Q10'];acc=[k for k in scopes if k not in partial]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS97.json');inv=cr(O/'PLOT_INVENTORY97.csv');assert len(inv)==734 and sha(O/'PLOT_INVENTORY97.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==741;cw(S/'PLOT_INVENTORY98.csv',inv)
ans=pm['animations'];assert [a['frames'] for a in ans]==[6]
for an in ans:
 assert sha(S/an['path'])==an['sha256']
 with Image.open(S/an['path']) as im:
  assert im.n_frames==6;hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert hs==an['frame_rgb_sha256'] and len(set(hs))==6
c=read(O/'CONVERGENCE97.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==48 and not set(scopes)&set(old)
cnt=q['counts']
newclaims=[
 dict(id='RG-PUSH-15',stage=n,items=['Q09','Q11','Q12','Q13'],claim='For specified nested dyadic map and fixed legal reference, exact induced block log-density is conditional logsumexp, KL loss equals conditional KL and frozen score Fisher loss equals conditional variance of retained score means. All 215040 actual recorded selection laws, 10400 prefix rows, six scales and 2400 derivative stencil cases checked.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','RG98.json','RG98.npz','INDEPENDENT98.json']],limit='Fixed recorded histories and finite legal action probability; not actual altered-beta path response, continuum QFT or physical RG fixed point.'),
 dict(id='RG-PROJECTION-16',stage=n,items=['Q09','Q10','Q11','Q13'],claim=f'Exact two-cosine log-coordinate closure fails for explicit checkerboard vs uniform with equal slopes. Actual native finite one-step projected push TV max {cnt["max_push_projection_TV"]:.9g}; rank-two next coupling mismatches {cnt["rank2_projection_mismatch_cases"]}/{cnt["rank2_next_cases"]} and iterated projection TV max {cnt["max_iterated_TV"]:.9g}; beta0 uniform control retained.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','SUMMARY98.json','RG98.npz','INDEPENDENT98.json']],limit='Only designated two-coordinate truncation and registered finite dataset; no impossibility of richer closure or population critical inference.')]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='PARTIAL_CORRECTED_DIAGNOSTIC' if k in partial else 'ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Physical critical boundary, scale-invariant fixed point and richer autonomous closure remain open.'))
next(x for x in items if x['id']=='Q12')['animations']=ans
dump(S/'ITEM_RECONSTRUCTION98.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=52,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=40,new_accepted_subclaims=2,new_corrected_artifacts=4,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION98.json'),scope='Four corrected scoped RG outputs, Q10 partial; no physical criticality or autonomous closure; six open claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==40
dump(S/'CONVERGENCE98.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS97.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R98_evidence']=scopes.get(k,'');x['R98_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k])+(';'+';'.join(a['path'] for a in ans) if k=='Q12' else ''),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R98_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS98.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE98.csv',ev)
qm=cr(O/'QUALITY_MATRIX97.csv')
for x in qm:
 k=x['id'];x['R98_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R98_open_subclaims']='SCALE-CRITICAL-03;MEM-CLOSURE-04' if k in scopes else '';x['R98_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='98';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R98'
cw(S/'QUALITY_MATRIX98.csv',qm)
dump(S/'ITEM_CONVERGENCE98.json',dict(items=[dict(id=x['id'],accepted=x['R98_convergence_subclaims'],scope=x['R98_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R98_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX97.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE97.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION98.json')))
prev.update(figure_pairs=741,distinct_png_sha=741,files_paired=1482,current_pairs=7,prior_pairs=734,current_animations=1,inventory_sha256=sha(S/'PLOT_INVENTORY98.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE98.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS98.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX98.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=40,final_scientific_certified=0,animation_scope='R98 six spatial scale frames, actual fixed prefix; prior animations retained')
prev['stages'][S.name]=7;prev['original_output_evidence'].update(current=5,R55_R98_union=121,scope='121 scoped coverage; 52 corrected artifact acceptances,40 scoped claims; six open claims.')
dump(S/'PROGRESS98.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT98.md').write_text('# RESTORE98 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  ks=[k for k in scopes if any(x['id']==k and x['owner']==node['id'] for x in rows)]
  node['restore98_followup']=dict(evidence='4320 raw receipts, 215040 selections, 10400 prefix rows, 62400 prefix-scale rows, 2400 derivative cases; independent weighted LS and exact dyadic map',accepted_subclaims=[z['id'] for z in newclaims if set(ks)&set(z['items'])],corrected_artifacts=[k for k in acc if k in ks],partial=[k for k in partial if k in ks],remaining='Physical critical point, continuous beta function and richer closure not established.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT98.md,ITEM_RECONSTRUCTION98.json,CONVERGENCE98.json}')
dag.update(revision='RESTORE98_20261128',scope='Convergence: exact finite native spatial pushforward, Fisher contraction and projected closure',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE99-CRITICAL-SCALING-OUTPUTS',mainline_priorities=['R99 Q14-Q18 Q30-Q34 finite scaling, report and animation; check accepted items, preserve Q02/Q31 partial','Direct original reconstruction; no optimizer drift','Six open science claims, Q10 Q36 Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX98.csv',convergence_ledger='CONVERGENCE98.json')
rec=dict(status='four_corrected_artifacts_one_partial_two_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=4320,reused_draws=215040,prefix_rows=10400,prefix_scale_rows=62400,derivative_cases=2400,figures=7,animations=1,retained_pairs=741,original_evidence_items=5,original_evidence_union_R55_R98=121,original_obligations=121,corrected_report_artifact_accepted=52,new_corrected_artifacts=4,accepted_corrected_subclaims=40,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='69 corrected artifact obligations; Q10/Q36 partial; six open science claims; finite critical scaling outputs next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED98.md','RG98.json','RG98.npz','SUMMARY98.json','INDEPENDENT98.json','ORIGINAL_SOURCE98.json','CONVERGENCE98.json','ITEM_RECONSTRUCTION98.json','ORIGINAL_OUTPUT_REPORT98.md','ITEM_CONVERGENCE98.json','QUALITY_MATRIX98.csv','PLOT_MANIFEST98.json','PLOT_INVENTORY98.csv','ORIGINAL_OUTPUT_EVIDENCE98.csv','ORIGINAL_OUTPUT_PROGRESS98.csv','PROGRESS98.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore98']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore98']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish98.log','backup98.log','RUN_MANIFEST98.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=v['raw_receipts'];assert len(raw)==4320
manifest=dict(schema='mqft_restore98_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute98.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent direct bincount/weighted LS, frozen score derivatives and exact chain; no physical fixed point, independent physical coupling axes or autonomous RG closure.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST98.json',manifest)
dst=P/'research/restore98';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST98.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS98.csv','ORIGINAL_OUTPUT_EVIDENCE98.csv','PLOT_INVENTORY98.csv','PROGRESS98.json','QUALITY_MATRIX98.csv','CONVERGENCE98.json','ITEM_RECONSTRUCTION98.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE98_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE98_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore98','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore98/logs',':(exclude)research/restore98/CORE_POLICY.md',':(exclude)research/restore98/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE98: exact dyadic pushforward and projected RG closure defects')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE98.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE98.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE98_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST98.json'),source_zip_sha256=sha(S/'MQFT_RESTORE98_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE98.git.bundle'),progress_sha256=sha(S/'PROGRESS98.json'),fresh_paths=0,fresh_backups=0,figures=7,animations=1,cumulative_retained_pairs=741,original_union=121,accepted_corrected_subclaims=40,corrected_report_artifact_accepted=52,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
