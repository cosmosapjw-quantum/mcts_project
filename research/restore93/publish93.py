from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore92_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
parent='c51e57021376661a2c8c042058f14778d86727d4'
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED93.md') and r['parent_dag_sha']==sha(O/'DAG.json')
v=read(S/'INDEPENDENT93.json');u=read(S/'RESPONSE93.json');pm=read(S/'PLOT_MANIFEST93.json');src=read(S/'ORIGINAL_SOURCE93.json')
assert v['status']==u['status']=='PASS_SCOPED' and len(pm['figures'])==5
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'G10':'Actual R74 direct versus exact conditional finite source response and contact derivative, 768 declared frozen stationary reversible references, root uncertainty and deviation; no native physical equilibrium FDT.','G11':'Actual R74 time-resolved pulse and finite response kernel estimate, separate R66 independent F/counter ramps; finite/differential contrasts, causal zero and mediated response; direct impulse only wave8, ramps do not identify every column; no aging or physical FDT.','G12':'Joint finite causal score/count/drift identity, direct/reweighted finite contrasts and independent schedule estimates/overlap, explicit frozen equilibrium reference and finite sample discrepancies; not a native equilibrium FDT.'}
acc=list(scopes);partial=[]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS92.json');inv=cr(O/'PLOT_INVENTORY92.csv');assert len(inv)==700 and sha(O/'PLOT_INVENTORY92.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==705;cw(S/'PLOT_INVENTORY93.csv',inv)
c=read(O/'CONVERGENCE92.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==26 and not set(acc)&set(old)
newclaims=[
 dict(id='RESP-KERNEL-01',stage=93,items=['G10','G11','G12'],claim='For finite common-kernel full-support quiescent categorical histories, source-path score S_s and bracket V_s yield first response E[Y S_s] and Hessian E[Y(S_s S_u-delta_su V_s)]. Hit response is causal, with exact contact E[q_s(1-q_s)] and future history-mediated E[q_t S_s]. Measurable prehistory anchors preserve finite odd/even contrast expectations.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','RESPONSE93.json','INDEPENDENT93.json']],limit='Finite specified ideal randomized law, four quiescent draws per wave, source-independent downstream at fixed history; independent exact rational examples, direct pulse only wave8; not physical equilibrium FDT or universal variance reduction.'),
 dict(id='RESP-MIXTURE-02',stage=93,items=['G11','G12'],claim='For equal mixture of two finite common-support prefix path laws, exact RN contrast uses bounded weights w_F=2 sigmoid(log(P_F/P_C)), w_C=2-w_F. Equal per-arm independent samples yield stratified unbiased finite contrast for source-independent observables. Actual R66 comparisons retain overlap and sampling discrepancies.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','RESPONSE93.json','INDEPENDENT93.json']],limit='Standard importance-weight identity applied and verified here, no novelty or independent estimator superiority claim. R66 schedules separate from R74, no reverse dynamics or equilibrium assumption.')]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Physical equilibrium FDT and full direct impulse-column identification open.'))
dump(S/'ITEM_RECONSTRUCTION93.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=29,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=30,new_accepted_subclaims=2,new_corrected_artifacts=3,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION93.json'),scope='Three corrected scoped FDT artifacts; six open science claims unchanged; physical FDT not admitted.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==30
dump(S/'CONVERGENCE93.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS92.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R93_evidence']=scopes.get(k,'');x['R93_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k]),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R93_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS93.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE93.csv',ev)
qm=cr(O/'QUALITY_MATRIX92.csv')
for x in qm:
 k=x['id'];x['R93_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R93_open_subclaims']='';x['R93_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='93';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R93'
cw(S/'QUALITY_MATRIX93.csv',qm)
dump(S/'ITEM_CONVERGENCE93.json',dict(items=[dict(id=x['id'],accepted=x['R93_convergence_subclaims'],scope=x['R93_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R93_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX92.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE92.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION93.json')))
prev.update(figure_pairs=705,distinct_png_sha=705,files_paired=1410,current_pairs=5,prior_pairs=700,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY93.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE93.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS93.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX93.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=30,final_scientific_certified=0)
prev['stages'][S.name]=5;prev['original_output_evidence'].update(current=3,R55_R93_union=121,scope='121 scoped coverage; 29 corrected artifact acceptances,30 scoped claims; six open claims.')
dump(S/'PROGRESS93.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT93.md').write_text('# RESTORE93 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  node['restore93_followup']=dict(evidence='5376 raw paths, 2304 paired pulse and 3072 independent time-dependent schedules, causal score/Hessian, exact binomial and stationary reference checks',accepted_subclaims=[z['id'] for z in newclaims],corrected_artifacts=[k for k in acc if any(x['id']==k and x['owner']==node['id'] for x in rows)],partial=partial,remaining='Physical equilibrium FDT, native bath and full direct impulse-column identification not established.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT93.md,ITEM_RECONSTRUCTION93.json,CONVERGENCE93.json}')
dag.update(revision='RESTORE93_20261128',scope='Convergence: corrected causal source response and FDT validation/summary',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE94-COMPLEX-STATE-POINTER-ENTROPY-OUTPUTS',mainline_priorities=['R94 G13-G17 corrected complex state, coherence, entropy, participation and pointer outputs','Direct original reconstruction; no optimizer drift','Six open science claims, Q02/Q31 and Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX93.csv',convergence_ledger='CONVERGENCE93.json')
rec=dict(status='three_corrected_original_artifacts_two_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=5376,pulse_paths=2304,schedule_paths=3072,figures=5,animations=0,retained_pairs=705,original_evidence_items=3,original_evidence_union_R55_R93=121,original_obligations=121,corrected_report_artifact_accepted=29,new_corrected_artifacts=3,accepted_corrected_subclaims=30,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='92 corrected artifact obligations; six open science claims, Q02/Q31 and Q19/Q20 partial; G13-G17 next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED93.md','RESPONSE93.json','INDEPENDENT93.json','ORIGINAL_SOURCE93.json','CONVERGENCE93.json','ITEM_RECONSTRUCTION93.json','ORIGINAL_OUTPUT_REPORT93.md','ITEM_CONVERGENCE93.json','QUALITY_MATRIX93.csv','PLOT_MANIFEST93.json','PLOT_INVENTORY93.csv','ORIGINAL_OUTPUT_EVIDENCE93.csv','ORIGINAL_OUTPUT_PROGRESS93.csv','PROGRESS93.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore93']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore93']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish93.log','backup93.log','RUN_MANIFEST93.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=u['raw_receipts'];assert len(raw)==5376
manifest=dict(schema='mqft_restore93_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute93.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent independent checks; no physical FDT or full directly tested impulse matrix; prior R87/R92 population distinct.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST93.json',manifest)
dst=P/'research/restore93';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST93.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS93.csv','ORIGINAL_OUTPUT_EVIDENCE93.csv','PLOT_INVENTORY93.csv','PROGRESS93.json','QUALITY_MATRIX93.csv','CONVERGENCE93.json','ITEM_RECONSTRUCTION93.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE93_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE93_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore93','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore93/logs',':(exclude)research/restore93/CORE_POLICY.md',':(exclude)research/restore93/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE93: reconstruct causal response and finite FDT diagnostics')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE93.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE93.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE93_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST93.json'),source_zip_sha256=sha(S/'MQFT_RESTORE93_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE93.git.bundle'),progress_sha256=sha(S/'PROGRESS93.json'),fresh_paths=0,fresh_backups=0,figures=5,cumulative_retained_pairs=705,original_union=121,accepted_corrected_subclaims=30,corrected_report_artifact_accepted=29,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
