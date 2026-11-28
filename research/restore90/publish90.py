from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore89_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';parent='d9094e195944edfe78de23c98b798777590f5568'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
q=read(S/'SUMMARY90.json');v=read(S/'INDEPENDENT90.json');c=read(S/'CONVERGENCE90.json');pm=read(S/'PLOT_MANIFEST90.json');r=read(S/'ITEM_RECONSTRUCTION90.json');reg=read(S/'REGISTRATION.json')
assert q['status']==v['status']=='PASS_SCOPED' and q['clean_paths']==7488 and q['unique_raw_receipts']==15648 and q['new_fixed_roots']==0
assert c['accepted_subclaims']==25 and c['new_accepted_subclaims']==2 and c['final_original_item_certifications']==0 and c['artifact_reconstruction_sha']==sha(S/'ITEM_RECONSTRUCTION90.json')
assert sha(S/'THEORY_REGISTERED90.md')==reg['contract_sha'] and r['corrected_artifacts_total']==16 and sum(x['decision'].startswith('ACCEPTED') for x in r['items'])==5
assert all(sha(S/k)==x for k,x in pm['sources'].items()) and all(sha(B/k)==x for k,x in pm['prior_sources'].items()) and len(pm['figures'])==7
src=read(S/'ORIGINAL_SOURCE90.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE89.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
m80=read(B/'mqft_restore80_20261127/RUN_MANIFEST80.json')
prior=v['raw_receipts'];assert len(prior)==15648 and len({x['path'] for x in prior})==15648
scopes={x['id']:x['method']+' '+x['remaining_limit'] for x in r['items']};artifact={x['id']:x for x in r['items']}
assert set(scopes)=={k for z in pm['figures'] for k in z['ids']}=={'L16','G03','G07','Q21','Q32','Q43'}
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS89.json');inv=cr(O/'PLOT_INVENTORY89.csv');assert len(inv)==678 and sha(O/'PLOT_INVENTORY89.csv')==prev['inventory_sha256']
plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==685;cw(S/'PLOT_INVENTORY90.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS89.csv');ev=[]
for x in rows:
 k=x['id'];x['R90_evidence']=scopes.get(k,'');x['R90_scope_status']=artifact[k]['decision'] if k in scopes else ''
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(artifact[k]['figures']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R90_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE90.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS90.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{k}_evidence') for k in range(55,91))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX89.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==90 and k in z['items']]
 x['R90_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R90_open_subclaims']=';'.join(z['id'] for z in linked if not z['status'].startswith('ACCEPTED'));x['R90_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='90';x['latest_scope']=scopes[k];x['corrected_output_state']=artifact[k]['decision']+'_R90'
cw(S/'QUALITY_MATRIX90.csv',qm)
dump(S/'ITEM_CONVERGENCE90.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX89.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE89.json'),items=[dict(id=x['id'],accepted=x['R90_convergence_subclaims'],scope=x['R90_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R90_scope']],claims=[z for z in c['claims'] if z['stage']==90],artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION90.json'),scope='Five corrected artifacts, Q43 partial; original science not certified.'))
accepted=c['corrected_report_artifacts_accepted'];assert len(accepted)==len(set(accepted))==16
prev.update(figure_pairs=685,distinct_png_sha=685,files_paired=1370,current_pairs=7,prior_pairs=678,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY90.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE90.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS90.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX90.csv'),corrected_report_artifact_accepted=accepted,accepted_corrected_subclaims=25,final_scientific_certified=0)
prev['stages'][S.name]=7;prev['original_output_evidence']=dict(current=len(scopes),R55_R90_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='121 scoped coverage; sixteen corrected artifact acceptances,25 corrected subclaims; Q43 partial, science separate.')
dump(S/'PROGRESS90.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('REPLAN-03','REPLAN-04.C','REPLAN-04.R','REPLAN-06','X-REP'):node['restore90_followup']=dict(evidence='7488 clean paths,2304 pulse paths,3456 phase paths,15648 unique historical receipts; 124 edges/vertices,832 work cases; original scoped reconstruction.',accepted_subclaims=['STAT-FACE-07','THERMO-WORK-CGF-05'],corrected_artifacts=[x['id'] for x in r['items'] if x['decision'].startswith('ACCEPTED')],partial=['Q43'],remaining='Physical hbar, thermometry, native coherence and broad transfer not admitted.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT90.md,ITEM_RECONSTRUCTION90.json,CONVERGENCE90.json}')
dag.update(revision='RESTORE90_20261128',scope='Convergence: statistical action, finite support and declared work CGF original reconstruction',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE91-STATISTICAL-MECHANICS-THERMAL-OUTPUTS',mainline_priorities=['R91 reconstruct remaining statistical mechanics/thermal outputs using actual evidence and exact contracts','Direct original artifact convergence; no repetitive diagnostic grids','Six open scientific claims and Q43 missing calibration remain distinct from corrected artifacts'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX90.csv',convergence_ledger='CONVERGENCE90.json',backup_policy='DROPBOX_ONLY_VERIFIED_SNAPSHOT_NO_REDUNDANT_LOCAL_BACKUP')
files=['REGISTRATION.json','THEORY_REGISTERED90.md','PATH90.json','DUAL90.json','RESPONSE90.json','WORK90.json','PHASE90.json','INDEPENDENT90.json','SUMMARY90.json','ORIGINAL_SOURCE90.json','CONVERGENCE90.json','ITEM_RECONSTRUCTION90.json','ORIGINAL_OUTPUT_REPORT90.md','ITEM_CONVERGENCE90.json','QUALITY_MATRIX90.csv','PLOT_MANIFEST90.json','PLOT_INVENTORY90.csv','ORIGINAL_OUTPUT_EVIDENCE90.csv','ORIGINAL_OUTPUT_PROGRESS90.csv','PROGRESS90.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='five_corrected_original_artifacts_two_new_scoped_claims_one_partial',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=7488,source_path_receipts=15648,figures=7,animations=0,retained_pairs=685,original_evidence_items=6,original_evidence_union_R55_R90=121,original_obligations=121,corrected_report_artifact_accepted=16,new_corrected_artifacts=5,accepted_corrected_subclaims=25,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='105 original corrected artifact obligations; six open science claims. Q43 partial. Dropbox only.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore90']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore90']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=m80['external'],weights=m80['weights']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','path90.py','dual90.py','response90.py','work90.py','phase90.py','verify90.py','plot90.py','report90.py','publish90.py','backup90.py','execute90.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=[str(x.relative_to(S)) for x in sorted((S/'logs').glob('*.log')) if x.name not in ('publish90.log','backup90.log')]
inputs={x['path']:x['sha256'] for x in v['prior_file_checks']}
for z in ['ITEM_RECONSTRUCTION89.json','CONVERGENCE89.json','QUALITY_MATRIX89.csv','DAG.json','ORIGINAL_SOURCE89.json']:inputs[str((O/z).relative_to(B))]=sha(O/z)
for z in ['ITEM_ACCEPTANCE79.json','ORIGINAL_SOURCE79.json']:
 f=B/'mqft_restore79_20261127'/z;inputs[str(f.relative_to(B))]=sha(f)
inputs.update(pm['prior_sources'])
manifest=dict(schema='mqft_restore90_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute90.sh on authenticated reused raw histories',files={z:sha(S/z) for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','PATH90.npz']},raw_receipts=[],prior_raw_receipts=prior,diagnostic_receipts=[],weights=m80['weights'],external=m80['external'],plan_hashes={},inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent independent algorithms, six new FFT reruns, inherited other native replay and 174 masses; no physical hbar, native thermal calibration or broad transfer.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST90.json',manifest)
dst=P/'research/restore90';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST90.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS90.csv','ORIGINAL_OUTPUT_EVIDENCE90.csv','PLOT_INVENTORY90.csv','PROGRESS90.json','QUALITY_MATRIX90.csv','CONVERGENCE90.json','ITEM_RECONSTRUCTION90.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE90_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE90_RESULTS.md')
git('add','research/restore90','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore90/logs',':(exclude)research/restore90/THEORY_REGISTERED90.md')
git('commit','-m','RESTORE90: reconstruct statistical action and work outputs; finite-support convergence')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE90.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE90.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE90_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST90.json'),source_zip_sha256=sha(S/'MQFT_RESTORE90_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE90.git.bundle'),progress_sha256=sha(S/'PROGRESS90.json'),fresh_paths=0,fresh_backups=0,figures=7,cumulative_retained_pairs=685,original_union=121,accepted_corrected_subclaims=25,corrected_report_artifact_accepted=16,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
