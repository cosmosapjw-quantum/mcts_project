from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore88_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';parent='25860fa8209493a1738e7a650e47d6695ac14ff3'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
q=read(S/'SUMMARY89.json');v=read(S/'INDEPENDENT89.json');c=read(S/'CONVERGENCE89.json');pm=read(S/'PLOT_MANIFEST89.json');r=read(S/'ITEM_RECONSTRUCTION89.json');reg=read(S/'REGISTRATION.json');n=read(S/'NATIVE89.json')
assert q['raw_paths']==3200 and q['registered_games']==80 and q['complete_games']==78 and q['eval_records']==579026 and q['new_fixed_roots']==0
assert v['status']=='PASS_SCOPED' and v['coherent_repeated_checks']==465 and v['posterior_cases']==24 and n['max_cut_error']==0
assert c['accepted_subclaims']==23 and c['new_accepted_subclaims']==2 and c['final_original_item_certifications']==0 and c['artifact_reconstruction_sha']==sha(S/'ITEM_RECONSTRUCTION89.json')
assert sha(S/'THEORY_REGISTERED89.md')==reg['contract_sha'] and r['corrected_artifacts_total']==11 and all(x['decision']=='ACCEPTED_CORRECTED_SCOPED_ARTIFACT' for x in r['items'])
assert all(sha(S/k)==x for k,x in pm['sources'].items()) and all(sha(B/k)==x for k,x in pm['prior_sources'].items()) and len(pm['figures'])==2
src=read(S/'ORIGINAL_SOURCE89.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE88.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
m67=read(B/'mqft_restore67_20261126/RUN_MANIFEST67.json');m68=read(B/'mqft_restore68_20261126/RUN_MANIFEST68.json')
prior=n['raw_receipts'];assert len(prior)==3280 and len({x['path'] for x in prior})==3280 and all(sha(B/x['path'])==x['sha256'] for x in prior)
scopes={x['id']:x['method']+' '+x['remaining_limit'] for x in r['items']};artifact={x['id']:x for x in r['items']}
assert set(scopes)=={k for z in pm['figures'] for k in z['ids']}=={'L21','L23'}
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS88.json');inv=cr(O/'PLOT_INVENTORY88.csv');assert len(inv)==676 and sha(O/'PLOT_INVENTORY88.csv')==prev['inventory_sha256']
plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==678;cw(S/'PLOT_INVENTORY89.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS88.csv');ev=[]
for x in rows:
 k=x['id'];x['R89_evidence']=scopes.get(k,'');x['R89_scope_status']='ACCEPTED_CORRECTED_SCOPED_ARTIFACT' if k in scopes else ''
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=artifact[k]['figure'],source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R89_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE89.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS89.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{k}_evidence') for k in range(55,90))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX88.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==89 and k in z['items']]
 x['R89_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R89_open_subclaims']=';'.join(z['id'] for z in linked if not z['status'].startswith('ACCEPTED'));x['R89_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='89';x['latest_scope']=scopes[k];x['corrected_output_state']='ACCEPTED_CORRECTED_SCOPED_ARTIFACT_R89'
cw(S/'QUALITY_MATRIX89.csv',qm)
dump(S/'ITEM_CONVERGENCE89.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX88.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE88.json'),items=[dict(id=x['id'],accepted=x['R89_convergence_subclaims'],scope=x['R89_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R89_scope']],claims=[z for z in c['claims'] if z['stage']==89],artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION89.json'),scope='Two corrected artifact acceptances; original science not certified.'))
accepted=c['corrected_report_artifacts_accepted'];assert len(accepted)==len(set(accepted))==11
prev.update(figure_pairs=678,distinct_png_sha=678,files_paired=1356,current_pairs=2,prior_pairs=676,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY89.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE89.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS89.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX89.csv'),corrected_report_artifact_accepted=accepted,accepted_corrected_subclaims=23,final_scientific_certified=0)
prev['stages'][S.name]=2;prev['original_output_evidence']=dict(current=len(scopes),R55_R89_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='121 scoped coverage; eleven corrected artifact acceptances,23 corrected subclaims; final science separate.')
dump(S/'PROGRESS89.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('X-STRUCT','X-REP'):node['restore89_followup']=dict(evidence='3200 actual fixed-root histories,80 registered games,579026 evaluator records; exact syndrome posterior and conditional coherent quantum recovery; L23 and L21 reconstruction.',accepted_subclaims=['QEC-POSTERIOR-01','QEC-COHERENT-02'],corrected_artifacts=list(scopes),remaining='No general native quantum/hardware or utility admission; original scientific certification separate.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT89.md,ITEM_RECONSTRUCTION89.json,CONVERGENCE89.json}')
dag.update(revision='RESTORE89_20261128',scope='Convergence: QEC and integrated advanced quantum original report',previous_entry=dag['entry'],entry='X-STRUCT',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE90-STATISTICAL-ACTION-ORIGINAL-OUTPUTS',mainline_priorities=['R90 reconstruct original action and effective potential/CGF panels using R80–R87 evidence','Direct original item reconstruction and acceptance, no repetitive diagnostic grids','Six open scientific claims remain distinct from corrected artifacts'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX89.csv',convergence_ledger='CONVERGENCE89.json',backup_policy='DROPBOX_ONLY_VERIFIED_SNAPSHOT_NO_REDUNDANT_LOCAL_BACKUP')
files=['REGISTRATION.json','THEORY_REGISTERED89.md','QEC89.json','NATIVE89.json','SOURCE_CORRECTIONS89.json','INDEPENDENT89.json','SUMMARY89.json','ORIGINAL_SOURCE89.json','CONVERGENCE89.json','ITEM_RECONSTRUCTION89.json','ORIGINAL_OUTPUT_REPORT89.md','ITEM_CONVERGENCE89.json','QUALITY_MATRIX89.csv','PLOT_MANIFEST89.json','PLOT_INVENTORY89.csv','ORIGINAL_OUTPUT_EVIDENCE89.csv','ORIGINAL_OUTPUT_PROGRESS89.csv','PROGRESS89.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='two_corrected_original_artifacts_two_new_qec_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=3200,reused_games=80,source_path_receipts=3280,eval_records=579026,figures=2,animations=0,retained_pairs=678,original_evidence_items=2,original_evidence_union_R55_R89=121,original_obligations=121,corrected_report_artifact_accepted=11,new_corrected_artifacts=2,accepted_corrected_subclaims=23,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='110 original corrected artifact obligations; six open science claims. Dropbox-only backup policy.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore89']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore89']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=m68['external'],weights=m67['weights']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','qec89.py','native89.py','summary89.py','verify89.py','plot89.py','report89.py','publish89.py','backup89.py','execute89.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=[f'logs/{z}.log' for z in ['qec89','native89','summary89','verify89','plot89']]
inputs={}
for st,day,fn in [(67,26,['AUDIT67.json','REGISTRATION.json','NATIVE67.npz','REFERENCE67.npz','RAPFI67.json','PHYSICS_RESULTS.json','QEC67.json','RUN_MANIFEST67.json']),(68,26,['CAMPAIGN68.json','AUDIT68.json','REGISTRATION.json','PHYSICS_RESULTS.json','RUN_MANIFEST68.json']),(79,27,['ITEM_ACCEPTANCE79.json','ORIGINAL_SOURCE79.json']),(88,28,['ITEM_RECONSTRUCTION88.json','CONVERGENCE88.json','QUALITY_MATRIX88.csv','GAUGE88.json','TOPOLOGY88.json'])]:
 p=B/f'mqft_restore{st}_202611{day}'
 for z in fn:inputs[str((p/z).relative_to(B))]=sha(p/z)
inputs.update(pm['prior_sources'])
manifest=dict(schema='mqft_restore89_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute89.sh on authenticated reused raw histories',files={z:sha(S/z) for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json']},raw_receipts=[],prior_raw_receipts=prior,diagnostic_receipts=[],weights=m67['weights'],external=m68['external'],plan_hashes={},inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent independent checks. No native physical QEC, universal error correction, or broad utility admission; unchanged native full replay inherited.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST89.json',manifest)
dst=P/'research/restore89';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST89.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS89.csv','ORIGINAL_OUTPUT_EVIDENCE89.csv','PLOT_INVENTORY89.csv','PROGRESS89.json','QUALITY_MATRIX89.csv','CONVERGENCE89.json','ITEM_RECONSTRUCTION89.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE89_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE89_RESULTS.md')
git('add','research/restore89','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore89/logs',':(exclude)research/restore89/THEORY_REGISTERED89.md')
git('commit','-m','RESTORE89: reconstruct QEC and advanced report; Dropbox-only backup policy')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE89.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE89.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE89_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST89.json'),source_zip_sha256=sha(S/'MQFT_RESTORE89_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE89.git.bundle'),progress_sha256=sha(S/'PROGRESS89.json'),fresh_paths=0,fresh_backups=0,figures=2,cumulative_retained_pairs=678,original_union=121,accepted_corrected_subclaims=23,corrected_report_artifact_accepted=11,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
