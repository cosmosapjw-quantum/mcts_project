from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore87_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';parent='874d38653b98fea197ccb849a479afa9c804d1ed'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
q=read(S/'SUMMARY88.json');v=read(S/'INDEPENDENT88.json');c=read(S/'CONVERGENCE88.json');pm=read(S/'PLOT_MANIFEST88.json');r=read(S/'ITEM_RECONSTRUCTION88.json');reg=read(S/'REGISTRATION.json')
assert q['fresh_paths']==0 and q['all_receipts']==3136 and q['eligible']==3071 and q['events']==2783 and q['censored']==288
assert v['status']=='PASS_SCOPED' and v['joint_loops']==4005 and v['threshold_checks']==25648 and read(S/'PERSISTENCE_RANK88.json')['rank_checks']==6048
assert c['accepted_subclaims']==21 and c['new_accepted_subclaims']==3 and c['final_original_item_certifications']==0 and c['artifact_reconstruction_sha']==sha(S/'ITEM_RECONSTRUCTION88.json')
assert sha(S/'THEORY_REGISTERED88.md')==reg['contract_sha'] and r['corrected_artifacts_total']==9 and all(x['decision']=='ACCEPTED_CORRECTED_SCOPED_ARTIFACT' for x in r['items'])
assert all(sha(S/k)==x for k,x in pm['sources'].items()) and len(pm['figures'])==8
src=read(S/'ORIGINAL_SOURCE88.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE87.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
m75=read(B/'mqft_restore75_20261127/RUN_MANIFEST75.json');m78=read(B/'mqft_restore78_20261127/RUN_MANIFEST78.json')
assert m75['weights']==m78['weights'] and m75['external']==m78['external']
prior=read(S/'BARRIER88.json')['raw_receipts']+read(S/'GAUGE88.json')['raw_receipts']+read(S/'RECOVERY88.json')['raw_receipts']
assert len(prior)==3136 and len({x['path'] for x in prior})==3136 and all(sha(B/x['path'])==x['sha256'] for x in prior)
scopes={x['id']:x['method']+' '+x['remaining_limit'] for x in r['items']};artifact={x['id']:x for x in r['items']}
assert set(scopes)=={k for z in pm['figures'] for k in z['ids']}
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS87.json');inv=cr(O/'PLOT_INVENTORY87.csv');assert len(inv)==668 and sha(O/'PLOT_INVENTORY87.csv')==prev['inventory_sha256']
plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==676;cw(S/'PLOT_INVENTORY88.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS87.csv');ev=[]
for x in rows:
 k=x['id'];x['R88_evidence']=scopes.get(k,'');x['R88_scope_status']='ACCEPTED_CORRECTED_SCOPED_ARTIFACT' if k in scopes else ''
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=artifact[k]['figure'],source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R88_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE88.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS88.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{k}_evidence') for k in range(55,89))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX87.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==88 and k in z['items']]
 x['R88_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R88_open_subclaims']=';'.join(z['id'] for z in linked if not z['status'].startswith('ACCEPTED'));x['R88_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='88';x['latest_scope']=scopes[k];x['corrected_output_state']='ACCEPTED_CORRECTED_SCOPED_ARTIFACT_R88'
cw(S/'QUALITY_MATRIX88.csv',qm)
dump(S/'ITEM_CONVERGENCE88.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX87.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE87.json'),items=[dict(id=x['id'],accepted=x['R88_convergence_subclaims'],scope=x['R88_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R88_scope']],claims=[z for z in c['claims'] if z['stage']==88],artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION88.json'),scope='Eight corrected artifact acceptances; original full science not certified.'))
accepted=c['corrected_report_artifacts_accepted'];assert len(accepted)==len(set(accepted))==9
prev.update(figure_pairs=676,distinct_png_sha=676,files_paired=1352,current_pairs=8,prior_pairs=668,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY88.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE88.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS88.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX88.csv'),corrected_report_artifact_accepted=accepted,accepted_corrected_subclaims=21,final_scientific_certified=0)
prev['stages'][S.name]=8;prev['original_output_evidence']=dict(current=len(scopes),R55_R88_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='121 scoped coverage unchanged; nine corrected artifact acceptances,21 corrected subclaims, final science separate.')
dump(S/'PROGRESS88.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('X-STRUCT','X-BARRIER','REPLAN-04.R'):node['restore88_followup']=dict(evidence='3120 actual fixed-root histories and16 games reauthenticated; 48 gauge,144 topology,96 transport contexts; exact joint invariants, persistence rank and minimax, eight corrected original plates/report.',accepted_subclaims=['STRUCT-HOLONOMY-01','STRUCT-PERSISTENCE-02','STRUCT-BARRIER-03'],corrected_artifacts=list(scopes),remaining='L23 QEC, L21 combined advanced report, native operational gauge/physical tunneling unestablished; original final scientific certification separate.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT88.md,ITEM_RECONSTRUCTION88.json,CONVERGENCE88.json}')
dag.update(revision='RESTORE88_20261128',scope='Convergence: direct corrected original structural outputs',previous_entry=dag['entry'],entry='X-STRUCT',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE89-QEC-AND-ADVANCED-REPORT-RECONSTRUCTION',mainline_priorities=['R89 L23 QEC and L21 integrated advanced report','Direct original item reconstruction and acceptance, no repetitive diagnostic grids','Existing six open scientific claims separate from corrected artifact completion'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX88.csv',convergence_ledger='CONVERGENCE88.json')
files=['REGISTRATION.json','THEORY_REGISTERED88.md','GAUGE88.json','TOPOLOGY88.json','BARRIER88.json','TRANSPORT88.json','RECOVERY88.json','LANDSCAPE88.json','SOURCE_CORRECTIONS88.json','INDEPENDENT88.json','PERSISTENCE_RANK88.json','SUMMARY88.json','ORIGINAL_SOURCE88.json','CONVERGENCE88.json','ITEM_RECONSTRUCTION88.json','ORIGINAL_OUTPUT_REPORT88.md','ITEM_CONVERGENCE88.json','QUALITY_MATRIX88.csv','PLOT_MANIFEST88.json','PLOT_INVENTORY88.csv','ORIGINAL_OUTPUT_EVIDENCE88.csv','ORIGINAL_OUTPUT_PROGRESS88.csv','PROGRESS88.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='eight_corrected_original_artifacts_three_new_structural_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=3120,reused_games=16,source_path_receipts=3136,gauge_contexts=48,topology_contexts=144,transport_contexts=96,figures=8,animations=0,retained_pairs=676,original_evidence_items=8,original_evidence_union_R55_R88=121,original_obligations=121,corrected_report_artifact_accepted=9,new_corrected_artifacts=8,accepted_corrected_subclaims=21,new_accepted_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='L23/L21 and other original output completion; native physical/operational identification, six registered open claims.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore88']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore88']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=m75['external'],weights=m75['weights']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','gauge78.py','transport75.py','barrier88.py','gauge88.py','topology88.py','transport88.py','verify88.py','landscape88.py','summary88.py','persistence_rank88.py','plot88.py','report88.py','publish88.py','backup88.py','execute88.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=[f'logs/{z}.log' for z in ['barrier88','gauge88','topology88','transport88','verify88','landscape88','persistence_rank88','plot88']]
inputs={}
for st,day,fn in [(75,27,['AUDIT75.json','FIRST_RESULTS75.json','FIRST75.npz','TRANSPORT75.json','GAMES75.json','RUN_MANIFEST75.json','INDEPENDENT75.json','transport75.py']),(78,27,['NATIVE78.json','GAUGE78.json','RUN_MANIFEST78.json','ORIGINAL_GAUGE_AUDIT78.json','gauge78.py']),(79,27,['ITEM_ACCEPTANCE79.json','ORIGINAL_SOURCE79.json']),(87,28,['CONVERGENCE87.json','QUALITY_MATRIX87.csv','ERRATUM86.json'])]:
 p=B/f'mqft_restore{st}_202611{day}'
 for z in fn:inputs[str((p/z).relative_to(B))]=sha(p/z)
for z in ['CAMPAIGN68.json','REGISTRATION.json']:p=B/'mqft_restore68_20261126'/z;inputs[str(p.relative_to(B))]=sha(p)
assert sha(S/'gauge78.py')==inputs['mqft_restore78_20261127/gauge78.py'] and sha(S/'transport75.py')==inputs['mqft_restore75_20261127/transport75.py']
manifest=dict(schema='mqft_restore88_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute88.sh on authenticated reused raw histories',files={z:sha(S/z) for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json']},raw_receipts=[],prior_raw_receipts=prior,diagnostic_receipts=[],weights=m75['weights'],external=m75['external'],plan_hashes={},inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate algorithms; no native physical gauge/tunneling or full original science certification; fixed oblique G18 projection, no full legacy regression.')
dump(S/'RUN_MANIFEST88.json',manifest)
dst=P/'research/restore88';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST88.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS88.csv','ORIGINAL_OUTPUT_EVIDENCE88.csv','PLOT_INVENTORY88.csv','PROGRESS88.json','QUALITY_MATRIX88.csv','CONVERGENCE88.json','ITEM_RECONSTRUCTION88.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE88_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE88_RESULTS.md')
git('add','research/restore88','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore88/logs',':(exclude)research/restore88/THEORY_REGISTERED88.md')
git('commit','-m','RESTORE88: reconstruct eight original structural outputs and joint holonomy')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE88.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE88.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE88_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST88.json'),source_zip_sha256=sha(S/'MQFT_RESTORE88_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE88.git.bundle'),progress_sha256=sha(S/'PROGRESS88.json'),fresh_paths=0,fresh_backups=0,figures=8,cumulative_retained_pairs=676,original_union=121,accepted_corrected_subclaims=21,corrected_report_artifact_accepted=9,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
