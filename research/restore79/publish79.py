from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore78_20261127';A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='2c64b451ad3f14b2c0a48ec55263fd88616d1293'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
a=read(S/'AUDIT79.json');i=read(S/'INVENTORY_AUDIT79.json');v=read(S/'INDEPENDENT79.json');h=read(S/'HONEST_SUMMARY79.json');q=read(S/'ITEM_ACCEPTANCE79.json');pm=read(S/'PLOT_MANIFEST79.json');accept=read(S/'L31_REPORT_ACCEPTANCE79.json');reg=read(S/'REGISTRATION.json')
assert a['paths']==768 and a['train_paths']==192 and a['backups']==98304 and a['train_backups']==24576
assert i['verified_pairs']==640 and i['status']=='PASS_SCOPED' and i['function_spans']==94 and i['guide_literal_matches']==27 and len(i['stage_manifest_checks'])==781 and not i['issues']
assert v['status']=='PASS_SCOPED' and v['raw_paths']==144 and v['independent_fits']==9 and v['occupancy_gradient_checks']==4608 and len(h['panels'])==9 and accept['status']=='PASS_CORRECTED_REPORT_ARTIFACT'
assert all(sha(B/p)==x for p,x in h['inputs'].items()) and all(sha(S/k)==x for k,x in pm['sources'].items()) and all(sha(S/k)==x for k,x in a['arrays'].items())
old=read(O/'ORIGINAL_SOURCE78.json');src=read(S/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==old['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
r0=read(O/'REGISTRATION.json');assert reg['weights']==r0['weights'] and reg['external']==r0['external'] and sha(S/'mqft_train')==reg['binary_sha']==r0['binary_sha'] and sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==r0['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha']
raw=[x for x in a['raw_receipts'] if x['cohort']==1];priorraw=[x for x in a['raw_receipts'] if x['cohort']==0];assert len(raw)==768 and len(priorraw)==192
scopes={'G02':'Independent conditional score scale and occupancy fit distinction, not physical temperature.', 'L03':'Independent native conditional likelihood and full-support alternatives; exact slope, flat cases and no physical temperature.', 'L04':'Budget-dependent endpoint occupancy projection with negative and boundary fits; no universal scaling exponent.', 'L05':'Typed measurement quality, original source and historical manifest/plot identity audit; no heterogeneous total measurement count.', 'L25':'Conditional scale versus occupancy Q projection with actual heldout loss, no physical T.', 'L26':'Actual budget-dependent endpoint fit trajectories including invalid/negative/boundary; separate known conditional beta.', 'L31':'Corrected original nine-panel honest summary from authenticated actual and declared reference data; artifact accepted, original physical claims not thereby certified.', 'L34':'Typed measurement quality, claim dispositions and actual source identity; no unmeasured temperature or critical point.'}
ids={k for f in pm['figures'] for k in f['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==len(q['items'])==121
prev=read(O/'PROGRESS78.json');inv=cr(O/'PLOT_INVENTORY78.csv');assert len(inv)==640 and sha(O/'PLOT_INVENTORY78.csv')==prev['inventory_sha256']
assert len(pm['figures'])==4 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(f.relative_to(S)) for f in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==644
cw(S/'PLOT_INVENTORY79.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS78.csv');ev=[]
for x in rows:
 k=x['id'];x['R79_evidence']=scopes.get(k,'');x['R79_scope_status']='CORRECTED_REPORT_ARTIFACT_ACCEPTED' if k=='L31' else 'SCOPED_INDEPENDENT_CALIBRATION_OR_QUALITY_PANEL' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[f['figure'] for f in pm['figures'] if k in f['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R79_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE79.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS79.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{r}_evidence') for r in range(55,79))}
union={x['id'] for x in rows if any(x.get(f'R{r}_evidence') for r in range(55,80))}
assert len(prior)==120 and len(union)==121 and union-prior=={'L31'} and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=644,distinct_png_sha=644,files_paired=1288,current_pairs=4,prior_pairs=640,current_animations=0,animation_scope='Historical animations retained.',inventory_sha256=sha(S/'PLOT_INVENTORY79.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE79.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS79.csv'),corrected_report_artifact_accepted=['L31'],final_scientific_certified=0)
prev['stages'][S.name]=4;prev['original_output_evidence']=dict(current=len(ids),R55_R79_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='R55-R79 evidence coverage complete; final original science certification separate. L31 corrected report artifact accepted.')
dump(S/'PROGRESS79.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for n in dag['nodes']:
 if n['id']=='REPLAN-01':n['restore79_followup']=dict(evidence='All 121 original source items authenticated and scoped evidence linked; 640 historical plot pairs, 781 run-manifest file identities checked. Corrected nine-panel L31 report accepted at artifact level. 121 item-specific convergence conditions.',remaining='Original physics certifications remain 0/121; L02/L19 legacy grouped numerical evidence lacks unique item figure mapping; no claim of all historical raw paths rerun.',artifact=S.name+'/{QUALITY_MATRIX79.csv,ITEM_ACCEPTANCE79.json,HONEST_SUMMARY79.json,L31_REPORT_ACCEPTANCE79.json}')
 if n['id']=='REPLAN-02':n['restore79_followup']=dict(evidence='768 independent native paths, 192 authenticated train paths, conditional score beta and heldout log score/KL, full-support occupancy fit and nonidentifiable first waves.',remaining='Physical temperature and energy not identified; occupancy not conditional law.',artifact=S.name+'/CALIBRATION79.json')
dag.update(revision='RESTORE79_20261127',scope='Original L31 corrected report, complete scoped item coverage, independent native calibration and convergence contracts',previous_entry=dag['entry'],entry='REPLAN-01',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE80-CONTROLLED-STATISTICAL-1PI-ACTION',mainline_priorities=['R80 controlled finite statistical action and 1PI','R81 operational complex phase intervention and matched dephased control','Item-specific final scientific acceptance, not evidence count; physical scale/thermal and memory/structure branches'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX79.csv')
files=['REGISTRATION.json','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS79.json','AUDIT79.json','CALIBRATION79.json','INDEPENDENT79.json','INVENTORY_AUDIT79.json','EVIDENCE_LINKS79.json','ORIGINAL_SOURCE79.json','ORIGINAL_SOURCE_SPANS79.md','HONEST_SUMMARY79.json','L31_REPORT_ACCEPTANCE79.json','QUALITY_MATRIX79.csv','ITEM_ACCEPTANCE79.json','ORIGINAL_QUALITY_REPORT79.md','PLOT_MANIFEST79.json','PLOT_INVENTORY79.csv','ORIGINAL_OUTPUT_EVIDENCE79.csv','ORIGINAL_OUTPUT_PROGRESS79.csv','PROGRESS79.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='scoped_original_coverage_completed_corrected_report_accepted',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=768,cumulative_fixed_root_histories=82936,fresh_backups=98304,reused_training_paths=192,raw_bytes=a['raw_bytes'],historical_plot_pairs_verified=640,historical_manifest_files_verified=781,source_functions=94,guide_labels=27,new_games=0,optimizer_updates=0,figures=4,animations=0,retained_pairs=644,original_evidence_items=len(ids),original_evidence_union_R55_R79=121,original_obligations=121,corrected_report_artifact_accepted=1,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Item-specific scientific acceptance, controlled statistical 1PI/action, operational phase, physical thermal/scale identification.')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore79']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore79']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights'],pulse_binary_sha=reg['pulse_binary_sha']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit_native74.py','audit_rapfi74.py','pulse74.patch','plan79.py','generate79.py','calibrate79.py','inventory79.py','verify79.py','assemble79.py','plot79.py','acceptance79.tsv','publish79.py','backup79.py','execute79.sh','RESULTS_KO.md','STAGE_LEDGER.md']
inputs=h['inputs'].copy();inputs[str((O/'AUDIT78.json').relative_to(B))]=sha(O/'AUDIT78.json')
manifest=dict(schema='mqft_restore79_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute79.sh (registered raw); plan79.py and generate79.py pinned backend',files={f:sha(S/f) for f in files+sources+plotfiles+['DAG.json','PROVENANCE.json','mqft_train','mqft_pulse74']+list(a['arrays'])},raw_receipts=raw,prior_raw_receipts=priorraw,weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation. No full historical raw replay, full legacy regression, independent seed-to-stream replay, expert adjudication or physical T/phase identification. L31 corrected report acceptance separate from scientific certification.')
dump(S/'RUN_MANIFEST79.json',manifest)
dst=P/'research/restore79';assert not dst.exists();dst.mkdir()
for f in files+sources+plotfiles+['DAG.json','PROVENANCE.json','RUN_MANIFEST79.json']:
 z=dst/f;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,z)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS79.csv','ORIGINAL_OUTPUT_EVIDENCE79.csv','PLOT_INVENTORY79.csv','PROGRESS79.json','QUALITY_MATRIX79.csv','ORIGINAL_QUALITY_REPORT79.md']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE79_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE79_RESULTS.md')
git('add','research/restore79','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore79/pulse74.patch',':(exclude)research/restore79/ORIGINAL_SOURCE_SPANS79.md');git('commit','-m','RESTORE79: original honest summary and convergence acceptance matrix')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE79.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE79.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE79_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST79.json'),source_zip_sha256=sha(S/'MQFT_RESTORE79_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE79.git.bundle'),progress_sha256=sha(S/'PROGRESS79.json'),fresh_paths=768,fresh_backups=98304,figures=4,cumulative_retained_pairs=644,original_union=121,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
