from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore82_20261127';R81=B/'mqft_restore81_20261127';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='fda7af649e105cd14cdc5099f0ea654c83267cb4'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
a=read(S/'AUDIT83.json');t=read(S/'TREE83.json');q=read(S/'SUMMARY83.json');r=read(S/'RECORD83.json');v=read(S/'INDEPENDENT83.json');fr=read(S/'FRAGMENT_AUDIT83.json');c=read(S/'CONVERGENCE83.json');pm=read(S/'PLOT_MANIFEST83.json');reg=read(S/'REGISTRATION.json');build=read(S/'BUILD83.json');patch=read(S/'BRANCH_PATCH83.json')
assert a['paths']==648 and a['forced_paths']==576 and a['reference_replays']==72 and a['diagnostic_backups']==82944 and a['fresh_paths']==0
assert v['status']=='PASS_SCOPED' and v['raw_forced_paths']==576 and v['independent_leaf_branches']==1152 and v['independent_record_matrices']==720 and fr['status']=='PASS_SCOPED' and fr['fragment_cases']==432
assert sha(S/'TREE83.json')==a['tree_sha']==r['tree_sha'] and sha(S/'RECORD_MATRICES83.npz')==r['matrix_sha']
assert len(t['contexts'])==len(r['contexts'])==72 and q['contexts_branch_dependent']==72 and q['native_NPT_any']==31 and q['diagnostic_NPT_any']==67 and q['fragment_90_cases']==0
assert build['status']=='PASS_BUILD_AND_LIB_SUITE' and sha(S/'logs/test83.log')==build['test_log_sha'] and '394 passed; 0 failed; 14 ignored' in (S/'logs/test83.log').read_text()
assert c['accepted_subclaims']==9 and c['new_accepted_subclaims']==2 and c['final_original_item_certifications']==0
assert all(sha(S/k)==x for k,x in pm['sources'].items())
src=read(S/'ORIGINAL_SOURCE83.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE82.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
r0=read(R81/'REGISTRATION.json');assert reg['weights']==r0['weights'] and reg['external']==r0['external'] and sha(S/'mqft_train')==reg['binary_sha']==r0['binary_sha'] and sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==r0['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha'] and sha(S/'mqft_phase81')==reg['phase_binary_sha']==r0['phase_binary_sha'] and sha(S/'phase81.patch')==reg['phase_patch_sha']
assert sha(S/'mqft_branch83')==reg['branch_binary_sha']==build['binary_sha'] and sha(S/'branch83.patch')==reg['branch_patch_sha']==build['patch_sha']
base=R81/'phase_src';assert all(sha(base/k)==h for k,h in patch['base_source_files'].items())
assert all(sha(S/'branch_src/src'/k)==x['modified_sha'] and sha(base/'src'/k)==x['original_sha'] for k,x in patch['patches'].items())
diag=a['raw_receipts'];prior=[dict(path=z['source'],sha256=z['source_sha'],role='R81_observed_H32_context') for z in t['contexts']]
assert len(diag)==len({z['path'] for z in diag})==648 and len(prior)==len({z['path'] for z in prior})==72
assert all(sha(B/z['path'])==z['sha256'] for z in diag+prior)
assert all(sha(S/'plans'/f'{k}.json')==z['sha256'] for k,z in reg['plans'].items()) and sha(S/'CONTEXTS83.json')==reg['contexts_sha']
scopes={'L15':'Actual adaptive three-wave declared record density on 576 selected native histories; exact abort mass, distinct diagnostic, no native physical coherence/objectivity.', 'L16':'Branch-dependent coherent operator products with actual native scores, exact selected path probability and rare-event visibility; no full physical complex action.', 'L20':'Adaptive CP selective instrument and complement abort effect, distinct controlled pair restriction; no continuous physical GKSL.', 'Q24':'Three-wave actual branch-dependent record construction with explicit bath labels and dephased control; physical bath not identified.', 'Q25':'Secondary first-bit pointer/fragment Holevo and classical MI on declared adaptive models; no 90% disjoint packing.', 'Q35':'Explicit adaptive record PSD density and NPT witnesses in 31/72 native success-conditional and 67/72 pair diagnostic contexts; no native physical entanglement.', 'Q37':'Actual branch probability and declared coherent readout/fragment information, success mass and nonunique dilation.', 'Q42':'Discrete adaptive CP construction with branch-dependent downstream and positive abort; no physical time or hbar.', 'Q43':'Actual selected path product probabilities and declared coherent amplitude product, full stopped normalization; no physical action.', 'Q45':'Bounded adaptive quantum record and constructive ambiguity, negative redundancy and rare-event limitations; full original acceptance open.'}
ids={k for z in pm['figures'] for k in z['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS82.json');inv=cr(O/'PLOT_INVENTORY82.csv');assert len(inv)==650+3 and sha(O/'PLOT_INVENTORY82.csv')==prev['inventory_sha256']
assert len(pm['figures'])==3 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==656;cw(S/'PLOT_INVENTORY83.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS82.csv');ev=[]
for x in rows:
 k=x['id'];x['R83_evidence']=scopes.get(k,'');x['R83_scope_status']='CORRECTED_SUBCLAIM_OR_DECLARED_ADAPTIVE_MODEL_EVIDENCE' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[p['figure'] for p in pm['figures'] if k in p['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R83_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE83.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS83.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{n}_evidence') for n in range(55,84))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX82.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==83 and k in z['items']]
 x['R83_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R83_open_subclaims']=';'.join(z['id'] for z in c['claims'] if not z['status'].startswith('ACCEPTED') and k in z['items']) if k in scopes else ''
 x['R83_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='83';x['latest_scope']=scopes[k];x['corrected_output_state']='CORRECTED_SUBCLAIM_OR_MODEL_EVIDENCE_R83'
cw(S/'QUALITY_MATRIX83.csv',qm)
dump(S/'ITEM_CONVERGENCE83.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX82.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE82.json'),items=[dict(id=x['id'],accepted=x['R83_convergence_subclaims'],open=x['R83_open_subclaims'],scope=x['R83_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R83_scope']],claims=[z for z in c['claims'] if z['stage']==83],scope='R83 delta; R79 item contracts and R80-R82 accepted claims retained.'))
prev.update(figure_pairs=656,distinct_png_sha=656,files_paired=1312,current_pairs=3,prior_pairs=653,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY83.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE83.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS83.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX83.csv'),corrected_report_artifact_accepted=['L31'],accepted_corrected_subclaims=9,final_scientific_certified=0)
prev['stages'][S.name]=3;prev['original_output_evidence']=dict(current=len(ids),R55_R83_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Complete scoped coverage unchanged; 9 corrected subclaims accepted (+2), L31 artifact retained; final original scientific certification separate.')
dump(S/'PROGRESS83.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('X-REP','REPLAN-05'):node['restore83_followup']=dict(evidence='72 actual observed H32 contexts, 576 forced leaf replays, 72 exact unforced replays, branch-dependent 504 settings, complete stopped CP instrument, explicit shared-bath vs history-tag completion, distinct normalized pair diagnostic and fragment checks.',accepted_subclaims=['Q-ADAPT-05','Q-DILATION-06'],remaining='Q-RECORD-03 physical/operational native coherence and objectivity not identified. Rare native repeat success, no disjoint 90% packing, no native physical entanglement.',artifact=S.name+'/{CONVERGENCE83.json,TREE83.json,RECORD83.json,FRAGMENTS83.json,INDEPENDENT83.json}')
dag.update(revision='RESTORE83_20261127',scope='Convergence: actual adaptive branch-dependent declared coherent records, exact selective mass and nonidentifiability',previous_entry=dag['entry'],entry='X-REP',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE84-PRESEARCH-CHILD-LANDSCAPE-POPULATION-TRANSFER',mainline_priorities=['R84 pre-search child evaluator landscape and independent root-only population transfer','R85 thermal/reverse calibration candidate; R86 autonomous memory/scale','Actual operational coherent record only with explicit independent readout; no repeat frozen models'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX83.csv',convergence_ledger='CONVERGENCE83.json')
files=['REGISTRATION.json','CONTEXTS83.json','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS83.json','BUILD83.json','BRANCH_PATCH83.json','AUDIT83.json','TREE83.json','RECORD83.json','FRAGMENTS83.json','FRAGMENT_AUDIT83.json','SUMMARY83.json','INDEPENDENT83.json','ORIGINAL_SOURCE83.json','CONVERGENCE83.json','ITEM_CONVERGENCE83.json','QUALITY_MATRIX83.csv','PLOT_MANIFEST83.json','PLOT_INVENTORY83.csv','ORIGINAL_OUTPUT_EVIDENCE83.csv','ORIGINAL_OUTPUT_PROGRESS83.csv','PROGRESS83.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='two_new_corrected_scientific_subclaims_accepted_physical_record_open',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=89080,fresh_backups=0,diagnostic_paths=648,forced_diagnostics=576,reference_replays=72,diagnostic_backups=82944,actual_contexts=72,actual_node_settings=504,raw_bytes=a['raw_bytes'],new_games=0,optimizer_updates=0,figures=3,animations=0,retained_pairs=656,original_evidence_items=len(ids),original_evidence_union_R55_R83=121,original_obligations=121,corrected_report_artifact_accepted=1,accepted_corrected_subclaims=9,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Root-only population transfer, physical native record identification/objectivity and utility; original item acceptance.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore83']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore83']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights'],phase_binary_sha=reg['phase_binary_sha'],branch_binary_sha=reg['branch_binary_sha']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','instrument81.py','record81.py','pulse74.patch','phase81.patch','branch83.patch','build83.py','plan83.py','generate83.py','audit83.py','adaptive83.py','verify83.py','fragments83.py','summary83.py','plot83.py','publish83.py','backup83.py','execute83.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=['logs/build83.log','logs/test83.log','logs/generate83.log','logs/audit83.log','logs/adaptive83.log','logs/verify83.log','logs/fragments83.log','logs/summary83.log','logs/plot83.log']
inputs={str((O/z).relative_to(B)):sha(O/z) for z in ['REGISTRATION.json','CONVERGENCE82.json','QUALITY_MATRIX82.csv','ORIGINAL_SOURCE82.json']}
for z in ['REGISTRATION.json','AUDIT81.json','PHASE_PATCH81.json','BUILD81.json']:inputs[str((R81/z).relative_to(B))]=sha(R81/z)
for k,h in patch['base_source_files'].items():inputs[str((base/k).relative_to(B))]=h
manifest=dict(schema='mqft_restore83_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute83.sh (registered diagnostic data); build83.py, plan83.py, generate83.py for acquisition',files={z:sha(S/z) for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','mqft_train','mqft_pulse74','mqft_phase81','mqft_branch83','RECORD_MATRICES83.npz']},raw_receipts=[],prior_raw_receipts=prior,diagnostic_receipts=diag,weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation, isolated Rust lib suite only. Forced diagnostic replays, not independent samples. Declared coherent completion not native coherent recombination or physical phase.')
dump(S/'RUN_MANIFEST83.json',manifest)
dst=P/'research/restore83';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST83.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
shutil.copytree(S/'plans',dst/'plans')
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS83.csv','ORIGINAL_OUTPUT_EVIDENCE83.csv','PLOT_INVENTORY83.csv','PROGRESS83.json','QUALITY_MATRIX83.csv','CONVERGENCE83.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE83_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE83_RESULTS.md')
git('add','research/restore83','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore83/pulse74.patch',':(exclude)research/restore83/phase81.patch',':(exclude)research/restore83/branch83.patch',':(exclude)research/restore83/logs')
git('commit','-m','RESTORE83: adaptive coherent record extension and exact selective mass')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE83.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE83.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE83_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST83.json'),source_zip_sha256=sha(S/'MQFT_RESTORE83_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE83.git.bundle'),progress_sha256=sha(S/'PROGRESS83.json'),fresh_paths=0,fresh_backups=0,diagnostic_paths=648,diagnostic_backups=82944,figures=3,cumulative_retained_pairs=656,original_union=121,accepted_corrected_subclaims=9,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
