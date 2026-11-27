from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore85_20261127';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='3a64373980f63c4daea49b6fd255ceb4cc027f04'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
a=read(S/'AUDIT86.json');q=read(S/'SUMMARY86.json');m=read(S/'MEMORY86.json');v=read(S/'INDEPENDENT86.json');sc=read(S/'SCALE86.json');u=read(S/'RNG_AUDIT86.json');c=read(S/'CONVERGENCE86.json');pm=read(S/'PLOT_MANIFEST86.json');reg=read(S/'REGISTRATION.json')
assert a['paths']==1152 and a['backups']==147456 and a['prior_paths']==6336 and a['prior_backups']==811008
assert v['status']=='PASS_SCOPED' and v['cross_tab_pairs']==408 and v['independent_kernel_cases']==276 and v['heldout_path_start_predictions']==677376 and v['raw_reductions']==90 and v['scale_cases']==300
assert sc['waves']==30720 and len(sc['raw_receipts'])==1920 and q['registered_passes']==3 and q['fine_NW_collision_pairs']==0
assert sha(S/'WAVES86.npz')==a['array_sha']==m['array_sha'] and sha(S/'MODEL86.json')==m['model_sha'] and sha(S/'THEORY_REGISTERED86.md')==reg['contract_sha']
assert c['accepted_subclaims']==16 and c['new_accepted_subclaims']==3 and c['final_original_item_certifications']==0 and u['streams']==768 and u['independent_uniforms']==98304
assert all(sha(S/k)==x for k,x in pm['sources'].items())
src=read(S/'ORIGINAL_SOURCE86.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE85.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
r0=read(B/'mqft_restore84_20261127/REGISTRATION.json');assert reg['weights']==r0['weights'] and reg['external']==r0['external'] and sha(S/'mqft_train')==reg['binary_sha']==r0['binary_sha'] and sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==r0['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha']
raw=[x for x in a['raw_receipts'] if x['group']==4];prior=[x for x in a['raw_receipts'] if x['group']!=4]+sc['raw_receipts']
assert len(raw)==1152 and len(prior)==8256 and len({x['path'] for x in raw+prior})==9408
assert all(sha(B/x['path'])==x['sha256'] for x in raw+prior)
assert all(sha(S/'plans'/f'{k}.json')==z['sha256'] for k,z in reg['plans'].items())
scopes={'L16':'Exact finite nonstationary conditional endpoint memory and actual one-wave selector sufficiency; no full marked autonomous path generator.', 'L37':'Actual coarse nonlumpability and signed finite projection memory obstruct naive autonomous RG; no new physical fixed point, original R77 negative finding retained.', 'L38':'Exact compensated martingale dyadic covariance and conditional exponential law with five sizes, three beta; finite-size fluctuation distinct from drift, no CLT or spatial RG claim.', 'L39':'Actual lag-resolved raw and compensated cross-moments, exact coarse collision witnesses and finite memory prediction; no stationary exponential relaxation time.'}
ids={k for z in pm['figures'] for k in z['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS85.json');inv=cr(O/'PLOT_INVENTORY85.csv');assert len(inv)==662 and sha(O/'PLOT_INVENTORY85.csv')==prev['inventory_sha256']
assert len(pm['figures'])==3 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==665;cw(S/'PLOT_INVENTORY86.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS85.csv');ev=[]
for x in rows:
 k=x['id'];x['R86_evidence']=scopes.get(k,'');x['R86_scope_status']='CORRECTED_SUBCLAIM_EVIDENCE' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[p['figure'] for p in pm['figures'] if k in p['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R86_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE86.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS86.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{k}_evidence') for k in range(55,87))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX85.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==86 and k in z['items']]
 x['R86_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R86_open_subclaims']=';'.join(z['id'] for z in linked if not z['status'].startswith('ACCEPTED'))
 x['R86_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='86';x['latest_scope']=scopes[k];x['corrected_output_state']='CORRECTED_SUBCLAIM_EVIDENCE_R86'
cw(S/'QUALITY_MATRIX86.csv',qm)
dump(S/'ITEM_CONVERGENCE86.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX85.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE85.json'),items=[dict(id=x['id'],accepted=x['R86_convergence_subclaims'],open=x['R86_open_subclaims'],scope=x['R86_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R86_scope']],claims=[z for z in c['claims'] if z['stage']==86],scope='R86 delta, original R79 item contracts and prior dispositions retained.'))
prev.update(figure_pairs=665,distinct_png_sha=665,files_paired=1330,current_pairs=3,prior_pairs=662,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY86.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE86.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS86.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX86.csv'),corrected_report_artifact_accepted=['L31'],accepted_corrected_subclaims=16,final_scientific_certified=0)
prev['stages'][S.name]=3;prev['original_output_evidence']=dict(current=len(ids),R55_R86_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Complete scoped coverage unchanged; 16 corrected subclaims accepted (+3); L31 artifact retained; final science separate.')
dump(S/'PROGRESS86.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('X-MEMORY','REPLAN-04.R'):node['restore86_followup']=dict(evidence='1152 actual new and6336 prior B128 paths,1920 B64 scale paths. Exact one-wave count sufficiency, actual coarse nonlumpability, exact nonstationary conditional projection kernel, compensated martingale scale law. Finite order4 external gain passes3/6.',accepted_subclaims=['MEM-SELECTOR-01','MEM-PROJECTION-02','MEM-SCALE-03'],remaining='MEM-CLOSURE-04 finite-memory transfer and autonomous marked closure remain open; no exact full NW collision or physical RG/CLT. Original final item acceptance separate.',artifact=S.name+'/{CONVERGENCE86.json,MEMORY86.json,SCALE86.json,COLLISION_WITNESSES86.json,INDEPENDENT86.json}')
dag.update(revision='RESTORE86_20261127',scope='Convergence: exact nonstationary projected memory and compensated scale',previous_entry=dag['entry'],entry='X-MEMORY',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE87-CONTROLLED-CRITICAL-SCALING',mainline_priorities=['R87 controlled finite-size critical scaling, drift/innovation separation and exact beta0 reference','Remaining structural/original item completion and open autonomous memory only with new constructive mechanism','Physical thermal/coherent identification only with independent calibration or readout; matched-cost utility separate'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX86.csv',convergence_ledger='CONVERGENCE86.json')
files=['REGISTRATION.json','THEORY_REGISTERED86.md','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS86.json','AUDIT86.json','RNG_AUDIT86.json','MODEL86.json','MEMORY86.json','SCALE86.json','COLLISION_WITNESSES86.json','AUTOCORRELATION86.json','SUMMARY86.json','INDEPENDENT86.json','ORIGINAL_SOURCE86.json','CONVERGENCE86.json','ITEM_CONVERGENCE86.json','QUALITY_MATRIX86.csv','PLOT_MANIFEST86.json','PLOT_INVENTORY86.csv','ORIGINAL_OUTPUT_EVIDENCE86.csv','ORIGINAL_OUTPUT_PROGRESS86.csv','PROGRESS86.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='three_new_corrected_memory_subclaims_accepted_autonomous_closure_open',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=1152,cumulative_fixed_root_histories=92536,fresh_backups=147456,reused_B128_paths=6336,reused_scale_paths=1920,source_path_receipts=9408,registered_prediction_passes=3,scale_wave_cases=30720,enumerated_sector_strings=2488320,raw_bytes=a['raw_bytes'],new_games=0,optimizer_updates=0,figures=3,animations=0,retained_pairs=665,original_evidence_items=len(ids),original_evidence_union_R55_R86=121,original_obligations=121,corrected_report_artifact_accepted=1,accepted_corrected_subclaims=16,new_accepted_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Controlled critical scaling, autonomous marked closure, broad population transfer, native coherent readout, physical thermal calibration and utility; original item acceptance.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore86']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore86']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights'],pulse_binary_sha=reg['pulse_binary_sha']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit_native74.py','audit_rapfi74.py','pulse74.patch','plan86.py','generate86.py','extract86.py','memory86.py','scale86.py','verify86.py','summary86.py','plot86.py','publish86.py','backup86.py','execute86.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=['logs/generate86.log','logs/extract86.log','logs/memory86.log','logs/scale86.log','logs/verify86.log','logs/plot86.log']
inputs={str((O/z).relative_to(B)):sha(O/z) for z in ['CONVERGENCE85.json','QUALITY_MATRIX85.csv','ORIGINAL_SOURCE85.json']}
for st,fn in [(82,['AUDIT82.json','POPULATION82.npz','REGISTRATION.json']),(84,['AUDIT84.json','FRESH84.npz','REGISTRATION.json']),(77,['THEORY.md','CLOSURE77.json','RG77.json','ORIGINAL_RG_AUDIT77.json','RUN_MANIFEST77.json'])]:
 r=B/f'mqft_restore{st}_20261127'
 for z in fn:inputs[str((r/z).relative_to(B))]=sha(r/z)
inputs.update(sc['arrays'])
for L in (13,15,17,19,21):
 r=B/'mqft_restore77_20261127'/f'AUDIT_L{L}.json';inputs[str(r.relative_to(B))]=sha(r)
rngp=B/'mqft_restore61_20261126/mqft_trained_rng61';assert sha(rngp)==u['binary_sha'];inputs[str(rngp.relative_to(B))]=u['binary_sha']
manifest=dict(schema='mqft_restore86_convergence_run_v1',parent=parent,status='completed_scoped',command='plan86.py, generate86.py, bash execute86.sh on registered immutable paths',files={z:sha(S/z) for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','mqft_train','mqft_pulse74','WAVES86.npz','RNG_INPUT86.json']},raw_receipts=raw,prior_raw_receipts=prior,diagnostic_receipts=[],weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate algorithms, no full legacy suite. Exact conditional projection and count martingale not autonomous marked path closure or physical RG; prediction gate3/6.')
dump(S/'RUN_MANIFEST86.json',manifest)
dst=P/'research/restore86';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST86.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
shutil.copytree(S/'plans',dst/'plans')
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS86.csv','ORIGINAL_OUTPUT_EVIDENCE86.csv','PLOT_INVENTORY86.csv','PROGRESS86.json','QUALITY_MATRIX86.csv','CONVERGENCE86.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE86_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE86_RESULTS.md')
git('add','research/restore86','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore86/pulse74.patch',':(exclude)research/restore86/logs',':(exclude)research/restore86/THEORY_REGISTERED86.md')
git('commit','-m','RESTORE86: exact projected memory and compensated scale law')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE86.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE86.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE86_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST86.json'),source_zip_sha256=sha(S/'MQFT_RESTORE86_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE86.git.bundle'),progress_sha256=sha(S/'PROGRESS86.json'),fresh_paths=1152,fresh_backups=147456,figures=3,cumulative_retained_pairs=665,original_union=121,accepted_corrected_subclaims=16,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
