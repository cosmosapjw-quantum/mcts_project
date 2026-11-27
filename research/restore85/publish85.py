from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore84_20261127';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='99c96a8586a11621b11d0de6c237f2ecb0f1eebb'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
n=read(S/'NATIVE85.json');q=read(S/'SUMMARY85.json');t=read(S/'THERMAL85.json');br=read(S/'BRANCH_ENERGY85.json');v=read(S/'INDEPENDENT85.json');c=read(S/'CONVERGENCE85.json');pm=read(S/'PLOT_MANIFEST85.json');reg=read(S/'REGISTRATION.json')
assert n['paths']==2304 and n['backups']==294912 and n['fresh_native_paths']==0 and len(n['contexts'])==768
assert len(br['contexts'])==104 and len(br['raw_receipts'])==2448 and len(t['cases'])==832 and q['enumerated_triples']==114360704 and q['auxiliary_samples']==6553600
assert v['status']=='PASS_SCOPED' and v['wave_cases']==768 and v['thermal_cases']==832 and v['raw_representatives']==144 and v['stationary_witnesses']==768
assert sha(S/'THEORY_REGISTERED85.md')==reg['contract_sha'] and sha(S/'WAVE_SAMPLES85.npz')==n['samples_sha'] and sha(S/'THERMAL_SAMPLES85.npz')==t['raw_array_sha'] and sha(S/'BRANCH_ENERGY85.json')==t['energy_sha']
assert c['accepted_subclaims']==13 and c['new_accepted_subclaims']==3 and c['final_original_item_certifications']==0
assert all(sha(S/k)==x for k,x in pm['sources'].items())
src=read(S/'ORIGINAL_SOURCE85.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE84.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
r0=read(O/'REGISTRATION.json');assert reg['weights']==r0['weights'] and reg['external']==r0['external'] and reg['binary_sha']==r0['binary_sha'] and reg['pulse_binary_sha']==r0['pulse_binary_sha']
assert all(sha(B/x['path'])==x['sha256'] for x in reg['weights'].values())
raw=[dict(path=x['path'],sha256=x['sha256'],role='R74_authenticated_native_pulse') for x in n['raw_receipts']]+br['raw_receipts']
assert len(raw)==4752 and len({x['path'] for x in raw})==4752 and all(sha(B/x['path'])==x['sha256'] for x in raw)
scopes={'L08':'Finite branch-driven feedback information/work reference, explicit controller/reset omissions; no native physical energy.', 'L18':'Actual reverse ledger, support-aware RN IFT, finite LDB and first-law reference; physical reversal and bath remain open.', 'L32':'Actual branch-derived auxiliary work and information law with explicit reverse; no net device extraction.', 'Q19':'Exact support-aware IFT and forward/reverse samples, including perfect-detector singularity; not native physical Jarzynski.', 'Q20':'Finite work/heat distributions and weight-degeneracy diagnostics under declared branch-driven reference; physical heat uncalibrated.', 'Q21':'Exact path likelihood ratio and reverse endpoint-deletion construction, not physical quantum action.', 'Q23':'Corrected support-aware thermodynamic report; exact versus sampled failures and Gibbs marginal ambiguity retained.'}
ids={k for z in pm['figures'] for k in z['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS84.json');inv=cr(O/'PLOT_INVENTORY84.csv');assert len(inv)==659 and sha(O/'PLOT_INVENTORY84.csv')==prev['inventory_sha256']
assert len(pm['figures'])==3 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==662;cw(S/'PLOT_INVENTORY85.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS84.csv');ev=[]
for x in rows:
 k=x['id'];x['R85_evidence']=scopes.get(k,'');x['R85_scope_status']='CORRECTED_SUBCLAIM_EVIDENCE' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[p['figure'] for p in pm['figures'] if k in p['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R85_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE85.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS85.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{m}_evidence') for m in range(55,86))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX84.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==85 and k in z['items']]
 x['R85_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R85_open_subclaims']=';'.join(z['id'] for z in linked if not z['status'].startswith('ACCEPTED'))
 x['R85_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='85';x['latest_scope']=scopes[k];x['corrected_output_state']='CORRECTED_SUBCLAIM_EVIDENCE_R85'
cw(S/'QUALITY_MATRIX85.csv',qm)
dump(S/'ITEM_CONVERGENCE85.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX84.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE84.json'),items=[dict(id=x['id'],accepted=x['R85_convergence_subclaims'],open=x['R85_open_subclaims'],scope=x['R85_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R85_scope']],claims=[z for z in c['claims'] if z['stage']==85],scope='R85 delta, original R79 item contracts and prior dispositions retained.'))
prev.update(figure_pairs=662,distinct_png_sha=662,files_paired=1324,current_pairs=3,prior_pairs=659,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY85.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE85.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS85.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX85.csv'),corrected_report_artifact_accepted=['L31'],accepted_corrected_subclaims=13,final_scientific_certified=0)
prev['stages'][S.name]=3;prev['original_output_evidence']=dict(current=len(ids),R55_R85_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Complete scoped coverage unchanged; 13 corrected subclaims accepted (+3); L31 artifact retained, final science separate.')
dump(S/'PROGRESS85.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('REPLAN-06','REPLAN-02'):node['restore85_followup']=dict(evidence='2304 authenticated native pulse histories reversed as ordered ledgers, 768 exact conditional wave support identities, 104 actual branch-derived energy settings from2448 records,832 feedback cycles and114360704 exact triples,6553600 auxiliary F/R paths,768 stationary Gibbs marginal ambiguity witnesses.',accepted_subclaims=['THERMO-LEDGER-01','THERMO-FEEDBACK-02','THERMO-ID-03'],remaining='THERMO-PHYSICAL-04 physical native energy/reservoir/time not established, no repeated auxiliary FT scan; record/control cost absent. Original final item acceptance separate.',artifact=S.name+'/{CONVERGENCE85.json,NATIVE85.json,THERMAL85.json,STATIONARY85.json,INDEPENDENT85.json}')
dag.update(revision='RESTORE85_20261127',scope='Convergence: reverse ledger and support-aware finite feedback thermodynamics',previous_entry=dag['entry'],entry='REPLAN-06',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE86-HISTORY-CLOSURE-MEMORY-SCALE',mainline_priorities=['R86 explicit history-conditioned dynamics, memory kernel, autonomous closure and controlled scale law','R87 controlled critical scaling or structural completion if evidence warrants','Physical thermal or coherent identification only with independently defined calibration/readout; remaining original item acceptance'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX85.csv',convergence_ledger='CONVERGENCE85.json')
files=['REGISTRATION.json','THEORY_REGISTERED85.md','NATIVE85.json','BRANCH_ENERGY85.json','THERMAL85.json','STATIONARY85.json','SUMMARY85.json','INDEPENDENT85.json','ORIGINAL_SOURCE85.json','CONVERGENCE85.json','ITEM_CONVERGENCE85.json','QUALITY_MATRIX85.csv','PLOT_MANIFEST85.json','PLOT_INVENTORY85.csv','ORIGINAL_OUTPUT_EVIDENCE85.csv','ORIGINAL_OUTPUT_PROGRESS85.csv','PROGRESS85.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='three_new_corrected_thermal_subclaims_accepted_physical_identification_open',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=91384,fresh_backups=0,reused_native_paths=2304,reused_native_backups=294912,authenticated_branch_source_paths=2448,branch_contexts=104,conditional_wave_contexts=768,thermal_cases=832,enumerated_triples=114360704,auxiliary_sequences=6553600,new_games=0,optimizer_updates=0,figures=3,animations=0,retained_pairs=662,original_evidence_items=len(ids),original_evidence_union_R55_R85=121,original_obligations=121,corrected_report_artifact_accepted=1,accepted_corrected_subclaims=13,new_accepted_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Physical thermal identification, autonomous memory/scale, broad population transfer, native coherent readout and utility; original item acceptance.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore85']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore85']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','native85.py','branches85.py','thermal85.py','verify85.py','summary85.py','plot85.py','publish85.py','backup85.py','execute85.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=['logs/native85.log','logs/branches85.log','logs/thermal85.log','logs/verify85.log','logs/plot85.log']
inputs={str((O/z).relative_to(B)):sha(O/z) for z in ['CONVERGENCE84.json','QUALITY_MATRIX84.csv','ORIGINAL_SOURCE84.json','REGISTRATION.json']}
for st,day,fn in [(66,26,['THEORY.md','FEEDBACK_CONTRACT66.md','THERMAL66.json','FEEDBACK66.json']),(73,26,['RUN_MANIFEST73.json','INSTRUMENT73.json','BRANCH_REGISTRATION73.json']),(74,26,['REGISTRATION.json','AUDIT74.json','RUN_MANIFEST74.json','THEORY.md','PETZ74.json']),(83,27,['RUN_MANIFEST83.json','TREE83.json'])]:
 r=B/f'mqft_restore{st}_202611{day}'
 for z in fn:inputs[str((r/z).relative_to(B))]=sha(r/z)
for name in ('mqft_train','mqft_pulse74','pulse74.patch'):
 r=B/'mqft_restore74_20261126'/name;inputs[str(r.relative_to(B))]=sha(r)
manifest=dict(schema='mqft_restore85_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute85.sh (authenticated reused paths, declared auxiliary sampling)',files={z:sha(S/z) for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','WAVE_SAMPLES85.npz','THERMAL_SAMPLES85.npz']},raw_receipts=[],prior_raw_receipts=raw,diagnostic_receipts=[],weights=reg['weights'],external=reg['external'],plan_hashes={},inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent independent algorithm, no native subagent; no physical native reservoir/reverse or net device extraction. No full legacy regression on unchanged native backend.')
dump(S/'RUN_MANIFEST85.json',manifest)
dst=P/'research/restore85';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST85.json','WAVE_SAMPLES85.npz','THERMAL_SAMPLES85.npz']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS85.csv','ORIGINAL_OUTPUT_EVIDENCE85.csv','PLOT_INVENTORY85.csv','PROGRESS85.json','QUALITY_MATRIX85.csv','CONVERGENCE85.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE85_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE85_RESULTS.md')
git('add','research/restore85','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore85/logs',':(exclude)research/restore85/THEORY_REGISTERED85.md')
git('commit','-m','RESTORE85: reverse ledger and support-aware branch feedback thermodynamics')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE85.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE85.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE85_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST85.json'),source_zip_sha256=sha(S/'MQFT_RESTORE85_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE85.git.bundle'),progress_sha256=sha(S/'PROGRESS85.json'),fresh_paths=0,fresh_backups=0,auxiliary_sequences=6553600,figures=3,cumulative_retained_pairs=662,original_union=121,accepted_corrected_subclaims=13,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
