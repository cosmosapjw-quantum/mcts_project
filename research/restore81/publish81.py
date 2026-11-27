from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore80_20261127';A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='9168e93ef9669ea6ea8c05d63a20c749b419f466'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
a=read(S/'AUDIT81.json');r=read(S/'RECORD81.json');e=read(S/'EFFECTS81.json');q=read(S/'SUMMARY81.json');v=read(S/'INDEPENDENT81.json');u=read(S/'RNG_AUDIT81.json');c=read(S/'CONVERGENCE81.json');pm=read(S/'PLOT_MANIFEST81.json');reg=read(S/'REGISTRATION.json');build=read(S/'BUILD81.json');rg=read(S/'REGRESSION81.json')
assert a['paths']==3456 and a['backups']==442368 and len(e['matched_pairs'])==576 and len(r['contexts'])==72 and q['erased_exact_cases']==576
assert sha(S/'PATHS81.npz')==a['array_sha']==r['input_sha']==e['input_sha'] and v['status']=='PASS_SCOPED' and v['raw_paths']==432 and v['branch_checks']==2592
assert u['streams']==384 and u['independent_uniforms']==49152 and u['max_error']<1e-12
assert rg['passed']==394 and rg['ignored']==14 and rg['failed']==0 and sha(S/'logs/lib_regression81.log')==rg['log_sha256']
assert c['accepted_subclaims']==5 and c['new_accepted_subclaims']==2 and c['final_original_item_certifications']==0
assert all(sha(S/k)==x for k,x in pm['sources'].items())
src=read(S/'ORIGINAL_SOURCE81.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE80.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
r0=read(O/'REGISTRATION.json');assert reg['weights']==r0['weights'] and reg['external']==r0['external'] and sha(S/'mqft_train')==reg['binary_sha']==r0['binary_sha'] and sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==r0['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha']
assert sha(S/'mqft_phase81')==reg['phase_binary_sha']==build['binary_sha'] and sha(S/'phase81.patch')==reg['phase_patch_sha']==build['patch_sha']
patch=read(S/'PHASE_PATCH81.json');base=B/'mqft_restore74_20261126/pulse_src'
assert all(sha(base/k)==h for k,h in patch['base_source_files'].items())
assert all(sha(S/'phase_src/src'/k)==x['modified_sha'] and sha(base/'src'/k)==x['original_sha'] for k,x in patch['patches'].items())
raw=a['raw_receipts'];assert len(raw)==len({x['path'] for x in raw})==3456 and all(sha(B/x['path'])==x['sha256'] for x in raw)
assert all(sha(S/'plans'/f'{k}.json')==z['sha256'] for k,z in reg['plans'].items())
scopes={'L09':'Actual conditional complex register entropy/coherence and phase erasure/revival across three noncommuting wave settings; not physical decay constant.', 'L15':'Declared frozen three-qutrit record state with causal joint law and PPT first/rest; no adaptive objectivity or redundancy established.', 'L16':'Actual normalized adaptive CQ path likelihood and same-history intervention ratios; physical complex path amplitude remains open.', 'L20':'Actual opt-in selective discrete CP instrument, full state and law replay; not continuous physical GKSL identification.', 'Q24':'Actual phase-sensitive three-wave software channel and independent matrix/Bloch checks; physical bath not identified.', 'Q25':'Frozen three-record QMI and joint probability, not independent native fragment redundancy.', 'Q35':'Explicit frozen coherent density and partial transpose, no detected first/rest negativity, not separability.', 'Q37':'Actual conditional register dynamics plus separate frozen record information and causal marginals; no physical native quantum information identification.', 'Q42':'Specified discrete CP dynamics and state-independent dephasing, no physical time or hbar.', 'Q43':'Actual causal marked history probability ratio under phase intervention, not physical complex action.', 'Q45':'Scoped operational phase result, frozen model and open physical/utility/record requirements, not comprehensive original science completion.'}
ids={k for z in pm['figures'] for k in z['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS80.json');inv=cr(O/'PLOT_INVENTORY80.csv');assert len(inv)==647 and sha(O/'PLOT_INVENTORY80.csv')==prev['inventory_sha256']
assert len(pm['figures'])==3 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 base0=S/'plots'/x['figure'];png=Path(str(base0)+'.png');pdf=Path(str(base0)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==650;cw(S/'PLOT_INVENTORY81.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS80.csv');ev=[]
for x in rows:
 k=x['id'];x['R81_evidence']=scopes.get(k,'');x['R81_scope_status']='CORRECTED_SUBCLAIM_OR_FROZEN_MODEL_EVIDENCE' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[p['figure'] for p in pm['figures'] if k in p['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R81_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE81.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS81.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{n}_evidence') for n in range(55,82))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX80.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==81 and k in z['items']]
 x['R81_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R81_open_subclaims']=';'.join(z['id'] for z in linked if not z['status'].startswith('ACCEPTED'))
 x['R81_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='81';x['latest_scope']=scopes[k];x['corrected_output_state']='CORRECTED_SUBCLAIM_OR_MODEL_EVIDENCE_R81'
cw(S/'QUALITY_MATRIX81.csv',qm)
dump(S/'ITEM_CONVERGENCE81.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX80.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE80.json'),items=[dict(id=x['id'],accepted=x['R81_convergence_subclaims'],open=x['R81_open_subclaims'],scope=x['R81_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R81_scope']],claims=[z for z in c['claims'] if z['stage']==81],scope='R81 delta; R79 original item contracts and R80 dispositions retained.'))
prev.update(figure_pairs=650,distinct_png_sha=650,files_paired=1300,current_pairs=3,prior_pairs=647,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY81.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE81.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS81.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX81.csv'),corrected_report_artifact_accepted=['L31'],accepted_corrected_subclaims=5,final_scientific_certified=0)
prev['stages'][S.name]=3;prev['original_output_evidence']=dict(current=len(ids),R55_R81_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Complete scoped coverage unchanged; 5 corrected subclaims accepted (+2), L31 artifact retained, final original certification separate.')
dump(S/'PROGRESS81.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('X-REP','REPLAN-05'):node['restore81_followup']=dict(evidence='3456 actual matched phase/erasure/order/base paths, 576 exact common prefixes and erasure controls, 2592 independent CP branch products, explicit 72 frozen three-time record models, no detected first/rest negativity.',accepted_subclaims=['Q-OP-01','Q-CAUSAL-02'],remaining='Native physical phase, actual adaptive coherent record/fragment objectivity, and matched-cost utility open; frozen record not native identification.',artifact=S.name+'/{CONVERGENCE81.json,EFFECTS81.json,RECORD81.json,INDEPENDENT81.json}')
dag.update(revision='RESTORE81_20261127',scope='Convergence: operational software phase instrument and normalized causal CQ history law',previous_entry=dag['entry'],entry='X-REP',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE82-ROOT-CONDITIONED-POPULATION-ACTION',mainline_priorities=['R82 root-conditioned population effective action and independent geometry transfer','R83 actual adaptive coherent records only if explicitly defined branch-dependent process','Remaining physical calibration, scale/memory, structural and application item-specific conditions'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX81.csv',convergence_ledger='CONVERGENCE81.json')
files=['REGISTRATION.json','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS81.json','BUILD81.json','PHASE_PATCH81.json','AUDIT81.json','EFFECTS81.json','RECORD81.json','SUMMARY81.json','INDEPENDENT81.json','RNG_AUDIT81.json','REGRESSION81.json','ORIGINAL_SOURCE81.json','CONVERGENCE81.json','ITEM_CONVERGENCE81.json','QUALITY_MATRIX81.csv','PLOT_MANIFEST81.json','PLOT_INVENTORY81.csv','ORIGINAL_OUTPUT_EVIDENCE81.csv','ORIGINAL_OUTPUT_PROGRESS81.csv','PROGRESS81.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='two_new_corrected_scientific_subclaims_accepted',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=3456,cumulative_fixed_root_histories=87544,fresh_backups=442368,matched_blocks=576,branch_checks=2592,frozen_contexts=72,raw_bytes=a['bytes'],new_games=0,optimizer_updates=0,figures=3,animations=0,retained_pairs=650,original_evidence_items=len(ids),original_evidence_union_R55_R81=121,original_obligations=121,corrected_report_artifact_accepted=1,accepted_corrected_subclaims=5,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Root-conditioned population transfer, actual adaptive coherent records, native physical phase and utility; final original acceptance.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore81']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore81']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights'],phase_binary_sha=reg['phase_binary_sha']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','pulse74.patch','phase81.patch','plan81.py','generate81.py','build81.py','instrument81.py','audit81.py','effects81.py','record81.py','verify81.py','summary81.py','plot81.py','publish81.py','backup81.py','execute81.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=['logs/build81.log','logs/test81.log','logs/lib_regression81.log','logs/lib_regression81.exit','logs/generate81.log','logs/audit81_first_failure.log','logs/audit81.log','logs/effects81.log','logs/record81.log','logs/verify81.log']
inputs={str((O/z).relative_to(B)):sha(O/z) for z in ['REGISTRATION.json','CONVERGENCE80.json','QUALITY_MATRIX80.csv','ORIGINAL_SOURCE80.json']}
for k,h in patch['base_source_files'].items():inputs[str((base/k).relative_to(B))]=h
rngp=B/'mqft_restore61_20261126/mqft_trained_rng61';assert sha(rngp)==u['binary_sha'];inputs[str(rngp.relative_to(B))]=u['binary_sha']
manifest=dict(schema='mqft_restore81_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute81.sh (existing raw traces and lib receipt); build81.py, plan81.py, generate81.py for acquisition',files={z:sha(S/z) for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','mqft_train','mqft_pulse74','mqft_phase81','PATHS81.npz','RNG_INPUT81.json']},raw_receipts=raw,prior_raw_receipts=[],weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation, isolated full lib suite only; declared software intervention and frozen record model not native physical quantum process.')
dump(S/'RUN_MANIFEST81.json',manifest)
dst=P/'research/restore81';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST81.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
shutil.copytree(S/'plans',dst/'plans')
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS81.csv','ORIGINAL_OUTPUT_EVIDENCE81.csv','PLOT_INVENTORY81.csv','PROGRESS81.json','QUALITY_MATRIX81.csv','CONVERGENCE81.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE81_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE81_RESULTS.md')
git('add','research/restore81','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore81/pulse74.patch',':(exclude)research/restore81/phase81.patch',':(exclude)research/restore81/logs')
git('commit','-m','RESTORE81: operational phase instrument and causal CQ path law')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE81.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE81.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE81_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST81.json'),source_zip_sha256=sha(S/'MQFT_RESTORE81_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE81.git.bundle'),progress_sha256=sha(S/'PROGRESS81.json'),fresh_paths=3456,fresh_backups=442368,figures=3,cumulative_retained_pairs=650,original_union=121,accepted_corrected_subclaims=5,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
