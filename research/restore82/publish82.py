from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore81_20261127';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='207d54e87cf5818aa84a8ae3eb5388a3d02c9574'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
a=read(S/'AUDIT82.json');q=read(S/'SUMMARY82.json');r=read(S/'TRANSFER82.json');v=read(S/'INDEPENDENT82.json');u=read(S/'RNG_AUDIT82.json');c=read(S/'CONVERGENCE82.json');comp=read(S/'COMPAT82.json');mod=read(S/'MODEL82.json');pm=read(S/'PLOT_MANIFEST82.json');reg=read(S/'REGISTRATION.json')
assert a['paths']==1536 and a['prior_paths']==2496 and a['backups']==196608 and a['prior_backups']==319488
assert v['status']=='PASS_SCOPED' and v['independent_predictions']==1008 and v['aggregate_predictions']==12672 and v['tilted_mixture_cases']==261 and v['raw_paths']==312
assert comp['status']=='PASS_SCOPED' and len(comp['cases'])==12 and comp['diagnostic_backups']==1536
assert sha(S/'POPULATION82.npz')==a['array_sha']==r['array_sha']==mod['input_sha'] and sha(S/'PREDICTIONS82.npz')==r['prediction_sha'] and sha(S/'MODEL82.json')==r['model_sha']
assert u['streams']==1024 and u['independent_uniforms']==131072 and u['max_error']<1e-12
assert q['root_only_passes']==5 and q['prefix_passes']==6 and c['accepted_subclaims']==7 and c['new_accepted_subclaims']==2 and c['final_original_item_certifications']==0
assert all(sha(S/k)==x for k,x in pm['sources'].items())
src=read(S/'ORIGINAL_SOURCE82.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE81.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
r0=read(O/'REGISTRATION.json');assert reg['weights']==r0['weights'] and reg['external']==r0['external'] and sha(S/'mqft_train')==reg['binary_sha']==r0['binary_sha'] and sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==r0['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha']
raw=[x for x in a['raw_receipts'] if x['group']==2];priorraw=[x for x in a['raw_receipts'] if x['group']!=2];assert len(raw)==1536 and len(priorraw)==2496 and len({x['path'] for x in raw+priorraw})==4032
assert all(sha(B/x['path'])==x['sha256'] for x in raw+priorraw)
assert all(sha(B/x['path'])==x['sha256'] and sha(B/x['source'])==x['source_sha'] for x in comp['cases'])
assert all(sha(S/'plans'/f'{k}.json')==z['sha256'] for k,z in reg['plans'].items())
scopes={'L16':'Exact root-mixture CGF/Hessian and bounded observed-prefix predictive conditional action; root-only transfer and full physical complex action remain open.', 'Q32':'Exact tilted population covariance split into within-root and between-root terms; actual independent conditional prediction gains with H32, no physical thermal susceptibility.', 'G07':'Corrected root-conditioned statistical response and fluctuation distinction; observed-prefix joint prediction tested on two independent cohorts, native causal field/FDT separate.'}
ids={k for z in pm['figures'] for k in z['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS81.json');inv=cr(O/'PLOT_INVENTORY81.csv');assert len(inv)==650 and sha(O/'PLOT_INVENTORY81.csv')==prev['inventory_sha256']
assert len(pm['figures'])==3 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==653;cw(S/'PLOT_INVENTORY82.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS81.csv');ev=[]
for x in rows:
 k=x['id'];x['R82_evidence']=scopes.get(k,'');x['R82_scope_status']='CORRECTED_SUBCLAIM_EVIDENCE' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[p['figure'] for p in pm['figures'] if k in p['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R82_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE82.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS82.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{n}_evidence') for n in range(55,83))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX81.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==82 and k in z['items']]
 x['R82_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R82_open_subclaims']='STAT-TRANSFER-04' if k in scopes else ''
 x['R82_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='82';x['latest_scope']=scopes[k];x['corrected_output_state']='CORRECTED_SUBCLAIM_EVIDENCE_R82'
cw(S/'QUALITY_MATRIX82.csv',qm)
dump(S/'ITEM_CONVERGENCE82.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX81.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE81.json'),items=[dict(id=x['id'],accepted=x['R82_convergence_subclaims'],open=x['R82_open_subclaims'],scope=x['R82_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R82_scope']],claims=[z for z in c['claims'] if z['stage']==82],scope='R82 delta; R79 item contracts and R80/R81 accepted claims retained.'))
prev.update(figure_pairs=653,distinct_png_sha=653,files_paired=1306,current_pairs=3,prior_pairs=650,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY82.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE82.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS82.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX82.csv'),corrected_report_artifact_accepted=['L31'],accepted_corrected_subclaims=7,final_scientific_certified=0)
prev['stages'][S.name]=3;prev['original_output_evidence']=dict(current=len(ids),R55_R82_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Complete scoped coverage unchanged; 7 corrected subclaims accepted (+2), L31 artifact retained; final original scientific certification separate.')
dump(S/'PROGRESS82.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('REPLAN-03','X-MEMORY'):node['restore82_followup']=dict(evidence='1536 new clean histories, 2496 authenticated prior histories, 104 geometry orbits; exact root-mixture covariance identities, frozen six-model conditional forecast, H32 root+history improves in all six external family-cohort cells; root-only registered criterion passes 5/6, Rapfi R81 fails.',accepted_subclaims=['STAT-MIX-05','STAT-HISTORY-06'],remaining='STAT-TRANSFER-04 open, large absolute W errors and low root ESS; no autonomous closure or physical temperature, full original item certification separate.',artifact=S.name+'/{CONVERGENCE82.json,TRANSFER82.json,SUMMARY82.json,INDEPENDENT82.json,QUALITY_MATRIX82.csv}')
dag.update(revision='RESTORE82_20261127',scope='Convergence: root-conditional population mixture and bounded observed-prefix prediction',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE83-ADAPTIVE-COHERENT-RECORD-DYNAMICS',mainline_priorities=['R83 actual adaptive multi-time coherent record branch-dependent extension','R84 root-only transfer using pre-search child evaluator features if prioritized','Remaining physical calibration, scale/memory, structural and application item-specific conditions'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX82.csv',convergence_ledger='CONVERGENCE82.json')
files=['REGISTRATION.json','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS82.json','AUDIT82.json','MODEL82.json','TRANSFER82.json','SUMMARY82.json','INDEPENDENT82.json','RNG_AUDIT82.json','COMPAT82.json','ORIGINAL_SOURCE82.json','CONVERGENCE82.json','ITEM_CONVERGENCE82.json','QUALITY_MATRIX82.csv','PLOT_MANIFEST82.json','PLOT_INVENTORY82.csv','ORIGINAL_OUTPUT_EVIDENCE82.csv','ORIGINAL_OUTPUT_PROGRESS82.csv','PROGRESS82.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='two_new_corrected_scientific_subclaims_accepted_root_only_open',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=1536,cumulative_fixed_root_histories=89080,fresh_backups=196608,reused_paths=2496,reused_backups=319488,compatibility_replays=12,compatibility_backups=1536,root_counts=[48,24,32],root_only_cells_passed=5,history_cells_passed=6,raw_bytes=a['raw_bytes'],new_games=0,optimizer_updates=0,figures=3,animations=0,retained_pairs=653,original_evidence_items=len(ids),original_evidence_union_R55_R82=121,original_obligations=121,corrected_report_artifact_accepted=1,accepted_corrected_subclaims=7,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Root-only population transfer, adaptive coherent record extension, native physical phase and utility; original item acceptance.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore82']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore82']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights'],pulse_binary_sha=reg['pulse_binary_sha']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit_native74.py','audit_rapfi74.py','pulse74.patch','plan82.py','generate82.py','extract82.py','compat82.py','population82.py','verify82.py','summary82.py','plot82.py','publish82.py','backup82.py','execute82.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=['logs/generate82.log','logs/extract82.log','logs/fit82.log','logs/evaluate82.log','logs/compat82.log','logs/verify82.log','logs/summary82.log','logs/plot82.log']
inputs={str((O/z).relative_to(B)):sha(O/z) for z in ['REGISTRATION.json','AUDIT81.json','CONVERGENCE81.json','QUALITY_MATRIX81.csv','ORIGINAL_SOURCE81.json']}
for n in (79,80):
 d=B/f'mqft_restore{n}_20261127'
 for z in ['REGISTRATION.json',f'AUDIT{n}.json']:inputs[str((d/z).relative_to(B))]=sha(d/z)
for z in comp['cases']:inputs[z['path']]=z['sha256'];inputs[z['source']]=z['source_sha']
rngp=B/'mqft_restore61_20261126/mqft_trained_rng61';assert sha(rngp)==u['binary_sha'];inputs[str(rngp.relative_to(B))]=u['binary_sha']
manifest=dict(schema='mqft_restore82_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute82.sh (registered data and compatibility receipts); plan82.py, generate82.py, compat82.py for acquisition',files={z:sha(S/z) for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','mqft_train','mqft_pulse74','POPULATION82.npz','PREDICTIONS82.npz','RNG_INPUT82.json']},raw_receipts=raw,prior_raw_receipts=priorraw,compatibility_diagnostic_receipts=comp['cases'],weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation, no full legacy suite on unchanged backend. Bounded observed-prefix gain, root-only transfer not admitted, no autonomous closure/physical interpretation.')
dump(S/'RUN_MANIFEST82.json',manifest)
dst=P/'research/restore82';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST82.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
shutil.copytree(S/'plans',dst/'plans')
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS82.csv','ORIGINAL_OUTPUT_EVIDENCE82.csv','PLOT_INVENTORY82.csv','PROGRESS82.json','QUALITY_MATRIX82.csv','CONVERGENCE82.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE82_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE82_RESULTS.md')
git('add','research/restore82','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore82/pulse74.patch',':(exclude)research/restore82/logs')
git('commit','-m','RESTORE82: root-conditioned population action and independent prefix forecasts')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE82.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE82.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE82_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST82.json'),source_zip_sha256=sha(S/'MQFT_RESTORE82_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE82.git.bundle'),progress_sha256=sha(S/'PROGRESS82.json'),fresh_paths=1536,fresh_backups=196608,figures=3,cumulative_retained_pairs=653,original_union=121,accepted_corrected_subclaims=7,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
