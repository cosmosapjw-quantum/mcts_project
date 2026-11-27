from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore83_20261127';O82=B/'mqft_restore82_20261127';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='68c0249a9a5991731f8bec4ee70c98d16268f00a'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
a=read(S/'AUDIT84.json');q=read(S/'SUMMARY84.json');t=read(S/'TRANSFER84.json');l=read(S/'LANDSCAPE84.json');v=read(S/'INDEPENDENT84.json');u=read(S/'RNG_AUDIT84.json');c=read(S/'CONVERGENCE84.json');pm=read(S/'PLOT_MANIFEST84.json');reg=read(S/'REGISTRATION.json');mod=read(S/'MODEL84.json');build=read(S/'BUILD84.json');patch=read(S/'LANDSCAPE_PATCH84.json')
assert a['paths']==2304 and a['backups']==294912 and a['prior_paths']==4032 and a['prior_backups']==516096
assert v['status']=='PASS_SCOPED' and v['child_checks']==54264 and v['selected_predictions']==288 and v['raw_paths']==144 and v['terminal_controls']==3
assert q['passed']==3 and q['direct_evaluator_calls']==54720 and all(x['registered_pass'] for x in t['results'])
assert sha(S/'FRESH84.npz')==a['array_sha']==t['fresh_array_sha'] and sha(S/'LANDSCAPE84.json')==mod['landscape_sha']==t['landscape_sha'] and sha(S/'MODEL84.json')==t['model_sha']
assert mod['status']=='TRAIN_ONLY_FROZEN' and sha(O82/'POPULATION82.npz')==mod['development_sha']
assert c['accepted_subclaims']==10 and c['new_accepted_subclaims']==1 and c['final_original_item_certifications']==0 and u['streams']==1536 and u['independent_uniforms']==196608
assert all(sha(S/k)==x for k,x in pm['sources'].items())
src=read(S/'ORIGINAL_SOURCE84.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O82/'ORIGINAL_SOURCE82.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
r0=read(O82/'REGISTRATION.json');assert reg['weights']==r0['weights'] and reg['external']==r0['external'] and sha(S/'mqft_train')==reg['binary_sha']==r0['binary_sha'] and sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==r0['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha']
assert build['status']=='PASS_BUILD' and sha(S/'mqft_landscape84')==build['binary_sha'] and sha(S/'landscape84.patch')==patch['patch_sha']==build['patch_sha']
base=B/'mqft_restore74_20261126/pulse_src';assert all(sha(base/k)==h for k,h in patch['base_source_files'].items())
assert sha(S/'landscape_src/src/mqft_train.rs')==patch['modified_sha'] and sha(base/'src/mqft_train.rs')==patch['original_sha']
raw=a['raw_receipts'];prior=a['prior_raw_receipts'];assert len(raw)==2304 and len(prior)==4032 and len({z['path'] for z in raw+prior})==6336
assert all(sha(B/z['path'])==z['sha256'] for z in raw+prior)
assert all(sha(S/'plans'/f'{k}.json')==z['sha256'] for k,z in reg['plans'].items())
scopes={'L16':'Bounded pre-search joint law and effective action predictions from actual full child evaluator landscape; registered relative gate passes all three evaluators. Absolute/general physical action remains open.', 'Q32':'Root-conditioned statistical generating function improvement on 48 independent geometries; root-wise vs population mixture, no physical susceptibility.', 'G07':'Actual pre-search statistical source prediction, separate direct child evaluator cost, clipping and tail diagnostics; causal field/FDT separate.'}
ids={k for z in pm['figures'] for k in z['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS83.json');inv=cr(O/'PLOT_INVENTORY83.csv');assert len(inv)==656 and sha(O/'PLOT_INVENTORY83.csv')==prev['inventory_sha256']
assert len(pm['figures'])==3 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==659;cw(S/'PLOT_INVENTORY84.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS83.csv');ev=[]
for x in rows:
 k=x['id'];x['R84_evidence']=scopes.get(k,'');x['R84_scope_status']='CORRECTED_SUBCLAIM_EVIDENCE' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[p['figure'] for p in pm['figures'] if k in p['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R84_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE84.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS84.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{n}_evidence') for n in range(55,85))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX83.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==84 and k in z['items']]
 x['R84_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R84_open_subclaims']=';'.join(z['id'] for z in c['claims'] if not z['status'].startswith('ACCEPTED') and k in z['items']) if k in scopes else ''
 x['R84_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='84';x['latest_scope']=scopes[k];x['corrected_output_state']='CORRECTED_SUBCLAIM_EVIDENCE_R84'
cw(S/'QUALITY_MATRIX84.csv',qm)
dump(S/'ITEM_CONVERGENCE84.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX83.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE83.json'),items=[dict(id=x['id'],accepted=x['R84_convergence_subclaims'],open=x['R84_open_subclaims'],scope=x['R84_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R84_scope']],claims=[z for z in c['claims'] if z['stage']==84],scope='R84 delta; R79 original item contracts and R80-R83 dispositions retained.'))
prev.update(figure_pairs=659,distinct_png_sha=659,files_paired=1318,current_pairs=3,prior_pairs=656,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY84.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE84.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS84.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX84.csv'),corrected_report_artifact_accepted=['L31'],accepted_corrected_subclaims=10,final_scientific_certified=0)
prev['stages'][S.name]=3;prev['original_output_evidence']=dict(current=len(ids),R55_R84_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Complete scoped coverage unchanged; 10 corrected subclaims accepted (+1), L31 artifact retained, final science separate.')
dump(S/'PROGRESS84.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('REPLAN-03','X-MEMORY'):node['restore84_followup']=dict(evidence='2304 fresh native paths; 104 development and 48 independent geometry orbits; 54720 direct root/child evaluator calls; frozen strongest root-only vs landscape ridge, relative ES/CGF criterion 3/3. Independent child, native mark, RNG and selected predictive law checks.',accepted_subclaims=['STAT-PRESEARCH-07'],remaining='Broad arbitrary-population transfer not admitted; finite absolute W, coverage, clipping, extra cost; no repeat feature grid absent new question.',artifact=S.name+'/{CONVERGENCE84.json,LANDSCAPE84.json,TRANSFER84.json,INDEPENDENT84.json,QUALITY_MATRIX84.csv}')
dag.update(revision='RESTORE84_20261127',scope='Convergence: bounded independent pre-search landscape transfer',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE85-THERMAL-REVERSE-PATH-IDENTIFICATION',mainline_priorities=['R85 explicit thermal and reverse path-law identification with support','R86 autonomous memory/scale or controlled critical limit','Actual operational coherent records only with independent readout; remaining item-specific acceptance'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX84.csv',convergence_ledger='CONVERGENCE84.json')
files=['REGISTRATION.json','ROOT_PREPARATION.json','LANDSCAPE_ROOTS84.json','LANDSCAPE84.json','LANDSCAPE_PATCH84.json','BUILD84.json','TERMINAL_TEST84.json','RUN_SHARDS.json','RAPFI_RECEIPTS84.json','AUDIT84.json','MODEL84.json','TRANSFER84.json','SUMMARY84.json','INDEPENDENT84.json','RNG_AUDIT84.json','ORIGINAL_SOURCE84.json','CONVERGENCE84.json','ITEM_CONVERGENCE84.json','QUALITY_MATRIX84.csv','PLOT_MANIFEST84.json','PLOT_INVENTORY84.csv','ORIGINAL_OUTPUT_EVIDENCE84.csv','ORIGINAL_OUTPUT_PROGRESS84.csv','PROGRESS84.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='bounded_presearch_population_transfer_accepted',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=2304,cumulative_fixed_root_histories=91384,fresh_backups=294912,reused_paths=4032,reused_backups=516096,direct_evaluator_calls=54720,development_roots=104,test_roots=48,registered_family_passes=3,raw_bytes=a['raw_bytes'],new_games=0,optimizer_updates=0,figures=3,animations=0,retained_pairs=659,original_evidence_items=len(ids),original_evidence_union_R55_R84=121,original_obligations=121,corrected_report_artifact_accepted=1,accepted_corrected_subclaims=10,new_accepted_subclaims=1,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Broad population generalization not admitted; physical native record and utility, thermal/reverse/scale/memory and item acceptance.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore84']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore84']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights'],pulse_binary_sha=reg['pulse_binary_sha'],landscape_binary_sha=build['binary_sha']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit_native74.py','audit_rapfi74.py','pulse74.patch','landscape84.patch','build84.py','plan84.py','landscape84.py','generate84.py','extract84.py','population84.py','verify84.py','summary84.py','plot84.py','publish84.py','backup84.py','execute84.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=['logs/build84.log','logs/landscape84.log','logs/fit84.log','logs/generate84.log','logs/extract84.log','logs/evaluate84_first_failure.log','logs/evaluate84.log','logs/verify84.log']
inputs={str((O/z).relative_to(B)):sha(O/z) for z in ['CONVERGENCE83.json','QUALITY_MATRIX83.csv','ORIGINAL_SOURCE83.json']}
for z in ['REGISTRATION.json','AUDIT82.json','POPULATION82.npz','ORIGINAL_SOURCE82.json']:inputs[str((O82/z).relative_to(B))]=sha(O82/z)
for k,h in patch['base_source_files'].items():inputs[str((base/k).relative_to(B))]=h
rngp=B/'mqft_restore61_20261126/mqft_trained_rng61';assert sha(rngp)==u['binary_sha'];inputs[str(rngp.relative_to(B))]=u['binary_sha']
extra=['data/terminal_check'+str(g)+'.json' for g in range(3)]
manifest=dict(schema='mqft_restore84_convergence_run_v1',parent=parent,status='completed_scoped',command='build84.py, plan84.py, landscape84.py, population84.py fit (frozen before generate84.py), bash execute84.sh on registered raw',files={z:sha(S/z) for z in files+sources+plotfiles+logs+extra+['DAG.json','PROVENANCE.json','mqft_train','mqft_pulse74','mqft_landscape84','FRESH84.npz','RNG_INPUT84.json']},raw_receipts=raw,prior_raw_receipts=prior,diagnostic_receipts=[],weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation, no full legacy suite; bounded extra-cost pre-search prediction, not exact fine history or universal population law.')
dump(S/'RUN_MANIFEST84.json',manifest)
dst=P/'research/restore84';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+extra+['DAG.json','PROVENANCE.json','RUN_MANIFEST84.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
shutil.copytree(S/'plans',dst/'plans')
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS84.csv','ORIGINAL_OUTPUT_EVIDENCE84.csv','PLOT_INVENTORY84.csv','PROGRESS84.json','QUALITY_MATRIX84.csv','CONVERGENCE84.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE84_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE84_RESULTS.md')
git('add','research/restore84','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore84/pulse74.patch',':(exclude)research/restore84/landscape84.patch',':(exclude)research/restore84/logs')
git('commit','-m','RESTORE84: independent pre-search child landscape population transfer')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE84.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE84.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE84_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST84.json'),source_zip_sha256=sha(S/'MQFT_RESTORE84_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE84.git.bundle'),progress_sha256=sha(S/'PROGRESS84.json'),fresh_paths=2304,fresh_backups=294912,direct_evaluator_calls=54720,figures=3,cumulative_retained_pairs=659,original_union=121,accepted_corrected_subclaims=10,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
