from pathlib import Path
import json,csv,hashlib,zipfile
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore55_20261126';F=B/'mqft_restore53_20261126';A=B/'mqft_restore54_20261126'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
r=load(S/'PHYSICS_RESULTS.json');m=load(S/'plots/manifest.json');progress=load(S/'PROGRESS56.json')
cs=[v for q in r['counts'].values() for v in q.values()];total=lambda k:sum(v.get(k,0) for v in cs)
assert total('jobs')==180 and total('searches')==528 and total('backups')==16608
assert r['fresh_histories']==144 and r['unique_seed_groups']==48 and len(r['games'])==36
assert all(x['status']=='natural_terminal' for x in r['games']) and len(r['longitudinal'])==384
assert total('rng_root')==16608 and total('rng_samples')==66180 and total('runtime_retries')==0
assert len(r['control'])==72 and len(r['mark_prediction'])==48 and len(r['control_models'])==6 and len(r['loops'])==12
assert sum(len(x['values']) for x in r['loops'])==36 and all(v['one_loop_error']<v['leading_error'] for x in r['loops'] for v in x['values'])
assert len(r['combined_game_summary'])==44 and sum(x['searched'] for x in r['combined_game_summary'])==533
assert len(r['quantum'])==108 and len(r['thermal'])==36 and len(r['q19'])==12
assert max(x['quadrature_error'] for y in r['thermal'] for x in y['rows'])<1e-8
assert max(x['finite_error'] for y in r['thermal'] for x in y['rows'])<1e-8
assert r['q19_source']['sha']==h(A/'WORK_SAMPLES.npz')
assert r['states_sha']==h(S/'CONDITIONAL_STATES.npz')
with zipfile.ZipFile(S/'CONDITIONAL_STATES.npz') as z:assert len(z.namelist())==972
assert len(m['groups'])==10 and m['source_sha256']==h(S/'PHYSICS_RESULTS.json') and m['states_sha256']==r['states_sha']
assert all(h(S/f)==v for a in m['groups'].values() for f,v in a['files'].items())
assert progress['figure_pairs']==445 and progress['distinct_png_sha']==445 and progress['current_pairs']==10
assert progress['inventory_sha256']==h(S/'PLOT_INVENTORY56.csv')
assert progress['original_evidence_sha256']==h(S/'ORIGINAL_OUTPUT_EVIDENCE56.csv')
assert progress['original_progress_sha256']==h(S/'ORIGINAL_OUTPUT_PROGRESS56.csv')
regs={}
for L in (13,21):
 reg=load(S/f'REGISTRATION_L{L}.json');old=load(O/f'REGISTRATION_L{L}.json');q=F/'family'/f'L{L}';exe=F/'runtime'/f'mqft-L{L}'
 assert h(exe)==reg['backend_sha']==old['backend_sha'] and reg['model']==old['model']
 assert h(q/'SOURCE_MANIFEST.json')==reg['source_manifest_sha']==old['source_manifest_sha']
 sm=load(q/'SOURCE_MANIFEST.json');assert all(h(q/f)==v for f,v in sm['files'].items())
 patch=A/'python'/f'L{L}';pm=load(patch/'PATCH_MANIFEST.json')
 assert h(patch/'PATCH_MANIFEST.json')==reg['analysis_patch_manifest_sha'] and pm['parent']==h(q/'python/mqft/analysis.py') and pm['modified']==h(patch/'mqft/analysis.py') and all(h(patch/f)==v for f,v in pm['files'].items())
 x=load(S/f'PHYSICS_L{L}.json');assert h(S/f'PHYSICS_L{L}.json')==r['size_results_sha'][f'L{L}'] and x['registration_sha']==h(S/f'REGISTRATION_L{L}.json')
 for n,c in x['counts'].items():
  stem=f'{n}-L{L}';a=S/'analysis'/stem;d=S/'data'/stem
  mm=load(a/'manifest.json');rp=load(next(d.glob('report-*.json')));pl=load(S/'plans'/(stem+'.json'));lock=load(d/'run_lock.json')
  assert lock['backend_sha']==reg['backend_sha'] and lock['external']==reg['model']
  assert lock['plan_hash']==mm['plan_hash']==pl['plan_hash']==reg['plans'][n]['hash']
  assert mm['completed']==mm['expected']==len(pl['jobs'])==len(rp['completed'])==c['jobs'] and not rp['failures'] and not rp['unattempted']
  assert all(h(a/f)==v for f,v in mm['outputs'].items())
  for t in rp['completed']:
   f=d/t['file'];assert h(f)==t['sha256'] and f.stat().st_size==t['bytes'] and load(d/(t['job_id']+'.receipt.json'))==t
 regs[str(L)]={'registration_sha256':h(S/f'REGISTRATION_L{L}.json'),'backend_sha256':h(exe),'source_manifest_sha256':reg['source_manifest_sha'],'analysis_patch_manifest_sha256':reg['analysis_patch_manifest_sha'],'plans':reg['plans']}
for n in ('CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv'):
 assert h(S/n)==h(O/n)==h(P/'docs'/n)
g=load(S/'PLOT_GROUPS.json');cat={x['id']:x for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())}
assert len(g['groups'])==13 and len(g['original_target_owner'])==121 and set(g['original_target_owner'])==set(cat)
ev=list(csv.DictReader((S/'ORIGINAL_OUTPUT_EVIDENCE56.csv').open()));assert len(ev)==23 and len(set(x['id'] for x in ev))==23
for x in ev:
 assert x['primary_node']==g['original_target_owner'][x['id']] and x['member_sha256']==cat[x['id']]['member_sha256'] and x['legacy_status']==cat[x['id']]['status']
 assert all(x['id'] in m['groups'][k]['original_items'] for k in x['evidence_nodes'].split(';'))
e=list(csv.DictReader((S/'ORIGINAL_OUTPUT_PROGRESS56.csv').open()))
assert len(e)==121 and {x['id'] for x in e}==set(cat) and all(x['final_completion']=='NOT_CERTIFIED' for x in e)
d=load(P/'docs/DAG.json');old=load(O/'DAG.json');assert d==old
d.update(revision='RESTORE56_20261126',scope='Grouped Layer A/Q: matched longitudinal games across budgets, independently trained predictable mark control, conditional erasure and driven response, original output progress ledger',previous_entry=d['entry'],entry='REPLAN-01',supersedes_next=d['next_stage_id'],next_stage_id='MQFT-RESTORE57-DEDICATED-ORIGINAL-OUTPUTS')
d['restore56']={'parent_dag_sha256':h(O/'DAG.json'),'status':'scoped_native_longitudinal_and_predictable_control_conditional_physics','sizes':[13,21],'fresh_games':36,'fresh_marked_histories':144,'total_marked_histories':8792,'fresh_backups':16608,'longitudinal_plies':384,'combined_games_R55_R56':44,'combined_game_plies':533,'new_core_geometries':6,'matched_blocks_B16_B32':8,'matched_blocks_B32_B64':6,'response_rows':72,'mark_control':'heldout predictor MSE improved; LR action-unit exploratory reduction, scaled/direct no general admission','restricted_lattice_cases':36,'conditional_record_cases':108,'conditional_states':972,'conditional_driven_cases':36,'original_evidence_items':23,'original_evidence_sha256':h(S/'ORIGINAL_OUTPUT_EVIDENCE56.csv'),'progress_sha256':h(S/'PROGRESS56.json'),'inventory_sha256':h(S/'PLOT_INVENTORY56.csv'),'original_progress_sha256':h(S/'ORIGINAL_OUTPUT_PROGRESS56.csv'),'group_figures':m['groups'],'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':r['states_sha'],'original_plot_tasks':121,'additional_plot_nodes':0,'prior_admissions':'preserved','critical_scaling':'unresolved','native_1PI':'unresolved','next_contract':'docs/NEXT_DYNAMICS.md','evidence':['docs/RESTORE56_THEORY.md','docs/RESTORE56_RESULTS.md','docs/ORIGINAL_OUTPUT_EVIDENCE56.csv','docs/PROGRESS56.json','docs/PLOT_INVENTORY56.csv','docs/ORIGINAL_OUTPUT_PROGRESS56.csv']}
assert d['nodes']==old['nodes'] and d['edges']==old['edges'] and len(d['nodes'])==len(d['edges'])==13
(P/'docs/DAG.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
v=load(P/'docs/PROVENANCE.json');v['restore56']={'parent_main':'3a024396037a9b75f8d09261fbf4793b55ca3523','family':regs,'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':r['states_sha'],'original_evidence_sha256':h(S/'ORIGINAL_OUTPUT_EVIDENCE56.csv'),'progress_sha256':h(S/'PROGRESS56.json'),'inventory_sha256':h(S/'PLOT_INVENTORY56.csv'),'original_progress_sha256':h(S/'ORIGINAL_OUTPUT_PROGRESS56.csv'),'q19_source':r['q19_source'],'group_map_sha256':h(S/'PLOT_GROUPS.json'),'original_identity_log_sha256':h(S/'logs/original-identities.log'),'old_production_runtime_unchanged':True}
(P/'docs/PROVENANCE.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
f=P/'README.md';x=f.read_text();needle='## RESTORE55 actual longitudinal and sequential-wave physics';assert needle in x
x=x.replace('project, RESTORE55 (grouped Layer A/Q physics; explicit evaluator identity).','project, RESTORE56 (grouped Layer A/Q physics; explicit evaluator identity).')
f.write_text(x.replace(needle,'## RESTORE56 matched longitudinal games and conditional dynamics\n\n36 new full games and 144 fixed-root histories; independent predictable mark\ncontrol, coherent erasure, driven response, conditional Q19 replot.\nSee [theory](docs/RESTORE56_THEORY.md), [results](docs/RESTORE56_RESULTS.md),\n[reproduction](research/restore56/README.md), [progress](docs/PROGRESS56.json),\n[original item ledger](docs/ORIGINAL_OUTPUT_PROGRESS56.csv).\n\n'+needle))
print('Verified',total('jobs'),'receipts,',total('backups'),'backups, 384 game plies, 972 states, 23 item scopes, 10 plots, 445-pair inventory, 13/121 owner map')
