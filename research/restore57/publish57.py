from pathlib import Path
import json,csv,hashlib,zipfile
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore56_20261126';F=B/'mqft_restore53_20261126';A=B/'mqft_restore54_20261126'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
r=load(S/'PHYSICS_RESULTS.json');m=load(S/'plots/manifest.json');progress=load(S/'PROGRESS57.json')
cs=[v for q in r['counts'].values() for v in q.values()];total=lambda k:sum(v.get(k,0) for v in cs)
assert total('jobs')==288 and total('searches')==288 and total('backups')==4608
assert r['fresh_histories']==288 and r['unique_seed_groups']==96
assert total('rng_root')==4608 and total('rng_samples')==19584 and total('runtime_retries')==3
assert len(r['controls'])==72 and len(r['mark_prediction'])==96
assert r['model_source_sha']==h(O/'PHYSICS_RESULTS.json')
assert len(r['games'])==44 and len(r['energy'])==533 and len(r['lawfit'])==44
assert sum(x['n'] for x in r['lawfit'])==17856 and all(x['status']=='finite' for x in r['lawfit'])
assert sum(len(x['rows']) for x in r['energy'])==3119
for n,v in r['reuse_source_sha'].items():assert h(B/n/'PHYSICS_RESULTS.json')==v
assert len(r['thermal'])==24 and len(r['engines'])==24 and len(r['rate'])==8
assert all(abs(x['deltaF'])>1e-10 and x['crooks_error']<1e-8 for x in r['thermal'])
assert sum(x[tag]['N'] for x in r['thermal'] for tag in ('forward','reverse'))==98304
assert all(x[tag]['mc']['path_residual']<1e-10 for x in r['thermal'] for tag in ('forward','reverse'))
for x in r['thermal']:
 for tag in ('forward','reverse'):
  for q in x[tag]['rows']:
   assert max(abs(q[k]) for k in ('first_residual','entropy_residual','free_residual'))<1e-8 and q['flux_sigma']>=-1e-10
assert r['gauge_error']<1e-10
assert all(x['Qh']>0 and x['W_in']<0 and x['sigma']>=0 and abs(x['W_in']+x['Qh']+x['Qc'])<1e-12 for x in r['engines'])
assert r['paths_sha']==h(S/'NONCYCLIC_PATHS.npz') and r['thermal_registration_sha']==h(S/'THERMAL_REGISTRATION.json')
with zipfile.ZipFile(S/'NONCYCLIC_PATHS.npz') as z:assert len(z.namelist())==192
assert len(m['groups'])==7 and m['source_sha256']==h(S/'PHYSICS_RESULTS.json') and m['paths_sha256']==r['paths_sha']
assert all(h(S/f)==v for a in m['groups'].values() for f,v in a['files'].items())
assert progress['figure_pairs']==452 and progress['distinct_png_sha']==452 and progress['current_pairs']==7
assert progress['inventory_sha256']==h(S/'PLOT_INVENTORY57.csv')
assert progress['original_evidence_sha256']==h(S/'ORIGINAL_OUTPUT_EVIDENCE57.csv')
assert progress['original_progress_sha256']==h(S/'ORIGINAL_OUTPUT_PROGRESS57.csv')
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
ev=list(csv.DictReader((S/'ORIGINAL_OUTPUT_EVIDENCE57.csv').open()));assert len(ev)==14 and len(set(x['id'] for x in ev))==14
for x in ev:
 assert x['primary_node']==g['original_target_owner'][x['id']] and x['legacy_status']==cat[x['id']]['status']
 assert all(x['id'] in m['groups'][k]['original_items'] for k in x['evidence_figures'].split(';'))
 if cat[x['id']]['archive']!='guide_only':assert x['source_sha256']==cat[x['id']]['member_sha256']
 else:assert x['source_sha256']=='bfac951744509e0398f58a801855acb355bf9d95754c0ccbe231baf570228db6'
e=list(csv.DictReader((S/'ORIGINAL_OUTPUT_PROGRESS57.csv').open()))
assert len(e)==121 and {x['id'] for x in e}==set(cat) and all(x['final_completion']=='NOT_CERTIFIED' for x in e)
d=load(P/'docs/DAG.json');old=load(O/'DAG.json');assert d==old
d.update(revision='RESTORE57_20261126',scope='Dedicated native original energy/temperature/entropy; frozen prospective mark control; declared noncyclic work/entropy and finite-contact heat-engine references',previous_entry=d['entry'],entry='REPLAN-06',supersedes_next=d['next_stage_id'],next_stage_id='MQFT-RESTORE58-CRITICAL-TRANSPORT-STRUCTURAL-OUTPUTS')
d['restore57']={'parent_dag_sha256':h(O/'DAG.json'),'status':'scoped_original_native_definition_and_conditional_thermodynamic_outputs','sizes':[13,21],'fresh_games':0,'fresh_marked_histories':288,'total_marked_histories':9080,'fresh_backups':4608,'reused_games_R55_R56':44,'reused_game_plies':533,'native_energy_cuts':3119,'native_selector_choices':17856,'new_core_geometries':2,'response_rows':72,'mark_control':'MSE improves; LR action-unit prospective variance gain not replicated, not admitted','conditional_noncyclic_protocols':24,'conditional_work_paths':98304,'conditional_cycle_cases':24,'original_evidence_items':14,'original_evidence_sha256':h(S/'ORIGINAL_OUTPUT_EVIDENCE57.csv'),'progress_sha256':h(S/'PROGRESS57.json'),'inventory_sha256':h(S/'PLOT_INVENTORY57.csv'),'original_progress_sha256':h(S/'ORIGINAL_OUTPUT_PROGRESS57.csv'),'group_figures':m['groups'],'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'paths_sha256':r['paths_sha'],'original_plot_tasks':121,'additional_plot_nodes':0,'prior_admissions':'preserved','critical_scaling':'unresolved','native_1PI':'unresolved','next_contract':'docs/NEXT_DYNAMICS.md','evidence':['docs/RESTORE57_THEORY.md','docs/RESTORE57_RESULTS.md','docs/ORIGINAL_OUTPUT_EVIDENCE57.csv','docs/PROGRESS57.json','docs/PLOT_INVENTORY57.csv','docs/ORIGINAL_OUTPUT_PROGRESS57.csv']}
assert d['nodes']==old['nodes'] and d['edges']==old['edges'] and len(d['nodes'])==len(d['edges'])==13
(P/'docs/DAG.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
v=load(P/'docs/PROVENANCE.json');v['restore57']={'parent_main':'4e1ec234265013507a1e5cd5dfca0f32e174114f','family':regs,'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'paths_sha256':r['paths_sha'],'thermal_registration_sha256':r['thermal_registration_sha'],'original_evidence_sha256':h(S/'ORIGINAL_OUTPUT_EVIDENCE57.csv'),'progress_sha256':h(S/'PROGRESS57.json'),'inventory_sha256':h(S/'PLOT_INVENTORY57.csv'),'original_progress_sha256':h(S/'ORIGINAL_OUTPUT_PROGRESS57.csv'),'reuse_source_sha':r['reuse_source_sha'],'group_map_sha256':h(S/'PLOT_GROUPS.json'),'original_identity_log_sha256':h(S/'logs/original-identities.log'),'old_production_runtime_unchanged':True}
(P/'docs/PROVENANCE.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
f=P/'README.md';x=f.read_text();needle='## RESTORE56 matched longitudinal games and conditional dynamics';assert needle in x
x=x.replace('project, RESTORE56 (grouped Layer A/Q physics; explicit evaluator identity).','project, RESTORE57 (grouped Layer A/Q physics; explicit evaluator identity).')
f.write_text(x.replace(needle,'## RESTORE57 dedicated original native and thermodynamic outputs\n\n288 independent native root histories; 44 verified games reused for original\nenergy/temperature and entropy; conditional noncyclic work and signed entropy,\nfinite-contact Otto and ideal Carnot references. Frozen mark LR variance gain\nnot replicated. See [theory](docs/RESTORE57_THEORY.md), [results](docs/RESTORE57_RESULTS.md),\n[reproduction](research/restore57/README.md), [progress](docs/PROGRESS57.json),\n[original item ledger](docs/ORIGINAL_OUTPUT_PROGRESS57.csv).\n\n'+needle))
print('Verified',total('jobs'),'receipts,',total('backups'),'backups, 533 reused game plies, 98304 conditional paths, 14 original scopes, 7 plots, 452-pair inventory, 13/121 map')
