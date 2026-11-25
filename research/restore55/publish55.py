from pathlib import Path
import json,csv,hashlib,zipfile
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore54_20261126';F=B/'mqft_restore53_20261126'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
r=load(S/'PHYSICS_RESULTS.json');m=load(S/'plots/grouped/manifest.json')
cs=[v for q in r['counts'].values() for v in q.values()];total=lambda k:sum(v.get(k,0) for v in cs)
assert total('jobs')==488 and total('searches')==629 and total('backups')==11232
assert r['fresh_histories']==480 and r['unique_seed_groups']==96 and len(r['games'])==8
assert all(x['status']=='natural_terminal' for x in r['games']) and len(r['longitudinal'])==149
assert total('rng_root')==11232 and total('rng_samples')==47052 and total('exact_prefix_pairs')==192 and total('exact_prefix_draws')==2304
assert len(r['control'])==60 and len(r['restricted'])==len(r['loops'])==12
assert max(x['derivative_error'] for x in r['restricted'])<1e-8
assert sum(len(x['values']) for x in r['loops'])==36 and all(v['one_loop_error']<v['leading_error'] for x in r['loops'] for v in x['values'])
assert len(r['quantum'])==72 and len(r['thermal'])==24
assert r['states_sha']==h(S/'CONDITIONAL_STATES.npz')
with zipfile.ZipFile(S/'CONDITIONAL_STATES.npz') as z:assert len(z.namelist())==216
assert len(m['groups'])==9 and m['source_sha256']==h(S/'PHYSICS_RESULTS.json') and m['states_sha256']==r['states_sha']
assert all(h(S/'plots/grouped'/f)==v for a in m['groups'].values() for f,v in a['files'].items())
regs={}
for L in (13,21):
 reg=load(S/f'REGISTRATION_L{L}.json');old=load(O/f'REGISTRATION_L{L}.json');q=F/'family'/f'L{L}';exe=F/'runtime'/f'mqft-L{L}'
 assert h(exe)==reg['backend_sha']==old['backend_sha'] and reg['model']==old['model']
 assert h(q/'SOURCE_MANIFEST.json')==reg['source_manifest_sha']==old['source_manifest_sha']
 sm=load(q/'SOURCE_MANIFEST.json');assert all(h(q/f)==v for f,v in sm['files'].items())
 patch=O/'python'/f'L{L}';pm=load(patch/'PATCH_MANIFEST.json')
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
ev=list(csv.DictReader((S/'ORIGINAL_OUTPUT_EVIDENCE55.csv').open()));assert len(ev)==20 and len(set(x['id'] for x in ev))==20
for x in ev:
 assert x['primary_node']==g['original_target_owner'][x['id']] and x['member_sha256']==cat[x['id']]['member_sha256'] and x['legacy_status']==cat[x['id']]['status']
 assert all(x['id'] in m['groups'][k]['original_items'] for k in x['evidence_nodes'].split(';'))
d=load(P/'docs/DAG.json');old=load(O/'DAG.json');assert d==old
d.update(revision='RESTORE55_20261126',scope='Grouped Layer A/Q: actual longitudinal Q fit and depth flow, sequential full-support martingale controls, restricted PL/reserve and declared quantum record/FDT models',previous_entry=d['entry'],entry='REPLAN-01',supersedes_next=d['next_stage_id'],next_stage_id='MQFT-RESTORE56-LONGITUDINAL-CONDITIONAL-DYNAMICS')
d['restore55']={'parent_dag_sha256':h(O/'DAG.json'),'status':'scoped_native_longitudinal_sequential_controls_and_conditional_physics','sizes':[13,21],'fresh_games':8,'fresh_marked_histories':480,'total_marked_histories':8648,'fresh_backups':11232,'longitudinal_plies':149,'new_core_geometries':2,'exact_prefix_pairs':192,'response_rows':60,'restricted_lattice_cases':36,'native_sector_control':'variance reduced scoped; future mark direct control NOT_ADMITTED','conditional_record_cases':72,'conditional_states':216,'conditional_equilibrium_cases':24,'original_evidence_items':20,'original_evidence_sha256':h(S/'ORIGINAL_OUTPUT_EVIDENCE55.csv'),'group_figures':m['groups'],'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':r['states_sha'],'original_plot_tasks':121,'additional_plot_nodes':0,'prior_admissions':'preserved','critical_scaling':'unresolved','native_1PI':'unresolved','next_contract':'docs/NEXT_DYNAMICS.md','evidence':['docs/RESTORE55_THEORY.md','docs/RESTORE55_RESULTS.md','docs/ORIGINAL_OUTPUT_EVIDENCE55.csv']}
assert d['nodes']==old['nodes'] and d['edges']==old['edges'] and len(d['nodes'])==len(d['edges'])==13
(P/'docs/DAG.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
v=load(P/'docs/PROVENANCE.json');v['restore55']={'parent_main':'8c0536e71dfd1d27b3ff87bf0be1022529cc5c8e','family':regs,'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':r['states_sha'],'original_evidence_sha256':h(S/'ORIGINAL_OUTPUT_EVIDENCE55.csv'),'group_map_sha256':h(S/'PLOT_GROUPS.json'),'original_identity_log_sha256':h(S/'logs/original-identities.log'),'old_production_runtime_unchanged':True}
(P/'docs/PROVENANCE.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
f=P/'README.md';x=f.read_text();needle='## RESTORE54 exact conditional IID wave and response precision';assert needle in x
x=x.replace('project, RESTORE54 (grouped Layer A/Q physics; explicit evaluator identity).','project, RESTORE55 (grouped Layer A/Q physics; explicit evaluator identity).')
f.write_text(x.replace(needle,'## RESTORE55 actual longitudinal and sequential-wave physics\n\n480 new fixed-root histories and eight complete games, 149 actual game\nsearch plies; exact Q-fit boundary status and tree-depth variance flow.\nSequential sector controls, restricted PL/reserve and declared\nquantum fragments/pointer and Gibbs FDT. See [theory](docs/RESTORE55_THEORY.md),\n[results](docs/RESTORE55_RESULTS.md), [reproduction](research/restore55/README.md),\n[original item evidence](docs/ORIGINAL_OUTPUT_EVIDENCE55.csv).\n\n'+needle))
print('Verified',total('jobs'),'receipts,',total('backups'),'backups, 149 game plies, 216 states, 20 original item scopes, 9 figures, 13/121 owner map')
