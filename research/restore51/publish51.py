"""Verify all current scientific artifacts before changing DAG or publication."""
from pathlib import Path
import json,csv,hashlib,zipfile,subprocess
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore50_20261126';F=B/'mqft_restore49_20261126'
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();load=lambda f:json.loads(f.read_text())
r=load(S/'PHYSICS_RESULTS.json');m=load(S/'plots/grouped/manifest.json');cs=[v for q in r['counts'].values() for v in q.values()]
total=lambda k:sum(v.get(k,0) for v in cs)
assert total('jobs')==1719 and r['fresh_histories']==1716 and len(r['games'])==3
assert all(x['status']=='natural_terminal' for x in r['games'])
assert total('rng_root')==30144 and total('exact_prefix_pairs')==672 and total('exact_prefix_draws')==7872
assert len(r['spatial'])==len(r['actions'])==54 and len(r['actions_n16'])==6 and len(r['temporal'])==72
assert len(r['diagnostics'])==78 and len(r['peripheral_variants'])==27 and len(r['memory'])==9
assert all(x['S0']==0 for x in r['spatial']) and len(r['loops'])==len(r['noise'])==len(r['score_bounds'])==54
assert sum(len(x['values']) for x in r['loops'])==162 and all(y['one_loop_error']<y['leading_error'] for x in r['loops'] for y in x['values'])
assert len(r['quantum'])==1296 and len(r['thermal'])==162 and len(r['thermal_check'])==9
assert max(x['residual'] for x in r['quantum'])<1e-10 and max(x['residual'] for x in r['thermal'])<1e-8
assert max(x['max_error'] for x in r['thermal_check'])<1e-8 and r['static_sha']==h(S/'STATIC_RESULTS.json')
assert r['states_sha256']==h(S/'CONDITIONAL_ENV_STATES.npz')
with zipfile.ZipFile(S/'CONDITIONAL_ENV_STATES.npz') as z:assert len(z.namelist())==432
assert len(m['groups'])==9 and m['source_sha256']==h(S/'PHYSICS_RESULTS.json') and m['states_sha256']==r['states_sha256']
assert all(h(S/'plots/grouped'/f)==v for a in m['groups'].values() for f,v in a['files'].items())
assert len(list((S/'plots/grouped').glob('*.png')))==len(list((S/'plots/grouped').glob('*.pdf')))==9
regs={}
for L in (13,17,21):
 reg=load(S/f'REGISTRATION_L{L}.json');old=load(F/f'REGISTRATION_L{L}.json')
 assert h(F/'runtime_verified'/f'mqft-L{L}')==reg['backend_sha']==old['backend_sha'] and reg['model']==old['model']
 assert h(F/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==reg['source_manifest_sha']
 x=load(S/f'PHYSICS_L{L}.json');assert h(S/f'PHYSICS_L{L}.json')==r['size_results_sha'][f'L{L}'] and x['registration_sha']==h(S/f'REGISTRATION_L{L}.json')
 for n,c in x['counts'].items():
  stem=f'{n}-L{L}';a=S/'analysis'/stem;d=S/'data'/stem
  mm=load(a/'manifest.json');rp=load(next(d.glob('report-*.json')));pl=load(S/'plans'/(stem+'.json'))
  assert mm['plan_hash']==pl['plan_hash']==reg['plans'][n]['hash']
  assert mm['completed']==mm['expected']==len(pl['jobs'])==len(rp['completed'])==c['jobs'] and not rp['failures'] and not rp['unattempted']
  assert all(h(a/f)==v for f,v in mm['outputs'].items())
  for t in rp['completed']:
   f=d/t['file'];assert h(f)==t['sha256'] and f.stat().st_size==t['bytes'] and load(d/(t['job_id']+'.receipt.json'))==t
 regs[str(L)]={'registration_sha256':h(S/f'REGISTRATION_L{L}.json'),'backend_sha256':reg['backend_sha'],'source_manifest_sha256':reg['source_manifest_sha'],'plans':reg['plans']}
for n in ('CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv'):
 assert h(S/n)==h(O/n)==h(P/'docs'/n)
g=load(S/'PLOT_GROUPS.json');assert len(g['groups'])==13 and len(g['original_target_owner'])==121 and set(g['original_target_owner'])=={x['id'] for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())}
d=load(P/'docs/DAG.json');old=load(O/'DAG.json');assert d==old
d.update(revision='RESTORE51_20261126',scope='Grouped Layer A/Q: prospective informative sources, independent central/peripheral geometry, direct temporal response and coherent/continuous thermal references',previous_entry=d['entry'],entry='REPLAN-04.R',supersedes_next=d['next_stage_id'],next_stage_id='MQFT-RESTORE52-ABSOLUTE-INFORMATION-SECTOR-FIELD')
d['restore51']={'parent_dag_sha256':h(O/'DAG.json'),'status':'scoped_native_source_geometry_and_conditional_dynamics','sizes':[13,17,21],'fresh_games':3,'fresh_marked_histories':1716,'total_marked_histories':5736,'fresh_backups':total('backups'),'pilot_histories':108,'independent_confirm_histories':432,'pulse_histories':1176,'new_core_geometries':3,'root_variants':18,'exact_prefix_pairs':672,'restricted_lattice_cases':162,'conditional_quantum_cases':1296,'conditional_thermal_cases':162,'source_information':'L21 high informative; L13 absolute information low','projected_action':'rank improved, no general heldout predictive admission','memory':'mark improves, sector worsens; no universal closure','group_figures':m['groups'],'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':h(S/'CONDITIONAL_ENV_STATES.npz'),'original_plot_tasks':121,'additional_plot_nodes':0,'prior_admissions':'preserved','critical_scaling':'unresolved','native_1PI':'unresolved','next_contract':'docs/NEXT_DYNAMICS.md','evidence':['docs/RESTORE51_THEORY.md','docs/RESTORE51_RESULTS.md']}
assert d['nodes']==old['nodes'] and d['edges']==old['edges'] and len(d['nodes'])==len(d['edges'])==13
(P/'docs/DAG.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
v=load(P/'docs/PROVENANCE.json');v['restore51']={'parent_main':'3ac8d282abdb15231e7c7cc8050c4bf48f0ab824','family':regs,'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'static_sha256':h(S/'STATIC_RESULTS.json'),'states_sha256':h(S/'CONDITIONAL_ENV_STATES.npz'),'reference_registration_sha256':h(S/'REFERENCE_REGISTRATION.json'),'group_map_sha256':h(S/'PLOT_GROUPS.json'),'original_identity_log_sha256':h(S/'logs/original-identities.log'),'production_runtime_unchanged':True}
(P/'docs/PROVENANCE.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
f=P/'README.md';x=f.read_text().replace('project, RESTORE50 (grouped Layer A/Q physics; explicit evaluator identity).','project, RESTORE51 (grouped Layer A/Q physics; explicit evaluator identity).')
needle='## RESTORE50 density-controlled spatial and temporal physics';assert needle in x
f.write_text(x.replace(needle,'## RESTORE51 prospective informative sources and geometry\n\n1716 new actual histories and three games, independent h0 pilot selection,\nthree central templates and peripheral variants, direct temporal pulses,\n672 exact prefix controls. Rank-aware action and conditional quantum\nenvironment and continuous thermal friction. See [theory](docs/RESTORE51_THEORY.md),\n[results](docs/RESTORE51_RESULTS.md), [reproduction](research/restore51/README.md).\n\n'+needle))
print('Verified',total('jobs'),'receipts,',total('backups'),'backups, exact 13/121 mapping and all figure/state identities')
