from pathlib import Path
import json,csv,hashlib,zipfile,subprocess
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore51_20261126';F=B/'mqft_restore49_20261126'
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();load=lambda f:json.loads(f.read_text())
r=load(S/'PHYSICS_RESULTS.json');m=load(S/'plots/grouped/manifest.json');cs=[v for q in r['counts'].values() for v in q.values()]
total=lambda k:sum(v.get(k,0) for v in cs)
assert total('jobs')==554 and r['fresh_histories']==552 and len(r['games'])==2 and r['unique_seed_groups']==120
assert total('backups')==10896 and total('rng_root')==9984 and total('rng_samples')==45844
assert total('exact_prefix_pairs')==240 and total('exact_prefix_draws')==3648
assert all(x['status']=='natural_terminal' for x in r['games'])
assert len(r['spatial'])==len(r['actions'])==len(r['diagnostics'])==18 and len(r['temporal'])==30
assert len(r['compare'])==12 and r['kl_identity']['draws']==9984
assert max(v for k,v in r['kl_identity'].items() if k!='draws')<1e-10
assert len(r['loops'])==len(r['noise'])==12 and len(r['bounds'])==18
assert sum(len(x['values']) for x in r['loops'])==36 and all(y['one_loop_error']<y['leading_error'] for x in r['loops'] for y in x['values'])
assert len(r['quantum'])==96 and len(r['thermal'])==36 and len(r['thermal_check'])==6
assert max(x['residual'] for x in r['quantum'])<1e-10 and max(x['crooks_error'] for x in r['thermal'])<1e-8
assert r['states_sha']==h(S/'CONDITIONAL_STATES.npz')
with zipfile.ZipFile(S/'CONDITIONAL_STATES.npz') as z:assert len(z.namelist())==96
assert len(m['groups'])==9 and m['source_sha256']==h(S/'PHYSICS_RESULTS.json') and m['states_sha256']==r['states_sha']
assert all(h(S/'plots/grouped'/f)==v for a in m['groups'].values() for f,v in a['files'].items())
regs={}
for L in (13,21):
 reg=load(S/f'REGISTRATION_L{L}.json');old=load(F/f'REGISTRATION_L{L}.json')
 exe=S/'runtime'/f'mqft-L{L}';q=S/'family'/f'L{L}';sm=load(q/'SOURCE_MANIFEST.json')
 assert h(exe)==reg['backend_sha'] and reg['model']==old['model']
 assert h(q/'SOURCE_MANIFEST.json')==reg['source_manifest_sha'] and all(h(q/f)==v for f,v in sm['files'].items())
 assert sm['parent_manifest_sha']==h(F/'family'/f'L{L}'/'SOURCE_MANIFEST.json')
 for f in (q/'backend/src/mcts').rglob('*.rs'):assert h(f)==h(F/'family'/f'L{L}'/f.relative_to(q))
 gate=load(S/f'GATE_L{L}.json');assert gate['backend_sha']==h(exe) and gate['singleton_scalar'] and gate['complement_gauge'] and gate['invalid_groups_rejected']==3
 x=load(S/f'PHYSICS_L{L}.json');assert h(S/f'PHYSICS_L{L}.json')==r['size_results_sha'][f'L{L}'] and x['registration_sha']==h(S/f'REGISTRATION_L{L}.json')
 for n,c in x['counts'].items():
  stem=f'{n}-L{L}';a=S/'analysis'/stem;d=S/'data'/stem
  mm=load(a/'manifest.json');rp=load(next(d.glob('report-*.json')));pl=load(S/'plans'/(stem+'.json'))
  assert mm['plan_hash']==pl['plan_hash']==reg['plans'][n]['hash']
  assert mm['completed']==mm['expected']==len(pl['jobs'])==len(rp['completed'])==c['jobs'] and not rp['failures'] and not rp['unattempted']
  assert all(h(a/f)==v for f,v in mm['outputs'].items())
  for t in rp['completed']:
   f=d/t['file'];assert h(f)==t['sha256'] and f.stat().st_size==t['bytes'] and load(d/(t['job_id']+'.receipt.json'))==t
 regs[str(L)]={'registration_sha256':h(S/f'REGISTRATION_L{L}.json'),'backend_sha256':h(exe),'source_manifest_sha256':reg['source_manifest_sha'],'plans':reg['plans']}
for n in ('CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv'):
 assert h(S/n)==h(O/n)==h(P/'docs'/n)
g=load(S/'PLOT_GROUPS.json');assert len(g['groups'])==13 and len(g['original_target_owner'])==121 and set(g['original_target_owner'])=={x['id'] for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())}
d=load(P/'docs/DAG.json');old=load(O/'DAG.json');assert d==old
d.update(revision='RESTORE52_20261126',scope='Grouped Layer A/Q: explicit native multi-action field, causal response, conditional complex state and continuous tilted generator',previous_entry=d['entry'],entry='REPLAN-04.R',supersedes_next=d['next_stage_id'],next_stage_id='MQFT-RESTORE53-GROUP-FIELD-PRECISION')
d['restore52']={'parent_dag_sha256':h(O/'DAG.json'),'status':'scoped_native_group_field_and_conditional_dynamics','sizes':[13,21],'fresh_games':2,'fresh_marked_histories':552,'total_marked_histories':6288,'fresh_backups':10896,'exact_prefix_pairs':240,'group_field_KL_verified_draws':9984,'restricted_lattice_cases':36,'conditional_quantum_states':96,'conditional_thermal_cases':36,'projected_action':'no general heldout admission','group_figures':m['groups'],'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':h(S/'CONDITIONAL_STATES.npz'),'original_plot_tasks':121,'additional_plot_nodes':0,'prior_admissions':'preserved','critical_scaling':'unresolved','native_1PI':'unresolved','next_contract':'docs/NEXT_DYNAMICS.md','evidence':['docs/RESTORE52_THEORY.md','docs/RESTORE52_RESULTS.md']}
assert d['nodes']==old['nodes'] and d['edges']==old['edges'] and len(d['nodes'])==len(d['edges'])==13
(P/'docs/DAG.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
v=load(P/'docs/PROVENANCE.json');v['restore52']={'parent_main':'3136e85a34b4e8d8440796e9b15cbc729681631e','family':regs,'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':h(S/'CONDITIONAL_STATES.npz'),'group_map_sha256':h(S/'PLOT_GROUPS.json'),'original_identity_log_sha256':h(S/'logs/original-identities.log'),'old_production_runtime_unchanged':True}
(P/'docs/PROVENANCE.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
f=P/'README.md';x=f.read_text().replace('project, RESTORE51 (grouped Layer A/Q physics; explicit evaluator identity).','project, RESTORE52 (grouped Layer A/Q physics; explicit evaluator identity).')
needle='## RESTORE51 prospective informative sources and geometry';assert needle in x
f.write_text(x.replace(needle,'## RESTORE52 native spatial group field\n\nSeparate v4 size-scoped opt-in group field, same MixNet, 552 new\nroot histories and two complete games, 240 exact pre-pulse controls.\nExact conditional binary KL, complex conditional states and continuous\nwork tilt. See [theory](docs/RESTORE52_THEORY.md), [results](docs/RESTORE52_RESULTS.md),\n[reproduction](research/restore52/README.md).\n\n'+needle))
print('Verified',total('jobs'),'receipts,',total('backups'),'backups, source manifests and exact 13/121 mapping')
