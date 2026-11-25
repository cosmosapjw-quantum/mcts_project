from pathlib import Path
import json,csv,hashlib,zipfile
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore53_20261126'
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();load=lambda f:json.loads(f.read_text())
r=load(S/'PHYSICS_RESULTS.json');m=load(S/'plots/grouped/manifest.json');c=load(S/'CRN_CALIBRATION.json')
cs=[v for q in r['counts'].values() for v in q.values()];total=lambda k:sum(v.get(k,0) for v in cs)
assert total('jobs')==850 and total('searches')==900 and total('backups')==16960
assert r['fresh_histories']==848 and r['unique_seed_groups']==80 and len(r['games'])==2
assert all(x['status']=='natural_terminal' for x in r['games'])
assert total('rng_root')==16128 and total('rng_samples')==71272 and total('exact_prefix_pairs')==768 and total('exact_prefix_draws')==11264
assert len(r['groups'])==13 and sum(len(x['rows']) for x in r['groups'])==66 and all(x['S0']==0 for x in r['groups'])
assert len(r['exact_count'])==72 and all(len(x['counts'])==35 for x in r['exact_count'])
vv=[v for x in r['exact_count'] for v in x['rows']]
assert len(vv)==288 and all(v['interior'] and v['one_loop_error']<v['leading_error'] for v in vv)
assert len(r['quantum'])==48 and len(r['thermal'])==12 and len(r['noise'])==13
assert max(x['complex_crooks_error'] for x in r['thermal'])<1e-8
assert max(x['residual'] for x in r['noise'])<1e-12 and r['max_tilt_error']<1e-10
assert c['source_sha']==h(S/'PHYSICS_RESULTS.json') and c['M']==16384 and c['simulated_uniforms']==2097152 and len(c['labels'])==6
assert r['states_sha']==h(S/'CHOI_STATES.npz') and r['work_samples_sha']==h(S/'WORK_SAMPLES.npz')
with zipfile.ZipFile(S/'CHOI_STATES.npz') as z:assert len(z.namelist())==192
with zipfile.ZipFile(S/'WORK_SAMPLES.npz') as z:assert len(z.namelist())==24
assert len(m['groups'])==9 and m['source_sha256']==h(S/'PHYSICS_RESULTS.json') and m['states_sha256']==r['states_sha'] and m['work_samples_sha256']==r['work_samples_sha'] and m['crn_sha256']==h(S/'CRN_CALIBRATION.json')
assert all(h(S/'plots/grouped'/f)==v for a in m['groups'].values() for f,v in a['files'].items())
regs={}
for L in (13,21):
 reg=load(S/f'REGISTRATION_L{L}.json');old=load(O/f'REGISTRATION_L{L}.json');q=O/'family'/f'L{L}';exe=O/'runtime'/f'mqft-L{L}'
 assert h(exe)==reg['backend_sha']==old['backend_sha'] and reg['model']==old['model']
 assert h(q/'SOURCE_MANIFEST.json')==reg['source_manifest_sha']==old['source_manifest_sha']
 sm=load(q/'SOURCE_MANIFEST.json');assert all(h(q/f)==v for f,v in sm['files'].items())
 patch=S/'python'/f'L{L}';pm=load(patch/'PATCH_MANIFEST.json')
 assert pm['parent']==h(q/'python/mqft/analysis.py') and pm['modified']==h(patch/'mqft/analysis.py') and all(h(patch/f)==v for f,v in pm['files'].items())
 x=load(S/f'PHYSICS_L{L}.json');assert h(S/f'PHYSICS_L{L}.json')==r['size_results_sha'][f'L{L}'] and x['registration_sha']==h(S/f'REGISTRATION_L{L}.json')
 for n,c0 in x['counts'].items():
  stem=f'{n}-L{L}';a=S/'analysis'/stem;d=S/'data'/stem
  mm=load(a/'manifest.json');rp=load(next(d.glob('report-*.json')));pl=load(S/'plans'/(stem+'.json'));lock=load(d/'run_lock.json')
  assert lock['backend_sha']==reg['backend_sha'] and lock['external']==reg['model']
  assert lock['plan_hash']==mm['plan_hash']==pl['plan_hash']==reg['plans'][n]['hash']
  assert mm['completed']==mm['expected']==len(pl['jobs'])==len(rp['completed'])==c0['jobs'] and not rp['failures'] and not rp['unattempted']
  assert all(h(a/f)==v for f,v in mm['outputs'].items())
  for t in rp['completed']:
   f=d/t['file'];assert h(f)==t['sha256'] and f.stat().st_size==t['bytes'] and load(d/(t['job_id']+'.receipt.json'))==t
 regs[str(L)]={'registration_sha256':h(S/f'REGISTRATION_L{L}.json'),'backend_sha256':h(exe),'source_manifest_sha256':reg['source_manifest_sha'],'analysis_patch_manifest_sha256':h(patch/'PATCH_MANIFEST.json'),'plans':reg['plans']}
for n in ('CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv'):
 assert h(S/n)==h(O/n)==h(P/'docs'/n)
g=load(S/'PLOT_GROUPS.json');assert len(g['groups'])==13 and len(g['original_target_owner'])==121 and set(g['original_target_owner'])=={x['id'] for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())}
d=load(P/'docs/DAG.json');old=load(O/'DAG.json');assert d==old
d.update(revision='RESTORE54_20261126',scope='Grouped Layer A/Q: exact conditional IID last-wave response, count action and Fisher pushforward, mixed finite precision, conditional quantum memory and reverse-work distribution',previous_entry=d['entry'],entry='REPLAN-03',supersedes_next=d['next_stage_id'],next_stage_id='MQFT-RESTORE55-WAVE-CLOSURE-LONGITUDINAL')
d['restore54']={'parent_dag_sha256':h(O/'DAG.json'),'status':'scoped_exact_iid_conditional_response_and_conditional_physics','sizes':[13,21],'fresh_games':2,'fresh_marked_histories':848,'total_marked_histories':8168,'fresh_backups':16960,'exact_prefix_pairs':768,'new_core_geometries':1,'precision_seed_groups':32,'mixed_pairs_measured':[[0,1],[0,2],[1,2]],'response_rows':66,'conditional_iid_count_laws':72,'frozen_count_cases':288,'CRN_calibration':{'M':16384,'max_stat_tail':c['max_stat_tail_MC'],'sha256':h(S/'CRN_CALIBRATION.json')},'adaptive_kernel':'exact IID final-wave occupancy only; PL/reserve approximate; marks/future adaptive kernel unresolved','conditional_quantum_cases':48,'conditional_choi_states':192,'conditional_thermal_cases':12,'conditional_work_paths':98304,'group_figures':m['groups'],'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':r['states_sha'],'work_samples_sha256':r['work_samples_sha'],'original_plot_tasks':121,'additional_plot_nodes':0,'prior_admissions':'preserved','critical_scaling':'unresolved','native_1PI':'unresolved','next_contract':'docs/NEXT_DYNAMICS.md','evidence':['docs/RESTORE54_THEORY.md','docs/RESTORE54_RESULTS.md']}
assert d['nodes']==old['nodes'] and d['edges']==old['edges'] and len(d['nodes'])==len(d['edges'])==13
(P/'docs/DAG.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
v=load(P/'docs/PROVENANCE.json');v['restore54']={'parent_main':'5b449021025417f70c8f7cc59c0ed890c45a1359','family':regs,'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'states_sha256':r['states_sha'],'work_samples_sha256':r['work_samples_sha'],'crn_sha256':h(S/'CRN_CALIBRATION.json'),'group_map_sha256':h(S/'PLOT_GROUPS.json'),'original_identity_log_sha256':h(S/'logs/original-identities.log'),'old_production_runtime_unchanged':True}
(P/'docs/PROVENANCE.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
f=P/'README.md';x=f.read_text();needle='## RESTORE53 native vector field and conditional dynamics';assert needle in x
x=x.replace('project, RESTORE53 (grouped Layer A/Q physics; explicit evaluator identity).','project, RESTORE54 (grouped Layer A/Q physics; explicit evaluator identity).')
f.write_text(x.replace(needle,'## RESTORE54 exact conditional IID wave and response precision\n\n848 new root histories, two complete games, 768 exact late-prefix controls.\nSame v5 native MixNet; exact final IID occupancy expectation and paired CDF\nvariance, full conditional sector count action, quantum memory Choi and\ncontinuous forward/reverse work checks. See [theory](docs/RESTORE54_THEORY.md),\n[results](docs/RESTORE54_RESULTS.md), [reproduction](research/restore54/README.md).\n\n'+needle))
print('Verified',total('jobs'),'receipts,',total('backups'),'backups, 13/121 owner map, 9 figures, 192 Choi, 24 work arrays, CRN')
