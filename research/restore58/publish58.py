from pathlib import Path
import json,csv,hashlib,zipfile
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore57_20261126';F=B/'mqft_restore49_20261126'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
r=load(S/'PHYSICS_RESULTS.json');m=load(S/'plots/manifest.json');progress=load(S/'PROGRESS58.json')
cs=[v for q in r['counts'].values() for v in q.values()];total=lambda k:sum(v.get(k,0) for v in cs)
assert total('jobs')==480 and total('searches')==480 and total('backups')==10560
assert r['fresh_histories']==480 and r['unique_seed_groups']==360 and r['baseline_histories']==360
assert total('rng_root')==10560 and total('rng_samples')==44160 and total('runtime_retries')==4
assert total('exact_prefix_pairs')==120 and total('exact_prefix_draws')==1440
assert len(r['spatial'])==60 and len(r['response'])==10 and len(r['scaling'])==60 and len(r['survival'])==30
assert len(r['topology'])==480 and len(r['barriers'])==480
assert sum(x['independent_thresholds'] for x in r['topology'])==6582
assert r['fault_counts']=={'single_corrected':1920,'double_detected_uncorrectable':1440} and r['fault_max_error']==0
assert all(x['trie']['b1']==0 and x['trie']['E']==x['trie']['V']-1 for x in r['topology'])
assert all(x['connected'] for x in r['barriers'])
assert all(x['rank']<=5 and abs(x['S0'])<1e-12 and x['decomposition_error']<1e-12 for x in r['spatial'])
crn=load(S/'CRN_VALIDATION.json');assert crn['source_sha']==h(S/'PHYSICS_RESULTS.json') and len(crn['rows'])==60 and len(crn['aggregates'])==6
for x in crn['aggregates']:
 for k in ('occupancy','detection'):assert abs(sum(x[k]['pmf'])-1)<1e-12
assert len(m['groups'])==7 and m['source_sha256']==h(S/'PHYSICS_RESULTS.json')
assert all(h(S/f)==v for a in m['groups'].values() for f,v in a['files'].items())
assert progress['figure_pairs']==459 and progress['distinct_png_sha']==459 and progress['current_pairs']==7
assert progress['inventory_sha256']==h(S/'PLOT_INVENTORY58.csv')
assert progress['original_evidence_sha256']==h(S/'ORIGINAL_OUTPUT_EVIDENCE58.csv')
assert progress['original_progress_sha256']==h(S/'ORIGINAL_OUTPUT_PROGRESS58.csv')
regs={}
for L in (13,15,17,19,21):
 reg=load(S/f'REGISTRATION_L{L}.json');old=load(F/f'REGISTRATION_L{L}.json');q=F/'family'/f'L{L}';exe=F/'runtime_verified'/f'mqft-L{L}'
 assert h(exe)==reg['backend_sha']==old['backend_sha'] and reg['model']==old['model']
 assert h(q/'SOURCE_MANIFEST.json')==reg['source_manifest_sha']==old['source_manifest_sha']
 sm=load(q/'SOURCE_MANIFEST.json');assert all(h(q/f)==v for f,v in sm['files'].items())
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
 regs[str(L)]={'registration_sha256':h(S/f'REGISTRATION_L{L}.json'),'backend_sha256':h(exe),'source_manifest_sha256':reg['source_manifest_sha'],'plans':reg['plans']}
for n in ('CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv'):
 assert h(S/n)==h(O/n)==h(P/'docs'/n)
g=load(S/'PLOT_GROUPS.json');cat={x['id']:x for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())}
assert len(g['groups'])==13 and len(g['original_target_owner'])==121 and set(g['original_target_owner'])==set(cat)
ev=list(csv.DictReader((S/'ORIGINAL_OUTPUT_EVIDENCE58.csv').open()));assert len(ev)==15 and len(set(x['id'] for x in ev))==15
for x in ev:
 assert x['primary_node']==g['original_target_owner'][x['id']] and x['legacy_status']==cat[x['id']]['status']
 assert all(x['id'] in m['groups'][k]['original_items'] for k in x['evidence_figures'].split(';'))
 if cat[x['id']]['archive']!='guide_only':assert x['source_sha256']==cat[x['id']]['member_sha256']
 else:assert x['source_sha256'] in ('bfac951744509e0398f58a801855acb355bf9d95754c0ccbe231baf570228db6','1789cedff042b0d1da1a2368ca2ce3f5dc60f5ec60e68c1834cef18caaa528aa')
e=list(csv.DictReader((S/'ORIGINAL_OUTPUT_PROGRESS58.csv').open()))
assert len(e)==121 and {x['id'] for x in e}==set(cat) and all(x['final_completion']=='NOT_CERTIFIED' for x in e)
d=load(P/'docs/DAG.json');old=load(O/'DAG.json');assert d==old
d.update(revision='RESTORE58_20261126',scope='Five-size native finite-budget covariance and adaptive drift; conditional first detection; geometric barrier, cubical filtration and offline classical fault recovery',previous_entry=d['entry'],entry='REPLAN-04.C',supersedes_next=d['next_stage_id'],next_stage_id='MQFT-RESTORE59-QUANTUM-CONTINUATION-TRAINING-OUTPUTS')
d['restore58']={'parent_dag_sha256':h(O/'DAG.json'),'status':'scoped_finite_size_transport_structure_native_outputs','sizes':[13,15,17,19,21],'fresh_games':0,'fresh_marked_histories':480,'total_marked_histories':9560,'fresh_backups':10560,'baseline_histories':360,'paired_seed_groups':60,'new_core_geometries':2,'spatial_groups':60,'conditional_response_groups':10,'exact_prefix_pairs':120,'CRN_sha256':h(S/'CRN_VALIDATION.json'),'topology_threshold_checks':6582,'offline_single_corrected':1920,'offline_double_detected':1440,'original_evidence_items':15,'original_evidence_sha256':h(S/'ORIGINAL_OUTPUT_EVIDENCE58.csv'),'progress_sha256':h(S/'PROGRESS58.json'),'inventory_sha256':h(S/'PLOT_INVENTORY58.csv'),'original_progress_sha256':h(S/'ORIGINAL_OUTPUT_PROGRESS58.csv'),'group_figures':m['groups'],'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'original_plot_tasks':121,'additional_plot_nodes':0,'prior_admissions':'preserved','critical_scaling':'unstable slopes; unresolved','native_1PI':'unresolved','tunneling':'geometric/first detection only','QEC':'offline classical reference only','next_contract':'docs/NEXT_DYNAMICS.md','evidence':['docs/RESTORE58_THEORY.md','docs/RESTORE58_RESULTS.md','docs/ORIGINAL_OUTPUT_EVIDENCE58.csv','docs/PROGRESS58.json','docs/PLOT_INVENTORY58.csv','docs/ORIGINAL_OUTPUT_PROGRESS58.csv']}
assert d['nodes']==old['nodes'] and d['edges']==old['edges'] and len(d['nodes'])==len(d['edges'])==13
(P/'docs/DAG.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
v=load(P/'docs/PROVENANCE.json');v['restore58']={'parent_main':'0d5b40720cc25952e0e47c1836f98c78bad4a0f8','family':regs,'physics_results_sha256':h(S/'PHYSICS_RESULTS.json'),'CRN_sha256':h(S/'CRN_VALIDATION.json'),'original_evidence_sha256':h(S/'ORIGINAL_OUTPUT_EVIDENCE58.csv'),'progress_sha256':h(S/'PROGRESS58.json'),'inventory_sha256':h(S/'PLOT_INVENTORY58.csv'),'original_progress_sha256':h(S/'ORIGINAL_OUTPUT_PROGRESS58.csv'),'group_map_sha256':h(S/'PLOT_GROUPS.json'),'original_identity_log_sha256':h(S/'logs/original-identities.log'),'old_production_runtime_unchanged':True}
(P/'docs/PROVENANCE.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
f=P/'README.md';x=f.read_text();needle='## RESTORE57 dedicated original native and thermodynamic outputs';assert needle in x
x=x.replace('project, RESTORE57 (grouped Layer A/Q physics; explicit evaluator identity).','project, RESTORE58 (grouped Layer A/Q physics; explicit evaluator identity).')
f.write_text(x.replace(needle,'## RESTORE58 critical, transport and structural native diagnostics\n\n480 actual five-size histories; fixed-budget covariance with adaptive drift;\nexact final-wave detection, geometric minimax barrier, cubical filtration,\noffline classical backup recovery. No critical exponent, physical tunneling\nor quantum code claim. See [theory](docs/RESTORE58_THEORY.md), [results](docs/RESTORE58_RESULTS.md),\n[reproduction](research/restore58/README.md), [progress](docs/PROGRESS58.json).\n\n'+needle))
print('Verified',total('jobs'),'receipts,',total('backups'),'backups, 60 spatial groups, 120 exact prefix pairs, 7 plots, 459-pair inventory, 13/121 map')
