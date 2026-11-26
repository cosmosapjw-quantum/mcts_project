from base61 import *
import csv
O=B/'mqft_restore64_20261126';P=B/'mqft_restore39_20261125/repo'
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/n)==sha(O/n)==sha(P/'docs'/n)
p=read(O/'PROGRESS64.json');assert sha(O/'PLOT_INVENTORY64.csv')==p['inventory_sha256']
inv=csvread(O/'PLOT_INVENTORY64.csv');assert len(inv)==501
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST65.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==509
write(S/'PLOT_INVENTORY65.csv',inv)
from PIL import Image
for x in pm['animations']:
 q=S/'plots'/x['name'];assert sha(q)==x['sha256'] and q.stat().st_size==x['bytes']
 with Image.open(q) as im:
  assert im.n_frames==x['frames']==32
  for k in range(im.n_frames):im.seek(k);im.load()
assert len(pm['animations'])==2
cat={x['id']:x for x in csvread(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=csvread(O/'ORIGINAL_OUTPUT_PROGRESS64.csv');assert len(rows)==121
scope={
'L09':'Actual persistent conditional density and selective coherence/purity; auxiliary software model.',
'L16':'Actual persistent native likelihood, exact within-wave tilted instrument and count mixture; no physical Feynman integral.',
'L20':'Explicit finite free channel plus selective Kraus state dynamics, not inferred physical Lindblad bath.',
'L40':'Actual within-wave probability changes and reset comparison; no general projected Markov sufficiency.',
'L41':'Exact conditional count mean/covariance and observed iid-versus-mixture NLL; finite scope.',
'L43':'Actual selective wave generating operator, exact commuting mixture and measured residual.',
'L48':'Scoped paired utility result, negative persistent-g3 evidence preserved.',
'L49':'Independent high-budget reference Q proxy/top5, not true regret or strength.',
'Q24':'Actual persistent auxiliary coherence and purity; no physical Darwinism claim.',
'Q27':'Actual 32-frame coherence/purity animation at completed native clock, static counterpart; no entanglement or size scan.',
'Q41':'Actual wave-averaged absolute visibility proxy with explicit unit-action principal-branch floor, not physical hbar.',
'Q42':'Actual persistent selective state, free propagation, measurement backaction; software calibration, no physical bath.',
'Q43':'Actual phase-conditioned native path outcomes and exact finite tilted wave instrument, not physical path integral.',
'Q44':'Actual 32-frame visibility-match proxy animation and static counterpart; original .5 threshold unreachable, no physical regime.',
'Q45':'Independent reference utility and actual phase response, no production admission.'
}
ev=[]
for x in rows:
 k=x['id'];x['R65_evidence']=scope.get(k,'');x['R65_scope_status']='SCOPED_NATIVE_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];f=[v['figure'] for v in pm['figures'] if k in v['ids']]+[v['name'] for v in pm['animations'] if k in v['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_PANEL',scope=scope[k]))
assert len(ev)==15 and all(x['evidence_figures'] for x in ev)
write(S/'ORIGINAL_OUTPUT_EVIDENCE65.csv',ev);write(S/'ORIGINAL_OUTPUT_PROGRESS65.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,66))]
assert len(union)==62 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(O/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={
'X-REP':('Persistent selective software state on actual full-support native paths, matrix replay, exact commuting wave mixture, actual Q27/Q44 animations.','Physical phase/action/hbar identification and independent physical dynamics unresolved; negative utility kept.'),
'REPLAN-03':('Exact finite tilted instrument count generator and rank-one covariance correction; all 49152 persistent waves analyzed, conditional mixture NLL improved.','Cross-wave quantum/coarse closure, physical action/1PI and reverse protocol remain.'),
'X-MEMORY':('Actual measurement-conditioned within-wave state memory, reset comparison and conditional mixture; old iid applies to frozen-score control only.','Cross-wave projected sufficient state not established.'),
'REPLAN-07':('Independent g3 high-budget Q-proxy and top5 comparisons: persistent plus worsens compared with baseline despite improved mean mark.','No production admission; independent complete-game utility, size transfer.'),
'X-TRAIN':('4800 fresh fixed-root paths including 192 independent g3 B1024 references, no new training.','Actual independent game strength and 15x15+ remain.'),
'REPLAN-05':('Actual outcome-conditioned auxiliary Kraus record and selective purity; Q27 animation.','Physical environmental records/redundancy and thermodynamic cost unresolved.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore65_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore65_20261126/{RESULTS_KO.md,PHYSICS_RESULTS.json,AUDIT65.json,GENERATOR65.json}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=509,distinct_png_sha=509,files_paired=1018,current_pairs=8,prior_pairs=501,current_animations=2,animation_scope='two new verified GIFs; cumulative historical animation inventory not counted',original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE65.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS65.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY65.csv'))
p['stages'][S.name]=8;p['original_output_evidence']=dict(current=15,R55_R65_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R65 overlays, not exhaustive historical evidence; no automatic final certification.')
dump(S/'PROGRESS65.json',p)
a=read(S/'AUDIT65.json');g=read(S/'GENERATOR65.json');r=read(S/'PHYSICS_RESULTS.json');assert a['n']==4800 and a['backups']==786432 and g['n_waves']==49152
dag.update(revision='RESTORE65_20261126',scope='Persistent native selective auxiliary instrument, exact conditional wave generator and independent utility proxy',previous_entry=dag['entry'],entry='REPLAN-06',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE66-THERMODYNAMIC-RECORD-WORK-PROTOCOL')
r0=dict(status='scoped_actual_persistent_native_instrument_negative_utility',parent_dag_sha256=sha(O/'DAG.json'),new_games=0,new_full_game_search_plies=0,fresh_fixed_root_histories=4800,low_budget_histories=4608,reference_histories=192,total_marked_histories=53816,fresh_backups=786432,evaluator_calls=a['evaluations'],raw_gzip_bytes=a['raw_gzip_bytes'],conditional_generator_waves=g['n_waves'],calibration_software_shots=read(S/'CALIBRATION65.json')['shots'],utility='g3 persistent plus worse top5 and Q regret proxy vs baseline; no production admission',original_plot_tasks=121,original_evidence_items=15,original_evidence_union_R55_R65=len(union),figures=8,new_animations=2,retained_pairs=509,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','AUDIT65.json','RNG_AUDIT65.json','EVALUATOR65.json','LEGACY_REGRESSION65.json','CALIBRATION65.json','CALIBRATION65.npz','PHYSICS_RESULTS.json','PHASE65.npz','REFERENCE65.npz','GENERATOR65.json','GENERATOR65.npz','ORIGINAL_OUTPUT_EVIDENCE65.csv','ORIGINAL_OUTPUT_PROGRESS65.csv','PLOT_INVENTORY65.csv','PROGRESS65.json','PLOT_MANIFEST65.json']:
 r0[f+'_sha256']=sha(S/f)
dag['restore65']=r0;dump(S/'DAG.json',dag)
v=read(O/'PROVENANCE.json');v['restore65']=dict(r0,dag_sha256=sha(S/'DAG.json'),parent='b77047387f09693466487193d195e80b3e2e7519',weights=read(S/'REGISTRATION.json')['weights']);dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'plot pairs;',len(union),'original-item overlay; DAG',sha(S/'DAG.json'))
