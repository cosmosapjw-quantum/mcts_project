from base61 import *
import csv
from PIL import Image
R=B/'mqft_restore65_20261126';P=B/'mqft_restore39_20261125/repo'
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/n)==sha(R/n)==sha(P/'docs'/n)
p=read(R/'PROGRESS65.json');assert sha(R/'PLOT_INVENTORY65.csv')==p['inventory_sha256']
inv=csvread(R/'PLOT_INVENTORY65.csv');assert len(inv)==509
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST66.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==520
write(S/'PLOT_INVENTORY66.csv',inv)
for x in pm['animations']:
 q=S/'plots'/x['name'];assert sha(q)==x['sha256'] and q.stat().st_size==x['bytes']
 with Image.open(q) as im:
  assert im.n_frames==x['frames']==9
  for k in range(im.n_frames):im.seek(k);im.load()
assert len(pm['animations'])==2
cat={x['id']:x for x in csvread(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=csvread(R/'ORIGINAL_OUTPUT_PROGRESS65.csv');assert len(rows)==121
scope={
'L06':'Explicit finite thermal model work/heat and exact energy balance, not native physical heat.',
'L08':'Explicit auxiliary measured-record feedback, information-corrected IFT and finite-rate efficiency; no device cycle.',
'L32':'Actual auxiliary finite-error erasure and record-conditioned work extraction replacing placeholder; no measured physical cost.',
'L33':'Finite thermal work/heat evolution on frozen native pilot score schedules.',
'Q01':'Explicit Gibbs and reversible driven auxiliary chain with actual forward/reverse paths, not physical MCTS bath.',
'Q03':'Driven quench and relaxation work/heat; not cyclic heat-engine efficiency.',
'Q04':'Actual exact model ensemble nine-stage work/heat/entropy GIF plus static counterpart.',
'Q05':'Native same-order likelihood and explicit auxiliary entropy-production balance; finite sample residuals retained.',
'Q19':'Actual auxiliary forward/reverse thermal experiments, exact DeltaF and finite-sample IFT failures.',
'Q20':'Actual auxiliary work trajectory and distribution, estimator uncertainty, not artificial random-walk surrogate.',
'Q21':'Actual normalized auxiliary path weights and Crooks bins, not physical Feynman integral.',
'Q22':'Actual finite sample normalized-weight convergence GIF with independent F/R data, not prescribed success.',
'Q23':'Honest comprehensive finite-model Jarzynski/feedback report, failed finite estimates retained.',
'Q38':'Explicit auxiliary Gibbs entropy/capacity and binary record erasure cost, no native criticality.'
}
ev=[]
for x in rows:
 k=x['id'];x['R66_evidence']=scope.get(k,'');x['R66_scope_status']='SCOPED_AUXILIARY_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];f=[v['figure'] for v in pm['figures'] if k in v['ids']]+[v['name'] for v in pm['animations'] if k in v['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_AUXILIARY_PANEL',scope=scope[k]))
assert len(ev)==14 and all(x['evidence_figures'] for x in ev)
write(S/'ORIGINAL_OUTPUT_EVIDENCE66.csv',ev);write(S/'ORIGINAL_OUTPUT_PROGRESS66.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,67))]
assert len(union)==70 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(R/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={
'REPLAN-06':('Actual native same-order source schedule contrast plus finite auxiliary Gibbs F/R work/heat, binary record erasure and measurement-conditioned feedback; independently audited 6291456 auxiliary paths.','Native physical reverse/energy calibration and device cycle unresolved; two finite-sample reverse IFT failures retained.'),
'REPLAN-02':('Actual native conditional record entropy and auxiliary Gibbs fixed-spectrum entropy/capacity; explicit energy unit declared.','Native thermometry/physical bath not identified; no criticality inference.'),
'REPLAN-03':('Actual same-order native path RN law and finite thermal tilted transfer, exact Crooks/Jarzynski in defined auxiliary kernel.','Native tree reverse and physical 1PI/field action still unresolved; rare-event ESS.'),
'REPLAN-05':('Binary pilot pair erasure and record-conditioned feedback with independent normalized reverse, explicit detector accuracy and costs omitted.','Physical record preparation/reset and complete joint cycle/quantum record redundancy remain.'),
'X-STRUCT':('Original thermodynamic/record target crosswalk and independent finite-model validation; next actual QEC fault injection.','Actual logical/noise recovery and native utility not tested in R66.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore66_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore66_20261126/{RESULTS_KO.md,PHYSICS_RESULTS.json,AUDIT66.json,VERIFY_AUX66.json,THERMAL66.json,ERASURE66.json,FEEDBACK66.json}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=520,distinct_png_sha=520,files_paired=1040,current_pairs=11,prior_pairs=509,current_animations=2,animation_scope='two new verified GIFs, historical animation inventory not counted',original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE66.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS66.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY66.csv'))
p['stages'][S.name]=11;p['original_output_evidence']=dict(current=14,R55_R66_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R66 overlays, not exhaustive historical evidence; no automatic final certification.')
dump(S/'PROGRESS66.json',p)
a=read(S/'AUDIT66.json');v=read(S/'VERIFY_AUX66.json');assert a['n']==3072 and a['backups']==393216 and v['paths']==6291456 and v['transitions']==92012544
dag.update(revision='RESTORE66_20261126',scope='Actual native same-order likelihood and explicit auxiliary finite thermodynamic work/heat/feedback',previous_entry=dag['entry'],entry='X-STRUCT',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE67-FAULT-INJECTED-QEC-RECOVERY')
r0=dict(status='scoped_native_schedule_and_auxiliary_thermodynamic_models',parent_dag_sha256=sha(R/'DAG.json'),new_games=0,new_full_game_search_plies=0,fresh_fixed_root_histories=3072,total_marked_histories=56888,fresh_backups=393216,evaluator_calls=a['evaluations'],raw_gzip_bytes=a['raw_gzip_bytes'],auxiliary_paths=v['paths'],auxiliary_transitions=v['transitions'],auxiliary_protocol_files=v['files'],native_reverse='NOT_IMPLEMENTED_same_forward_tree_clock',finite_estimator_limit='alpha1 reverse IFT two individual intervals exclude unity; g3 F native conditional KL residual excludes zero; BAR improves all four MAE',original_plot_tasks=121,original_evidence_items=14,original_evidence_union_R55_R66=len(union),figures=11,new_animations=2,retained_pairs=520,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','THERMAL_SCHEDULES.json','AUDIT66.json','RNG_AUDIT66.json','LEGACY_REGRESSION66.json','THERMAL66.json','ERASURE66.json','FEEDBACK66.json','FEEDBACK_CONTRACT66.md','VERIFY_AUX66.json','PHYSICS_RESULTS.json','NATIVE66.npz','ORIGINAL_SOURCE_AUDIT66.json','ORIGINAL_OUTPUT_EVIDENCE66.csv','ORIGINAL_OUTPUT_PROGRESS66.csv','PLOT_INVENTORY66.csv','PROGRESS66.json','PLOT_MANIFEST66.json']:
 r0[f+'_sha256']=sha(S/f)
dag['restore66']=r0;dump(S/'DAG.json',dag)
v0=read(R/'PROVENANCE.json');v0['restore66']=dict(r0,dag_sha256=sha(S/'DAG.json'),parent=r0['parent_dag_sha256'],parent_commit='6340af6613074737daa30f95115648ce1d51be80',weights=read(S/'REGISTRATION.json')['weights']);dump(S/'PROVENANCE.json',v0)
print('VERIFIED',len(inv),'plot pairs;',len(union),'original-item overlay; DAG',sha(S/'DAG.json'))
