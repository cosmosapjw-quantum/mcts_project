from base61 import *
import csv
from PIL import Image
R=B/'mqft_restore68_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/n)==sha(R/n)==sha(P/'docs'/n)
p=read(R/'PROGRESS68.json');assert sha(R/'PLOT_INVENTORY68.csv')==p['inventory_sha256'];inv=cr(R/'PLOT_INVENTORY68.csv');assert len(inv)==536
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST69.json')
for k,h in pm['sources'].items():assert sha(S/k)==h
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==543;cw(S/'PLOT_INVENTORY69.csv',inv)
for x in pm['animations']:
 f=S/'plots'/x['name'];assert sha(f)==x['sha256'] and f.stat().st_size==x['bytes']
 with Image.open(f) as im:
  assert im.n_frames==x['frames']
  for k in range(im.n_frames):im.seek(k);im.load()
assert len(pm['animations'])==1
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=cr(R/'ORIGINAL_OUTPUT_PROGRESS68.csv');assert len(rows)==121
scopes={
'L11':'System-fragment entanglement of explicit complementary instrument; not branch entanglement.',
'L15':'Pointer label information and fragment readout threshold, not actual final decision redundancy.',
'L30':'Finite conditional record information and disjoint packing; no robust Darwinism plateau.',
'Q24':'Exact nonselective decoherence/coherence and entropy of conditional instrument; not physical environment.',
'Q25':'Exact record fragment information and readout; pointer versus final decision distinction.',
'Q26':'Declared rotated pointer basis and 90% record information; no unique physical pointer identification.',
'Q27':'Four-frame actual conditional instrument fragment evolution; no physical time calibration.',
'Q28':'Scoped decoherence/record information panels, not whole original summary.',
'Q35':'Explicit S:F negativity and concurrence; not entanglement of game-tree branches.',
'Q37':'Quantum MI and fixed-pointer information gap, not optimized discord.',
'G13':'Exact system pointer coherence after recorded conditional draws.',
'G14':'Exact finite-record von Neumann/linear entropy, not thermodynamic entropy.',
'G16':'Pointer information for fixed instrument; no emergent unique physical basis.',
'G17':'Frozen realized score schedules and noncommuting memory controls; not adaptive multi-time path law.',
'G23':'Record-partition quantum MI matrix, not tree-branch matrix.',
'G24':'Record-partition information network with exact zero R:R negativity; not entangled branch network.',
'G25':'System-fragment concurrence over retained record count, not native branch concurrence.',
'G26':'Optimal CHSH in explicit coherent two-qubit compression, not physical Bell experiment.',
'G27':'Scoped quantum record model, information/entanglement distinctions; not full original summary.'}
ev=[]
for x in rows:
 k=x['id'];x['R69_evidence']=scopes.get(k,'');x['R69_scope_status']='SCOPED_MODEL_PANEL' if k in scopes else ''
 if k in scopes:
  y=cat[k];f=[v['figure'] for v in pm['figures'] if k in v['ids']]+[v['name'] for v in pm['animations'] if k in v['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_MODEL_PANEL',scope=scopes[k]))
assert len(ev)==19 and all(x['evidence_figures'] for x in ev);cw(S/'ORIGINAL_OUTPUT_EVIDENCE69.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS69.csv',rows)
prior=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,69))]
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,70))]
assert len(prior)==76 and len(union)==90 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(R/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={
'REPLAN-05':('Exact complementary record law, Holevo, actual basis MI, explicit disjoint 90% fragment counts; no robust pointer redundancy.','Final decision D distinct; actual adaptive multi-time record and disjoint final-decision redundancy remain.'),
'X-REP':('Exact complex S:F states, negativity/concurrence and optimal model CHSH; independent matrix checks.','No physical Bell experiment or native unique coherent extension.'),
'X-MEMORY':('Analytic separable adjacent-record pair for any intervening CPTP map, actual score schedules frozen for numerical controls.','Adaptive joint memory/record dynamics on actual alternative trajectories.'),
'REPLAN-03':('Exact finite-record influence action and controlled quadratic remainder.','Actual adaptive marked W[J], tangent 1PI and independently controlled saddle/loop expansion next.'),
'REPLAN-07':('Prior-informed optimization proposal superseded; existing negative/limited arena outcomes retained.','Application-only; physics-derived intervention vs same evaluator conventional baseline MCTS only.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore69_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore69_20261126/{RESULTS_KO.md,THEORY.md,PHYSICS_RESULTS.json,INDEPENDENT69.json,BLOCK_RESULTS69.json}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=543,distinct_png_sha=543,files_paired=1086,current_pairs=7,prior_pairs=536,current_animations=1,animation_scope='One fresh four-frame GIF; cumulative historical animations not exhaustively counted.',original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE69.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS69.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY69.csv'))
p['stages'][S.name]=7;p['original_output_evidence']=dict(current=19,R55_R69_union=len(union),new_scoped_ids=sorted(set(union)-set(prior)),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R69 overlays; scoped native/model correspondences only, not full historical evidence or final certification.')
dump(S/'PROGRESS69.json',p)
a=read(S/'AUDIT69.json');assert a['native_paths']==256 and a['rapfi_paths']==32 and a['backups']==36864
dag.update(revision='RESTORE69_20261126',scope='Physics-first charter reset; exact complementary instrument records and finite influence action',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE70-ADAPTIVE-MARKED-EFFECTIVE-ACTION',active_research_charter='docs/RESEARCH_CHARTER.md',mainline_priorities=['R70 adaptive marked effective action and controlled saddle','R71 graph RG and genuine size-family closure','R72 complex adaptive multi-time records and decision information'],application_policy='same-evaluator MQFT-on vs conventional baseline MCTS; conditional application only')
z=dict(status='scoped_complementary_record_physics',parent_dag_sha256=sha(R/'DAG.json'),fresh_fixed_root_histories=288,total_marked_histories=60376,fresh_backups=a['backups'],evaluator_calls=a['evals'],raw_gzip_bytes=a['raw_bytes'],fresh_native_conditional_waves=8192,reused_R65_waves=49152,rapfi_auxiliary_waves=1024,frozen_crosswave_pair_modes=35712,new_games=0,optimizer_updates=0,original_plot_tasks=121,original_evidence_items=19,original_evidence_union_R55_R69=90,figures=7,new_animations=1,retained_pairs=543,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md',prior_optimization='DEFERRED')
for f in ['REGISTRATION.json','AUDIT69.json','PHYSICS_RESULTS.json','INDEPENDENT69.json','BLOCK_RESULTS69.json','RNG_AUDIT69.json','ORIGINAL_SSOT_RECEIPT69.json','ORIGINAL_ARCHIVE_MEMBERS69.json','ORIGINAL_SOURCE_AUDIT69.json','HISTORY_CENSUS69.json','GIT_HISTORY69.json','RESEARCH_CHARTER.md','HISTORICAL_RECONCILIATION69.md','PLOT_MANIFEST69.json','PLOT_INVENTORY69.csv','ORIGINAL_OUTPUT_EVIDENCE69.csv','ORIGINAL_OUTPUT_PROGRESS69.csv','PROGRESS69.json']:
 z[f+'_sha256']=sha(S/f)
dag['restore69']=z;dump(S/'DAG.json',dag);v=read(R/'PROVENANCE.json');v['restore69']=dict(z,dag_sha256=sha(S/'DAG.json'),parent_commit=read(S/'REGISTRATION.json')['parent'],external=read(S/'REGISTRATION.json')['external']);dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'pairs',len(union),'scoped original items; new',sorted(set(union)-set(prior)),'DAG',sha(S/'DAG.json'))
