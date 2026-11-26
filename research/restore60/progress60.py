import csv,json,hashlib
from pathlib import Path
S=Path(__file__).resolve().parent;B=S.parent;O=B/'mqft_restore59_20261126';P=B/'mqft_restore39_20261125/repo'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/n)==sha(O/n)==sha(P/'docs'/n)
p=read(O/'PROGRESS59.json');assert sha(O/'PLOT_INVENTORY59.csv')==p['inventory_sha256']
inv=csvread(O/'PLOT_INVENTORY59.csv');assert len(inv)==463
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST60.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==468
write(S/'PLOT_INVENTORY60.csv',inv)
cat={x['id']:x for x in csvread(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=csvread(O/'ORIGINAL_OUTPUT_PROGRESS59.csv');assert len(rows)==121
scope={'L46':'Original-format metric series from actual three-generation self-play, 1400 optimizer updates, held-out losses, LR and measured matched arena; no global Elo or GPU fields.',
'L47':'Actual iterative training curves, matched arena, resource/game stats, 384 common-root native observables. 9x9 early-opening CPU scope, original ResNet and thermodynamic heads not recovered; no final certification.'}
ev=[]
for x in rows:
 k=x['id'];x['R60_evidence']=scope.get(k,'');x['R60_scope_status']='SCOPED_NATIVE_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in pm['figures'] if k in f['ids']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_PANEL',scope=scope[k]))
assert len(ev)==2 and all(x['evidence_figures'] for x in ev)
write(S/'ORIGINAL_OUTPUT_EVIDENCE60.csv',ev);write(S/'ORIGINAL_OUTPUT_PROGRESS60.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,61))]
assert len(union)==52 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(O/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
n=next(x for x in dag['nodes'] if x['id']=='X-TRAIN')
n['restore60_prior_status']={k:n[k] for k in ('execution_status','latest_evidence','remaining_scope','next_action')}
n.update(execution_status='scoped_actual_iterative_selfplay_policy_value_three_generations_matched_arena_verified',latest_evidence='mqft_restore60_20261126/{THEORY.md,RESULTS_KO.md,REGISTRATION.json,SCALE_REGISTRATION.json,AUDIT60.json,TRAINING1.json,TRAINING2.json,TRAINING3.json,TRAINING_METRICS_ORIGINAL_FORMAT60.json,PHYSICS_RESULTS.json,PLOT_MANIFEST60.json}',remaining_scope='9x9 early-opening single orientation CPU only; 15x15 and original 19-plane network, D4 controlled inference, trained wave/GPU, identified thermodynamic/phase heads and general strength remain unresolved; original R23 fixed-teacher/negative complex-feature evidence retained',next_action='use matched trained native traces for original physics; controlled source/wave path and D4 control; no generic further training before identified objective')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x==old[x['id']] for x in dag['nodes'] if x['id']!='X-TRAIN')
p.update(figure_pairs=468,distinct_png_sha=468,files_paired=936,current_pairs=5,prior_pairs=463,original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE60.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS60.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY60.csv'))
p['stages'][S.name]=5
p['DAG']['node_status']['X-TRAIN']=n['execution_status'];p['DAG']['execution_status_prefix']={'complete':1,'scoped':7,'partial':5}
p['original_output_evidence']=dict(current=2,R55_R60_union=52,baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R60 overlays, not exhaustive historical evidence; no automatic final certification.')
dump(S/'PROGRESS60.json',p)
a=read(S/'AUDIT60.json');assert a['total']['games']==1592 and a['total']['backups']==1132064 and a['matched_initial_roots']==384
dag.update(revision='RESTORE60_20261126',scope='Actual iterative policy/value learning and matched trained native observables',previous_entry=dag['entry'],entry='X-TRAIN',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE61-TRAINED-NATIVE-PHYSICS')
r=dict(status='scoped_actual_three_generation_feedback_and_matched_arena',parent_dag_sha256=sha(O/'DAG.json'),new_games=1592,selfplay_games=760,arena_games=832,new_full_game_search_plies=11956,fresh_backups=1132064,evaluator_calls=871025,raw_bytes=621081076,optimizer_updates=1400,independent_replay_updates=800,training_positions=4542,heldout_positions=1124,matched_initial_roots=384,total_marked_histories=9624,fresh_fixed_root_histories=0,original_plot_tasks=121,original_evidence_items=2,original_evidence_union_R55_R60=52,figures=5,retained_pairs=468,additional_plot_nodes=0,native_subagent='NOT_RUN',g3_g0_B128_score=.78125,prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','SCALE_REGISTRATION.json','REGISTRATION_CORRECTION.json','BRIDGE_SOURCE60.json','AUDIT60.json','OPTIMIZER_REPLAY60.json','PHYSICS_RESULTS.json','MATCHED_ROOTS60.json','TRAINING_METRICS_EXPORT60.json','TRAINING_METRICS_ORIGINAL_FORMAT60.json','ORIGINAL_OUTPUT_EVIDENCE60.csv','ORIGINAL_OUTPUT_PROGRESS60.csv','PLOT_INVENTORY60.csv','PROGRESS60.json','PLOT_MANIFEST60.json']:
 r[f+'_sha256']=sha(S/f)
dag['restore60']=r
dump(S/'DAG.json',dag)
v=read(O/'PROVENANCE.json');v['restore60']=dict(r,dag_sha256=sha(S/'DAG.json'),parent='7bd48b692fbfcb3ed89bbe1dee052b5a5445b3cf',checkpoint_lineage=a['lineage'])
dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'plot pairs;',len(union),'original-item overlay; DAG',sha(S/'DAG.json'))
