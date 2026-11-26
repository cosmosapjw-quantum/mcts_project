from base61 import *
import csv
R=B/'mqft_restore66_20261126';P=B/'mqft_restore39_20261125/repo'
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/n)==sha(R/n)==sha(P/'docs'/n)
p=read(R/'PROGRESS66.json');assert sha(R/'PLOT_INVENTORY66.csv')==p['inventory_sha256']
inv=csvread(R/'PLOT_INVENTORY66.csv');assert len(inv)==520
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST67.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==528
write(S/'PLOT_INVENTORY67.csv',inv)
cat={x['id']:x for x in csvread(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=csvread(R/'ORIGINAL_OUTPUT_PROGRESS66.csv');assert len(rows)==121
scope={'L23':'Actual native sign transport errors, decoder correction success conditioned on syndrome; explicit auxiliary quantum X code and common/phase/coherent failures.', 'L21':'QEC subpanel only of original advanced quantum composite; actual native fault/auxiliary recovery evidence, gauge/topology not newly reproduced.', 'L12':'Scoped actual QEC cross-validation, independent PyTorch, high-budget and Rapfi reference comparison; not full original physics summary.'}
ev=[]
for x in rows:
 k=x['id'];x['R67_evidence']=scope.get(k,'');x['R67_scope_status']='SCOPED_NATIVE_AND_AUXILIARY_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];f=[v['figure'] for v in pm['figures'] if k in v['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_AND_AUXILIARY_PANEL',scope=scope[k]))
assert len(ev)==3 and all(x['evidence_figures'] for x in ev)
write(S/'ORIGINAL_OUTPUT_EVIDENCE67.csv',ev);write(S/'ORIGINAL_OUTPUT_PROGRESS67.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,68))]
assert len(union)==72 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(R/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={'X-STRUCT':('Actual evaluator sign fault and syndrome recovery, explicit restricted quantum code, coherent/correlated failure and actual independent reference.','General noise, hardware and matched 15x15+ full-game utility remain.'),'X-REP':('768 prior actual persistent states explicitly encoded, noisy, syndrome-decoded; coherent accumulation and Z failure retained.','No physical native phase or quantum hardware.'),'X-TRAIN':('Actual g0/g3 and pinned Rapfi external 9 fallback / 15 NNUE executed; 3200 trained native paths.','Size compatible 15x15 trained native full-game and independent utility next.'),'REPLAN-07':('Recovery improves noisy internal high-budget proxy, not established external g3 agreement.','No production admission or Elo, full paired games next.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore67_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore67_20261126/{RESULTS_KO.md,PHYSICS_RESULTS.json,AUDIT67.json,QEC67.json,STATE_RECOVERY67.json,RAPFI67.json}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=528,distinct_png_sha=528,files_paired=1056,current_pairs=8,prior_pairs=520,current_animations=0,original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE67.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS67.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY67.csv'))
p['stages'][S.name]=8;p['original_output_evidence']=dict(current=3,R55_R67_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R67 overlays; not exhaustive historical evidence or final certification.')
dump(S/'PROGRESS67.json',p)
a=read(S/'AUDIT67.json');assert a['n']==3200 and a['backups']==524288
dag.update(revision='RESTORE67_20261126',scope='Actual native evaluator sign transport fault recovery and explicit auxiliary quantum code; external engine independent comparison',previous_entry=dag['entry'],entry='X-TRAIN',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE68-SIZE-COMPATIBLE-PRETRAINED-ENGINE-ARENA')
r0=dict(status='scoped_actual_native_fault_recovery_and_auxiliary_QEC',parent_dag_sha256=sha(R/'DAG.json'),new_games=0,new_full_game_search_plies=0,fresh_fixed_root_histories=3200,total_marked_histories=60088,fresh_backups=524288,evaluator_calls=a['evaluations'],raw_gzip_bytes=a['raw_gzip_bytes'],auxiliary_noise_masks=6000000,prior_state_channel_applications=3840,external_queries=96,external_scope='9x9 non-NNUE fallback 64;15x15 center embedding verified NNUE 32',utility='g3 internal proxy improved versus single noisy; g3 independent Rapfi9 agreement not improved',original_plot_tasks=121,original_evidence_items=3,original_evidence_union_R55_R67=len(union),figures=8,new_animations=0,retained_pairs=528,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','AUDIT67.json','RNG_AUDIT67.json','EVALUATOR67.json','LEGACY_REGRESSION67.json','COMMON_CONTROL67.json','QEC67.json','STATE_RECOVERY67.json','RAPFI67.json','EXTERNAL_CONTRACT67.md','PHYSICS_RESULTS.json','NATIVE67.npz','REFERENCE67.npz','ORIGINAL_SOURCE_AUDIT67.json','ORIGINAL_OUTPUT_EVIDENCE67.csv','ORIGINAL_OUTPUT_PROGRESS67.csv','PLOT_INVENTORY67.csv','PROGRESS67.json','PLOT_MANIFEST67.json']:
 r0[f+'_sha256']=sha(S/f)
dag['restore67']=r0;dump(S/'DAG.json',dag)
v=read(R/'PROVENANCE.json');v['restore67']=dict(r0,dag_sha256=sha(S/'DAG.json'),parent_commit='30f231e9a81f0ca3ba8624b5d75a9e9f0f60d475',weights=read(S/'REGISTRATION.json')['weights']);dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'plot pairs;',len(union),'original-item overlay; DAG',sha(S/'DAG.json'))
