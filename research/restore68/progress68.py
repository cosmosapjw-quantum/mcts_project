from base61 import *
import csv
from PIL import Image
R=B/'mqft_restore67_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/n)==sha(R/n)==sha(P/'docs'/n)
p=read(R/'PROGRESS67.json');assert sha(R/'PLOT_INVENTORY67.csv')==p['inventory_sha256'];inv=cr(R/'PLOT_INVENTORY67.csv');assert len(inv)==528
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST68.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==536;cw(S/'PLOT_INVENTORY68.csv',inv)
for x in pm['animations']:
 f=S/'plots'/x['name'];assert sha(f)==x['sha256'] and f.stat().st_size==x['bytes']
 with Image.open(f) as im:
  assert im.n_frames==x['frames']
  for k in range(im.n_frames):im.seek(k);im.load()
assert len(pm['animations'])==2
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=cr(R/'ORIGINAL_OUTPUT_PROGRESS67.csv');assert len(rows)==121
scope={'L12':'Scoped actual complete-game cross-validation with independent search, exact NNUE model identity and failures retained. Not full original summary.','L23':'Actual 15x15 native evaluator sign transport error and recovery, complete-game comparison; not physical QEC hardware.','Q33':'Actual fixed L15 spatial connected visit-field correlation over five backup cuts and verified GIF; no multi-size or equilibrium exponent.','Q36':'Actual L15 entropy versus search budget and game clock only; original area/volume and system-size scaling remain.','Q37':'Actual classical conditional selection entropy and empirical wave-index mixture information; not quantum MI or discord.','Q39':'Actual native game-ply visit entropy animation with changing at-risk counts; not physical thermodynamic or entanglement entropy.'}
ev=[]
for x in rows:
 k=x['id'];x['R68_evidence']=scope.get(k,'');x['R68_scope_status']='SCOPED_NATIVE_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];f=[v['figure'] for v in pm['figures'] if k in v['ids']]+[v['name'] for v in pm['animations'] if k in v['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_PANEL',scope=scope[k]))
assert len(ev)==6 and all(x['evidence_figures'] for x in ev);cw(S/'ORIGINAL_OUTPUT_EVIDENCE68.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS68.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,69))]
assert len(union)==76 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(R/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={
'X-TRAIN':('Actual direct 15x15 pretrained MixNet policy/WDL with 78 complete games and 2 incomplete external responses, native and Rapfi color-swapped arena.','Strong independent opponent, no proven recovery strength, prior-informed root law heldout and compute matching next.'),
'REPLAN-01':('New actual complete-game trace path on 15x15, all 80 receipts replayed and external NNUE pinned.','No complete strength certification; two response failures retained.'),
'REPLAN-07':('Actual recovery vs noisy and clean complete-game comparison, limited 8 opening blocks; first-wave uniform diagnostic.','No general utility admission; prior-informed stochastic root law and conventional PUCT comparator.'),
'REPLAN-04.C':('Actual L15 connected spatial visit-field correlation across five budget cuts, original Q33 animation.','Multi-size correlation length and critical scaling unresolved.'),
'REPLAN-05':('Actual L15 classical visit/conditional entropy and empirical wave-index MI, Q36/Q37/Q39 scoped.','Entanglement, quantum discord, across-size entropy/records remain.'),
'X-STRUCT':('Actual sign transport recovery on 15x15 complete-game traces, same evaluator and cost proxy.','No physical hardware fault tolerance or general utility claim.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore68_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore68_20261126/{RESULTS_KO.md,PHYSICS_RESULTS.json,AUDIT68.json,DIAGNOSTICS68.json,TACTICAL68.json}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=536,distinct_png_sha=536,files_paired=1072,current_pairs=8,prior_pairs=528,current_animations=2,original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE68.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS68.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY68.csv'))
p['stages'][S.name]=8;p['original_output_evidence']=dict(current=6,R55_R68_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R68 overlays; not exhaustive historical evidence or final certification.')
dump(S/'PROGRESS68.json',p)
a=read(S/'AUDIT68.json');assert a['games']==80 and a['complete']==78 and a['native_searches']==853 and a['backups']==54592
cp=sum(len(x['native']) for x in a['rows'] if x['status']=='complete')
dag.update(revision='RESTORE68_20261126',scope='Actual 15x15 pretrained direct NNUE arena, fault recovery and native game-temporal observables',previous_entry=dag['entry'],entry='REPLAN-07',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE69-PRIOR-INFORMED-ROOT-GENERATOR')
z=dict(status='scoped_15x15_actual_complete_game_arena',parent_dag_sha256=sha(R/'DAG.json'),registered_games=80,new_games=78,incomplete_games=2,new_full_game_search_plies=cp,native_searched_plies_including_incomplete=853,fresh_fixed_root_histories=0,total_marked_histories=60088,fresh_backups=54592,evaluator_calls=a['evals'],raw_gzip_bytes=a['raw_bytes'],external_moves=a['external_moves'],utility='limited paired results not established; cold-start uniform despite peaked NNUE prior; immediate win and unique defense all selected',original_plot_tasks=121,original_evidence_items=6,original_evidence_union_R55_R68=76,figures=8,new_animations=2,retained_pairs=536,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','CAMPAIGN68.json','AUDIT68.json','DIAGNOSTICS68.json','PHYSICS_RESULTS.json','TACTICAL68.json','SPATIAL68.json','INFORMATION68.json','ORIGINAL_SOURCE_AUDIT68.json','PLOT_MANIFEST68.json','PLOT_INVENTORY68.csv','ORIGINAL_OUTPUT_EVIDENCE68.csv','ORIGINAL_OUTPUT_PROGRESS68.csv','PROGRESS68.json']:
 z[f+'_sha256']=sha(S/f)
dag['restore68']=z;dump(S/'DAG.json',dag);v=read(R/'PROVENANCE.json');v['restore68']=dict(z,dag_sha256=sha(S/'DAG.json'),parent_commit='f52b593f6a3ed5682b3b583b927ac865ea594d0e',external=read(S/'REGISTRATION.json')['external']);dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'pairs',len(union),'original items; complete game native plies',cp,'DAG',sha(S/'DAG.json'))
