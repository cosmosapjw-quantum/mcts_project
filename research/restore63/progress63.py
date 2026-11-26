from base61 import *
import csv
O=B/'mqft_restore62_20261126';P=B/'mqft_restore39_20261125/repo'
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/n)==sha(O/n)==sha(P/'docs'/n)
p=read(O/'PROGRESS62.json');assert sha(O/'PLOT_INVENTORY62.csv')==p['inventory_sha256']
inv=csvread(O/'PLOT_INVENTORY62.csv');assert len(inv)==485
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST63.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==493
write(S/'PLOT_INVENTORY63.csv',inv)
cat={x['id']:x for x in csvread(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=csvread(O/'ORIGINAL_OUTPUT_PROGRESS62.csv');assert len(rows)==121
scope={
'G07':'Actual independent fresh-seed matrix martingale bracket and legacy source response, no physical FDT.',
'Q32':'Same fresh-seed residual and response; no critical point.',
'L16':'Exact native conditional multinomial logMGF and local conjugate, independently checked tilted transfer composition; not calibrated quantum action.',
'L38':'Actual finite wave action curvature and separate historical Rapfi 13–21 local-law transfer, not critical scaling.',
'L39':'Actual lag products of causal conditional innovations, not intrinsic relaxation.',
'L40':'Heldout multiwave conditional kernel prediction and count-consistent propagation; Markov sufficiency not established.',
'L41':'Exact local conditional count law versus actual heldout counts and fitted rest normalizer; separate Rapfi transfer.',
'L42':'Actual raw correlations and conditional innovation lag products, not physical memory law.',
'L43':'Time-dependent fitted joint count/next-state kernel and independent tilted operator propagation, heldout errors retained.',
'L44':'Observed lags, no physical time separation or exponential decay claim.',
'L48':'Scoped heldout generator/closure results, failed and uncertain tests retained.',
'L49':'Native local count calibration and heldout multiscale summary; not original quantum final certification.'}
ev=[]
for x in rows:
 k=x['id'];x['R63_evidence']=scope.get(k,'');x['R63_scope_status']='SCOPED_NATIVE_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in pm['figures'] if k in f['ids']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_PANEL',scope=scope[k]))
assert len(ev)==12 and all(x['evidence_figures'] for x in ev)
write(S/'ORIGINAL_OUTPUT_EVIDENCE63.csv',ev);write(S/'ORIGINAL_OUTPUT_PROGRESS63.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,64))]
assert len(union)==55 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(O/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={
'REPLAN-03':('Exact native frozen multinomial logMGF and local convex conjugate, verified tilted operator product; fresh conditional Fisher bracket and finite response recheck.','Individual vs simultaneous residual uncertainty, composed MGF error, physical 1PI and phase unresolved; next explicit phase instrument.'),
'REPLAN-04.R':('Time-dependent count-additive transition kernel and heldout composition; 5-size Rapfi conditional-law transfer.','Reduced-state closure not universal, no physical autonomous RG or trained size transfer.'),
'REPLAN-04.C':('Fresh independent same-geometry Fisher/legacy scalar source rechecks; no physical critical response inference.','Fixed-root and geometry uncertainty differ; critical scaling unresolved.'),
'X-MEMORY':('Actual heldout multiscale kernel and causal conditional innovation lag products; p composition beats occupancy 32/32, full-p freeze 5/32 point estimates.','Projected kernel not generally sufficient, MGF error retained; no physical memory time.'),
'X-REP':('Original Q41/Q43 source audit, explicit independent quadrature identification argument, no new native phase mechanism.','Next actual phase-sensitive instrument using R21/R27, independent energy/time calibration or scoped blocker.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore63_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore63_20261126/{RESULTS_KO.md,PHYSICS_RESULTS.json,KERNEL_RESULTS63.json,VERIFICATION63.json,RAPFI_TRANSFER63.json,ORIGINAL_QUANTUM_TARGET63.md}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=493,distinct_png_sha=493,files_paired=986,current_pairs=8,prior_pairs=485,original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE63.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS63.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY63.csv'))
p['stages'][S.name]=8;p['original_output_evidence']=dict(current=12,R55_R63_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R63 overlays, not exhaustive historical evidence; no automatic final certification.')
dump(S/'PROGRESS63.json',p)
a=read(S/'AUDIT63.json');v=read(S/'VERIFICATION63.json');assert a['n_new']==9216 and a['backups_new']==1179648 and a['n_reused']==1536 and len(v['operator'])==2
dag.update(revision='RESTORE63_20261126',scope='Native conditional count generator, martingale residual and heldout multiscale closure',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE64-PHASE-SENSITIVE-NATIVE-INSTRUMENT')
r=dict(status='scoped_native_conditional_generator_and_heldout_closure',parent_dag_sha256=sha(O/'DAG.json'),new_games=0,new_full_game_search_plies=0,fresh_fixed_root_histories=9216,total_marked_histories=42872,fresh_backups=1179648,evaluator_calls=1184714,raw_gzip_bytes=1450906245,reused_R62_histories=1536,reused_R58_Rapfi_histories=180,heldout_geometry_blocks=64,heldout_waves=65536,fisher_residual='fresh fixed-root bracket and matrix simultaneous intervals include zero; individual geometry bootstrap and combined individual intervals exclude zero',legacy_response='prior discrepancy not reproduced by new seeds, finite h uncertainty remains',closure='p kernel beats occupancy 32/32 and frozen full p 5/32 NLL point estimates; MGF discrepancy retained',original_plot_tasks=121,original_evidence_items=12,original_evidence_union_R55_R63=len(union),figures=8,retained_pairs=493,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','AUDIT63.json','RNG_AUDIT63.json','PHYSICS_RESULTS.json','KERNEL_RESULTS63.json','MODEL_LOCK63.json','VERIFICATION63.json','RAPFI_TRANSFER63.json','FINAL63.npz','WAVES63.npz','ORIGINAL_QUANTUM_TARGET63.md','ORIGINAL_OUTPUT_EVIDENCE63.csv','ORIGINAL_OUTPUT_PROGRESS63.csv','PLOT_INVENTORY63.csv','PROGRESS63.json','PLOT_MANIFEST63.json']:
 r[f+'_sha256']=sha(S/f)
dag['restore63']=r;dump(S/'DAG.json',dag)
v=read(O/'PROVENANCE.json');v['restore63']=dict(r,dag_sha256=sha(S/'DAG.json'),parent='0c597cc0520322cd305ff27c25a36c25088a31cf',weights=read(S/'REGISTRATION.json')['weights']);dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'plot pairs;',len(union),'original-item overlay; DAG',sha(S/'DAG.json'))
