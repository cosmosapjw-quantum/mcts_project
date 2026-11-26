from base61 import *
import csv
P=B/'mqft_restore39_20261125/repo'
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/n)==sha(O/n)==sha(P/'docs'/n)
p=read(O/'PROGRESS60.json');assert sha(O/'PLOT_INVENTORY60.csv')==p['inventory_sha256']
inv=csvread(O/'PLOT_INVENTORY60.csv');assert len(inv)==468
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST61.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==476
write(S/'PLOT_INVENTORY61.csv',inv)
cat={x['id']:x for x in csvread(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=csvread(O/'ORIGINAL_OUTPUT_PROGRESS60.csv');assert len(rows)==121
scope={
'G01':'Actual trained g0–g3 negative mean root mark vs completed backup clock at 48 matched openings; no physical energy.',
'G03':'Actual native visit/prior Shannon entropy distinguished on matched learned checkpoints, not thermodynamic entropy.',
'L04':'Actual trained finite-budget prefix evolution; no physical scaling exponent.',
'G07':'Native adaptive source count finite contrast, likelihood score response, Fisher and variance decomposition; no thermal FDT.',
'Q32':'Same actual measured susceptibility as G07, registered independent roots, h±.35; no critical peak or equilibrium identification.',
'L16':'Actual normalized path action, logit response, empirical count CGF and restricted Legendre; D4 readout/path control; not calibrated quantum action or continuum 1PI.',
'L13':'Actual trained native finite-budget trajectory in depth, support, entropy and mark statistics, not autonomous RG.',
'L29':'Actual trained native finite-budget flow; no physical coupling-space vector field.',
'L35':'Measured finite-budget trajectory in H–variance plane; no physical RG trajectory.',
'L36':'Two-dimensional measured H–variance flow, no assumed field or fixed point.',
'L38':'Actual native scale-dependent observables at k=4…128, no critical exponent.',
'Q09':'Measured dyadic finite differences of native H and E_Q against log B; not autonomous RG beta-functions.',
'Q11':'Measured H–variance trajectories at fixed registered sampler law and checkpoint, not a physical RG phase diagram.'
}
ev=[]
for x in rows:
 k=x['id'];x['R61_evidence']=scope.get(k,'');x['R61_scope_status']='SCOPED_NATIVE_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in pm['figures'] if k in f['ids']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_PANEL',scope=scope[k]))
assert len(ev)==13 and all(x['evidence_figures'] for x in ev)
write(S/'ORIGINAL_OUTPUT_EVIDENCE61.csv',ev);write(S/'ORIGINAL_OUTPUT_PROGRESS61.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,62))]
assert len(union)==55 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(O/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={
'REPLAN-03':('Actual trained g0/g3 native source response, score/Fisher/path likelihood, empirical Legendre; 28 D4 coupling order witnesses.','Coordinate-covariant action coupling and multi-source path Hessian; g0 serial importance-reweight discrepancy; native continuum 1PI and phase identification unresolved.'),
'REPLAN-04.R':('Actual 63438 old native prefix records and 20992 new prefix records, measured log-budget increments and H–variance trajectories.','Independent conditional transition closure and calibrated coarse generator; no autonomous RG or fixed point.'),
'REPLAN-04.C':('Trained finite native source-count response; exact score/drift/variance split, Fisher distinction; positive direct response and unresolved checkpoint contrast.','New geometry/source strata and independent h-grid; no physical critical exponents or equilibrium FDT.'),
'REPLAN-02':('Matched trained native prior versus visit entropy and negative mean root mark; different entropy trend under new stochastic selector/design.','Physical temperature and equilibrium entropy identification unresolved.'),
'X-TRAIN':('R60 trained weights reused and D4 group average readout exact in tested fixtures; no new optimizer updates.','Pathwise D4 coupling, original 15x15 architecture and independent physical labels remain.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore61_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore61_20261126/{RESULTS_KO.md,PHYSICS_RESULTS.json,DIAGNOSTICS61.json,CONTROL_AUDIT61.json}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=476,distinct_png_sha=476,files_paired=952,current_pairs=8,prior_pairs=468,original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE61.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS61.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY61.csv'))
p['stages'][S.name]=8
p['original_output_evidence']=dict(current=13,R55_R61_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R61 overlays, not exhaustive historical evidence; no automatic final certification.')
dump(S/'PROGRESS61.json',p)
a=read(S/'CONTROL_AUDIT61.json');b=read(S/'R60_REDUCTION61.json');d=read(S/'DIAGNOSTICS61.json')
assert a['n']==3584 and a['backups']==425984 and b['n_rows']==63438 and d['first_divergences']==28
dag.update(revision='RESTORE61_20261126',scope='Trained native source response, finite action/flow and D4 path-coupling diagnosis',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE62-COVARIANT-PATH-AND-CONDITIONAL-CLOSURE')
r=dict(status='scoped_trained_native_controlled_response_and_D4_readout',parent_dag_sha256=sha(O/'DAG.json'),new_games=0,new_full_game_search_plies=0,prior_full_game_searches_reduced=11956,prior_full_game_backups_reduced=1132064,prior_prefix_records=63438,fresh_fixed_root_histories=3584,total_marked_histories=13208,fresh_backups=425984,evaluator_calls=426797,raw_bytes=1883628991,new_prefix_records=20992,response_root_blocks=32,response_seeds_per_block=8,response_histories=3072,symmetry_histories=512,d4_inference_equivariance='zero mapped residual on tested fixtures',d4_path_coupling='28/224 first divergences all equal-prior ordering; marginal root law same at divergence',learning_source_response='not resolved',original_plot_tasks=121,original_evidence_items=13,original_evidence_union_R55_R61=len(union),figures=8,retained_pairs=476,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','R60_REDUCTION61.json','CONTROL_AUDIT61.json','RNG_AUDIT61.json','EVALUATOR_AUDIT61.json','LEGACY_INFERENCE61.json','PHYSICS_RESULTS.json','DIAGNOSTICS61.json','BUDGET_FLOW61.json','CONTROL61.npz','R60_PREFIX61.npz','ORIGINAL_OUTPUT_EVIDENCE61.csv','ORIGINAL_OUTPUT_PROGRESS61.csv','PLOT_INVENTORY61.csv','PROGRESS61.json','PLOT_MANIFEST61.json']:
 r[f+'_sha256']=sha(S/f)
dag['restore61']=r
dump(S/'DAG.json',dag)
v=read(O/'PROVENANCE.json');v['restore61']=dict(r,dag_sha256=sha(S/'DAG.json'),parent='be9fc78107d78d7e9c2d8d9806de7467a6f7a40b',weights=read(S/'REGISTRATION.json')['weights'])
dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'plot pairs;',len(union),'original-item overlay; DAG',sha(S/'DAG.json'))
