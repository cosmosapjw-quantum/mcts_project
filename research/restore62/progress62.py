from base61 import *
import csv
O=B/'mqft_restore61_20261126';P=B/'mqft_restore39_20261125/repo'
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/n)==sha(O/n)==sha(P/'docs'/n)
p=read(O/'PROGRESS61.json');assert sha(O/'PLOT_INVENTORY61.csv')==p['inventory_sha256']
inv=csvread(O/'PLOT_INVENTORY61.csv');assert len(inv)==476
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST62.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==485
write(S/'PLOT_INVENTORY62.csv',inv)
cat={x['id']:x for x in csvread(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=csvread(O/'ORIGINAL_OUTPUT_PROGRESS61.csv');assert len(rows)==121
scope={
'G07':'Actual two-source path response matrix, score and direct interventions; no physical susceptibility or equilibrium FDT.',
'Q32':'Actual multivariate source response, score dispersion and independent legacy scalar replication; no critical point.',
'L16':'Actual conditional and mixture finite empirical count action/curvature, path response and D4 transport; not physical loop or continuum 1PI.',
'L38':'Actual constrained matrix spectra k=4..128, not physical critical exponents.',
'L40':'Heldout predictive count-fraction closure; no Markov sufficiency claim.',
'L41':'Exact last-wave conditional estimator and measured baseline/RB dispersion; not physical analytical bath model.',
'L43':'Actual heldout binned count-fraction transition matrices 64 to 128; not stationary Markov kernel.',
'L48':'Heldout closure and actual negative results preserved, partial summary.',
'L49':'Native heldout closure, D4 pathwise and response diagnostic summary; original quantum summary not fully restored.'
}
ev=[]
for x in rows:
 k=x['id'];x['R62_evidence']=scope.get(k,'');x['R62_scope_status']='SCOPED_NATIVE_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in pm['figures'] if k in f['ids']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_PANEL',scope=scope[k]))
assert len(ev)==9 and all(x['evidence_figures'] for x in ev)
write(S/'ORIGINAL_OUTPUT_EVIDENCE62.csv',ev);write(S/'ORIGINAL_OUTPUT_PROGRESS62.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,63))]
assert len(union)==55 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(O/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={
'REPLAN-03':('Native multi-source matrix response and exact path Hessian; canonical D4 path transport 280/280; finite empirical Legendre with conditional and mixture measure distinguished.','g0/w4 Fisher residual and legacy finite FD/centered contrast; native calibrated physical 1PI and phase unresolved.'),
'REPLAN-04.R':('Actual finite-budget constrained matrix spectra and heldout dyadic count transitions, conditional versus mixture empirical curvature.','Need conditional kernel composition and calibrated coarse generator; no autonomous physical RG.'),
'REPLAN-04.C':('Actual native 64-root 2-source response; scalar h-grid and independent legacy replication.','No physical critical exponent or FDT; independent matrix uncertainty and scaling.'),
'X-MEMORY':('32/32 geometry train/test split: occupancy closure wins 6 of 8 strata; mark or count ridge each wins one, history not robust.','Predictive composition, no Markov sufficiency or physical memory kernel admitted.'),
'X-TRAIN':('Existing trained g0/g3 unchanged; actual canonical-frame coupling in D4 witness suite.','15x15 trained architecture and physical labels, runtime uniform frame sampler for nontrivial stabilizers remain.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore62_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore62_20261126/{RESULTS_KO.md,PHYSICS_RESULTS.json,LEGENDRE62.json,AUDIT62.json,SYMMETRY62.json}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=485,distinct_png_sha=485,files_paired=970,current_pairs=9,prior_pairs=476,original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE62.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS62.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY62.csv'))
p['stages'][S.name]=9;p['original_output_evidence']=dict(current=9,R55_R62_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R62 overlays, not exhaustive historical evidence; no automatic final certification.')
dump(S/'PROGRESS62.json',p)
a=read(S/'AUDIT62.json');d=read(S/'SYMMETRY62.json');assert a['n']==20448 and a['backups']==2586624 and a['rows']==122208 and d['all_exact']
dag.update(revision='RESTORE62_20261126',scope='Canonical trained native multi-source response, finite local action and heldout closure',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE63-CONDITIONAL-GENERATOR-AND-ORIGINAL-PHYSICS')
r=dict(status='scoped_native_multisource_path_and_d4_transport',parent_dag_sha256=sha(O/'DAG.json'),new_games=0,new_full_game_search_plies=0,fresh_fixed_root_histories=20448,total_marked_histories=33656,fresh_backups=2586624,evaluator_calls=2592077,raw_gzip_bytes=3606115280,new_prefix_records=122208,response_root_blocks=64,response_seeds_per_block=12,response_histories=18432,legacy_histories=1536,symmetry_histories=480,d4_path_transport='224 unique plus 56 stabilizer frame multiset comparisons exact; previous 28 witnesses included',legacy_regression=6,learning_source_response='not resolved',fisher_residual='g0/w4 aa interval excludes zero; finite sampling discrepancy retained',closure='occupancy wins 6/8, mark and count each 1/8; no Markov sufficiency',original_plot_tasks=121,original_evidence_items=9,original_evidence_union_R55_R62=len(union),figures=9,retained_pairs=485,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','AUDIT62.json','RNG_AUDIT62.json','EVALUATOR62.json','LEGACY_REGRESSION62.json','PHYSICS_RESULTS.json','LEGENDRE62.json','EXACT62.json','SYMMETRY62.json','TRANSITIONS62.json','CONTROL62.npz','ORIGINAL_OUTPUT_EVIDENCE62.csv','ORIGINAL_OUTPUT_PROGRESS62.csv','PLOT_INVENTORY62.csv','PROGRESS62.json','PLOT_MANIFEST62.json']:
 r[f+'_sha256']=sha(S/f)
dag['restore62']=r;dump(S/'DAG.json',dag)
v=read(O/'PROVENANCE.json');v['restore62']=dict(r,dag_sha256=sha(S/'DAG.json'),parent='16f292a5baf4849d14265735058a44e3f84cb112',weights=read(S/'REGISTRATION.json')['weights']);dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'plot pairs;',len(union),'original-item overlay; DAG',sha(S/'DAG.json'))
