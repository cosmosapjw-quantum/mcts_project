from base61 import *
import csv
O=B/'mqft_restore63_20261126';P=B/'mqft_restore39_20261125/repo'
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for n in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/n)==sha(O/n)==sha(P/'docs'/n)
p=read(O/'PROGRESS63.json');assert sha(O/'PLOT_INVENTORY63.csv')==p['inventory_sha256']
inv=csvread(O/'PLOT_INVENTORY63.csv');assert len(inv)==493
for x in inv:
 base=B/x['stage']/x['plot'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=read(S/'PLOT_MANIFEST64.json')
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==501
write(S/'PLOT_INVENTORY64.csv',inv)
cat={x['id']:x for x in csvread(S/'ORIGINAL_TARGET_CATALOG.csv')};rows=csvread(O/'ORIGINAL_OUTPUT_PROGRESS63.csv');assert len(rows)==121
scope={
'L09':'Auxiliary actual-history conditional complex rho and S_vN with explicit phase/dephase controls; no inferred native physical decoherence.',
'L16':'Actual phase-instrument native path likelihood, selected baseline path pair lift; not physical path integral or continuum 1PI.',
'Q24':'Actual conditional auxiliary density evolution and coherence quadratures, chosen CPTP channel; no physical Darwinism claim.',
'Q41':'Separate calibrated auxiliary phase coupling, absolute/relative visibility match, original dimensionless proxy and unit-action floor; no physical hbar.',
'Q42':'Explicit rotation/dephasing Kraus and independent Lindblad ODE checks, heldout software X/Y tomography and native state readout; no measured physical bath.',
'Q43':'Actual full-support native phase-controlled selection, path law and actual selected two-path complex lift; not physical Feynman measure.',
'Q45':'Measured native phase contrast, paired mark null/uncertainty, independent software calibration and likelihood check; scoped report only.'}
ev=[]
for x in rows:
 k=x['id'];x['R64_evidence']=scope.get(k,'');x['R64_scope_status']='SCOPED_NATIVE_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in pm['figures'] if k in f['ids']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_PANEL',scope=scope[k]))
assert len(ev)==7 and all(x['evidence_figures'] for x in ev)
write(S/'ORIGINAL_OUTPUT_EVIDENCE64.csv',ev);write(S/'ORIGINAL_OUTPUT_PROGRESS64.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,65))]
assert len(union)==60 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
dag=read(O/'DAG.json');old={x['id']:x.copy() for x in dag['nodes']}
updates={
'X-REP':('Actual 6144 full-support trained native phase-controlled searches; auxiliary complex density/Choi/ODE and software quadrature calibration; exact real/dephase native control and measured visit shifts.','Physical phase/energy/time/hbar not identified; fresh conditional preparation, persistent instrument and independent utility next.'),
'REPLAN-03':('Actual controlled native path likelihood, independently calibrated model readout and 3072 selected same-root baseline path-pair lifts.','Physical action, continuum quantum path measure, 1PI and persistent phase dynamics unresolved.'),
'REPLAN-05':('Actual auxiliary conditional complex register and separate X/Y readouts; identical real/dephased native path law for selected readout.','Native physical measurement/record environment and redundancy identification remain.'),
'X-TRAIN':('Actual trained g0/g3 phase intervention on 9x9 full support, no new training.','No established mean mark benefit; complete-game/independent regret and 15x15 compatible transport remain.')}
for n in dag['nodes']:
 if n['id'] in updates:
  a,b=updates[n['id']];n['restore64_followup']=dict(evidence=a,remaining=b,artifact='mqft_restore64_20261126/{RESULTS_KO.md,PHYSICS_RESULTS.json,CALIBRATION64.json,AUDIT64.json,PATH_PAIRS64.json}')
assert len(dag['nodes'])==len(dag['edges'])==13 and all(x['execution_status']==old[x['id']]['execution_status'] for x in dag['nodes'])
p.update(figure_pairs=501,distinct_png_sha=501,files_paired=1002,current_pairs=8,prior_pairs=493,original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE64.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS64.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY64.csv'))
p['stages'][S.name]=8;p['original_output_evidence']=dict(current=7,R55_R64_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55–R64 overlays, not exhaustive historical evidence; no automatic final certification.')
dump(S/'PROGRESS64.json',p)
a=read(S/'AUDIT64.json');c=read(S/'CALIBRATION64.json');r=read(S/'PHYSICS_RESULTS.json');assert a['n']==6144 and a['backups']==786432 and a['real_dephased_path_matches']==1024 and a['real_dephased_native_law_exact']
dag.update(revision='RESTORE64_20261126',scope='Full-support native phase-sensitive auxiliary instrument, independent software quadratures and path-pair lift',previous_entry=dag['entry'],entry='X-REP',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE65-PERSISTENT-PHASE-INSTRUMENT-UTILITY')
r0=dict(status='scoped_actual_full_support_auxiliary_phase_instrument',parent_dag_sha256=sha(O/'DAG.json'),new_games=0,new_full_game_search_plies=0,fresh_fixed_root_histories=6144,total_marked_histories=49016,fresh_backups=786432,evaluator_calls=a['evaluations'],raw_gzip_bytes=a['raw_gzip_bytes'],conditional_wave_states=a['wave_records'],calibration_software_shots=c['shots'],native_state_software_readout_shots=sum(x['shots'] for x in r['readouts']),path_pairs=r['path_pairs']['n'],real_dephased_exact_paths=1024,original_plot_tasks=121,original_evidence_items=7,original_evidence_union_R55_R64=len(union),figures=8,retained_pairs=501,additional_plot_nodes=0,native_subagent='NOT_RUN',prior_admissions='preserved',next_contract='docs/NEXT_DYNAMICS.md')
for f in ['REGISTRATION.json','AUDIT64.json','RNG_AUDIT64.json','EVALUATOR64.json','LEGACY_REGRESSION64.json','CALIBRATION64.json','CALIBRATION64.npz','PHYSICS_RESULTS.json','PATH_PAIRS64.json','PHASE64.npz','ORIGINAL_OUTPUT_EVIDENCE64.csv','ORIGINAL_OUTPUT_PROGRESS64.csv','PLOT_INVENTORY64.csv','PROGRESS64.json','PLOT_MANIFEST64.json']:
 r0[f+'_sha256']=sha(S/f)
dag['restore64']=r0;dump(S/'DAG.json',dag)
v=read(O/'PROVENANCE.json');v['restore64']=dict(r0,dag_sha256=sha(S/'DAG.json'),parent='a236944c943c8119a7e4f4a6dec66542136ffb96',weights=read(S/'REGISTRATION.json')['weights']);dump(S/'PROVENANCE.json',v)
print('VERIFIED',len(inv),'plot pairs;',len(union),'original-item overlay; DAG',sha(S/'DAG.json'))
