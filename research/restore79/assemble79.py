"""Assemble authenticated nine-panel L31 report and item-specific convergence matrix."""
from base61 import *
import csv
I=read(S/'INVENTORY_AUDIT79.json');V=read(S/'INDEPENDENT79.json');A=read(S/'AUDIT79.json');C=read(S/'CALIBRATION79.json');L=read(S/'EVIDENCE_LINKS79.json');O=read(S/'ORIGINAL_SOURCE79.json')
assert V['status']=='PASS_SCOPED' and I['status']=='PASS_SCOPED'
inputs=I['summary_inputs'].copy()
for f in ('CALIBRATION79.json','AUDIT79.json','INDEPENDENT79.json','INVENTORY_AUDIT79.json','ORIGINAL_SOURCE79.json','EVIDENCE_LINKS79.json','acceptance79.tsv'):inputs[str((S/f).relative_to(B))]=sha(S/f)
def get(f):
 p=next(B.glob(f'mqft_restore{f[0]}_*'))/f[1];assert sha(p)==inputs[str(p.relative_to(B))];return read(p)
Q=get((76,'SUMMARY76.json'));X=get((76,'ORIGINAL_CLAIM_AUDIT76.json'));R=get((77,'SUMMARY77.json'));RGA=get((77,'ORIGINAL_RG_AUDIT77.json'));D=get((69,'PHYSICS_RESULTS.json'));F=get((66,'FEEDBACK66.json'));T=get((74,'PETZ74.json'));E=get((66,'ERASURE66.json'));W=get((73,'DECISION_RESULTS73.json'))
z=next(x for x in D['summaries'] if x['dataset']=='fresh_R69' and x['group']=='all');stats=z['stats']
steps=sorted({x['steps'] for x in F['results']});work=[]
for k in steps:
 q=[x for x in F['results'] if x['steps']==k];e=[x for x in E['results'] if x['steps']==k]
 work.append(dict(steps=k,n_protocols=len(q),information=float(np.mean([x['information'] for x in q])),beta_extracted_work=float(-(10/3)*np.mean([x['mean_work_exact'] for x in q])),beta_sample_extracted_work=float(-(10/3)*np.mean([x['mean_work_sample'] for x in q])),beta_erasure_work=float((10/3)*np.mean([x['mean_work_exact'] for x in e])),erasure_entropy_bound=float(np.mean([x['bound'] for x in e]))))
sw=[x['switch_ledger'] for x in T['contexts']];energy=np.array([x['energy'] for x in sw]);switch=np.array([x['switch_work'] for x in sw]);assert len(sw)==64 and max(x['first_law_error'] for x in sw)<1e-10
ref=[x for x in R['block_rg'] if x['beta']==16];rg={f:[x for x in ref if x['field']==f] for f in ('central_residual','mark')}
models=[dict(family=x['family'],scale=next(y['beta'] for y in x['models'] if y['model']=='score'),slope_count=x['score_log_slope_count'],unidentifiable=x['unidentifiable_waves'],slope_error=x['score_log_slope_max_error'],occupancy=x['occupancy'],conditional_KL={y['model']:y['conditional_KL'][-1] for y in x['models']}) for x in C['results']]
counts=dict(new_paths=A['paths'],training_reused=A['train_paths'],fresh_backups=A['backups'],actual_conditional_slope_waves=sum(x['slope_count'] for x in models),unidentifiable_initial_waves=sum(x['unidentifiable'] for x in models),historical_plot_pairs_verified=I['verified_pairs'],historical_manifest_files_verified=sum(x['status']=='VERIFIED' for x in I['stage_manifest_checks']),source_functions=94,guide_labels=27,quantum_reference_record_waves=z['n'],reference_thermal_contexts=len(sw),original_physical_temperature='NOT_IDENTIFIED',critical_point='NOT_ESTABLISHED',native_phase='NOT_IDENTIFIED')
panels=[
 dict(id='P01',title='Temperature extraction: conditional selector scale',evidence='NATIVE_CHECKED',sources=['CALIBRATION79.json'],data=models),
 dict(id='P02',title='Temperature scaling: occupancy projection',evidence='NATIVE_CHECKED',sources=['CALIBRATION79.json'],data=models),
 dict(id='P03',title='Finite-budget fluctuation and crossover',evidence='NATIVE_CHECKED',sources=['mqft_restore76_20261127/SUMMARY76.json'],data=Q['profiles']),
 dict(id='P04',title='Original critical exponent claim vs Ising reference',evidence='REFUTED_LINKED_CLAIM',sources=['mqft_restore76_20261127/ORIGINAL_CLAIM_AUDIT76.json'],data=X),
 dict(id='P05',title='Defined block RG and native dependence',evidence='NATIVE_PLUS_REFERENCE',sources=['mqft_restore77_20261127/SUMMARY77.json','mqft_restore77_20261127/ORIGINAL_RG_AUDIT77.json'],data=dict(block=rg,original_WF=RGA['wilson_fisher_beta_g_over_g'])),
 dict(id='P06',title='Explicit quantum record information and redundancy',evidence='DECLARED_MODEL',sources=['mqft_restore69_20261126/PHYSICS_RESULTS.json','mqft_restore73_20261126/DECISION_RESULTS73.json'],data=dict(waves=z['n'],chi=stats['chi'],classical=stats['classical'],chi_ratio=stats['chi_ratio'],classical_ratio=stats['classical_ratio'],packing2=stats['chi_packing2'],classical_packing2=stats['classical_packing2'],decision_logloss_gain=[dict(generation=x['generation'],arm=x['arm'],gain=x['gain'][-1]) for x in W['results'] if x['dataset']=='fresh_R73'])),
 dict(id='P07',title='Information-to-work and finite erasure',evidence='DECLARED_MODEL',sources=['mqft_restore66_20261126/FEEDBACK66.json','mqft_restore66_20261126/ERASURE66.json'],data=dict(rows=work,scope='beta times model work, beta E*=10/3; feedback and erasure are distinct protocols, not a net cycle; erasure bound is entropy decrease for degenerate endpoints')),
 dict(id='P08',title='Declared switched-H energy ledger',evidence='DECLARED_MODEL',sources=['mqft_restore74_20261126/PETZ74.json'],data=dict(contexts=len(sw),mean_energy=energy.mean(0).tolist(),mean_switch_work=switch.mean(0).tolist(),max_balance_error=max(x['first_law_error'] for x in sw),max_free_energy_exchange=max(abs(x['free_energy_exchange']) for x in sw))),
 dict(id='P09',title='Typed quality and identification',evidence='AUDIT',sources=['INVENTORY_AUDIT79.json','AUDIT79.json','INDEPENDENT79.json'],data=counts)]
assert all(x['data'] and x['sources'] for x in panels) and len(panels)==9
dump(S/'HONEST_SUMMARY79.json',dict(status='CORRECTED_NINE_PANEL_DATA_READY',panels=panels,inputs=inputs,scope='Original L31 nine-panel intent; no invented physical temperature, exponent, native gauge/phase or physical heat. Separate artifact-level acceptance and scientific certification.'))
# Curated per-item acceptance, anchored to exact original function/guide source and all historical evidence.
contract={x['id']:x for x in csv.DictReader((S/'acceptance79.tsv').open(),delimiter='\t')}
cat={x['id']:x for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())};progress={x['id']:x for x in csv.DictReader((B/'mqft_restore78_20261127/ORIGINAL_OUTPUT_PROGRESS78.csv').open())}
assert len(contract)==len(cat)==len(progress)==121
src={x['id']:x for x in O['spans']};rows=[];allitems=[]
for k,c in sorted(contract.items()):
 h=L['items'].get(k,[]);x=src[k];y=cat[k];z=progress[k];assert z['final_completion']=='NOT_CERTIFIED'
 latest=h[-1] if h else None;refs=[q['stage'] for q in h]
 if k=='L31':scope='R79 corrected L31 report; nine authenticated panels, physical quantities not asserted';latest_stage=79;render='RECONSTRUCTED_REPORT_DATA_READY'
 elif k in ('L03','L04','L25','L26','L34','G02','L05'):
  scope='R79 independent native score-law calibration, full-support occupancy projection or typed measurement-quality report; prior evidence retained.';latest_stage=79;render='SCOPED_R79_PANEL'
 else:scope=latest['scope'];latest_stage=latest['stage'];render='HISTORICAL_SCOPED_EVIDENCE'
 has_figure=any(q['figure_links'] or q.get('animation_links') for q in h)
 row=dict(id=k,owner=z['owner'],route=c['route'],source_function=y['function'],source_member=x['member'],source_sha=x['member_sha'] or x.get('guide_source_sha',''),source_span=f"{x['start']}-{x['end']}" if x['start'] else ';'.join(map(str,x.get('guide_lines',[]))),original_status=y['status'],claim_disposition=c['disposition'],latest_scope=scope,latest_stage=latest_stage,prior_evidence_stages=';'.join(map(str,refs)),historical_item_figure_link='YES' if has_figure else 'GROUPED_OR_NUMERIC_ONLY' if h else 'NEW_R79',corrected_output_state=render,final_scientific_certification='NOT_CERTIFIED',final_acceptance=c['final_acceptance'])
 rows.append(row);allitems.append(dict(row,original_source=x,historical_evidence=h))
p=S/'QUALITY_MATRIX79.csv'
with p.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
dump(S/'ITEM_ACCEPTANCE79.json',dict(scope='121 item-specific proposed acceptance contracts and typed current disposition. Existing evidence is not final original science certification; item-level semantic review by same agent, no independent expert endorsement.',items=allitems,disposition_counts=dict(collections.Counter(x['claim_disposition'] for x in rows)),route_counts=dict(collections.Counter(x['route'] for x in rows))))
report=['# MQFT original-output quality report (RESTORE79)','', '## Scope','RESTORE69 charter; RESTORE78 parent. All 94 source spans and 27 guide labels authenticated; all 640 retained pre-R79 plot pairs checked. Corrected L31 nine-panel report reconstructs measurements, declared models and negative results. Final original scientific certification remains separate.','', '## Evidence classification','NATIVE_SCOPED = measured native finite result; CORRECTED_SCOPED = corrected definition or formula with bounded evidence; MODEL_SCOPED = explicit auxiliary quantum/thermal/statistical model; REFUTED_LINKED_CLAIM = a linked original assertion has a direct counterexample or literature mismatch (not whole item refuted); PARTIAL = larger original requirement unresolved; RECONSTRUCTED_REPORT = report-level output reconstruction. These are authored scope judgments, not independent certification.','', '## All 121 item-specific acceptance requirements','See QUALITY_MATRIX79.csv and ITEM_ACCEPTANCE79.json for exact hashes, source spans, all historical evidence rows, plot/animation links, and route. Three legacy items L02/L19/L26 lacked unique prior item-specific figure links; L26 receives a new scoped panel. Grouped numerical evidence is not discarded.','', '| ID | Scope disposition | Latest stage | Remaining original acceptance |','|---|---|---:|---|']
for x in rows:report.append('| '+' | '.join(str(x[k]).replace('|','\\|') for k in ('id','claim_disposition','latest_stage','final_acceptance'))+' |')
report+=['','## Next constructive work','R80: normalized path law and controlled statistical 1PI/loop expansion on an independent actual marked path ensemble. R81: operational coherent phase intervention and matched phase-blind control. R82+: conditional memory/scale and physical calibration only where meaningful; separate publishable finite results from unsupported physical identifications.']
(S/'ORIGINAL_QUALITY_REPORT79.md').write_text('\n'.join(report)+'\n')
print('QUALITY',len(rows),collections.Counter(x['claim_disposition'] for x in rows),collections.Counter(x['route'] for x in rows),flush=True)
print('P07',work,'P08',panels[7]['data'],'P09',counts,flush=True)
