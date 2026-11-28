from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore91_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
parent='e35626561951a04494ea73945d07db8df1affae7'
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED92.md') and r['parent_dag_sha']==sha(O/'DAG.json')
v=read(S/'INDEPENDENT92.json');u=read(S/'SUMMARY92.json');pm=read(S/'PLOT_MANIFEST92.json');src=read(S/'ORIGINAL_SOURCE92.json')
assert v['status']==u['status']=='PASS_SCOPED' and len(pm['figures'])==7
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'G06':'Actual normalized occupancy concentration, iid-collision correction, entropy, signed centered modes, five sizes and two density arms; distinguish independent budgets and prefixes; no spontaneous order.','G08':'Actual connected legal-pair covariance, ordered open-board displacement shells, fixed-total zero mode and exact uniform null; conditional geometric extent is not physical correlation length.','G09':'Joint finite order, instantaneous conditional variance, covariance and controlled inherited R87 negative crossing/held-out test, root uncertainty; no critical point.','Q02':'Partial: actual finite selector-beta/density/size and concentration/variance/extent; missing physical T, C, external-field protocol and phase boundary.','Q29':'Actual spatial shell and nonstationary fixed-clock temporal correlation, paired root uncertainty; physical correlation length separately unidentified.','Q31':'Partial: size/budget, path action, occupancy entropy, score gap, geometric extent and variance; no quantum spectrum, entanglement, physical length or universal exponent.'}
acc=['G06','G08','G09','Q29'];partial=['Q02','Q31']
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS91.json');inv=cr(O/'PLOT_INVENTORY91.csv');assert len(inv)==693 and sha(O/'PLOT_INVENTORY91.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==700;cw(S/'PLOT_INVENTORY92.csv',inv)
c=read(O/'CONVERGENCE91.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==22 and not set(acc)&set(old)
newclaims=[
 dict(id='SCALE-COLLISION-04',stage=92,items=['G06','G09','Q31'],claim='For n fixed legal sites and B>=2 iid categorical samples, E[m(X)]=1/B+(1-1/B)m(p), hence corrected collision m*=(B m(X)-1)/(B-1) is unbiased for m(p). For adapted selectors pair probability is E[p_t(a_s)] for s<t, not an iid probability product; the correction is a realized collision statistic, not generally an unbiased adaptive concentration estimator.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','SCALE92.json.gz','SUMMARY92.json','INDEPENDENT92.json']],limit='Finite n,B, fixed legal support, exact iid enumeration n3 B4 two laws; actual adaptive results descriptive, no symmetry-breaking limit.'),
 dict(id='SCALE-SHELL-05',stage=92,items=['G08','G09','Q29','Q31'],claim='For fixed legal sites, ordered squared-displacement shell masses of predictable categorical covariance sum to zero, with nonpositive off-diagonal masses and exact distance moment -sum_d d C_d=2 mean_t Var_{p_t}(position). Empirical raw covariance splits into drift, innovation and cross, shellwise and at fixed clocks; compensated time innovations have zero population off-lag covariance without stationarity.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','SCALE92.json.gz','SUMMARY92.json','INDEPENDENT92.json']],limit='Native finite conditional categorical laws and repeated-root sample covariance; geometric spread not physical xi; finite sample pointwise deviations not universal autocorrelation; same-agent direct methods.') ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='ACCEPTED_CORRECTED_SCOPED_ARTIFACT' if k in acc else 'PARTIAL_CORRECTED_MODEL_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Finite controlled ensemble only; physical critical/length/gap identification open.'))
dump(S/'ITEM_RECONSTRUCTION92.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=26,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=28,new_accepted_subclaims=2,new_corrected_artifacts=4,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION92.json'),scope='Four corrected scoped artifacts; Q02/Q31 partial; six open science claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==28
dump(S/'CONVERGENCE92.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS91.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R92_evidence']=scopes.get(k,'');x['R92_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k]),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R92_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS92.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE92.csv',ev)
qm=cr(O/'QUALITY_MATRIX91.csv')
for x in qm:
 k=x['id'];x['R92_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R92_open_subclaims']='';x['R92_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='92';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R92'
cw(S/'QUALITY_MATRIX92.csv',qm)
dump(S/'ITEM_CONVERGENCE92.json',dict(items=[dict(id=x['id'],accepted=x['R92_convergence_subclaims'],scope=x['R92_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R92_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX91.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE91.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION92.json')))
prev.update(figure_pairs=700,distinct_png_sha=700,files_paired=1400,current_pairs=7,prior_pairs=693,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY92.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE92.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS92.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX92.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=28,final_scientific_certified=0)
prev['stages'][S.name]=7;prev['original_output_evidence'].update(current=6,R55_R92_union=121,scope='121 scoped coverage; 26 corrected artifact acceptances,28 scoped claims; Q02/Q31 partial.')
dump(S/'PROGRESS92.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT92.md').write_text('# RESTORE92 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('REPLAN-04.C','REPLAN-03'):
  node['restore92_followup']=dict(evidence='4320 raw paths, 10400 prefixes, 400 spatial and 120 temporal strata; five sizes, exact fixed-total shell and iid concentration correction',accepted_subclaims=[z['id'] for z in newclaims],corrected_artifacts=acc,partial=partial,remaining='Controlled physical critical limit, equilibrium FDT and quantum gap not identified.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT92.md,ITEM_RECONSTRUCTION92.json,CONVERGENCE92.json}')
dag.update(revision='RESTORE92_20261128',scope='Convergence: corrected finite size, concentration and signed spatiotemporal covariance',previous_entry=dag['entry'],entry='REPLAN-04.C',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE93-CAUSAL-RESPONSE-FDT-OUTPUTS',mainline_priorities=['R93 G10-G12 corrected causal response and FDT outputs','Direct original reconstruction; no repetitive diagnostic sweep','Six open science claims, Q02/Q31 and Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX92.csv',convergence_ledger='CONVERGENCE92.json')
rec=dict(status='four_corrected_original_artifacts_two_new_scoped_claims_two_partial',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=4320,prefix_rows=10400,B64_paths=3040,figures=7,animations=0,retained_pairs=700,original_evidence_items=6,original_evidence_union_R55_R92=121,original_obligations=121,corrected_report_artifact_accepted=26,new_corrected_artifacts=4,accepted_corrected_subclaims=28,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='95 corrected artifact obligations; six open science claims, Q02/Q31 and Q19/Q20 partial; G10-G12 next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED92.md','SCALE92.json.gz','SCALE_PACKING92.json','SUMMARY92.json','INDEPENDENT92.json','ORIGINAL_SOURCE92.json','CONVERGENCE92.json','ITEM_RECONSTRUCTION92.json','ORIGINAL_OUTPUT_REPORT92.md','ITEM_CONVERGENCE92.json','QUALITY_MATRIX92.csv','PLOT_MANIFEST92.json','PLOT_INVENTORY92.csv','ORIGINAL_OUTPUT_EVIDENCE92.csv','ORIGINAL_OUTPUT_PROGRESS92.csv','PROGRESS92.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore92']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore92']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=u['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish92.log','backup92.log','RUN_MANIFEST92.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw={x['path']:x['sha256'] for a in v['audit'] for x in a['raw_receipts']};assert len(raw)==4320
manifest=dict(schema='mqft_restore92_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute92.sh; initial scale92.py completed before interruption; verified retained output',files={f:sha(S/f) for f in sorted(files)},inputs=u['source_hashes'],prior_raw_receipts=raw,software=u['software'],limits='Same-agent independent checks; no physical xi/T/gap/critical limit; prior R87 heldout evidence inherited.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST92.json',manifest)
dst=P/'research/restore92';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST92.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS92.csv','ORIGINAL_OUTPUT_EVIDENCE92.csv','PLOT_INVENTORY92.csv','PROGRESS92.json','QUALITY_MATRIX92.csv','CONVERGENCE92.json','ITEM_RECONSTRUCTION92.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE92_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE92_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore92','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore92/logs',':(exclude)research/restore92/CORE_POLICY.md',':(exclude)research/restore92/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE92: reconstruct finite-size concentration and signed spatial-temporal covariance')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE92.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE92.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE92_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST92.json'),source_zip_sha256=sha(S/'MQFT_RESTORE92_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE92.git.bundle'),progress_sha256=sha(S/'PROGRESS92.json'),fresh_paths=0,fresh_backups=0,figures=7,cumulative_retained_pairs=700,original_union=121,accepted_corrected_subclaims=28,corrected_report_artifact_accepted=26,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
