from base61 import *
import csv,shutil,subprocess
from PIL import Image
O=B/'mqft_restore99_20261129';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';n=100
parent='a79789439c3bf3ca55c320ef2ac5b3083b2b5c7f'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED100.md') and r['parent_dag_sha']==sha(O/'DAG.json') and r['parent']==parent
for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f)
v=read(S/'INDEPENDENT100.json');u=read(S/'NATIVE100.json');q=read(S/'SUMMARY100.json');pm=read(S/'PLOT_MANIFEST100.json');src=read(S/'ORIGINAL_SOURCE100.json')
assert v['status']==u['status']=='PASS_SCOPED' and v['reference_paths']==2555904 and u['prefix_rows']==10400 and v['reference_cases']==312 and u['paths']==4320 and len(pm['figures'])==10 and len(pm['animations'])==2
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'Q01': 'Actual native likelihood and score ledgers plus new explicit no-feedback driven reference, exact signed work/heat, Crooks and dissipation; not a native physical bath.', 'Q03': 'Explicit periodic scaled-spectrum two-bath Otto-type reference, 624 closed marginal cycles, work/heat and efficiency, finite versus full refresh; no physical native engine or actual Carnot cycle.', 'Q04': 'Five exact reference cycle stages and animation, periodic closure and cumulative signed work/heat; no invented interpolation.', 'Q06': 'Partial: reference full versus finite refresh and separate actual native score ledger, no common physical state variables/reservoirs, no native versus ideal efficiency admission.', 'Q07': 'Comprehensive scoped nonequilibrium report, actual native information and score ledger, explicit reference work/heat and entropy, cyclic closure, capacity and prior feedback/quantum/response evidence; no measured bath.', 'Q19': 'Partial: independent exact F/R reference work laws, 312 cases and 4096 independent samples/direction, support-aware Crooks/Jarzynski, uncertainty and failures; native calibrated work and reverse measure absent.', 'Q20': 'Partial: actual reference sample trajectories, exact cumulants, component covariance and finite tilted dual; no native thermodynamic work or asymptotic rate/FDT.', 'Q22': 'Four sample-count frames of actual independently generated F/R reference estimators and ESS, retained nonconvergence; not physical native equality.', 'Q23': 'Comprehensive scoped declared reference work and fluctuation report; explicit Gibbs starts, exact DeltaF, independent F/R paths, finite sample uncertainty, prior feedback support correction distinct.', 'Q38': 'Explicit finite Gibbs spectrum entropy, free energy, heat capacity, occupied energy-bin counts and driven entropy production; R91/R97 thermal derivative and R93 reference response retained; no continuum microcanonical derivative or native T.'}
partial=['Q06','Q19','Q20'];acc=[k for k in scopes if k not in partial]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS99.json');inv=cr(O/'PLOT_INVENTORY99.csv');assert len(inv)==752 and sha(O/'PLOT_INVENTORY99.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==762;cw(S/'PLOT_INVENTORY100.csv',inv)
ans=pm['animations'];assert sorted(a['frames'] for a in ans)==[4,5]
for an in ans:
 assert sha(S/an['path'])==an['sha256']
 with Image.open(S/an['path']) as im:
  assert im.n_frames==an['frames'];hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert hs==an['frame_rgb_sha256'] and len(set(hs))==im.n_frames
c=read(O/'CONVERGENCE99.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==57 and not set(scopes)&set(old)
newclaims=[
 dict(id='NATIVE-LR-20',stage=n,items=['Q01','Q06','Q07'],claim='On full legal support with normalized predictable q, exp(-Sigma) is a mean-one likelihood-ratio martingale and cumulative conditional KL is its log-ratio compensator. Actual 4320 histories, 215040 root choices, 10400 prefixes and two comparisons checked; finite log sample IFT may be as low as -314.777, no equality imposed. Score switch and action change telescope exactly.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','NATIVE100.json','NATIVE100.npz','INDEPENDENT100.json']],limit='Native path likelihood against a comparator, not physical time reversal, work, heat or bath entropy.'),
 dict(id='THERMO-DRIVE-21',stage=n,items=['Q01','Q07','Q19','Q20','Q22','Q23','Q38'],claim='For declared Gibbs-start two-quench detailed-balance refresh protocol and properly ordered reverse, exact F/R ratio exp(beta(W-DeltaF)) and Jarzynski hold. 312 reference cases on 104 authenticated score spectra, 2555904 independent auxiliary draws; independent mixture/estimator checks, work cumulants and finite dual; final sample failures retained.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','THERMAL100.json','THERMAL_SAMPLES100.npz','SUMMARY100.json','INDEPENDENT100.json']],limit='Explicit finite reference only; no native calibrated work/heat, physical bath or asymptotic large deviation inference.'),
 dict(id='THERMO-CYCLE-22',stage=n,items=['Q03','Q04','Q06','Q07','Q38'],claim='For scaled-spectrum periodic two-reservoir reference with equal partial refresh r and scale a, y-x=r/(2-r)(ph-pc), Wout=(a-1)d, eta=1-1/a, below Carnot in declared regime. All 624 periodic cases checked by independent iteration and analytic heat formula; finite/ideal output attenuation r/(2-r).',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','THERMAL100.json','SUMMARY100.json','INDEPENDENT100.json']],limit='Reference cycle with explicit reservoirs, not actual MCTS heat engine, Carnot implementation or coherence enhancement.')
 ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='PARTIAL_CORRECTED_DIAGNOSTIC' if k in partial else 'ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Native thermal calibration, physical reverse and work/heat remain open.'))
for an in ans:next(x for x in items if x['id']==an['id'])['animations']=[an]
dump(S/'ITEM_RECONSTRUCTION100.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=64,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=46,new_accepted_subclaims=3,new_corrected_artifacts=7,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION100.json'),scope='Seven corrected scoped thermal/reference outputs, three partial; native thermal calibration remains open; six open claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==46
dump(S/'CONVERGENCE100.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS99.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R100_evidence']=scopes.get(k,'');x['R100_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k])+(';'+';'.join(a['path'] for a in ans if a['id']==k) if k in ('Q04','Q22') else ''),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R100_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS100.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE100.csv',ev)
qm=cr(O/'QUALITY_MATRIX99.csv')
for x in qm:
 k=x['id'];x['R100_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R100_open_subclaims']='THERMO-PHYSICAL-04' if k in scopes else '';x['R100_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='100';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R100'
cw(S/'QUALITY_MATRIX100.csv',qm)
dump(S/'ITEM_CONVERGENCE100.json',dict(items=[dict(id=x['id'],accepted=x['R100_convergence_subclaims'],scope=x['R100_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R100_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX99.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE99.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION100.json')))
prev.update(figure_pairs=762,distinct_png_sha=762,files_paired=1524,current_pairs=10,prior_pairs=752,current_animations=2,inventory_sha256=sha(S/'PLOT_INVENTORY100.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE100.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS100.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX100.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=46,final_scientific_certified=0,animation_scope='R100 five exact cycle stages and four independent estimator sample-count frames; prior animations retained')
prev['stages'][S.name]=10;prev['original_output_evidence'].update(current=10,R55_R100_union=121,scope='121 scoped coverage; 64 corrected artifact acceptances,46 scoped claims; six open claims.')
dump(S/'PROGRESS100.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT100.md').write_text('# RESTORE100 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  ks=[k for k in scopes if any(x['id']==k and x['owner']==node['id'] for x in rows)]
  node['restore100_followup']=dict(evidence='6768 raw receipts, 4320 actual native histories, 215040 selections, 10400 prefixes; 312 reference cases, 2555904 auxiliary paths, 624 periodic cycles',accepted_subclaims=[z['id'] for z in newclaims if set(ks)&set(z['items'])],corrected_artifacts=[k for k in acc if k in ks],partial=[k for k in partial if k in ks],remaining='Native thermal calibration, physical work/heat/reverse not established.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT100.md,ITEM_RECONSTRUCTION100.json,CONVERGENCE100.json}')
dag.update(revision='RESTORE100_20261129',scope='Convergence: native likelihood measures, independent driven thermal reference and periodic cycles',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE101-FIELD-MAPPING-OUTPUTS',mainline_priorities=['R101 exact Q41-Q43 and legacy L-series field/mapping obligations; inspect acceptance list','Direct original reconstruction; no optimizer drift','Six open science claims, Q10 Q36 Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX100.csv',convergence_ledger='CONVERGENCE100.json')
rec=dict(status='seven_corrected_artifacts_three_partial_three_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=4320,reused_draws=215040,prefix_rows=10400,reference_raw_paths=2448,reference_cases=312,reference_paths=2555904,reference_cycles=624,figures=10,animations=2,retained_pairs=762,original_evidence_items=10,original_evidence_union_R55_R100=121,original_obligations=121,corrected_report_artifact_accepted=64,new_corrected_artifacts=7,accepted_corrected_subclaims=46,new_accepted_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='57 corrected artifact obligations; Q06/Q19/Q20 partial; six open science claims; field/mapping outputs next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED100.md','NATIVE100.json','NATIVE100.npz','THERMAL100.json','THERMAL_SAMPLES100.npz','SUMMARY100.json','INDEPENDENT100.json','ORIGINAL_SOURCE100.json','CONVERGENCE100.json','ITEM_RECONSTRUCTION100.json','ORIGINAL_OUTPUT_REPORT100.md','ITEM_CONVERGENCE100.json','QUALITY_MATRIX100.csv','PLOT_MANIFEST100.json','PLOT_INVENTORY100.csv','ORIGINAL_OUTPUT_EVIDENCE100.csv','ORIGINAL_OUTPUT_PROGRESS100.csv','PROGRESS100.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore100']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore100']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish100.log','backup100.log','RUN_MANIFEST100.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=v['raw_receipts'];assert len(raw)==6768
manifest=dict(schema='mqft_restore100_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute100.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent independent mixture, raw-index sample replay and long-double weights, analytic periodic cycle; no native thermal calibration.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST100.json',manifest)
dst=P/'research/restore100';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST100.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS100.csv','ORIGINAL_OUTPUT_EVIDENCE100.csv','PLOT_INVENTORY100.csv','PROGRESS100.json','QUALITY_MATRIX100.csv','CONVERGENCE100.json','ITEM_RECONSTRUCTION100.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE100_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE100_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore100','docs')
# Verify byte-locked preregistration.
assert sha(S/'THEORY_REGISTERED100.md')==r['contract_sha']==sha(dst/'THEORY_REGISTERED100.md')
git('diff','--cached','--check','--','.',':(exclude)research/restore100/logs',':(exclude)research/restore100/CORE_POLICY.md',':(exclude)research/restore100/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE100: native likelihood and independent thermal reference cycles')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE100.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE100.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE100_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST100.json'),source_zip_sha256=sha(S/'MQFT_RESTORE100_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE100.git.bundle'),progress_sha256=sha(S/'PROGRESS100.json'),fresh_paths=0,fresh_backups=0,figures=10,animations=2,cumulative_retained_pairs=762,original_union=121,accepted_corrected_subclaims=46,corrected_report_artifact_accepted=64,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
