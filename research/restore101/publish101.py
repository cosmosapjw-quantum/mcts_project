from base61 import *
import csv,shutil,subprocess
from PIL import Image
O=B/'mqft_restore100_20261129';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';n=101
parent='3a2a5b702f3e61cb9eb21ece5de94664d8f4aff2'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED101.md') and r['parent_dag_sha']==sha(O/'DAG.json') and r['parent']==parent
for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f)
v=read(S/'INDEPENDENT101.json');u=read(S/'FIELD101.json');q=read(S/'SUMMARY101.json');pm=read(S/'PLOT_MANIFEST101.json');src=read(S/'ORIGINAL_SOURCE101.json')
assert v['status']==u['status']=='PASS_SCOPED' and v['path_evaluations']==2160 and v['local_events']==41472 and v['record_cases']==5184 and u['paths']==2304 and len(pm['figures'])==9 and len(pm['animations'])==1
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'Q41': 'Actual frozen-history local visibility with explicit unit dimensionless action, finite/singular mask, principal-branch floor and exact dephase-vs-unitary matched-X counterexample; no physical hbar.', 'Q42': 'Exact declared discrete Kraus/Choi and Bloch dynamics on 41472 actual active history-conditioned steps, local dimensionless generators, singular erase and noncommuting order; frozen sequence not global adaptive channel.', 'Q43': 'Partial: empirical native path logMGF, Legendre dual, causal score and linear source contraction, heldout block comparison; explicit complex joint record phase/readout and success weighting. Physical phase/action, calibrated hbar, continuum path integral and one-loop identification absent.', 'Q44': 'Twelve actual selection-clock frames for finite visibility proxy and singular mask, no fictitious physical crossover threshold.', 'Q45': 'Comprehensive scoped native path source/response, dimensionless visibility, complex CP and record mapping report; retained negative predictive and postselection limitations, no physical quantum-classical limit.', 'L09': 'Actual conditional programmed qubit coherence trajectories with erase, phase and swap controls, explicit local frozen generator; no unjustified global relaxation-time fit.', 'L11': 'Explicit complex system-record joint density and eigenvalues, independent partial transpose/negativity, phase-tag and dephased controls with Gram/Born readout; not branch correlations or native entanglement.', 'L20': 'Exact complex history-conditioned CP instrument, Kraus completeness/Choi PSD and trace, noncommuting and singular channels; local piecewise dimensionless GKSL reference only.'}
partial=['Q43'];acc=[k for k in scopes if k not in partial]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS100.json');inv=cr(O/'PLOT_INVENTORY100.csv');assert len(inv)==762 and sha(O/'PLOT_INVENTORY100.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==771;cw(S/'PLOT_INVENTORY101.csv',inv)
ans=pm['animations'];assert sorted(a['frames'] for a in ans)==[12]
for an in ans:
 assert sha(S/an['path'])==an['sha256']
 with Image.open(S/an['path']) as im:
  assert im.n_frames==an['frames'];hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert hs==an['frame_rgb_sha256'] and len(set(hs))==im.n_frames
c=read(O/'CONVERGENCE100.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==64 and not set(scopes)&set(old)
newclaims=[
 dict(id='FIELD-PATH-23',stage=n,items=['Q43','Q45'],claim='For fixed finite empirical path law W(J,h), gradient is tilted mean, Hessian covariance, dual is KL and mixed source derivative is covariance with normalized path score. 2160 cases, 1935 rank4 and225 rank3, 192 exact linear source restrictions, disjoint rep halves. Ordinary correlation differs from causal response (exact rational counterexample). Heldout joint vs factorized paired intervals all include zero.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','FIELD101.json','SUMMARY101.json','INDEPENDENT101.json']],limit='Finite empirical tilted path law, not physical QFT, continuum 1PI or new-root transfer.'),
 dict(id='Q-CP-MAP-24',stage=n,items=['Q41','Q42','Q44','Q45','L09','L20'],claim='41472 actual history-conditioned local programmed maps verified by complex Kraus and independent Choi. Fixed-axis generator in unit selection segment kappa=-ln gamma/2; singular erase and 4608 noncommuting order cases retained. Unit-action visibility proxy finite34560/singular6912, same-X dephased vs pure rotation trace distance sqrt(1-gamma^2)/2; no physical hbar.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','QUANTUM101.json','QUANTUM101.npz','SUMMARY101.json','INDEPENDENT101.json']],limit='Frozen local nonselective sequence at actual histories not global adaptive unselective dynamics or a physical Lindblad generator.'),
 dict(id='Q-PHASE-TAG-25',stage=n,items=['Q43','Q45','L11'],claim='5184 declared current-prefix complex joint states; local record phase tag leaves computational diagonal, joint spectrum and system-record NPT invariant but changes fixed Hadamard readout in3456 cases. Independent complex Gram amplitude verifies readout; system-record NPT1728. Native-repeat conditional and success-weighted phase TV distinct from pair diagnostic.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','QUANTUM101.json','SUMMARY101.json','INDEPENDENT101.json']],limit='Explicit declared record density and fixed readout, not native coherent hardware, observed physical path amplitudes or physical entanglement.')
 ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='PARTIAL_CORRECTED_DIAGNOSTIC' if k in partial else 'ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Physical hbar, native coherent readout, physical time and continuum QFT remain open.'))
for an in ans:next(x for x in items if x['id']==an['id'])['animations']=[an]
dump(S/'ITEM_RECONSTRUCTION101.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=71,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=49,new_accepted_subclaims=3,new_corrected_artifacts=7,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION101.json'),scope='Seven corrected scoped field and quantum representation outputs, Q43 partial; six open claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==49
dump(S/'CONVERGENCE101.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS100.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R101_evidence']=scopes.get(k,'');x['R101_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k])+(';'+';'.join(a['path'] for a in ans if a['id']==k) if k=='Q44' else ''),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R101_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS101.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE101.csv',ev)
qm=cr(O/'QUALITY_MATRIX100.csv')
for x in qm:
 k=x['id'];x['R101_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R101_open_subclaims']='Q-RECORD-03;MEM-CLOSURE-04' if k in scopes else '';x['R101_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='101';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R101'
cw(S/'QUALITY_MATRIX101.csv',qm)
dump(S/'ITEM_CONVERGENCE101.json',dict(items=[dict(id=x['id'],accepted=x['R101_convergence_subclaims'],scope=x['R101_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R101_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX100.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE100.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION101.json')))
prev.update(figure_pairs=771,distinct_png_sha=771,files_paired=1542,current_pairs=9,prior_pairs=762,current_animations=1,inventory_sha256=sha(S/'PLOT_INVENTORY101.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE101.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS101.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX101.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=49,final_scientific_certified=0,animation_scope='R101 actual 12 selection-clock visibility frames; prior animations retained')
prev['stages'][S.name]=9;prev['original_output_evidence'].update(current=8,R55_R101_union=121,scope='121 scoped coverage; 71 corrected artifact acceptances,49 scoped claims; six open claims.')
dump(S/'PROGRESS101.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT101.md').write_text('# RESTORE101 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  ks=[k for k in scopes if any(x['id']==k and x['owner']==node['id'] for x in rows)]
  node['restore101_followup']=dict(evidence='6408 raw receipts, 2160 path source evaluations and 192 coarse checks; 41472 local channels, 5184 complex joint states',accepted_subclaims=[z['id'] for z in newclaims if set(ks)&set(z['items'])],corrected_artifacts=[k for k in acc if k in ks],partial=[k for k in partial if k in ks],remaining='Physical hbar, native coherent observation, global adaptive channel and continuum QFT not identified.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT101.md,ITEM_RECONSTRUCTION101.json,CONVERGENCE101.json}')
dag.update(revision='RESTORE101_20261129',scope='Convergence: native path source dual, history-conditioned complex CP and phase-tag readout',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE102-LEGACY-MARKOV-CLOSURE',mainline_priorities=['R102 L39-L44/L48/L49 temporal Markov and closure reports; inspect acceptance list','Direct original reconstruction; no optimizer drift','Six open science claims, Q10 Q36 Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX101.csv',convergence_ledger='CONVERGENCE101.json')
rec=dict(status='seven_corrected_artifacts_one_partial_three_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_pulse_paths=2304,reused_phase_paths=3456,raw_receipts=6408,path_evaluations=2160,coarse_source_checks=192,local_events=41472,record_cases=5184,figures=9,animations=1,retained_pairs=771,original_evidence_items=8,original_evidence_union_R55_R101=121,original_obligations=121,corrected_report_artifact_accepted=71,new_corrected_artifacts=7,accepted_corrected_subclaims=49,new_accepted_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='50 corrected artifact obligations; Q43 partial; six open science claims; legacy Markov closure next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED101.md','FIELD101.json','QUANTUM101.json','QUANTUM101.npz','SUMMARY101.json','INDEPENDENT101.json','ORIGINAL_SOURCE101.json','CONVERGENCE101.json','ITEM_RECONSTRUCTION101.json','ORIGINAL_OUTPUT_REPORT101.md','ITEM_CONVERGENCE101.json','QUALITY_MATRIX101.csv','PLOT_MANIFEST101.json','PLOT_INVENTORY101.csv','ORIGINAL_OUTPUT_EVIDENCE101.csv','ORIGINAL_OUTPUT_PROGRESS101.csv','PROGRESS101.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore101']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore101']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish101.log','backup101.log','RUN_MANIFEST101.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=v['raw_receipts'];assert len(raw)==6408
manifest=dict(schema='mqft_restore101_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute101.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent long-double source checks, independent Choi and Gram amplitude, exact phase tag; no physical hbar or continuum QFT.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST101.json',manifest)
dst=P/'research/restore101';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST101.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS101.csv','ORIGINAL_OUTPUT_EVIDENCE101.csv','PLOT_INVENTORY101.csv','PROGRESS101.json','QUALITY_MATRIX101.csv','CONVERGENCE101.json','ITEM_RECONSTRUCTION101.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE101_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE101_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore101','docs')
# Verify byte-locked preregistration.
assert sha(S/'THEORY_REGISTERED101.md')==r['contract_sha']==sha(dst/'THEORY_REGISTERED101.md')
git('diff','--cached','--check','--','.',':(exclude)research/restore101/logs',':(exclude)research/restore101/CORE_POLICY.md',':(exclude)research/restore101/QUANTUM_REPRESENTATION_POLICY.md',':(exclude)research/restore101/THEORY_REGISTERED101.md')
git('commit','-m','RESTORE101: path source dual and complex CP phase mapping')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE101.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE101.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE101_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST101.json'),source_zip_sha256=sha(S/'MQFT_RESTORE101_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE101.git.bundle'),progress_sha256=sha(S/'PROGRESS101.json'),fresh_paths=0,fresh_backups=0,figures=9,animations=1,cumulative_retained_pairs=771,original_union=121,accepted_corrected_subclaims=49,corrected_report_artifact_accepted=71,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
