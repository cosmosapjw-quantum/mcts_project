from base61 import *
import csv,shutil,subprocess
from PIL import Image
O=B/'mqft_restore98_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';n=99
parent='aac6e5afef6662781143856a7a0e7b64cf3e41c8'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED99.md') and r['parent_dag_sha']==sha(O/'DAG.json') and r['parent']==parent
for f in ['CORE_POLICY.md','RESEARCH_CHARTER.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f)
v=read(S/'INDEPENDENT99.json');u=read(S/'SCALE99.json');q=read(S/'SUMMARY99.json');pm=read(S/'PLOT_MANIFEST99.json');src=read(S/'ORIGINAL_SOURCE99.json')
assert v['status']==u['status']=='PASS_SCOPED' and v['action_evaluations']==104000 and u['prefix_rows']==10400 and v['model_fits']==96 and u['paths']==4320 and len(pm['figures'])==11 and len(pm['animations'])==2
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'Q14': 'Partial: actual finite effective size slopes, sampled score controls and analytic noncritical peak/fourth-moment null; physical critical point and independent exponents not identified.', 'Q15': 'Partial: registered transformed-coordinate diagnostic and heldout model comparison, not critical 3D data collapse or universality.', 'Q16': 'Scoped finite scaling functions of actual categorical A/X, matched-peak tail residual, frozen Fisher, one-draw centered moments and explicit scalar source Legendre dual Gamma; separate independent sizes/budgets. Physical 1PI, thermal and equilibrium scaling not identified.', 'Q17': 'Four actual finite score-control frames across five sizes and two density arms, fixed B64 and root uncertainty, not physical critical animation.', 'Q18': 'Comprehensive scoped critical report: 96 fits, 48 L21 comparisons, negative/positive critical-type results and analytic noncritical null; no physical exponent or universality admission.', 'Q30': 'Partial: actual heldout model/collapse diagnostic against noncritical alternatives, no physical critical collapse.', 'Q31': 'Partial: independent B16 vs B64 and five sizes, actual categorical entropy, path action comparator, score gap, geometric spread and conditional Fisher; no quantum gap, physical correlation length or universal exponent.', 'Q33': 'Actual R92 signed radial covariance across five sizes, fresh dense beta16, h16/32/64 frames from same B64 paths; independent bin reconstruction and fixed per-panel axes, not equilibrium time or correlation length.', 'Q34': 'Comprehensive scoped finite statistical report: exact peak-tail identities, actual size/budget and radial covariance, frozen Fisher, one-draw source dual, heldout fits with earlier response evidence retained. No assumed phase transition.'}
partial=['Q14','Q15','Q30','Q31'];acc=[k for k in scopes if k not in partial]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS98.json');inv=cr(O/'PLOT_INVENTORY98.csv');assert len(inv)==741 and sha(O/'PLOT_INVENTORY98.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==752;cw(S/'PLOT_INVENTORY99.csv',inv)
ans=pm['animations'];assert sorted(a['frames'] for a in ans)==[3,4]
for an in ans:
 assert sha(S/an['path'])==an['sha256']
 with Image.open(S/an['path']) as im:
  assert im.n_frames==an['frames'];hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert hs==an['frame_rgb_sha256'] and len(set(hs))==im.n_frames
c=read(O/'CONVERGENCE98.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==52 and not set(scopes)&set(old)
cnt=q['counts']
newclaims=[
 dict(id='SCALE-TAIL-17',stage=n,items=['Q16','Q31','Q34'],claim='Exact peak-tail KL and concentration decompositions for positive finite categorical distributions; all 10400 actual prefix mixtures, 62400 dyadic rows, 41600 one-draw moments checked. Tail KL >1e-9 in 7680 prefixes, mean tail fraction .406062 among KL>1e-12, max matched-peak TV .755110. X Binder undefined86 retained.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','SCALE99.json','SCALE99.npz','SUMMARY99.json','INDEPENDENT99.json']],limit='Peak-matched diagnostic, not independent prediction of M; no physical order parameter or interaction identification.'),
 dict(id='SCALE-CROSSOVER-18',stage=n,items=['Q14','Q15','Q16','Q17','Q18','Q30','Q34'],claim='Noninteracting one-spike categorical law has exact shifted sigmoid collapse, finite score Fisher peak and centered Bernoulli U4=2/3 at half mass. 96 development-only fits, 4536 grid candidates, 48 L21 external comparisons; critical-width beats all three alternatives 12/48, h64 3/16, one paired pointwise improvement against development-selected noncritical model. No universal exponent or impossibility claim.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','FITS99.json','SUMMARY99.json','INDEPENDENT99.json']],limit='Existing independent cohorts previously studied; finite score-control design, no calibrated transition or multiplicity-adjusted inference.'),
 dict(id='SCALE-ACTION-19',stage=n,items=['Q16','Q34'],claim='For named one-draw categorical law W(J)=log E exp(Jf), Gamma(m(J))=Jm-W=D(p_J||p), W''=Var(f)>0 and derivative of dual fixed by J. 104000 actual A/peak-matched source evaluations with direct KL residual2.484e-15.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','EFFECTIVE_ACTION_EXTENSION99.md','ACTION99.json','ACTION99.npz','INDEPENDENT99.json']],limit='Categorical one-draw Legendre dual, not path-space or physical QFT 1PI effective action. Post-initial-run extension documented.')
 ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='PARTIAL_CORRECTED_DIAGNOSTIC' if k in partial else 'ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Physical critical point, exponents, length and thermal calibration remain open.'))
for an in ans:next(x for x in items if x['id']==an['id'])['animations']=[an]
dump(S/'ITEM_RECONSTRUCTION99.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=57,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=43,new_accepted_subclaims=3,new_corrected_artifacts=5,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION99.json'),scope='Five corrected scoped finite scaling outputs, four partial; no physical criticality; six open claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==43
dump(S/'CONVERGENCE99.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS98.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R99_evidence']=scopes.get(k,'');x['R99_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k])+(';'+';'.join(a['path'] for a in ans if a['id']==k) if k in ('Q17','Q33') else ''),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R99_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS99.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE99.csv',ev)
qm=cr(O/'QUALITY_MATRIX98.csv')
for x in qm:
 k=x['id'];x['R99_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R99_open_subclaims']='SCALE-CRITICAL-03;MEM-CLOSURE-04' if k in scopes else '';x['R99_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='99';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R99'
cw(S/'QUALITY_MATRIX99.csv',qm)
dump(S/'ITEM_CONVERGENCE99.json',dict(items=[dict(id=x['id'],accepted=x['R99_convergence_subclaims'],scope=x['R99_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R99_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX98.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE98.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION99.json')))
prev.update(figure_pairs=752,distinct_png_sha=752,files_paired=1504,current_pairs=11,prior_pairs=741,current_animations=2,inventory_sha256=sha(S/'PLOT_INVENTORY99.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE99.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS99.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX99.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=43,final_scientific_certified=0,animation_scope='R99 four score-control and three prefix covariance frames; prior animations retained')
prev['stages'][S.name]=11;prev['original_output_evidence'].update(current=9,R55_R99_union=121,scope='121 scoped coverage; 57 corrected artifact acceptances,43 scoped claims; six open claims.')
dump(S/'PROGRESS99.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT99.md').write_text('# RESTORE99 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  ks=[k for k in scopes if any(x['id']==k and x['owner']==node['id'] for x in rows)]
  node['restore99_followup']=dict(evidence='4320 raw receipts rehashed, 10400 prefixes, 62400 scales, 41600 moments, 104000 source duals, 96 fits and 48 L21 comparisons',accepted_subclaims=[z['id'] for z in newclaims if set(ks)&set(z['items'])],corrected_artifacts=[k for k in acc if k in ks],partial=[k for k in partial if k in ks],remaining='Physical critical point, critical exponents, physical correlation length and thermal calibration not established.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT99.md,ITEM_RECONSTRUCTION99.json,CONVERGENCE99.json}')
dag.update(revision='RESTORE99_20261129',scope='Convergence: finite categorical scaling, exact tail decomposition, noncritical null and heldout model competition',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE100-NONEQUILIBRIUM-THERMAL-OUTPUTS',mainline_priorities=['R100 Q01/Q03/Q04/Q06/Q07/Q19/Q20/Q22/Q23/Q38 nonequilibrium and thermal outputs','Direct original reconstruction; no optimizer drift','Six open science claims, Q10 Q36 Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX99.csv',convergence_ledger='CONVERGENCE99.json')
rec=dict(status='five_corrected_artifacts_four_partial_three_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=4320,prefix_rows=10400,prefix_scale_rows=62400,moment_cases=41600,action_evaluations=104000,model_fits=96,heldout_comparisons=48,figures=11,animations=2,retained_pairs=752,original_evidence_items=9,original_evidence_union_R55_R99=121,original_obligations=121,corrected_report_artifact_accepted=57,new_corrected_artifacts=5,accepted_corrected_subclaims=43,new_accepted_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='64 corrected artifact obligations; Q14/Q15/Q30/Q31 partial; six open science claims; nonequilibrium/thermal outputs next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED99.md','SCALE99.json','SCALE99.npz','SUMMARY99.json','FITS99.json','ACTION99.json','ACTION99.npz','EFFECTIVE_ACTION_EXTENSION99.md','INDEPENDENT99.json','ORIGINAL_SOURCE99.json','CONVERGENCE99.json','ITEM_RECONSTRUCTION99.json','ORIGINAL_OUTPUT_REPORT99.md','ITEM_CONVERGENCE99.json','QUALITY_MATRIX99.csv','PLOT_MANIFEST99.json','PLOT_INVENTORY99.csv','ORIGINAL_OUTPUT_EVIDENCE99.csv','ORIGINAL_OUTPUT_PROGRESS99.csv','PROGRESS99.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore99']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore99']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish99.log','backup99.log','RUN_MANIFEST99.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=v['raw_receipts'];assert len(raw)==4320
manifest=dict(schema='mqft_restore99_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute99.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent independent moments, exact tail identities, rederived R92 radial bins, QR checks and finite one-draw dual; no physical critical point or calibrated bath.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST99.json',manifest)
dst=P/'research/restore99';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST99.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS99.csv','ORIGINAL_OUTPUT_EVIDENCE99.csv','PLOT_INVENTORY99.csv','PROGRESS99.json','QUALITY_MATRIX99.csv','CONVERGENCE99.json','ITEM_RECONSTRUCTION99.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE99_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE99_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore99','docs')
# The preregistration is byte-locked; preserve the exact original trailing-space line.
assert sha(S/'THEORY_REGISTERED99.md')==r['contract_sha']==sha(dst/'THEORY_REGISTERED99.md')
git('diff','--cached','--check','--','.',':(exclude)research/restore99/logs',':(exclude)research/restore99/CORE_POLICY.md',':(exclude)research/restore99/QUANTUM_REPRESENTATION_POLICY.md',':(exclude)research/restore99/THEORY_REGISTERED99.md')
git('commit','-m','RESTORE99: finite scaling null, tail information and heldout crossover tests')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE99.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE99.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE99_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST99.json'),source_zip_sha256=sha(S/'MQFT_RESTORE99_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE99.git.bundle'),progress_sha256=sha(S/'PROGRESS99.json'),fresh_paths=0,fresh_backups=0,figures=11,animations=2,cumulative_retained_pairs=752,original_union=121,accepted_corrected_subclaims=43,corrected_report_artifact_accepted=57,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
