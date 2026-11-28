from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore93_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
parent='0e7ecce731118823884165f66554d7ee27eac036'
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED94.md') and r['parent_dag_sha']==sha(O/'DAG.json')
v=read(S/'INDEPENDENT94.json');u=read(S/'NATIVE94.json');pm=read(S/'PLOT_MANIFEST94.json');src=read(S/'ORIGINAL_SOURCE94.json')
assert v['status']==u['status']=='PASS_SCOPED' and len(pm['figures'])==7
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'G13': 'Actual recorded complex qubit conditional pre/prepared/post coherence and noncommuting phase intervention, fixed basis C_l1 and C_rel; no physical bath.', 'G14': 'Actual conditional entropy and local unselective counterfactual; explicit S-record-bath/purifier prefix partition, entropy spectra and classical diagonal comparator; not native physical entanglement.', 'G15': 'Full legal-action conditional inverse participation on actual histories vs base score law and realized prefix occupancy; normalized probability and finite-count bounds.', 'G16': 'Local pointer-axis fixed states and finite frozen singular-vector stability, explicit record fragment Holevo and actual paired recommendation disagreement; no emergent global pointer or objectivity.', 'G17': 'Joint adaptive noncommuting conditional trajectories and declared coherent prefix record, entropy, participation and pointer metrics; rare-event weights and distinct diagnostic intervention retained.'}
acc=list(scopes);partial=[]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS93.json');inv=cr(O/'PLOT_INVENTORY93.csv');assert len(inv)==705 and sha(O/'PLOT_INVENTORY93.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==712;cw(S/'PLOT_INVENTORY94.csv',inv)
c=read(O/'CONVERGENCE93.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==29 and not set(acc)&set(old)
newclaims=[
 dict(id='Q-INSTRUMENT-07',stage=94,items=['G13','G14','G15','G16','G17'],claim='For the specified efficient grouped square-root qubit instrument, its nonselective map fixes the measured longitudinal Bloch component and contracts transverse components by gamma. Pointer readout TV=m lambda. Purification concavity gives mean selective entropy no larger than prepared entropy, while unital measurement increases unselective entropy; their difference is the Holevo balance. A frozen noncommuting sequence minimizes final pure-input entropy along its largest right singular direction.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','NATIVE94.json','NATIVE94.npz','INDEPENDENT94.json']],limit='Explicit instrument, fixed-history local channel and frozen sequence only. Actual selected entropy may increase or decrease; no global adaptive unselective law, native physical pointer or novelty claim.'),
 dict(id='Q-PREFIX-08',stage=94,items=['G14','G16','G17'],claim='For the specified branch-dependent coherent extension, each continuing prefix joint S-record matrix is a positive Gram product of Kraus path amplitudes with a mixed-input purifier. Its normalized system marginal equals the direct CP continuing density recurrence; accumulated complement failures restore trace one for the stopped system. Prefix survival normalization is not final postselection. Entropy of the joint SR equals the Schmidt entropy across SR versus bath-plus-purifier, not S-record entanglement alone.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','RECORD94.json','INDEPENDENT94.json']],limit='Declared common bath convention, native-repeat and pair-diagnostic distinct; no identification of actual coherent branch state or physical Darwinism from measured diagonal outcomes.')]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Native physical coherence, unique pointer/objectivity and coherent branch record identification open.'))
dump(S/'ITEM_RECONSTRUCTION94.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=34,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=32,new_accepted_subclaims=2,new_corrected_artifacts=5,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION94.json'),scope='Five corrected scoped quantum/decoherence artifacts; six open science claims unchanged; native physical coherence not admitted.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==32
dump(S/'CONVERGENCE94.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS93.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R94_evidence']=scopes.get(k,'');x['R94_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k]),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R94_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS94.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE94.csv',ev)
qm=cr(O/'QUALITY_MATRIX93.csv')
for x in qm:
 k=x['id'];x['R94_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R94_open_subclaims']='';x['R94_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='94';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R94'
cw(S/'QUALITY_MATRIX94.csv',qm)
dump(S/'ITEM_CONVERGENCE94.json',dict(items=[dict(id=x['id'],accepted=x['R94_convergence_subclaims'],scope=x['R94_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R94_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX93.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE93.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION94.json')))
prev.update(figure_pairs=712,distinct_png_sha=712,files_paired=1424,current_pairs=7,prior_pairs=705,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY94.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE94.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS94.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX94.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=32,final_scientific_certified=0)
prev['stages'][S.name]=7;prev['original_output_evidence'].update(current=5,R55_R94_union=121,scope='121 scoped coverage; 34 corrected artifact acceptances,32 scoped claims; six open claims.')
dump(S/'PROGRESS94.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT94.md').write_text('# RESTORE94 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  node['restore94_followup']=dict(evidence='4104 raw paths, 3456 stochastic phase histories and 648 diagnostics; 2304 declared joint prefix states, entropy and pointer verification',accepted_subclaims=[z['id'] for z in newclaims],corrected_artifacts=[k for k in acc if any(x['id']==k and x['owner']==node['id'] for x in rows)],partial=partial,remaining='Native physical coherent branch identification and objectivity not established.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT94.md,ITEM_RECONSTRUCTION94.json,CONVERGENCE94.json}')
dag.update(revision='RESTORE94_20261128',scope='Convergence: actual conditional complex qubit and declared adaptive record prefix, coherence entropy pointer',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE95-ENTANGLEMENT-NETWORK-CONCURRENCE-BELL-OUTPUTS',mainline_priorities=['R95 G23-G27 explicit entanglement, QMI, concurrence and Bell controls','Direct original reconstruction; no optimizer drift','Six open science claims, Q02/Q31 and Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX94.csv',convergence_ledger='CONVERGENCE94.json')
rec=dict(status='five_corrected_original_artifacts_two_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=3456,diagnostic_paths=648,declared_prefix_states=2304,figures=7,animations=0,retained_pairs=712,original_evidence_items=5,original_evidence_union_R55_R94=121,original_obligations=121,corrected_report_artifact_accepted=34,new_corrected_artifacts=5,accepted_corrected_subclaims=32,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='87 corrected artifact obligations; six open science claims, Q02/Q31 and Q19/Q20 partial; G23-G27 next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED94.md','NATIVE94.json','NATIVE94.npz','RECORD94.json','INDEPENDENT94.json','ORIGINAL_SOURCE94.json','CONVERGENCE94.json','ITEM_RECONSTRUCTION94.json','ORIGINAL_OUTPUT_REPORT94.md','ITEM_CONVERGENCE94.json','QUALITY_MATRIX94.csv','PLOT_MANIFEST94.json','PLOT_INVENTORY94.csv','ORIGINAL_OUTPUT_EVIDENCE94.csv','ORIGINAL_OUTPUT_PROGRESS94.csv','PROGRESS94.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore94']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore94']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish94.log','backup94.log','RUN_MANIFEST94.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=u['raw_receipts'];assert len(raw)==4104
manifest=dict(schema='mqft_restore94_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute94.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent scalar/complex CP/purification checks; no native physical coherence or objectivity; forced diagnostics not stochastic sample.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST94.json',manifest)
dst=P/'research/restore94';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST94.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS94.csv','ORIGINAL_OUTPUT_EVIDENCE94.csv','PLOT_INVENTORY94.csv','PROGRESS94.json','QUALITY_MATRIX94.csv','CONVERGENCE94.json','ITEM_RECONSTRUCTION94.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE94_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE94_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore94','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore94/logs',':(exclude)research/restore94/CORE_POLICY.md',':(exclude)research/restore94/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE94: reconstruct complex instrument coherence entropy and pointer records')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE94.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE94.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE94_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST94.json'),source_zip_sha256=sha(S/'MQFT_RESTORE94_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE94.git.bundle'),progress_sha256=sha(S/'PROGRESS94.json'),fresh_paths=0,fresh_backups=0,figures=7,cumulative_retained_pairs=712,original_union=121,accepted_corrected_subclaims=32,corrected_report_artifact_accepted=34,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
