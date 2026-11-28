from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore94_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
parent='aadfc5db1d17641dd58969a80984baed8f6c90fb'
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED95.md') and r['parent_dag_sha']==sha(O/'DAG.json')
v=read(S/'INDEPENDENT95.json');u=read(S/'ENTANGLE95.json');pm=read(S/'PLOT_MANIFEST95.json');src=read(S/'ORIGINAL_SOURCE95.json')
assert v['status']==u['status']=='PASS_SCOPED' and len(pm['figures'])==8
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'G23': 'Current-depth explicit R1..Rd,S two-qubit reduced QMI and classical Z MI matrices with independent partial trace; not native branch MI identification.', 'G24': 'Pairwise negativity network and all unique joint bipartition cuts, tagged separable controls; PPT limits and pair-vs-cut distinction.', 'G25': 'Actual branch-derived prefix concurrence for named two-qubit factors and independent spin-flip/SVD checks; no absent pair treated as zero.', 'G26': 'Maximal nondegenerate Pauli CHSH with explicit local directions, Born probability and no-signalling checks, Bell/Werner/separable controls, success-weighted indicator and lambda0 counterexample; not physical Bell experiment.', 'G27': 'Combined QMI, negativity, concurrence, CHSH and tag/phase/base controls on declared adaptive prefix records; rare-event success retained, no native physical identification.'}
acc=list(scopes);partial=[]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS94.json');inv=cr(O/'PLOT_INVENTORY94.csv');assert len(inv)==712 and sha(O/'PLOT_INVENTORY94.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==720;cw(S/'PLOT_INVENTORY95.csv',inv)
c=read(O/'CONVERGENCE94.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==34 and not set(acc)&set(old)
newclaims=[
 dict(id='Q-WITNESS-09',stage=95,items=['G23','G24','G25','G26','G27'],claim='For the declared current-prefix coherent record extension, product local record tag channels preserve all computational diagonals; full tag gives a fully separable mixture, even when classical MI persists. Named pairwise reductions and all joint cuts distinguish NPT from classical correlation. The two-qubit CHSH maximum is realized by explicit local SVD directions; success-weighted excess is an indicator, not an unconditional native Bell result.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','ENTANGLE95.json','JOINT95.npz','INDEPENDENT95.json']],limit='Known entanglement criteria applied to finite declared model; actual native coherent state and physical independent settings unestablished; no novelty claim.'),
 dict(id='Q-BASE-10',stage=95,items=['G24','G26','G27'],claim='For lambda0 on the registered branch settings and common bath convention, each Kraus path is a scalar sqrt(w_h) times a branch-independent free product. The conditional joint state factors into the pure coherent probability-amplitude record tensor system marginal. A dependent classical probability table can therefore yield nonzero record entanglement and large record-record CHSH while system-record entanglement is exactly zero. This control is distinct from full record tagging.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','BASE_CONTROL95.json','ENTANGLE95.json','JOINT95.npz']],limit='Finite counterfactual harvested plus81 node settings, not actual native baseline law or experimental Bell test. Verified 432 positive-depth cases; no general physical advantage.') ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Native physical entanglement and coherent branch identification, physical Bell locality and setting independence remain open.'))
dump(S/'ITEM_RECONSTRUCTION95.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=39,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=34,new_accepted_subclaims=2,new_corrected_artifacts=5,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION95.json'),scope='Five corrected scoped quantum information and entanglement artifacts; six open science claims unchanged; native physical Bell not admitted.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==34
dump(S/'CONVERGENCE95.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS94.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R95_evidence']=scopes.get(k,'');x['R95_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k]),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R95_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS95.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE95.csv',ev)
qm=cr(O/'QUALITY_MATRIX94.csv')
for x in qm:
 k=x['id'];x['R95_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R95_open_subclaims']='';x['R95_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='95';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R95'
cw(S/'QUALITY_MATRIX95.csv',qm)
dump(S/'ITEM_CONVERGENCE95.json',dict(items=[dict(id=x['id'],accepted=x['R95_convergence_subclaims'],scope=x['R95_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R95_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX94.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE94.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION95.json')))
prev.update(figure_pairs=720,distinct_png_sha=720,files_paired=1440,current_pairs=8,prior_pairs=712,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY95.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE95.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS95.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX95.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=34,final_scientific_certified=0)
prev['stages'][S.name]=8;prev['original_output_evidence'].update(current=5,R55_R95_union=121,scope='121 scoped coverage; 39 corrected artifact acceptances,34 scoped claims; six open claims.')
dump(S/'PROGRESS95.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT95.md').write_text('# RESTORE95 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  node['restore95_followup']=dict(evidence='720 raw receipts; 2304 joint prefixes, 5184 controlled states, 17280 pair and 19008 cut evaluations; 432 lambda0 factorization checks',accepted_subclaims=[z['id'] for z in newclaims],corrected_artifacts=[k for k in acc if any(x['id']==k and x['owner']==node['id'] for x in rows)],partial=partial,remaining='Native physical entanglement, coherent branch identification and operational Bell settings not established.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT95.md,ITEM_RECONSTRUCTION95.json,CONVERGENCE95.json}')
dag.update(revision='RESTORE95_20261128',scope='Convergence: declared adaptive prefix pair/cut entanglement, concurrence and CHSH',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE96-DARWINISM-DEDICATED-OUTPUTS',mainline_priorities=['R96 Q24-Q28 dedicated decoherence/Darwinism and accessible information','Direct original reconstruction; no optimizer drift','Six open science claims, Q02/Q31 and Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX95.csv',convergence_ledger='CONVERGENCE95.json')
rec=dict(status='five_corrected_original_artifacts_two_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=72,diagnostic_paths=648,declared_prefix_states=2304,controlled_prefixes=5184,pair_evaluations=17280,cut_evaluations=19008,figures=8,animations=0,retained_pairs=720,original_evidence_items=5,original_evidence_union_R55_R95=121,original_obligations=121,corrected_report_artifact_accepted=39,new_corrected_artifacts=5,accepted_corrected_subclaims=34,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='82 corrected artifact obligations; six open science claims, Q02/Q31 and Q19/Q20 partial; Q24-Q28 next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED95.md','ENTANGLE95.json','JOINT95.npz','SUMMARY95.json','BASE_CONTROL95.json','INDEPENDENT95.json','ORIGINAL_SOURCE95.json','CONVERGENCE95.json','ITEM_RECONSTRUCTION95.json','ORIGINAL_OUTPUT_REPORT95.md','ITEM_CONVERGENCE95.json','QUALITY_MATRIX95.csv','PLOT_MANIFEST95.json','PLOT_INVENTORY95.csv','ORIGINAL_OUTPUT_EVIDENCE95.csv','ORIGINAL_OUTPUT_PROGRESS95.csv','PROGRESS95.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore95']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore95']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish95.log','backup95.log','RUN_MANIFEST95.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=u['raw_receipts'];assert len(raw)==720
manifest=dict(schema='mqft_restore95_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute95.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent bit-index, matrix/SVD and exact controls; no physical Bell or native coherent record identification; forced diagnostics not stochastic trials.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST95.json',manifest)
dst=P/'research/restore95';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST95.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS95.csv','ORIGINAL_OUTPUT_EVIDENCE95.csv','PLOT_INVENTORY95.csv','PROGRESS95.json','QUALITY_MATRIX95.csv','CONVERGENCE95.json','ITEM_RECONSTRUCTION95.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE95_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE95_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore95','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore95/logs',':(exclude)research/restore95/CORE_POLICY.md',':(exclude)research/restore95/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE95: reconstruct declared prefix entanglement and Bell witnesses')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE95.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE95.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE95_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST95.json'),source_zip_sha256=sha(S/'MQFT_RESTORE95_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE95.git.bundle'),progress_sha256=sha(S/'PROGRESS95.json'),fresh_paths=0,fresh_backups=0,figures=8,cumulative_retained_pairs=720,original_union=121,accepted_corrected_subclaims=34,corrected_report_artifact_accepted=39,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
