from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore95_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
parent='03593ba04a1f685d0d762cb6986593b721214dfe'
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED96.md') and r['parent_dag_sha']==sha(O/'DAG.json')
v=read(S/'INDEPENDENT96.json');u=read(S/'DARWIN96.json');pm=read(S/'PLOT_MANIFEST96.json');src=read(S/'ORIGINAL_SOURCE96.json')
assert v['status']==u['status']=='PASS_SCOPED' and len(pm['figures'])==6 and len(pm['animations'])==1
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'Q24': 'Actual R81 software phase interventions and R94 conditional qubit, R95 declared adaptive prefix coherence and entropy, explicit controls; no physical bath or higher-time coherent hardware readout.', 'Q25': 'All nonempty fragments, attainable measurement lower and Holevo/TV upper, exact disjoint packing and entropy guard; negative plus-arm result, base81 positive control retained; no actual objectivity.', 'Q26': 'Local pointer discrimination and disturbance, frozen stability and actual phase-dependent decision comparator; no calibrated pointer energy spectrum.', 'Q27': 'Four-frame GIF H32,36,40,44 from verified actual conditional qubit and declared prefix data, explicit clock; static snapshot pair; no physical time calibration.', 'Q28': 'Objectivity diagnostic report with conditional product deviation, pointer measurement disturbance, disjoint packing, tag and lambda0 controls and success-weighted contribution; no physical redundancy or objectivity certification.'}
acc=list(scopes);partial=[]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS95.json');inv=cr(O/'PLOT_INVENTORY95.csv');assert len(inv)==720 and sha(O/'PLOT_INVENTORY95.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==726;cw(S/'PLOT_INVENTORY96.csv',inv)
an=pm['animations'][0];assert sha(S/an['path'])==an['sha256']
with Image.open(S/an['path']) as im:
 assert im.n_frames==an['frames']==4
 hs=[]
 for i in range(4):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
 assert hs==an['frame_rgb_sha256'] and len(set(hs))==4
c=read(O/'CONVERGENCE95.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==39 and not set(acc)&set(old)
newclaims=[
 dict(id='Q-ACCESS-11',stage=96,items=['Q25','Q28'],claim='For a binary pointer ensemble and any POVM, classical common-overlap decomposition yields I(X:Y)<=H(X) TV(P0,P1)<=H(X) D(rho0,rho1). Together with Holevo and explicit projective candidates this bounds unrestricted accessible information. In the registered dataset the TV bound did not tighten Holevo at tolerance, and attainable candidates improved Z information in 10670 of 25920 evaluations.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','DARWIN96.json','SUMMARY96.json','INDEPENDENT96.json']],limit='Finite specified binary ensembles, no claimed unrestricted POVM optimum or new general theorem; no native physical objectivity.'),
 dict(id='Q-PACK-12',stage=96,items=['Q24','Q25','Q26','Q28'],claim='Exact disjoint packing from lower and upper fragment information bounds gives a certified cardinality interval. For H>1e-6 and 90% threshold, no two qualifying fragments are possible for either registered pointer in final plus81 xi1. All 18 attained two-fragment conditions are six native-repeat base81 contexts at three tags; one extra unresolved condition. System-Z has no possible two-fragment packing in any registered condition.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','DARWIN96.json','SUMMARY96.json','INDEPENDENT96.json']],limit='Finite conditional model, not a universal no-go. Lambda0 counterfactual uses harvested plus settings; repeated tags not independent contexts; degeneracy and success weights retained.') ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Native physical coherent record identification, bath/time/energy and objectivity remain open.'))
next(x for x in items if x['id']=='Q27')['animation']=an
dump(S/'ITEM_RECONSTRUCTION96.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=44,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=36,new_accepted_subclaims=2,new_corrected_artifacts=5,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION96.json'),scope='Five corrected scoped Darwinism outputs; no native physical objectivity; six open science claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==36
dump(S/'CONVERGENCE96.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS95.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R96_evidence']=scopes.get(k,'');x['R96_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k])+(';' + an['path'] if k=='Q27' else ''),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R96_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS96.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE96.csv',ev)
qm=cr(O/'QUALITY_MATRIX95.csv')
for x in qm:
 k=x['id'];x['R96_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R96_open_subclaims']='';x['R96_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='96';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R96'
cw(S/'QUALITY_MATRIX96.csv',qm)
dump(S/'ITEM_CONVERGENCE96.json',dict(items=[dict(id=x['id'],accepted=x['R96_convergence_subclaims'],scope=x['R96_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R96_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX95.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE95.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION96.json')))
prev.update(figure_pairs=726,distinct_png_sha=726,files_paired=1452,current_pairs=6,prior_pairs=720,current_animations=1,inventory_sha256=sha(S/'PLOT_INVENTORY96.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE96.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS96.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX96.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=36,final_scientific_certified=0)
prev['stages'][S.name]=6;prev['original_output_evidence'].update(current=5,R55_R96_union=121,scope='121 scoped coverage; 44 corrected artifact acceptances,36 scoped claims; six open claims.')
dump(S/'PROGRESS96.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT96.md').write_text('# RESTORE96 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  node['restore96_followup']=dict(evidence='4104 rehashed raw receipts; 5184 controlled states, 8640 pointer ensembles, 25920 fragment evaluations, exact packing and 4-frame animation',accepted_subclaims=[z['id'] for z in newclaims],corrected_artifacts=[k for k in acc if any(x['id']==k and x['owner']==node['id'] for x in rows)],partial=partial,remaining='Native physical objectivity and coherent record identification not established.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT96.md,ITEM_RECONSTRUCTION96.json,CONVERGENCE96.json}')
dag.update(revision='RESTORE96_20261128',scope='Convergence: declared adaptive prefix accessible information and disjoint fragment packing',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE97-ENTROPY-PARTITION-OUTPUTS',mainline_priorities=['R97 Q35-Q37 Q39-Q40 entropy and partition outputs','Direct original reconstruction; no optimizer drift','Six open science claims, Q02/Q31 and Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX96.csv',convergence_ledger='CONVERGENCE96.json')
rec=dict(status='five_corrected_original_artifacts_two_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=3456,diagnostic_paths=648,controlled_prefixes=5184,pointer_ensembles=8640,fragment_evaluations=25920,figures=6,animations=1,retained_pairs=726,original_evidence_items=5,original_evidence_union_R55_R96=121,original_obligations=121,corrected_report_artifact_accepted=44,new_corrected_artifacts=5,accepted_corrected_subclaims=36,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='77 corrected artifact obligations; six open science claims, Q02/Q31 and Q19/Q20 partial; Q35-Q37 Q39-Q40 next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED96.md','DARWIN96.json','SUMMARY96.json','INDEPENDENT96.json','ORIGINAL_SOURCE96.json','CONVERGENCE96.json','ITEM_RECONSTRUCTION96.json','ORIGINAL_OUTPUT_REPORT96.md','ITEM_CONVERGENCE96.json','QUALITY_MATRIX96.csv','PLOT_MANIFEST96.json','PLOT_INVENTORY96.csv','ORIGINAL_OUTPUT_EVIDENCE96.csv','ORIGINAL_OUTPUT_PROGRESS96.csv','PROGRESS96.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore96']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore96']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish96.log','backup96.log','RUN_MANIFEST96.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=v['raw_receipts'];assert len(raw)==4104
manifest=dict(schema='mqft_restore96_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute96.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent bit-index, actual measurement probabilities and exact packing; no unrestricted POVM optimum or native physical objectivity.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST96.json',manifest)
dst=P/'research/restore96';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST96.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS96.csv','ORIGINAL_OUTPUT_EVIDENCE96.csv','PLOT_INVENTORY96.csv','PROGRESS96.json','QUALITY_MATRIX96.csv','CONVERGENCE96.json','ITEM_RECONSTRUCTION96.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE96_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE96_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore96','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore96/logs',':(exclude)research/restore96/CORE_POLICY.md',':(exclude)research/restore96/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE96: bound accessible information and disjoint record packing')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE96.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE96.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE96_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST96.json'),source_zip_sha256=sha(S/'MQFT_RESTORE96_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE96.git.bundle'),progress_sha256=sha(S/'PROGRESS96.json'),fresh_paths=0,fresh_backups=0,figures=6,animations=1,cumulative_retained_pairs=726,original_union=121,accepted_corrected_subclaims=36,corrected_report_artifact_accepted=44,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
