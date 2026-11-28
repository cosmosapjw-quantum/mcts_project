from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore96_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
parent='ef1feb40f444715b8312d0cb1a0fbcd207d50cfc'
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED97.md') and r['parent_dag_sha']==sha(O/'DAG.json')
v=read(S/'INDEPENDENT97.json');u=read(S/'ENTROPY97.json');pm=read(S/'PLOT_MANIFEST97.json');src=read(S/'ORIGINAL_SOURCE97.json')
assert v['status']==u['status']=='PASS_SCOPED' and len(pm['figures'])==8 and len(pm['animations'])==2
assert all(sha(S/k)==h for k,h in pm['sources'].items())
scopes={'Q35': 'Declared joint state, all reductions and unique cuts, purification Schmidt spectra, conditional entropy, QMI and NPT with independently checked PT; original physical coherent record identification remains open.', 'Q36': 'Partial only: finite record cut-cardinality and partition spectra plus separate R92 native categorical size series; no independent spatial coherent size/cut geometry, area or volume law not certified.', 'Q37': 'Named one-sided qubit measurement and finite projective discord bounds, Z-Z MI, relative entropy identities, prior actual native phase/decision comparator; no causal flow or unrestricted optimal discord.', 'Q39': 'Actual R68 native game-ply entropy with contributing games and changing risk set, separate R94 actual conditional qubit and declared R95 prefix state clocks; two GIFs with 8 and 4 distinct frames.', 'Q40': 'Comprehensive scoped report of entropy, original mixed-state cuts, measurement bounds, explicit R91 fixed-spectrum thermodynamic derivatives and feedback identities, R92/R93 references; spatial scaling partial, native bath and physical FDT not identified.'}
partial=['Q36'];acc=[k for k in scopes if k not in partial]
fmap={k:[f for f in pm['figures'] if k in f['ids']] for k in scopes};assert all(fmap.values());sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS96.json');inv=cr(O/'PLOT_INVENTORY96.csv');assert len(inv)==726 and sha(O/'PLOT_INVENTORY96.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==734;cw(S/'PLOT_INVENTORY97.csv',inv)
ans=pm['animations'];assert sorted(a['frames'] for a in ans)==[4,8]
for an in ans:
 assert sha(S/an['path'])==an['sha256']
 with Image.open(S/an['path']) as im:
  assert im.n_frames==an['frames']
  hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert hs==an['frame_rgb_sha256'] and len(set(hs))==im.n_frames
c=read(O/'CONVERGENCE96.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==44 and not set(acc)&set(old)
newclaims=[
 dict(id='ENT-PART-13',stage=97,items=['Q35','Q36','Q40'],claim='For any retained subset A of the declared current-prefix density, the canonical purification sqrt(rho) gives squared Schmidt values equal to the eigenvalues of rho_A across A:(BE). This is not original mixed A:B entanglement. Of 19008 cuts, 10260 NPT, 5674 negative conditional entropy witnesses, and 4586 NPT with nonnegative conditional entropies. All 36288 CMI values satisfy SSA within numerical tolerance. Product/dephase relative entropy identities checked with marginal log support.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','ENTROPY97.json','ENTROPY97.json.gz','INDEPENDENT97.json']],limit='Named finite conditional model, not physical native entanglement or area-law scaling. No general new theorem.'),
 dict(id='Q-DISCORD-14',stage=97,items=['Q37','Q39','Q40'],claim='One-sided qubit discord lies between max(0,S(B)-S(AB)) and I-J_best for explicitly measured projective candidates, with Z-Z classical MI<=J_Z<=J_best. Across 15552 cases 4338 improve Z, mean fixed gap .13103674 nat and mean attainable upper .08495533 nat. R68 visit entropy and R91 fixed-spectrum thermodynamic identities are separate estimands.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','SUMMARY97.json','NATIVE_THERMAL97.json','INDEPENDENT97.json']],limit='Not unrestricted POVM optimum or native causal information flow, no bath, physical temperature or FDT. Independent clocks and populations retained.')
 ]
assert not set(x['id'] for x in newclaims)&set(x['id'] for x in c['claims']);c['claims']+=newclaims
items=[]
for k,s in scopes.items():
 ff=fmap[k];items.append(dict(id=k,decision='PARTIAL_CORRECTED_DIAGNOSTIC' if k in partial else 'ACCEPTED_CORRECTED_SCOPED_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure'] for f in ff],png_sha=ff[0]['png_sha'],pdf_sha=ff[0]['pdf_sha'],figure_hashes=ff,original_science='NOT_CERTIFIED',remaining_limit='Native coherent record identification, physical spatial area law and calibrated bath/time/energy remain open.'))
next(x for x in items if x['id']=='Q39')['animations']=ans
dump(S/'ITEM_RECONSTRUCTION97.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=48,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=38,new_accepted_subclaims=2,new_corrected_artifacts=4,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION97.json'),scope='Four corrected scoped entropy outputs, Q36 partial; no spatial area law or native thermal calibration; six open claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==38
dump(S/'CONVERGENCE97.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS96.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R97_evidence']=scopes.get(k,'');x['R97_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in fmap[k])+(';'+';'.join(a['path'] for a in ans) if k=='Q39' else ''),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R97_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS97.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE97.csv',ev)
qm=cr(O/'QUALITY_MATRIX96.csv')
for x in qm:
 k=x['id'];x['R97_convergence_subclaims']=';'.join(z['id'] for z in newclaims if k in z['items']);x['R97_open_subclaims']='Q-RECORD-03;SCALE-CRITICAL-03' if k=='Q36' else '';x['R97_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='97';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R97'
cw(S/'QUALITY_MATRIX97.csv',qm)
dump(S/'ITEM_CONVERGENCE97.json',dict(items=[dict(id=x['id'],accepted=x['R97_convergence_subclaims'],scope=x['R97_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R97_scope']],claims=newclaims,parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX96.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE96.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION97.json')))
prev.update(figure_pairs=734,distinct_png_sha=734,files_paired=1468,current_pairs=8,prior_pairs=726,current_animations=2,inventory_sha256=sha(S/'PLOT_INVENTORY97.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE97.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS97.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX97.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=38,final_scientific_certified=0)
prev['stages'][S.name]=8;prev['original_output_evidence'].update(current=5,R55_R97_union=121,scope='121 scoped coverage; 48 corrected artifact acceptances,38 scoped claims; six open claims.')
dump(S/'PROGRESS97.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT97.md').write_text('# RESTORE97 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({'; '.join(f['figure'] for f in fmap[k])})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
owners={x['owner'] for x in rows if x['id'] in scopes};print('OWNERS',owners,flush=True)
for node in dag['nodes']:
 if node['id'] in owners:
  node['restore97_followup']=dict(evidence='4184 raw receipts; 5760 states, 43776 subsets, 19008 cuts, 36288 CMI, 15552 measurements, 4265 native game cut replays, 4264 thermal derivative points',accepted_subclaims=[z['id'] for z in newclaims],corrected_artifacts=[k for k in acc if any(x['id']==k and x['owner']==node['id'] for x in rows)],partial=[k for k in partial if any(x['id']==k and x['owner']==node['id'] for x in rows)],remaining='Spatial area-law and coherent size series not established; native bath, physical entanglement identification open.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT97.md,ITEM_RECONSTRUCTION97.json,CONVERGENCE97.json}')
dag.update(revision='RESTORE97_20261128',scope='Convergence: mixed partition entropy, purification and discord bounds, native entropy clock and reference thermodynamics',previous_entry=dag['entry'],entry=sorted(owners)[0],supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE98-RG-OUTPUTS',mainline_priorities=['R98 Q09-Q13 exact pushforward, induced beta field and RG closure','Direct original reconstruction; no optimizer drift','Six open science claims, Q02/Q31 and Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX97.csv',convergence_ledger='CONVERGENCE97.json')
rec=dict(status='four_corrected_artifacts_one_partial_two_new_scoped_claims',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=3456,diagnostic_paths=648,raw_game_histories=80,native_game_searches=853,controlled_prefixes=5760,subsets=43776,cuts=19008,conditional_MI=36288,measurement_cases=15552,figures=8,animations=2,retained_pairs=734,original_evidence_items=5,original_evidence_union_R55_R97=121,original_obligations=121,corrected_report_artifact_accepted=48,new_corrected_artifacts=4,accepted_corrected_subclaims=38,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='73 corrected artifact obligations; Q36 remains partial, Q38 historical; six open science claims; Q09-Q13 next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED97.md','ENTROPY97.json','ENTROPY97.json.gz','SUMMARY97.json','NATIVE_THERMAL97.json','INDEPENDENT97.json','ORIGINAL_SOURCE97.json','CONVERGENCE97.json','ITEM_RECONSTRUCTION97.json','ORIGINAL_OUTPUT_REPORT97.md','ITEM_CONVERGENCE97.json','QUALITY_MATRIX97.csv','PLOT_MANIFEST97.json','PLOT_INVENTORY97.csv','ORIGINAL_OUTPUT_EVIDENCE97.csv','ORIGINAL_OUTPUT_PROGRESS97.csv','PROGRESS97.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore97']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore97']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish97.log','backup97.log','RUN_MANIFEST97.json','PUBLICATION_RECEIPT.json','BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json')]
raw=v['raw_receipts'];assert len(raw)==4184
manifest=dict(schema='mqft_restore97_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute97.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent indexed trace/PT, purification SVD, actual projective measurement, raw game counts and fixed-spectrum reference; no spatial area law, unrestricted discord or native thermal calibration.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST97.json',manifest)
dst=P/'research/restore97';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST97.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS97.csv','ORIGINAL_OUTPUT_EVIDENCE97.csv','PLOT_INVENTORY97.csv','PROGRESS97.json','QUALITY_MATRIX97.csv','CONVERGENCE97.json','ITEM_RECONSTRUCTION97.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE97_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE97_RESULTS.md')
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(A69/f)==sha(dst/f)
git('add','research/restore97','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore97/logs',':(exclude)research/restore97/CORE_POLICY.md',':(exclude)research/restore97/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE97: partition entropy and discord bounds, native entropy replay')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE97.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE97.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE97_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST97.json'),source_zip_sha256=sha(S/'MQFT_RESTORE97_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE97.git.bundle'),progress_sha256=sha(S/'PROGRESS97.json'),fresh_paths=0,fresh_backups=0,figures=8,animations=2,cumulative_retained_pairs=734,original_union=121,accepted_corrected_subclaims=38,corrected_report_artifact_accepted=48,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
