from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore86_20261127';P=B/'mqft_restore39_20261125/repo';A69=B/'mqft_restore69_20261126';parent='7df13ae754f3f16f24f81c9ef1bf446ffe688292';LS=[13,15,17,19,21]
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A69/f),f
q=read(S/'SUMMARY87.json');v=read(S/'INDEPENDENT87.json');c=read(S/'CONVERGENCE87.json');pm=read(S/'PLOT_MANIFEST87.json');e=read(S/'ERRATUM86.json')
assert q['fresh_paths']==800 and q['fresh_backups']==51200 and q['reused_paths']==3520 and not q['critical_gate'] and v['status']=='PASS_SCOPED' and v['raw_paths']==400 and v['spectral_cases']==400
assert c['accepted_subclaims']==18 and c['new_accepted_subclaims']==2 and c['final_original_item_certifications']==0 and e['status']=='CORRECTED_STARTUP_SCOPE'
assert all(sha(S/k)==x for k,x in pm['sources'].items())
src=read(S/'ORIGINAL_SOURCE87.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==read(O/'ORIGINAL_SOURCE86.json')['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
raw=[];prior=[];inputs={};regs={};plans={};files=[]
ext=read(B/'mqft_restore77_20261127/RUN_MANIFEST77.json')['external']
for L in LS:
 r=read(S/f'REGISTRATION_L{L}.json');a=read(S/f'AUDIT_L{L}.json');regs[str(L)]=r
 assert r['model']==dict(ext,rules=f'freestyle{L}') and r['contract_sha']==sha(S/'THEORY_REGISTERED87.md') and a['paths']==864 and a['fresh_paths']==160 and a['fresh_backups']==10240
 assert sha(S/f'OBS_L{L}.npz')==a['array_sha'] and sha(S/f'SECTOR_L{L}.npz')==a['sector_sha']
 for z in a['raw_receipts']:(raw if z['stage']==87 else prior).append(z)
 plans[f'L{L}_base']=r['plans']['base'];assert sha(S/f'plans/L{L}_base.json')==plans[f'L{L}_base']['sha256']
 for z,h in [(B/'mqft_restore49_20261126/runtime_verified'/f'mqft-L{L}',r['backend_sha']),(B/'mqft_restore49_20261126/family'/f'L{L}'/'SOURCE_MANIFEST.json',r['source_manifest_sha'])]:
  assert sha(z)==h;inputs[str(z.relative_to(B))]=h
 for st in (76,77):
  p=B/f'mqft_restore{st}_20261127'
  for z in [f'AUDIT_L{L}.json',f'REGISTRATION_L{L}.json',f'plans/L{L}_base.json']+([f'PATHS_L{L}.npz'] if st==77 else []):inputs[str((p/z).relative_to(B))]=sha(p/z)
 files += [f'{k}_L{L}.json' for k in ('REGISTRATION','GEN','PILOT','AUDIT')]
assert len(raw)==800 and len(prior)==3520 and len({x['path'] for x in raw+prior})==4320 and all(sha(B/x['path'])==x['sha256'] for x in raw+prior)
for k,h in ext['files'].items():assert sha(P/'.external/rapfi'/k)==h
scopes={'L17':'Bounded full-action innovation and independent finite-size critical/crossover comparisons; no critical admission.', 'L26':'Exact entropy-gap crossover and actual controlled log-size selection schedule; no physical temperature.', 'L36':'Noninteracting categorical scaling null and frozen versus adaptive controls; no autonomous physical RG flow.', 'L37':'Controlled heldout critical-width models and crossing diagnostics; original fixed-point obligation remains open.', 'L38':'Per-selection predictable covariance, finite-prefix and board-size observations, corrected R49 startup.', 'Q32':'Bounded normalized innovation susceptibility, raw/drift/innovation covariance decomposition and exact null; physical susceptibility not identified.'}
assert set(scopes)=={k for z in pm['figures'] for k in z['ids']}
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS86.json');inv=cr(O/'PLOT_INVENTORY86.csv');assert len(inv)==665 and sha(O/'PLOT_INVENTORY86.csv')==prev['inventory_sha256']
plotfiles=[]
for x in pm['figures']:
 b=S/'plots'/x['figure'];png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==668;cw(S/'PLOT_INVENTORY87.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS86.csv');ev=[]
for x in rows:
 k=x['id'];x['R87_evidence']=scopes.get(k,'');x['R87_scope_status']='CORRECTED_SUBCLAIM_EVIDENCE' if k in scopes else ''
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(p['figure'] for p in pm['figures'] if k in p['ids']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R87_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE87.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS87.csv',rows)
union={x['id'] for x in rows if any(x.get(f'R{k}_evidence') for k in range(55,88))}
assert len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX86.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if z['stage']==87 and k in z['items']]
 x['R87_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R87_open_subclaims']=';'.join(z['id'] for z in linked if not z['status'].startswith('ACCEPTED'))
 x['R87_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='87';x['latest_scope']=scopes[k];x['corrected_output_state']='CORRECTED_SUBCLAIM_EVIDENCE_R87'
 if k in ('L38','L39'):x['R87_scope']+=' R86 startup bracket/exponential erratum applies; compensated D and lag cross moments unchanged.'
cw(S/'QUALITY_MATRIX87.csv',qm)
dump(S/'ITEM_CONVERGENCE87.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX86.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE86.json'),items=[dict(id=x['id'],accepted=x['R87_convergence_subclaims'],open=x['R87_open_subclaims'],scope=x['R87_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R87_scope']],claims=[z for z in c['claims'] if z['stage']==87],scope='R87 delta and R86 startup correction; R79 original item contracts unchanged.'))
prev.update(figure_pairs=668,distinct_png_sha=668,files_paired=1336,current_pairs=3,prior_pairs=665,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY87.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE87.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS87.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX87.csv'),corrected_report_artifact_accepted=['L31'],accepted_corrected_subclaims=18,final_scientific_certified=0)
prev['stages'][S.name]=3;prev['original_output_evidence']=dict(current=len(scopes),R55_R87_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Complete scoped coverage unchanged; 18 corrected subclaims accepted, L31 retained; final science separate.')
dump(S/'PROGRESS87.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('REPLAN-04.C','REPLAN-04.R','X-MEMORY'):node['restore87_followup']=dict(evidence='800 fresh scale paths,3520 reused, exact bounded innovation and categorical entropy crossover, independent critical-width comparison; explicit R86 startup correction.',accepted_subclaims=['SCALE-BOUND-01','SCALE-ENTROPY-02'],remaining='Critical limit not admitted; autonomous closure open. R86 bracket/exponential corrected in ERRATUM86.json, raw D unchanged.',artifact=S.name+'/{CRITICAL87.json,ERRATUM86.json,SUMMARY87.json,CONVERGENCE87.json}')
dag.update(revision='RESTORE87_20261128',scope='Convergence: bounded innovation and controlled criticality, R86 startup erratum',previous_entry=dag['entry'],entry='REPLAN-04.C',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE88-STRUCTURAL-ORIGINAL-OUTPUT-RECONSTRUCTION',mainline_priorities=['R88 structural gauge/topology/barrier claims and direct original output reconstruction','Resolve remaining original item-specific acceptance conditions without repetitive scans','Open physical identification/utility only with explicit independent observables or intervention'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX87.csv',convergence_ledger='CONVERGENCE87.json')
files+=['THEORY_REGISTERED87.md','CRITICAL87.json','ERRATUM86.json','SUMMARY87.json','INDEPENDENT87.json','ORIGINAL_SOURCE87.json','CONVERGENCE87.json','ITEM_CONVERGENCE87.json','QUALITY_MATRIX87.csv','PLOT_MANIFEST87.json','PLOT_INVENTORY87.csv','ORIGINAL_OUTPUT_EVIDENCE87.csv','ORIGINAL_OUTPUT_PROGRESS87.csv','PROGRESS87.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='two_scoped_scale_claims_accepted_critical_limit_open_R86_erratum',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=800,cumulative_fixed_root_histories=93336,fresh_backups=51200,reused_paths=3520,source_path_receipts=4320,registered_critical_wins=q['registered_critical_wins'],registered_cells=q['registered_cells'],raw_bytes=q['raw_bytes'],new_games=0,optimizer_updates=0,figures=3,animations=0,retained_pairs=668,original_evidence_items=len(scopes),original_evidence_union_R55_R87=121,original_obligations=121,corrected_report_artifact_accepted=1,accepted_corrected_subclaims=18,new_accepted_subclaims=2,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Controlled critical limit, structural and original final item acceptance; R86 startup correction supersedes bracket/exponential tables.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore87']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore87']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=ext,backends={L:r['backend_sha'] for L,r in regs.items()});dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','plan87.py','generate87.py','extract87.py','extract87_initial.py','extract87_fast.py','erratum86.py','critical87.py','verify87.py','summary87.py','plot87.py','publish87.py','backup87.py','execute87.sh','RESULTS_KO.md','STAGE_LEDGER.md']
logs=[f'logs/{n}_L{L}.log' for L in LS for n in ('generate','extract')]+[f'logs/extract_L{L}_initial.log' for L in LS]+['logs/erratum86.log','logs/critical87.log','logs/verify87.log','logs/summary87.log','logs/plot87.log','logs/publish87_first_failure.log']
for z in ['CONVERGENCE86.json','QUALITY_MATRIX86.csv','SCALE86.json','SUMMARY86.json','THEORY.md']:inputs[str((O/z).relative_to(B))]=sha(O/z)
for z in ('CRITICAL76.json','THEORY.md'):p=B/'mqft_restore76_20261127'/z;inputs[str(p.relative_to(B))]=sha(p)
p=B/'mqft_restore46_20261126/runtime/mqft-rng-replay';inputs[str(p.relative_to(B))]=sha(p)
arrays=[f'{k}_L{L}.npz' for L in LS for k in ('OBS','SECTOR')]
manifest=dict(schema='mqft_restore87_convergence_run_v1',parent=parent,status='completed_scoped',command='plan87.py, generate87.py per size; bash execute87.sh on immutable data; initial startup assertion failed, corrected per-selection analysis and erratum',files={z:sha(S/z) for z in files+sources+plotfiles+logs+arrays+['DAG.json','PROVENANCE.json']},raw_receipts=raw,prior_raw_receipts=prior,diagnostic_receipts=[],weights={},external=ext,plan_hashes=plans,inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent independent checks; no full legacy suite. First-wave R86 scope corrected; no critical limit admission.')
dump(S/'RUN_MANIFEST87.json',manifest)
dst=P/'research/restore87';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+logs+['DAG.json','PROVENANCE.json','RUN_MANIFEST87.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
shutil.copytree(S/'plans',dst/'plans')
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS87.csv','ORIGINAL_OUTPUT_EVIDENCE87.csv','PLOT_INVENTORY87.csv','PROGRESS87.json','QUALITY_MATRIX87.csv','CONVERGENCE87.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE87_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE87_RESULTS.md')
git('add','research/restore87','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore87/logs',':(exclude)research/restore87/THEORY_REGISTERED87.md')
git('commit','-m','RESTORE87: bounded innovation and controlled scaling; correct R86 startup')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE87.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE87.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE87_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST87.json'),source_zip_sha256=sha(S/'MQFT_RESTORE87_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE87.git.bundle'),progress_sha256=sha(S/'PROGRESS87.json'),fresh_paths=800,fresh_backups=51200,figures=3,cumulative_retained_pairs=668,original_union=121,accepted_corrected_subclaims=18,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
