from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore79_20261127';A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='149e0bd757e0244ded64b96149c7faab40cf77ed'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
a=read(S/'AUDIT80.json');r=read(S/'ACTION80.json');q=read(S/'SUMMARY80.json');v=read(S/'INDEPENDENT80.json');f=read(S/'FORCED_REPLAY80.json');u=read(S/'RNG_AUDIT80.json');c=read(S/'CONVERGENCE80.json');pm=read(S/'PLOT_MANIFEST80.json');reg=read(S/'REGISTRATION.json')
assert a['paths']==1152 and a['prior_paths']==768 and a['backups']==147456 and a['prior_backups']==98304
assert sha(S/'MARKED80.npz')==a['array_sha']==r['input_sha'] and len(r['saddles'])==180 and all(x['status']=='OK' for x in r['saddles'])
assert v['status']=='PASS_SCOPED' and v['saddles_checked']==180 and len(v['direct_convolutions'])==6 and len(v['binomial_controls'])==9 and v['raw_paths']==48
assert f['status']=='PASS_SCOPED' and len(f['cases'])==24 and f['diagnostic_backups']==3072
assert u['streams']==768 and u['independent_uniforms']==98304 and u['max_error']<1e-12
assert c['accepted_subclaims']==3 and c['final_original_item_certifications']==0
assert all(sha(S/k)==x for k,x in pm['sources'].items())
old=read(O/'ORIGINAL_SOURCE79.json');src=read(S/'ORIGINAL_SOURCE80.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert src['archives']==old['archives'] and all(sha(paths[k])==x for k,x in src['archives'].items())
r0=read(O/'REGISTRATION.json');assert reg['weights']==r0['weights'] and reg['external']==r0['external'] and sha(S/'mqft_train')==reg['binary_sha']==r0['binary_sha'] and sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==r0['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha']
raw=[x for x in a['raw_receipts'] if x['cohort']==1];priorraw=[x for x in a['raw_receipts'] if x['cohort']==0];assert len(raw)==1152 and len(priorraw)==768
assert all(sha(B/x['path'])==x['sha256'] for x in raw+priorraw)
assert all(sha(B/x['path'])==x['sha256'] and sha(B/x['source'])==x['source_sha'] for x in f['cases'])
assert all(sha(S/'plans'/f'{k}.json')==z['sha256'] for k,z in reg['plans'].items())
scopes={'L16':'Accepted corrected finite normalized marked path action, exact finite 1PI duality and controlled bivariate independent-history replica saddle. Physical phase/action and geometry transfer remain open.', 'Q21':'Exact finite marked path law and bivariate replica mass expansion; not physical work or Jarzynski identification.', 'Q43':'Accepted finite statistical saddle correction with explicit independent-replica small parameter; no physical hbar or complex amplitude identified.', 'Q32':'Exact whole-path tilted covariance and Legendre response; independent matched/novel W transfer tested; not native causal thermal susceptibility.', 'G07':'Exact finite tilted statistical source response and covariance with heldout geometry error; causal field/FDT scope remains separate.'}
ids={k for z in pm['figures'] for k in z['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(cat)==121
prev=read(O/'PROGRESS79.json');inv=cr(O/'PLOT_INVENTORY79.csv');assert len(inv)==644 and sha(O/'PLOT_INVENTORY79.csv')==prev['inventory_sha256']
assert len(pm['figures'])==3 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha']
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size));plotfiles.extend(str(z.relative_to(S)) for z in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==647
cw(S/'PLOT_INVENTORY80.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS79.csv');ev=[]
for x in rows:
 k=x['id'];x['R80_evidence']=scopes.get(k,'');x['R80_scope_status']='CORRECTED_SUBCLAIM_EVIDENCE' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[p['figure'] for p in pm['figures'] if k in p['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R80_scope_status'],scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE80.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS80.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{n}_evidence') for n in range(55,80))}
union={x['id'] for x in rows if any(x.get(f'R{n}_evidence') for n in range(55,81))}
assert len(prior)==len(union)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
qm=cr(O/'QUALITY_MATRIX79.csv');assert len(qm)==121
for x in qm:
 k=x['id'];linked=[z for z in c['claims'] if k in z['items']]
 x['R80_convergence_subclaims']=';'.join(z['id'] for z in linked if z['status'].startswith('ACCEPTED'))
 x['R80_open_subclaims']=';'.join(z['id'] for z in linked if not z['status'].startswith('ACCEPTED'))
 x['R80_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='80';x['latest_scope']=scopes[k];x['corrected_output_state']='CORRECTED_SUBCLAIM_EVIDENCE_R80'
cw(S/'QUALITY_MATRIX80.csv',qm)
# Preserve exact prior item records without silently rewriting prior acceptance.
dump(S/'ITEM_CONVERGENCE80.json',dict(parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX79.csv'),parent_acceptance_sha256=sha(O/'ITEM_ACCEPTANCE79.json'),items=[dict(id=x['id'],accepted=x['R80_convergence_subclaims'],open=x['R80_open_subclaims'],scope=x['R80_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R80_scope']],claims=c['claims'],scope='R80 delta; R79 121 item contracts retained.'))
prev.update(figure_pairs=647,distinct_png_sha=647,files_paired=1294,current_pairs=3,prior_pairs=644,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY80.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE80.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS80.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX80.csv'),corrected_report_artifact_accepted=['L31'],accepted_corrected_subclaims=3,final_scientific_certified=0)
prev['stages'][S.name]=3;prev['original_output_evidence']=dict(current=len(ids),R55_R80_union=121,new_scoped_ids=[],baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Complete scoped coverage unchanged; 3 corrected mathematical subclaims accepted, L31 corrected report retained, original item final certification separate.')
dump(S/'PROGRESS80.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id']=='REPLAN-03':node['restore80_followup']=dict(evidence='Corrected ideal categorical marked path action, exact finite Legendre/KL and Hessian duality, 180 bivariate independent-history lattice saddle cases with alias control. 1152 new searches; 768 reused reference; 24 forced replays; independent RNG/coefficients and direct convolution.',accepted_subclaims=[z['id'] for z in c['claims'] if z['status'].startswith('ACCEPTED')],remaining='Native population W transfer, full autonomous joint law, operational complex phase and physical quantum action remain open; original complete node status not inflated.',artifact=S.name+'/{CONVERGENCE80.json,ACTION80.json,INDEPENDENT80.json,QUALITY_MATRIX80.csv}')
dag.update(revision='RESTORE80_20261127',scope='Convergence: corrected statistical path action, exact finite 1PI and controlled bivariate replica expansion',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE81-OPERATIONAL-QUANTUM-PHASE-INSTRUMENT',mainline_priorities=['R81 operational noncommuting multi-time phase intervention and matched dephased comparator','R82 root-conditioned population generating law and geometry-transfer correction','Remaining physical calibration, memory/scale, structural and application item-specific acceptance'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX80.csv',convergence_ledger='CONVERGENCE80.json')
files=['REGISTRATION.json','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS80.json','AUDIT80.json','ACTION80.json','SUMMARY80.json','INDEPENDENT80.json','RNG_AUDIT80.json','REPLAY_PATCH80.json','REPLAY_REGISTRATION80.json','FORCED_REPLAY80.json','ORIGINAL_SOURCE80.json','CONVERGENCE80.json','ITEM_CONVERGENCE80.json','QUALITY_MATRIX80.csv','PLOT_MANIFEST80.json','PLOT_INVENTORY80.csv','ORIGINAL_OUTPUT_EVIDENCE80.csv','ORIGINAL_OUTPUT_PROGRESS80.csv','PROGRESS80.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='three_corrected_scientific_subclaims_accepted',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=1152,cumulative_fixed_root_histories=84088,fresh_backups=147456,reused_reference_paths=768,forced_replays=24,forced_diagnostic_backups=3072,raw_bytes=a['raw_bytes'],saddle_cases=180,empirical_source_cases=v['source_cases'],new_games=0,optimizer_updates=0,figures=3,animations=0,retained_pairs=647,original_evidence_items=len(ids),original_evidence_union_R55_R80=121,original_obligations=121,corrected_report_artifact_accepted=1,accepted_corrected_subclaims=3,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='Operational phase, geometry transfer and individual original science acceptance.')
rec.update({z+'_sha256':sha(S/z) for z in files});dag['restore80']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore80']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights'],pulse_binary_sha=reg['pulse_binary_sha'],replay_binary_sha=sha(S/'mqft_replay80'));dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit_native74.py','audit_rapfi74.py','pulse74.patch','plan80.py','generate80.py','build80.py','replay80.py','replay80.patch','extract80.py','rng80.py','action80.py','action80_initial_grid.py','verify80.py','summary80.py','plot80.py','publish80.py','backup80.py','execute80.sh','RESULTS_KO.md','STAGE_LEDGER.md']
inputs={str((O/z).relative_to(B)):sha(O/z) for z in ['REGISTRATION.json','AUDIT79.json','QUALITY_MATRIX79.csv','ITEM_ACCEPTANCE79.json','ORIGINAL_SOURCE79.json']}
for x in f['cases']:inputs[x['path']]=x['sha256'];inputs[x['source']]=x['source_sha']
rngp=B/'mqft_restore61_20261126/mqft_trained_rng61';assert sha(rngp)==u['binary_sha'];inputs[str(rngp.relative_to(B))]=u['binary_sha']
manifest=dict(schema='mqft_restore80_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute80.sh (existing registered raw and forced replay); plan80.py/generate80.py, build80.py/replay80.py for acquisition',files={z:sha(S/z) for z in files+sources+plotfiles+['DAG.json','PROVENANCE.json','mqft_train','mqft_pulse74','mqft_replay80','MARKED80.npz','RNG_INPUT80.json']},raw_receipts=raw,prior_raw_receipts=priorraw,forced_diagnostic_receipts=f['cases'],weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation; no full legacy regression or rigorous floating interval; exact finite law versus empirical population inference separate.')
dump(S/'RUN_MANIFEST80.json',manifest)
dst=P/'research/restore80';assert not dst.exists();dst.mkdir()
for z in files+sources+plotfiles+['DAG.json','PROVENANCE.json','RUN_MANIFEST80.json']:
 p=dst/z;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/z,p)
shutil.copytree(S/'plans',dst/'plans')
for z in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS80.csv','ORIGINAL_OUTPUT_EVIDENCE80.csv','PLOT_INVENTORY80.csv','PROGRESS80.json','QUALITY_MATRIX80.csv','CONVERGENCE80.json']:shutil.copy2(S/z,P/'docs'/z)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE80_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE80_RESULTS.md')
git('add','research/restore80','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore80/pulse74.patch',':(exclude)research/restore80/replay80.patch')
git('commit','-m','RESTORE80: converge normalized marked action and bivariate replica expansion')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE80.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE80.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE80_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST80.json'),source_zip_sha256=sha(S/'MQFT_RESTORE80_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE80.git.bundle'),progress_sha256=sha(S/'PROGRESS80.json'),fresh_paths=1152,fresh_backups=147456,forced_diagnostics=24,figures=3,cumulative_retained_pairs=647,original_union=121,accepted_corrected_subclaims=3,corrected_report_artifact_accepted=1,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
